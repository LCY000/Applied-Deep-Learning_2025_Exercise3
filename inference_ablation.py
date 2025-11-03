"""
消融實驗：比較有無 Reranker 的效果
Ablation Study: Compare with/without Reranker

此腳本會進行以下實驗：
1. 只使用 Retriever (取前3名直接送入LLM)
2. 使用 Retriever + Reranker (取前3名送入LLM)
3. 使用 Retriever 但增加輸入筆數 (取前5名送入LLM，測試是否能彌補沒有 Reranker 的效果)

用於回答 Q3：
- 比較 Reranker 模型是否能明顯提升 MRR
- 增加輸入的資料筆數可不可以解決這問題，效果是否可以打平有 Reranker 模型
"""

import numpy as np
import json, faiss, torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
import os
import sqlite3
from utils import get_inference_user_prompt, get_inference_system_prompt, parse_generated_answer

load_dotenv()
hf_token = os.getenv("hf_token")
login(token=hf_token)

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_folder", type=str, default="./data")
argparser.add_argument("--index_folder", type=str, default="./vector_database")
argparser.add_argument("--index_file", type=str, default="passage_index.faiss")
argparser.add_argument("--sqlite_file", type=str, default="passage_store.db")
argparser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
argparser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
argparser.add_argument("--retriever_model_path", type=str, default="./models/retriever")
argparser.add_argument("--reranker_model_path", type=str, default="./models/reranker")
argparser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-1.7B")
argparser.add_argument("--mode", type=str, default="all", 
                       choices=["retriever_only", "with_reranker", "retriever_more", "all"],
                       help="實驗模式：retriever_only=只用retriever前3, with_reranker=用reranker前3, retriever_more=retriever前5, all=全部跑")
args = argparser.parse_args()

###############################################################################
# 0. Parameters
TOP_K = 10  # Retriever 檢索數量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_Q = 16  # 與 inference_batch.py 一致
BATCH_GEN = 2  # 與 inference_batch.py 一致，避免記憶體問題
TEST_DATA_SIZE = -1  # -1 表示跑全部測試集

###############################################################################
# 1. Load DB and Index
sqlite_path = f"{args.index_folder}/{args.sqlite_file}"
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

retriever = SentenceTransformer(args.retriever_model_path, device=DEVICE)
print(f"✅ Retriever 已載入: {args.retriever_model_path}")

index = faiss.read_index(os.path.join(args.index_folder, args.index_file))
print(f"✅ FAISS Index 已載入")

###############################################################################
# 2. Load Dataset
def load_qrels_gold_pids(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold
    return qid2gold

tests = []
qid2gold = load_qrels_gold_pids(args.qrels_path)

with open(args.test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj.get("qid")
        query = obj.get("rewrite")
        gold_answer = (obj.get("answer")).get("text", "")
        gold_pids = qid2gold.get(qid, set())
        tests.append({"qid": qid, "query": query, "gold_answer": gold_answer, "gold_pids": gold_pids})

tests = tests[:TEST_DATA_SIZE]
print(f"✅ 載入 {len(tests)} 筆測試資料")

###############################################################################
# 3. Evaluation Metrics
def recall_at_k(retrieved_pids, gold_pids, k):
    topk = retrieved_pids[:k]
    return 1.0 if any(pid in gold_pids for pid in topk) else 0.0

def mrr_at_k(ranked_pids, gold_pids, k):
    for rank, pid in enumerate(ranked_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (rank + 1)
    return 0.0

###############################################################################
# 4. Experiment Functions

def run_retriever_only(top_m=3):
    """
    實驗 1: 只使用 Retriever，取前 top_m 名直接送入 LLM
    不使用 Reranker
    """
    print(f"\n{'='*60}")
    print(f"🔬 實驗：只使用 Retriever (前 {top_m} 名)")
    print(f"{'='*60}")
    
    # 載入 LLM
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    model = AutoModelForCausalLM.from_pretrained(args.generator_model, dtype="auto", device_map="auto")
    print(f"✅ LLM 已載入: {args.generator_model}")
    
    R_at_K_sum = 0.0
    MRR_sum = 0.0
    output_records = []
    
    for b_start in tqdm(range(0, len(tests), BATCH_Q), desc="Processing"):
        batch = tests[b_start:b_start+BATCH_Q]
        qids = [ex["qid"] for ex in batch]
        queries = [ex["query"] for ex in batch]
        gold_sets = [ex["gold_pids"] for ex in batch]
        gold_ans = [ex["gold_answer"] for ex in batch]
        
        # Retrieve from FAISS
        prefix_queries = ["query: " + q for q in queries]
        q_embs = retriever.encode(prefix_queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=BATCH_Q)
        D, I = index.search(q_embs, TOP_K)
        
        # Get passages
        need_rowids = set(int(rid) for row in I for rid in row.tolist())
        placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
        sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
        rows = cur.execute(sql, tuple(need_rowids)).fetchall()
        rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}
        
        # Process each query
        messages_list = []
        for b, row in enumerate(I):
            rid_list = row.tolist()
            cand_ids, cand_texts = [], []
            for rid in rid_list:
                tup = rowid2pt.get(int(rid))
                if tup is None: continue
                pid, text = tup
                cand_ids.append(pid)
                cand_texts.append(text)
            
            # Calculate metrics (基於 Retriever 的順序)
            R_at_K_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)
            MRR_sum += mrr_at_k(cand_ids, gold_sets[b], TOP_K)
            
            # 取前 top_m 名送入 LLM (不經過 Reranker)
            context_list = cand_texts[:min(top_m, len(cand_texts))]
            messages = [
                {"role": "system", "content": get_inference_system_prompt()},
                {"role": "user", "content": get_inference_user_prompt(queries[b], context_list)}
            ]
            messages_list.append(messages)
        
        # Generate answers
        pending = [(idx, msg) for idx, msg in enumerate(messages_list) if msg is not None]
        for g_start in range(0, len(pending), BATCH_GEN):
            chunk = pending[g_start:g_start+BATCH_GEN]
            idxs, msgs_batch = zip(*chunk)
            tokenizer.padding_side = "left"
            rendered_prompts = [
                tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                for m in msgs_batch
            ]
            inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for glob_i, ans in zip(idxs, decoded):
                pred_ans = parse_generated_answer(ans.strip())
                output_records.append({
                    "qid": qids[glob_i],
                    "query": queries[glob_i],
                    "generated_answer": pred_ans,
                    "gold_answer": gold_ans[glob_i]
                })
    
    # 清理記憶體
    del model, tokenizer
    torch.cuda.empty_cache()
    
    N = len(tests)
    recall = R_at_K_sum / N
    mrr = MRR_sum / N
    
    # 計算 Bi-Encoder Cosine Similarity (第三個指標)
    print(f"\n📊 計算答案相似度...")
    res = [record["generated_answer"] for record in output_records]
    ans = [record["gold_answer"] for record in output_records]
    
    sentence_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    emb_res = sentence_scorer.encode(res, convert_to_tensor=True, normalize_embeddings=True)
    emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(emb_res, emb_gold)
    diag_scores = scores.diag().tolist()
    bi_encoder_similarity = np.mean(diag_scores)
    
    del sentence_scorer
    torch.cuda.empty_cache()
    
    print(f"\n📊 結果 (只用 Retriever 前 {top_m} 名):")
    print(f"  Recall@{TOP_K}: {recall:.4f}")
    print(f"  MRR@{TOP_K}: {mrr:.4f}")
    print(f"  Bi-Encoder CosSim: {bi_encoder_similarity:.4f}")
    
    # 儲存結果
    result_file = f"./results/ablation_retriever_only_top{top_m}.json"
    os.makedirs("./results", exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": f"Retriever Only (Top {top_m})",
            "recall_at_k": recall,
            "mrr_at_k": mrr,
            "bi_encoder_cossim": bi_encoder_similarity,
            "top_k": TOP_K,
            "top_m": top_m,
            "predictions": output_records
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 結果已儲存至: {result_file}")
    
    return {"recall": recall, "mrr": mrr, "bi_encoder_cossim": bi_encoder_similarity, "mode": f"retriever_only_top{top_m}"}


def run_with_reranker(top_m=3):
    """
    實驗 2: 使用 Retriever + Reranker，取前 top_m 名送入 LLM
    """
    print(f"\n{'='*60}")
    print(f"🔬 實驗：Retriever + Reranker (前 {top_m} 名)")
    print(f"{'='*60}")
    
    # 載入 Reranker
    reranker = CrossEncoder(args.reranker_model_path, device=DEVICE)
    print(f"✅ Reranker 已載入: {args.reranker_model_path}")
    
    # 載入 LLM
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    model = AutoModelForCausalLM.from_pretrained(args.generator_model, dtype="auto", device_map="auto")
    print(f"✅ LLM 已載入: {args.generator_model}")
    
    R_at_K_sum = 0.0
    MRR_sum = 0.0
    output_records = []
    
    for b_start in tqdm(range(0, len(tests), BATCH_Q), desc="Processing"):
        batch = tests[b_start:b_start+BATCH_Q]
        qids = [ex["qid"] for ex in batch]
        queries = [ex["query"] for ex in batch]
        gold_sets = [ex["gold_pids"] for ex in batch]
        gold_ans = [ex["gold_answer"] for ex in batch]
        
        # Retrieve from FAISS
        prefix_queries = ["query: " + q for q in queries]
        q_embs = retriever.encode(prefix_queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=BATCH_Q)
        D, I = index.search(q_embs, TOP_K)
        
        # Get passages
        need_rowids = set(int(rid) for row in I for rid in row.tolist())
        placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
        sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
        rows = cur.execute(sql, tuple(need_rowids)).fetchall()
        rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}
        
        # Create candidates
        batch_cand_ids, batch_cand_texts = [], []
        for b, row in enumerate(I):
            rid_list = row.tolist()
            cand_ids, cand_texts = [], []
            for rid in rid_list:
                tup = rowid2pt.get(int(rid))
                if tup is None: continue
                pid, text = tup
                cand_ids.append(pid)
                cand_texts.append(text)
            batch_cand_ids.append(cand_ids)
            batch_cand_texts.append(cand_texts)
            R_at_K_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)
        
        # Reranking
        flat_pairs = []
        idx_slices = []
        cursor = 0
        for q, ctexts in zip(queries, batch_cand_texts):
            n = len(ctexts)
            if n == 0:
                idx_slices.append((cursor, cursor))
                continue
            flat_pairs.extend(zip([q] * n, ctexts))
            idx_slices.append((cursor, cursor + n))
            cursor += n
        
        if len(flat_pairs) == 0:
            continue
        
        flat_scores = reranker.predict(flat_pairs)
        
        # Process reranked results
        messages_list = []
        for b, (q, (low, high)) in enumerate(zip(queries, idx_slices)):
            if low == high:
                MRR_sum += 0.0
                messages_list.append(None)
                continue
            
            scores = flat_scores[low:high]
            cand_ids = batch_cand_ids[b]
            cand_text = batch_cand_texts[b]
            reranked = sorted(zip(scores, cand_ids, cand_text), key=lambda x: x[0], reverse=True)
            reranked_pids = [pid for _, pid, _ in reranked]
            
            # Calculate MRR based on reranked order
            MRR_sum += mrr_at_k(reranked_pids, gold_sets[b], TOP_K)
            
            # 取前 top_m 名送入 LLM
            context_list = [text for _, _, text in reranked][:min(top_m, len(reranked))]
            messages = [
                {"role": "system", "content": get_inference_system_prompt()},
                {"role": "user", "content": get_inference_user_prompt(queries[b], context_list)}
            ]
            messages_list.append(messages)
        
        # Generate answers
        pending = [(idx, msg) for idx, msg in enumerate(messages_list) if msg is not None]
        for g_start in range(0, len(pending), BATCH_GEN):
            chunk = pending[g_start:g_start+BATCH_GEN]
            idxs, msgs_batch = zip(*chunk)
            tokenizer.padding_side = "left"
            rendered_prompts = [
                tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                for m in msgs_batch
            ]
            inputs = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for glob_i, ans in zip(idxs, decoded):
                pred_ans = parse_generated_answer(ans.strip())
                output_records.append({
                    "qid": qids[glob_i],
                    "query": queries[glob_i],
                    "generated_answer": pred_ans,
                    "gold_answer": gold_ans[glob_i]
                })
    
    # 清理記憶體
    del model, tokenizer, reranker
    torch.cuda.empty_cache()
    
    N = len(tests)
    recall = R_at_K_sum / N
    mrr = MRR_sum / N
    
    # 計算 Bi-Encoder Cosine Similarity (第三個指標)
    print(f"\n📊 計算答案相似度...")
    res = [record["generated_answer"] for record in output_records]
    ans = [record["gold_answer"] for record in output_records]
    
    sentence_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    emb_res = sentence_scorer.encode(res, convert_to_tensor=True, normalize_embeddings=True)
    emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(emb_res, emb_gold)
    diag_scores = scores.diag().tolist()
    bi_encoder_similarity = np.mean(diag_scores)
    
    del sentence_scorer
    torch.cuda.empty_cache()
    
    print(f"\n📊 結果 (Retriever + Reranker 前 {top_m} 名):")
    print(f"  Recall@{TOP_K}: {recall:.4f}")
    print(f"  MRR@{TOP_K}: {mrr:.4f}")
    print(f"  Bi-Encoder CosSim: {bi_encoder_similarity:.4f}")
    
    # 儲存結果
    result_file = f"./results/ablation_with_reranker_top{top_m}.json"
    os.makedirs("./results", exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": f"Retriever + Reranker (Top {top_m})",
            "recall_at_k": recall,
            "mrr_at_k": mrr,
            "bi_encoder_cossim": bi_encoder_similarity,
            "top_k": TOP_K,
            "top_m": top_m,
            "predictions": output_records
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 結果已儲存至: {result_file}")
    
    return {"recall": recall, "mrr": mrr, "bi_encoder_cossim": bi_encoder_similarity, "mode": f"with_reranker_top{top_m}"}


###############################################################################
# 5. Main Execution

if __name__ == "__main__":
    results_summary = []
    
    if args.mode == "retriever_only" or args.mode == "all":
        result = run_retriever_only(top_m=3)
        results_summary.append(result)
    
    if args.mode == "with_reranker" or args.mode == "all":
        result = run_with_reranker(top_m=3)
        results_summary.append(result)
    
    if args.mode == "retriever_more" or args.mode == "all":
        result = run_retriever_only(top_m=5)
        results_summary.append(result)
    
    # 顯示總結
    print(f"\n{'='*80}")
    print("📊 實驗總結 - 三個評估指標")
    print(f"{'='*80}")
    print(f"{'實驗配置':<30} | {'Recall@10':<12} | {'MRR@10':<12} | {'Bi-Encoder CosSim':<18}")
    print(f"{'-'*80}")
    for r in results_summary:
        print(f"{r['mode']:<30} | {r['recall']:<12.4f} | {r['mrr']:<12.4f} | {r['bi_encoder_cossim']:<18.4f}")
    print(f"{'='*80}")
    
    # 儲存總結
    summary_file = "./results/ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 總結已儲存至: {summary_file}")
