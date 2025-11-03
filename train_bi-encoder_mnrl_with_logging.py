"""
此腳本用於訓練 ADL HW3 資料集的 Bi-Encoder 模型，並記錄詳細的訓練損失。

查詢和段落會獨立傳入 transformer 網路以產生固定大小的嵌入向量。
這些嵌入向量可以透過餘弦相似度來尋找給定查詢的匹配段落。

訓練使用 MultipleNegativesRankingLoss，格式為：
(query, positive_passage, negative_passage)

執行此腳本：
python train_bi-encoder_mnrl_with_logging.py --model_name intfloat/multilingual-e5-small --use_pre_trained_model --epochs 3 --train_batch_size 16
"""

import argparse
import json
import logging
import os
from datetime import datetime

import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util

#### 印出除錯資訊到標準輸出
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--model_name", default="intfloat/multilingual-e5-small")
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=500, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--use_pre_trained_model", default=True, action="store_true")  # 改為預設 True
parser.add_argument("--log_every_n_steps", default=50, type=int, help="每 N 步記錄一次損失")
args = parser.parse_args()

print(args)

# 參數設定
model_name = args.model_name
train_batch_size = args.train_batch_size
max_seq_length = args.max_seq_length
num_epochs = args.epochs

# 載入嵌入模型
if args.use_pre_trained_model:
    logging.info("使用預訓練的 SBERT 模型")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("建立新的 SBERT 模型")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = "output/train_bi-encoder-mnrl-{}-{}".format(
    model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(model_save_path, exist_ok=True)

### 讀取訓練資料
logging.info("從 data/train.txt 讀取訓練資料")
train_samples = []

with open("./data/train.txt", encoding="utf8") as f:
    for line in tqdm.tqdm(f, desc="載入訓練資料"):
        item = json.loads(line)
        query = item["rewrite"]
        evidences = item["evidences"]
        labels = item["retrieval_labels"]
        
        # 找出正樣本和負樣本
        pos_passages = [evidences[i] for i, label in enumerate(labels) if label == 1]
        neg_passages = [evidences[i] for i, label in enumerate(labels) if label == 0]
        
        # 為 E5 模型加上必要的前綴
        # E5 模型要求 query 前面加 "query: "，passage 前面加 "passage: "
        query_with_prefix = f"query: {query}"
        
        # 為每個正樣本配對所有負樣本，建立訓練三元組
        for pos in pos_passages:
            for neg in neg_passages:
                pos_with_prefix = f"passage: {pos}"
                neg_with_prefix = f"passage: {neg}"
                train_samples.append(InputExample(texts=[query_with_prefix, pos_with_prefix, neg_with_prefix]))

logging.info(f"訓練樣本數: {len(train_samples)}")

# 建立 DataLoader 和 Loss 函數
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss_fn = losses.MultipleNegativesRankingLoss(model=model)

# 訓練參數
steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * num_epochs
warmup_steps = args.warmup_steps

logging.info(f"每個 epoch: {steps_per_epoch} 步")
logging.info(f"總訓練步數: {total_steps} 步")
logging.info(f"Warmup 步數: {warmup_steps} 步")
logging.info(f"每 {args.log_every_n_steps} 步記錄一次損失")

# 使用 model.fit 但記錄損失
# 創建一個 callback 來記錄損失
training_loss_history = []

class LossLoggingCallback:
    def __init__(self, log_every_n_steps):
        self.log_every_n_steps = log_every_n_steps
        self.step = 0
        
    def __call__(self, score, epoch, steps):
        self.step += 1
        if self.step % self.log_every_n_steps == 0 or steps == steps_per_epoch:
            training_loss_history.append({
                'epoch': epoch,
                'step': self.step,
                'global_step': (epoch - 1) * steps_per_epoch + steps,
                'loss': score
            })
            logging.info(f"Epoch {epoch}/{num_epochs}, Step {steps}/{steps_per_epoch}, Global Step {self.step}, Loss: {score:.4f}")

# 訓練模型
logging.info("開始訓練...")
model.fit(
    train_objectives=[(train_dataloader, train_loss_fn)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": args.lr},
    checkpoint_path=model_save_path,
    checkpoint_save_steps=steps_per_epoch,
    show_progress_bar=True,
    use_amp=True,
)

# 儲存模型
logging.info("儲存最終模型...")
model.save(model_save_path)

# 繪製訓練損失曲線
logging.info("繪製訓練損失曲線...")

# 如果有記錄到損失資料
if len(training_loss_history) > 0:
    # 提取資料
    global_steps = [item['global_step'] for item in training_loss_history]
    losses = [item['loss'] for item in training_loss_history]
    
    # 建立圖表
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Curve - Bi-Encoder with MNRL', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # 標記 epoch 分界線
    for epoch in range(1, num_epochs):
        epoch_step = epoch * steps_per_epoch
        plt.axvline(x=epoch_step, color='r', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(epoch_step, max(losses), f'Epoch {epoch}', rotation=90, va='top', ha='right')
    
    # 儲存圖片
    loss_curve_path = os.path.join(model_save_path, "training_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ 訓練損失曲線已儲存至: {loss_curve_path}")
    plt.close()
    
    # 儲存損失數據
    loss_data_path = os.path.join(model_save_path, "training_loss_history.json")
    with open(loss_data_path, 'w', encoding='utf8') as f:
        json.dump(training_loss_history, f, indent=2, ensure_ascii=False)
    logging.info(f"✓ 訓練損失數據已儲存至: {loss_data_path}")
    logging.info(f"✓ 總共記錄了 {len(training_loss_history)} 個損失資料點")
else:
    logging.warning("⚠ 沒有記錄到訓練損失資料")

# 儲存訓練配置
config = {
    "model_name": model_name,
    "train_batch_size": train_batch_size,
    "max_seq_length": max_seq_length,
    "num_epochs": num_epochs,
    "learning_rate": args.lr,
    "warmup_steps": warmup_steps,
    "total_training_samples": len(train_samples),
    "steps_per_epoch": steps_per_epoch,
    "total_steps": total_steps,
    "loss_function": "MultipleNegativesRankingLoss",
}

config_path = os.path.join(model_save_path, "training_config.json")
with open(config_path, 'w', encoding='utf8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
logging.info(f"✓ 訓練配置已儲存至: {config_path}")

logging.info("=" * 50)
logging.info("訓練完成！")
logging.info(f"模型儲存位置: {model_save_path}")
logging.info("=" * 50)
