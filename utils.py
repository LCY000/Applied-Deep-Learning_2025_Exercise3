from typing import List
import re
    
def get_inference_system_prompt() -> str:
    """第四版：更嚴格的答案要求"""
    return (
        "You are a question-answering assistant. "
        "You must answer directly and accurately based on the provided passages."
    )


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    """第四版：強調必須完全引用原文，不得改寫"""
    # 上下文段落編號
    context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context_list)])
    
    return (
        f"Context passages:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"1. Read all passages carefully to find the only answer\n"
        f"2. Your answer MUST be copied EXACTLY from the passage text - do NOT paraphrase or change any words\n"
        f"3. Copy the relevant sentence(s) word-for-word from the passage\n"
        f"4. If the answer is not found in any passage, write exactly: CANNOTANSWER\n\n"
        f"Answer:"
    )


def parse_generated_answer(pred_ans: str) -> str:
    """解析模型生成的答案，提取 assistant\\n<think>\\n\\n</think>\\n\\n 後面的內容"""
    
    # 方法1: 尋找 </think> 後的內容
    think_pattern = r'</think>\s*\n\s*(.+?)(?:\n|$)'
    match = re.search(think_pattern, pred_ans, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer
    
    # 方法2: 尋找 assistant 後的內容（如果沒有 think 標籤）
    assistant_pattern = r'assistant\s*\n\s*(.+?)(?:\n|$)'
    match = re.search(assistant_pattern, pred_ans, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 如果內容中沒有 <think>，直接返回
        if '<think>' not in content:
            return content
    
    # 方法3: 如果都找不到，返回最後一行非空內容
    lines = [line.strip() for line in pred_ans.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    # 方法4: 如果以上都失敗，返回原始答案
    return pred_ans.strip()
