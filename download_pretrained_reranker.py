"""
下載預訓練 Reranker 模型腳本

此腳本會下載 cross-encoder/ms-marco-MiniLM-L-12-v2 預訓練模型
並儲存到 ./models/reranker 目錄，以符合作業繳交要求
"""

import os
from sentence_transformers import CrossEncoder

def download_pretrained_reranker():
    """下載並儲存預訓練 Reranker 模型"""
    
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    save_path = "./models/reranker"
    
    print(f"📥 開始下載預訓練 Reranker 模型: {model_name}")
    print(f"💾 儲存路徑: {save_path}")
    
    # 建立目錄
    os.makedirs(save_path, exist_ok=True)
    
    # 載入並儲存模型
    print("⏳ 正在下載模型...")
    model = CrossEncoder(model_name)
    
    print("💾 正在儲存模型到本地...")
    model.save_pretrained(save_path)
    
    print("✅ 下載完成!")
    print(f"✅ 模型已儲存至: {save_path}")
    print("\n📝 現在可以使用以下指令進行推論:")
    print("python inference_batch.py \\")
    print("    --test_data_path ./data/test_open.txt \\")
    print("    --retriever_model_path ./models/retriever \\")
    print("    --reranker_model_path ./models/reranker")

if __name__ == "__main__":
    download_pretrained_reranker()
