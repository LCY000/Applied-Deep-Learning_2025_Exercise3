#!/bin/bash
# Q3 消融實驗執行腳本
# Ablation Study Runner for Q3

set -e

echo "=========================================="
echo "  Q3 消融實驗 - Reranker 效果分析"
echo "=========================================="
echo ""

# 檢查必要檔案
echo "🔍 檢查必要檔案..."
if [ ! -d "./models/retriever" ]; then
    echo "❌ 錯誤: ./models/retriever 不存在"
    echo "   請先執行 bash download.sh 下載模型"
    exit 1
fi

if [ ! -f "./vector_database/passage_index.faiss" ]; then
    echo "❌ 錯誤: 向量資料庫不存在"
    echo "   請先執行: python save_embeddings.py --retriever_model_path ./models/retriever --build_db"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "⚠️  警告: .env 檔案不存在,請確保有設定 HF token"
fi

echo "✅ 檔案檢查完成"
echo ""

# 詢問執行模式
echo "請選擇執行模式:"
echo "  1) 執行所有實驗 (推薦)"
echo "  2) 只執行實驗 1: Retriever Only (Top 3)"
echo "  3) 只執行實驗 2: Retriever + Reranker (Top 3)"
echo "  4) 只執行實驗 3: Retriever Only (Top 5)"
echo ""
read -p "請輸入選項 [1-4]: " choice

case $choice in
    1)
        MODE="all"
        echo ""
        echo "🚀 開始執行所有實驗..."
        ;;
    2)
        MODE="retriever_only"
        echo ""
        echo "🚀 執行實驗 1: Retriever Only (Top 3)..."
        ;;
    3)
        MODE="with_reranker"
        echo ""
        echo "🚀 執行實驗 2: Retriever + Reranker (Top 3)..."
        ;;
    4)
        MODE="retriever_more"
        echo ""
        echo "🚀 執行實驗 3: Retriever Only (Top 5)..."
        ;;
    *)
        echo "❌ 無效的選項"
        exit 1
        ;;
esac

# 執行實驗
python inference_ablation.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker \
    --mode $MODE

echo ""
echo "=========================================="
echo "  ✅ 實驗完成!"
echo "=========================================="
echo ""
echo "📊 結果已儲存至 ./results/ 目錄:"
echo "   - ablation_retriever_only_top3.json"
echo "   - ablation_with_reranker_top3.json"
echo "   - ablation_retriever_only_top5.json"
echo "   - ablation_summary.json"
echo ""
echo "請查看 ablation_summary.json 獲得實驗總結"
