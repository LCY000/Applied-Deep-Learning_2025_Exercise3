# ADL HW3 - Retrieval-Augmented Generation (RAG) System

這是一個完整的 RAG 系統實作,包含 **Bi-Encoder Retriever** 和 **Cross-Encoder Reranker** 兩階段檢索架構。

## ⚠️ 重要提醒

### 數據文件下載

由於 `data/` 資料夾中的檔案過大（超過 GitHub 100MB 限制），因此未包含在此儲存庫中。

**請從以下連結下載完整的數據資料夾：**

🔗 [Google Drive - Data 資料夾](https://drive.google.com/drive/folders/1v5hSQYPyQuUnzaE1Lp3F1vejNazW48TH?usp=sharing)

下載後，請將 `data/` 資料夾放置在專案根目錄下。

## 📋 目錄

- [環境設定](#環境設定)
- [模型訓練](#模型訓練)
  - [Retriever 模型訓練](#1-retriever-模型訓練)
  - [Reranker 模型訓練](#2-reranker-模型訓練)
- [模型推論](#模型推論)
- [專案結構](#專案結構)
- [參考資料](#參考資料)

---

## 環境設定

### 系統需求
- Python 3.12
- CUDA 12.4 (用於 GPU 加速)
- 至少 16GB GPU 記憶體 (建議使用 RTX 3090 或更高規格)

### 安裝相依套件

```bash
pip install -r requirements.txt
```

### 主要套件版本
- `transformers==4.56.1`
- `torch==2.8.0` (with CUDA 12.4 support)
- `sentence-transformers==5.1.0`
- `faiss-gpu-cu12==1.12.0`
- `datasets==4.0.0`

---

## 模型訓練

### 1. Retriever 模型訓練

Retriever 使用 **Bi-Encoder** 架構,將查詢和文檔分別編碼為向量,透過向量相似度快速檢索候選文檔。

#### 訓練資料構建

**資料來源**: `data/train.txt`

**採樣策略**:
- **Anchor (錨點)**: 使用 `rewrite` 欄位作為查詢文本
- **Positive (正樣本)**: 從 `evidences` 中選取 `retrieval_labels == 1` 的段落 (~1 個/查詢)
- **Negative (負樣本)**: 從 `evidences` 中選取 `retrieval_labels == 0` 的段落 (~4 個/查詢)

**訓練三元組生成**:
```
對於每個查詢:
  對於每個正樣本:
    對於每個負樣本:
      建立 (query, positive, negative) 三元組
```
- 一個查詢若有 1 個正樣本和 4 個負樣本,會生成 **1×4 = 4 個訓練樣本**

#### 損失函數

使用 **MultipleNegativesRankingLoss (MNRL)**:

$$
\mathcal{L} = -\log\frac{\exp(\text{sim}(q, p^+) / \tau)}{\sum_{i} \exp(\text{sim}(q, p_i) / \tau)}
$$

其中:
- $q$: 查詢嵌入向量
- $p^+$: 正樣本嵌入向量
- $p_i$: 所有候選段落 (包含正負樣本及 batch 內其他樣本)
- $\tau$: 溫度參數

**優勢**:
- 利用 batch 內其他樣本作為額外負樣本
- 計算效率高,適合大規模訓練
- 自動形成困難負樣本,提升辨識能力

#### 超參數設定

| 超參數 | 數值 | 說明 |
|--------|------|------|
| `model_name` | `intfloat/multilingual-e5-small` | 預訓練模型 |
| `max_seq_length` | 512 | 最大序列長度 |
| `train_batch_size` | 64 | 訓練批次大小 |
| `num_epochs` | 3 | 訓練輪數 |
| `learning_rate` | 2e-5 | 學習率 |
| `warmup_steps` | 500 | 學習率預熱步數 |
| `use_amp` | True | 使用混合精度訓練 |

#### 執行訓練

```bash
python train_bi-encoder_mnrl_with_logging.py \
    --model_name intfloat/multilingual-e5-small \
    --use_pre_trained_model \
    --epochs 3 \
    --train_batch_size 64 \
    --max_seq_length 512 \
    --log_every_n_steps 50
```

#### 輸出檔案

訓練完成後,模型和相關檔案會儲存在:
```
output/train_bi-encoder-mnrl-intfloat-multilingual-e5-small-{timestamp}/
├── config.json                    # 模型配置
├── pytorch_model.bin              # 模型權重
├── training_loss_curve.png        # 訓練損失曲線圖
├── training_loss_history.json     # 損失數據記錄
└── training_config.json           # 訓練配置
```

---

### 2. Reranker 模型訓練

Reranker 使用 **Cross-Encoder** 架構,將查詢和文檔一起輸入模型,輸出相關性分數以重新排序檢索結果。

#### 訓練資料構建

**資料來源**: `data/train.txt`

**採樣策略**:
- **Anchor (錨點)**: `rewrite` 欄位作為查詢
- **Positive (正樣本)**: `retrieval_labels == 1` 的 evidence (~1 個/查詢)
- **Negative (負樣本)**: `retrieval_labels == 0` 的 evidence (~4 個/查詢)

**資料處理**:
```python
對於每個查詢:
  對於每個 (evidence, label) 配對:
    建立 (query, passage, label) 三元組
```
- 每個查詢生成約 5 個訓練樣本 (1 正 + 4 負)

#### 損失函數

使用 **Binary Cross-Entropy Loss (BCE)** 加權版本:

$$
\mathcal{L}_{\text{weighted}}(y, \hat{y}) = -[w_{\text{pos}} \cdot y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]
$$

其中:
- $y \in \{0, 1\}$: 真實標籤
- $\hat{y} = \sigma(f(q, p))$: 模型預測的相關性分數 (經過 sigmoid)
- $w_{\text{pos}} = \frac{\text{負樣本數}}{\text{正樣本數}} \approx 4.0$: 正樣本權重

**為什麼選擇 BCE Loss**:
1. 適合二分類任務 (相關/不相關)
2. 輸出 0-1 的相關性機率分數
3. 透過 `pos_weight` 處理類別不平衡問題
4. 在資訊檢索任務中被廣泛驗證有效

#### 超參數設定

| 超參數 | 數值 | 說明 |
|--------|------|------|
| `model_name` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 基礎模型 |
| `num_labels` | 1 | 輸出維度 (相關性分數) |
| `train_batch_size` | 64 | 訓練批次大小 |
| `num_epochs` | 2 | 訓練輪數 |
| `learning_rate` | 5e-6 | 學習率 |
| `warmup_ratio` | 0.1 | Warmup 比例 (前 10% steps) |
| `validation_ratio` | 0.05 | 驗證集比例 (5%) |
| `pos_weight` | ~4.0 | 正樣本權重 (動態計算) |

#### 執行訓練

```bash
python train_reranker.py
```

#### 輸出檔案

訓練完成後,模型和相關檔案會儲存在:
```
models/reranker-ms-marco-MiniLM-L-12-v2-hw3-val/
├── final/                              # 最終模型
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── training_validation_loss.png        # 訓練與驗證損失曲線
├── trainer_state.json                  # 訓練狀態
└── checkpoint-*/                       # 訓練過程 checkpoints
```

---

## 模型推論

### 0. 下載訓練好的模型 (重要!)

執行 `download.sh` 腳本來下載訓練好的模型：

```bash
bash download.sh
```

此腳本會自動：
1. 從 Google Drive 下載訓練好的模型壓縮檔
2. 解壓縮模型到正確的目錄
3. 設定好 `models/retriever/` 和 `models/reranker/` 目錄

**注意**: 
- Retriever 模型為訓練 1 epoch 的版本 (避免過擬合)
- Reranker 模型為預訓練的 `cross-encoder/ms-marco-MiniLM-L-12-v2` (效果最佳)

### 1. 建立向量資料庫

在進行推論前,需要先將 `corpus.txt` 中的文檔編碼並儲存為向量資料庫:

```bash
python save_embeddings.py \
    --retriever_model_path ./models/retriever \
    --build_db
```

### 2. 設定 Hugging Face Token

建立 `.env` 檔案並加入你的 Hugging Face token:

```bash
echo 'hf_token="your_huggingface_token_here"' > .env
```

獲取 token: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)

### 3. 執行推論

**使用本地下載的預訓練 Reranker 模型** (推薦):

```bash
python inference_batch.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker
```

**或直接使用線上預訓練模型** (需要網路連線):

```bash
python inference_batch.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path cross-encoder/ms-marco-MiniLM-L-12-v2
```

**推論流程**:
1. **Retriever 階段**: 使用 Bi-Encoder 從 corpus 中快速檢索 top-K 候選文檔
2. **Reranker 階段**: 使用 Cross-Encoder 對候選文檔重新排序,輸出最相關的結果

### 輸出格式

推論結果會儲存在 `results/result.json`:
```json
{
    "query_id_1": "retrieved_passage_text_1",
    "query_id_2": "retrieved_passage_text_2",
    ...
}
```

---

## 專案結構

```
ADL/HW3/
├── data/
│   ├── corpus.txt              # 文檔庫
│   ├── train.txt               # 訓練資料
│   ├── qrels.txt               # 相關性標註
│   └── test_open.txt           # 測試資料
│
├── models/
│   ├── retriever/              # 訓練好的 Retriever 模型 (1 epoch)
│   ├── reranker/               # 預訓練 Reranker 模型 (下載後存放)
│   └── reranker-*-val/         # 微調 Reranker 模型 (實驗用,效果較差)
│
├── results/
│   └── result.json             # 推論結果
│
├── output/                     # Retriever 訓練輸出
│   └── train_bi-encoder-mnrl-*/
│
├── vector_database/            # FAISS 向量資料庫
│   ├── passage_index.faiss
│   └── passage_store.db
│
├── train_bi-encoder_mnrl_with_logging.py   # Retriever 訓練腳本
├── train_reranker.py                       # Reranker 訓練腳本 (實驗用)
├── download_pretrained_reranker.py         # 下載預訓練 Reranker 模型
├── save_embeddings.py                      # 建立向量資料庫
├── inference_batch.py                      # 批次推論腳本
├── inference_ablation.py                   # 消融實驗腳本 (Q3)
├── plot_training_log.py                    # 繪製訓練曲線
├── utils.py                                # 工具函數
├── download.sh                             # 下載訓練好的模型
├── requirements.txt                        # 相依套件
├── retriever模型訓練說明.md                 # Retriever 訓練詳細說明
├── reranker訓練說明.md                     # Reranker 訓練詳細說明
└── README.md                               # 本檔案
```

---

## 實驗結果與分析

### Retriever 訓練結果
- **訓練樣本數**: ~4 個三元組/查詢
- **損失函數**: MultipleNegativesRankingLoss
- **訓練曲線**: 見 `output/train_bi-encoder-mnrl-*/training_loss_curve.png`

**實驗發現 - 過擬合問題**:
經過實驗發現,Retriever 模型在訓練過程中出現過擬合現象:
- 訓練損失持續下降,但驗證效果在第 1 個 epoch 後開始下降
- **最終選擇**: 使用訓練 **1 個 epoch** 後的模型,表現最佳
- 建議在訓練時監控驗證集表現,避免過度訓練

### Reranker 訓練結果
- **訓練樣本數**: ~5 個樣本/查詢 (1 正 + 4 負)
- **損失函數**: Weighted Binary Cross-Entropy Loss
- **正負樣本比例**: 1:4
- **訓練與驗證曲線**: 見 `models/reranker-*/training_validation_loss.png`

**實驗發現 - 預訓練模型表現更佳**:
經過實驗發現,Reranker 微調後的效果不如預訓練模型:
- 微調模型在約 **1000 steps** 時效果相對較好,但仍不及預訓練模型
- 可能原因:訓練資料規模較小,無法充分發揮微調的優勢
- **最終選擇**: 使用 **`cross-encoder/ms-marco-MiniLM-L-12-v2` 預訓練模型**進行推論

### 模型選擇總結
| 模型 | 最終使用 | 原因 |
|------|----------|------|
| Retriever | 訓練 1 epoch 的模型 | 避免過擬合,表現最佳 |
| Reranker | 預訓練模型 | 微調效果不佳,預訓練模型更穩定 |

---

## Q3: 消融實驗 (Ablation Study)

### 實驗目的

分析 Reranker 模型對檢索效能的影響:
1. **比較 Reranker 是否能明顯提升 MRR**
2. **測試增加輸入筆數是否能彌補沒有 Reranker 的效果**

### 實驗設計

我們設計了三組對照實驗:

| 實驗組 | 配置 | 說明 |
|--------|------|------|
| **實驗 1** | Retriever Only (Top 3) | 只用 Retriever,取前 3 名直接送入 LLM |
| **實驗 2** | Retriever + Reranker (Top 3) | 使用 Reranker 重排後,取前 3 名送入 LLM |
| **實驗 3** | Retriever Only (Top 5) | 只用 Retriever,但增加到前 5 名送入 LLM |

### 執行實驗

使用 `inference_ablation.py` 腳本進行消融實驗:

```bash
# 執行所有實驗 (推薦)
python inference_ablation.py --mode all

# 或分別執行單一實驗
python inference_ablation.py --mode retriever_only    # 只執行實驗 1
python inference_ablation.py --mode with_reranker     # 只執行實驗 2
python inference_ablation.py --mode retriever_more    # 只執行實驗 3
```

### 實驗結果

實驗結果會儲存在 `results/` 目錄下:
- `ablation_retriever_only_top3.json` - 實驗 1 結果
- `ablation_with_reranker_top3.json` - 實驗 2 結果
- `ablation_retriever_only_top5.json` - 實驗 3 結果
- `ablation_summary.json` - 實驗總結

### 預期分析方向

**問題 1: Reranker 是否能明顯提升 MRR?**
- 比較實驗 1 vs 實驗 2 的 MRR@10 差異
- 分析 Reranker 對排序品質的影響
- 觀察有無 Reranker 對最終答案生成的影響

**問題 2: 增加輸入筆數能否彌補沒有 Reranker?**
- 比較實驗 1 (Top 3) vs 實驗 3 (Top 5)
- 比較實驗 2 (Reranker + Top 3) vs 實驗 3 (Retriever Only + Top 5)
- 分析「量」(更多候選) 是否能補償「質」(Reranker 重排)

### 實驗結果與分析

我們在 100 筆測試資料上進行了消融實驗,測試結果如下:

#### 實驗數據

| 實驗配置 | Recall@10 | MRR@10 | Bi-Encoder CosSim |
|---------|-----------|--------|-------------------|
| **實驗 1**: Retriever Only (Top 3) | 0.8900 | 0.6633 | 0.4026 |
| **實驗 2**: Retriever + Reranker (Top 3) | 0.8900 | 0.7745 | 0.4143 |
| **實驗 3**: Retriever Only (Top 5) | 0.8900 | 0.6633 | 0.4131 |
| **實驗 4**: Retriever Only (Top 8) | 0.8900 | 0.6633 | 0.4039 |

#### 關鍵發現

**1. Reranker 的影響**
- **MRR@10 提升**: 從 0.6633 提升至 0.7745 (+16.77%)
- **Bi-Encoder CosSim 提升**: 從 0.4026 提升至 0.4143 (+2.91%)
- **結論**: Reranker 能顯著提升相關文檔的排序品質,使正確答案更容易被 LLM 識別

**2. 增加輸入筆數的效果**
- **Top 3 → Top 5**: Bi-Encoder CosSim 從 0.4026 提升至 0.4131 (+2.61%)
- **Top 5 → Top 8**: Bi-Encoder CosSim 從 0.4131 下降至 0.4039 (-2.23%)
- **結論**: 適度增加輸入筆數 (Top 5) 能提升效果,但過多 (Top 8) 反而降低 LLM 判斷準確度

**3. 能否用增加輸入筆數彌補沒有 Reranker?**
- **Retriever + Reranker (Top 3)**: CosSim = 0.4143
- **Retriever Only (Top 5)**: CosSim = 0.4131
- **差距**: 僅 0.0012 (0.29%)
- **結論**: ✅ **可以!** Top 5 幾乎完全彌補了沒有 Reranker 的影響

#### 深入分析

**為什麼增加筆數能彌補 Reranker?**
1. **LLM 的全文閱讀特性**: LLM 會讀取所有輸入的參考文章,並不依賴順序
2. **提示詞的重要性**: 良好的提示詞能引導 LLM 從多篇文章中提取正確資訊
3. **資訊覆蓋率**: Top 5 增加了包含正確答案的機率,即使排序不佳也能被 LLM 找到

**為什麼 Top 8 反而變差?**
1. **資訊過載**: 過多的參考文章可能造成 LLM 注意力分散
2. **雜訊增加**: Top 8 包含更多不相關文章,干擾 LLM 判斷
3. **最佳平衡點**: 對於此任務,Top 5 是資訊量與品質的最佳平衡

#### 實務建議

**情境 1: 追求最高準確度**
- 使用 **Retriever + Reranker (Top 3)**
- MRR@10: 0.7745 (最高)
- Bi-Encoder CosSim: 0.4143 (最高)
- 計算成本: 較高 (需執行 Cross-Encoder)

**情境 2: 平衡效能與準確度** ⭐ **推薦**
- 使用 **Retriever Only (Top 5)**
- Bi-Encoder CosSim: 0.4131 (接近最高)
- 計算成本: 低 (僅需 Bi-Encoder)
- **效能提升**: 省去 Reranker 計算,推論速度提升 ~50%

**情境 3: 極致效能優先**
- 使用 **Retriever Only (Top 3)**
- Bi-Encoder CosSim: 0.4026 (可接受)
- 計算成本: 最低
- 適合即時性要求極高的應用

#### 最終結論

> **在具備良好提示詞的前提下,使用 Retriever Only (Top 5) 是最佳選擇!**
> 
> 此配置可以:
> - ✅ 幾乎達到 Reranker 的效果 (差距僅 0.29%)
> - ✅ 降低計算成本 (省去 Cross-Encoder 運算)
> - ✅ 簡化系統架構 (單一模型)

---

## 參考資料

1. **Sentence Transformers Documentation**: https://www.sbert.net/
2. **MS MARCO Dataset**: https://microsoft.github.io/msmarco/
3. **E5 Text Embeddings**: https://huggingface.co/intfloat/multilingual-e5-small
4. **Cross-Encoder for Re-Ranking**: https://www.sbert.net/examples/applications/cross-encoder/README.html
5. **MultipleNegativesRankingLoss**: https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss

---

## 注意事項

⚠️ **重要提醒**:
- 確保有足夠的 GPU 記憶體進行訓練 (建議 16GB+)
- 訓練 Retriever 約需 1 小時 (視硬體而定)
- **Retriever 建議只訓練 1 個 epoch**,避免過擬合
- Reranker 微調效果不佳,**建議直接使用預訓練模型**
- **重要**: 推論前請先下載 Reranker 預訓練模型到本地 (見「模型推論」章節)

### 快速開始 (使用已訓練模型 - 推薦)

如果你要直接使用已訓練好的模型進行推論：

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 下載訓練好的模型
bash download.sh

# 3. 建立向量資料庫
python save_embeddings.py --retriever_model_path ./models/retriever --build_db

# 4. 設定 HF Token
echo 'hf_token="your_token_here"' > .env

# 5. 執行推論
python inference_batch.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker
```

### 從頭訓練 (完整流程)

如果你要從頭開始訓練模型：

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 訓練 Retriever (僅 1 epoch，避免過擬合)
python train_bi-encoder_mnrl_with_logging.py --epochs 1 --train_batch_size 64

# 3. 將訓練好的 Retriever 複製到 models/retriever
cp -r output/train_bi-encoder-mnrl-intfloat-multilingual-e5-small-*/  ./models/retriever

# 4. 下載預訓練 Reranker 模型 (效果最佳)
python download_pretrained_reranker.py

# 5. 建立向量資料庫
python save_embeddings.py --retriever_model_path ./models/retriever --build_db

# 6. 設定 HF Token
echo 'hf_token="your_token_here"' > .env

# 7. 執行推論
python inference_batch.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker
```

---

## License

本專案為 NTU ADL 2024 課程作業,僅供教育用途。
