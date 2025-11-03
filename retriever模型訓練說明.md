# ADL HW3 訓練說明

## 1. 訓練資料建構方式

### 資料來源
- 資料集：`data/train.txt`
- 每行為一個 JSON 物件，包含查詢 (query)、證據段落 (evidences) 和檢索標籤 (retrieval_labels)

### 採樣策略

#### 錨點 (Anchor) 採樣
- **錨點**：使用 `rewrite` 欄位作為查詢文本 (query)
- 每個訓練樣本的錨點是經過改寫的對話查詢，確保語意清晰

#### 正樣本 (Positive) 採樣
- **正樣本來源**：從 `evidences` 列表中選取 `retrieval_labels == 1` 的段落
- **採樣數量**：每個查詢通常有 1 個正樣本
- **正樣本定義**：與查詢語意相關且包含答案的文本段落

#### 負樣本 (Negative) 採樣
- **負樣本來源**：從同一查詢的 `evidences` 列表中選取 `retrieval_labels == 0` 的段落
- **採樣數量**：每個查詢通常有 4 個負樣本
- **負樣本定義**：與查詢主題相關但不包含答案的困難負樣本 (Hard Negatives)
- **採樣策略**：使用資料中預先標註的負樣本，確保負樣本具有挑戰性

#### 訓練三元組生成
對於每個查詢，生成訓練樣本的方式：
```
for each query:
    for each positive in positives:
        for each negative in negatives:
            create_triplet(query, positive, negative)
```
- 一個查詢如有 1 個正樣本和 4 個負樣本，會生成 **1 × 4 = 4 個訓練三元組**

## 2. 損失函數 (Loss Function)

### MultipleNegativesRankingLoss (MNRL)

#### 函數定義
```
L = -log(exp(sim(q, p+) / τ) / Σ_i exp(sim(q, p_i) / τ))
```

其中：
- `q`: 查詢 (query) 的嵌入向量
- `p+`: 正樣本 (positive passage) 的嵌入向量
- `p_i`: 包含正樣本和負樣本的所有段落嵌入向量
- `sim(·,·)`: 相似度函數（通常使用餘弦相似度或點積）
- `τ`: 溫度參數（temperature），用於控制分布的平滑度

#### 工作原理
1. **In-Batch Negatives**：除了明確提供的負樣本外，batch 內其他樣本的正樣本也作為額外的負樣本
2. **對比學習**：透過最大化查詢與正樣本的相似度，同時最小化查詢與負樣本的相似度
3. **優勢**：
   - 充分利用 batch 內的樣本，增加有效負樣本數量
   - 計算效率高，適合大規模訓練
   - 自動形成困難負樣本，提升模型辨識能力

## 3. 超參數設定

### 模型相關
| 超參數 | 數值 | 說明 |
|--------|------|------|
| `model_name` | `intfloat/multilingual-e5-small` | 預訓練模型 |
| `max_seq_length` | 512 | 最大序列長度（tokens） |
| `pooling` | mean | 詞嵌入池化方式 |

### 訓練相關
| 超參數 | 數值 | 說明 |
|--------|------|------|
| `train_batch_size` | 64 | 訓練批次大小 |
| `num_epochs` | 3 | 訓練輪數 |
| `learning_rate` | 2e-5 | 學習率 |
| `warmup_steps` | 500 | 學習率預熱步數 |
| `use_amp` | True | 使用混合精度訓練 |

### 資料相關
| 超參數 | 數值 | 說明 |
|--------|------|------|
| 正樣本/查詢 | ~1 | 平均每個查詢的正樣本數 |
| 負樣本/查詢 | ~4 | 平均每個查詢的負樣本數 |
| 訓練三元組/查詢 | ~4 | 平均每個查詢生成的訓練樣本數 |

### 超參數選擇理由
1. **max_seq_length=512**：
   - 分析顯示 95% 的段落長度 < 960 tokens
   - 512 是標準長度，平衡覆蓋率（~65%）與訓練效率

2. **batch_size=64**：
   - 3090 GPU 扛得住，我就設很大

3. **learning_rate=2e-5**：
   - BERT 系列模型微調的標準學習率
   - 避免破壞預訓練權重

4. **warmup_steps=1000**：
   - 約佔總訓練步數的 10-15%
   - 穩定訓練初期的學習過程

## 4. 訓練損失曲線

訓練程式會自動記錄並繪製訓練損失曲線：

### 輸出檔案
- `training_loss_curve.png`：損失曲線圖片
- `training_loss_history.json`：詳細損失數據
- `training_config.json`：完整訓練配置

### 記錄頻率
- 每 50 個訓練步驟記錄一次損失
- 每個 epoch 結束時記錄
- 預期記錄點數：> 5 個資料點（通常 > 20 個）

## 執行方式

### 使用帶損失記錄的版本（推薦）
```bash
python train_bi-encoder_mnrl_with_logging.py \
    --model_name intfloat/multilingual-e5-small \
    --use_pre_trained_model \
    --epochs 3 \
    --train_batch_size 16 \
    --max_seq_length 512 \
    --log_every_n_steps 50
```

### 基本版本
```bash
python train_bi-encoder_mnrl.py \
    --model_name intfloat/multilingual-e5-small \
    --use_pre_trained_model \
    --epochs 3 \
    --train_batch_size 16 \
    --max_seq_length 512
```

## 輸出結果

模型和訓練資訊會儲存在：
```
output/train_bi-encoder-mnrl-intfloat-multilingual-e5-small-{時間戳記}/
├── config.json                    # 模型配置
├── pytorch_model.bin              # 模型權重
├── training_loss_curve.png        # 訓練損失曲線圖
├── training_loss_history.json     # 損失數據
└── training_config.json           # 訓練配置
```

## 資料格式範例

```json
{
  "qid": "query_id",
  "rewrite": "Where is Malayali located?",
  "evidences": [
    "Evidence passage 1...",
    "Evidence passage 2...",
    "Evidence passage 3...",
    "Evidence passage 4...",
    "Evidence passage 5..."
  ],
  "retrieval_labels": [0, 0, 0, 0, 1]
}
```
- 訓練程式會生成 4 個三元組：`(query, evidence_5, evidence_i)` for i in {1,2,3,4}
