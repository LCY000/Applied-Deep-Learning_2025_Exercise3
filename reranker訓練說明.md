# Reranker 模型訓練說明文件

## 1. 訓練資料構建 (Training Data Construction)

### 資料來源
- **檔案**: `data/train.txt`
- **格式**: 每行為一個 JSON 物件

### 資料結構說明
每個訓練樣本包含以下欄位:
```json
{
  "rewrite": "查詢問題 (query)",
  "evidences": ["段落1", "段落2", "段落3", "段落4", "段落5"],
  "retrieval_labels": [0, 0, 0, 0, 1]
}
```

### Sampling 策略

#### 1. Anchor (錨點)
- **來源**: `rewrite` 欄位
- **說明**: 改寫後的查詢問題,作為每個訓練樣本的 anchor
- **用途**: 與 passage 配對計算相關性分數

#### 2. Positive Sampling (正樣本採樣)
- **策略**: 使用資料集中已標記的正樣本
- **標記方式**: `retrieval_labels` 中值為 `1` 的對應 evidence
- **特點**: 
  - 每個 query 通常有 1 個正樣本
  - 正樣本是與 query 真正相關的文檔段落
  - 不需要額外的 hard negative mining

#### 3. Negative Sampling (負樣本採樣)
- **策略**: 使用資料集中已標記的負樣本
- **標記方式**: `retrieval_labels` 中值為 `0` 的對應 evidence
- **特點**:
  - 每個 query 通常有 4 個負樣本
  - 負樣本可能是語義相似但不相關的文檔
  - 正負樣本比例約為 1:4
  - 這些負樣本已經是 "hard negatives",因為它們來自初始檢索系統的候選集

### 資料預處理流程
```python
def load_train_data(file_path):
    data = []
    for line in file:
        item = json.loads(line)
        query = item["rewrite"]  # Anchor
        
        # 建立 query-passage-label 三元組
        for evidence, label in zip(item["evidences"], item["retrieval_labels"]):
            data.append({
                "query": query,
                "passage": evidence,
                "label": float(label)  # 1.0 for positive, 0.0 for negative
            })
    return data
```

### 資料統計
- **總訓練樣本數**: (待訓練時統計)
- **正樣本數**: (待訓練時統計)
- **負樣本數**: (待訓練時統計)
- **正負比例**: 約 1:4


---

## 2. 損失函數 (Loss Function)

### 使用的損失函數: Binary Cross-Entropy Loss (BCE)

#### 數學定義
對於單一樣本 $(q, p, y)$,其中:
- $q$ = query (查詢)
- $p$ = passage (段落)
- $y \in \{0, 1\}$ = label (標籤)
- $\hat{y} = \sigma(f(q, p))$ = 模型預測的相關性分數 (經過 sigmoid)

BCE Loss 定義為:
$$
\mathcal{L}(y, \hat{y}) = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]
$$

#### 加權 BCE Loss
為了處理類別不平衡問題,使用加權 BCE:
$$
\mathcal{L}_{weighted}(y, \hat{y}) = -[w_{pos} \cdot y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]
$$

其中 $w_{pos}$ 為正樣本權重,計算方式:
$$
w_{pos} = \frac{\text{負樣本數}}{\text{正樣本數}} \approx 4.0
$$

### 為什麼選擇 BCE Loss?

1. **適合二分類任務**: Reranker 的目標是判斷 query-passage 的相關性 (相關/不相關)
2. **輸出機率分數**: BCE 配合 sigmoid 可以輸出 0-1 的相關性機率
3. **處理不平衡**: 透過 `pos_weight` 參數可以調整正負樣本的權重
4. **廣泛使用**: 在資訊檢索和 reranking 任務中被廣泛驗證有效

### 實作細節
```python
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# 計算正樣本權重
pos_weight = 負樣本數 / 正樣本數  # 約 4.0

# 建立損失函數
loss = BinaryCrossEntropyLoss(
    model=model, 
    pos_weight=torch.tensor(pos_weight)
)
```


---

## 3. 超參數設定 (Hyperparameters)

### 模型相關
| 參數 | 值 | 說明 |
|------|-----|------|
| `model_name` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 基礎模型 |
| `num_labels` | `1` | 輸出維度 (相關性分數) |
| `max_length` | `512` | 最大輸入長度 (由模型決定) |

### 訓練相關
| 參數 | 值 | 說明 |
|------|-----|------|
| `num_train_epochs` | `3` | 訓練輪數 |
| `per_device_train_batch_size` | `8` | 每個 GPU 的 batch size |
| `learning_rate` | `2e-5` | 學習率 |
| `warmup_ratio` | `0.1` | Warmup 比例 (前 10% steps) |

### 優化器相關
| 參數 | 值 | 說明 |
|------|-----|------|
| `optimizer` | `AdamW` | 優化器 (預設) |
| `weight_decay` | `0.01` | 權重衰減 (預設) |

### 硬體相關
| 參數 | 值 | 說明 |
|------|-----|------|
| `fp16` | `False` | 不使用半精度訓練 |
| `bf16` | `False` | 不使用 bfloat16 (可視硬體調整) |
| `dataloader_num_workers` | `0` | 資料載入的執行緒數 |

### 儲存與記錄
| 參數 | 值 | 說明 |
|------|-----|------|
| `output_dir` | `./models/reranker-*` | 模型儲存路徑 |
| `logging_steps` | `100` | 每 100 steps 記錄一次 |
| `save_strategy` | `epoch` | 每個 epoch 結束後儲存 |
| `save_total_limit` | `2` | 最多保留 2 個 checkpoint |
| `seed` | `42` | 隨機種子 |

### 損失函數相關
| 參數 | 值 | 說明 |
|------|-----|------|
| `pos_weight` | `~4.0` | 正樣本權重 (動態計算) |

### 超參數選擇理由

1. **Learning Rate (2e-5)**: 
   - 是 BERT-based 模型微調的標準學習率
   - 避免過大導致不穩定,或過小導致收斂太慢

2. **Batch Size (8)**:
   - 在記憶體限制下的合理選擇
   - 可以根據 GPU 記憶體調整 (16GB 可用 8-16)

3. **Epochs (3)**:
   - 通常 2-4 epochs 足夠微調 cross-encoder
   - 避免過度擬合

4. **Warmup Ratio (0.1)**:
   - 前 10% 的訓練步驟進行學習率 warmup
   - 幫助訓練初期的穩定性


---

## 4. 訓練流程

### 步驟說明

1. **環境準備**
   ```bash
   pip install -r requirements.txt
   ```

2. **執行訓練**
   ```bash
   python train_reranker.py
   ```

3. **輸出檔案**
   - 模型 checkpoints: `./models/reranker-ms-marco-MiniLM-L-12-v2-hw3/`
   - 訓練日誌: 包含在終端輸出
   - 最終模型: `./models/reranker-ms-marco-MiniLM-L-12-v2-hw3/final/`

### 訓練資料處理流程圖
```
train.txt (JSON Lines)
    ↓
解析 JSON 並提取欄位
    ↓
為每個 query 展開 evidences
    ↓
建立 (query, passage, label) 三元組
    ↓
轉換為 Dataset 格式
    ↓
輸入 CrossEncoderTrainer
    ↓
訓練並儲存模型
```


---

## 5. 訓練損失曲線 (Training Loss Curve)

### 如何產生訓練曲線

訓練腳本會自動記錄訓練損失。訓練完成後,可以使用以下方法繪製曲線:

#### 方法 1: 使用現有的 `plot_training_log.py`
```bash
python plot_training_log.py
```

#### 方法 2: 從 trainer_state.json 繪製
```python
import json
import matplotlib.pyplot as plt

# 載入訓練狀態
with open('./models/reranker-ms-marco-MiniLM-L-12-v2-hw3/trainer_state.json', 'r') as f:
    state = json.load(f)

# 提取訓練損失
steps = [log['step'] for log in state['log_history'] if 'loss' in log]
losses = [log['loss'] for log in state['log_history'] if 'loss' in log]

# 繪製曲線
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o')
plt.xlabel('Training Steps')
plt.ylabel('Training Loss')
plt.title('Reranker Training Loss Curve')
plt.grid(True)
plt.savefig('reranker_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 訓練損失數據點 (待訓練後填入)

| Step | Loss | Epoch |
|------|------|-------|
| 0 | - | 0.00 |
| 100 | - | ~0.33 |
| 200 | - | ~0.67 |
| 300 | - | ~1.00 |
| 400 | - | ~1.33 |
| 500 | - | ~1.67 |
| 600 | - | ~2.00 |
| 700 | - | ~2.33 |
| 800 | - | ~2.67 |
| 最終 | - | 3.00 |

### 預期趨勢
- **初期** (Step 0-200): 損失快速下降
- **中期** (Step 200-600): 損失持續下降但速度變慢
- **後期** (Step 600+): 損失趨於平穩,可能輕微震盪

### 訓練曲線圖
*(訓練完成後將圖片插入此處)*

```
[待插入訓練損失曲線圖 reranker_training_loss.png]
```


---

## 6. 模型評估與使用

### 載入訓練好的模型
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('./models/reranker-ms-marco-MiniLM-L-12-v2-hw3/final')
```

### 使用模型進行 Reranking
```python
query = "Where is Malayali located?"
passages = [
    "Passage 1 about something else...",
    "Passage 2 with relevant information about Malayali location...",
    "Passage 3 about another topic..."
]

# 計算相關性分數
scores = model.predict([(query, passage) for passage in passages])

# 根據分數排序
ranked_indices = scores.argsort()[::-1]
for idx in ranked_indices:
    print(f"Rank {idx+1}: Score={scores[idx]:.4f}, Passage={passages[idx][:50]}...")
```


---

## 7. 與 Base Model 比較

即使最終決定使用 base model 作為提交版本,以下是本次實驗的記錄:

### Base Model 資訊
- **模型**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **預訓練資料**: MS MARCO passage ranking dataset
- **預訓練任務**: Binary relevance classification

### 微調的優勢
1. **領域適應**: 針對特定資料集的問答格式
2. **Hard Negatives**: 學習區分相似但不相關的段落
3. **資料分佈**: 適應特定的查詢和文檔風格

### 微調的成本
1. **訓練時間**: 約 X 小時 (視硬體而定)
2. **運算資源**: 需要 GPU 支援
3. **過擬合風險**: 在小資料集上可能過擬合


---

## 8. 實驗紀錄與心得

### 實驗設定
- **訓練日期**: (填入訓練日期)
- **硬體環境**: (填入 GPU 型號和記憶體)
- **訓練時間**: (填入總訓練時間)

### 觀察與發現
(訓練完成後填入)
- 訓練穩定性:
- 損失收斂情況:
- 遇到的問題:
- 解決方案:

### 改進方向
(可選)
- [ ] 嘗試不同的 learning rate
- [ ] 調整 batch size
- [ ] 增加 training epochs
- [ ] 使用更大的模型
- [ ] 資料增強策略


---

## 參考資料

1. **Sentence Transformers Documentation**: https://www.sbert.net/
2. **MS MARCO Dataset**: https://microsoft.github.io/msmarco/
3. **Cross-Encoder for Re-Ranking**: https://www.sbert.net/examples/applications/cross-encoder/README.html
4. **Binary Cross-Entropy Loss**: PyTorch Documentation


---

## 附錄: 完整訓練腳本

完整的訓練腳本請參考: `train_reranker.py`

主要功能:
- ✅ 載入 `data/train.txt` 訓練資料
- ✅ 自動計算正負樣本比例
- ✅ 使用 Binary Cross-Entropy Loss
- ✅ 記錄訓練過程和損失
- ✅ 自動儲存 checkpoints
- ✅ 輸出最終模型
