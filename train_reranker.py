import logging
import json
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.training_args import BatchSamplers
import matplotlib.pyplot as plt
import os

# 設定日誌等級為 INFO 以獲得更多訓練資訊
logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def load_train_data(file_path):
    """從 train.txt 檔案載入訓練資料"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["rewrite"]
            for evidence, label in zip(item["evidences"], item["retrieval_labels"]):
                data.append({
                    "query": query,
                    "passage": evidence,
                    "label": float(label)
                })
    pos_count = sum(1 for d in data if d["label"] == 1.0)
    neg_count = len(data) - pos_count
    logging.info(f"已從 {file_path} 載入 {len(data)} 筆資料, 正:{pos_count} / 負:{neg_count} (1:{neg_count/pos_count:.2f})")
    return data


def plot_training_and_validation_loss(trainer, output_dir):
    """繪製訓練與驗證損失曲線"""
    log_history = trainer.state.log_history

    steps, train_losses, val_losses = [], [], []

    for log in log_history:
        if "loss" in log:
            steps.append(log["step"])
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            val_losses.append((log["step"], log["eval_loss"]))

    plt.figure(figsize=(12, 7))

    # Plot training loss
    plt.plot(steps, train_losses, label="Training Loss", color="#2E86AB", linewidth=2)

    # Plot validation loss
    if val_losses:
        val_steps, val_values = zip(*val_losses)
        plt.plot(val_steps, val_values, label="Validation Loss", color="#E74C3C", linewidth=2, linestyle="--")
        plt.scatter(val_steps, val_values, color="#E74C3C", s=60, zorder=5)

    plt.xlabel("Training Steps", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    plt.title("Reranker Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "training_validation_loss.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    logging.info(f"✅ 已繪製訓練與驗證損失曲線: {output_file}")


def main():
    # ===== 配置參數 =====
    model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    train_file = "./data/train.txt"

    train_batch_size = 64
    num_epochs = 2
    learning_rate = 5e-6
    warmup_ratio = 0.1
    logging_steps = 20
    validation_ratio = 0.05  # 5% 資料用於驗證

    # ===== 1. 載入模型 =====
    logging.info(f"載入模型: {model_name}")
    model = CrossEncoder(
        model_name,
        num_labels=1,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="MS MARCO MiniLM reranker for ADL HW3"
        ),
    )

    # ===== 2. 載入訓練資料 =====
    train_data = load_train_data(train_file)

    dataset = Dataset.from_dict({
        "sentence1": [d["query"] for d in train_data],
        "sentence2": [d["passage"] for d in train_data],
        "label": [d["label"] for d in train_data]
    })

    # === 分割 train / validation ===
    split_dataset = dataset.train_test_split(test_size=validation_ratio, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logging.info(f"訓練集: {len(train_dataset)} 筆, 驗證集: {len(eval_dataset)} 筆")

    # ===== 3. 定義損失函數 =====
    pos_count = sum(1 for d in train_data if d["label"] == 1.0)
    neg_count = len(train_data) - pos_count
    pos_weight = min(neg_count / pos_count, 3.0) if pos_count > 0 else 1.0
    logging.info(f"使用 pos_weight={pos_weight:.2f}")
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(pos_weight))

    # ===== 4. 定義訓練參數 =====
    short_model_name = model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-hw3-val"

    args = CrossEncoderTrainingArguments(
        output_dir=f"./models/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,  # ✅ 保持每 batch 正負比例
        load_best_model_at_end=True,                # ✅ 用最優模型
        metric_for_best_model="eval_loss",
        eval_strategy="steps",                      # ✅ 每 N steps 驗證一次
        eval_steps=200,                             # 每 200 steps 驗證一次
        save_strategy="steps",
        save_steps=200,
        save_total_limit=20,
        logging_steps=logging_steps,
        logging_first_step=True,
        run_name=run_name,
        seed=42,
    )

    # ===== 5. 建立訓練器 =====
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    # ===== 6. 開始訓練 =====
    logging.info("開始訓練...")
    trainer.train()

    # ===== 7. 繪製訓練+驗證損失曲線 =====
    logging.info("繪製訓練與驗證損失曲線...")
    plot_training_and_validation_loss(trainer, f"./models/{run_name}")

    # ===== 8. 儲存最終模型 =====
    final_output_dir = f"./models/{run_name}/final"
    model.save_pretrained(final_output_dir)
    logging.info(f"✅ 模型已儲存至: {final_output_dir}")
    logging.info("✅ 訓練完成!")


if __name__ == "__main__":
    main()
