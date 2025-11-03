"""
從終端輸出或日誌文件中提取訓練損失數據並繪製曲線

使用方式:
1. 直接貼上訓練輸出:
   python plot_training_log.py

2. 從文件讀取:
   python plot_training_log.py --log_file training_output.txt

3. 手動輸入數據:
   編輯下面的 MANUAL_DATA
"""

import re
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ===== 手動輸入數據 (從你的輸出複製) =====
MANUAL_DATA = """
{'loss': 0.8507, 'grad_norm': 4.928348541259766, 'learning_rate': 1.9960000000000002e-05, 'epoch': 0.25}
{'loss': 0.2239, 'grad_norm': 2.8967132568359375, 'learning_rate': 1.8155268022181148e-05, 'epoch': 0.51}
{'loss': 0.1751, 'grad_norm': 2.831136465072632, 'learning_rate': 1.6306839186691314e-05, 'epoch': 0.76}
{'loss': 0.1364, 'grad_norm': 2.2448956966400146, 'learning_rate': 1.4458410351201479e-05, 'epoch': 1.01}
{'loss': 0.1026, 'grad_norm': 1.8830444812774658, 'learning_rate': 1.2613678373382625e-05, 'epoch': 1.27}
{'loss': 0.0892, 'grad_norm': 1.1866358518600464, 'learning_rate': 1.076894639556377e-05, 'epoch': 1.52}
{'loss': 0.0819, 'grad_norm': 3.0646090507507324, 'learning_rate': 8.920517560073938e-06, 'epoch': 1.78}
{'loss': 0.0761, 'grad_norm': 1.8665401935577393, 'learning_rate': 7.072088724584104e-06, 'epoch': 2.03}
{'loss': 0.0628, 'grad_norm': 2.6477181911468506, 'learning_rate': 5.22365988909427e-06, 'epoch': 2.28}
{'loss': 0.0619, 'grad_norm': 1.3950588703155518, 'learning_rate': 3.3752310536044366e-06, 'epoch': 2.54}
{'loss': 0.0607, 'grad_norm': 1.7225053310394287, 'learning_rate': 1.5268022181146029e-06, 'epoch': 2.79}
"""

def parse_training_data(text):
    """從文本中提取訓練數據"""
    data = []
    
    # 匹配形如 {'loss': 0.8507, 'grad_norm': ..., 'epoch': 0.25} 的行
    pattern = r"\{'loss':\s*([\d.]+),.*?'epoch':\s*([\d.]+)\}"
    
    matches = re.findall(pattern, text)
    
    for loss, epoch in matches:
        data.append({
            'loss': float(loss),
            'epoch': float(epoch)
        })
    
    return data

def plot_loss_curve(data, output_path='training_loss_curve.png'):
    """繪製損失曲線"""
    if not data:
        print("❌ 沒有找到訓練數據!")
        return
    
    epochs = [d['epoch'] for d in data]
    losses = [d['loss'] for d in data]
    
    # 創建圖表
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=6, label='Training Loss')
    
    # 標記每個 epoch 的分界線
    for epoch_num in range(1, int(max(epochs)) + 1):
        plt.axvline(x=epoch_num, color='r', linestyle='--', alpha=0.3, linewidth=1)
        plt.text(epoch_num, max(losses) * 0.95, f'Epoch {epoch_num}', 
                rotation=90, va='top', ha='right', fontsize=10)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Curve - Bi-Encoder with MNRL', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # 添加統計信息
    stats_text = f'Data Points: {len(data)}\n'
    stats_text += f'Initial Loss: {losses[0]:.4f}\n'
    stats_text += f'Final Loss: {losses[-1]:.4f}\n'
    stats_text += f'Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 儲存圖片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 訓練損失曲線已儲存至: {output_path}")
    
    # 同時儲存 JSON 數據
    json_path = output_path.replace('.png', '.json')
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 訓練數據已儲存至: {json_path}")
    
    # 印出統計
    print(f"\n📊 訓練統計:")
    print(f"  數據點數: {len(data)}")
    print(f"  初始 Loss: {losses[0]:.4f}")
    print(f"  最終 Loss: {losses[-1]:.4f}")
    print(f"  下降幅度: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='從訓練日誌中提取並繪製損失曲線')
    parser.add_argument('--log_file', type=str, help='日誌文件路徑')
    parser.add_argument('--output', type=str, default='training_loss_curve.png', 
                       help='輸出圖片路徑')
    args = parser.parse_args()
    
    # 讀取數據
    if args.log_file:
        print(f"📖 從文件讀取: {args.log_file}")
        with open(args.log_file, 'r', encoding='utf8') as f:
            text = f.read()
    else:
        print(f"📖 使用內建數據")
        text = MANUAL_DATA
    
    # 解析並繪圖
    data = parse_training_data(text)
    
    if data:
        print(f"✅ 找到 {len(data)} 個訓練數據點")
        plot_loss_curve(data, args.output)
    else:
        print("❌ 沒有找到訓練數據,請檢查輸入")
        print("\n提示: 請確保輸入包含類似以下格式的行:")
        print("{'loss': 0.8507, 'grad_norm': 4.928, 'learning_rate': 1.996e-05, 'epoch': 0.25}")

if __name__ == '__main__':
    main()
