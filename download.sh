# download.sh
#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "[INFO] Creating model directories..."

echo "[INFO] Downloading trained models..."
# 範例：從 Google Drive 下載 (gdown --id FILE_ID)
# 你要先把訓練好的模型壓縮成 zip 丟到雲端，再把 FILE_ID 換掉
python3 -m gdown --id 1_zq28m2RRBGreBrxn7fSwCUqt0LmGuyv -O ./models.zip

echo "[INFO] Unzipping models..."
unzip -o ./models.zip -d ./

echo "[INFO] Finished setup!"
