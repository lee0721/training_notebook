#!/bin/bash
#SBATCH --job-name=basketball_yolo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=training_notebook/slurm_train.out

# 移動到提交作業的目錄
cd $SLURM_SUBMIT_DIR

# 載入 Anaconda 並啟用 yolov8 環境（請確認已安裝好）
module load anaconda3/2022.10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# Player 模型訓練
echo "🎽 開始訓練 player detector..."
python training_notebook/basketball_player_detection_training.py

# Ball 模型訓練
echo "🏀 開始訓練 ball detector..."
python training_notebook/basketball_ball_training.py

# Court Keypoint 模型訓練
echo "🟩 開始訓練 court keypoint pose..."
python training_notebook/basketball_court_keypoint_training.py

echo "✅ 全部訓練完成，請確認 training_notebook 內 best.pt 模型是否產生"