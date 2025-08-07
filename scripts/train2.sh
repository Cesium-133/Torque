#!/bin/bash

# =================================================================
# GRU时序预测模型训练脚本
# 在项目根目录运行此脚本: ./scripts/train.sh
# =================================================================

# --- 在这里修改你的训练参数 ---

# 模型类型: gru (推荐), cnn, lstm, 或 flatten
MODEL_TYPE="gru"

# 时序预测参数
N_FRAMES=15        # 输入的历史帧数（过去n帧）
M_FRAMES=15        ## 预测的未来帧数（未来m帧）

# 训练参数
EPOCHS=500         # 最大训练轮次 (EarlyStopping 可能会提前终止)
BATCH_SIZE=128      # 批次大小
OPTIMIZER="adamw"  # 优化器类型: adam 或 adamw
LEARNING_RATE=1e-3  # 初始学习率

# 回调参数
EARLY_STOPPING_PATIENCE=50  # 早停耐心值
LR_PATIENCE=20             # 学习率调度器耐心值

# 数据文件路径 (相对于项目根目录)
DATA_PATH="data/processed/train+val/"

# --- 脚本核心 ---
echo "========================================"
echo "Starting GRU Sequence Prediction Training"
echo "========================================"
echo "Model Type:     $MODEL_TYPE"
echo "N Frames:       $N_FRAMES (input history)"
echo "M Frames:       $M_FRAMES (prediction horizon)"
echo "Epochs:         $EPOCHS"
echo "Batch Size:     $BATCH_SIZE"
echo "Optimizer:      $OPTIMIZER"
echo "Learning Rate:  $LEARNING_RATE"
echo "Data Path:      $DATA_PATH"
echo "========================================"

# 检查数据目录是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据目录未找到: $DATA_PATH"
    exit 1
fi

# 激活你的 conda 环境 (如果需要)
# conda activate torque

# 运行 Python 训练脚本，并传递参数
python src/train.py \
    --model_type "$MODEL_TYPE" \
    --n_frames "$N_FRAMES" \
    --m_frames "$M_FRAMES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --optimizer "$OPTIMIZER" \
    --learning_rate "$LEARNING_RATE" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --lr_patience "$LR_PATIENCE" \
    --data_path "$DATA_PATH" \
    --no_cross_validation 

echo "========================================"
echo "Training script finished."
echo "Check the 'checkpoints/' directory for saved models."
echo "Use 'tensorboard --logdir logs/' to view training progress."
echo "========================================"