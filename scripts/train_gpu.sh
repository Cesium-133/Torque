#!/bin/bash

# 优化训练脚本（支持GPU/CPU切换）
# 使用方法: 
#   GPU训练: ./scripts/train_gpu.sh [模型类型]
#   CPU训练: ./scripts/train_gpu.sh [模型类型] --use_cpu
#   安静模式: ./scripts/train_gpu.sh [模型类型] --quiet

# 解析参数
MODEL_TYPE=${1:-"gru"}
shift  # 移除第一个参数

# 默认参数
DATA_PATH="data/processed/train+val"
EPOCHS=200
DEVICE_FLAG="--use_gpu"  # 默认使用GPU
VERBOSE_FLAG="--verbose"  # 默认详细输出

# 处理额外参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_cpu|--cpu)
            DEVICE_FLAG="--use_cpu"
            shift
            ;;
        --quiet|-q)
            VERBOSE_FLAG="--quiet"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# 确定设备类型和输出模式
if [[ $DEVICE_FLAG == "--use_gpu" ]]; then
    DEVICE_NAME="GPU"
else
    DEVICE_NAME="CPU"
fi

if [[ $VERBOSE_FLAG == "--verbose" ]]; then
    echo "==================== ${DEVICE_NAME}优化训练 ===================="
    echo "模型类型: $MODEL_TYPE"
    echo "数据路径: $DATA_PATH"
    echo "训练轮数: $EPOCHS"
    echo "设备类型: $DEVICE_NAME"
    echo "===================================================="

    # 检查GPU是否可用（仅在详细模式下显示）
    python -c "import tensorflow as tf; print('GPU可用:', len(tf.config.experimental.list_physical_devices('GPU')) > 0)"
fi

# 执行优化训练
python src/train_optimized.py \
    --model_type $MODEL_TYPE \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    $DEVICE_FLAG \
    $VERBOSE_FLAG \
    --auto_batch_size \
    --optimizer adamw \
    --learning_rate 0.001 \
    --early_stopping_patience 15 \
    --lr_patience 10 \
    --train_split 0.8 \
    --n_frames 50 \
    --m_frames 2

if [[ $VERBOSE_FLAG == "--verbose" ]]; then
    echo "训练完成！检查 checkpoints/ 目录查看保存的模型。"
    echo "使用 tensorboard --logdir logs/ 查看训练过程。"
fi 