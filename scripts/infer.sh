#!/bin/bash

# =================================================================
# GRU时序预测模型推理脚本
# 在项目根目录运行此脚本: ./scripts/infer.sh
# =================================================================

# --- 在这里修改你的推理参数 ---

# 1. 指定你要使用的已训练好的模型文件路径
#    注意: 文件名应遵循格式 modeltype_nX_mY_lenZ_foldW_model.h5
#    例如: gru_n10_m2_len150_fold1_model.h5
MODEL_PATH="checkpoints/gru_regressor_n15_m15_len150_model.pth"

# 2. 指定包含待推理文件的目录gru_n50_m2_len150_cpu_cpu_model
#    脚本会递归查找这个目录下的所有 .h5 文件
INPUT_DIR="data/processed/train+val/"

# 3. 指定结果输出文件的路径
OUTPUT_FILE="output/inference_results_n15m15.txt"

# 4. 预测聚合方法
#    max_confidence: 选择置信度最高的预测
#    majority_vote: 多数投票
#    average: 对所有预测取平均
AGGREGATION_METHOD="mean"

# --- 脚本核心 ---

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件未找到: $MODEL_PATH"
    echo "请检查以下几点:"
    echo "1. 模型文件路径是否正确"
    echo "2. 是否已完成模型训练"
    echo "3. 文件名是否遵循格式: modeltype_nX_mY_lenZ_foldW_model.h5"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录未找到: $INPUT_DIR"
    exit 1
fi

echo "========================================"
echo "Starting GRU Sequence Prediction Inference"
echo "========================================"
echo "Model Path:         $MODEL_PATH"
echo "Input Directory:    $INPUT_DIR"
echo "Output File:        $OUTPUT_FILE"
echo "Aggregation Method: $AGGREGATION_METHOD"
echo "Detailed Output:    $DETAILED_OUTPUT"
echo "========================================"

# 激活你的 conda 环境 (如果需要)
# conda activate torque

# 构建推理命令
INFERENCE_CMD="python src/infer.py \
    --model_path \"$MODEL_PATH\" \
    --input_dir \"$INPUT_DIR\" \
    --output_file \"$OUTPUT_FILE\" \
    --aggregation_method \"$AGGREGATION_METHOD\" \
    --detailed_output \
    --visualize \
    --output_dir \"output/html_visualization_m15_new\""

# 运行推理脚本
eval $INFERENCE_CMD

echo "========================================"
echo "Inference script finished."
echo "Results saved to: $OUTPUT_FILE"
echo "========================================"

# 显示结果统计
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "预测结果统计:"
    echo "=============="
    
    # 统计各类别的预测数量 (排除注释行和错误行)
    LEFT_COUNT=$(grep -v "^#" "$OUTPUT_FILE" | grep -v "ERROR" | grep -c " - left")
    MID_COUNT=$(grep -v "^#" "$OUTPUT_FILE" | grep -v "ERROR" | grep -c " - mid")
    RIGHT_COUNT=$(grep -v "^#" "$OUTPUT_FILE" | grep -v "ERROR" | grep -c " - right")
    ERROR_COUNT=$(grep -c "ERROR" "$OUTPUT_FILE")
    
    echo "Left (左):   $LEFT_COUNT 个文件"
    echo "Mid (中):    $MID_COUNT 个文件"
    echo "Right (右):  $RIGHT_COUNT 个文件"
    
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "Errors:      $ERROR_COUNT 个文件"
    fi
    
    echo "=============="
fi