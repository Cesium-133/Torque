# 扭矩回归预测器 (Torque Regression Predictor)

这是一个基于深度学习的扭矩传感器数据回归预测项目，用于预测机器人手臂未来时间步的力矩值。

## 项目概述

本项目使用时序深度学习模型对H5格式的扭矩传感器数据进行回归预测：
- **输入**: 过去n帧的7维扭矩传感器数据
- **输出**: 未来m帧的扭矩预测值
- **任务类型**: 回归任务（使用MSE损失函数）
- **模型特点**: 集成了位置嵌入(Position Embedding)以增强时序建模能力

支持两种不同的模型架构：GRU和LSTM，均配备了窗口级别的位置嵌入机制。

## 项目结构

```
Torque_Classifier/
├── README.md              # 项目说明文档
├── requirements.txt       # Python依赖包列表
├── data/                  # 数据目录
│   ├── raw/              # 原始H5数据文件
│   │   ├── train+val/    # 训练和验证数据
│   │   └── test/         # 测试数据
│   └── processed/        # 预处理后的数据（自动生成）
├── checkpoints/          # 保存的模型和标准化器
├── logs/                 # TensorBoard日志文件
├── src/                  # 源代码目录
│   ├── model.py          # 回归模型定义
│   ├── train.py          # 训练脚本
│   ├── infer.py          # 推理脚本
│   └── pre_process.py    # 数据预处理脚本
└── scripts/              # 执行脚本
    ├── train.sh          # 训练执行脚本
    └── infer.sh          # 推理执行脚本
```

## 安装和环境配置

### 环境要求
- Python 3.8+
- TensorFlow 2.x
- CUDA（可选，用于GPU加速）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 功能模块说明

### 1. 模型模块 (`src/model.py`)
提供两种不同的深度学习回归模型架构：

- **GRU模型** (`model_type='gru'`): 使用门控循环单元，配备双层GRU架构和位置嵌入
- **LSTM模型** (`model_type='lstm'`): 使用长短期记忆网络，同样集成了位置嵌入机制

**核心特性:**
- **位置嵌入**: 一维的窗口级别位置编码，通过Embedding层映射到与GRU输出相同的维度
- **特征融合**: 位置嵌入与GRU输出直接相加，增强时序位置感知能力
- **回归输出**: MLP预测头输出m维向量，代表未来m帧的力矩信息

**使用方法:**
```python
from src.model import create_torque_regressor

# 创建GRU回归模型
model = create_torque_regressor(
    input_shape=(10, 7),   # 输入：10个时间步，7维特征
    output_dim=2,          # 输出：预测未来2帧的力矩值
    model_type='gru',      # 模型类型
    n_frames=10,           # 输入历史帧数
    m_frames=2             # 预测未来帧数
)
```

### 2. 训练模块 (`src/train.py`)
负责数据加载、预处理、模型训练和评估。支持两种训练模式：5-Fold交叉验证和简单训练验证分割。

**主要功能:**
- 从H5文件加载扭矩数据（不再需要标签数据）
- 滑动窗口样本生成，自动创建位置编码
- 自动序列长度标准化（填充/截断）
- 数据标准化
- 使用MSE损失函数进行回归训练
- 支持多种训练模式（交叉验证/简单分割）
- 自动保存最佳模型和标准化器

**使用方法:**
```bash
# 使用默认参数训练GRU回归模型（5-Fold交叉验证）
python src/train.py --data_path data/raw/train+val --model_type gru

# 使用简单训练验证分割（更快的训练）
python src/train.py --data_path data/raw/train+val --model_type gru --no_cross_validation

# 训练LSTM模型，指定训练轮数和批大小
python src/train.py --data_path data/raw/train+val --model_type lstm --epochs 100 --batch_size 64 --no_cross_validation --train_split 0.85

# 调整滑动窗口参数
python src/train.py --data_path data/raw/train+val --model_type gru --n_frames 15 --m_frames 3
```

**参数说明:**

*数据相关:*
- `--data_path`: 数据目录路径，包含H5文件
- `--n_frames`: 输入历史帧数（默认10）
- `--m_frames`: 预测未来帧数（默认2）

*模型相关:*
- `--model_type`: 模型类型，可选 'gru', 'lstm'
- `--optimizer`: 优化器类型，可选 'adam', 'adamw'（默认adamw）
- `--learning_rate`: 初始学习率（默认0.001）

*训练相关:*
- `--epochs`: 最大训练轮数（默认200）
- `--batch_size`: 批处理大小（默认128）
- `--early_stopping_patience`: 早停耐心值（默认15）
- `--lr_patience`: 学习率调度器耐心值（默认10）

*训练模式:*
- `--use_cross_validation`: 使用5-Fold交叉验证（默认True）
- `--no_cross_validation`: 使用简单训练验证分割（更快）
- `--train_split`: 简单分割时的训练集比例（默认0.8）

### 3. 推理模块 (`src/infer.py`)
用于对新的H5文件进行力矩值回归预测。

**功能特点:**
- 自动加载训练好的回归模型和标准化器
- 应用与训练时完全相同的预处理流程
- 支持多种聚合方法处理多窗口预测结果
- 输出预测的力矩数值而不是分类结果

**使用方法:**
```bash
# 对目录中的所有H5文件进行回归预测
python src/infer.py --model_path checkpoints/gru_regressor_n10_m2_len150_model.h5 --input_dir data/test --output_file output/torque_predictions.txt

# 使用不同的聚合方法
python src/infer.py --model_path checkpoints/gru_regressor_n10_m2_len150_model.h5 --input_dir data/test --aggregation_method median

# 输出详细的每个时间窗口预测结果
python src/infer.py --model_path checkpoints/gru_regressor_n10_m2_len150_model.h5 --input_dir data/test --detailed_output

# 生成交互式可视化图表（对比真实值与预测值）
python src/infer.py --model_path checkpoints/gru_regressor_n10_m2_len150_model.h5 --input_dir data/test --visualize --output_dir output/visualizations
```

**参数说明:**
- `--model_path`: 训练好的回归模型文件路径
- `--input_dir`: 包含待预测H5文件的目录路径
- `--output_file`: 保存预测结果的文件路径
- `--aggregation_method`: 多窗口预测聚合方法 ('mean', 'median', 'last')
- `--detailed_output`: 是否输出每个时间窗口的详细预测
- `--visualize`: 生成交互式可视化HTML文件，对比真实值与预测值
- `--output_dir`: 可视化HTML文件的保存目录（默认：output）

**可视化功能特点:**
- **交互式图表**: 使用Plotly生成可缩放、可拖拽的交互式HTML图表
- **真实值对比**: 显示蓝色实线表示真实的torque值（仅第一维度）
- **预测值展示**: 显示红色虚线和标记点表示每个时间窗口的预测值
- **单帧预测支持**: 特别优化了m_frames=1时的显示效果，使用标记点增强可见性
- **悬停信息**: 鼠标悬停显示精确的数值和时间信息
- **图例控制**: 可以切换显示/隐藏不同的数据序列
- **自动保存**: 为每个H5文件生成独立的HTML可视化文件
- **调试信息**: 输出数据形状信息帮助诊断问题

## 数据格式要求

### H5文件结构
每个H5文件应包含以下数据结构：
```
/right_arm_effort
```

该数据集应为形状为 `(sequence_length, 7)` 的时序数据，其中：
- `sequence_length`: 时序长度（可变）
- `7`: 7维特征向量（扭矩传感器数据）

### 数据组织方式
训练数据可以直接放在数据目录中，不再需要按类别分组。所有H5文件都将被用于回归训练。

## 训练流程

1. **数据加载**: 从指定目录递归加载所有H5文件
2. **滑动窗口采样**: 创建输入-输出对，同时生成位置编码
3. **序列标准化**: 将所有序列填充/截断到固定长度150
4. **数据标准化**: 使用StandardScaler进行Z-score标准化
5. **数据划分**: 根据选择的模式进行数据划分
6. **模型训练**: 使用MSE损失函数、早停和模型检查点机制
7. **模型保存**: 自动保存最佳模型和标准化器

## 模型输出

训练完成后会在`checkpoints/`目录下生成：
- `{model_type}_regressor_n{N}_m{M}_len150_model.h5`: 训练好的回归模型文件
- `{model_type}_regressor_n{N}_m{M}_len150_scaler.joblib`: 数据标准化器

## 监控和日志

项目集成了TensorBoard支持：
```bash
# 启动TensorBoard查看训练过程
tensorboard --logdir logs/
```

## 常见问题

### 1. 导入错误
如果遇到Keras导入错误，请确保安装了正确版本的TensorFlow：
```bash
pip install tensorflow>=2.8.0
```

### 2. CUDA警告信息
训练时可能出现以下CUDA相关警告，这些警告不影响训练过程：
```
Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```
这些是TensorFlow的内部警告，可以安全忽略。

### 3. CUDA内存不足
如果GPU内存不足，可以：
- 减小批处理大小：`--batch_size 64`
- 减少滑动窗口帧数：`--n_frames 8`
- 使用CPU训练：设置环境变量 `CUDA_VISIBLE_DEVICES=""`

### 4. 数据加载失败
请检查：
- H5文件路径是否正确
- H5文件是否包含正确的数据结构 (`/right_arm_effort`)
- 文件权限是否正确

## 性能优化建议

1. **数据预处理**: 考虑使用数据缓存减少重复加载时间
2. **模型选择**: GRU模型通常在时序回归任务中表现良好，且计算效率高于LSTM
3. **超参数调优**: 可以调整学习率、网络层数、位置嵌入维度等参数
4. **数据增强**: 考虑添加时序数据增强技术，如噪声注入、时间扭曲等
5. **位置编码**: 根据数据特性调整位置嵌入的最大长度参数

## 模型架构详解

### 位置嵌入机制
- **输入**: 窗口在序列中的起始位置索引
- **嵌入**: 通过Embedding层映射到与GRU输出相同的维度
- **融合**: 与GRU输出进行元素级相加，增强位置感知能力

### 回归预测头
- **结构**: 多层全连接网络 (64 -> 32 -> output_dim)
- **激活**: ReLU激活函数（隐藏层），线性激活（输出层）
- **正则化**: Dropout防止过拟合

## 版本更新日志

- **v2.0**: 
  - 从分类任务改为回归任务
  - 添加位置嵌入机制
  - 使用MSE损失函数
  - 移除标签依赖，直接预测力矩值
  - 支持多帧力矩预测
- **v1.0**: 初始版本，支持CNN/LSTM/Flatten三种分类模型

---
*如有问题或建议，请查看代码注释或联系开发团队*


