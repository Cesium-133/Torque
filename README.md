# 扭矩分类器 (Torque Classifier)

这是一个基于深度学习的扭矩传感器数据分类项目，用于对机器人手臂的力矩数据进行自动分类识别。

## 项目概述

本项目使用时序深度学习模型对H5格式的扭矩传感器数据进行三分类：
- **Left (左侧)**: 标签 0
- **Mid (中间)**: 标签 1  
- **Right (右侧)**: 标签 2

支持三种不同的模型架构：CNN、LSTM和Flatten，可根据数据特性选择最适合的模型类型。

## 项目结构

```
Torque_Classifier/
├── README.md              # 项目说明文档
├── requirements.txt       # Python依赖包列表
├── data/                  # 数据目录
│   ├── raw/              # 原始H5数据文件
│   │   ├── train+val/    # 训练和验证数据
│   │   │   ├── left/     # 左侧类别数据
│   │   │   ├── mid/      # 中间类别数据
│   │   │   └── right/    # 右侧类别数据
│   │   └── test/         # 测试数据
│   │       ├── left/
│   │       ├── mid/
│   │       └── right/
│   └── processed/        # 预处理后的数据（自动生成）
├── checkpoints/          # 保存的模型和标准化器
├── logs/                 # TensorBoard日志文件
├── src/                  # 源代码目录
│   ├── model.py          # 模型定义
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
提供三种不同的深度学习模型架构：

- **CNN模型** (`model_type='cnn'`): 使用1D卷积神经网络，适合捕捉时序数据中的局部特征
- **LSTM模型** (`model_type='lstm'`): 使用长短期记忆网络，适合处理长序列依赖关系
- **Flatten模型** (`model_type='flatten'`): 简单的全连接网络，作为基线模型

**使用方法:**
```python
from src.model import create_torque_classifier

# 创建CNN模型
model = create_torque_classifier(
    input_shape=(150, 7),  # 时序长度150，特征维度7
    num_classes=3,         # 三分类
    model_type='cnn'       # 模型类型
)
```

### 2. 训练模块 (`src/train.py`)
负责数据加载、预处理、模型训练和评估。

**主要功能:**
- 从H5文件加载扭矩数据
- 自动序列长度标准化（填充/截断）
- 数据标准化和标签编码
- 模型训练和验证
- 自动保存最佳模型和标准化器

**使用方法:**
```bash
# 使用默认参数训练CNN模型
python src/train.py --data_path data/raw/train+val --model_type cnn

# 训练LSTM模型，指定训练轮数和批大小
python src/train.py --data_path data/raw/train+val --model_type lstm --epochs 100 --batch_size 64
```

**参数说明:**
- `--data_path`: 数据目录路径，应包含left/mid/right三个子文件夹
- `--model_type`: 模型类型，可选 'cnn', 'lstm', 'flatten'
- `--epochs`: 最大训练轮数（默认200）
- `--batch_size`: 批处理大小（默认32）

### 3. 推理模块 (`src/infer.py`)
用于对新的H5文件进行分类预测。

**功能特点:**
- 自动加载训练好的模型和标准化器
- 应用与训练时完全相同的预处理流程
- 输出预测类别和置信度

**使用方法:**
```bash
# 对单个文件进行预测
python src/infer.py --model_path checkpoints/cnn_len150_model.h5 --input_file data/test/left/sample.h5
```

**参数说明:**
- `--model_path`: 训练好的模型文件路径
- `--input_file`: 要分类的H5文件路径

## 数据格式要求

### H5文件结构
每个H5文件应包含以下数据结构：
```
/upper_body_observations/right_arm_effort
```

该数据集应为形状为 `(sequence_length, 7)` 的时序数据，其中：
- `sequence_length`: 时序长度（可变）
- `7`: 7维特征向量（扭矩传感器数据）

### 数据组织方式
训练数据按类别组织在不同文件夹中：
- `left/`: 左侧类别的H5文件
- `mid/`: 中间类别的H5文件  
- `right/`: 右侧类别的H5文件

## 训练流程

1. **数据加载**: 从指定目录递归加载所有H5文件
2. **序列标准化**: 将所有序列填充/截断到固定长度150
3. **数据标准化**: 使用StandardScaler进行Z-score标准化
4. **数据划分**: 8:2比例划分训练集和验证集
5. **模型训练**: 使用早停和模型检查点机制
6. **模型保存**: 自动保存最佳模型和标准化器

## 模型输出

训练完成后会在`checkpoints/`目录下生成：
- `{model_type}_len150_model.h5`: 训练好的模型文件
- `{model_type}_len150_scaler.joblib`: 数据标准化器

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

### 2. CUDA内存不足
如果GPU内存不足，可以：
- 减小批处理大小：`--batch_size 16`
- 使用CPU训练：设置环境变量 `CUDA_VISIBLE_DEVICES=""`

### 3. 数据加载失败
请检查：
- H5文件路径是否正确
- H5文件是否包含正确的数据结构
- 文件权限是否正确

## 性能优化建议

1. **数据预处理**: 考虑使用数据缓存减少重复加载时间
2. **模型选择**: CNN模型通常在时序分类任务中表现最佳
3. **超参数调优**: 可以调整学习率、网络层数等参数
4. **数据增强**: 考虑添加时序数据增强技术

## 版本更新日志

- **v1.0**: 初始版本，支持CNN/LSTM/Flatten三种模型
- 修复了Keras导入兼容性问题
- 完善了错误处理和日志输出

---
*如有问题或建议，请查看代码注释或联系开发团队*


