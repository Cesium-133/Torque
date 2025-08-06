import h5py
import numpy as np
import argparse
import datetime
from pathlib import Path
from joblib import dump
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf

# 从我们自己的model.py中导入模型创建函数
from model import create_torque_classifier, get_optimizer, get_lr_scheduler

def create_sliding_window_samples(sequences, labels, n_frames=10, m_frames=2, step_size=1):
    """
    使用滑动窗口从时序数据中创建训练样本
    
    Args:
        sequences (np.ndarray): 形状为 (n_samples, seq_len, n_features) 的时序数据
        labels (np.ndarray): 形状为 (n_samples, seq_len) 的标签数据
        n_frames (int): 输入的历史帧数
        m_frames (int): 预测的未来帧数
        step_size (int): 滑动窗口的步长
        
    Returns:
        tuple: (X, y) 其中X是输入序列，y是目标标签
    """
    X_windows = []
    y_windows = []
    
    print(f"创建滑动窗口样本: n_frames={n_frames}, m_frames={m_frames}, step_size={step_size}")
    
    for seq_idx in tqdm(range(len(sequences)), desc="Processing sequences"):
        sequence = sequences[seq_idx]  # (seq_len, n_features)
        label_sequence = labels[seq_idx]  # (seq_len,)
        seq_len = len(sequence)
        
        # 滑动窗口采样
        for start_idx in range(0, seq_len - n_frames - m_frames + 1, step_size):
            # 输入：过去n_frames帧
            input_window = sequence[start_idx:start_idx + n_frames]  # (n_frames, n_features)
            
            # 输出：未来m_frames帧的标签
            target_labels = label_sequence[start_idx + n_frames:start_idx + n_frames + m_frames]  # (m_frames,)
            
            X_windows.append(input_window)
            y_windows.append(target_labels)
    
    X_windows = np.array(X_windows)  # (n_windows, n_frames, n_features)
    y_windows = np.array(y_windows)  # (n_windows, m_frames)
    
    print(f"滑动窗口采样完成: X_windows.shape={X_windows.shape}, y_windows.shape={y_windows.shape}")
    return X_windows, y_windows

def load_data_from_directory(dir_path: Path, target_length: int = 150):
    """
    从目录加载H5文件中的时序数据和标签
    
    Args:
        dir_path (Path): 包含H5文件的数据目录路径
        target_length (int): 目标序列长度，用于padding或truncation
        
    Returns:
        tuple: (sequences, labels) 其中sequences是特征数据，labels是标签数据
    """
    all_h5_files = list(dir_path.rglob('*.h5')) + list(dir_path.rglob('*.hdf5'))
    if not all_h5_files:
        raise FileNotFoundError(f"在目录 '{dir_path}' 中没有找到任何 .h5 或 .hdf5 文件。")

    print(f"找到了 {len(all_h5_files)} 个 H5 文件。开始加载和处理...")

    all_sequences = []
    all_labels = []

    for file_path in tqdm(all_h5_files, desc="Loading files"):
        try:
            with h5py.File(file_path, 'r') as f:
                # 新的H5文件格式
                if '/right_arm_effort' not in f or '/labels' not in f:
                    print(f"\033[93m警告: 在文件 {file_path.name} 中没有找到所需数据，已跳过。\033[0m")
                    continue
                
                # 加载特征和标签
                effort_data = f['/right_arm_effort'][:]  # (seq_len, 7)
                label_data = f['/labels'][:]  # (seq_len,)
                
                # 确保数据长度一致
                min_len = min(len(effort_data), len(label_data))
                effort_data = effort_data[:min_len]
                label_data = label_data[:min_len]
                
                # Padding或截断到目标长度
                if len(effort_data) > target_length:
                    # 截断
                    effort_data = effort_data[:target_length]
                    label_data = label_data[:target_length]
                elif len(effort_data) < target_length:
                    # Padding
                    pad_length = target_length - len(effort_data)
                    effort_pad = np.zeros((pad_length, effort_data.shape[1]))
                    label_pad = np.full(pad_length, label_data[-1])  # 用最后一个标签填充
                    
                    effort_data = np.concatenate([effort_data, effort_pad], axis=0)
                    label_data = np.concatenate([label_data, label_pad], axis=0)
                
                all_sequences.append(effort_data)
                all_labels.append(label_data)

        except Exception as e:
            print(f"\033[91m处理文件 {file_path.name} 时出错: {e}，已跳过。\033[0m")

    if not all_sequences:
        raise ValueError("未能从任何文件中成功加载数据。")

    sequences = np.array(all_sequences)  # (n_files, target_length, 7)
    labels = np.array(all_labels)  # (n_files, target_length)
    
    print(f"数据加载完成。sequences.shape: {sequences.shape}, labels.shape: {labels.shape}")
    return sequences, labels

def custom_multi_output_loss(y_true_list, y_pred_list, m_frames=2):
    """
    自定义多输出损失函数，对m个输出取平均
    
    Args:
        y_true_list: 真实标签列表
        y_pred_list: 预测结果列表
        m_frames: 输出帧数
        
    Returns:
        平均交叉熵损失
    """
    total_loss = 0
    for i in range(m_frames):
        loss_i = tf.keras.losses.categorical_crossentropy(y_true_list[i], y_pred_list[i])
        total_loss += loss_i
    
    return total_loss / m_frames

def main(args):
    # --- 1. 定义路径和参数 ---
    project_root = Path(__file__).parent.parent
    saved_models_dir = project_root / "checkpoints"
    logs_dir = project_root / "logs"
    data_dir_path = project_root / args.data_path

    saved_models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # 超参数
    TARGET_SEQUENCE_LENGTH = 150  # 原始序列长度
    N_FRAMES = args.n_frames  # 输入历史帧数
    M_FRAMES = args.m_frames  # 预测未来帧数
    STEP_SIZE = 1  # 滑动窗口步长
    N_FOLDS = 5  # 交叉验证折数

    print(f"超参数设置: N_FRAMES={N_FRAMES}, M_FRAMES={M_FRAMES}, STEP_SIZE={STEP_SIZE}")

    # --- 2. 加载数据 ---
    print("开始加载数据...")
    all_sequences = []
    all_labels = []
    
    # 从三个类别目录加载数据
    for class_dir in ['left', 'mid', 'right']:
        class_path = data_dir_path / class_dir
        if class_path.exists():
            sequences, labels = load_data_from_directory(class_path, TARGET_SEQUENCE_LENGTH)
            all_sequences.append(sequences)
            all_labels.append(labels)
        else:
            print(f"\033[93m警告: 目录 {class_path} 不存在，已跳过。\033[0m")
    
    if not all_sequences:
        print("\033[91m没有加载到任何数据，程序终止。\033[0m")
        return
    
    # 合并所有数据
    sequences = np.concatenate(all_sequences, axis=0)  # (total_files, seq_len, 7)
    labels = np.concatenate(all_labels, axis=0)  # (total_files, seq_len)
    
    print(f"总数据形状: sequences={sequences.shape}, labels={labels.shape}")

    # --- 3. 创建滑动窗口样本 ---
    X_windows, y_windows = create_sliding_window_samples(
        sequences, labels, N_FRAMES, M_FRAMES, STEP_SIZE
    )
    
    # --- 4. 数据标准化 ---
    print("开始数据标准化...")
    scaler = StandardScaler()
    n_windows, n_frames, n_features = X_windows.shape
    
    # 重塑数据进行标准化
    X_reshaped = X_windows.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_windows, n_frames, n_features)
    
    # 保存scaler
    model_name_base = f"{args.model_type}_n{N_FRAMES}_m{M_FRAMES}_len{TARGET_SEQUENCE_LENGTH}"
    scaler_path = saved_models_dir / f"{model_name_base}_scaler.joblib"
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- 5. 5-Fold 交叉验证 ---
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\n开始 {N_FOLDS}-Fold 交叉验证...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        # 分割数据
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_windows[train_idx], y_windows[val_idx]
        
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
        
        # 准备标签数据
        if M_FRAMES == 1:
            # 单输出情况
            y_train_cat = to_categorical(y_train.squeeze(), num_classes=3)
            y_val_cat = to_categorical(y_val.squeeze(), num_classes=3)
        else:
            # 多输出情况
            y_train_cat = [to_categorical(y_train[:, i], num_classes=3) for i in range(M_FRAMES)]
            y_val_cat = [to_categorical(y_val[:, i], num_classes=3) for i in range(M_FRAMES)]
        
        # --- 6. 构建和编译模型 ---
        model = create_torque_classifier(
            input_shape=(N_FRAMES, n_features),
            num_classes=3,
            model_type=args.model_type,
            n_frames=N_FRAMES,
            m_frames=M_FRAMES
        )
        
        # 获取优化器和学习率调度器
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        lr_scheduler = get_lr_scheduler(patience=args.lr_patience)
        
        # 编译模型
        if M_FRAMES == 1:
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            # 多输出情况
            model.compile(
                optimizer=optimizer,
                loss=['categorical_crossentropy'] * M_FRAMES,
                loss_weights=[1.0] * M_FRAMES,  # 等权重
                metrics=['accuracy']
            )
        
        if fold == 0:  # 只在第一折显示模型结构
            model.summary()
        
        # --- 7. 设置回调函数 ---
        log_dir = logs_dir / "fit" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_fold{fold+1}"
        model_path = saved_models_dir / f"{model_name_base}_fold{fold+1}_model.h5"
        
        callbacks_list = [
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=args.early_stopping_patience,
                verbose=1,
                mode='min',
                restore_best_weights=True
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1),
            lr_scheduler
        ]
        
        # --- 8. 训练模型 ---
        print(f"\n开始训练 Fold {fold + 1}...")
        history = model.fit(
            X_train, y_train_cat,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # --- 9. 评估模型 ---
        print(f"\n评估 Fold {fold + 1}...")
        val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)[:2]
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        print(f"Fold {fold + 1} - 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
    
    # --- 10. 汇总结果 ---
    print(f"\n{'='*60}")
    print("5-Fold 交叉验证结果汇总")
    print(f"{'='*60}")
    
    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
    std_loss = np.std([r['val_loss'] for r in fold_results])
    std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
    
    for result in fold_results:
        print(f"Fold {result['fold']}: Loss={result['val_loss']:.4f}, Accuracy={result['val_accuracy']:.4f}")
    
    print(f"\n平均验证损失: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"平均验证准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # 保存结果
    results_file = saved_models_dir / f"{model_name_base}_cv_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("5-Fold Cross Validation Results\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"N_frames: {N_FRAMES}, M_frames: {M_FRAMES}\n")
        f.write(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: Loss={result['val_loss']:.4f}, Accuracy={result['val_accuracy']:.4f}\n")
    
    print(f"\n结果已保存至: {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GRU-based sequence prediction model with 5-fold cross validation.")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, default="data/processed/train+val",
                        help="Path to the directory containing HDF5 data files, relative to project root.")
    
    # 模型相关参数
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "cnn", "lstm", "flatten"],
                        help="Type of model to train.")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of input frames (past frames).")
    parser.add_argument("--m_frames", type=int, default=2,
                        help="Number of output frames (future frames to predict).")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                        help="Optimizer type.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate.")
    
    # 回调相关参数
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Early stopping patience.")
    parser.add_argument("--lr_patience", type=int, default=10,
                        help="Learning rate scheduler patience.")
    
    args = parser.parse_args()
    main(args)