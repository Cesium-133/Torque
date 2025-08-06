import h5py
import numpy as np
import argparse
import datetime
import os
import multiprocessing as mp
from pathlib import Path
from joblib import dump
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf

# 从我们自己的model.py中导入模型创建函数
from model import create_torque_classifier, get_optimizer, get_lr_scheduler

# 设备配置
def configure_device(use_gpu=True, verbose=True):
    """配置训练设备（GPU或CPU）"""
    if verbose:
        print("配置训练设备...")
    
    if not use_gpu:
        # 强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if verbose:
            print("强制使用CPU训练")
        return False
    
    # 获取可用的GPU设备
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用内存增长，避免占用所有GPU内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU内存配置已通过set_memory_growth处理，无需额外设置虚拟设备
            
            if verbose:
                print(f"找到 {len(gpus)} 个GPU设备")
                for i, gpu in enumerate(gpus):
                    print(f"GPU {i}: {gpu.name}")
            
            # 启用混合精度训练（仅GPU）
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            if verbose:
                print(f"启用混合精度训练: {policy.name}")
            
            # 启用XLA编译优化
            tf.config.optimizer.set_jit(True)
            if verbose:
                print("启用XLA编译优化")
            
            return True
            
        except RuntimeError as e:
            if verbose:
                print(f"GPU配置错误: {e}")
                print("回退到CPU训练")
            return False
    else:
        if verbose:
            print("未找到GPU设备，将使用CPU训练")
        return False

def load_single_h5_file(file_path, target_length=150):
    """加载单个H5文件的数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            # 检查数据结构
            if '/right_arm_effort' not in f or '/labels' not in f:
                return None, None, file_path.name
            
            # 加载特征和标签
            effort_data = f['/right_arm_effort'][:]
            label_data = f['/labels'][:]
            
            # 确保数据长度一致
            min_len = min(len(effort_data), len(label_data))
            effort_data = effort_data[:min_len]
            label_data = label_data[:min_len]
            
            # Padding或截断到目标长度
            if len(effort_data) > target_length:
                effort_data = effort_data[:target_length]
                label_data = label_data[:target_length]
            elif len(effort_data) < target_length:
                pad_length = target_length - len(effort_data)
                effort_pad = np.zeros((pad_length, effort_data.shape[1]))
                label_pad = np.full(pad_length, label_data[-1])
                
                effort_data = np.concatenate([effort_data, effort_pad], axis=0)
                label_data = np.concatenate([label_data, label_pad], axis=0)
            
            return effort_data, label_data, None
    except Exception as e:
        return None, None, f"{file_path.name}: {e}"

def load_data_parallel(dir_path: Path, target_length: int = 150, max_workers: int = None, verbose: bool = True):
    """并行加载H5文件数据"""
    all_h5_files = list(dir_path.rglob('*.h5')) + list(dir_path.rglob('*.hdf5'))
    if not all_h5_files:
        raise FileNotFoundError(f"在目录 '{dir_path}' 中没有找到任何 .h5 或 .hdf5 文件。")

    if verbose:
        print(f"找到了 {len(all_h5_files)} 个 H5 文件。开始并行加载...")
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # 限制最大线程数避免过多I/O竞争
    
    all_sequences = []
    all_labels = []
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有加载任务
        future_to_file = {
            executor.submit(load_single_h5_file, file_path, target_length): file_path 
            for file_path in all_h5_files
        }
        
        # 收集结果
        progress_bar = tqdm(total=len(all_h5_files), desc="并行加载文件", disable=not verbose)
        for future in as_completed(future_to_file):
            effort_data, label_data, error = future.result()
            
            if effort_data is not None and label_data is not None:
                all_sequences.append(effort_data)
                all_labels.append(label_data)
            else:
                failed_files.append(error)
            
            progress_bar.update(1)
        progress_bar.close()
    
    if failed_files and verbose:
        print(f"\033[93m警告: {len(failed_files)} 个文件加载失败:\033[0m")
        for error in failed_files[:3]:  # 只显示前3个错误
            print(f"  - {error}")
        if len(failed_files) > 3:
            print(f"  ... 以及其他 {len(failed_files) - 3} 个文件")
    
    if not all_sequences:
        raise ValueError("未能从任何文件中成功加载数据。")
    
    sequences = np.array(all_sequences)
    labels = np.array(all_labels)
    
    if verbose:
        print(f"并行数据加载完成。sequences.shape: {sequences.shape}, labels.shape: {labels.shape}")
    return sequences, labels

def create_sliding_window_samples_optimized(sequences, labels, n_frames=10, m_frames=2, step_size=1, verbose=True):
    """优化的滑动窗口样本生成，减少内存分配"""
    if verbose:
        print(f"创建滑动窗口样本: n_frames={n_frames}, m_frames={m_frames}, step_size={step_size}")
    
    # 预计算总样本数
    total_samples = 0
    for seq in sequences:
        seq_len = len(seq)
        samples_per_seq = max(0, (seq_len - n_frames - m_frames + 1 + step_size - 1) // step_size)
        total_samples += samples_per_seq
    
    if verbose:
        print(f"预计生成 {total_samples} 个滑动窗口样本")
    
    # 预分配数组
    n_features = sequences[0].shape[1]
    X_windows = np.empty((total_samples, n_frames, n_features), dtype=np.float32)
    y_windows = np.empty((total_samples, m_frames), dtype=np.int32)
    
    sample_idx = 0
    progress_bar = tqdm(range(len(sequences)), desc="生成滑动窗口", disable=not verbose)
    for seq_idx in progress_bar:
        sequence = sequences[seq_idx]
        label_sequence = labels[seq_idx]
        seq_len = len(sequence)
        
        # 批量生成滑动窗口
        for start_idx in range(0, seq_len - n_frames - m_frames + 1, step_size):
            # 输入：过去n_frames帧
            X_windows[sample_idx] = sequence[start_idx:start_idx + n_frames]
            
            # 输出：未来m_frames帧的标签
            y_windows[sample_idx] = label_sequence[start_idx + n_frames:start_idx + n_frames + m_frames]
            
            sample_idx += 1
    
    # 裁剪到实际大小（防止预估不准确）
    X_windows = X_windows[:sample_idx]
    y_windows = y_windows[:sample_idx]
    
    if verbose:
        print(f"滑动窗口采样完成: X_windows.shape={X_windows.shape}, y_windows.shape={y_windows.shape}")
    return X_windows, y_windows

def create_tf_dataset(X, y, batch_size, shuffle=True, prefetch_buffer=tf.data.AUTOTUNE):
    """创建优化的TensorFlow数据集"""
    # 创建数据集
    if isinstance(y, list):
        # 多输出情况
        dataset = tf.data.Dataset.from_tensor_slices((X, tuple(y)))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        # 使用较大的缓冲区进行洗牌
        buffer_size = min(len(X), 10000)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    # 批处理
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # 预取数据以重叠数据加载和训练
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset

def get_optimal_batch_size(use_gpu=True):
    """根据设备类型自动确定最优批处理大小"""
    if not use_gpu:
        return 32  # CPU默认批处理大小
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return 32  # 没有GPU时使用CPU默认值
    
    try:
        # 获取GPU内存信息
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        # 基于GPU内存估算合适的批处理大小
        # 这是一个简单的启发式方法，实际应该根据模型大小调整
        return 256  # 对于大多数现代GPU的推荐值
    except:
        return 128  # 默认值

def load_data_from_directory(dir_path: Path, target_length: int = 150, verbose: bool = True):
    """从目录加载H5文件中的时序数据和标签（优化版本）"""
    return load_data_parallel(dir_path, target_length, verbose=verbose)

def train_single_fold_optimized(X_train, X_val, y_train, y_val, args, model_name_base, 
                               saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES, fold=None):
    """优化版本的单fold训练"""
    
    # 准备标签数据
    if M_FRAMES == 1:
        y_train_cat = to_categorical(y_train.squeeze(), num_classes=3)
        y_val_cat = to_categorical(y_val.squeeze(), num_classes=3)
    else:
        y_train_cat = [to_categorical(y_train[:, i], num_classes=3) for i in range(M_FRAMES)]
        y_val_cat = [to_categorical(y_val[:, i], num_classes=3) for i in range(M_FRAMES)]
    
    # 创建TensorFlow数据集
    train_dataset = create_tf_dataset(X_train, y_train_cat, args.batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_val, y_val_cat, args.batch_size, shuffle=False)
    
    # 构建和编译模型
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
        model.compile(
            optimizer=optimizer,
            loss=['categorical_crossentropy'] * M_FRAMES,
            loss_weights=[1.0] * M_FRAMES,
            metrics=['accuracy'] * M_FRAMES
        )
    
    # 只在详细模式或第一折时显示模型结构
    if args.verbose and (fold is None or fold == 0):
        model.summary()
    
    # 设置回调函数
    if fold is not None:
        log_dir = logs_dir / "fit" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_fold{fold+1}"
        model_path = saved_models_dir / f"{model_name_base}_fold{fold+1}_model.h5"
    else:
        device_suffix = "gpu" if args.use_gpu else "cpu"
        log_dir = logs_dir / "fit" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{device_suffix}"
        model_path = saved_models_dir / f"{model_name_base}_{device_suffix}_model.h5"
    
    # 根据verbose设置回调的详细程度
    verbose_level = 1 if args.verbose else 0
    
    callbacks_list = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=verbose_level,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            verbose=verbose_level,
            mode='min',
            restore_best_weights=True
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        lr_scheduler
    ]
    
    # 训练模型
    fold_text = f"Fold {fold + 1}" if fold is not None else "模型"
    device_text = "GPU" if args.use_gpu else "CPU"
    if args.verbose:
        print(f"\n开始训练 {fold_text} ({device_text}优化版本)...")
    
    # 使用fit方法训练，传入数据集
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=verbose_level
    )
    
    # 评估模型
    if args.verbose:
        print(f"\n评估 {fold_text}...")
    val_results = model.evaluate(val_dataset, verbose=0)
    val_loss = val_results[0]
    val_accuracy = val_results[1] if M_FRAMES == 1 else np.mean(val_results[1:M_FRAMES+1])
    
    # 清理内存
    del model, train_dataset, val_dataset
    tf.keras.backend.clear_session()
    gc.collect()
    
    return val_loss, val_accuracy

def main(args):
    # 配置设备
    is_gpu = configure_device(use_gpu=args.use_gpu, verbose=args.verbose)
    args.use_gpu = is_gpu  # 更新实际使用的设备状态
    
    # --- 1. 定义路径和参数 ---
    project_root = Path(__file__).parent.parent
    saved_models_dir = project_root / "checkpoints"
    logs_dir = project_root / "logs"
    data_dir_path = project_root / args.data_path

    saved_models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # 超参数
    TARGET_SEQUENCE_LENGTH = 150
    N_FRAMES = args.n_frames
    M_FRAMES = args.m_frames
    STEP_SIZE = 1
    
    # 自动优化批处理大小
    if args.auto_batch_size:
        optimal_batch_size = get_optimal_batch_size(use_gpu=args.use_gpu)
        if args.verbose:
            print(f"自动设置批处理大小: {optimal_batch_size}")
        args.batch_size = optimal_batch_size

    if args.verbose:
        device_name = "GPU" if args.use_gpu else "CPU"
        print(f"{device_name}优化超参数设置: N_FRAMES={N_FRAMES}, M_FRAMES={M_FRAMES}, BATCH_SIZE={args.batch_size}")
        print(f"训练模式: {'5-Fold 交叉验证' if args.use_cross_validation else '简单训练验证分割'}")

    # --- 2. 并行加载数据 ---
    if args.verbose:
        print("开始并行加载数据...")
    all_sequences = []
    all_labels = []
    
    # 从三个类别目录并行加载数据
    for class_dir in ['left', 'mid', 'right']:
        class_path = data_dir_path / class_dir
        if class_path.exists():
            sequences, labels = load_data_from_directory(class_path, TARGET_SEQUENCE_LENGTH, verbose=args.verbose)
            all_sequences.append(sequences)
            all_labels.append(labels)
        else:
            if args.verbose:
                print(f"\033[93m警告: 目录 {class_path} 不存在，已跳过。\033[0m")
    
    if not all_sequences:
        print("\033[91m没有加载到任何数据，程序终止。\033[0m")
        return
    
    # 合并所有数据
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if args.verbose:
        print(f"总数据形状: sequences={sequences.shape}, labels={labels.shape}")

    # --- 3. 优化的滑动窗口样本创建 ---
    X_windows, y_windows = create_sliding_window_samples_optimized(
        sequences, labels, N_FRAMES, M_FRAMES, STEP_SIZE, verbose=args.verbose
    )
    
    # --- 4. 数据标准化 ---
    if args.verbose:
        print("开始数据标准化...")
    scaler = StandardScaler()
    n_windows, n_frames, n_features = X_windows.shape
    
    # 重塑数据进行标准化
    X_reshaped = X_windows.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(n_windows, n_frames, n_features)
    
    # 根据设备类型选择数据类型
    if args.use_gpu:
        X_scaled = X_scaled.astype(np.float32)  # GPU使用float32节省内存
    else:
        X_scaled = X_scaled.astype(np.float64)  # CPU可以使用float64获得更好精度
    
    # 保存scaler
    device_suffix = "gpu" if args.use_gpu else "cpu"
    model_name_base = f"{args.model_type}_n{N_FRAMES}_m{M_FRAMES}_len{TARGET_SEQUENCE_LENGTH}_{device_suffix}"
    scaler_path = saved_models_dir / f"{model_name_base}_scaler.joblib"
    dump(scaler, scaler_path)
    if args.verbose:
        print(f"Scaler saved to {scaler_path}")

    # --- 5. 选择训练模式 ---
    if args.use_cross_validation:
        if args.verbose:
            print("优化版本暂时使用简单分割训练以获得最佳性能")
        args.use_cross_validation = False
    
    if not args.use_cross_validation:
        # 使用优化版本的简单分割训练
        if args.verbose:
            device_name = "GPU" if args.use_gpu else "CPU"
            print(f"\n使用{device_name}优化版本的训练验证分割 (训练集: {args.train_split:.1%}, 验证集: {1-args.train_split:.1%})...")
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_windows, 
            test_size=1-args.train_split, 
            random_state=42, 
            stratify=y_windows[:, 0] if len(y_windows.shape) > 1 else y_windows
        )
        
        if args.verbose:
            print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
        
        # 训练模型
        val_loss, val_accuracy = train_single_fold_optimized(
            X_train, X_val, y_train, y_val, args, model_name_base, 
            saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES, fold=None
        )
        
        # 输出结果
        device_name = "GPU" if args.use_gpu else "CPU"
        print(f"\n{'='*60}")
        print(f"{device_name}优化训练结果")
        print(f"{'='*60}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        
        # 保存结果
        results_file = saved_models_dir / f"{model_name_base}_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"{device_name} Optimized Training Results\n")
            f.write(f"Model: {args.model_type}\n")
            f.write(f"Device: {device_name}\n")
            f.write(f"N_frames: {N_FRAMES}, M_frames: {M_FRAMES}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            if args.use_gpu:
                f.write(f"Mixed Precision: Enabled\n")
                f.write(f"XLA Compilation: Enabled\n")
            f.write(f"Parallel Data Loading: Enabled\n")
            f.write(f"Train Split: {args.train_split:.1%}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        
        if args.verbose:
            print(f"\n结果已保存至: {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="优化版本的扭矩分类器训练脚本（支持GPU/CPU切换）")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, default="data/processed/train+val",
                        help="数据目录路径")
    
    # 模型相关参数
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "cnn", "lstm", "flatten"],
                        help="模型类型")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="输入历史帧数")
    parser.add_argument("--m_frames", type=int, default=2,
                        help="预测未来帧数")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=200,
                        help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="批处理大小")
    parser.add_argument("--auto_batch_size", action="store_true",
                        help="自动根据设备类型确定最优批处理大小")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                        help="优化器类型")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="初始学习率")
    
    # 设备相关参数
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="使用GPU训练（默认：True）")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false",
                        help="强制使用CPU训练")
    
    # 输出控制参数
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="详细输出模式（默认：True）")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="安静模式，减少输出信息")
    
    # 回调相关参数
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="早停耐心值")
    parser.add_argument("--lr_patience", type=int, default=10,
                        help="学习率调度器耐心值")
    
    # 训练模式相关参数
    parser.add_argument("--use_cross_validation", action="store_true", default=False,
                        help="使用5-Fold交叉验证（优化版本默认使用简单分割）")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="训练集比例")
    
    args = parser.parse_args()
    main(args) 