import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
import numpy as np
import argparse
import datetime
from pathlib import Path
from joblib import dump
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 从我们自己的model.py中导入模型创建函数
from model import create_torque_regressor, get_optimizer

class TorqueDataset(Dataset):
    """PyTorch数据集类，用于处理时序回归数据"""
    
    def __init__(self, X, y, positions):
        """
        Args:
            X: 输入序列数据 (n_samples, n_frames, n_features)
            y: 目标torque值 (n_samples, m_frames)
            positions: 位置编码 (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.positions = torch.LongTensor(positions)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.positions[idx]

def create_sliding_window_samples(sequences, n_frames=10, m_frames=2, step_size=1):
    """
    使用滑动窗口从时序数据中创建训练样本（回归任务）
    
    Args:
        sequences (np.ndarray): 形状为 (n_samples, seq_len, n_features) 的时序数据
        n_frames (int): 输入的历史帧数
        m_frames (int): 预测的未来帧数
        step_size (int): 滑动窗口的步长
        
    Returns:
        tuple: (X, y, positions) 其中X是输入序列，y是目标torque值，positions是位置编码
    """
    X_windows = []
    y_windows = []
    positions = []
    
    print(f"创建滑动窗口样本（回归任务）: n_frames={n_frames}, m_frames={m_frames}, step_size={step_size}")
    
    for seq_idx in tqdm(range(len(sequences)), desc="Processing sequences"):
        sequence = sequences[seq_idx]  # (seq_len, n_features)
        seq_len = len(sequence)
        
        # 滑动窗口采样
        for start_idx in range(0, seq_len - n_frames - m_frames + 1, step_size):
            # 输入：过去n_frames帧
            input_window = sequence[start_idx:start_idx + n_frames]  # (n_frames, n_features)
            
            # 输出：未来m_frames帧的torque信息（假设使用第一个特征维度作为torque）
            # 这里我们使用未来帧的第一个特征作为目标torque值
            target_torque = sequence[start_idx + n_frames:start_idx + n_frames + m_frames, 0]  # (m_frames,)
            
            X_windows.append(input_window)
            y_windows.append(target_torque)
            # 位置编码：使用窗口在序列中的相对位置
            positions.append(start_idx)
    
    X_windows = np.array(X_windows)  # (n_windows, n_frames, n_features)
    y_windows = np.array(y_windows)  # (n_windows, m_frames)
    positions = np.array(positions)  # (n_windows,)
    
    print(f"滑动窗口采样完成: X_windows.shape={X_windows.shape}, y_windows.shape={y_windows.shape}, positions.shape={positions.shape}")
    return X_windows, y_windows, positions

def load_data_from_directory(dir_path: Path, target_length: int = 150):
    """
    从目录加载H5文件中的时序数据（仅特征数据，不再读取标签）
    
    Args:
        dir_path (Path): 包含H5文件的数据目录路径
        target_length (int): 目标序列长度，用于padding或truncation
        
    Returns:
        sequences: 特征数据
    """
    all_h5_files = list(dir_path.rglob('*.h5')) + list(dir_path.rglob('*.hdf5'))
    if not all_h5_files:
        raise FileNotFoundError(f"在目录 '{dir_path}' 中没有找到任何 .h5 或 .hdf5 文件。")

    print(f"找到了 {len(all_h5_files)} 个 H5 文件。开始加载和处理...")

    all_sequences = []

    for file_path in tqdm(all_h5_files, desc="Loading files"):
        try:
            with h5py.File(file_path, 'r') as f:
                # 读取torque数据
                if '/right_arm_effort' not in f:
                    print(f"\033[93m警告: 在文件 {file_path.name} 中没有找到所需数据，已跳过。\033[0m")
                    continue
                
                # 加载特征数据
                effort_data = f['/right_arm_effort'][:]  # (seq_len, 7)
                
                # Padding或截断到目标长度
                if len(effort_data) > target_length:
                    # 截断
                    effort_data = effort_data[:target_length]
                elif len(effort_data) < target_length:
                    # Padding
                    pad_length = target_length - len(effort_data)
                    effort_pad = np.zeros((pad_length, effort_data.shape[1]))
                    effort_data = np.concatenate([effort_data, effort_pad], axis=0)
                
                all_sequences.append(effort_data)

        except Exception as e:
            print(f"\033[91m处理文件 {file_path.name} 时出错: {e}，已跳过。\033[0m")

    if not all_sequences:
        raise ValueError("未能从任何文件中成功加载数据。")

    sequences = np.array(all_sequences)  # (n_files, target_length, 7)
    
    print(f"数据加载完成。sequences.shape: {sequences.shape}")
    return sequences

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_x, batch_y, batch_pos in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_pos = batch_pos.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(batch_x, batch_pos)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)
    
    return total_loss / total_samples

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_pos in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_pos = batch_pos.to(device)
            
            outputs = model(batch_x, batch_pos)
            loss = criterion(outputs, batch_y)
            mae = nn.L1Loss()(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            total_mae += mae.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
    
    return total_loss / total_samples, total_mae / total_samples

def train_with_cross_validation(X_scaled, y_windows, positions, args, model_name_base, saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES):
    """使用交叉验证进行训练（回归任务）"""
    N_FOLDS = 5
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\n开始 {N_FOLDS}-Fold 交叉验证（回归任务）...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        # 分割数据
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_windows[train_idx], y_windows[val_idx]
        pos_train, pos_val = positions[train_idx], positions[val_idx]
        
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
        
        # 训练单个fold
        val_loss, val_mae = train_single_fold(
            X_train, X_val, y_train, y_val, pos_train, pos_val, args, model_name_base, 
            saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES, fold
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        print(f"Fold {fold + 1} - 验证MSE: {val_loss:.4f}, 验证MAE: {val_mae:.4f}")
    
    # 汇总结果
    print_cv_results(fold_results, model_name_base, saved_models_dir, args, N_FRAMES, M_FRAMES)

def train_simple_split(X_scaled, y_windows, positions, args, model_name_base, saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES):
    """使用简单的训练验证分割进行训练（回归任务）"""
    print(f"\n使用简单的训练验证分割 (训练集: {args.train_split:.1%}, 验证集: {1-args.train_split:.1%})...")
    
    # 分割数据
    X_train, X_val, y_train, y_val, pos_train, pos_val = train_test_split(
        X_scaled, y_windows, positions,
        test_size=1-args.train_split, 
        random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
    
    # 训练模型
    val_loss, val_mae = train_single_fold(
        X_train, X_val, y_train, y_val, pos_train, pos_val, args, model_name_base, 
        saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES, fold=None
    )
    
    print(f"\n{'='*60}")
    print("简单分割训练结果（回归任务）")
    print(f"{'='*60}")
    print(f"验证MSE: {val_loss:.4f}")
    print(f"验证MAE: {val_mae:.4f}")
    
    # 保存结果
    results_file = saved_models_dir / f"{model_name_base}_simple_split_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Simple Train-Validation Split Results (Regression)\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"N_frames: {N_FRAMES}, M_frames: {M_FRAMES}\n")
        f.write(f"Train Split: {args.train_split:.1%}\n")
        f.write(f"Validation MSE: {val_loss:.4f}\n")
        f.write(f"Validation MAE: {val_mae:.4f}\n")
    
    print(f"\n结果已保存至: {results_file}")

def train_single_fold(X_train, X_val, y_train, y_val, pos_train, pos_val, args, model_name_base, saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES, fold=None):
    """训练单个fold或单次训练（回归任务）"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    train_dataset = TorqueDataset(X_train, y_train, pos_train)
    val_dataset = TorqueDataset(X_val, y_val, pos_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 构建模型
    model = create_torque_regressor(
        input_shape=(N_FRAMES, n_features),
        output_dim=M_FRAMES,
        model_type=args.model_type,
        n_frames=N_FRAMES,
        m_frames=M_FRAMES
    )
    model = model.to(device)
    
    # 获取优化器和学习率调度器
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=1e-7
    )
    
    # 损失函数
    criterion = nn.MSELoss()
    
    if fold is None or fold == 0:  # 只在第一折或单次训练时显示模型结构
        print(f"模型结构:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
    
    # 设置日志和模型保存路径
    if fold is not None:
        log_dir = logs_dir / "fit" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_fold{fold+1}"
        model_path = saved_models_dir / f"{model_name_base}_fold{fold+1}_model.pth"
    else:
        log_dir = logs_dir / "fit" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_simple_split"
        model_path = saved_models_dir / f"{model_name_base}_model.pth"
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    fold_text = f"Fold {fold + 1}" if fold is not None else "模型"
    print(f"\n开始训练 {fold_text}（回归任务）...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE/Val', val_mae, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # 打印进度
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'model_config': {
                    'input_shape': (N_FRAMES, n_features),
                    'output_dim': M_FRAMES,
                    'model_type': args.model_type,
                    'n_frames': N_FRAMES,
                    'm_frames': M_FRAMES
                }
            }, model_path)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= args.early_stopping_patience:
            print(f"早停于 epoch {epoch+1}")
            break
    
    writer.close()
    
    print(f"\n{fold_text}训练完成！最佳验证MSE: {best_val_loss:.4f}, 最佳验证MAE: {best_val_mae:.4f}")
    print(f"模型已保存至: {model_path}")
    
    return best_val_loss, best_val_mae

def print_cv_results(fold_results, model_name_base, saved_models_dir, args, N_FRAMES, M_FRAMES):
    """打印交叉验证结果（回归任务）"""
    print(f"\n{'='*60}")
    print("5-Fold 交叉验证结果汇总（回归任务）")
    print(f"{'='*60}")
    
    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_mae = np.mean([r['val_mae'] for r in fold_results])
    std_loss = np.std([r['val_loss'] for r in fold_results])
    std_mae = np.std([r['val_mae'] for r in fold_results])
    
    for result in fold_results:
        print(f"Fold {result['fold']}: MSE={result['val_loss']:.4f}, MAE={result['val_mae']:.4f}")
    
    print(f"\n平均验证MSE: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"平均验证MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    
    # 保存结果
    results_file = saved_models_dir / f"{model_name_base}_cv_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("5-Fold Cross Validation Results (Regression)\n")
        f.write(f"Model: {args.model_type}\n")
        f.write(f"N_frames: {N_FRAMES}, M_frames: {M_FRAMES}\n")
        f.write(f"Average MSE: {avg_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f} ± {std_mae:.4f}\n\n")
        for result in fold_results:
            f.write(f"Fold {result['fold']}: MSE={result['val_loss']:.4f}, MAE={result['val_mae']:.4f}\n")
    
    print(f"\n结果已保存至: {results_file}")

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

    print(f"超参数设置（回归任务）: N_FRAMES={N_FRAMES}, M_FRAMES={M_FRAMES}, STEP_SIZE={STEP_SIZE}")
    print(f"训练模式: {'5-Fold 交叉验证' if args.use_cross_validation else '简单训练验证分割'}")

    # --- 2. 加载数据 ---
    print("开始加载数据（回归任务）...")
    all_sequences = []
    
    # 从数据目录加载所有H5文件（不再区分类别目录）
    sequences = load_data_from_directory(data_dir_path, TARGET_SEQUENCE_LENGTH)
    
    print(f"总数据形状: sequences={sequences.shape}")

    # --- 3. 创建滑动窗口样本 ---
    X_windows, y_windows, positions = create_sliding_window_samples(
        sequences, N_FRAMES, M_FRAMES, STEP_SIZE
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
    model_name_base = f"{args.model_type}_regressor_n{N_FRAMES}_m{M_FRAMES}_len{TARGET_SEQUENCE_LENGTH}"
    scaler_path = saved_models_dir / f"{model_name_base}_scaler.joblib"
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # --- 5. 选择训练模式 ---
    if args.use_cross_validation:
        train_with_cross_validation(X_scaled, y_windows, positions, args, model_name_base, saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES)
    else:
        train_simple_split(X_scaled, y_windows, positions, args, model_name_base, saved_models_dir, logs_dir, n_features, N_FRAMES, M_FRAMES)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GRU-based sequence regression model with 5-fold cross validation.")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, default="data/processed/train+val",
                        help="Path to the directory containing HDF5 data files, relative to project root.")
    
    # 模型相关参数
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "lstm"],
                        help="Type of model to train.")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of input frames (past frames).")
    parser.add_argument("--m_frames", type=int, default=2,
                        help="Number of output frames (future frames to predict).")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128,
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
    
    # 训练模式相关参数
    parser.add_argument("--use_cross_validation", action="store_true", default=True,
                        help="Use 5-fold cross validation for training (default: True).")
    parser.add_argument("--no_cross_validation", dest="use_cross_validation", action="store_false",
                        help="Use simple train-validation split instead of cross validation.")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Training set ratio when not using cross validation (default: 0.8).")
    
    args = parser.parse_args()
    main(args)