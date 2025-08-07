import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import h5py
import numpy as np
import argparse
from pathlib import Path
from joblib import load
import tensorflow as tf
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

def load_and_preprocess_single_file(file_path: Path, scaler, n_frames: int, target_length: int = 150):
    """
    加载单个H5文件，并应用与训练时完全相同的预处理（回归任务）。
    
    Args:
        file_path (Path): H5文件路径
        scaler: 训练时保存的标准化器
        n_frames (int): 输入历史帧数
        target_length (int): 目标序列长度
        
    Returns:
        tuple: (processed_sequences, positions) 处理后的序列和位置编码
    """
    with h5py.File(file_path, 'r') as f:
        if '/right_arm_effort' not in f:
            raise ValueError(f"在文件 {file_path.name} 中没有找到所需的数据结构。")
        
        # 加载数据
        effort_data = f['/right_arm_effort'][:]  # (seq_len, 7)
    
    # Padding或截断到目标长度
    if len(effort_data) > target_length:
        effort_data = effort_data[:target_length]
    elif len(effort_data) < target_length:
        pad_length = target_length - len(effort_data)
        effort_pad = np.zeros((pad_length, effort_data.shape[1]))
        effort_data = np.concatenate([effort_data, effort_pad], axis=0)
    
    # 创建滑动窗口序列用于推理
    sequences = []
    positions = []
    
    seq_len = len(effort_data)
    for start_idx in range(seq_len - n_frames + 1):
        input_window = effort_data[start_idx:start_idx + n_frames]  # (n_frames, n_features)
        sequences.append(input_window)
        positions.append(start_idx)  # 位置编码
    
    if not sequences:
        raise ValueError(f"序列长度 {seq_len} 小于所需的输入帧数 {n_frames}")
    
    # 转换为numpy数组并标准化
    sequences = np.array(sequences)  # (n_windows, n_frames, n_features)
    positions = np.array(positions)  # (n_windows,)
    n_windows, n_frames_check, n_features = sequences.shape
    
    # 标准化
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = scaler.transform(sequences_reshaped)
    sequences_final = sequences_scaled.reshape(n_windows, n_frames_check, n_features)
    
    return sequences_final, positions

def predict_torque_sequence(model, sequences, positions, m_frames=2):
    """
    对序列进行torque回归预测
    
    Args:
        model: 训练好的回归模型
        sequences: 输入序列 (n_windows, n_frames, n_features)
        positions: 位置编码 (n_windows,)
        m_frames: 预测的未来帧数
        
    Returns:
        predictions: 预测的torque值
    """
    predictions = model.predict([sequences, positions], verbose=0)
    print(f"🔍 Debug: model prediction shape = {predictions.shape}")
    print(f"🔍 Debug: prediction range - min: {predictions.min():.6f}, max: {predictions.max():.6f}, mean: {predictions.mean():.6f}")
    print(f"🔍 Debug: first few predictions = {predictions[:3].flatten()}")
    return predictions

def aggregate_torque_predictions(predictions, method='mean'):
    """
    聚合多个时间窗口的torque预测结果
    
    Args:
        predictions: 模型预测结果 (n_windows, m_frames)
        method: 聚合方法 ('mean', 'median', 'last')
        
    Returns:
        aggregated_prediction: 聚合后的预测结果
    """
    if method == 'mean':
        return np.mean(predictions, axis=0)  # 对所有窗口取平均
    elif method == 'median':
        return np.median(predictions, axis=0)  # 对所有窗口取中位数
    elif method == 'last':
        return predictions[-1]  # 使用最后一个窗口的预测
    else:
        return np.mean(predictions, axis=0)  # 默认使用平均值

def load_truth_values_from_file(file_path: Path, n_frames: int, m_frames: int, target_length: int = 150):
    """
    从H5文件中加载真实的torque值，用于可视化对比
    
    Args:
        file_path (Path): H5文件路径
        n_frames (int): 输入历史帧数
        m_frames (int): 预测的未来帧数
        target_length (int): 目标序列长度
        注意：真实值使用原始尺度，与训练时的目标值保持一致
        
    Returns:
        tuple: (truth_values, time_indices) 真实值和对应的时间索引
    """
    with h5py.File(file_path, 'r') as f:
        if '/right_arm_effort' not in f:
            raise ValueError(f"在文件 {file_path.name} 中没有找到所需的数据结构。")
        
        # 加载原始torque数据
        effort_data = f['/right_arm_effort'][:]  # (seq_len, 7)
    
    # 应用与训练时相同的预处理
    if len(effort_data) > target_length:
        effort_data = effort_data[:target_length]
    elif len(effort_data) < target_length:
        pad_length = target_length - len(effort_data)
        effort_pad = np.zeros((pad_length, effort_data.shape[1]))
        effort_data = np.concatenate([effort_data, effort_pad], axis=0)
    
    # 🚨 重要修复：训练时目标值y_windows是从原始数据提取的（未标准化）
    # 所以推理时的真实值也应该使用原始尺度，不进行标准化
    print("🔧 修复：使用原始尺度的真实值，与训练时的目标值保持一致")
    print(f"🔍 Debug: 原始数据范围 - min: {effort_data[:, 0].min():.6f}, max: {effort_data[:, 0].max():.6f}, mean: {effort_data[:, 0].mean():.6f}")
    effort_data_scaled = effort_data  # 不进行标准化
    
    # 提取真实的未来值用于对比
    # 对于每个预测窗口，提取对应的真实未来m_frames值
    truth_values = []
    time_indices = []
    
    seq_len = len(effort_data_scaled)
    for start_idx in range(seq_len - n_frames + 1):
        # 预测的时间点从 start_idx + n_frames 开始
        future_start = start_idx + n_frames
        if future_start + m_frames <= seq_len:
            # 提取真实的未来m_frames值（只取第一维）
            truth_future = effort_data_scaled[future_start:future_start + m_frames, 0]  # 只取第一维
            truth_values.append(truth_future)
            # 时间索引对应预测的时间点
            time_indices.append(list(range(future_start, future_start + m_frames)))
        else:
            # 如果超出范围，用零填充
            remaining_frames = seq_len - future_start
            if remaining_frames > 0:
                truth_future = effort_data_scaled[future_start:seq_len, 0]
                # 用零填充不足的帧数
                if m_frames - remaining_frames > 0:
                    truth_future = np.concatenate([truth_future, np.zeros(m_frames - remaining_frames)])
            else:
                truth_future = np.zeros(m_frames)
            truth_values.append(truth_future)
            time_indices.append(list(range(future_start, future_start + m_frames)))
    
    # 确保返回的数组有正确的形状
    truth_values = np.array(truth_values)
    print(f"Debug: truth_values.shape after processing = {truth_values.shape}")
    
    # 当m_frames=1时，确保形状是(n_windows, 1)而不是(n_windows,)
    if truth_values.ndim == 1:
        truth_values = truth_values.reshape(-1, 1)
    
    return truth_values, time_indices

def create_interactive_visualization(truth_values, predictions, time_indices, file_name, output_dir):
    """
    创建交互式可视化图表，显示真实值和预测值的对比
    
    Args:
        truth_values: 真实值 (n_windows, m_frames)
        predictions: 预测值 (n_windows, m_frames) 或 (n_windows,) 当m_frames=1时
        time_indices: 时间索引列表
        file_name: 文件名
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理预测数据的维度问题
    print(f"Debug: predictions.shape = {predictions.shape}")
    print(f"Debug: truth_values.shape = {truth_values.shape}")
    
    # 确保预测数据有正确的形状
    if predictions.ndim == 1:
        # 当m_frames=1时，predictions可能是(n_windows,)，需要reshape为(n_windows, 1)
        predictions = predictions.reshape(-1, 1)
    elif predictions.ndim == 2 and predictions.shape[1] == 1:
        # 已经是正确的形状(n_windows, 1)
        pass
    
    # 创建图表
    fig = go.Figure()
    
    # 添加真实值的实线
    # 将所有真实值连接成一条连续的线
    all_truth_times = []
    all_truth_values = []
    
    for i, (truth_seq, time_seq) in enumerate(zip(truth_values, time_indices)):
        # 确保truth_seq是一维数组
        if truth_seq.ndim > 1:
            truth_seq = truth_seq.flatten()
        all_truth_times.extend(time_seq)
        all_truth_values.extend(truth_seq)
    
    # 去重并排序，创建连续的真实值线
    time_truth_pairs = list(zip(all_truth_times, all_truth_values))
    time_truth_pairs = sorted(list(set(time_truth_pairs)))
    unique_times, unique_truths = zip(*time_truth_pairs)
    
    # 添加真实值实线
    fig.add_trace(go.Scatter(
        x=unique_times,
        y=unique_truths,
        mode='lines',
        name='Truth Values',
        line=dict(color='blue', width=3),
        hovertemplate='Time: %{x}<br>Truth: %{y:.6f}<extra></extra>'
    ))
    
    # 添加每个预测窗口的虚线
    for i, (pred_seq, time_seq) in enumerate(zip(predictions, time_indices)):
        # 处理预测值的维度
        if pred_seq.ndim > 1:
            pred_values = pred_seq[:, 0]  # 只取第一维
        else:
            pred_values = pred_seq if isinstance(pred_seq, np.ndarray) else [pred_seq]
        
        # 确保pred_values和time_seq长度匹配
        if len(pred_values) != len(time_seq):
            print(f"Warning: pred_values length {len(pred_values)} != time_seq length {len(time_seq)}")
            continue
            
        fig.add_trace(go.Scatter(
            x=time_seq,
            y=pred_values,
            mode='lines+markers',  # 添加markers使单点更明显
            name=f'Prediction Window {i+1}',
            line=dict(dash='dot', width=2, color=f'rgba(255, 0, 0, 0.7)'),
            marker=dict(size=4, color='red'),
            hovertemplate=f'Window {i+1}<br>Time: %{{x}}<br>Prediction: %{{y:.6f}}<extra></extra>',
            showlegend=(i == 0)  # 只在第一条预测线显示图例
        ))
    
    # 设置图表布局
    fig.update_layout(
        title=f'Torque Prediction Visualization - {file_name}',
        xaxis_title='Time Step',
        yaxis_title='Torque Value (First Dimension)',
        hovermode='closest',
        legend=dict(x=0.02, y=0.98),
        width=1200,
        height=600,
        template='plotly_white'
    )
    
    # 保存交互式HTML文件
    html_file = output_path / f'{file_name}_visualization.html'
    fig.write_html(html_file)
    
    print(f"交互式可视化已保存至: {html_file}")
    return html_file

def main(args):
    # --- 1. 解析模型参数 ---
    model_path = Path(args.model_path)
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 从模型文件名解析参数
    model_name_stem = model_path.stem
    try:
        # 解析模型文件名格式：model_type_regressor_nX_mY_lenZ_foldW_model 或 model_type_regressor_nX_mY_lenZ_model
        parts = model_name_stem.split('_')
        
        # 查找n_frames和m_frames参数
        n_frames = None
        m_frames = None
        target_length = None
        
        for part in parts:
            if part.startswith('n') and part[1:].isdigit():
                n_frames = int(part[1:])
            elif part.startswith('m') and part[1:].isdigit():
                m_frames = int(part[1:])
            elif part.startswith('len') and part[3:].isdigit():
                target_length = int(part[3:])
        
        if n_frames is None or m_frames is None or target_length is None:
            raise ValueError("无法解析模型参数")
            
    except (IndexError, ValueError) as e:
        print(f"\033[91m错误: 无法从模型文件名 '{model_path.name}' 中推断参数。\033[0m")
        print(f"文件名应遵循格式: 'modeltype_regressor_nX_mY_lenZ_model.h5'")
        return

    # 构建scaler路径
    scaler_name = model_name_stem.replace('_model', '_scaler.joblib')
    # 如果是fold模型，需要移除fold信息来找到对应的scaler
    scaler_name = '_'.join([part for part in scaler_name.split('_') if not part.startswith('fold')])
    scaler_path = model_path.parent / scaler_name

    if not model_path.exists() or not scaler_path.exists():
        print(f"\033[91m错误: 模型或Scaler未找到。请检查路径。\n模型: {model_path}\nScaler: {scaler_path}\033[0m")
        return

    print(f"模型参数: n_frames={n_frames}, m_frames={m_frames}, target_length={target_length}")

    # --- 2. 加载模型和Scaler ---
    print(f"正在加载回归模型: {model_path}")
    # 解决Keras版本兼容性问题：显式指定自定义对象
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError()
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"正在加载Scaler: {scaler_path}")
    scaler = load(scaler_path)

    # --- 3. 查找所有待推理的文件 ---
    h5_files_to_infer = list(input_dir.rglob('*.h5')) + list(input_dir.rglob('*.hdf5'))
    if not h5_files_to_infer:
        print(f"\033[93m在目录 '{input_dir}' 中没有找到任何 .h5 或 .hdf5 文件。\033[0m")
        return
    
    print(f"找到 {len(h5_files_to_infer)} 个文件。开始批量torque回归推理...")

    # --- 4. 批量推理并保存结果 ---
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 写入头部信息
        f_out.write(f"# GRU Torque Regression Results\n")
        f_out.write(f"# Model: {model_path.name}\n")
        f_out.write(f"# Parameters: n_frames={n_frames}, m_frames={m_frames}\n")
        f_out.write(f"# Aggregation Method: {args.aggregation_method}\n")
        f_out.write(f"# Format: filepath - predicted_torque_values\n\n")
        
        for file_path in tqdm(h5_files_to_infer, desc="Inferring"):
            try:
                # 4.1 预处理单个文件
                sequences, positions = load_and_preprocess_single_file(
                    file_path, scaler, n_frames, target_length
                )
                
                # 4.2 进行torque预测
                predictions = predict_torque_sequence(model, sequences, positions, m_frames)
                
                # 4.3 聚合预测结果
                aggregated_prediction = aggregate_torque_predictions(predictions, args.aggregation_method)
                
                # 4.4 格式化输出
                torque_values = [f"{val:.6f}" for val in aggregated_prediction]
                torque_str = "[" + ", ".join(torque_values) + "]"
                
                # 4.5 准备输出行
                output_line = f"{file_path.resolve()} - {torque_str}\n"
                f_out.write(output_line)
                
                # 如果需要详细输出，也可以保存每个时间步的预测
                if args.detailed_output:
                    f_out.write(f"  Detailed predictions for {file_path.name}:\n")
                    for i, pred in enumerate(predictions):
                        pred_values = [f"{val:.6f}" for val in pred]
                        pred_str = "[" + ", ".join(pred_values) + "]"
                        f_out.write(f"    Window {i}: {pred_str}\n")
                    f_out.write("\n")

                # 5. 加载真实值并进行可视化
                if args.visualize:
                    truth_values, time_indices = load_truth_values_from_file(
                        file_path, n_frames, m_frames, target_length
                    )
                    create_interactive_visualization(
                        truth_values, predictions, time_indices, file_path.stem, args.output_dir
                    )

            except Exception as e:
                error_line = f"{file_path.resolve()} - ERROR: {e}\n"
                f_out.write(error_line)
                tqdm.write(f"\033[91m处理文件 {file_path.name} 时出错: {e}\033[0m")

    print("\n" + "="*50)
    print("      GRU Torque回归推理完成")
    print("="*50)
    print(f"结果已保存至: {output_file.resolve()}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform GRU-based torque regression inference on H5 files.")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved Keras regression model file (e.g., 'checkpoints/gru_regressor_n10_m2_len150_model.h5').")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing H5 files for inference.")
    parser.add_argument("--output_file", type=str, default="output/torque_inference_results.txt",
                        help="Path to save the torque inference results.")
    parser.add_argument("--aggregation_method", type=str, default="mean", 
                        choices=["mean", "median", "last"],
                        help="Method to aggregate multiple window predictions.")
    parser.add_argument("--detailed_output", action="store_true",
                        help="Include detailed predictions for each time window.")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate interactive visualization HTML files.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save visualization HTML files.")

    args = parser.parse_args()
    main(args)