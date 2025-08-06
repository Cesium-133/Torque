import h5py
import numpy as np
import argparse
from pathlib import Path
from joblib import load
import tensorflow as tf
from tqdm import tqdm

# 将数字类别映射到有意义的标签
CLASS_MAP = {
    0: 'left',
    1: 'mid', 
    2: 'right'
}

def load_and_preprocess_single_file(file_path: Path, scaler, n_frames: int, target_length: int = 150):
    """
    加载单个H5文件，并应用与训练时完全相同的预处理。
    
    Args:
        file_path (Path): H5文件路径
        scaler: 训练时保存的标准化器
        n_frames (int): 输入历史帧数
        target_length (int): 目标序列长度
        
    Returns:
        tuple: (processed_sequences, original_labels) 处理后的序列和原始标签
    """
    with h5py.File(file_path, 'r') as f:
        if '/right_arm_effort' not in f or '/labels' not in f:
            raise ValueError(f"在文件 {file_path.name} 中没有找到所需的数据结构。")
        
        # 加载数据
        effort_data = f['/right_arm_effort'][:]  # (seq_len, 7)
        label_data = f['/labels'][:]  # (seq_len,)
    
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
    
    # 创建滑动窗口序列用于推理
    sequences = []
    valid_positions = []  # 记录有效的预测位置
    
    seq_len = len(effort_data)
    for start_idx in range(seq_len - n_frames + 1):
        input_window = effort_data[start_idx:start_idx + n_frames]  # (n_frames, n_features)
        sequences.append(input_window)
        valid_positions.append(start_idx + n_frames - 1)  # 预测位置是窗口的最后一个位置
    
    if not sequences:
        raise ValueError(f"序列长度 {seq_len} 小于所需的输入帧数 {n_frames}")
    
    # 转换为numpy数组并标准化
    sequences = np.array(sequences)  # (n_windows, n_frames, n_features)
    n_windows, n_frames_check, n_features = sequences.shape
    
    # 标准化
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = scaler.transform(sequences_reshaped)
    sequences_final = sequences_scaled.reshape(n_windows, n_frames_check, n_features)
    
    return sequences_final, label_data, valid_positions

def predict_sequence(model, sequences, m_frames=2):
    """
    对序列进行预测
    
    Args:
        model: 训练好的模型
        sequences: 输入序列 (n_windows, n_frames, n_features)
        m_frames: 预测的未来帧数
        
    Returns:
        predictions: 预测结果
    """
    predictions = model.predict(sequences, verbose=0)
    
    if m_frames == 1:
        # 单输出情况
        return predictions
    else:
        # 多输出情况，返回每个输出的预测结果
        return predictions

def aggregate_predictions(predictions, m_frames=2):
    """
    聚合多帧预测结果
    
    Args:
        predictions: 模型预测结果
        m_frames: 预测帧数
        
    Returns:
        aggregated_predictions: 聚合后的预测结果
    """
    if m_frames == 1:
        return np.argmax(predictions, axis=1)
    else:
        # 对多个输出取平均
        avg_predictions = np.mean(predictions, axis=0)  # 对m_frames维度取平均
        return np.argmax(avg_predictions, axis=1)

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
        # 解析模型文件名格式：model_type_nX_mY_lenZ_foldW_model 或 model_type_nX_mY_lenZ_model
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
        print(f"文件名应遵循格式: 'modeltype_nX_mY_lenZ_model.h5'")
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
    print(f"正在加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"正在加载Scaler: {scaler_path}")
    scaler = load(scaler_path)

    # --- 3. 查找所有待推理的文件 ---
    h5_files_to_infer = list(input_dir.rglob('*.h5')) + list(input_dir.rglob('*.hdf5'))
    if not h5_files_to_infer:
        print(f"\033[93m在目录 '{input_dir}' 中没有找到任何 .h5 或 .hdf5 文件。\033[0m")
        return
    
    print(f"找到 {len(h5_files_to_infer)} 个文件。开始批量推理...")

    # --- 4. 批量推理并保存结果 ---
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 写入头部信息
        f_out.write(f"# GRU Sequence Prediction Results\n")
        f_out.write(f"# Model: {model_path.name}\n")
        f_out.write(f"# Parameters: n_frames={n_frames}, m_frames={m_frames}\n")
        f_out.write(f"# Format: filepath - predicted_direction (confidence)\n\n")
        
        for file_path in tqdm(h5_files_to_infer, desc="Inferring"):
            try:
                # 4.1 预处理单个文件
                sequences, original_labels, valid_positions = load_and_preprocess_single_file(
                    file_path, scaler, n_frames, target_length
                )
                
                # 4.2 进行预测
                predictions = predict_sequence(model, sequences, m_frames)
                
                # 4.3 处理预测结果
                if m_frames == 1:
                    # 单输出情况
                    predicted_classes = np.argmax(predictions, axis=1)
                    confidence_scores = np.max(predictions, axis=1)
                else:
                    # 多输出情况 - 对每个时间步的多个预测取平均
                    # predictions是一个列表，包含m_frames个预测结果
                    avg_predictions = np.mean(predictions, axis=0)  # 对m_frames个输出取平均
                    predicted_classes = np.argmax(avg_predictions, axis=1)
                    confidence_scores = np.max(avg_predictions, axis=1)
                
                # 4.4 选择最有信心的预测作为文件的整体预测
                # 可以选择置信度最高的预测，或者使用多数投票
                if args.aggregation_method == 'max_confidence':
                    best_idx = np.argmax(confidence_scores)
                    final_prediction = predicted_classes[best_idx]
                    final_confidence = confidence_scores[best_idx]
                elif args.aggregation_method == 'majority_vote':
                    # 多数投票
                    unique, counts = np.unique(predicted_classes, return_counts=True)
                    final_prediction = unique[np.argmax(counts)]
                    # 计算该类别的平均置信度
                    mask = predicted_classes == final_prediction
                    final_confidence = np.mean(confidence_scores[mask])
                else:  # 'average'
                    # 对所有预测取平均
                    class_probs = np.zeros(3)
                    if m_frames == 1:
                        class_probs = np.mean(predictions, axis=0)
                    else:
                        class_probs = np.mean(avg_predictions, axis=0)
                    final_prediction = np.argmax(class_probs)
                    final_confidence = class_probs[final_prediction]
                
                predicted_label = CLASS_MAP.get(final_prediction, 'Unknown')
                
                # 4.5 准备输出行
                output_line = f"{file_path.resolve()} - {predicted_label} ({final_confidence:.3f})\n"
                f_out.write(output_line)
                
                # 如果需要详细输出，也可以保存每个时间步的预测
                if args.detailed_output:
                    f_out.write(f"  Detailed predictions for {file_path.name}:\n")
                    for i, (pos, pred_class, conf) in enumerate(zip(valid_positions, predicted_classes, confidence_scores)):
                        pred_label = CLASS_MAP.get(pred_class, 'Unknown')
                        f_out.write(f"    Position {pos}: {pred_label} ({conf:.3f})\n")
                    f_out.write("\n")

            except Exception as e:
                error_line = f"{file_path.resolve()} - ERROR: {e}\n"
                f_out.write(error_line)
                tqdm.write(f"\033[91m处理文件 {file_path.name} 时出错: {e}\033[0m")

    print("\n" + "="*50)
    print("      GRU序列预测推理完成")
    print("="*50)
    print(f"结果已保存至: {output_file.resolve()}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform GRU-based sequence prediction inference on H5 files.")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved Keras model file (e.g., 'checkpoints/gru_n10_m2_len150_model.h5').")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing H5 files for inference.")
    parser.add_argument("--output_file", type=str, default="output/inference_results.txt",
                        help="Path to save the inference results.")
    parser.add_argument("--aggregation_method", type=str, default="max_confidence", 
                        choices=["max_confidence", "majority_vote", "average"],
                        help="Method to aggregate multiple predictions per file.")
    parser.add_argument("--detailed_output", action="store_true",
                        help="Include detailed predictions for each time step.")

    args = parser.parse_args()
    main(args)