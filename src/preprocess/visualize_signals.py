#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号可视化脚本

用于分析joint_position信号的特征，帮助调整SG滤波器参数

作者：AI Assistant
日期：2025-01-05
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from preprocess import TorqueDataPreprocessor


def visualize_signal_analysis(file_path: str, preprocessor: TorqueDataPreprocessor, 
                            save_dir: str = "signal_analysis"):
    """
    可视化单个文件的信号分析过程
    
    Args:
        file_path: H5文件路径
        preprocessor: 预处理器实例
        save_dir: 保存图片的目录
    """
    # 读取数据
    with h5py.File(file_path, 'r') as f:
        joint_position = f['upper_body_observations/right_arm_joint_position'][:]
    
    # 提取第一维度信号
    signal = joint_position[:, 0]
    
    # 应用SG滤波器
    filtered_signal = preprocessor.apply_sg_filter(signal)
    
    # 计算斜率
    slopes = np.gradient(filtered_signal)
    slopes_filtered = preprocessor.apply_sg_filter(slopes) if len(slopes) >= preprocessor.sg_window_length else slopes
    
    # 生成标签
    labels, debug_info = preprocessor.generate_labels_sg(joint_position)
    
    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Signal Analysis: {os.path.basename(file_path)}', fontsize=14)
    
    time_steps = np.arange(len(signal))
    
    # 原始信号 vs 滤波信号
    axes[0].plot(time_steps, signal, 'b-', alpha=0.7, label='original signal', linewidth=1)
    axes[0].plot(time_steps, filtered_signal, 'r-', label='SG signal', linewidth=2)
    axes[0].axhline(y=preprocessor.joint_1_threshold, color='g', linestyle='--', alpha=0.7, label=f'阈值 +{preprocessor.joint_1_threshold}')
    axes[0].axhline(y=-preprocessor.joint_1_threshold, color='g', linestyle='--', alpha=0.7, label=f'阈值 -{preprocessor.joint_1_threshold}')
    axes[0].set_ylabel('joint_1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 斜率信号
    axes[1].plot(time_steps, slopes, 'b-', alpha=0.7, label='original slope', linewidth=1)
    axes[1].plot(time_steps, slopes_filtered, 'r-', label='filtered slope', linewidth=2)
    axes[1].axhline(y=preprocessor.slope_threshold, color='g', linestyle='--', alpha=0.7, label=f'斜率阈值 +{preprocessor.slope_threshold}')
    axes[1].axhline(y=-preprocessor.slope_threshold, color='g', linestyle='--', alpha=0.7, label=f'斜率阈值 -{preprocessor.slope_threshold}')
    axes[1].set_ylabel('斜率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 标签可视化
    label_colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = labels == i
        if np.any(mask):
            axes[2].scatter(time_steps[mask], np.full(np.sum(mask), i), 
                          c=label_colors[i], alpha=0.7, s=20, label=f'label {i}')
    
    # 标记转换点
    if debug_info['pos_transition_idx'] != -1:
        axes[2].axvline(x=debug_info['pos_transition_idx'], color='orange', linestyle='-', 
                       linewidth=2, label=f'positive shift: {debug_info["pos_transition_idx"]}')
    if debug_info['neg_transition_idx'] != -1:
        axes[2].axvline(x=debug_info['neg_transition_idx'], color='purple', linestyle='-', 
                       linewidth=2, label=f'negative shift: {debug_info["neg_transition_idx"]}')
    
    axes[2].set_ylabel('label')
    axes[2].set_ylim(-0.5, 2.5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 信号与标签叠加显示
    axes[3].plot(time_steps, filtered_signal, 'k-', linewidth=2, label='filtered signal')
    
    # 用不同颜色显示不同标签区域
    for i in range(3):
        mask = labels == i
        if np.any(mask):
            axes[3].fill_between(time_steps, filtered_signal.min(), filtered_signal.max(), 
                               where=mask, alpha=0.3, color=label_colors[i], 
                               label=f'label {i} region')
    
    axes[3].set_xlabel('time step')
    axes[3].set_ylabel('joint_1')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_dir, f"{filename}_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n文件: {os.path.basename(file_path)}")
    print(f"信号范围: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"滤波信号范围: [{filtered_signal.min():.4f}, {filtered_signal.max():.4f}]")
    print(f"斜率范围: [{slopes_filtered.min():.4f}, {slopes_filtered.max():.4f}]")
    print(f"标签分布: {[np.sum(labels==i) for i in range(3)]}")
    print(f"转换点: 正向={debug_info['pos_transition_idx']}, 负向={debug_info['neg_transition_idx']}")
    print(f"图片已保存: {save_path}")


def analyze_directory_signals(data_dir: str, preprocessor: TorqueDataPreprocessor, 
                            max_files: int = 5):
    """
    分析目录中的信号文件
    
    Args:
        data_dir: 数据目录
        preprocessor: 预处理器实例
        max_files: 最大分析文件数
    """
    # 从每个子目录选择几个文件进行分析
    subdirs = ['left', 'mid', 'right']
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"目录不存在: {subdir_path}")
            continue
            
        # 获取H5文件列表
        h5_files = [f for f in os.listdir(subdir_path) if f.endswith('.h5')]
        h5_files = sorted(h5_files)[:max_files]  # 只取前几个文件
        
        print(f"\n=== 分析 {subdir} 目录 ===")
        
        for h5_file in h5_files:
            file_path = os.path.join(subdir_path, h5_file)
            save_dir = f"signal_analysis/{subdir}"
            
            try:
                visualize_signal_analysis(file_path, preprocessor, save_dir)
            except Exception as e:
                print(f"分析文件 {h5_file} 失败: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='信号可视化分析脚本')
    parser.add_argument('--data_dir', type=str, default='data/raw/test',
                      help='数据目录路径 (默认: data/raw/test)')
    parser.add_argument('--file_path', type=str, default=None,
                      help='分析单个文件路径')
    parser.add_argument('--max_files', type=int, default=5,
                      help='每个目录最大分析文件数 (默认: 3)')
    parser.add_argument('--threshold', type=float, default=0.1,
                      help='关节1阈值 (默认: 0.1)')
    parser.add_argument('--sg_window', type=int, default=21,
                      help='SG滤波器窗口长度 (默认: 21)')
    parser.add_argument('--sg_poly', type=int, default=3,
                      help='SG滤波器多项式阶数 (默认: 3)')
    parser.add_argument('--slope_threshold', type=float, default=0.01,
                      help='斜率阈值 (默认: 0.01)')
    parser.add_argument('--min_stable', type=int, default=20,
                      help='最小稳定段长度 (默认: 20)')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = TorqueDataPreprocessor(
        joint_1_threshold=args.threshold,
        verbose=True,
        sg_window_length=args.sg_window,
        sg_polyorder=args.sg_poly,
        slope_threshold=args.slope_threshold,
        min_stable_length=args.min_stable
    )
    
    if args.file_path:
        # 分析单个文件
        visualize_signal_analysis(args.file_path, preprocessor)
    else:
        # 分析目录
        analyze_directory_signals(args.data_dir, preprocessor, args.max_files)


if __name__ == '__main__':
    main() 