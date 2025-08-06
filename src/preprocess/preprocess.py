#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扭矩数据预处理脚本

功能：
1. 读取data/raw目录下的所有HDF5文件
2. 提取right_arm_effort和right_arm_joint_position数据
3. 基于joint_1_threshold生成标签
4. 保存处理后的数据到data/processed目录

作者：AI Assistant
日期：2025-01-05
"""

import os
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
from scipy.signal import savgol_filter
from scipy import ndimage


class TorqueDataPreprocessor:
    """扭矩数据预处理器"""
    
    def __init__(self, joint_1_threshold: float = 0.05, verbose: bool = True,
                 sg_window_length: int = 21, sg_polyorder: int = 3,
                 slope_threshold: float = 0.005, min_stable_length: int = 20):
        """
        初始化预处理器
        
        Args:
            joint_1_threshold: 关节1的阈值，用于生成标签（作为备选方法）
            verbose: 是否输出详细日志
            sg_window_length: SG滤波器窗口长度（必须为奇数）
            sg_polyorder: SG滤波器多项式阶数
            slope_threshold: 斜率阈值，用于检测信号开始上升
            min_stable_length: 最小稳定段长度，用于确定平稳区域
        """
        self.joint_1_threshold = joint_1_threshold
        self.verbose = verbose
        
        # SG滤波器参数
        self.sg_window_length = sg_window_length if sg_window_length % 2 == 1 else sg_window_length + 1
        self.sg_polyorder = sg_polyorder
        
        # 信号分析参数
        self.slope_threshold = slope_threshold
        self.min_stable_length = min_stable_length
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def apply_sg_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        应用Savitzky-Golay滤波器平滑信号
        
        Args:
            signal: 输入信号
            
        Returns:
            filtered_signal: 滤波后的信号
        """
        if len(signal) < self.sg_window_length:
            # 如果信号长度小于窗口长度，使用较小的窗口
            window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
            if window_length < 3:
                return signal  # 信号太短，直接返回
            polyorder = min(self.sg_polyorder, window_length - 1)
        else:
            window_length = self.sg_window_length
            polyorder = self.sg_polyorder
            
        return savgol_filter(signal, window_length, polyorder)
    
    def detect_signal_transitions(self, signal: np.ndarray) -> Tuple[int, int]:
        """
        检测信号从平稳到上升的转换点
        
        Args:
            signal: 输入信号（已滤波）
            
        Returns:
            pos_transition_idx: 正向转换点索引（-1表示未找到）
            neg_transition_idx: 负向转换点索引（-1表示未找到）
        """
        # 计算一阶导数（斜率）
        slopes = np.gradient(signal)
        
        # 应用SG滤波器平滑斜率
        if len(slopes) >= self.sg_window_length:
            slopes_filtered = self.apply_sg_filter(slopes)
        else:
            slopes_filtered = slopes
        
        # 寻找正向转换点（从平稳到上升）
        pos_transition_idx = -1
        for i in range(self.min_stable_length, len(slopes_filtered) - self.min_stable_length):
            # 检查前面一段是否相对平稳（斜率小且方差小）
            prev_segment = slopes_filtered[i-self.min_stable_length:i]
            prev_mean = np.mean(prev_segment)
            prev_std = np.std(prev_segment)
            
            if (np.abs(prev_mean) < self.slope_threshold * 0.3 and 
                prev_std < self.slope_threshold * 0.5):
                
                # 检查当前点及后续是否开始持续上升
                next_segment = slopes_filtered[i:i+self.min_stable_length]
                next_mean = np.mean(next_segment)
                
                # 更严格的上升条件：平均斜率大于阈值，且至少70%的点斜率为正
                if (next_mean > self.slope_threshold and 
                    np.sum(next_segment > 0) >= len(next_segment) * 0.7):
                    pos_transition_idx = i
                    break
        
        # 寻找负向转换点（从平稳到下降）
        neg_transition_idx = -1
        for i in range(self.min_stable_length, len(slopes_filtered) - self.min_stable_length):
            # 检查前面一段是否相对平稳
            prev_segment = slopes_filtered[i-self.min_stable_length:i]
            prev_mean = np.mean(prev_segment)
            prev_std = np.std(prev_segment)
            
            if (np.abs(prev_mean) < self.slope_threshold * 0.3 and 
                prev_std < self.slope_threshold * 0.5):
                
                # 检查当前点及后续是否开始持续下降
                next_segment = slopes_filtered[i:i+self.min_stable_length]
                next_mean = np.mean(next_segment)
                
                # 更严格的下降条件：平均斜率小于负阈值，且至少70%的点斜率为负
                if (next_mean < -self.slope_threshold and 
                    np.sum(next_segment < 0) >= len(next_segment) * 0.7):
                    neg_transition_idx = i
                    break
        
        return pos_transition_idx, neg_transition_idx
    
    def generate_labels_sg(self, joint_position: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        基于SG滤波器和信号分析生成精准标签
        
        标签规则：
        - 检测到正向转换（上升）时，该点及之后标记为 0
        - 检测到负向转换（下降）时，该点及之后标记为 2
        - 其余情况标签为 1
        
        Args:
            joint_position: 关节位置数据，形状为 (time_steps, 6)
            
        Returns:
            labels: 标签数组，形状为 (time_steps,)
            debug_info: 调试信息字典
        """
        time_steps = joint_position.shape[0]
        labels = np.ones(time_steps, dtype=int)  # 默认标签为1
        
        # 提取第一维度数据
        first_joint = joint_position[:, 0]
        
        # 应用SG滤波器
        filtered_signal = self.apply_sg_filter(first_joint)
        
        # 检测转换点
        pos_transition_idx, neg_transition_idx = self.detect_signal_transitions(filtered_signal)
        
        # 应用标签规则
        if pos_transition_idx != -1 and neg_transition_idx != -1:
            # 两个转换点都存在，以先出现的为准
            if pos_transition_idx < neg_transition_idx:
                labels[pos_transition_idx:] = 0
            else:
                labels[neg_transition_idx:] = 2
        elif pos_transition_idx != -1:
            # 只有正向转换
            labels[pos_transition_idx:] = 0
        elif neg_transition_idx != -1:
            # 只有负向转换
            labels[neg_transition_idx:] = 2
        
        # 构建调试信息
        debug_info = {
            'original_signal': first_joint,
            'filtered_signal': filtered_signal,
            'slopes': np.gradient(filtered_signal),
            'pos_transition_idx': pos_transition_idx,
            'neg_transition_idx': neg_transition_idx,
            'signal_range': [first_joint.min(), first_joint.max()],
            'filtered_range': [filtered_signal.min(), filtered_signal.max()]
        }
        
        return labels, debug_info
    
    def generate_labels_threshold(self, joint_position: np.ndarray) -> np.ndarray:
        """
        基于阈值的传统标签生成方法（备选方案）
        
        Args:
            joint_position: 关节位置数据，形状为 (time_steps, 6)
            
        Returns:
            labels: 标签数组，形状为 (time_steps,)
        """
        time_steps = joint_position.shape[0]
        labels = np.ones(time_steps, dtype=int)  # 默认标签为1
        
        # 提取第一维度数据
        first_joint = joint_position[:, 0]
        
        # 查找阈值越界点
        pos_threshold_indices = np.where(first_joint > self.joint_1_threshold)[0]
        neg_threshold_indices = np.where(first_joint < -self.joint_1_threshold)[0]
        
        # 应用标签规则
        if len(pos_threshold_indices) > 0:
            first_pos_idx = pos_threshold_indices[0]
            labels[first_pos_idx:] = 0
            
        if len(neg_threshold_indices) > 0:
            first_neg_idx = neg_threshold_indices[0]
            labels[first_neg_idx:] = 2
        
        # 如果同时存在正负阈值越界，以先出现的为准
        if len(pos_threshold_indices) > 0 and len(neg_threshold_indices) > 0:
            first_pos_idx = pos_threshold_indices[0]
            first_neg_idx = neg_threshold_indices[0]
            
            if first_pos_idx < first_neg_idx:
                labels[first_pos_idx:] = 0
            else:
                labels[first_neg_idx:] = 2
                
        return labels
    
    def process_single_file(self, input_path: str, use_threshold: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单个H5文件
        
        Args:
            input_path: 输入H5文件路径
            use_threshold: 是否使用传统阈值方法
            
        Returns:
            effort_data: right_arm_effort数据
            labels: 生成的标签
        """
        try:
            with h5py.File(input_path, 'r') as f:
                # 提取数据
                effort_data = f['upper_body_observations/right_arm_effort'][:]
                joint_position = f['upper_body_observations/right_arm_joint_position'][:]
                
                # 验证数据形状
                if effort_data.shape[0] != joint_position.shape[0]:
                    raise ValueError(f"数据时间步不匹配: effort={effort_data.shape}, joint={joint_position.shape}")
                
                # 生成标签
                if use_threshold:
                    # 使用传统阈值方法
                    labels = self.generate_labels_threshold(joint_position)
                    self.logger.info(f"处理文件: {os.path.basename(input_path)} [阈值方法]")
                    self.logger.info(f"  - 数据形状: effort={effort_data.shape}, labels={labels.shape}")
                    self.logger.info(f"  - 标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}")
                    self.logger.info(f"  - 关节1范围: [{joint_position[:, 0].min():.4f}, {joint_position[:, 0].max():.4f}]")
                else:
                    # 使用SG滤波器方法
                    labels, debug_info = self.generate_labels_sg(joint_position)
                    self.logger.info(f"处理文件: {os.path.basename(input_path)} [SG滤波器方法]")
                    self.logger.info(f"  - 数据形状: effort={effort_data.shape}, labels={labels.shape}")
                    self.logger.info(f"  - 标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}")
                    self.logger.info(f"  - 关节1范围: [{debug_info['signal_range'][0]:.4f}, {debug_info['signal_range'][1]:.4f}]")
                    self.logger.info(f"  - 滤波后范围: [{debug_info['filtered_range'][0]:.4f}, {debug_info['filtered_range'][1]:.4f}]")
                    if debug_info['pos_transition_idx'] != -1:
                        self.logger.info(f"  - 正向转换点: {debug_info['pos_transition_idx']}")
                    if debug_info['neg_transition_idx'] != -1:
                        self.logger.info(f"  - 负向转换点: {debug_info['neg_transition_idx']}")
                
                return effort_data, labels
                
        except Exception as e:
            self.logger.error(f"处理文件 {input_path} 时出错: {str(e)}")
            raise
    
    def save_processed_data(self, effort_data: np.ndarray, labels: np.ndarray, 
                          output_path: str) -> None:
        """
        保存处理后的数据
        
        Args:
            effort_data: 力矩数据
            labels: 标签数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with h5py.File(output_path, 'w') as f:
                # 保存数据
                f.create_dataset('right_arm_effort', data=effort_data)
                f.create_dataset('labels', data=labels)
                
                # 添加元数据
                f.attrs['joint_1_threshold'] = self.joint_1_threshold
                f.attrs['data_shape'] = effort_data.shape
                f.attrs['label_distribution'] = [np.sum(labels==i) for i in range(3)]
                
            self.logger.info(f"数据已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存数据到 {output_path} 时出错: {str(e)}")
            raise
    
    def find_h5_files(self, root_dir: str) -> List[str]:
        """
        递归查找所有H5文件
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            h5_files: H5文件路径列表
        """
        h5_files = []
        root_path = Path(root_dir)
        
        for file_path in root_path.rglob('*.h5'):
            h5_files.append(str(file_path))
            
        self.logger.info(f"在 {root_dir} 中找到 {len(h5_files)} 个H5文件")
        return sorted(h5_files)
    
    def get_relative_path(self, file_path: str, raw_dir: str) -> str:
        """
        获取相对于raw目录的路径
        
        Args:
            file_path: 完整文件路径
            raw_dir: raw目录路径
            
        Returns:
            relative_path: 相对路径
        """
        raw_path = Path(raw_dir).resolve()
        file_path_obj = Path(file_path).resolve()
        
        try:
            relative_path = file_path_obj.relative_to(raw_path)
            return str(relative_path)
        except ValueError:
            # 如果无法获取相对路径，返回文件名
            return file_path_obj.name
    
    def process_all_files(self, raw_dir: str, processed_dir: str, use_threshold: bool = False) -> Dict[str, int]:
        """
        处理所有H5文件
        
        Args:
            raw_dir: 原始数据目录
            processed_dir: 处理后数据目录
            use_threshold: 是否使用传统阈值方法
            
        Returns:
            statistics: 处理统计信息
        """
        # 查找所有H5文件
        h5_files = self.find_h5_files(raw_dir)
        
        if not h5_files:
            self.logger.warning(f"在 {raw_dir} 中未找到H5文件")
            return {'processed': 0, 'failed': 0}
        
        processed_count = 0
        failed_count = 0
        
        for file_path in h5_files:
            try:
                # 获取相对路径以保持目录结构
                relative_path = self.get_relative_path(file_path, raw_dir)
                output_path = os.path.join(processed_dir, relative_path)
                
                # 处理文件
                effort_data, labels = self.process_single_file(file_path, use_threshold)
                
                # 保存结果
                self.save_processed_data(effort_data, labels, output_path)
                
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 失败: {str(e)}")
                failed_count += 1
                continue
        
        # 输出统计信息
        statistics = {
            'total': len(h5_files),
            'processed': processed_count,
            'failed': failed_count
        }
        
        self.logger.info("="*50)
        self.logger.info("处理完成统计:")
        self.logger.info(f"  - 总文件数: {statistics['total']}")
        self.logger.info(f"  - 成功处理: {statistics['processed']}")
        self.logger.info(f"  - 处理失败: {statistics['failed']}")
        self.logger.info("="*50)
        
        return statistics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扭矩数据预处理脚本')
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                      help='原始数据目录路径 (默认: data/raw)')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='处理后数据目录路径 (默认: data/processed)')
    parser.add_argument('--threshold', type=float, default=0.10,
                      help='关节1阈值 (默认: 0.1)')
    parser.add_argument('--sg_window', type=int, default=21,
                      help='SG滤波器窗口长度 (默认: 21)')
    parser.add_argument('--sg_poly', type=int, default=3,
                      help='SG滤波器多项式阶数 (默认: 3)')
    parser.add_argument('--slope_threshold', type=float, default=0.01,
                      help='斜率阈值，用于检测信号变化 (默认: 0.005)')
    parser.add_argument('--min_stable', type=int, default=20,
                      help='最小稳定段长度 (默认: 20)')
    parser.add_argument('--use_threshold', action='store_true',
                      help='使用传统阈值方法而不是SG滤波器方法')
    parser.add_argument('--verbose', action='store_true',
                      help='输出详细日志')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = TorqueDataPreprocessor(
        joint_1_threshold=args.threshold,
        verbose=args.verbose,
        sg_window_length=args.sg_window,
        sg_polyorder=args.sg_poly,
        slope_threshold=args.slope_threshold,
        min_stable_length=args.min_stable
    )
    
    try:
        # 处理所有文件
        statistics = preprocessor.process_all_files(args.raw_dir, args.processed_dir, args.use_threshold)
        
        if statistics['failed'] > 0:
            exit(1)  # 有失败的文件时返回非零退出码
            
    except KeyboardInterrupt:
        print("\n用户中断处理过程")
        exit(1)
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
