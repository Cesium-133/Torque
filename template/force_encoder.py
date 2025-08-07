import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

# TODO 电流力矩使用GRU编码,且需要检测向右的数据是否就是视觉输入不一样,模型从视觉学到了捷径
class ForceEncoder(nn.Module):
    """
    基于GRU的力编码器 (版本三，适配序列输入)
    
    该模型结构为:
    1. 两层GRU网络 (64单元 -> 32单元)。
    2. 一个多层感知机 (MLP)，包含Dropout和ReLU激活。
    3. 适配形状为 (batch_size, sequence_length, input_dim) 的序列输入。
    
    Args:
        input_dim (int): 输入力矩数据的维度 (默认7)
        output_dim (int): 输出embedding的维度 (默认512)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 512,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # GRU层定义无需改变
        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        
        self.gru2 = nn.GRU(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        
        # MLP层定义无需改变
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(32, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络中所有线性层的权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, force_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            force_data (torch.Tensor): 输入力矩序列数据，
                                     形状应为 (batch_size, sequence_length, input_dim)
                                     例如 (16, 10, 7)
        
        Returns:
            torch.Tensor: 编码后的力特征，形状为 (batch_size, output_dim)
        """
        # --- 主要修改点 ---
        # 1. 更新输入验证，现在期望3D张量
        if force_data.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {force_data.dim()}D. Shape should be (batch, seq_len, dim).")
        
        if force_data.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {force_data.size(-1)}")
        
        # 2. 移除 unsqueeze(1)，因为输入已经是序列了
        # force_data_seq = force_data.unsqueeze(1) # <-- REMOVED
        
        # 第一层GRU
        gru1_output, _ = self.gru1(force_data)
        
        # 第二层GRU
        _, h_n2 = self.gru2(gru1_output)
        
        # 提取GRU的最终隐藏状态并调整形状: (1, batch_size, 32) -> (batch_size, 32)
        gru_output = h_n2.squeeze(0)
        
        # 通过MLP层
        mlp_output = self.mlp(gru_output)
        
        # 通过最后的输出层，得到最终的embedding
        force_embedding = self.fc_out(mlp_output)
        
        return force_embedding
    
    def encode_single(self, force_sequence: Union[torch.Tensor, List[List[float]]]) -> torch.Tensor:
        """
        编码单个力矩序列样本。
        
        Args:
            force_sequence: 单个力矩序列，应为2D tensor或嵌套列表，
                            形状如 (10, 7)
        
        Returns:
            torch.Tensor: 编码后的单个力特征，形状为 (1, output_dim)
        """
        # --- 修改以适配序列输入 ---
        if isinstance(force_sequence, list):
            force_tensor = torch.tensor(force_sequence, dtype=torch.float32)
        else:
            force_tensor = force_sequence
        
        # 期望输入是代表单个序列的2D张量
        if force_tensor.dim() != 2:
            raise ValueError(f"Expected a 2D tensor for a single sequence, but got {force_tensor.dim()}D.")
            
        # 为单个序列添加batch维度: (seq_len, dim) -> (1, seq_len, dim)
        force_tensor = force_tensor.unsqueeze(0)
        
        return self.forward(force_tensor)