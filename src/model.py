import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TorqueRegressor(nn.Module):
    """
    创建一个基于GRU的时序回归模型，用于预测机械臂未来m帧的力矩信息。
    
    实验设计：基于过去n帧预测未来m帧的机械臂力矩值
    
    Args:
        input_size (int): 输入特征维度，默认为7（7维扭矩传感器数据）
        hidden_size (int): GRU隐藏层大小
        num_layers (int): GRU层数
        output_dim (int): 输出维度，等于m_frames，代表预测的未来帧数
        model_type (str): 模型类型，主要支持'gru'，保留其他类型以兼容原接口
        n_frames (int): 输入的历史帧数，默认10
        m_frames (int): 预测的未来帧数，默认2
        dropout (float): Dropout率
        max_position (int): 位置编码的最大值
    """
    
    def __init__(self, input_size=7, hidden_size=256, num_layers=2, output_dim=2, 
                 model_type='gru', n_frames=10, m_frames=2, dropout=0.2, max_position=1000):
        super(TorqueRegressor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model_type = model_type
        self.n_frames = n_frames
        self.m_frames = m_frames
        
        print(f"Building {model_type.upper()} regression model for sequence prediction...")
        print(f"Input: past {n_frames} frames -> Output: future {m_frames} torque values")
        
        if model_type == 'gru':
            # 第一层GRU，返回序列
            self.gru1 = nn.GRU(
                input_size=input_size,
                hidden_size=512,
                num_layers=1,
                batch_first=True,
                dropout=0.0,  # 单层不需要dropout
                bidirectional=False
            )
            
            # 第二层GRU，只返回最后的输出
            self.gru2 = nn.GRU(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False
            )
            
            # 手动添加dropout层
            self.dropout_gru1 = nn.Dropout(dropout)
            self.dropout_gru2 = nn.Dropout(dropout)
            
        else:
            raise ValueError("model_type must be 'gru'")
        
        # Position Embedding: 将窗口位置编码映射到与GRU输出相同的维度
        self.position_embedding = nn.Embedding(max_position, hidden_size)
        
        # 全连接层作为预测头
        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512, 256)
        
        # 输出层 - 回归任务，预测m_frames个未来时间步的力矩值
        self.output_layer = nn.Linear(256, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x, positions):
        """
        前向传播
        
        Args:
            x: 输入序列 (batch_size, n_frames, input_size)
            positions: 位置编码 (batch_size,)
            
        Returns:
            output: 预测的torque值 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 第一层GRU
        gru1_out, _ = self.gru1(x)  # (batch_size, n_frames, 512)
        gru1_out = self.dropout_gru1(gru1_out)
        
        # 第二层GRU，只需要最后的输出
        gru2_out, _ = self.gru2(gru1_out)  # (batch_size, n_frames, hidden_size)
        gru2_out = self.dropout_gru2(gru2_out)
        
        # 取最后一个时间步的输出
        gru_output = gru2_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Position Embedding
        pos_emb = self.position_embedding(positions)  # (batch_size, hidden_size)
        
        # 将GRU输出与位置嵌入相加
        x = gru_output + pos_emb  # (batch_size, hidden_size)
        
        # 全连接层
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))
        
        # 输出层（线性激活）
        output = self.output_layer(x)
        
        return output

def create_torque_regressor(input_shape=(10, 7), output_dim=2, model_type='gru', n_frames=10, m_frames=2):
    """
    创建一个基于GRU的时序回归模型，用于预测机械臂未来m帧的力矩信息。
    
    Args:
        input_shape (tuple): 输入数据的形状 (n_frames, 7)，其中n_frames是输入的时间步数
        output_dim (int): 输出维度，等于m_frames，代表预测的未来帧数
        model_type (str): 模型类型，主要支持'gru'，保留其他类型以兼容原接口
        n_frames (int): 输入的历史帧数，默认10
        m_frames (int): 预测的未来帧数，默认2
        
    Returns:
        A PyTorch Model instance for regression task.
    """
    
    input_size = input_shape[1]  # 特征维度
    
    model = TorqueRegressor(
        input_size=input_size,
        output_dim=output_dim,
        model_type=model_type,
        n_frames=n_frames,
        m_frames=m_frames
    )
    
    return model

def get_optimizer(model, optimizer_type='adamw', learning_rate=0.001):
    """
    获取优化器
    
    Args:
        model: PyTorch模型
        optimizer_type (str): 优化器类型，'adam' 或 'adamw'
        learning_rate (float): 学习率
        
    Returns:
        optimizer: PyTorch优化器实例
    """
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        return optim.Adam(model.parameters(), lr=learning_rate)
