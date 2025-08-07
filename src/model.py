import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, GlobalMaxPooling1D, Flatten, Dropout, TimeDistributed, Embedding, Add
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau

def create_torque_regressor(input_shape=(10, 7), output_dim=2, model_type='gru', n_frames=10, m_frames=2):
    """
    创建一个基于GRU的时序回归模型，用于预测机械臂未来m帧的力矩信息。
    
    实验设计：基于过去n帧预测未来m帧的机械臂力矩值
    
    Args:
        input_shape (tuple): 输入数据的形状 (n_frames, 7)，其中n_frames是输入的时间步数
        output_dim (int): 输出维度，等于m_frames，代表预测的未来帧数
        model_type (str): 模型类型，主要支持'gru'，保留其他类型以兼容原接口
        n_frames (int): 输入的历史帧数，默认10
        m_frames (int): 预测的未来帧数，默认2
        
    Returns:
        A Keras Model instance for regression task.
    """
    
    # 主输入：时序特征数据
    input_layer = Input(shape=input_shape, name='input_sequence')
    
    # Position Embedding输入：窗口级别的位置编码
    position_input = Input(shape=(1,), dtype='int32', name='position_input')
    
    x = input_layer

    if model_type == 'gru':
        # GRU模型 - 主要推荐的架构
        print(f"Building GRU regression model for sequence prediction...")
        print(f"Input: past {n_frames} frames -> Output: future {m_frames} torque values")
        
        
        x = GRU(units=64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0, name='gru_1')(x)
        gru_output = GRU(units=32, return_sequences=False, dropout=0.0, recurrent_dropout=0.0, name='gru_2')(x)
        
        # Position Embedding: 将窗口位置编码映射到与GRU输出相同的维度
        # 假设最大位置数为1000，可以根据实际需要调整
        position_embedding = Embedding(input_dim=1000, output_dim=32, name='position_embedding')(position_input)
        position_embedding = Flatten(name='position_flatten')(position_embedding)
        
        # 将GRU输出与位置嵌入相加
        x = Add(name='add_position_embedding')([gru_output, position_embedding])
        
        # Dropout防止过拟合
        #x = Dropout(0.3, name='dropout_1')(x)
        
        # 全连接层作为预测头
        x = Dense(512, activation='relu', name='dense_1')(x)
        #x = Dropout(0.2, name='dropout_2')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)

    elif model_type == 'lstm':
        # 方案 C: LSTM (保留兼容性)
        print("Building LSTM regression model...")
        x = tf.keras.layers.LSTM(units=64, return_sequences=False)(x)
        
        # Position Embedding for LSTM
        position_embedding = Embedding(input_dim=1000, output_dim=64, name='position_embedding')(position_input)
        position_embedding = Flatten(name='position_flatten')(position_embedding)
        
        # 将LSTM输出与位置嵌入相加
        x = Add(name='add_position_embedding')([x, position_embedding])
        
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
    
    else:
        raise ValueError("model_type must be one of 'gru', 'lstm'")

    # 输出层 - 回归任务，预测m_frames个未来时间步的力矩值
    # 输出一个m维向量，代表未来m帧的力矩信息
    output_layer = Dense(output_dim, activation='linear', name='torque_output')(x)
    
    # 创建模型，包含两个输入：序列数据和位置编码
    model = Model(inputs=[input_layer, position_input], outputs=output_layer)
    return model

def create_torque_classifier(input_shape=(10, 7), num_classes=3, model_type='gru', n_frames=10, m_frames=2):
    """
    创建一个基于GRU的时序预测模型，用于预测机械臂运动方向。
    
    实验设计：基于过去n帧预测未来m帧的机械臂运动方向
    
    Args:
        input_shape (tuple): 输入数据的形状 (n_frames, 7)，其中n_frames是输入的时间步数
        num_classes (int): 输出类别的数量 (0=左, 1=不动, 2=右)
        model_type (str): 模型类型，主要支持'gru'，保留其他类型以兼容原接口
        n_frames (int): 输入的历史帧数，默认10
        m_frames (int): 预测的未来帧数，默认2
        
    Returns:
        A Keras Model instance.
    """
    
    input_layer = Input(shape=input_shape, name='input_sequence')
    x = input_layer

    if model_type == 'gru':
        # GRU模型 - 主要推荐的架构
        print(f"Building GRU model for sequence prediction...")
        print(f"Input: past {n_frames} frames -> Output: future {m_frames} frames")
        
        # 第一层GRU，返回序列以便后续层处理
        x = GRU(units=64, return_sequences=True, name='gru_1')(x)
        
        # 第二层GRU，只返回最后的输出
        x = GRU(units=32, return_sequences=False, name='gru_2')(x)
        
        # Dropout防止过拟合
        #x = Dropout(0.3, name='dropout_1')(x)
        
        
        # 全连接层
        x = Dense(64, activation='relu', name='dense_1')(x)
        #x = Dropout(0.2, name='dropout_2')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)

    elif model_type == 'lstm':
        # 方案 C: LSTM (保留兼容性)
        print("Building LSTM model...")
        x = tf.keras.layers.LSTM(units=64, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
    
    else:
        raise ValueError("model_type must be one of 'gru', 'cnn', 'lstm', or 'flatten'")

    # 输出层 - 预测m_frames个时间步的类别
    # 为了简化，我们预测每个未来时间步的类别，然后在loss中取平均
    if model_type == 'gru':
        # 为每个预测的未来帧输出一个分类结果
        outputs = []
        for i in range(m_frames):
            frame_output = Dense(num_classes, activation='softmax', name=f'output_frame_{i+1}')(x)
            outputs.append(frame_output)
        
        # 如果只预测一帧，直接返回；否则返回多个输出
        if m_frames == 1:
            output_layer = outputs[0]
        else:
            output_layer = outputs
    else:
        # 其他模型类型保持原有的单输出结构
        output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def get_optimizer(optimizer_type='adamw', learning_rate=0.001):
    """
    获取优化器
    
    Args:
        optimizer_type (str): 优化器类型，'adam' 或 'adamw'
        learning_rate (float): 学习率
        
    Returns:
        optimizer: Keras优化器实例
    """
    if optimizer_type.lower() == 'adamw':
        return AdamW(learning_rate=learning_rate, weight_decay=0.01)
    else:
        return Adam(learning_rate=learning_rate)

def get_lr_scheduler(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7):
    """
    获取学习率调度器
    
    Args:
        monitor (str): 监控的指标
        factor (float): 学习率衰减因子
        patience (int): 等待轮数
        min_lr (float): 最小学习率
        
    Returns:
        callback: ReduceLROnPlateau回调
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=1
    )
