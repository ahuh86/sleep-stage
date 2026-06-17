# ============================================================
# config/sleepedf.py — Sleep-EDF 数据集训练配置
#
# 这里是整个项目的"控制面板"。
# 所有训练超参数都集中在这里，方便调整。
# 
# 新手建议先理解这些参数的含义，再尝试调整它们。
# ============================================================

params = {

    # ==================== 训练超参数 ====================

    # n_epochs：训练轮数
    # 整个数据集被反复训练多少遍
    "n_epochs": 200,

    # learning_rate：学习率
    # 控制模型参数每次更新的步长大小
    # 太大：模型发散不收敛    太小：训练缓慢
    # 1e-4 = 0.0001 是CNN+LSTM常用的稳定值
    "learning_rate": 1e-4,

    # Adam优化器参数
    "adam_beta_1": 0.9,       # 一阶矩衰减系数（控制梯度动量）
    "adam_beta_2": 0.999,     # 二阶矩衰减系数（控制学习率衰减速度）
    "adam_epsilon": 1e-8,     # 防止除零错误的小常数

    # clip_grad_value：梯度裁剪阈值
    # 如果梯度的范数超过5，会被缩放到5
    # 防止RNN/LSTM中常见的梯度爆炸问题
    "clip_grad_value": 5.0,

    # evaluate_span：每隔多少epoch评估一次
    "evaluate_span": 50,

    # checkpoint_span：每隔多少epoch保存一次模型
    "checkpoint_span": 50,

    # ==================== Early Stopping（早停机制）====================
    # 如果验证集连续50个epoch都没有提升，就提前结束训练
    # 防止过度训练导致过拟合
    "no_improve_epochs": 50,

    # ==================== 模型结构参数 ====================

    # 模型名称（用于标识不同实验）
    "model": "model-mod-8",

    # LSTM层数（这里只用1层，TinySleepNet追求轻量）
    "n_rnn_layers": 1,

    # LSTM隐藏单元数（记忆容量）
    # 越大模型记忆能力越强，但参数量和计算量也越大
    "n_rnn_units": 128,

    # 采样率（Hz）
    # Sleep-EDF数据集常用的EEG采样率是100Hz
    "sampling_rate": 100.0,

    # 输入信号长度
    # 100Hz * 30秒 = 3000个采样点（一个sleep epoch）
    "input_size": 3000,

    # 分类类别数
    # Wake(0) / N1(1) / N2(2) / N3(3) / REM(4)
    "n_classes": 5,

    # L2正则化系数（weight decay）
    # 越大则模型参数被约束得越小，过拟合风险越低
    "l2_weight_decay": 1e-3,

    # ==================== 数据集参数 ====================

    "dataset": "sleepedf",
    "data_dir": "../tinysleepnet/data/sleepedf/sleep-cassette/eeg_fpz_cz",
    "n_folds": 20,       # K折交叉验证（20折）
    "n_subjects": 20,    # 受试者数量

    # ==================== 数据增强 ====================

    "augment_seq": True,           # 是否对时间序列做增强（随机跳过开头）
    "augment_signal_full": True,   # 是否对信号做增强（平移缩放等）
    "weighted_cross_ent": True,    # 是否使用类别加权交叉熵
}

# ==================== 训练模式配置 ====================
train = params.copy()
train.update({
    "seq_length": 20,    # 一次看20个连续epoch（=10分钟睡眠）
    "batch_size": 15,    # 每批处理15个受试者
})

# ==================== 预测模式配置 ====================
predict = params.copy()
predict.update({
    "batch_size": 1,     # 预测时一次只处理1个受试者
    "seq_length": 1,     # 预测时逐个epoch判断（不用时序上下文）
})
