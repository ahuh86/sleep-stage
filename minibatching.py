# ============================================================
# minibatching.py — 批量数据生成器（核心数据处理模块）
#
# 【这个文件是干什么的？】
# 深度学习训练时，数据不能一次性全部送入模型（内存放不下），
# 而是要"分批"（batch）送入。这个文件实现了三种分批策略。
#
# 【核心概念】
# 为了理解这个文件，你需要先明白几个关键术语：
#
# 1. epoch（小写）：30秒的睡眠片段，是模型判断的最小单位
#    每个epoch有3000个采样点（100Hz × 30秒）
#
# 2. seq_length：时间窗口长度
#    LSTM每次看多少个连续的epoch
#    这里设为20，相当于一次看20×30秒 = 10分钟的睡眠记录
#
# 3. batch_size：批大小
#    一批同时处理多少个受试者的数据
#    这里设为15，相当于每次训练15个受试者各20个epoch
#
# 4. 为什么要有"多重序列批处理"？
#    不同受试者的睡眠时长不同（有人睡6小时，有人睡8小时），
#    所以每个人的epoch数量不同。我们需要把不等长的数据
#    "对齐"到同一个batch中，短序列用0填充。
#    weights数组就是用来标记哪些位置是真实数据（1）、哪些是填充（0）。
#
# 【数据形状变化】
# 输入：15个受试者，每人约800~1000个epoch
#         → 批处理生成器每次产出：
# 输出：batch_x: (300, 1, 3000)   ← 15人 × 20epoch = 300个样本
#       batch_y: (300,)            ← 300个标签
#       batch_weights: (300,)      ← 哪些是真实数据
#       batch_seq_len: (15,)       ← 每人当前批次的真实长度
#       start_loop: bool           ← 是否是新受试者的第一个batch
# ============================================================

import math
import numpy as np


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """【简单批处理生成器】
    
    最简单的分批方式：把数据按顺序切成等大小的batch。
    不支持时序处理，在这个项目中不常用。
    
    参数:
        inputs:  numpy array, 形状 (n_samples, ...)
        targets: numpy array, 形状 (n_samples,)
        batch_size: int, 每批的样本数
        shuffle:  bool, 是否在每个epoch开始时打乱顺序
    
    生成:
        (batch_inputs, batch_targets): 一个batch的数据
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    """【简单时序批处理生成器】
    
    把数据按受试者均分成 batch_size 份，
    每份再按 seq_length 切成小段。
    
    限制：所有受试者的数据长度必须相同。
    实际中不同受试者的睡眠时长不同，所以这个函数不太实用。
    """
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size
    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:], dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:], dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y


def iterate_batch_multiple_seq_minibatches(inputs, targets, batch_size, seq_length,
                                          shuffle_idx=None, augment_seq=False):
    """★ 【核心函数】多重序列批处理生成器 ★
    
    【设计动机】
    假设我们有15个受试者（batch_size=15），
    每个人睡了约7-9小时（约840-1080个epoch），
    每个人的睡眠时长（epoch数量）不同。
    
    我们希望每次从每个人身上取20个连续epoch（seq_length=20），
    然后把这15个"20-epoch片段"拼成一个batch送给模型。
    
    如果某个人剩下的epoch不够20个，就用0填充，
    并用 weights 标记哪些是真实数据。
    
    【处理流程】
    Step 1: 将受试者排序（训练时打乱顺序，验证时保持原序）
    Step 2: 每 batch_size 个受试者为一组（这里是15人）
    Step 3: （可选）数据增强：随机跳过开头几个epoch
    Step 4: 找出组中最长的受试者，计算需要切成多少段
    Step 5: 逐段生成：每次取所有人的第 i 段（各20个epoch）
    Step 6: 短序列用0填充，记录 weights 和实际长度
    
    参数:
        inputs:      list of numpy arrays, 每个元素形状 (n_epochs, ...)
        targets:     list of numpy arrays, 每个元素形状 (n_epochs,)
        batch_size:  int, 一批处理多少个受试者
        seq_length:  int, 时间窗口长度（连续看多少epoch）
        shuffle_idx: numpy array or None, 打乱后的受试者索引
        augment_seq: bool, 是否做序列增强（训练时用）
    
    生成:
        batch_x:       numpy array (batch_size*seq_length, 1, 3000)
                       当前batch的所有脑电信号
        batch_y:       numpy array (batch_size*seq_length,)
                       对应的睡眠阶段标签
        batch_weights: numpy array (batch_size*seq_length,)
                       标记哪些是真实数据（1.0）vs 填充数据（0.0）
        batch_seq_len: numpy array (batch_size,)
                       每个受试者当前批次的真实epoch数
        start_loop:    bool
                       True=这是新受试者的第一段（需要重置LSTM隐藏状态）
    """
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)

    # ---- Step 1: 决定受试者的处理顺序 ----
    if shuffle_idx is None:
        seq_idx = np.arange(n_inputs)          # 按原始顺序（验证/测试时）
    else:
        seq_idx = shuffle_idx                   # 按打乱顺序（训练时）

    input_sample_shape = inputs[0].shape[1:]    # 单个epoch的形状：(3000, 1, 1)
    target_sample_shape = targets[0].shape[1:]  # 单个标签的形状：()

    # ---- Step 2: 计算需要多少轮才能处理完所有受试者 ----
    n_loops = int(math.ceil(len(seq_idx) / batch_size))

    for l in range(n_loops):
        start_idx = l * batch_size
        end_idx = (l + 1) * batch_size
        seq_inputs = np.asarray(inputs)[seq_idx[start_idx:end_idx]]
        seq_targets = np.asarray(targets)[seq_idx[start_idx:end_idx]]

        # ---- Step 3（可选）：序列增强 ----
        # 随机跳过每个受试者开头的0~5个epoch
        # 【为什么要这样做？】
        # 模型不能"依赖"看到序列的绝对起始位置。
        # 增强后，模型学会了从任意中间位置开始推理，
        # 泛化能力更强。
        if augment_seq:
            max_skips = 5
            for s_idx in range(len(seq_inputs)):
                n_skips = np.random.randint(max_skips)
                seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]
                seq_targets[s_idx] = seq_targets[s_idx][n_skips:]

        # ---- Step 4: 计算当前组中最长受试者需要切成几段 ----
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_inputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)
        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))

        # ---- Step 5 & 6: 逐段生成 ----
        for b in range(n_batch_seqs):
            # start_loop = True 表示这是新受试者的第一段
            # 此时需要重置LSTM的隐藏状态（h0, c0归零）
            start_loop = True if b == 0 else False

            start_idx_b = b * seq_length
            end_idx_b = (b + 1) * seq_length

            # 初始化全零数组（短序列的剩余部分自动保持为0）
            batch_inputs = np.zeros(
                (batch_size, seq_length) + input_sample_shape, dtype=np.float32
            )
            batch_targets = np.zeros(
                (batch_size, seq_length) + target_sample_shape, dtype=np.int
            )
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int)

            # ---- 逐个受试者填入实际数据 ----
            for s_idx, s in enumerate(zip(seq_inputs, seq_targets)):
                each_seq_inputs = s[0][start_idx_b:end_idx_b]   # 取第b段
                each_seq_targets = s[1][start_idx_b:end_idx_b]
                real_len = len(each_seq_inputs)

                # 将真实数据填入batch的对应位置
                batch_inputs[s_idx, :real_len] = each_seq_inputs
                batch_targets[s_idx, :real_len] = each_seq_targets
                batch_weights[s_idx, :real_len] = 1.0   # 标记为真实数据
                batch_seq_len[s_idx] = real_len

            # 展平：将 (15, 20, ...) 变为 (300, ...)
            # 这样CNN可以一次性处理300个独立样本
            batch_x = batch_inputs.reshape((-1,) + input_sample_shape)
            batch_y = batch_targets.reshape((-1,) + target_sample_shape)
            batch_weights = batch_weights.reshape(-1)

            yield batch_x, batch_y, batch_weights, batch_seq_len, start_loop
