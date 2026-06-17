# ============================================================
# network.py — TinySleepNet 神经网络结构（★ 最核心文件）
#
# 【这个文件是干什么的？】
# 定义深度学习模型的"骨架"——数据从输入到输出经历了哪些数学变换。
# 如果把模型比作一个"脑电波翻译器"，这个文件就是它的"电路图"。
#
# ============================================================
# ★ 整体架构：CNN + LSTM
# ============================================================
#
# 【为什么这样设计？—— 两步走策略】
# 睡眠分期不像普通的图像分类（比如识别猫和狗），
# 它有一个重要的特点：时序连续性。
#
# 一个人不会突然从"清醒"跳到"深睡"，睡眠阶段的变化是平滑的。
# 因此模型需要两种能力：
#
# ① CNN（卷积神经网络）：处理"空间"信息
#    → 在一个30秒的epoch内部，识别出特定的波形模式
#    → 比如检测有没有"睡眠纺锤波"来判断是不是N2期
#
# ② LSTM（长短期记忆网络）：处理"时间"信息
#    → 把最近20个epoch（10分钟）串联起来看
#    → 利用睡眠阶段的前后依赖关系做判断
#    → 比如"刚才还在N3深睡，不太可能马上进REM"
#
# 【通俗类比】
# CNN就像是"看图识字"——看每个epoch的波形图案
# LSTM像是"上下文理解"——结合前后文来判断
# 两者结合，就像人类专家既看当前波形、又看变化趋势。
#
# ============================================================
# ★ 数据形状变化全流程
# ============================================================
# 输入（raw signal）:
#   (batch*seq, 1, 3000)
#   ↑ 比如300个样本，每个是1通道、3000个采样点的EEG信号
#
# ↓ CNN编码
#
# 特征向量:
#   (batch*seq, 2048)
#   ↑ 每个30秒epoch被压缩成2048维的特征向量
#
# ↓ 重塑为序列
#
# 时序数据:
#   (batch, seq_length, 2048)
#   ↑ 变成15个序列，每个序列20个时间步，每步2048维
#
# ↓ LSTM时序建模
#
# LSTM输出:
#   (batch, seq_length, 128)
#   ↑ LSTM隐状态维度为128
#
# ↓ 全连接分类
#
# 输出（logits）:
#   (batch*seq, 5)
#   ↑ 每个epoch属于5个睡眠类别的"得分"
# ============================================================

import torch
import torch.nn as nn
from collections import OrderedDict


class TinySleepNet(nn.Module):
    """TinySleepNet: 轻量级睡眠分期网络。

    继承自 nn.Module，这是PyTorch中所有神经网络模型的基类。
    我们需要实现两个方法：
      - __init__(): 定义网络有哪些"层"（layer）
      - forward():  定义数据如何从输入流到输出（前向传播）
    """

    def __init__(self, config):
        """构造函数：搭建网络的"骨架"

        参数:
            config: dict, 配置参数（来自 config/sleepedf.py 的 train 字典）
                    包含: sampling_rate, seq_length, n_rnn_units 等
        """
        super(TinySleepNet, self).__init__()

        # ==================== 1. Padding设置 ====================
        #
        # 【什么是padding？】
        # 卷积操作会缩小数据的尺寸。比如kernel_size=50时，
        # 输入3000点的信号会缩小为 (3000-50)/stride+1 点。
        # 但我们有时希望输出尺寸保持为输入尺寸的某个比例，
        # 这就需要在原始信号两端补充一些"假数据"（通常补0）。
        #
        # 【这里的padding策略】
        # 这是为了模拟TensorFlow的"SAME padding"模式，
        # 即：输出长度 = ceil(输入长度 / stride)
        # 元组 (left, right) 表示左边补left个0，右边补right个0
        # -------------------------------------------------------
        self.padding_edf = {
            'conv1': (22, 22),       # 第一层卷积前：左右各补22个0
            'max_pool1': (2, 2),      # 第一次池化前：左右各补2个0
            'conv2': (3, 4),          # 后续卷积前：左边3个、右边4个（不对称！）
            'max_pool2': (0, 1),      # 第二次池化前：右边补1个0
        }
        self.config = config

        # ==================== 2. 第一个卷积核 ====================
        #
        # 【卷积核大小为什么取采样率的一半？】
        # sampling_rate=100Hz -> kernel_size=50
        # 这意味着卷积核覆盖50个采样点 = 0.5秒的脑电波
        # 0.5秒足够捕捉到一些基本的EEG波形特征
        # （比如睡眠纺锤波约0.5-1.5秒，K复合波约0.5-1.0秒）
        # -------------------------------------------------------
        first_filter_size = int(self.config["sampling_rate"] / 2.0)

        # 步长（stride）：每次卷积移动的距离
        # sampling_rate/16 = 100/16 约等于 6
        # stride=6意味着每6个采样点做一次卷积计算
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        # ============================================================
        # 3. CNN部分（特征提取器）
        #
        # nn.Sequential：按顺序执行的一系列层
        # 数据像流水线一样，依次经过每一层
        #
        # 【整体结构】
        # 输入(1,3000) -> Conv1(128通道) -> BN -> ReLU -> MaxPool -> Dropout
        #   -> Conv2(128) -> BN -> ReLU -> Conv3(128) -> BN -> ReLU
        #   -> Conv4(128) -> BN -> ReLU -> MaxPool -> Flatten -> Dropout
        #   -> 输出(2048,)
        #
        # 【CNN的核心思想】
        # 用多个不同大小的卷积核在信号上滑动，检测各种波形模式。
        # 第一层大卷积核（size=50）：检测宽范围的波形
        # 后面的小卷积核（size=8）：在提取到的特征图上进一步提取精细特征
        # 池化层（MaxPool）：降采样，保留最显著的特征
        # BatchNorm：让每层的输出保持稳定的统计分布，训练更快更稳定
        # ReLU：激活函数，引入非线性
        # Dropout：随机丢弃部分神经元，防止过拟合
        # ============================================================
        self.cnn = nn.Sequential(

            # -------------------------------------------------------
            # 第1层：大卷积核 Conv1 (kernel_size=50, stride=6)
            # -------------------------------------------------------
            # 输入形状: (batch, 1, 3000)
            # 先padding再卷积，保证尺寸计算正确
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),
            # 在序列两端各补22个0
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(
                    in_channels=1,                     # 输入通道数：单通道EEG
                    out_channels=128,                  # 输出通道数：128个特征图
                    kernel_size=first_filter_size,     # 卷积核大小：50
                    stride=first_filter_stride,        # 步长：6
                    bias=False                         # 不用偏置（BatchNorm会做平移）
                ))
            ])),
            # 输出形状: (batch, 128, ~496)
            # 128个特征图，每个特征图长度约496
            # 每个特征图代表一种"波形模式检测器"

            # BatchNorm（批归一化）
            # 【作用】让每层输出的均值为0、方差为1
            # 【为什么需要？】深度学习在训练时，每层输入的分布会不断变化
            # （内部协变量偏移），BatchNorm能稳定这个分布，让训练更稳定
            # momentum=0.01：均值和方差的移动平均更新速度
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            # ReLU（修正线性单元）：f(x) = max(0, x)
            # 【作用】引入非线性，让网络能学习复杂的模式
            # 如果不加激活函数，多层线性叠加还是线性，根本学不了复杂的东西
            # inplace=True：直接修改输入张量，节省内存
            nn.ReLU(inplace=True),

            # -------------------------------------------------------
            # 池化层1：MaxPool (kernel_size=8, stride=8)
            # -------------------------------------------------------
            # 【什么是最大池化？】
            # 在一个小窗口内取最大值作为输出
            # 比如窗口大小8、步长8：每8个点取最大值，长度降为1/8
            # 【作用】
            # 1. 降采样：减少数据量，降低计算量
            # 2. 提取最显著特征：只保留最强的响应
            # 3. 一定的平移不变性：特征位置稍微偏移也不影响
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),
            nn.MaxPool1d(kernel_size=8, stride=8),

            # Dropout（随机丢弃）
            # 每次训练时，随机让50%的神经元输出为0
            # 【作用】防止协同适应（co-adaptation），强迫网络学到更鲁棒的特征
            # 相当于同时训练了多个不同的"子网络"，取集成效果
            nn.Dropout(p=0.5),

            # -------------------------------------------------------
            # 第2层：小卷积核 Conv2 (kernel_size=8, stride=1)
            # -------------------------------------------------------
            # stride=1：不降采样，逐点滑动
            # 在Conv1提取的特征基础上，进一步提取组合特征
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),
            nn.Sequential(OrderedDict([
                ('conv2', nn.Conv1d(
                    in_channels=128, out_channels=128,
                    kernel_size=8, stride=1, bias=False
                ))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            # -------------------------------------------------------
            # 第3层：小卷积核 Conv3 (kernel_size=8, stride=1)
            # -------------------------------------------------------
            # 堆叠多层小卷积核可以增加感受野
            # 2层3x3卷积 约等于 1层5x5卷积（在2D中）
            # 堆叠的好处：中间有非线性激活，表达能力更强
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),
            nn.Sequential(OrderedDict([
                ('conv3', nn.Conv1d(
                    in_channels=128, out_channels=128,
                    kernel_size=8, stride=1, bias=False
                ))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            # -------------------------------------------------------
            # 第4层：小卷积核 Conv4 (kernel_size=8, stride=1)
            # -------------------------------------------------------
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(
                    in_channels=128, out_channels=128,
                    kernel_size=8, stride=1, bias=False
                ))
            ])),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            # -------------------------------------------------------
            # 池化层2：MaxPool (kernel_size=4, stride=4)
            # -------------------------------------------------------
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),
            nn.MaxPool1d(kernel_size=4, stride=4),

            # Flatten：把多维特征"展平"为一维向量
            # (batch, 128, 16) -> (batch, 128*16) = (batch, 2048)
            # 这2048个数字就是CNN从30秒脑电中提取的"特征摘要"
            nn.Flatten(),

            nn.Dropout(p=0.5),
        )

        # ============================================================
        # 4. LSTM部分（时序建模器）
        #
        # 【为什么CNN后面还要加LSTM？】
        # CNN处理的每个epoch是"独立"的，它不知道这个epoch前后发生了什么。
        # 但睡眠分期不是独立事件——你刚在N2睡了5分钟，下一分钟
        # 最可能是继续N2，或者进入N3/REM，但不太可能突然变成Wake。
        # LSTM就是用来捕捉这种"时序上下文"的。
        #
        # 【LSTM的原理简述】
        # LSTM有三个"门"来控制信息流动：
        # - 遗忘门：决定要丢弃过去记忆中的哪些信息
        # - 输入门：决定要记住当前输入中的哪些信息
        # - 输出门：决定要输出哪些信息给下一时刻
        # 通过这种机制，LSTM可以选择性地长期"记住"重要的时序模式。
        #
        # 【参数解释】
        # input_size=2048: 每个时间步的输入维度（CNN输出的特征）
        # hidden_size=128: 隐藏状态的维度（记忆容量）
        # num_layers=1: 只用一层LSTM（TinySleepNet追求轻量）
        # batch_first=True: 输入形状为 (batch, seq, feature)
        # ============================================================
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=self.config['n_rnn_units'],  # 128
            num_layers=1,
            batch_first=True
        )

        # LSTM输出后的Dropout
        self.rnn_dropout = nn.Dropout(p=0.5)

        # ============================================================
        # 5. 全连接分类层
        #
        # nn.Linear(in_features, out_features): 全连接层
        # 把LSTM输出的128维特征映射到5个类别的"分数"（logits）
        # 后面会接Softmax把这些分数转成概率
        #
        # 【为什么是5类？】
        # 对应5个睡眠阶段：Wake(0)、N1(1)、N2(2)、N3(3)、REM(4)
        # Move(5)和Unk(6)在实际训练中被忽略
        # ============================================================
        self.fc = nn.Linear(self.config['n_rnn_units'], 5)

    def forward(self, x, state):
        """前向传播：定义数据如何从输入到输出

        参数:
            x: tensor, 形状 (batch*seq_length, 1, 3000)
               批量输入的脑电信号
            state: tuple (h0, c0)
                   LSTM的初始隐藏状态
                   h0: (1, batch, 128) — 隐藏状态
                   c0: (1, batch, 128) — 细胞状态（LSTM的"长期记忆"）

        返回:
            y_pred: tensor, 形状 (batch*seq_length, 5)
                    每个epoch属于5个类别的分数（logits）
            state:  tuple, 更新后的LSTM状态
        """
        # ==================== Step 1: CNN编码 ====================
        # 输入 (300, 1, 3000) -> CNN -> 输出 (300, 2048)
        # 每个30秒epoch被压缩为2048维特征向量
        x = self.cnn(x)

        # ==================== Step 2: 重塑为LSTM输入格式 ====================
        # LSTM需要的输入形状：(batch, seq_length, input_size)
        # 所以将 (300, 2048) 重塑为 (15, 20, 2048)
        # -1 表示自动推断这一维的大小（300/20=15）
        x = x.view(-1, self.config['seq_length'], 2048)

        # 断言检查特征维度是否正确
        assert x.shape[-1] == 2048

        # ==================== Step 3: LSTM时序建模 ====================
        # 输入 (15, 20, 2048) -> LSTM -> 输出 (15, 20, 128)
        # LSTM逐个处理20个时间步，每个时间步输出128维的隐藏状态
        # state也会被更新：新的h和c包含了前20个epoch的"记忆"
        x, state = self.rnn(x, state)

        # ==================== Step 4: 展平 + 分类 ====================
        # (15, 20, 128) -> (300, 128)
        x = x.reshape(-1, self.config['n_rnn_units'])

        # Dropout
        x = self.rnn_dropout(x)

        # 全连接层：128维 -> 5维分类分数
        x = self.fc(x)

        return x, state


if __name__ == '__main__':
    from torchsummaryX import summary
    from config.sleepedf import train

    model = TinySleepNet(config=train)
    state = (torch.zeros(size=(1, 2, 128)),
             torch.zeros(size=(1, 2, 128)))
    summary(model, torch.randn(size=(2, 1, 3000)), state)
