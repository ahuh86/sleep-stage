# ============================================================
# data.py — 数据加载模块
#
# 【这个文件是干什么的？】
# 深度学习的第一步就是把数据从硬盘读到内存里。
# 这个文件负责：
#   1. 根据受试者ID找到对应的数据文件
#   2. 从 .npz 文件中加载脑电信号（x）和睡眠标签（y）
#   3. 统一数据类型（float32用于模型输入，int32用于标签）
#
# 【数据格式说明】
# 预处理后的数据存储在 .npz 文件中（NumPy的压缩格式），
# 每个文件包含一整夜的睡眠记录。
# .npz 文件里有三个键：
#   - 'x': 脑电信号数组，形状 (n_epochs, n_samples)
#   - 'y': 标签数组，形状 (n_epochs,)  取值 0~6
#   - 'fs': 采样率（Hz）
#
# 一个"epoch" = 30秒的脑电信号片段
# 采样率100Hz → 每个epoch有 100×30 = 3000个采样点
# ============================================================

import os
import re
import numpy as np


def get_subject_files(dataset, files, sid):
    """根据受试者编号（Subject ID），从文件列表中找到该受试者的数据文件。

    【为什么需要这个函数？】
    一个数据集的文件夹里有很多 .npz 文件，每个文件代表一晚的睡眠记录。
    一个受试者可能有多个晚上的记录（例如连续采集了2晚）。
    文件名包含受试者编号信息，我们需要用正则表达式来匹配。

    参数:
        dataset: str, 数据集名称。不同数据集的文件命名规则不同：
                 - "sleepedf": 如 SC4001E0.npz（SC=睡眠 cassette，4=数据集版本，001=受试者编号）
                 - "mass":     如 SS3-001 PSG.npz
                 - "isruc":    如 subject1.npz
        files: list of str, 文件夹中所有 .npz 文件的路径列表
        sid:   int, 受试者编号（从0开始）

    返回:
        list of str, 该受试者对应的数据文件路径列表（一个人可能有多个文件）
    """
    # ---- 不同数据集有不同的命名规则，这里用正则表达式匹配 ----
    if "mass" in dataset:
        # ---- MASS 数据集：文件名格式 SS3-001 PSG.npz ----
        # 正则表达式：f".*-00{str(sid+1).zfill(2)} PSG.npz"
        #
        # 以 sid=0 为例，str(0+1).zfill(2) = "01":
        #   .*     → 匹配任意字符任意次（匹配 "SS3-" 部分）
        #   -00    → 精确匹配 "-00"
        #   01     → sid+1，补零到2位 → "01"
        #   " "    → 匹配空格
        #   PSG    → 精确匹配 "PSG"
        #   .npz   → 点号 + "npz"（.npz 后缀）
        # 完整匹配：SS3-001 PSG.npz
        #
        # 注意：MASS 用 sid+1（从1开始计），sid=0 对应文件中的 "001"
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"

    elif "sleepedf" in dataset:
        # ---- Sleep-EDF 数据集的正则表达式（逐段拆解） ----
        # 文件名如 "SC4001E0.npz"，以 sid=0 为例（受试者编号 001）：
        #
        # 【文件命名规则】
        #   SC = Sleep Cassette（卧室采集，自然睡眠）
        #   ST = Sleep Telemetry（实验室采集，药物干扰）
        #   4/7 = 数据集版本号
        #   001 = 受试者编号（3位，必须补零到3位）
        #   E0/E1 = 晚数编号（E0=第1晚，E1=第2晚）
        #
        # 【正则表达式逐段解释】
        #   f"S[CT][47]{str(sid+1).zfill(3)}[a-zA-Z0-9]+\\.npz$"
        #
        #   当 sid=0，str(0+1).zfill(3)="001"，完整正则：
        #   S[CT][47]001[a-zA-Z0-9]+\.npz$
        #
        #   逐段匹配 "SC4001E0.npz"：
        #   ┌ S      → 固定前缀，匹配 "S"（Sleep 首字母）
        #   ├ [CT]   → 匹配 C 或 T（字符类，选一个）
        #   ├ [47]   → 匹配 4 或 7（数据集版本）
        #   ├ 001    → sid+1 补零到3位，精确匹配受试者编号
        #   ├ [a-zA-Z0-9]+  → 匹配晚数编号 "E0"（一到多个字母数字）
        #   ├ \.     → 转义点号（正则中 . 表示任意字符，\. 才是真正点号）
        #   └ npz$   → 以 "npz" 结尾（$ 锚定到字符串末尾）
        #
        # 【为什么 zfill(3) 比 zfill(2) 好？】
        # 这3位编号必须"完整匹配"，不能只匹配前2位。
        # 假设用 zfill(2)：sid=0 → "00"
        #   则 "00" 不仅能匹配 "001"，还能匹配 "002"、"003"...
        #   这样 sid=0 会错误地拿到其他受试者的文件！
        # zfill(3) 保证 "001" 只精确匹配 "001"，绝不串到别的受试者。
        #
        reg_exp = f"S[CT][47]{str(sid+1).zfill(3)}[a-zA-Z0-9]+\\.npz$"
    elif "isruc" in dataset:
        # ISRUC 数据集：subject1.npz
        reg_exp = f"subject{sid+1}.npz"
    else:
        raise Exception("Invalid datasets.")

    # ---- 用正则表达式逐个匹配文件名 ----
    #   这里用的是 re.compile(...).search(f) 而不是 re.match(...)
    #
    #   re.match()   → 只从字符串开头匹配
    #   re.search()  → 在整个字符串中搜索，找到第一个匹配位置
    #   因为我们的正则表达式以 "S" 或 "subject" 开头，
    #   无论用 match 还是 search，效果是一样的。
    #   但用 search 更安全，万一文件路径前面还有目录名，
    #   search 也能正确匹配（比如 "data/sleepedf/SC4001E0.npz"）。
    #
    #   re.compile(reg_exp)：
    #   把字符串"编译"成正则对象，提升多次匹配的效率。
    #   虽然这里只有少量文件，但养成 compile 的习惯是好的。
    #
    #   enumerate(files) 返回 (索引, 文件名) 对：
    #   i = 0, f = "SC4001E0.npz"
    #   i = 1, f = "SC4002E0.npz"
    #   这里只用到了 f（文件名），但有时调试时需要 i（文件序号）。
    #
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
    return subject_files
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
    return subject_files


def load_data(subject_files):
    """加载一个或多个 .npz 文件中的脑电信号和睡眠阶段标签。

    【数据加载流程】
    1. 遍历该受试者的所有 .npz 文件（可能有多个晚上的记录）
    2. 对每个文件，用 np.load() 读取脑电信号、标签、采样率
    3. 检查所有文件的采样率是否一致
    4. 调整数据形状以匹配模型输入要求
    5. 统一数据类型

    参数:
        subject_files: list of str, 数据文件的路径列表

    返回:
        signals:       list of numpy arrays, 每个元素形状 (n_epochs, 3000, 1, 1)
                       n_epochs = 该晚的epoch数量
                       3000 = 30秒 × 100Hz
                       后面的两个1是额外的维度（为了兼容旧版代码）
        labels:        list of numpy arrays, 每个元素形状 (n_epochs,)
        sampling_rate: float, 采样率（Hz），确保所有文件一致
    """
    signals = []
    labels = []
    sampling_rate = None

    for sf in subject_files:
        # np.load() 读取 .npz 文件
        # .npz 是 NumPy 的压缩存档格式，可以包含多个数组
        with np.load(sf) as f:
            x = f['x']   # 脑电信号，形状 (n_epochs, n_samples)
            y = f['y']   # 睡眠阶段标签，形状 (n_epochs,)
            fs = f['fs'] # 采样率（标量），如 100.0

            # ---- 检查采样率一致性 ----
            # 如果多个文件的采样率不一致，模型无法处理
            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # ---- 数据形状调整 ----
            # np.squeeze: 去掉维度为1的轴（如果有的话）
            x = np.squeeze(x)

            # 添加两个维度：为了让数据兼容2D卷积的输入格式
            # 原始:    (n_epochs, 3000)
            # 处理后:  (n_epochs, 3000, 1, 1)
            # 注意：实际上后面用的是1D卷积，这两个额外的维度是历史遗留
            x = x[:, :, np.newaxis, np.newaxis]

            # ---- 数据类型转换 ----
            # float32: PyTorch默认的浮点精度，比float64省一半内存
            x = x.astype(np.float32)
            # int32: 标签通常用整数表示
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate

