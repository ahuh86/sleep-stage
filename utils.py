# ============================================================
# utils.py — 工具函数集合
#
# 【这个文件是干什么的？】
# 包含一些训练中用到的辅助功能：
#   1. 保存/加载受试者ID列表（用于K折交叉验证的划分）
#   2. 统计每个睡眠阶段有多少样本
#   3. 类别平衡处理（解决某些阶段样本过少的问题）
#
# 【为什么需要类别平衡？】
# 睡眠数据中，不同阶段的样本数量差异很大：
#   - N2 阶段最多（约占50%）
#   - N1 阶段很少（约占5%）
# 如果不做处理，模型会倾向于预测N2而忽略N1。
# ============================================================

import numpy as np
from sleepstage import class_dict
import logging

logger = logging.getLogger('default_log')


# ==================== ID保存/加载 ====================

def save_seq_ids(fname, ids):
    """将受试者ID序列保存到文本文件，每行一个ID。

    参数:
        fname: str, 文件名
        ids:   list or array, 受试者ID列表
    """
    with open(fname, 'w') as f:
        for _id in ids:
            f.write(str(_id) + '\n')


def load_seq_ids(fname):
    """从文本文件加载受试者ID序列。

    参数:
        fname: str, 文件名

    返回:
        numpy array, 受试者ID数组
    """
    ids = []
    with open(fname, 'r') as f:
        for line in f:
            ids.append(int(line.strip()))
    ids = np.asarray(ids)
    return ids


# ==================== 数据统计 ====================

def print_n_samples_each_class(labels):
    """打印每个睡眠阶段在数据集中的样本数量。

    【为什么关心这个？】
    如果某个阶段的样本特别少，模型的训练就会"偏科"。
    知道各类别的分布有助于决定是否需要做数据平衡处理。

    参数:
        labels: numpy array, 所有样本的标签
    """
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        logger.info('{}: {}'.format(class_dict[c], n_samples))


def compute_portion_each_class(labels):
    """计算每个睡眠阶段在数据集中的占比（0~1之间）。

    参数:
        labels: numpy array, 所有样本的标签

    返回:
        numpy array, 每个类别的占比
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    class_portions = np.zeros(len(unique_labels), dtype=np.float32)
    for c in unique_labels:
        n_class_samples = len(np.where(labels == c)[0])
        class_portions[c] = n_class_samples / float(n_samples)
    return class_portions


# ==================== 类别平衡策略 ====================

def get_balance_class_oversample(x, y):
    """【过采样法】让所有类别的样本数量相等。

    【做法】
    1. 找到样本数最多的那个类别
    2. 对其他类别进行"复制采样"，使它们也达到同样数量
    3. 如果复制后还不够，就随机抽取剩余的样本补上

    【优点】不会丢失任何数据
    【缺点】重复样本可能导致模型"过拟合"到这些重复的模式上
    【适用场景】数据量较少时

    参数:
        x: numpy array, 特征数据
        y: numpy array, 标签

    返回:
        (balance_x, balance_y): 平衡后的数据和标签
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        # 重复采样（复制现有样本）
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        # 如果还有剩余，随机补采
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    return balance_x, balance_y


def get_balance_class_sample(x, y):
    """【欠采样法】让所有类别的样本数量相等。

    【做法】
    1. 找到样本数最少的那个类别
    2. 对其他类别随机抽取同样数量的样本

    【优点】没有重复数据，更"干净"
    【缺点】丢弃了大量数据，可能丢失重要信息
    【适用场景】数据量足够大时

    参数:
        x: numpy array, 特征数据
        y: numpy array, 标签

    返回:
        (balance_x, balance_y): 平衡后的数据和标签
    """
    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        sample_idx = np.random.choice(idx, size=n_min_classes, replace=False)
        balance_x.append(x[sample_idx])
        balance_y.append(y[sample_idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    return balance_x, balance_y
