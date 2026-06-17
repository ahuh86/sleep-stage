# ============================================================
# script/util.py — 辅助函数
#
# 提供计算1D卷积"same padding"的数学工具函数。
#
# 【什么是"same padding"？】
# 在卷积操作中，padding决定输出尺寸的计算方式：
#   - 'valid' padding: 不补0，输出尺寸 = (输入 - 卷积核)/步长 + 1
#   - 'same' padding:  补0使输出尺寸 = ceil(输入/步长)
#
# 这个函数就是用来计算"same padding"需要左右各补多少个0。
# ============================================================

def same_padding_1d(in_length, kernel_size, stride):
    """计算1D卷积的same padding值。

    参数:
        in_length:   int, 输入序列的长度
        kernel_size: int, 卷积核大小
        stride:      int, 步长

    返回:
        (pad_left, pad_right): int tuple
        左边和右边各需要补多少个0
    """
    if in_length % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (in_length % stride), 0)
    pad_left = pad // 2
    pad_right = pad - pad_left
    return pad_left, pad_right


if __name__ == '__main__':
    in_length = 63
    kernel_size = 4
    stride = 4
    pad_left, pad_right = same_padding_1d(in_length, kernel_size, stride)
    print(pad_left, pad_right)
