# ============================================================
# trainer.py — 训练调度器（多折训练的"总指挥"）
#
# 【这个文件是干什么的？】
# train.py 是训练"一折"的脚本，而 trainer.py 负责：
#   1. 配置参数解析
#   2. 遍历指定范围的折（比如从折0到折19）
#   3. 对每一折调用 train.py 里的 train() 函数
#   4. 为每折设置不同的随机种子（保证结果可复现且不重复）
#
# 【典型用法】
# # 训练所有20折（Sleep-EDF数据集）：
# python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19
#
# # 分批训练（比如在一台4GPU机器上，每块GPU跑5折）：
# GPU 0: python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 4
# GPU 1: python trainer.py --db sleepedf --gpu 1 --from_fold 5 --to_fold 9
# GPU 2: python trainer.py --db sleepedf --gpu 2 --from_fold 10 --to_fold 14
# GPU 3: python trainer.py --db sleepedf --gpu 3 --from_fold 15 --to_fold 19
# ============================================================

import warnings
warnings.filterwarnings('ignore')
import argparse
import importlib
import os

from train import train


def run(args, db, gpu, from_fold, to_fold, suffix='', random_seed=42):
    """遍历指定范围的折，对每折执行训练。

    参数:
        db:          str, 数据集名称（'sleepedf' 或 'sleepedfx'）
        gpu:         int, GPU设备编号
        from_fold:   int, 起始折索引（包含）
        to_fold:     int, 结束折索引（包含）
        suffix:      str, 输出目录后缀（用于区分不同实验）
        random_seed: int, 基础随机种子（每折在此基础上+fold_idx）
    """
    # ---- 配置文件路径 ----
    # 根据数据集名称找到对应的配置文件
    config_file = os.path.join('config', '{}.py'.format(db))
    spec = importlib.util.spec_from_file_location('*', config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # ---- 输出目录 ----
    output_dir = 'out_{}{}'.format(db, suffix)

    # ---- 参数验证 ----
    assert from_fold <= to_fold
    assert to_fold < config.params['n_folds']

    # ---- 逐折训练 ----
    for fold_idx in range(from_fold, to_fold + 1):
        train(
            args=args,
            config_file=config_file,
            fold_idx=fold_idx,
            output_dir=os.path.join(output_dir, 'train'),
            log_file=os.path.join(output_dir, 'train_{}.log'.format(gpu)),
            restart=True,
            random_seed=random_seed + fold_idx,  # 每折种子不同
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TinySleepNet 多折训练调度器')
    parser.add_argument('--db', type=str, required=True,
                        help='数据集名称: sleepedf 或 sleepedfx')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU编号（从0开始）')
    parser.add_argument('--from_fold', type=int, required=True,
                        help='起始折索引（包含）')
    parser.add_argument('--to_fold', type=int, required=True,
                        help='结束折索引（包含）')
    parser.add_argument('--suffix', type=str, default='',
                        help='输出目录后缀')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--test_seq_len', type=int, default=20,
                        help='测试时序列长度')
    parser.add_argument('--test_batch_size', type=int, default=15,
                        help='测试时批大小')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='训练轮数')
    args = parser.parse_args()

    run(
        args=args,
        db=args.db,
        gpu=args.gpu,
        from_fold=args.from_fold,
        to_fold=args.to_fold,
        suffix=args.suffix,
        random_seed=args.random_seed,
    )
