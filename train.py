# ============================================================
# train.py — 单折训练脚本
#
# 【这个文件是干什么的？】
# 这是执行"一折"训练的完整流程脚本。
# "一折"（fold）是K折交叉验证中的一次训练。
#
# 【K折交叉验证（K-Fold Cross Validation）】
# 为什么要把数据分成K份轮流训练？
# 因为我们需要"公正"地评估模型性能。
# 只用一次训练/测试划分可能运气不好（比如测试集太难或太简单）。
# K折交叉验证让每份数据都当过测试集，取K次的平均结果，
# 这样评估更稳定、更可靠。
#
# 【处理流程概览】
# 1. 加载配置 → 2. 读取受试者ID → 3. 划分训练/验证/测试集
# 4. 加载EEG数据 → 5. 创建模型 → 6. 训练200个epoch
# 7. 每个epoch：数据增强 → 训练 → 验证 → 测试 → 记录
# 8. 保存最优模型
# ============================================================

import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import torch

from data import load_data, get_subject_files
from model import Model
from minibatching import iterate_batch_multiple_seq_minibatches
from utils import print_n_samples_each_class, load_seq_ids
from logger import get_logger


def train(args, config_file, fold_idx, output_dir, log_file,
          restart=False, random_seed=42):
    """执行一折的完整训练流程。

    参数:
        args:        argparse.Namespace, 命令行参数
        config_file: str, 配置文件路径（如 config/sleepedf.py）
        fold_idx:    int, 当前折的索引 (0 ~ n_folds-1)
        output_dir:  str, 输出目录
        log_file:    str, 日志文件路径
        restart:     bool, 是否重新开始（删除旧的输出目录）
        random_seed: int, 随机种子（保证每次运行结果可复现）
    """
    # ========== Step 1: 加载配置文件 ==========
    # importlib 可以从任意文件路径动态加载Python模块
    # 这里加载的是 config/sleepedf.py 或 config/sleepedfx.py
    spec = importlib.util.spec_from_file_location('*', config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.train  # 使用训练模式配置

    # ========== Step 2: 创建输出目录 ==========
    output_dir = os.path.join(output_dir, str(fold_idx))
    if restart:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # 删除旧的输出
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # ========== Step 3: 初始化日志 ==========
    logger = get_logger(log_file, level='info')

    # ========== Step 4: 查找数据文件 ==========
    # glob 搜索文件夹中所有 .npz 文件
    subject_files = glob.glob(os.path.join(config['data_dir'], '*.npz'))

    # ========== Step 5: 加载受试者ID ==========
    # 从 sleepedf.txt 文件中读取受试者编号
    # 这个文件预先生成了所有受试者的编号和顺序
    fname = '{}.txt'.format(config['dataset'])
    seq_sids = load_seq_ids(fname)
    logger.info('Load generated SIDs from {}'.format(fname))
    logger.info('SIDs ({}): {}'.format(len(seq_sids), seq_sids))

    # ========== Step 6: K折划分 ==========
    # 将受试者ID均匀分成K份（fold）
    fold_pids = np.array_split(seq_sids, config['n_folds'])

    # 当前折：第 fold_idx 份作为测试集
    test_sids = fold_pids[fold_idx]

    # 其余作为训练集（后续还会从中分出一部分作为验证集）
    train_sids = np.setdiff1d(seq_sids, test_sids)

    # 从训练集中随机分出10%作为验证集
    n_valids = round(len(train_sids) * 0.10)

    # 设置随机种子，确保每次划分一致
    np.random.seed(random_seed)
    valid_sids = np.random.choice(train_sids, size=n_valids, replace=False)
    train_sids = np.setdiff1d(train_sids, valid_sids)

    logger.info('Train SIDs: ({}) {}'.format(len(train_sids), train_sids))
    logger.info('Valid SIDs: ({}) {}'.format(len(valid_sids), valid_sids))
    logger.info('Test SIDs: ({}) {}'.format(len(test_sids), test_sids))

    # ========== Step 7: 加载数据 ==========
    # ---- 训练集 ----
    train_files = []
    for sid in train_sids:
        train_files.append(get_subject_files(
            dataset=config['dataset'], files=subject_files, sid=sid,
        ))
    train_files = np.hstack(train_files)
    train_x, train_y, _ = load_data(train_files)  # train_x是list，每个元素是一个受试者的数据

    # ---- 验证集 ----
    valid_files = []
    for sid in valid_sids:
        valid_files.append(get_subject_files(
            dataset=config['dataset'], files=subject_files, sid=sid,
        ))
    valid_files = np.hstack(valid_files)
    valid_x, valid_y, _ = load_data(valid_files)

    # ---- 测试集 ----
    test_files = []
    for sid in test_sids:
        test_files.append(get_subject_files(
            dataset=config['dataset'], files=subject_files, sid=sid,
        ))
    test_files = np.hstack(test_files)
    test_x, test_y, _ = load_data(test_files)

    # ---- 打印数据统计信息 ----
    logger.info('Training set (n_night_sleeps={})'.format(len(train_y)))
    for _x in train_x:
        logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(train_y))

    logger.info('Validation set (n_night_sleeps={})'.format(len(valid_y)))
    for _x in valid_x:
        logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(valid_y))

    logger.info('Test set (n_night_sleeps={})'.format(len(test_y)))
    for _x in test_x:
        logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(test_y))

    # ========== Step 8: 设置类别权重 ==========
    if config.get('weighted_cross_ent') is None:
        config['weighted_cross_ent'] = False
        logger.info('  Weighted cross entropy: Not specified --> default: {}'.format(
            config['weighted_cross_ent']))
    else:
        logger.info('  Weighted cross entropy: {}'.format(
            config['weighted_cross_ent']))

    if config['weighted_cross_ent']:
        # N1 (索引1) 的权重为1.5
        # 因为N1阶段样本最少、特征最模糊、最难分类
        config['class_weights'] = np.asarray([1., 1.5, 1., 1., 1.], dtype=np.float32)
    else:
        config['class_weights'] = np.asarray([1., 1., 1., 1., 1.], dtype=np.float32)
    logger.info('  Weighted cross entropy: {}'.format(config['class_weights']))

    # ========== Step 9: 创建模型 ==========
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info('using device {}'.format(args.gpu))
    model = Model(
        config=config,
        output_dir=output_dir,
        use_rnn=True,
        testing=False,
        use_best=False,
        device=device,
    )

    # ---- 打印数据增强配置 ----
    logger.info('Data Augmentation')
    logger.info('  Sequence: {}'.format(config['augment_seq']))
    logger.info('  Signal full: {}'.format(config['augment_signal_full']))

    # ============================================================
    # Step 10: 主训练循环
    # ============================================================
    best_acc = -1   # 记录最优验证准确率
    best_mf1 = -1   # 记录最优验证F1分数
    update_epoch = -1

    config['n_epochs'] = args.n_epochs

    for epoch in range(model.get_current_epoch(), config['n_epochs']):

        # -------------------------------------------------------
        # 10a: 数据增强
        # -------------------------------------------------------
        # 打乱受试者顺序（每次epoch不一样，增加训练随机性）
        shuffle_idx = np.random.permutation(np.arange(len(train_x)))

        # Signal-level augmentation: 随机平移信号
        # 对原始EEG信号做小幅度（10%）的循环平移
        # 【为什么有用？】
        # 想象一下：脑电波检测睡眠纺锤波时，纺锤波可能出现在epoch的
        # 任意位置。如果不做平移，模型可能学到"纺锤波只在第500个采样点附近"，
        # 这显然是错的。平移增强了模型的平移不变性。
        percent = 0.1
        aug_train_x = np.copy(train_x)
        aug_train_y = np.copy(train_y)
        for i in range(len(aug_train_x)):
            offset = np.random.uniform(-percent, percent) * aug_train_x[i].shape[1]
            roll_x = np.roll(aug_train_x[i], int(offset))
            if offset < 0:
                aug_train_x[i] = roll_x[:-1]
                aug_train_y[i] = aug_train_y[i][:-1]
            if offset > 0:
                aug_train_x[i] = roll_x[1:]
                aug_train_y[i] = aug_train_y[i][1:]
            roll_x = None
            assert len(aug_train_x[i]) == len(aug_train_y[i])

        # 创建训练数据生成器（用增强后的数据）
        aug_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            aug_train_x, aug_train_y,
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            shuffle_idx=shuffle_idx,
            augment_seq=config['augment_seq'],
        )

        # -------------------------------------------------------
        # 10b: 训练一个epoch
        # -------------------------------------------------------
        train_outs = model.train_with_dataloader(aug_minibatch_fn)

        # -------------------------------------------------------
        # 10c: 验证
        # -------------------------------------------------------
        valid_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            valid_x, valid_y,
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            shuffle_idx=None,
            augment_seq=False,
        )
        valid_outs = model.evaluate_with_dataloader(valid_minibatch_fn)

        # -------------------------------------------------------
        # 10d: 测试
        # -------------------------------------------------------
        test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            test_x, test_y,
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            shuffle_idx=None,
            augment_seq=False,
        )
        test_outs = model.evaluate_with_dataloader(test_minibatch_fn)

        # -------------------------------------------------------
        # 10e: 记录到TensorBoard
        # -------------------------------------------------------
        writer = model.train_writer
        writer.add_scalar(tag='e_losses/train', scalar_value=train_outs['train/loss'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_losses/valid', scalar_value=valid_outs['test/loss'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_losses/test', scalar_value=test_outs['test/loss'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_accuracy/train', scalar_value=train_outs['train/accuracy'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_accuracy/valid', scalar_value=valid_outs['test/accuracy'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_accuracy/test', scalar_value=test_outs['test/accuracy'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_f1_score/train', scalar_value=train_outs['train/f1_score'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_f1_score/valid', scalar_value=valid_outs['test/f1_score'],
                          global_step=train_outs['global_step'])
        writer.add_scalar(tag='e_f1_score/test', scalar_value=test_outs['test/f1_score'],
                          global_step=train_outs['global_step'])

        # -------------------------------------------------------
        # 10f: 打印本轮结果
        # -------------------------------------------------------
        logger.info(
            '[e{}/{} s{}] TR (n={}) l={:.4f} a={:.1f} f1={:.1f} ({:.1f}s)| '
            'VA (n={}) l={:.4f} a={:.1f}, f1={:.1f} ({:.1f}s) | '
            'TE (n={}) l={:.4f} a={:.1f}, f1={:.1f} ({:.1f}s)'.format(
                epoch + 1, config['n_epochs'], train_outs['global_step'],
                len(train_outs['train/trues']),
                train_outs['train/loss'],
                train_outs['train/accuracy'] * 100,
                train_outs['train/f1_score'] * 100,
                train_outs['train/duration'],
                len(valid_outs['test/trues']),
                valid_outs['test/loss'],
                valid_outs['test/accuracy'] * 100,
                valid_outs['test/f1_score'] * 100,
                valid_outs['test/duration'],
                len(test_outs['test/trues']),
                test_outs['test/loss'],
                test_outs['test/accuracy'] * 100,
                test_outs['test/f1_score'] * 100,
                test_outs['test/duration'],
            )
        )

        # -------------------------------------------------------
        # 10g: 保存最优模型
        # -------------------------------------------------------
        # 如果验证集上的准确率和F1分数都提高了，保存当前模型
        if best_acc < valid_outs['test/accuracy'] and \
           best_mf1 <= valid_outs['test/f1_score']:
            best_acc = valid_outs['test/accuracy']
            best_mf1 = valid_outs['test/f1_score']
            update_epoch = epoch + 1
            model.save_best_checkpoint(name='best_model')

        # -------------------------------------------------------
        # 10h: 打印混淆矩阵
        # -------------------------------------------------------
        if (epoch + 1) % config['evaluate_span'] == 0 or (epoch + 1) == config['n_epochs']:
            logger.info('>> Confusion Matrix')
            logger.info(test_outs['test/cm'])


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TinySleepNet 单折训练')
    parser.add_argument('--config_file', type=str, required=True,
                        help='配置文件路径（如 config/sleepedf.py）')
    parser.add_argument('--fold_idx', type=int, required=True,
                        help='当前折的索引（0 ~ n_folds-1）')
    parser.add_argument('--output_dir', type=str, default='./output/train',
                        help='输出目录')
    parser.add_argument('--restart', dest='restart', action='store_true',
                        help='是否重新开始（删除旧输出）')
    parser.add_argument('--no-restart', dest='restart', action='store_false')
    parser.add_argument('--log_file', type=str, default='./output/output.log',
                        help='日志文件路径')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号')
    parser.set_defaults(restart=False)
    args = parser.parse_args()

    train(
        args=args,
        config_file=args.config_file,
        fold_idx=args.fold_idx,
        output_dir=args.output_dir,
        log_file=args.log_file,
        restart=args.restart,
        random_seed=args.random_seed,
    )
