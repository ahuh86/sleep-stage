# ============================================================
# predict.py — 预测与模型评估脚本
#
# 【这个文件是干什么的？】
# 用训练好的模型对测试集进行预测，并计算各种评估指标。
# 这是判断模型"好不好用"的最终环节。
#
# 【评估指标详解】
# 1. Accuracy（准确率）：正确分类的比例
#    公式：(TP+TN) / (TP+TN+FP+FN)
#    缺点：各类别样本不平衡时，准确率可能"骗人"
#    比如：90%的样本是N2，模型全猜N2也有90%准确率，但实际毫无用处
#
# 2. Macro F1-Score（宏平均F1）
#    先算每个类别的F1分数，再取算术平均
#    公式：F1 = 2 * Precision * Recall / (Precision + Recall)
#    优点：不受类别不平衡影响，对每个类别"一视同仁"
#    这是睡眠分期中最重要的评价指标
#
# 3. Confusion Matrix（混淆矩阵）
#    矩阵的第i行第j列 = "真实类别i被预测为j的次数"
#    可以直观看到哪些类别容易混淆
#    比如N1经常被误判为Wake或N2（因为N1的特征本身就模糊）
#
# 4. Precision（精确率）和 Recall（召回率）
#    Precision = TP / (TP+FP) — 模型说"这是N1"时，有多大把握？
#    Recall = TP / (TP+FN) — 真正的N1中，模型找出了多少？
# ============================================================

import argparse
import glob
import importlib
import os
import numpy as np
import sklearn.metrics as skmetrics
import torch

from model import Model
from data import load_data, get_subject_files
from minibatching import iterate_batch_multiple_seq_minibatches
from utils import print_n_samples_each_class, load_seq_ids
from logger import get_logger


def compute_performance(cm):
    """从混淆矩阵中计算各类评估指标。

    混淆矩阵 cm[i][j] = 真实类别为i、预测为j的样本数

    参数:
        cm: numpy array, 形状 (n_classes, n_classes), 混淆矩阵

    返回:
        total:        int, 总样本数
        n_each_class: array, 每个类别的样本数
        acc:          float, 准确率
        mf1:          float, 宏平均F1
        precision:    array, 每个类别的精确率
        recall:       array, 每个类别的召回率
        f1:           array, 每个类别的F1分数
    """
    tp = np.diagonal(cm).astype(np.float)       # 真正例（对角线元素）
    tpfp = np.sum(cm, axis=0).astype(np.float)   # 每列之和 = 预测为该类的总数
    tpfn = np.sum(cm, axis=1).astype(np.float)   # 每行之和 = 实际为该类的总数
    acc = np.sum(tp) / np.sum(cm)                 # 准确率
    precision = tp / tpfp                         # 精确率 = TP / (TP+FP)
    recall = tp / tpfn                            # 召回率 = TP / (TP+FN)
    f1 = (2 * precision * recall) / (precision + recall)  # F1 = 2PR/(P+R)
    mf1 = np.mean(f1)                             # 宏平均F1

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(config_file, model_dir, output_dir, log_file, use_best=True):
    """执行预测流程。

    【流程】
    1. 遍历所有折（0 ~ n_folds-1）
    2. 对每折：加载对应折训练好的模型
    3. 对该折的测试集：逐一受试者、逐一晚上进行预测
    4. 保存每夜的预测结果（.npz文件）
    5. 汇总所有折的结果，计算总体指标

    参数:
        config_file: str, 配置文件路径
        model_dir:   str, 模型checkpoint所在目录（不同折的子目录为 model_dir/0/, model_dir/1/, ...）
        output_dir:  str, 输出目录
        log_file:    str, 日志文件路径
        use_best:    bool, 是否使用最优checkpoint
    """
    # ---- 加载配置 ----
    spec = importlib.util.spec_from_file_location('*', config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict  # 使用预测模式配置

    # ---- 创建输出目录 ----
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = get_logger(log_file, level='info')

    # ---- 获取数据文件 ----
    subject_files = glob.glob(os.path.join(config['data_dir'], '*.npz'))
    fname = '{}.txt'.format(config['dataset'])
    seq_sids = load_seq_ids(fname)

    # K折划分
    fold_pids = np.array_split(seq_sids, config['n_folds'])

    # 添加虚拟类别权重（预测时所有类别权重相同，不影响结果）
    config['class_weights'] = np.ones(config['n_classes'], dtype=np.float32)

    trues = []
    preds = []

    # ========== 遍历所有折进行预测 ==========
    for fold_idx in range(config['n_folds']):
        logger.info('------ Fold {}/{} ------'.format(fold_idx + 1, config['n_folds']))
        test_sids = fold_pids[fold_idx]
        logger.info('Test SIDs: ({}) {}'.format(len(test_sids), test_sids))

        # 加载对应折的训练好的模型
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        model = Model(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            use_rnn=True,
            testing=True,
            use_best=use_best,
            device=device,
        )

        s_trues = []
        s_preds = []

        # ---- 逐一受试者进行预测 ----
        for sid in test_sids:
            logger.info('Subject ID: {}'.format(sid))

            test_files = get_subject_files(
                dataset=config['dataset'], files=subject_files, sid=sid,
            )
            for vf in test_files:
                logger.info('Load files {} ...'.format(vf))

            test_x, test_y, _ = load_data(test_files)
            logger.info('Test set (n_night_sleeps={})'.format(len(test_y)))
            for _x in test_x:
                logger.info(_x.shape)
            print_n_samples_each_class(np.hstack(test_y))

            # ---- 逐夜预测 ----
            for night_idx, night_data in enumerate(zip(test_x, test_y)):
                night_x, night_y = night_data
                test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                    [night_x], [night_y],
                    batch_size=config['batch_size'],
                    seq_length=config['seq_length'],
                    shuffle_idx=None,
                    augment_seq=False,
                )
                test_outs = model.evaluate_with_dataloader(test_minibatch_fn)

                s_trues.extend(test_outs['test/trues'])
                s_preds.extend(test_outs['test/preds'])
                trues.extend(test_outs['test/trues'])
                preds.extend(test_outs['test/preds'])

                # ---- 保存每夜的预测结果 ----
                save_dict = {
                    'y_true': test_outs['test/trues'],
                    'y_pred': test_outs['test/preds'],
                }
                fname = os.path.basename(test_files[night_idx]).split('.')[0]
                save_path = os.path.join(output_dir, 'pred_{}.npz'.format(fname))
                np.savez(save_path, **save_dict)
                logger.info('Saved outputs to {}'.format(save_path))

        # ---- 打印该折的结果 ----
        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average='macro')
        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0, 1, 2, 3, 4])
        logger.info('n={}, acc={:.1f}, mf1={:.1f}'.format(
            len(s_preds), s_acc * 100.0, s_f1_score * 100.0,
        ))
        logger.info('>> Confusion Matrix')
        logger.info(s_cm)

    # ========== 打印总体结果 ==========
    acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average='macro')
    cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])

    logger.info('')
    logger.info('=== Overall ===')
    print_n_samples_each_class(trues)
    logger.info('n={}, acc={:.1f}, mf1={:.1f}'.format(
        len(preds), acc * 100.0, f1_score * 100.0,
    ))
    logger.info('>> Confusion Matrix')
    logger.info(cm)

    # ---- 详细指标 ----
    metrics = compute_performance(cm=cm)
    logger.info('Total: {}'.format(metrics[0]))
    logger.info('Number of samples from each class: {}'.format(metrics[1]))
    logger.info('Accuracy: {:.1f}'.format(metrics[2] * 100.0))
    logger.info('Macro F1-Score: {:.1f}'.format(metrics[3] * 100.0))
    logger.info('Per-class Precision: {}'.format(
        ' '.join(['{:.1f}'.format(m * 100.0) for m in metrics[4]])))
    logger.info('Per-class Recall: {}'.format(
        ' '.join(['{:.1f}'.format(m * 100.0) for m in metrics[5]])))
    logger.info('Per-class F1-Score: {}'.format(
        ' '.join(['{:.1f}'.format(m * 100.0) for m in metrics[6]])))

    # ---- 保存汇总结果 ----
    save_dict = {
        'y_true': trues,
        'y_pred': preds,
        'seq_sids': seq_sids,
        'config': config,
    }
    save_path = os.path.join(output_dir, '{}.npz'.format(config['dataset']))
    np.savez(save_path, **save_dict)
    logger.info('Saved summary to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TinySleepNet 预测与评估')
    parser.add_argument('--config_file', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--model_dir', type=str, default='./out_sleepedf/finetune',
                        help='模型目录')
    parser.add_argument('--output_dir', type=str, default='./output/predict',
                        help='输出目录')
    parser.add_argument('--log_file', type=str, default='./output/output.log',
                        help='日志文件路径')
    parser.add_argument('--use-best', dest='use_best', action='store_true',
                        help='使用最优checkpoint')
    parser.add_argument('--no-use-best', dest='use_best', action='store_false')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU编号')
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
    )
