# ============================================================
# model.py — 模型封装类（训练与评估的核心逻辑）
#
# 【这个文件是干什么的？】
# 如果说 network.py 定义了模型的"骨架"（网络结构），
# 那 model.py 就是给骨架注入了"灵魂"——训练流程。
#
# 它负责：
#   1. 创建网络实例并移动到GPU
#   2. 配置优化器（Adam）和损失函数（交叉熵）
#   3. 实现完整的"训练一个epoch"流程
#   4. 实现"评估/测试"流程
#   5. 保存最优模型
#   6. 用TensorBoard记录训练曲线
#
# 【训练一个batch的完整流程（背下来！）】
#   ① 获取一个batch的数据
#   ② 梯度清零（optimizer.zero_grad()）
#   ③ 前向传播（forward）：计算预测值
#   ④ 计算损失（loss）：比较预测值和真实值
#   ⑤ 反向传播（backward）：计算梯度
#   ⑥ 梯度裁剪（clip_grad）：防止梯度爆炸
#   ⑦ 更新参数（optimizer.step()）
#   ⑧ 记录指标（loss、准确率、F1等）
# ============================================================

import torch
import torch.nn as nn
import os
import timeit
import numpy as np
import sklearn.metrics as skmetrics
from network import TinySleepNet
from torch.optim import Adam
from tensorboardX import SummaryWriter
import logging

logger = logging.getLogger("default_log")


class Model:
    """模型封装类。

    把网络、优化器、损失函数、TensorBoard等组合在一起，
    提供 train_with_dataloader() 和 evaluate_with_dataloader() 两个核心方法。
    """

    def __init__(self, config=None, output_dir='./output', use_rnn=False,
                 testing=False, use_best=False, device=None):
        """初始化模型。

        参数:
            config:     dict, 配置参数（来自 config/sleepedf.py）
            output_dir: str,  输出目录（保存模型checkpoint、日志、TensorBoard数据）
            use_rnn:    bool, 是否使用RNN（这个参数未实际使用，保留兼容）
            testing:    bool, 是否为测试模式
            use_best:   bool, 是否加载最优checkpoint（测试时用）
            device:     torch.device, 计算设备（'cuda:0'或'cpu'）
        """
        # ========== 1. 创建网络 ==========
        self.tsn = TinySleepNet(config)
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint')
        self.best_ckpt_path = os.path.join(self.output_dir, 'best_ckpt')
        self.weights_path = os.path.join(self.output_dir, 'weights')
        self.log_dir = os.path.join(self.output_dir, 'log')

        # ---- 将网络移动到GPU/CPU ----
        self.device = device
        self.tsn.to(device)

        # ========== 2. 优化器：Adam ==========
        #
        # 【Adam优化器简介】
        # Adam = Adaptive Moment Estimation
        # 是目前最常用的深度学习优化器之一。
        # 它结合了两种方法的思想：
        #   - Momentum（动量）：积累历史梯度方向，加速收敛
        #   - RMSProp（自适应学习率）：每个参数有独立的学习率
        #
        # 【参数含义】
        # lr (learning_rate): 学习率=1e-4，控制更新步长
        # betas=(0.9, 0.999):
        #   beta1=0.9  -> 动量衰减率（考虑最近10次梯度）
        #   beta2=0.999 -> 学习率衰减率（考虑最近1000次梯度平方）
        # eps=1e-8: 防止除零错误
        # -------------------------------------------------------
        self.optimizer_all = Adam(
            self.tsn.parameters(),
            lr=config['learning_rate'],
            betas=(config['adam_beta_1'], config['adam_beta_2']),
            eps=config['adam_epsilon']
        )

        # ========== 3. 损失函数：交叉熵损失 ==========
        #
        # 【交叉熵（Cross-Entropy）是什么？】
        # 衡量预测概率分布和真实分布之间的"距离"。
        # 对于分类任务，如果模型预测的类别概率和真实标签越接近，loss越小。
        #
        # 公式：Loss = -Sigma y_true * log(y_pred)
        # 其中 y_true 是 one-hot 编码的真实标签
        # y_pred 是模型预测的各类别概率
        #
        # reduce=False: 不对loss求平均，保留每个样本的loss值
        # 这样后面可以按样本加权、按类别加权
        # -------------------------------------------------------
        self.CE_loss = nn.CrossEntropyLoss(reduce=False)

        # ========== 4. TensorBoard 可视化 ==========
        #
        # SummaryWriter 会将训练过程中的标量（loss、accuracy等）、
        # 计算图等信息写入文件。
        # 训练完成后，在终端运行：
        #   tensorboard --logdir=./output
        # 就可以在浏览器中查看训练曲线了。
        # -------------------------------------------------------
        self.train_writer = SummaryWriter(os.path.join(self.log_dir, 'train'))

        # 记录计算图（可以在TensorBoard中看到网络结构）
        # 这里需要给一个示例输入来"追踪"计算图
        example_batch = self.config['batch_size'] * self.config['seq_length']
        self.train_writer.add_graph(
            self.tsn,
            input_to_model=(
                torch.rand(size=(example_batch, 1, 3000)).to(device),
                (torch.zeros(size=(1, self.config['batch_size'], 128)).to(device),
                 torch.zeros(size=(1, self.config['batch_size'], 128)).to(device))
            )
        )

        # ---- 训练计数器 ----
        self.global_epoch = 0   # 当前进行的epoch数
        self.global_step = 0    # 当前进行的参数更新步数（每个batch+1）

        # ========== 5. 测试模式：加载最优模型 ==========
        if testing and use_best:
            best_ckpt_path = os.path.join(self.best_ckpt_path, 'best_model.ckpt')
            self.tsn.load_state_dict(torch.load(best_ckpt_path))
            logger.info(f'load best model from {best_ckpt_path}')

    def get_current_epoch(self):
        return self.global_epoch

    def pass_one_epoch(self):
        self.global_epoch = self.global_epoch + 1

    # ============================================================
    # train_with_dataloader — 训练一个epoch
    #
    # 【什么是"一个epoch"？】
    # 1个epoch = 使用训练集的所有数据训练一遍模型
    # 完整训练通常需要200个epoch
    #
    # 【流程】
    # 遍历数据生成器产生的所有batch，对每个batch：
    #   1. 准备数据（转tensor、重塑形状、送GPU）
    #   2. （如果是新序列）初始化LSTM状态
    #   3. 前向传播 -> 计算预测值
    #   4. 计算损失（交叉熵 + 样本加权 + 类别加权 + L2正则）
    #   5. 反向传播 -> 计算梯度
    #   6. 梯度裁剪 -> 防止梯度爆炸
    #   7. 更新参数
    #   8. 记录结果
    # ============================================================
    def train_with_dataloader(self, minibatches):
        """使用数据生成器训练一个epoch。

        参数:
            minibatches: generator, 数据生成器
                每次 yield (x, y, w, sl, re)
                x: 脑电信号  y: 标签  w: 权重（0=填充，1=真实数据）
                sl: 各受试者当前批次的真正长度
                re: 是否是新序列的开始（重置LSTM状态）

        返回:
            outputs: dict, 包含训练指标
                - train/loss: 平均损失
                - train/accuracy: 准确率
                - train/f1_score: 宏平均F1分数
                - train/cm: 混淆矩阵
                - train/duration: 耗时
        """
        # ---- 设置为训练模式 ----
        # 训练模式与评估模式的区别：
        #   - Dropout: 训练时随机丢弃，评估时全部保留
        #   - BatchNorm: 训练时用batch统计量，评估时用全局统计量
        self.tsn.train()
        start = timeit.default_timer()

        preds, trues, losses, outputs = ([], [], [], {})

        # ---- 核心循环：遍历每个batch ----
        for x, y, w, sl, re in minibatches:

            # ---------- 1. 数据准备 ----------
            # 输入的 x 形状为 (batch_size, seq_length, ...)
            # 展平为 (batch_size*seq_length, 1, 3000)
            # 这样CNN可以一次性处理所有样本
            x = torch.from_numpy(x).view(
                self.config['batch_size'] * self.config['seq_length'],
                1, 3000
            )
            y = torch.from_numpy(y)
            w = torch.from_numpy(w)

            # ---------- 2. 初始化LSTM状态 ----------
            # LSTM需要"记忆"之前的睡眠状态
            # 但如果这是新受试者的第一个batch，记忆应该清空
            # re=True 表示需要重置
            if re:
                # h0: (1, batch, 128) - 初始隐藏状态全0
                # c0: (1, batch, 128) - 初始细胞状态全0
                state = (
                    torch.zeros(size=(1, self.config['batch_size'],
                                      self.config['n_rnn_units'])),
                    torch.zeros(size=(1, self.config['batch_size'],
                                      self.config['n_rnn_units']))
                )
                state = (state[0].to(self.device), state[1].to(self.device))

            # ---------- 3. 梯度清零 ----------
            # PyTorch的反向传播会累积梯度
            # 每个batch开始前必须清零，否则梯度会叠加
            self.optimizer_all.zero_grad()

            # 数据移到GPU/CPU
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)

            # ---------- 4. 前向传播 ----------
            # 输入 x -> TinySleepNet -> 输出 y_pred (300, 5)
            y_pred, state = self.tsn.forward(x, state)

            # 分离LSTM状态：阻止梯度沿时间反向传播到上一个batch
            # 因为我们只在当前batch内反向传播
            state = (state[0].detach(), state[1].detach())

            # ---------- 5. 计算损失 ----------
            # ① 基础交叉熵损失
            loss = self.CE_loss(y_pred, y)

            # ② 按样本加权：填充位置的loss置0
            #   w=1：真实数据，正常计算loss
            #   w=0：填充数据，loss归零（不参与训练）
            loss = torch.mul(loss, w)

            # ③ 按类别加权：解决类别不平衡
            # 【为什么需要类别加权？】
            # 在睡眠数据中，N2阶段样本最多，N1和N3样本很少。
            # 如果不加权，模型会"忽略"样本少的类别（因为贡献loss小）。
            # 加权后，模型会更关注这些"少数派"。
            one_hot = torch.zeros(len(y), self.config['n_classes']).to(self.device)\
                           .scatter_(1, y.unsqueeze(dim=1), 1)
            # class_weights: [1.0, 1.5, 1.0, 1.0, 1.0]
            # N1的权重1.5最高，因为N1最难分
            sample_weight = torch.mm(
                one_hot,
                torch.Tensor(self.config['class_weights']).to(self.device).unsqueeze(dim=1)
            ).view(-1)
            loss = torch.mul(loss, sample_weight).sum() / w.sum()

            # ④ L2正则化（Weight Decay）
            # 【为什么要有L2正则化？】
            # 防止模型参数变得过大 -> 防止过拟合
            # 原理：大的权重意味着模型对某些特征"过于敏感"
            # 惩罚大的权重，让模型参数保持"紧凑"
            cnn_weights = [parm for name, parm in self.tsn.cnn.named_parameters()
                          if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2   # L2 = 1/2 * sum(w^2)
            reg_loss = self.config['l2_weight_decay'] * reg_loss

            # 总损失 = 加权交叉熵 + L2正则项
            loss = loss + reg_loss

            # ---------- 6. 反向传播 ----------
            loss.backward()

            # ---------- 7. 梯度裁剪 ----------
            # 【为什么需要梯度裁剪？】
            # RNN/LSTM训练中常见的问题：梯度可能变得极大（梯度爆炸），
            # 导致参数更新过大，训练崩溃。
            # 梯度裁剪把梯度的范数限制在max_norm以内。
            # 这里限制梯度的L2范数不超过5.0。
            nn.utils.clip_grad_norm_(
                self.tsn.parameters(),
                max_norm=self.config['clip_grad_value'],
                norm_type=2
            )

            # ---------- 8. 更新参数 ----------
            self.optimizer_all.step()

            losses.append(loss.detach().cpu().numpy())
            self.global_step += 1

            # ---------- 9. 记录预测结果 ----------
            # argmax: 取5个类别中得分最高的作为预测结果
            # 例如 y_pred=[-2.1, 0.5, 3.2, -1.0, 0.8] -> argmax=2 -> N2
            tmp_preds = np.reshape(
                np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                (self.config['batch_size'], self.config['seq_length'])
            )
            tmp_trues = np.reshape(
                y.cpu().detach().numpy(),
                (self.config['batch_size'], self.config['seq_length'])
            )
            # 只保留真实数据部分（排除填充的0）
            for i in range(self.config['batch_size']):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        # ========== 计算评估指标 ==========
        # Accuracy: 预测正确的比例
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()

        # Macro F1-Score: 每个类别的F1分数的平均值
        # F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
        # macro平均: 先算每个类的F1，再取平均（不考虑类别样本数量）
        # 相比accuracy，F1对样本少的类别更公平
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average='macro')

        # 混淆矩阵：cm[i][j] = 真实类别i、预测为j的样本数
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])

        stop = timeit.default_timer()
        duration = stop - start

        outputs.update({
            'global_step': self.global_step,
            'train/trues': trues,
            'train/preds': preds,
            'train/accuracy': acc,
            'train/loss': all_loss,
            'train/f1_score': f1_score,
            'train/cm': cm,
            'train/duration': duration,
        })

        self.global_epoch += 1
        return outputs

    # ============================================================
    # evaluate_with_dataloader — 评估/验证一个epoch
    #
    # 【和训练的区别】
    # 1. 切换到评估模式（self.tsn.eval()）
    #    - Dropout被关闭（所有神经元都激活）
    #    - BatchNorm使用全局统计量（而不是batch统计量）
    # 2. torch.no_grad()：不计算梯度
    #    - 节省大量内存（不需要存储中间变量用于反向传播）
    #    - 计算速度更快
    # 3. 不反向传播、不更新参数
    # ============================================================
    def evaluate_with_dataloader(self, minibatches):
        """评估一个epoch（不更新参数，只计算指标）。

        参数/返回值格式与 train_with_dataloader 相同。
        """
        self.tsn.eval()  # 切换到评估模式
        start = timeit.default_timer()

        preds, trues, losses, outputs = ([], [], [], {})

        # 不计算梯度
        with torch.no_grad():
            for x, y, w, sl, re in minibatches:
                x = torch.from_numpy(x).view(
                    self.config['batch_size'] * self.config['seq_length'],
                    1, 3000
                )
                y = torch.from_numpy(y)
                w = torch.from_numpy(w)

                if re:
                    state = (
                        torch.zeros(size=(1, self.config['batch_size'],
                                          self.config['n_rnn_units'])),
                        torch.zeros(size=(1, self.config['batch_size'],
                                          self.config['n_rnn_units']))
                    )
                    state = (state[0].to(self.device), state[1].to(self.device))

                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)

                y_pred, state = self.tsn.forward(x, state)
                state = (state[0].detach(), state[1].detach())

                # 损失计算与训练时相同（但不反向传播）
                loss = self.CE_loss(y_pred, y)
                loss = torch.mul(loss, w)
                one_hot = torch.zeros(len(y), self.config['n_classes']).to(self.device)\
                               .scatter_(1, y.unsqueeze(dim=1), 1)
                sample_weight = torch.mm(
                    one_hot,
                    torch.Tensor(self.config['class_weights']).to(self.device).unsqueeze(dim=1)
                ).view(-1)
                loss = torch.mul(loss, sample_weight).sum() / w.sum()

                losses.append(loss.detach().cpu().numpy())
                tmp_preds = np.reshape(
                    np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                    (self.config['batch_size'], self.config['seq_length'])
                )
                tmp_trues = np.reshape(
                    y.cpu().detach().numpy(),
                    (self.config['batch_size'], self.config['seq_length'])
                )
                for i in range(self.config['batch_size']):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        # 计算指标
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average='macro')
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])

        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            'test/trues': trues,
            'test/preds': preds,
            'test/loss': all_loss,
            'test/accuracy': acc,
            'test/f1_score': f1_score,
            'test/cm': cm,
            'test/duration': duration,
        }
        return outputs

    def save_best_checkpoint(self, name):
        """保存当前模型的参数为最优checkpoint。

        torch.save: 只保存模型参数（state_dict），不保存整个模型对象
        这样：
          - 文件更小
          - 兼容性更好（模型结构变了也能加载参数）
        加载时需要先创建同样结构的网络，再 load_state_dict()
        """
        if not os.path.exists(self.best_ckpt_path):
            os.makedirs(self.best_ckpt_path)
        save_path = os.path.join(self.best_ckpt_path, '{}.ckpt'.format(name))
        torch.save(self.tsn.state_dict(), save_path)
        logger.info('Saved best checkpoint to {}'.format(save_path))
