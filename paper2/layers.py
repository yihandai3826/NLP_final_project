import math
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm as SN
from config import device


class BidirEncoder(nn.Module):
    """
    双向编码器，使用GRU、LSTM或Elman RNN作为基本单元。
    """
    
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        """
        初始化双向编码器。
        
        :param input_size: 输入特征的维度。
        :param hidden_size: 隐藏层的维度。
        :param cell: 使用的RNN类型，可以是'GRU'、'Elman'或'LSTM'。
        :param n_layers: RNN的层数。
        :param drop_ratio: Dropout比例。
        """
        super(BidirEncoder, self).__init__()
        self.cell_type = cell
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 根据RNN类型创建不同的RNN层
        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)

        # Dropout层
        self.dropout_layer = nn.Dropout(drop_ratio)

    def forward(self, embed_seq, input_lens=None):
        """
        前向传播函数。
        
        :param embed_seq: 输入序列的嵌入表示，形状为(B, L, emb_dim)。
        :param input_lens: 输入序列的长度，形状为(B)。
        :return: RNN的输出和状态。
        """
        embed_inps = self.dropout_layer(embed_seq)

        # 根据输入长度是否给定选择是否打包序列
        if input_lens is None:
            outputs, state = self.rnn(embed_inps, None)
        else:
            # 动态RNN
            total_len = embed_inps.size(1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embed_inps,
                input_lens, batch_first=True, enforce_sorted=False)
            outputs, state = self.rnn(packed, None)
            # 将打包的序列解包
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                batch_first=True, total_length=total_len)

        return outputs, state

    def init_state(self, batch_size):
        """
        初始化RNN的状态。
        
        :param batch_size: 批次大小。
        :return: 初始化的隐藏状态和（如果使用LSTM）单元状态。
        """
        # 初始化隐藏状态
        init_h = torch.zeros((self.n_layers*2, batch_size, self.hidden_size),
            requires_grad=False, device=device)

        # 如果是LSTM，还需要初始化单元状态
        if self.cell_type == 'LSTM':
            init_c = torch.zeros((self.n_layers*2, batch_size, self.hidden_size),
                requires_grad=False, device=device)
            return (init_h, init_c)
        else:
            return init_h

class Decoder(nn.Module):
    """
    解码器，用于生成序列的输出。
    使用GRU、LSTM或Elman RNN作为基本单元。
    """
    
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        """
        初始化解码器。
        
        :param input_size: 输入特征的维度。
        :param hidden_size: 隐藏层的维度。
        :param cell: 使用的RNN类型，可以是'GRU'、'Elman'或'LSTM'。
        :param n_layers: RNN的层数。
        :param drop_ratio: Dropout比例。
        """
        super(Decoder, self).__init__()
        self.dropout_layer = nn.Dropout(drop_ratio)

        # 根据RNN类型创建不同的RNN层
        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, embed_seq, last_state):
        """
        前向传播函数。
        
        :param embed_seq: 输入序列的嵌入表示，形状为(B, emb_dim)。
        :param last_state: 上一个时间步的状态，形状为(B, H)或(B, N, H)。
        :return: RNN的输出和状态。
        """
        embed_inps = self.dropout_layer(embed_seq)
        # 将状态增加一个维度以匹配RNN的输入要求
        output, state = self.rnn(embed_inps, last_state.unsqueeze(0))
        # 移除单一的时间步维度
        output = output.squeeze(1)  # (B, 1, N) -> (B, N)
        return output, state.squeeze(0)  # (B, H)

class MLP(nn.Module):
    """
    多层感知机（MLP），用于构建深度前馈神经网络。
    
    :param ori_input_size: 输入层的大小。
    :param layer_sizes: 各隐藏层的大小列表。
    :param activs: 各层激活函数的列表，可以是'tanh'、'relu'或'leakyrelu'。
    :param drop_ratio: Dropout比例。
    :param no_drop: 是否禁用Dropout。
    """

    def __init__(self, ori_input_size, layer_sizes, activs=None,
        drop_ratio=0.0, no_drop=False):
        super(MLP, self).__init__()

        layer_num = len(layer_sizes)

        # 使用OrderedDict来保持层的顺序
        orderedDic = OrderedDict()
        input_size = ori_input_size
        for i, (layer_size, activ) in enumerate(zip(layer_sizes, activs)):
            linear_name = 'linear_' + str(i)
            # 添加全连接层
            orderedDic[linear_name] = nn.Linear(input_size, layer_size)
            input_size = layer_size  # 更新输入大小为当前层的输出大小

            if activ is not None:
                assert activ in ['tanh', 'relu', 'leakyrelu']  # 确保激活函数有效

            active_name = 'activ_' + str(i)
            if activ == 'tanh':
                # 根据激活函数类型添加相应的激活层
                orderedDic[active_name] = nn.Tanh()
            elif activ == 'relu':
                orderedDic[active_name] = nn.ReLU()
            elif activ == 'leakyrelu':
                orderedDic[active_name] = nn.LeakyReLU(0.2)

            # 根据需要添加Dropout层
            if (drop_ratio > 0) and (i < layer_num-1) and (not no_drop):
                orderedDic["drop_" + str(i)] = nn.Dropout(drop_ratio)

        # 将所有层封装成一个Sequential模块
        self.mlp = nn.Sequential(orderedDic)

    def forward(self, inps):
        """
        前向传播函数。
        
        :param inps: 输入数据。
        :return: MLP的输出。
        """
        return self.mlp(inps)

class ContextLayer(nn.Module):
    """
    上下文层，用于结合解码器状态和先前的上下文信息来更新上下文向量。
    """
    
    def __init__(self, inp_size, out_size, kernel_size=3):
        """
        初始化上下文层。
        
        :param inp_size: 输入尺寸（解码器状态的特征维度）。
        :param out_size: 输出尺寸（上下文向量的维度）。
        :param kernel_size: 卷积核的大小，默认为3。
        """
        super(ContextLayer, self).__init__()
        # 一维卷积层，用于处理解码器状态
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)
        # 全连接层，用于结合卷积输出和先前的上下文信息
        self.linear = nn.Linear(out_size+inp_size, out_size)

    def forward(self, last_context, dec_states):
        """
        前向传播函数。
        
        :param last_context: 上一个上下文向量，形状为(B, context_size)。
        :param dec_states: 解码器状态，形状为(B, H, L)。
        :return: 更新后的上下文向量。
        """
        # 使用卷积层处理解码器状态，并调整维度
        hidden_feature = self.conv(dec_states).permute(0, 2, 1)  # (B, L_out, out_size)
        # 应用激活函数并计算平均值
        feature = torch.tanh(hidden_feature).mean(dim=1)  # (B, out_size)
        # 结合先前的上下文信息和卷积输出
        new_context = torch.tanh(self.linear(torch.cat([last_context, feature], dim=1)))
        return new_context

class Discriminator(nn.Module):
    """
    判别器，用于区分真实数据和生成数据。
    包含特征提取和分类两个部分，用于判断输入的潜在变量是否符合给定的类别标签。
    """
    
    def __init__(self, n_class1, n_class2, factor_emb_size,
        latent_size, drop_ratio):
        """
        初始化判别器。
        
        :param n_class1: 第一个分类任务的类别数。
        :param n_class2: 第二个分类任务的类别数。
        :param factor_emb_size: 因子嵌入的维度。
        :param latent_size: 潜在变量的维度。
        :param drop_ratio: Dropout比例。
        """
        super(Discriminator, self).__init__()
        # 特征提取层
        self.inp2feature = nn.Sequential(
            SN(nn.Linear(latent_size, latent_size)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(latent_size, factor_emb_size*2)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(factor_emb_size*2, factor_emb_size*2)),
            nn.LeakyReLU(0.2)
        )
        # 分类层
        self.feature2logits = SN(nn.Linear(factor_emb_size*2, 1))
        
        # 因子嵌入层，用于分类任务
        self.dis_fembed1 = SN(nn.Embedding(n_class1, factor_emb_size))
        self.dis_fembed2 = SN(nn.Embedding(n_class2, factor_emb_size))

    def forward(self, x, labels1, labels2):
        """
        前向传播函数。
        
        :param x: 潜在变量，形状为(B, latent_size)。
        :param labels1: 第一个分类任务的标签。
        :param labels2: 第二个分类任务的标签。
        :return: 分类逻辑值。
        """
        # 因子嵌入
        femb1 = self.dis_fembed1(labels1)
        femb2 = self.dis_fembed2(labels2)
        factor_emb = torch.cat([femb1, femb2], dim=1)  # (B, factor_emb_size*2)
        
        # 特征提取
        feature = self.inp2feature(x)  # (B, factor_emb_size*2)
        logits0 = self.feature2logits(feature).squeeze(1)  # (B)
        
        # 计算逻辑值
        logits = logits0 + torch.sum(feature*factor_emb, dim=1)
        return logits

class CBNLayer(nn.Module):
    """
    条件批归一化变换层（Conditional BatchNorm Transform Layer）。
    结合了线性变换、条件批归一化和激活函数。
    """
    
    def __init__(self, inp_size, out_size, n_classes):
        """
        初始化条件批归一化层。
        
        :param inp_size: 输入特征的维度。
        :param out_size: 输出特征的维度。
        :param n_classes: 类别数量，用于条件批归一化。
        """
        super(CBNLayer, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.n_classes = n_classes

        # 线性变换层
        self.linear = nn.Linear(self.inp_size, self.out_size)
        # 条件批归一化层
        self.cbn = CondtionalBatchNorm(self.n_classes, self.out_size)
        # 激活函数
        self.activ = nn.LeakyReLU(0.2)

    def forward(self, inps):
        """
        前向传播函数。
        
        :param inps: 输入数据和对应的标签，格式为[(B, inp_size), (B,)]。
        :return: 经过条件批归一化和激活函数处理的输出，以及原始标签。
        """
        x, y = inps[0], inps[1]
        # x: 输入特征，形状为(B, inp_size)
        # y: 类别标签，形状为(B,)
        
        # 应用线性变换
        h = self.linear(x)
        # 应用条件批归一化
        out = self.cbn(h, y)
        # 应用激活函数
        return (self.activ(out), y)

class PriorGenerator(nn.Module):
    """
    先验生成器，用于基于类别标签和关键状态生成潜在变量的先验分布。
    """
    
    def __init__(self, inp_size, latent_size, n_class1, n_class2, factor_emb_size):
        """
        初始化先验生成器。
        
        :param inp_size: 输入特征的维度。
        :param latent_size: 潜在变量的维度。
        :param n_class1: 第一个分类任务的类别数。
        :param n_class2: 第二个分类任务的类别数。
        :param factor_emb_size: 因子嵌入的维度。
        """
        super(PriorGenerator, self).__init__()

        # 因子嵌入层
        self.factor_embed1 = nn.Embedding(n_class1, factor_emb_size)
        self.factor_embed2 = nn.Embedding(n_class2, factor_emb_size)

        # 潜在变量的一半维度
        self.slatent_size = int(latent_size//2)

        # 生成第一个潜在变量的MLP
        self.mlp1 = nn.Sequential(
            CBNLayer(inp_size+factor_emb_size, latent_size, n_class1),
            CBNLayer(latent_size, latent_size, n_class1))
        # 第一个潜在变量的BatchNorm层
        self.bn1 = nn.Sequential(
            nn.Linear(latent_size, self.slatent_size),
            nn.BatchNorm1d(self.slatent_size))

        # 生成第二个潜在变量的MLP
        self.mlp2 = nn.Sequential(
            CBNLayer(inp_size+factor_emb_size, latent_size, n_class2),
            CBNLayer(latent_size, latent_size, n_class2))
        # 第二个潜在变量的BatchNorm层
        self.bn2 = nn.Sequential(
            nn.Linear(latent_size, self.slatent_size),
            nn.BatchNorm1d(self.slatent_size))

    def forward(self, key_state, labels1, labels2):
        """
        前向传播函数。
        
        :param key_state: 关键状态，形状为(B, inp_size)。
        :param labels1: 第一个分类任务的标签。
        :param labels2: 第二个分类任务的标签。
        :return: 生成的潜在变量，形状为(B, latent_size)。
        """
        # 因子嵌入
        factor1 = self.factor_embed1(labels1)
        factor2 = self.factor_embed2(labels2)

        # 批次大小
        batch_size = key_state.size(0)
        
        # 为第一个潜在变量生成噪声
        eps1 = torch.randn((batch_size, self.slatent_size),
            dtype=torch.float, device=key_state.device)
        # 拼接噪声、关键状态和因子嵌入
        cond1 = torch.cat([eps1, key_state, factor1], dim=1)
        # 通过MLP生成第一个潜在变量的先验
        prior1 = self.mlp1((cond1, labels1))[0]
        prior1 = self.bn1(prior1)

        # 为第二个潜在变量生成噪声
        eps2 = torch.randn((batch_size, self.slatent_size),
            dtype=torch.float, device=key_state.device)
        # 拼接噪声、关键状态和因子嵌入
        cond2 = torch.cat([eps2, key_state, factor2], dim=1)
        # 通过MLP生成第二个潜在变量的先验
        prior2 = self.mlp2((cond2, labels2))[0]
        prior2 = self.bn2(prior2)

        # 拼接两个潜在变量的先验
        return torch.cat([prior1, prior2], dim=1)

class PosterioriGenerator(nn.Module):
    """
    后验生成器，用于基于输入状态、关键状态和类别标签生成潜在变量的后验分布。
    """
    
    def __init__(self, inp_size, latent_size, n_class1, n_class2, factor_emb_size):
        """
        初始化后验生成器。
        
        :param inp_size: 输入特征的维度。
        :param latent_size: 潜在变量的维度。
        :param n_class1: 第一个分类任务的类别数。
        :param n_class2: 第二个分类任务的类别数。
        :param factor_emb_size: 因子嵌入的维度。
        """
        super(PosterioriGenerator, self).__init__()

        self.latent_size = latent_size

        # 因子嵌入层
        self.post_embed1 = nn.Embedding(n_class1, factor_emb_size)
        self.post_embed2 = nn.Embedding(n_class2, factor_emb_size)

        # 多层感知机，用于生成后验潜在变量
        self.mlp = MLP(
            inp_size+factor_emb_size*2+latent_size,
            layer_sizes=[512, latent_size, latent_size],
            activs=['leakyrelu', 'leakyrelu', None], no_drop=True)

    def forward(self, sen_state, key_state, labels1, labels2):
        """
        前向传播函数。
        
        :param sen_state: 输入状态，形状为(B, inp_size)。
        :param key_state: 关键状态，形状为(B, latent_size)。
        :param labels1: 第一个分类任务的标签。
        :param labels2: 第二个分类任务的标签。
        :return: 生成的后验潜在变量，形状为(B, latent_size)。
        """
        # 因子嵌入
        factor1 = self.post_embed1(labels1)
        factor2 = self.post_embed2(labels2)

        # 批次大小
        batch_size = key_state.size(0)
        
        # 为后验潜在变量生成噪声
        eps = torch.randn((batch_size, self.latent_size),
            dtype=torch.float, device=key_state.device)

        # 拼接输入状态、关键状态、因子嵌入和噪声
        cond = torch.cat([sen_state, key_state, factor1, factor2, eps], dim=1)
        # 通过MLP生成后验潜在变量
        z_post = self.mlp(cond)

        return z_post

class CondtionalBatchNorm(nn.Module):
    """
    条件批归一化层（Conditional BatchNorm），用于根据类别标签调整批归一化参数。
    """
    
    _version = 2

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1):
        """
        初始化条件批归一化层。
        
        :param num_classes: 类别数。
        :param num_features: 特征数（即输入通道数）。
        :param eps: 用于数值稳定性的小值。
        :param momentum: 移动平均的动量。
        """
        super(CondtionalBatchNorm, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 为每个类别学习一组 gamma 和 beta 参数
        self.delta_gamma = nn.Embedding(num_classes, num_features)
        self.delta_beta = nn.Embedding(num_classes, num_features)

        # 共享的 gamma 和 beta 参数
        self.gamma = nn.Parameter(torch.Tensor(1, num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features), requires_grad=True)

        # 运行时统计参数
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_running_stats(self):
        """
        重置运行时统计参数。
        """
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        """
        重置网络参数。
        """
        self.reset_running_stats()

        # 初始化 delta_gamma 和 delta_beta 为恒等矩阵和零向量
        self.delta_gamma.weight.data.fill_(1)
        self.delta_beta.weight.data.zero_()

        # 初始化 gamma 和 beta 参数
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, inps, labels):
        """
        前向传播函数。
        
        :param inps: 输入特征，形状为(B, D)。
        :param labels: 类别标签，形状为(B)。
        :return: 条件批归一化后的输出。
        """
        exp_avg_factor = 0.0
        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exp_avg_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exp_avg_factor = self.momentum

        B, D = inps.size(0), inps.size(1)

        mean = inps.mean(dim=0)  # (D)
        variance = inps.var(dim=0)  # (D)

        if self.training:
            running_mean = self.running_mean * (1 - exp_avg_factor) + mean * exp_avg_factor
            running_var = self.running_var * (1 - exp_avg_factor) + variance * exp_avg_factor

            mu = mean
            var = variance

            self.running_mean = running_mean
            self.running_var = running_var

        else:
            mu = self.running_mean
            var = self.running_var

        # 归一化输入特征
        x = (inps - mu.view(1, D).repeat(B, 1)) / torch.sqrt(var.view(1, D).repeat(B, 1) + self.eps)

        # 根据类别标签获取 delta_weight 和 delta_bias
        delta_weight = self.delta_gamma(labels)  # (B, D)
        delta_bias = self.delta_beta(labels)

        # 计算条件批归一化的权重和偏置
        weight = self.gamma.repeat(B, 1) + delta_weight
        bias = self.beta.repeat(B, 1) + delta_bias

        # 应用条件批归一化
        out = weight * x + bias

        return out

class GumbelSampler(object):
    """
    使用Gumbel-Softmax函数来返回离散型采样标签。
    该类提供了一种方法来生成硬标签（long type），而不是传统的one-hot标签或软标签（概率分布）。
    """
    
    def __init__(self):
        """
        初始化GumbelSampler。
        """
        super(GumbelSampler, self).__init__()
        self.__tau = 1.0  # 设置温度参数τ的初始值

    def set_tau(self, tau):
        """
        设置温度参数τ的值。
        τ的值需要在(0, 1]范围内。
        
        :param tau: 温度参数τ。
        """
        if 0.0 < tau <= 1.0:
            self.__tau = tau

    def get_tau(self):
        """
        获取当前的温度参数τ的值。
        
        :return: 当前的温度参数τ。
        """
        return self.__tau

    def __call__(self, logits):
        """
        调用Gumbel-Softmax函数生成采样标签。
        
        :param logits: 模型输出的logits，形状为(B, n_class)。
        :return: 采样后的硬标签，形状为(B,)。
        """
        # 使用Gumbel-Softmax函数生成软标签
        y_soft = F.gumbel_softmax(logits, tau=self.__tau, hard=False)
        
        # 通过取最大值生成硬标签
        y_hard = y_soft.max(dim=-1, keepdim=True)[1]  # [B, n_class]
        
        # 将硬标签转换为与软标签相同分布的标签，以便用于梯度下降
        y_hard = (y_hard.float() - y_soft).detach() + y_soft
        
        # 返回第一个类别的采样标签
        return y_hard[:, 0]

#-------------------------------------
#-------------------------------------
class LossWrapper(object):
    """
    损失函数包装器，用于封装不同的损失计算方法。
    """
    
    def __init__(self, pad_idx, sens_num, sen_len):
        """
        初始化损失函数包装器。
        
        :param pad_idx: 填充索引，用于在计算交叉熵损失时忽略填充的部分。
        :param sens_num: 句子的数量。
        :param sen_len: 句子的最大长度。
        """
        self.__sens_num = sens_num
        self.__sen_len = sen_len
        self.___pad_idx = pad_idx

        self.__gen_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # 用于生成任务的交叉熵损失
        self.__cl_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 用于分类任务的交叉熵损失

    def seq_ce_loss(self, outs, trgs):
        """
        计算序列的交叉熵损失。
        
        :param outs: 模型输出，形状为(B, L, V)。
        :param trgs: 目标标签，形状为(B, L)。
        :return: 交叉熵损失。
        """
        vocab_size = outs.size(2)
        trg_max_len = trgs.size(1)

        output = outs[:, 0:trg_max_len, :].contiguous().view(-1, vocab_size)
        target = trgs.contiguous().view(-1)
        return self.__gen_criterion(output, target)

    def cross_entropy_loss(self, all_outs, all_trgs):
        """
        计算所有句子的交叉熵损失。
        
        :param all_outs: 所有句子的模型输出，形状为(B, L, V) * sens_num。
        :param all_trgs: 所有句子的目标标签，形状为(B, L) * sens_num。
        :return: 所有句子的平均交叉熵损失。
        """
        batch_size, vocab_size = all_outs[0].size(0), all_outs[0].size(2)

        all_loss = []
        for step in range(0, self.__sens_num):
            all_loss.append(self.seq_ce_loss(all_outs[step], all_trgs[step]).unsqueeze(0))

        rec_loss = torch.cat(all_loss, dim=0)
        rec_loss = torch.mean(rec_loss)  # (sens_num)

        return rec_loss

    def bow_loss(self, bow_logits, all_trgs):
        """
        计算BOW（Bag of Words）损失。
        
        :param bow_logits: BOW模型的输出，形状为(B, V)。
        :param all_trgs: 所有句子的目标标签，形状为(B, L) * sens_num。
        :return: BOW损失。
        """
        all_loss = []
        for step in range(0, self.__sens_num):
            trgs = all_trgs[step]
            max_dec_len = trgs.size(1)
            for i in range(0, max_dec_len):
                all_loss.append(self.__gen_criterion(bow_logits, trgs[:, i]).unsqueeze(0))

        all_loss = torch.cat(all_loss, dim=0)
        bow_loss = all_loss.mean()  # [B, T, sens_num]
        return bow_loss

    def cl_loss(self, logits_w, logits_xw, combined_label, mask):
        """
        计算分类损失。
        
        :param logits_w: 有标签样本的logits，形状为(B, n_class)。
        :param logits_xw: 无标签样本的logits，形状为(B, n_class)。
        :param combined_label: 合并的标签，形状为(B,)。
        :param mask: 掩码，用于区分有标签和无标签样本，形状为(B,)。
        :return: 分别是有标签样本损失、无标签样本损失和熵损失。
        """
        cl_loss_w = self.__cl_criterion(logits_w, combined_label).mean()  # (B) -> (1)

        entropy_loss_xw = self.__get_entropy(logits_xw, 1-mask)

        cl_loss_xw = self.__cl_criterion(logits_xw, combined_label) * mask  # (B)
        cl_loss_xw = cl_loss_xw.sum() / (mask.sum()+1e-10)

        return cl_loss_w, cl_loss_xw, entropy_loss_xw

    def __get_entropy(self, logits, mask):
        """
        计算熵损失。
        
        :param logits: logits，形状为(B, n_class)。
        :param mask: 掩码，用于区分有标签和无标签样本，形状为(B,)。
        :return: 熵损失。
        """
        probs = F.softmax(logits, dim=-1)
        entropy = torch.log(probs+1e-10) * probs  # (B, n_class)

        entropy_loss = entropy.mean(dim=-1) * mask  # (B)

        entropy_loss = entropy_loss.sum() / (mask.sum()+1e-10)

        return entropy_loss

#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ScheduledOptim(object):
    """
    优化器包装器，实现预定的学习率调度策略。
    该类包装了一个PyTorch优化器，并根据预定的策略调整学习率。
    """

    def __init__(self, optimizer, warmup_steps, max_lr=5e-4, min_lr=3e-5, beta=0.55):
        """
        初始化ScheduledOptim。

        :param optimizer: 被包装的PyTorch优化器。
        :param warmup_steps: 热身步骤数，在学习率增加的阶段。
        :param max_lr: 最大学习率。
        :param min_lr: 最小学习率，用于热身后的学习率下限。
        :param beta: 学习率衰减的指数参数。
        """
        self.__optimizer = optimizer

        self._step = 0  # 当前步骤
        self._rate = 0  # 当前学习率

        self.__warmup_steps = warmup_steps  # 热身步数
        self.__max_lr = max_lr  # 最大学习率
        self.__min_lr = min_lr  # 最小学习率

        # 学习率衰减参数
        self.__alpha = warmup_steps**(-beta-1.0)
        self.__beta = -beta

        # 学习率缩放因子
        self.__scale = 1.0 / (self.__alpha*warmup_steps)

    def step(self):
        """
        更新参数和学习率。
        在每次优化器步骤后调用，更新学习率并执行优化器的步骤。
        """
        self._step += 1
        rate = self.rate()
        for p in self.__optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.__optimizer.step()

    def rate(self, step=None):
        """
        计算当前步骤的学习率。
        实现预定的学习率增加和衰减策略。

        :param step: 指定的步骤，如果为None，则使用当前步骤。
        :return: 计算得到的学习率。
        """
        if step is None:
            step = self._step
        # 根据步骤计算学习率
        lr = self.__max_lr*self.__scale*min(step*self.__alpha, step**(self.__beta))
        if step > self.__warmup_steps:
            lr = max(lr, self.__min_lr)
        return lr

    def zero_grad(self):
        """
        清零梯度。
        调用优化器的zero_grad方法。
        """
        self.__optimizer.zero_grad()

    def state_dict(self):
        """
        获取优化器的状态字典。
        返回被包装优化器的状态字典。
        """
        return self.__optimizer.state_dict()

    def load_state_dict(self, dic):
        """
        加载优化器的状态字典。
        将传入的状态字典加载到被包装的优化器中。

        :param dic: 优化器的状态字典。
        """
        self.__optimizer.load_state_dict(dic)

#---------------------------------------------------
class RateDecay(object):
    """
    基本的速率衰减类，用于不同类型的速率衰减，例如：
    - 教学强制比率（teacher forcing ratio）
    - Gumbel 温度
    - KL 退火（KL annealing）
    """
    
    def __init__(self, burn_down_steps, decay_steps, limit_v):
        """
        初始化速率衰减类。

        :param burn_down_steps: 燃烧阶段步数，超过此步数后开始衰减。
        :param decay_steps: 衰减阶段步数，用于计算衰减率。
        :param limit_v: 衰减的下限值。
        """
        self.step = 0  # 当前步数
        self.rate = 1.0  # 当前衰减率

        self.burn_down_steps = burn_down_steps  # 燃烧阶段步数
        self.decay_steps = decay_steps  # 衰减阶段步数

        self.limit_v = limit_v  # 衰减的下限值

    def decay_function(self):
        """
        计算衰减率的函数。
        具体的衰减函数需要在子类中实现。
        :return: 衰减后的速率。
        """
        # 该方法需要在子类中重构
        return self.rate

    def do_step(self):
        """
        执行一步衰减操作。
        更新当前步数和衰减率。
        :return: 当前的衰减率。
        """
        # 更新步数
        self.step += 1
        # 如果步数超过燃烧阶段，则更新衰减率
        if self.step > self.burn_down_steps:
            self.rate = self.decay_function()
        
        return self.rate

    def get_rate(self):
        """
        获取当前的衰减率。
        :return: 当前的衰减率。
        """
        return self.rate

class ExponentialDecay(RateDecay):
    """
    指数衰减类，用于模拟指数衰减过程。
    
    参数:
    burn_down_steps (int): 预热步数，即衰减开始前的步数。
    decay_steps (int): 衰减步数，即衰减过程持续的步数。
    min_v (float): 衰减的最小值。
    """
    def __init__(self, burn_down_steps, decay_steps, min_v):
        super(ExponentialDecay, self).__init__(
            burn_down_steps, decay_steps, min_v)
        # 计算衰减率
        self.__alpha = np.log(self.limit_v) / (-decay_steps)

    def decay_function(self):
        """
        计算当前步数下的衰减值。
        
        返回:
        float: 当前步数下的衰减值。
        """
        new_rate = max(np.exp(-self.__alpha * self.step), self.limit_v)
        return new_rate

class InverseLinearDecay(RateDecay):
    """
    逆线性衰减类，用于模拟逆线性衰减过程。
    
    参数:
    burn_down_steps (int): 预热步数，即衰减开始前的步数。
    decay_steps (int): 衰减步数，即衰减过程持续的步数。
    max_v (float): 衰减的最大值。
    """
    def __init__(self, burn_down_steps, decay_steps, max_v):
        super(InverseLinearDecay, self).__init__(
            burn_down_steps, decay_steps, max_v)
        # 计算衰减率
        self.__alpha = (max_v - 0.0) / decay_steps

    def decay_function(self):
        """
        计算当前步数下的衰减值。
        
        返回:
        float: 当前步数下的衰减值。
        """
        new_rate = min(self.__alpha * self.step, self.limit_v)
        return new_rate

class LinearDecay(RateDecay):
    """
    线性衰减类，用于模拟线性衰减过程。
    
    参数:
    burn_down_steps (int): 预热步数，即衰减开始前的步数。
    decay_steps (int): 衰减步数，即衰减过程持续的步数。
    max_v (float): 衰减的最大值。
    min_v (float): 衰减的最小值。
    """
    def __init__(self, burn_down_steps, decay_steps, max_v, min_v):
        super(LinearDecay, self).__init__(
            burn_down_steps, decay_steps, min_v)
        # 存储最大值和最小值
        self.__max_v = max_v
        # 计算衰减率
        self.__alpha = (self.__max_v - min_v) / decay_steps

    def decay_function(self):
        """
        计算当前步数下的衰减值。
        
        返回:
        float: 当前步数下的衰减值。
        """
        new_rate = max(self.__max_v - self.__alpha * self.step, self.limit_v)
        return new_rate
