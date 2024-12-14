import math
import random
from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F
from layers import BidirEncoder, Decoder, MLP, ContextLayer, GumbelSampler
from layers import Discriminator, PriorGenerator, PosterioriGenerator
from config import device

def get_non_pad_mask(seq, pad_idx):
    """
    生成非填充（非pad）掩码。
    
    参数:
    seq (Tensor): 输入序列，形状为 [B, L]，其中 B 是批次大小，L 是序列长度。
    pad_idx (int): 填充值的索引。
    
    返回:
    Tensor: 非填充掩码，形状为 [B, L]，非填充元素为 1.0，填充元素为 0.0。
    """
    # 确保输入序列是二维的
    assert seq.dim() == 2
    # 创建一个与输入序列形状相同的掩码，非填充元素为 True，填充元素为 False
    mask = seq.ne(pad_idx).type(torch.float)
    # 将掩码转移到指定的设备上
    return mask.to(device)

def get_seq_length(seq, pad_idx):
    """
    计算序列的实际长度（不包括填充）。
    
    参数:
    seq (Tensor): 输入序列，形状为 [B, L]。
    pad_idx (int): 填充值的索引。
    
    返回:
    Tensor: 序列长度，形状为 [B]，每个元素表示对应序列的实际长度。
    """
    # 获取非填充掩码
    mask = get_non_pad_mask(seq, pad_idx)
    # 计算每个序列的非填充元素数量，即序列的实际长度
    lengths = mask.sum(dim=-1)
    # 将长度转换为长整型
    lengths = lengths.type(torch.long)
    return lengths


class MixPoetAUS(nn.Module):
    """
    MixPoetAUS模型，用于生成诗歌并预测诗歌的主题。

    参数:
    hps (dict): 包含模型超参数的字典。
    """
    def __init__(self, hps):
        super(MixPoetAUS, self).__init__()
        self.hps = hps

        # 模型参数
        self.vocab_size = hps.vocab_size  # 词汇表大小
        self.n_class1 = hps.n_class1  # 第一个主题类别的数量
        self.n_class2 = hps.n_class2  # 第二个主题类别的数量
        self.emb_size = hps.emb_size  # 嵌入层大小
        self.hidden_size = hps.hidden_size  # 隐藏层大小
        self.factor_emb_size = hps.factor_emb_size  # 因子嵌入层大小
        self.latent_size = hps.latent_size  # 潜在空间大小
        self.context_size = hps.context_size  # 上下文向量大小
        self.poem_len = hps.poem_len  # 诗歌长度
        self.sens_num = hps.sens_num  # 诗句数量
        self.sen_len = hps.sen_len  # 每句长度

        # 特殊索引
        self.pad_idx = hps.pad_idx  # 填充索引
        self.bos_idx = hps.bos_idx  # 起始索引

        # 起始向量
        self.bos_tensor = torch.tensor(hps.bos_idx, dtype=torch.long, device=device).view(1, 1)

        # 高斯采样工具
        self.gumbel_tool = GumbelSampler()

        # 构建位置输入以区分不同位置的行
        self.pos_inps = F.one_hot(torch.arange(0, self.sens_num), self.sens_num)
        self.pos_inps = self.pos_inps.type(torch.FloatTensor).to(device)

        # 构建组件
        self.layers = nn.ModuleDict()
        self.layers['embed'] = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.pad_idx)

        # 编码器
        self.layers['encoder'] = BidirEncoder(self.emb_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        # 解码器
        self.layers['decoder'] = Decoder(self.hidden_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        # 字符编码器
        self.layers['word_encoder'] = BidirEncoder(self.emb_size, self.emb_size, cell='Elman', drop_ratio=hps.drop_ratio)

        # 主题分类器
        self.layers['cl_xw1'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class1], activs=['relu', 'relu', None], drop_ratio=hps.drop_ratio)
        self.layers['cl_xw2'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class2], activs=['relu', 'relu', None], drop_ratio=hps.drop_ratio)

        # 无上下文的主题分类器
        self.layers['cl_w1'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class1], activs=['relu', 'relu', None], drop_ratio=hps.drop_ratio)
        self.layers['cl_w2'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class2], activs=['relu', 'relu', None], drop_ratio=hps.drop_ratio)

        # 因子嵌入
        self.layers['factor_embed1'] = nn.Embedding(self.n_class1, self.factor_emb_size)
        self.layers['factor_embed2'] = nn.Embedding(self.n_class2, self.factor_emb_size)

        # 先验和后验生成器
        self.layers['prior'] = PriorGenerator(
            self.emb_size*2+int(self.latent_size//2),
            self.latent_size, self.n_class1, self.n_class2, self.factor_emb_size)
        self.layers['posteriori'] = PosterioriGenerator(
            self.hidden_size*2+self.emb_size*2, self.latent_size,
            self.n_class1, self.n_class2, self.factor_emb_size)

        # 判别器
        self.layers['discriminator'] = Discriminator(self.n_class1, self.n_class2,
            self.factor_emb_size, self.latent_size, drop_ratio=hps.drop_ratio)

        # 输出投影
        self.layers['out_proj'] = nn.Linear(hps.hidden_size, hps.vocab_size)

        # 解码器初始状态MLP
        self.layers['dec_init'] = MLP(self.latent_size+self.emb_size*2+self.factor_emb_size*2,
            layer_sizes=[self.hidden_size-6], activs=['tanh'], drop_ratio=hps.drop_ratio)

        # 上下文向量映射
        self.layers['map_x'] = MLP(self.context_size+self.emb_size,
            layer_sizes=[self.hidden_size], activs=['tanh'], drop_ratio=hps.drop_ratio)

        # 更新上下文向量
        self.layers['context'] = ContextLayer(self.hidden_size, self.context_size)

        # 退火参数
        self.__tau = 1.0
        self.__teach_ratio = 1.0

        # 预训练解码器初始状态MLP
        self.layers['dec_init_pre'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size-6], activs=['tanh'], drop_ratio=hps.drop_ratio)

    # 设置退火参数
    def set_tau(self, tau):
        self.gumbel_tool.set_tau(tau)

    # 获取退火参数
    def get_tau(self):
        return self.gumbel_tool.get_tau()

    # 设置教学比例
    def set_teach_ratio(self, teach_ratio):
        if 0.0 < teach_ratio <= 1.0:
            self.__teach_ratio = teach_ratio

    # 获取教学比例
    def get_teach_ratio(self):
        return self.__teach_ratio

    #---------------------------------
    def dec_step(self, inp, state, context):
        
        emb_inp = self.layers['embed'](inp)

        x = self.layers['map_x'](torch.cat([emb_inp, context.unsqueeze(1)], dim=2))

        cell_out, new_state = self.layers['decoder'](x, state)
        out = self.layers['out_proj'](cell_out)
        return out, new_state


    def generator(self, dec_init_state, dec_inps, lengths, specified_teach=None):
        """
        生成诗歌。

        参数:
        dec_init_state (Tensor): 解码器的初始状态，形状为 (batch_size, hidden_size)。
        dec_inps (List[Tensor]): 每个元素为 (batch_size, max_dec_len) 的输入序列列表。
        lengths (Tensor): 每个序列的长度，形状为 (batch_size,)。
        specified_teach (float, optional): 指定的教学比例。若为 None，则使用 self.__teach_ratio。

        返回:
        all_outs (List[Tensor]): 每个元素为 (batch_size, max_dec_len, vocab_size) 的输出序列列表。
        """
        batch_size = dec_init_state.size(0)
        context = torch.zeros((batch_size, self.context_size),
            dtype=torch.float, device=device) # (B, context_size)

        all_outs = []
        if specified_teach is None:
            teach_ratio = self.__teach_ratio
        else:
            teach_ratio = specified_teach

        for step in range(0, self.sens_num):
            pos_inps = self.pos_inps[step, :].unsqueeze(0).repeat(batch_size, 1)

            state = torch.cat([dec_init_state, lengths, pos_inps], dim=-1) # (B, H)
            max_dec_len = dec_inps[step].size(1)

            outs = torch.zeros(batch_size, max_dec_len, self.vocab_size, device=device)
            dec_states = []

            # generate each line
            inp = self.bos_tensor.expand(batch_size, 1)
            for t in range(0, max_dec_len):
                out, state = self.dec_step(inp, state, context)
                outs[:, t, :] = out

                # teach force with a probability
                is_teach = random.random() < teach_ratio
                if is_teach or (not self.training):
                    inp = dec_inps[step][:, t].unsqueeze(1)
                else:
                    normed_out = F.softmax(out, dim=-1)
                    top1 = normed_out.data.max(1)[1]
                    inp  = top1.unsqueeze(1)

                dec_states.append(state.unsqueeze(2)) # (B, H, 1)

            # save each generated line
            all_outs.append(outs)

            # update the context vector
            # (B, 1, L)
            dec_mask = get_non_pad_mask(dec_inps[step], self.pad_idx).unsqueeze(1)
            states = torch.cat(dec_states, dim=2) # (B, H, L)
            context = self.layers['context'](context, states*dec_mask)

        return all_outs


    def computer_enc(self, inps, encoder):
        """
        计算编码器的输出。

        参数:
        inps (Tensor): 输入序列，形状为 (batch_size, length)。
        encoder (nn.Module): 编码器模块。

        返回:
        enc_outs (Tensor): 编码器的输出，形状为 (batch_size, length, hidden_size)。
        enc_state (Tensor): 编码器的最终状态，形状为 (batch_size, hidden_size)。
        """
        lengths = get_seq_length(inps, self.pad_idx)

        emb_inps = self.layers['embed'](inps) # (batch_size, length, emb_size)

        enc_outs, enc_state = encoder(emb_inps, lengths)

        return enc_outs, enc_state


    def get_factor_emb(self, condition, factor_id, label, mask):
        """
        获取因子嵌入。

        参数:
        condition (Tensor): 条件向量，形状为 (batch_size, hidden_size*2)。
        factor_id (int): 因子 ID，1 或 2。
        label (Tensor): 标签，形状为 (batch_size,)。
        mask (Tensor): 掩码，形状为 (batch_size,)。

        返回:
        factor_emb (Tensor): 因子嵌入，形状为 (batch_size, factor_emb_size)。
        logits_cl (Tensor): 分类器的输出，形状为 (batch_size, n_class)。
        fin_label (Tensor): 最终标签，形状为 (batch_size,)。
        """
        logits_cl = self.layers['cl_xw'+str(factor_id)](condition)
        sampled_label = self.gumbel_tool(logits_cl)

        fin_label = label.float() * mask + (1-mask) * sampled_label
        fin_label = fin_label.long()

        factor_emb = self.layers['factor_embed'+str(factor_id)](fin_label)

        return factor_emb, logits_cl, fin_label


    def get_prior_and_posterior(self, key_inps, vae_inps, factor_labels,
        factor_mask, ret_others=False):
        """
        获取先验和后验。

        参数:
        key_inps (Tensor): 关键词输入，形状为 (batch_size, key_len)。
        vae_inps (Tensor): VAE 输入，形状为 (batch_size, poem_len)。
        factor_labels (Tensor): 因子标签，形状为 (batch_size, 2)。
        factor_mask (Tensor): 因子掩码，形状为 (batch_size, 2)。
        ret_others (bool, optional): 是否返回其他信息。

        返回:
        z_prior (Tensor): 先验，形状为 (batch_size, latent_size)。
        z_post (Tensor): 后验，形状为 (batch_size, latent_size)。
        key_state (Tensor): 关键词状态，形状为 (batch_size, hidden_size*2)。
        factors (Tensor): 因子，形状为 (batch_size, factor_emb_size*2)。
        logits_cl_xw1 (Tensor): 第一个因子的分类器输出，形状为 (batch_size, n_class1)。
        logits_cl_xw2 (Tensor): 第二个因子的分类器输出，形状为 (batch_size, n_class2)。
        combined_label1 (Tensor): 第一个因子的最终标签，形状为 (batch_size,)。
        combined_label2 (Tensor): 第二个因子的最终标签，形状为 (batch_size,)。
        """
        _, vae_state = self.computer_enc(vae_inps, self.layers['encoder']) # (2, B, H)
        sen_state = torch.cat([vae_state[0, :, :], vae_state[1, :, :]], dim=-1) # [B, 2*H]
        # TODO: incorporate multiple keywords
        _, key_state0 = self.computer_enc(key_inps, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([sen_state, key_state], dim=1)
        factor_emb1, logits_cl_xw1, combined_label1 = self.get_factor_emb(condition, 1,
            factor_labels[:, 0], factor_mask[:, 0])
        factor_emb2, logits_cl_xw2, combined_label2 = self.get_factor_emb(condition, 2,
            factor_labels[:, 1], factor_mask[:, 1])

        factors = torch.cat([factor_emb1, factor_emb2], dim=-1)

        batch_size = key_state.size(0)
        eps = torch.randn((batch_size, self.latent_size), dtype=torch.float, device=device)
        z_post = self.layers['posteriori'](sen_state, key_state, combined_label1, combined_label2)
        z_prior = self.layers['prior'](key_state, combined_label1, combined_label2)

        if ret_others:

            return z_prior, z_post, key_state, factors,\
                logits_cl_xw1, logits_cl_xw2, combined_label1, combined_label2
        else:
            return z_prior, z_post, combined_label1, combined_label2


    def forward(self, key_inps, vae_inps, dec_inps, factor_labels, factor_mask, lengths,
        use_prior=False, specified_teach=None):

        z_prior, z_post, key_state, factors, logits_cl_xw1, logits_cl_xw2, cb_label1, cb_label2\
            = self.get_prior_and_posterior(key_inps, vae_inps, factor_labels, factor_mask, True)

        if use_prior:
            z = z_prior
        else:
            z = z_post

        dec_init_state = self.layers['dec_init'](torch.cat([z, key_state, factors], dim=-1)) # (B, H-2)
        all_gen_outs = self.generator(dec_init_state, dec_inps, lengths, specified_teach)

        logits_cl_w1 = self.layers['cl_w1'](key_state)
        logits_cl_w2 = self.layers['cl_w2'](key_state)

        return all_gen_outs, cb_label1, cb_label2, \
            logits_cl_xw1, logits_cl_xw2, logits_cl_w1, logits_cl_w2


    def classifier_graph(self, keys, poems, factor_id):
        """
        根据诗歌和关键词，使用给定因子ID进行分类。

        参数:
        keys (Tensor): 关键词序列。
        poems (Tensor): 诗歌序列。
        factor_id (int): 分类因子的ID（1或2）。

        返回:
        logits_xw (Tensor): 综合条件（诗歌+关键词）的分类 logits。
        logits_w (Tensor): 仅关键词的分类 logits。
        probs_xw (Tensor): 综合条件的分类概率。
        probs_w (Tensor): 仅关键词的分类概率。
        """
        _, poem_state0 = self.computer_enc(poems, self.layers['encoder'])
        poem_state = torch.cat([poem_state0[0, :, :], poem_state0[1, :, :]], dim=-1) # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([poem_state, key_state], dim=-1)

        logits_w = self.layers['cl_w'+str(factor_id)](key_state)
        logits_xw = self.layers['cl_xw'+str(factor_id)](condition)

        probs_w = F.softmax(logits_w, dim=-1)
        probs_xw = F.softmax(logits_xw, dim=-1)


        return logits_xw, logits_w, probs_xw, probs_w


    def dae_graph(self, keys, poems, dec_inps, lengths):
        """
        构建去噪自编码器图，用于预训练。

        参数:
        keys (Tensor): 关键词序列。
        poems (Tensor): 诗歌序列。
        dec_inps (List[Tensor]): 解码器输入序列列表。
        lengths (Tensor): 诗歌序列长度。

        返回:
        all_gen_outs (List[Tensor]): 生成的诗歌序列列表。
        """
        _, poem_state0 = self.computer_enc(poems, self.layers['encoder'])
        poem_state = torch.cat([poem_state0[0, :, :], poem_state0[1, :, :]], dim=-1) # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        dec_init_state = self.layers['dec_init_pre'](torch.cat([poem_state, key_state], dim=-1))

        all_gen_outs = self.generator(dec_init_state, dec_inps, lengths)

        return all_gen_outs

    def dae_parameter_names(self):
        """
        返回去噪自编码器所需的参数名称列表。
        """
        required_names = ['embed', 'encoder', 'word_encoder',
            'dec_init_pre', 'decoder', 'out_proj', 'context', 'map_x']
        return required_names

    def dae_parameters(self):
        """
        返回去噪自编码器所需的参数迭代器。
        """
        names = self.dae_parameter_names()

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    # -------------------------------------
    def classifier_parameter_names(self, factor_id):
        """
        返回分类器所需的参数名称列表。

        参数:
        factor_id (int): 分类因子的ID（1或2）。
        """
        assert factor_id == 1 or factor_id == 2
        if factor_id == 1:
            required_names = ['embed', 'encoder', 'word_encoder',
                'cl_w1', 'cl_xw1']
        else:
            required_names = ['embed', 'encoder', 'word_encoder',
            'cl_w2', 'cl_xw2']
        return required_names

    def cl_parameters(self, factor_id):
        """
        返回分类器所需的参数迭代器。

        参数:
        factor_id (int): 分类因子的ID（1或2）。
        """
        names = self.classifier_parameter_names(factor_id)

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    def rec_parameters(self):
        """
        返回识别网络所需的参数迭代器。
        """
        names = ['embed', 'encoder', 'decoder', 'word_encoder',
            'cl_xw1', 'cl_xw2', 'cl_w1', 'cl_w2',
            'factor_embed1', 'factor_embed2', 'posteriori',
            'out_proj', 'dec_init', 'context', 'map_x']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    def dis_parameters(self):
        """
        返回判别器所需的参数迭代器。
        """
        return self.layers['discriminator'].parameters()


    def gen_parameters(self):
        """
        返回生成网络所需的参数迭代器。
        """
        names = ['prior', 'posteriori', 'encoder', 'word_encoder',
            'embed']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    def compute_key_state(self, keys):
        """
        计算关键词的状态。

        参数:
        keys (Tensor): 关键词序列。

        返回:
        key_state (Tensor): 关键词编码状态。
        """
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]
        return key_state

    def compute_inferred_label(self, key_state, factor_id):
        """
        计算推断出的标签。

        参数:
        key_state (Tensor): 关键词编码状态。
        factor_id (int): 分类因子的ID（1或2）。

        返回:
        pred (Tensor): 推断出的标签。
        """
        logits = self.layers['cl_w'+str(factor_id)](key_state)
        probs = F.softmax(logits, dim=-1)
        pred = probs.max(dim=-1)[1] # (B)
        return pred

    def compute_dec_init_state(self, key_state, labels1, labels2):
        """
        计算解码器的初始状态。

        参数:
        key_state (Tensor): 关键词编码状态。
        labels1 (Tensor): 第一个因子的标签。
        labels2 (Tensor): 第二个因子的标签。

        返回:
        dec_init_state (Tensor): 解码器初始状态。
        """
        z_prior = self.layers['prior'](key_state, labels1, labels2)

        factor_emb1 = self.layers['factor_embed1'](labels1)
        factor_emb2 = self.layers['factor_embed2'](labels2)

        dec_init_state = self.layers['dec_init'](
            torch.cat([z_prior, key_state, factor_emb1, factor_emb2], dim=-1)) # (B, H-2)

        return dec_init_state


    def compute_prior(self, keys, labels1, labels2):
        """
        计算先验分布。

        参数:
        keys (Tensor): 关键词序列。
        labels1 (Tensor): 第一个因子的标签。
        labels2 (Tensor): 第二个因子的标签。

        返回:
        z_prior (Tensor): 先验分布。
        """
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]


        z_prior = self.layers['prior'](key_state, labels1, labels2)

        return z_prior