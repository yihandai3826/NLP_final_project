from collections import namedtuple
import torch

# 定义一个名为HParams的具名元组，用于存储模型和训练的超参数。
HParams = namedtuple('HParams',
    'vocab_size, pad_idx, bos_idx,'  # 词汇表大小、填充索引、句子开始索引
    'emb_size, hidden_size, context_size, latent_size, factor_emb_size,'  # 嵌入层大小、隐藏层大小、上下文大小、潜在空间大小、因素嵌入层大小
    'n_class1, n_class2, key_len, sens_num, sen_len, poem_len,'  # 分类任务的类别数、关键词长度、句子数、每句长度、诗歌长度
    'batch_size, drop_ratio, weight_decay, clip_grad_norm,'  # 批量大小、dropout比例、权重衰减、梯度裁剪

    'max_lr, min_lr, warmup_steps, ndis,'  # 最大学习率、最小学习率、预热步数、判别器训练次数
    'min_tr, burn_down_tr, decay_tr,'  # 最小教学比例、教学比例衰减、衰减周期
    'tau_annealing_steps, min_tau,'  # 温度退火步数、最小温度
    'rec_warm_steps,noise_decay_steps,'  # 重建预热步数、噪声衰减步数

    'log_steps, sample_num, max_epoches,'  # 日志记录步数、样本数量、最大周期数
    'save_epoches, validate_epoches,'  # 保存周期、验证周期
    'fbatch_size, fmax_epoches, fsave_epoches,'  # 微调批量大小、微调最大周期数、微调保存周期
    'vocab_path, ivocab_path, train_data, valid_data,'  # 词汇表路径、逆词汇表路径、训练数据路径、验证数据路径
    'model_dir, data_dir, train_log_path, valid_log_path,'  # 模型保存目录、数据目录、训练日志路径、验证日志路径
    'fig_log_path,'  # 图形日志路径

    'corrupt_ratio, dae_epoches, dae_batch_size,'  # 数据损坏比例、去噪自编码器周期数、去噪自编码器批量大小
    'dae_max_lr, dae_min_lr, dae_warmup_steps,'  # 去噪自编码器最大学习率、最小学习率、预热步数
    'dae_min_tr, dae_burn_down_tr, dae_decay_tr,'  # 去噪自编码器最小教学比例、教学比例衰减、衰减周期

    'dae_log_steps, dae_validate_epoches, dae_save_epoches,'  # 去噪自编码器日志记录步数、验证周期、保存周期
    'dae_train_log_path, dae_valid_log_path,'  # 去噪自编码器训练日志路径、验证日志路径

    'cl_batch_size, cl_epoches,'  # 分类器批量大小、周期数
    'cl_max_lr, cl_min_lr, cl_warmup_steps,'  # 分类器最大学习率、最小学习率、预热步数
    'cl_log_steps, cl_validate_epoches,'  # 分类器日志记录步数、验证周期
    'cl_save_epoches, cl_train_log_path,'  # 分类器保存周期、训练日志路径
    'cl_valid_log_path'  # 分类器验证日志路径
)

# 实例化HParams具名元组，设置模型和训练的超参数。
hparams = HParams(
    # 模型和数据的基本参数
    vocab_size=-1,  # 词汇表大小，初始值设为-1，实际使用时需替换为加载的词汇表大小
    pad_idx=-1,  # 填充索引，用于序列填充，初始值设为-1，实际使用时需替换
    bos_idx=-1,  # 句子开始索引，用于标记句子的开始，初始值设为-1，实际使用时需替换

    emb_size=256,  # 词嵌入层的维度
    hidden_size=512,  # 隐藏层的维度
    context_size=512,  # 上下文向量的维度
    latent_size=256,  # 潜在空间的维度，用于变分自编码器
    factor_emb_size=64,  # 风格因素嵌入层的维度

    # 诗歌分类任务的参数
    n_class1=3,  # 第一个风格因素的类别数（生活经历）
    n_class2=2,  # 第二个风格因素的类别数（历史背景）

    # 诗歌结构的参数
    key_len=4,  # 关键词长度，每个关键词最多包含2个中文字符
    sens_num=4,  # 句子数，每首诗由4句组成
    sen_len=9,  # 每句的长度
    poem_len=30,  # 整首诗的长度

    # 训练的正则化和梯度参数
    drop_ratio=0.15,  # Dropout比例
    weight_decay=2.5e-4,  # 权重衰减，用于L2正则化
    clip_grad_norm=2.0,  # 梯度裁剪阈值

    # 数据文件路径
    vocab_path="../corpus/vocab.pickle",  # 词汇表文件路径
    ivocab_path="../corpus/ivocab.pickle",  # 逆词汇表文件路径
    train_data="../corpus/semi_train.pickle",  # 训练数据文件路径
    valid_data="../corpus/semi_valid.pickle",  # 验证数据文件路径
    model_dir="../checkpoint/",  # 模型保存目录
    data_dir="../data/",  # 数据目录

    # 混合模型训练参数
    batch_size=128,  # 训练批量大小
    ndis=3,  # 判别器训练次数
    max_lr=8e-4,  # 最大学习率
    min_lr=5e-8,  # 最小学习率
    warmup_steps=6000,  # 学习率预热步数
    min_tr=0.85,  # 最小教学比例
    burn_down_tr=3,  # 教学比例衰减周期
    decay_tr=6,  # 教学比例衰减周期
    tau_annealing_steps=6000,  # 温度退火步数
    min_tau=0.01,  # 最小Gumbel温度
    rec_warm_steps=1500,  # 重建预热步数
    noise_decay_steps=8500,  # 噪声衰减步数

    log_steps=200,  # 日志记录步数
    sample_num=1,  # 样本数量
    max_epoches=12,  # 最大训练周期数
    save_epoches=3,  # 模型保存周期
    validate_epoches=1,  # 验证周期

    # 微调参数
    fbatch_size=64,  # 微调批量大小
    fmax_epoches=3,  # 微调最大周期数
    fsave_epoches=1,  # 微调保存周期

    train_log_path="../log/mix_train_log.txt",  # 训练日志路径
    valid_log_path="../log/mix_valid_log.txt",  # 验证日志路径
    fig_log_path="../log/",  # 图形日志路径

    # 去噪自编码器预训练参数
    dae_batch_size=128,  # 去噪自编码器批量大小
    corrupt_ratio=0.1,  # 数据损坏比例
    dae_max_lr=8e-4,  # 去噪自编码器最大学习率
    dae_min_lr=5e-8,  # 去噪自编码器最小学习率
    dae_warmup_steps=4500,  # 去噪自编码器学习率预热步数
    dae_min_tr=0.85,  # 去噪自编码器最小教学比例
    dae_burn_down_tr=2,  # 去噪自编码器教学比例衰减周期
    dae_decay_tr=6,  # 去噪自编码器教学比例衰减周期
    dae_epoches=10,  # 去噪自编码器训练周期数

    dae_log_steps=300,  # 去噪自编码器日志记录步数
    dae_validate_epoches=1,  # 去噪自编码器验证周期
    dae_save_epoches=2,  # 去噪自编码器保存周期
    dae_train_log_path="../log/dae_train_log.txt",  # 去噪自编码器训练日志路径
    dae_valid_log_path="../log/dae_valid_log.txt",  # 去噪自编码器验证日志路径

    # 分类器预训练参数
    cl_batch_size=64,  # 分类器批量大小
    cl_max_lr=8e-4,  # 分类器最大学习率
    cl_min_lr=5e-8,  # 分类器最小学习率
    cl_warmup_steps=800,  # 分类器学习率预热步数
    cl_epoches=10,  # 分类器训练周期数

    cl_log_steps=100,  # 分类器日志记录步数
    cl_validate_epoches=1,  # 分类器验证周期
    cl_save_epoches=2,  # 分类器保存周期
    cl_train_log_path="../log/cl_train_log.txt",  # 分类器训练日志路径
    cl_valid_log_path="../log/cl_valid_log.txt",  # 分类器验证日志路径
)

# 设置运行设备，优先使用GPU，如果没有GPU，则使用CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")