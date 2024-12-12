import pickle
import numpy as np
import random
import copy
import torch

# 定义函数 readPickle，用于从指定路径读取 pickle 文件，并返回文件内容。
def readPickle(data_path):
    corpus_file = open(data_path, 'rb')    
    # 使用 pickle 模块的 load 函数从文件中加载数据。
    corpus = pickle.load(corpus_file)    
    corpus_file.close()
    return corpus

class Tool(object):

    # Tool 类的构造函数，用于初始化工具类的基本属性。
    def __init__(self, sens_num, key_len, sen_len, poem_len, corrupt_ratio=0):
        # 记录每个诗歌中句子的数量。
        self.sens_num = sens_num

        # 记录每个关键词的最大长度，通常为2个中文字符。
        self.key_len = key_len

        # 记录每个句子的最大长度，通常为9个中文字符。
        self.sen_len = sen_len

        # 记录整个诗歌的最大长度，通常为30个中文字符。
        self.poem_len = poem_len

        # 去噪自编码器（DAE）的数据损坏比例。
        self.corrupt_ratio = corrupt_ratio

        # 初始化词汇表和逆词汇表为None，将在后续加载词汇表时赋值。
        self.__vocab = None
        self.__ivocab = None

        # 初始化特殊符号的索引为None，这些索引将在加载词汇表后被赋值。
        # PAD_ID: 用于填充序列的索引。
        # B_ID: 句子开始的标识符。
        # E_ID: 句子结束的标识符。
        # UNK_ID: 未知词的标识符。
        self.__PAD_ID = None
        self.__B_ID = None
        self.__E_ID = None
        self.__UNK_ID = None

    # -----------------------------------
    # map functions
    # 将索引列表转换为诗歌行。
    # idxes: 包含词汇索引的列表。
    # truncate: 是否截断索引列表，如果为True且列表中包含结束索引（self.__E_ID），则截断。
    def idxes2line(self, idxes, truncate=True):
        # 如果需要截断，并且索引列表中包含结束索引，则截断列表。
        if truncate and self.__E_ID in idxes:
            idxes = idxes[:idxes.index(self.__E_ID)]

        # 将索引列表转换为标记列表。
        tokens = self.idxes2tokens(idxes, truncate)
        # 将标记列表转换为字符串形式的诗歌行。
        line = self.tokens2line(tokens)
        return line

    # 将诗歌行转换为索引列表。
    # line: 字符串形式的诗歌行。
    def line2idxes(self, line):
        # 将诗歌行转换为标记列表。
        tokens = self.line2tokens(line)
        # 将标记列表转换为索引列表。
        return self.tokens2idxes(tokens)

    # 将诗歌行转换为标记列表。
    # line: 字符串形式的诗歌行。
    def line2tokens(self, line):
        '''
        在这项工作中，我们将每个中文字符视为一个标记。
        '''
        # 去除诗歌行两端的空白字符。
        line = line.strip()
        # 将诗歌行拆分为单个字符的列表。
        tokens = [c for c in line]
        return tokens

    # 将标记列表转换为诗歌行。
    # tokens: 包含诗歌中字符的列表。
    def tokens2line(self, tokens):
        # 将标记列表连接成一个字符串，形成诗歌行。
        return "".join(tokens)

    # 将标记列表转换为索引列表。
    # tokens: 包含诗歌中字符的列表。
    def tokens2idxes(self, tokens):
        '''
        将字符列表转换为索引列表。
        '''
        # 初始化索引列表。
        idxes = []
        # 遍历每个字符。
        for w in tokens:
            # 如果字符在词汇表中，添加其索引；否则，添加未知字符的索引。
            if w in self.__vocab:
                idxes.append(self.__vocab[w])
            else:
                idxes.append(self.__UNK_ID)
        return idxes

    # 将索引列表转换为标记列表。
    # idxes: 包含词汇索引的列表。
    # omit_special: 是否省略特殊标记，如填充、开始和结束标记。
    def idxes2tokens(self, idxes, omit_special=True):
        # 初始化标记列表。
        tokens = []
        # 遍历每个索引。
        for idx in idxes:
            # 如果索引是特殊标记，并且omit_special为True，则跳过。
            if (idx == self.__PAD_ID or idx == self.__B_ID or idx == self.__E_ID) and omit_special:
                continue
            # 添加索引对应的字符到标记列表。
            tokens.append(self.__ivocab[idx])
        return tokens

        # -------------------------------------------------def greedy_search(self, probs):
    def greedy_search(self, probs):
        """
        贪婪搜索，从概率分布中选取最可能的标记序列。
        
        :param probs: 概率分布数组，形状为(B, L, V)，分别代表批次、序列长度和词汇表大小。
        :return: 转换后的文本行。
        """
        # 选取概率最大的标记索引
        outidx = [int(np.argmax(prob, axis=-1)) for prob in probs]
        
        # 遇到序列结束标记则截断
        if self.__E_ID in outidx:
            outidx = outidx[:outidx.index(self.__E_ID)]

        # 索引转标记
        tokens = self.idxes2tokens(outidx)
        
        # 标记转文本行
        return self.tokens2line(tokens)

    # ----------------------------
    def get_vocab(self):
        """
        获取词汇表的深拷贝。
        """
        return copy.deepcopy(self.__vocab)

    def get_ivocab(self):
        """
        获取逆词汇表的深拷贝。
        """
        return copy.deepcopy(self.__ivocab)

    def get_vocab_size(self):
        """
        获取词汇表大小。
        """
        if self.__vocab:
            return len(self.__vocab)
        else:
            return -1

    def get_PAD_ID(self):
        """
        获取填充标记ID。
        """
        assert self.__PAD_ID is not None
        return self.__PAD_ID

    def get_B_ID(self):
        """
        获取批次开始标记ID。
        """
        assert self.__B_ID is not None
        return self.__B_ID

    def get_E_ID(self):
        """
        获取序列结束标记ID。
        """
        assert self.__E_ID is not None
        return self.__E_ID

    def get_UNK_ID(self):
        """
        获取未知标记ID。
        """
        assert self.__UNK_ID is not None
        return self.__UNK_ID

    # ----------------------------------------------------------------
    def load_dic(self, vocab_path, ivocab_path):
        dic = readPickle(vocab_path)
        idic = readPickle(ivocab_path)

        assert len(dic) == len(idic)

        self.__vocab = dic
        self.__ivocab = idic

        self.__PAD_ID = dic['PAD']
        self.__UNK_ID = dic['UNK']
        self.__E_ID = dic['<E>']
        self.__B_ID = dic['<B>']


    def build_data(self, train_data_path, valid_data_path, batch_size, mode):
        """
        构建训练和验证数据批次。
        
        :param train_data_path: 训练数据路径。
        :param valid_data_path: 验证数据路径。
        :param batch_size: 批次大小。
        :param mode: 模式，包括分类预训练、去噪自编码器预训练和MixPoet训练。
        """
        assert mode in ['cl1', 'cl2', 'dae', 'mixpoet_pre', 'mixpoet_tune']
        train_data = readPickle(train_data_path)
        valid_data = readPickle(valid_data_path)

        print (len(train_data))
        print (len(valid_data))

        self.train_batches = self.__build_data_core(train_data, batch_size, mode)
        self.valid_batches = self.__build_data_core(valid_data, batch_size, mode)

        self.train_batch_num = len(self.train_batches)
        self.valid_batch_num = len(self.valid_batches)

    def __build_data_core(self, data, batch_size, mode, data_limit=None):
        """
        核心数据构建函数。
        
        :param data: 数据列表。
        :param batch_size: 批次大小。
        :param mode: 模式。
        :param data_limit: 调试时的数据限制。
        """
        if data_limit is not None:
            data = data[0:data_limit]

        if mode == 'cl1' or mode == 'cl2':
            return self.build_classifier_batches(data, batch_size, mode)
        elif mode == 'dae':
            return self.build_dae_batches(data, batch_size)
        elif mode == 'mixpoet_pre' or mode == 'mixpoet_tune':
            return self.build_mixpoet_batches(data, batch_size, mode)

    def build_classifier_batches(self, ori_data, batch_size, mode):
        """
        为分类任务构建批次数据。
        
        :param ori_data: 原始数据。
        :param batch_size: 批次大小。
        :param mode: 模式。
        """
        data = []
        for instance in ori_data:
            if mode == 'cl1':
                label = instance[2]
            elif mode == 'cl2':
                label = instance[3]
            if label == -1:
                continue
            assert label >= 0
            data.append((instance[0], instance[1], label))

        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            poems = [instance[1] for instance in instances]
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True)

            keys = [instance[0] for instance in instances]
            batch_keys = self.__build_batch_seqs(keys, True, True)

            labels = [instance[2] for instance in instances]
            batch_labels = torch.tensor(labels, dtype=torch.long)

            batched_data.append((batch_keys, batch_poems, batch_labels))

        return batched_data

    def build_dae_batches(self, data, batch_size):
        """
        为去噪自编码器任务构建批次数据。
        
        :param data: 数据。
        :param batch_size: 批次大小。
        """
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            poems = [instance[1] for instance in instances]
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True, corrupt=True)

            keys = [instance[0] for instance in instances]
            batch_keys = self.__build_batch_seqs(keys, True, True)

            batch_dec_inps = []
            for step in range(0, self.sens_num):
                lines = [poem[step] for poem in poems]
                batch_lengths = self.__build_batch_length(lines)
                batch_lines = self.__build_batch_seqs(lines, False, True)
                batch_dec_inps.append(batch_lines)

            batched_data.append((batch_keys, batch_poems, batch_dec_inps, batch_lengths))

        return batched_data

    def build_mixpoet_batches(self, ori_data, batch_size, mode):
        """
        为MixPoet任务构建批次数据。
        
        :param ori_data: 原始数据。
        :param batch_size: 批次大小。
        :param mode: 模式。
        """
        if mode == 'mixpoet_pre':
            data = ori_data
        else:
            data = []
            for instance in ori_data:
                label1 = instance[2]
                label2 = instance[3]
                if label1 == -1 and label2 == -1:
                    continue
                data.append(instance)

        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            poems = [instance[1] for instance in instances]
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True)

            keys = [instance[0] for instance in instances]
            batch_keys = self.__build_batch_seqs(keys, True, True)

            batch_dec_inps = []
            for step in range(0, self.sens_num):
                lines = [poem[step] for poem in poems]
                batch_lengths = self.__build_batch_length(lines)
                batch_lines = self.__build_batch_seqs(lines, False, True)
                batch_dec_inps.append(batch_lines)

            labels = [(instance[2], instance[3]) for instance in instances]
            label_mask = [(float(pair[0] != -1), float(pair[1] != -1)) for pair in labels]

            batch_labels = torch.tensor(labels, dtype=torch.long)
            batch_label_mask = torch.tensor(label_mask, dtype=torch.float)

            batched_data.append((batch_keys, batch_poems, batch_dec_inps,
                batch_labels, batch_label_mask, batch_lengths))

        return batched_data


    def __build_batch_length(self, lines):
        """
        构建批次行的长度信息。
        
        :param lines: 行的列表。
        :return: 长度信息的张量。
        """
        lengths = []
        for line in lines:
            yan = len(line)
            assert yan == 5 or yan == 7
            if yan == 5:
                lengths.append([0.0, 1.0])
            else:
                lengths.append([1.0, 0.0])

        batch_lengths = torch.tensor(lengths, dtype=torch.float)
        return batch_lengths

    def __build_batch_seqs(self, instances, with_B, with_E, corrupt=False):
        """
        将序列打包成张量。
        
        :param instances: 序列实例。
        :param with_B: 是否包含开始标记。
        :param with_E: 是否包含结束标记。
        :param corrupt: 是否需要对序列进行损坏处理。
        :return: 序列的张量。
        """
        seqs = self.__get_batch_seq(instances, with_B, with_E, corrupt)
        seqs_tensor = self.__sens2tensor(seqs)
        return seqs_tensor

    def __get_batch_seq(self, seqs, with_B, with_E, corrupt):
        """
        获取批次序列。
        
        :param seqs: 序列列表。
        :param with_B: 是否包含开始标记。
        :param with_E: 是否包含结束标记。
        :param corrupt: 是否需要对序列进行损坏处理。
        :return: 批次序列。
        """
        batch_size = len(seqs)
        max_len = max([len(seq) for seq in seqs]) + int(with_B) + int(with_E)

        batched_seqs = []
        for i in range(batch_size):
            ori_seq = copy.deepcopy(seqs[i])

            if corrupt:
                seq = self.__do_corruption(ori_seq)
            else:
                seq = ori_seq

            pad_size = max_len - len(seq) - int(with_B) - int(with_E)
            pads = [self.__PAD_ID] * pad_size

            new_seq = [self.__B_ID] * int(with_B) + seq + [self.__E_ID] * int(with_E) + pads

            batched_seqs.append(new_seq)

        return batched_seqs

    def __sens2tensor(self, sens):
        """
        将句子转换为张量。
        
        :param sens: 句子列表。
        :return: 句子的张量。
        """
        batch_size = len(sens)
        sen_len = max([len(sen) for sen in sens])
        tensor = torch.zeros(batch_size, sen_len, dtype=torch.long)
        for i, sen in enumerate(sens):
            for j, token in enumerate(sen):
                tensor[i][j] = token
        return tensor

    def __do_corruption(self, inp):
        """
        对输入序列进行损坏处理，将部分标记设置为未知标记。
        
        :param inp: 输入序列。
        :return: 损坏后的序列。
        """
        m = int(np.ceil(len(inp) * self.corrupt_ratio))
        m = min(m, len(inp))

        unk_id = self.get_UNK_ID()

        corrupted_inp = copy.deepcopy(inp)
        pos = random.sample(list(range(0, len(inp))), m)
        for p in pos:
            corrupted_inp[p] = unk_id

        return corrupted_inp

    def shuffle_train_data(self):
        """
        随机打乱训练数据批次。
        """
        random.shuffle(self.train_batches)

    def keys2tensor(self, keys):
        """
        将关键词转换为张量。
        
        :param keys: 关键词列表。
        :return: 关键词索引的张量。
        """
        key_idxes = []
        for key in keys:
            tokens = self.line2tokens(key)
            idxes = self.tokens2idxes(tokens)
            key_idxes.append([self.__B_ID] + idxes + [self.__E_ID])
        return self.__sens2tensor(key_idxes)

    def lengths2tensor(self, lengths):
        """
        将长度信息转换为张量。
        
        :param lengths: 长度列表。
        :return: 长度信息的张量。
        """
        vec = []
        for length in lengths:
            assert length == 5 or length == 7
            if length == 5:
                vec.append([0.0, 1.0])
            else:
                vec.append([1.0, 0.0])

        batch_lengths = torch.tensor(vec, dtype=torch.float)
        return batch_lengths

    def pos2tensor(self, step):
        """
        将步骤位置转换为张量。
        
        :param step: 当前步骤位置。
        :return: 步骤位置的张量。
        """
        assert step in list(range(0, self.sens_num))
        pos = [0.0] * self.sens_num
        pos[step] = 1.0

        pos_tensor = torch.tensor([pos], dtype=torch.float)
        return pos_tensor