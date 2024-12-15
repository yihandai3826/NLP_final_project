# -*- coding: utf-8 -*-

import os
from config import Config
import numpy as np
from model import PoetryModel
import torch

class Sample(object):
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda') if self.config.use_gpu else torch.device('cpu')

        self.processed_data_path = self.config.processed_data_path
        self.model_path = self.config.model_path
        self.max_len = self.config.max_gen_len
        self.sentence_max_len = self.config.sentence_max_len

        self.load_data()
        self.load_model()

    def load_data(self):
        if os.path.exists(self.processed_data_path):
            data = np.load(self.processed_data_path, allow_pickle=True)  # 设置 allow_pickle=True 以允许加载包含对象的数组
            self.data, self.word_to_ix, self.ix_to_word = data['data'], data['word2ix'].item(), data['ix2word'].item()

    def load_model(self):
        model = PoetryModel(len(self.word_to_ix),
                            self.config.embedding_dim,
                            self.config.hidden_dim,
                            self.device,
                            self.config.layer_num)
        map_location = lambda s, l: s
        state_dict = torch.load(self.config.model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.to(self.device)
        self.model = model

    def generate_random(self, start_words='<START>', nmf_topic_vector=None):
        """自由生成一首诗歌"""
        poetry = []
        sentence_len = 0

        input = (torch.Tensor([self.word_to_ix[start_words]]).view(1, 1).long()).to(self.device)
        hidden = self.model.init_hidden(1, self.config.layer_num)

        for i in range(self.max_len):
            # 前向计算出概率最大的当前词
            output, hidden = self.model.forward(input, hidden)
            entropy = self.model.calculate_entropy(output)
            if nmf_topic_vector is not None and torch.mean(entropy).item() > self.config.entropy_threshold:
                output = self.model.adjust_output_with_nmf(output, nmf_topic_vector)
            top_index = output[0].argmax().item()
            char = self.ix_to_word[top_index]

            # 遇到终结符则输出
            if char == '<EOP>':
                break

            # 有8个句子则停止预测
            if char in ['。', '！']:
                sentence_len += 1
                if sentence_len == 8:
                    poetry.append(char)
                    break

            input = (torch.Tensor([top_index]).view(1, 1).long()).to(self.device)
            poetry.append(char)

        return ''.join(poetry)

    def generate_head(self, head_sentence, nmf_topic_vector=None):
        """生成藏头诗"""
        poetry = []
        head_char_len = len(head_sentence)  # 要生成的句子的数量
        sentence_len = 0  # 当前句子的数量
        pre_char = '<START>'  # 前一个已经生成的字

        # 准备第一步要输入的数据
        input = (torch.Tensor([self.word_to_ix['<START>']]).view(1, 1).long()).to(self.device)
        hidden = self.model.init_hidden(1, self.config.layer_num)

        for i in range(self.max_len):
            # 前向计算出概率最大的当前词
            output, hidden = self.model.forward(input, hidden)
            entropy = self.model.calculate_entropy(output)
            if nmf_topic_vector is not None and torch.mean(entropy).item() > self.config.entropy_threshold:
                output = self.model.adjust_output_with_nmf(output, nmf_topic_vector)
            top_index = output[0].argmax().item()
            char = self.ix_to_word[top_index]

            # 句首的字用藏头字代替
            if pre_char in ['。', '！', '<START>']:
                if sentence_len == head_char_len:
                    break
                else:
                    char = head_sentence[sentence_len]
                    sentence_len += 1
                    input = (torch.Tensor([self.word_to_ix[char]]).view(1, 1).long()).to(self.device)
            else:
                input = (torch.Tensor([top_index]).view(1, 1).long()).to(self.device)

            poetry.append(char)
            pre_char = char

        return ''.join(poetry)

    def generate_poetry(self, mode=1, head_sentence=None, nmf_topic_vector=None):
        """
        模式一：随机生成诗歌
        模式二：生成藏头诗
        模式三：给定首句生成诗
        :return:
        """
        poetry = ''
        if mode == 1 or (mode == 2 and head_sentence is None):
            poetry = self.generate_random(nmf_topic_vector=nmf_topic_vector)
        elif mode == 2 and head_sentence is not None:
            head_sentence = head_sentence.replace(',', '，').replace('.', '。').replace('?', '？')
            poetry = self.generate_head(head_sentence, nmf_topic_vector=nmf_topic_vector)

        return poetry

    def generate_samples(self, num_samples, nmf_topic_vector=None):
        samples = []
        for _ in range(num_samples):
            sample = self.generate_poetry(nmf_topic_vector=nmf_topic_vector)
            samples.append(sample)
        return samples

    def save_samples(self, samples, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')

if __name__ == '__main__':
    obj = Sample()
    poetry = obj.generate_poetry(mode=2, head_sentence="七月流火")
    print(poetry)

    # 将生成的诗歌保存到本地文件
    with open("./generated_poetry.txt", "w", encoding="utf-8") as file:
        file.write(poetry)