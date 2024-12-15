import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        attn_weights = F.softmax(self.attn(encoder_outputs), dim=1)
        context_vector = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context_vector, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)

    def forward(self, input_, hidden):
        output, hidden = self.gru(input_)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, attention):
        super(Decoder, self).__init__()
        self.attention = attention
        self.gru = nn.GRU(hidden_dim + 1, hidden_dim)  # 注意这里的hidden_dim + 1
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_, hidden, encoder_outputs, nmf_topic_vector=None):
        context_vector, _ = self.attention(hidden, encoder_outputs)
        context_vector = context_vector.unsqueeze(1)  # 添加一个维度，使其成为 (batch_size, 1, hidden_dim)
        input_ = torch.cat((input_, context_vector), dim=2)  # 沿着最后一个维度拼接

        if nmf_topic_vector is not None:
            # 将NMF主题向量与输入序列拼接
            input_ = torch.cat((input_, nmf_topic_vector.unsqueeze(1).repeat(1, input_.size(1), 1)), dim=2)

        output, hidden = self.gru(input_, hidden)
        output = F.log_softmax(self.fc(output[:, -1, :]), dim=1)
        return output, hidden

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device, layer_num, nmf_model=None):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.layer_num = layer_num
        self.nmf_model = nmf_model

        # 创建embedding层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 创建Encoder
        self.encoder = Encoder(embedding_dim, hidden_dim)

        # 创建Decoder
        self.decoder = Decoder(hidden_dim, vocab_size, Attention(hidden_dim))

        # 创建一个线性层
        self.linear1 = nn.Linear(hidden_dim, vocab_size)

        # 创建一个dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs, targets, teacher_forcing_ratio=0.5, nmf_topic_vector=None):
        batch_size = inputs.size(0)
        target_len = targets.size(1)
        vocab_size = self.embeddings.weight.size(0)

        # 初始化隐藏状态
        hidden = self.init_hidden(batch_size, self.layer_num)

        # 编码器输入
        input_ = self.embeddings(inputs)

        # 编码器输出
        encoder_outputs, hidden = self.encoder(input_, hidden)

        # 解码器输入（初始为<sos> tokens）
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
        decoder_input = self.embeddings(decoder_input)

        # 初始化解码器的隐藏状态为编码器的最终隐藏状态
        decoder_hidden = hidden[-1]  # 使用最后一层的隐藏状态

        # 逐时间步解码
        for t in range(1, target_len + 1):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, nmf_topic_vector)
            decoder_output = decoder_output[:, -1, :]  # 取最后一个时间步的输出

            # 决定是否使用teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                next_input = targets[:, t-1].unsqueeze(1)
            else:
                next_input = decoder_output.argmax(1).unsqueeze(1)
            next_input = self.embeddings(next_input)
            decoder_input = next_input

        # 将最终输出通过线性层和dropout层
        output = self.linear1(decoder_output.view(-1, decoder_output.size(-1)))
        output = self.dropout(output)

        return output

    def init_hidden(self, batch_size, layer_num):
        # 返回两个3D张量，形状为 (num_layers, batch_size, hidden_size)
        return (torch.zeros(layer_num, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(layer_num, batch_size, self.hidden_dim).to(self.device))

    def calculate_entropy(self, output):
        # 计算熵
        prob = F.softmax(output, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # 避免对0取对数
        return entropy

    def adjust_output_with_nmf(self, output, nmf_topic_vector):
        # 根据NMF主题向量调整输出概率分布
        topic_weights = torch.matmul(output, nmf_topic_vector.T)
        topic_weights = F.softmax(topic_weights, dim=1)
        adjusted_output = output + topic_weights
        return adjusted_output