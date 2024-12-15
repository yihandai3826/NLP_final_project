# nmf_utils.py
import os
import json
import re
from gensim import corpora, models
import numpy as np

def parse_raw_data(data_path, category, author, constrain):
    """
    获取原数据并预处理
    :param data_path: 数据存放的路径
    :param category: 数据的类型
    :param author: 作者名称
    :param constrain: 长度限制
    :return: list
    """
    def sentence_parse(para):
        """对文本进行处理，取出脏数据"""
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s
        r, number = re.subn("。。", "。", r)
        return r

    def handle_json(file):
        """读入json文件，返回诗句list，每一个元素为一首诗歌(str类型表示)"""
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if author is not None and poetry.get("author") != author:
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue

            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentence_parse(pdata)
            if pdata != "" and len(pdata) > 1:
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(data_path):
        if filename.startswith(category):
            data.extend(handle_json(data_path + filename))
    return data

def perform_nmf(documents, num_topics=10, max_features=1000):
    """
    对文档集合进行NMF处理。
    参数:
    documents (list of str): 文档集合，每个元素是一个文档的文本。
    num_topics (int): 要提取的主题数量。
    max_features (int): 词汇表中的最大特征数。
    返回:
    model (gensim.models.Nmf): NMF模型。
    """
    # 文本预处理和向量化
    texts = [[word for word in document.split()] for document in documents]
    dictionary = corpora.Dictionary(texts)  # 构建字典
    corpus = [dictionary.doc2bow(text) for text in texts]  # 将文本转换为词袋模型

    # 应用NMF
    model = models.Nmf(corpus, num_topics=num_topics, id2word=dictionary, passes=100)
    return model

def get_topic_vector(model, topic_id, max_features=1000):
    """
    获取指定主题的词汇分布向量。
    参数:
    model (gensim.models.Nmf): 训练好的NMF模型。
    topic_id (int): 主题ID。
    max_features (int): 返回的词汇分布向量的最大特征数。
    返回:
    topic_vector (numpy.ndarray): 主题的词汇分布向量。
    """
    # 获取主题-词汇分布矩阵
    W = model.wm
    # 获取指定主题的词汇分布向量
    topic_vector = W[topic_id]
    # 将向量按值降序排列，并取前max_features个特征
    topic_vector = topic_vector.argsort()[-1:-max_features-1:-1]
    return topic_vector

# 测试代码
if __name__ == '__main__':
    from config import Config
    config = Config()
    data_path = config.data_path
    category = config.category
    author = config.author
    constrain = config.constrain

    # 获取数据
    data = parse_raw_data(data_path, category, author, constrain)
    
    # 执行NMF
    model = perform_nmf(data, num_topics=config.nmf_num_topics, max_features=config.nmf_max_features)

    # 获取指定主题的词汇分布向量
    topic_id = 0  # 假设我们关注的是第一个主题
    topic_vector = get_topic_vector(model, topic_id, max_features=config.nmf_max_features)
    print(topic_vector)