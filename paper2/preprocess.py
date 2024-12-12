import pickle
import json
import random
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="The parameters for the preprocessing tool.")
    parser.add_argument("-n", "--num_unlabelled", type=int, default=-1, help="The number of unlabelled data to be used.")
    return parser.parse_args()

# 将数据写入文件
def outFile(data, file_name):
    print("output data to %s, num: %d" % (file_name, len(data)))
    with open(file_name, 'w', encoding='utf-8') as fout:  # 指定编码格式
        for d in data:
            fout.write(d+"\n")

class PreProcess(object):
    """文本预处理类"""
    def __init__(self, min_freq=1):
        super(PreProcess, self).__init__()
        self.min_freq = min_freq

    def line2idxes(self, line):
        """将文本行转换为索引列表"""
        chars = [c for c in line]
        idxes = []
        for c in chars:
            if c in self.dic:
                idx = self.dic[c]
            else:
                idx = self.dic['UNK']
            idxes.append(idx)

        return idxes

    def create_dic(self, poems):
        """创建词汇字典"""
        print ("creating the word dictionary...")
        print ("input poems: %d" % (len(poems)))
        count_dic = {}
        for p in poems:
            poem = p.strip().replace("|", "")

            for c in poem:
                if c in count_dic:
                    count_dic[c] += 1
                else:
                    count_dic[c] = 1

        vec = sorted(count_dic.items(), key=lambda d:d[1], reverse=True)
        print ("original word num:%d" % (len(vec)))

        # 添加特殊符号
        dic = {}
        idic = {}
        dic['PAD'] = 0
        idic[0] = 'PAD'

        dic['UNK'] = 1
        idic[1] = 'UNK'

        dic['<E>'] = 2
        idic[2] = '<E>'

        dic['<B>'] = 3
        idic[3] = '<B>'

        idx = 4
        print ("min freq:%d" % (self.min_freq))

        for c, v in vec:
            if v < self.min_freq:
                continue
            if not c in dic:
                dic[c] = idx
                idic[idx] = c
                idx += 1

        print ("total word num: %s" % (len(dic)))

        return dic, idic

    def build_dic(self, infile):
        """从文件构建词汇字典"""
        with open(infile, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        poems = []
        training_lines = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            poems.append(poem)
            training_lines.extend(poem.split("|"))

        dic, idic = self.create_dic(poems)
        self.dic = dic
        self.idic = idic

        # 输出字典文件
        dic_file = "vocab.pickle"
        idic_file = "ivocab.pickle"

        print ("saving dictionary to %s" % (dic_file))
        with open(dic_file, 'wb') as fout:
            pickle.dump(dic, fout, -1)

        print ("saving inverting dictionary to %s" % (idic_file))
        with open(idic_file, 'wb') as fout:
            pickle.dump(idic, fout, -1)

        # 构建训练行
        outFile(training_lines, "training_lines.txt")

    def read_corpus(self, infile, with_label=False):
        """读取语料库文件"""
        with open(infile, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        corpus = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            keywords = dic['keywords'].split(" ")

            if with_label:
                para = dic['label'].split(" ")
                labels = (int(para[0]), int(para[1]))
            else:
                labels = (-1, -1)

            for keyword in keywords:
                corpus.append((keyword, poem, labels))

        return corpus

    def build_data(self, corpus):
        """构建数据集"""
        max_len = -1
        skip_plen_count = 0
        skip_llen_count = 0

        data = []
        for d in corpus:
            keyword = d[0]
            poem = d[1]

            lines = poem.split("|")

            if len(lines) != 4:
                skip_plen_count += 1
                continue

            skip = False
            line_idxes_vec = []
            for line in lines:
                idxes = self.line2idxes(line)
                length = len(idxes)
                if length != 5 and length != 7:
                    skip = True
                    break

                max_len = max(max_len, length)
                line_idxes_vec.append(idxes)

            if skip:
                skip_llen_count += 1
                continue

            assert len(line_idxes_vec) == 4

            assert len(keyword) <= 2
            key_idxes = self.line2idxes(keyword)

            data.append((key_idxes, line_idxes_vec, d[2][0], d[2][1]))

        print ("data num: %d, skip plen: %d, skip llen: %d, max length: %d" %\
            (len(data), skip_plen_count, skip_llen_count, max_len))

        return data

    def build_test_data(self, infile, out_inp_file, out_trg_file):
        """构建测试数据集"""
        with open(infile, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        keywords_vec = []
        poems_vec = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            keywords = dic['keywords'].split(" ")

            keywords_vec.append(random.sample(keywords, 1)[0])
            poems_vec.append(poem)

        outFile(keywords_vec, out_inp_file)
        outFile(poems_vec, out_trg_file)

    def process(self, unlabelled_num=None):
        """处理数据"""
        self.build_dic("ccpc_train_v1.0.json")

        unlabelled_train = self.read_corpus("ccpc_train_v1.0.json", False)
        unlabelled_valid = self.read_corpus("ccpc_valid_v1.0.json", False)

        if unlabelled_num is not None:
            unlabelled_train = random.sample(unlabelled_train, unlabelled_num)

        labelled_train = self.read_corpus("cqcf_train_sample.json", True)
        labelled_valid = self.read_corpus("cqcf_valid_sample.json", True)

        semi_train_data = self.build_data(unlabelled_train+labelled_train)
        semi_valid_data = self.build_data(unlabelled_valid+labelled_valid)

        random.shuffle(semi_train_data)
        random.shuffle(semi_valid_data)

        print ("training data: %d" % (len(semi_train_data)))
        print ("validation data: %d" % (len(semi_valid_data)))

        train_file = "semi_train.pickle"
        print ("saving training data to %s" % (train_file))
        with open(train_file, 'wb') as fout:
            pickle.dump(semi_train_data, fout, -1)

        valid_file = "semi_valid.pickle"
        print ("saving validation data to %s" % (valid_file))
        with open(valid_file, 'wb') as fout:
            pickle.dump(semi_valid_data, fout, -1)

        self.build_test_data("ccpc_test_v1.0.json", "test_inps.txt", "test_trgs.txt")

def main():
    """主函数"""
    args = parse_args()
    if args.num_unlabelled == -1:
        n = None
    else:
        n = args.num_unlabelled

    processor = PreProcess()
    processor.process(n)

if __name__ == "__main__":
    main()