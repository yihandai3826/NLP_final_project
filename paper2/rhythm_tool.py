class RhythmRecognizer(object):
    """用于识别输入诗句的节奏ID，仅适用于中文古典绝句。"""

    def __init__(self, ping_file, ze_file):
        """初始化平声和仄声字符列表"""
        with open(ping_file, 'r', encoding='utf-8') as fin:
            self.__ping = fin.read()

        with open(ze_file, 'r', encoding='utf-8') as fin:
            self.__ze = fin.read()

    def get_rhythm(self, sentence):
        """根据诗句内容返回其节奏ID"""
        if len(sentence) == 5:
            # 五言诗节奏匹配规则
            if (sentence[0] in self.__ping and sentence[1] in self.__ping and 
                sentence[2] in self.__ping and sentence[3] in self.__ze and 
                sentence[4] in self.__ze):
                return 0
            if (sentence[0] in self.__ping and sentence[1] in self.__ping and 
                sentence[2] in self.__ze and sentence[3] in self.__ze and 
                sentence[4] in self.__ze):
                return 0
            if (sentence[0] in self.__ze and sentence[1] in self.__ping and 
                sentence[2] in self.__ping and sentence[3] in self.__ze and 
                sentence[4] in self.__ze):
                return 0
            if (sentence[0] in self.__ze and sentence[1] in self.__ping and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 0
            if (sentence[0] in self.__ping and sentence[1] in self.__ping and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 0
            if (sentence[0] in self.__ze and sentence[1] in self.__ze and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ping):
                return 1
            if (sentence[0] in self.__ping and sentence[1] in self.__ze and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ping):
                return 1
            if (sentence[0] in self.__ping and sentence[1] in self.__ze and 
                sentence[2] in self.__ping and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 3
            if (sentence[0] in self.__ping and sentence[1] in self.__ze and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 3
            if (sentence[0] in self.__ze and sentence[1] in self.__ze and 
                sentence[2] in self.__ping and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 3
            if (sentence[0] in self.__ze and sentence[1] in self.__ze and 
                sentence[2] in self.__ze and sentence[3] in self.__ping and 
                sentence[4] in self.__ze):
                return 3
            if (sentence[0] in self.__ping and sentence[1] in self.__ping and 
                sentence[2] in self.__ze and sentence[3] in self.__ze and 
                sentence[4] in self.__ping):
                return 2
            if (sentence[0] in self.__ze and sentence[1] in self.__ping and 
                sentence[2] in self.__ping and sentence[3] in self.__ze and 
                sentence[4] in self.__ping):
                return 2
            if (sentence[0] in self.__ping and sentence[1] in self.__ping and 
                sentence[2] in self.__ping and sentence[3] in self.__ze and 
                sentence[4] in self.__ping):
                return 2
        elif len(sentence) == 7:
            # 七言诗节奏匹配规则
            if (sentence[1] in self.__ze and sentence[2] in self.__ping and 
                sentence[3] in self.__ping and sentence[4] in self.__ping and 
                sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            if (sentence[1] in self.__ze and sentence[2] in self.__ping and 
                sentence[3] in self.__ping and sentence[4] in self.__ze and 
                sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            if (sentence[1] in self.__ze and sentence[2] in self.__ze and 
                sentence[3] in self.__ping and sentence[4] in self.__ping and 
                sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            if (sentence[1] in self.__ze and sentence[2] in self.__ping and 
                sentence[3] in self.__ping and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            if (sentence[1] in self.__ze and sentence[2] in self.__ze and 
                sentence[3] in self.__ping and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            if (sentence[1] in self.__ping and sentence[2] in self.__ze and 
                sentence[3] in self.__ze and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            if (sentence[1] in self.__ping and sentence[2] in self.__ping and 
                sentence[3] in self.__ze and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            if (sentence[1] in self.__ping and sentence[2] in self.__ping and 
                sentence[3] in self.__ze and sentence[4] in self.__ping and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            if (sentence[1] in self.__ping and sentence[2] in self.__ping and 
                sentence[3] in self.__ze and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            if (sentence[1] in self.__ping and sentence[2] in self.__ze and 
                sentence[3] in self.__ze and sentence[4] in self.__ping and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            if (sentence[1] in self.__ping and sentence[2] in self.__ze and 
                sentence[3] in self.__ze and sentence[4] in self.__ze and 
                sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            if (sentence[1] in self.__ze and sentence[2] in self.__ping and 
                sentence[3] in self.__ping and sentence[4] in self.__ze and 
                sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            if (sentence[1] in self.__ze and sentence[2] in self.__ze and 
                sentence[3] in self.__ping and sentence[4] in self.__ping and 
                sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            if (sentence[1] in self.__ze and sentence[2] in self.__ping and 
                sentence[3] in self.__ping and sentence[4] in self.__ping and 
                sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
        else:
            return -2  # 无效的诗句长度
        return -1  # 默认返回值，表示无匹配的节奏