import json
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class QwenModel:
    def __init__(self, name, temperature, log_directory=None):
        self.log_directory = log_directory
        if log_directory:
            self.log_counter = 0
        self.name = name
        self.temperature = temperature

        self.client = OpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _query(self, user_prompt):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content


poetry_prompt_template = '''
请你从以下五个角度对我提供给你的中国古典诗歌打分：
1.	流畅性：诗歌是否符合语法、结构和音韵规则？
2.	意义：诗歌是否传达了某种明确的信息？
3.	连贯性：整首诗在意义和主题上是否连贯一致？
4.	相关性：诗歌是否很好地表达了用户的主题？
5.	美感：诗歌是否具有诗意和艺术美感？
每个角度的满分都是20，你只需要回答一个含有五个元素的列表(不用回答其他的文字)，代表这五个方面的分数。

分数的格式一定要是列表！！！比如[15,16,18,12,10]

主题：{theme}
诗歌：{poem}
'''


def generate(input_p, out):
    qwen_model = QwenModel(name="qwq-32b-preview", temperature=0.7)
    dicts = []
    ans = []
    with open(input_p, 'r', encoding='utf-8') as poem_f:
        # 按行读取文本文件
        poem_data = [line.strip() for line in poem_f if line.strip()]  # 去除空行和两端空白

    count = 0

    for idx, item in enumerate(poem_data):

        result_list = []
        score = ""
        if "<|extra_1|>" not in item:
            try:
                prompt = poetry_prompt_template.format_map(
                    {
                        "poem": item,
                        "theme": "离愁别绪"
                    }
                )
                score = qwen_model._query(prompt).strip()
                match = re.search(r'\[([\d\s,]+)\]', score)
                if match:
                    # 将匹配的内容转化为列表
                    list_str = match.group(1)  # 获取匹配的内容，不包括外部的 []
                    result_list = [int(x.strip()) for x in list_str.split(',')]  # 转化为整数列表
            except Exception as e:
                print(f"调用模型时出错：{e}")
        else:
            result_list = [0, 0, 0, 0, 0]
        print(score)
        print(f"处理完成: {idx + 1}/{len(poem_data)}")
        print(result_list)
        dict1 = {
            "poem": item,
            "score": str(result_list),
            "total": sum(result_list),
            "exp": score if len(result_list) == 0 else None
        }
        dicts.append(dict1)

    with open(out, 'w', encoding='utf-8') as outf:
        json.dump(dicts, outf, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 调用函数
    input_file = "poem.txt"  # 输入文件路径
    output_file = "score.json"
    generate(input_file, output_file)
