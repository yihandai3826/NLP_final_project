import json
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

import re
import json


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
以下是一首中文古典诗歌，你需要检查每个句号（即“。”）前的那个字（称为句尾字）的韵脚是否相同。如果韵脚相同，返回“YES”；如果韵脚不同，则对句尾字进行替换，使得所有句尾字具有相同的韵脚，同时尽量保持诗句的语义连贯。返回修改后的诗歌以及“NO”。在必要时，你可以略微调整其他字词，但要优先保持原文的语义。
注意，你只需要考虑句号前的每个字，而不是逗号前。比如对于“春眠不觉晓，处处闻啼鸟。”，你需要考虑的句尾字只有句号前的“鸟”。
以下是几个例子，供你参考：

**示例 1：**
**输入：**
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
**输出：**
YES

**示例 2：**
**输入：**
山重水复疑无路，柳暗花明又一村。千里莺啼绿映红，春风十里不胜寒。
**输出：**
NO
修改后：
山重水复疑无路，柳暗花明又一村。千里莺啼绿映红，春风十里不胜云。

**示例 3：**
**输入：**
人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。
**输出：**
NO
修改后：
人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。


**输入：**
现在，请根据以下诗句完成任务：
{poem}
**输出：**
（请按照上述格式返回结果，不需要回答其他内容）
'''


def generate(input_p, out):
    qwen_model = QwenModel(name="qwen-max", temperature=0.7)
    dicts = []
    with open(input_p, 'r', encoding='utf-8') as poem_f:
        # 按行读取文本文件
        poem_data = [line.strip() for line in poem_f if line.strip()]  # 去除空行和两端空白

    for idx, item in enumerate(poem_data):
        print(idx)
        if "<|extra_1|>" in item:
            continue
        try:
            prompt = poetry_prompt_template.format_map(
                {
                    "poem": item,
                }
            )
            response = qwen_model._query(prompt).strip()
            # print(response)
            pattern = re.compile(
                r"(YES|NO)(?:\n修改后：\n([\s\S]+))?"
            )
            matches = list(pattern.finditer(response))
            if not matches:
                print("未找到匹配内容")
                result = response
                modified_poem = ""
            else:
                match = matches[-1]
                result = match.group(1)  # YES 或 NO
                modified_poem = match.group(2)  # 修改后的诗歌内容
            output_data = {
                "result": result,
                "modified_poem": modified_poem.strip() if modified_poem else response
            }
            dicts.append(output_data)
        except Exception as e:
            print(f"调用模型时出错：{e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dicts, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    # 调用函数
    input_file = "poem_few.txt"  # 输入文件路径
    output_file = "change.json"
    generate(input_file, output_file)
