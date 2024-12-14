import json
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
            api_key="sk-9488a84020dc42d5a194d066ef43972b",
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

masked_poem_dict = {
"五言绝句": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"七言绝句": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"五言律诗": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"七言律诗": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"减字木兰花": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"满江红": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。",
"水调歌头": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"蝶恋花": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"如梦令": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"清平乐": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"沁园春": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
"菩萨蛮": "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。\n<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。"
}

poetry_prompt_template = '''<|im_start|>user
Fill in all the <|extra_1|>.
{user_prompt}
诗词: {masked_poem}<|im_end|>
<|im_start|>assistant
'''


def summarize_and_generate_instruction(input_p, output):
    # 初始化 QwenModel
    qwen_model = QwenModel(name="qwen-plus", temperature=0.7)

    with open(input_p, 'r', encoding='utf-8') as poem_f:
        poem_data = json.load(poem_f)

    # 打开输出文件以附加模式写入
    with open(output, 'w', encoding='utf-8') as poem_f:
        poem_f.write('[\n')  # JSON数组起始

        for idx, item in enumerate(poem_data):
            poem = item["poem"]
            rhythmic = item["rhythmic"]

            # 调用 Qwen 模型总结主题
            try:
                theme = qwen_model._query(
                    f"请用一句简练的话（不超过8个字）总结这首诗《{rhythmic}》的主题：\n\n{poem}。请你不要直接使用诗歌中的原句，回答的风格现代、白话一点，并且你只需要直接回答这个主题，不用回答其他的内容或者标点符号。"
                ).strip()
            except Exception as e:
                print(f"调用模型时出错：{e}")
                theme = "主题未知"

            # 生成新的内容
            new_request = f"请围绕'{theme}'创作"

            # 构造输出对象
            output_item = {
                "rhythmic": rhythmic,
                "poem": poem,
                "theme": theme,
                "request": new_request
            }

            # 写入文件
            json.dump(output_item, poem_f, ensure_ascii=False, indent=2)

            # 添加逗号，除最后一项外
            if idx < len(poem_data) - 1:
                poem_f.write(',\n')

            print(f"处理完成: {idx + 1}/{len(poem_data)}")

        poem_f.write('\n]')  # JSON数组结束

    print(f"所有诗歌处理完毕，结果已保存到文件：{output}")


if __name__ == "__main__":
    # 调用函数
    input_file = "valid_poems.json"  # 输入文件路径
    output_file = "valid_poems.json"  # 输出文件路径
    # summarize_and_generate_instruction(input_file, output_file)
    # Load JSON data from file
    with open(output_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Template to fill
    template = []
    # Iterate over the JSON dataset
    for idx, data in enumerate(json_data):
        masked_poem = masked_poem_dict[data["rhythmic"]]  # masked poem as format hint.
        prompt = poetry_prompt_template.format_map({
            "user_prompt": data["request"],
            "masked_poem": masked_poem
        })
        conversation = [
            {
                "from": "user",
                "value": prompt
            },
            {
                "from": "assistant",
                "value": data["poem"]
            }
        ]
        template.append({
            "id": f"identity_{idx}",
            "conversations": conversation
        })

    # Print the length of the template
    print(len(template))

    # Print an example of the filled template
    print(json.dumps(template[2], ensure_ascii=False, indent=2))

    # Write the template to a local file
    output_file_path = "new_poem_data.json"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"Processed data has been written to the file: {output_file_path}")