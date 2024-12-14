import transformers
import re
from bestChatPoet.utils import write_poem
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained('./bestChatPoet', device_map='auto', trust_remote_code=True, quantization_config=quantization_config)
# model = AutoModelForCausalLM.from_pretrained('./CharPoet', device_map='auto', trust_remote_code=True, torch_dtype=torch.float16, quantization_config=None)
tokenizer = AutoTokenizer.from_pretrained('./bestChatPoet', trust_remote_code=True)

# 定义主题和韵律格式
theme = "请围绕'离愁别绪'创作"
rhythmic_formats = [
    "五言绝句/Wuyanjueju",
    "七言绝句/Qiyanjueju",
    "五言律诗/Wuyanlvshi",
    "七言律诗/Qiyanlvshi",
    "如梦令/Rumengling",
    "减字木兰花/Jianzimulanhua",
    "清平乐/Qingpingyue",
    "蝶恋花/Dielianhua",
    "满江红/Manjianghong",
    "沁园春/Qinyuanchun",
    "水调歌头/Shuidiaogetou",
    "菩萨蛮/Pusaman"
]

# 函数实现：以每种韵律生成10次，并检查准确率
def test_poem_accuracy(model, tokenizer, theme, rhythmic_formats, num_trials=100):
    results = {}
    for rhythmic in rhythmic_formats:
        success_count = 0  # 记录成功生成的诗歌数量

        for _ in range(num_trials):
            response_formated = write_poem(model, tokenizer, theme, rhythmic)

            # 检查生成的内容是否含有 <|extra_1|>
            if "<|extra_1|>" not in response_formated:
                success_count += 1

        # 记录每种韵律的准确率
        accuracy = success_count / num_trials
        results[rhythmic] = accuracy

    # 打印每种韵律的结果
    for rhythmic, accuracy in results.items():
        print(f"韵律: {rhythmic}, 准确率: {accuracy:.2%}")

    return results

# 示例：测试函数调用
# 假设 `model` 和 `tokenizer` 已经定义并加载
# 调用测试函数
results = test_poem_accuracy(model, tokenizer, theme, rhythmic_formats)

# 打印最终结果
print("各韵律生成准确率统计：")
for rhythmic, accuracy in results.items():
    print(f"{rhythmic}: {accuracy:.2%}")
