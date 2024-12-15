import json
import os

# 定义文件路径模板
input_file_template = "./chinese-poetry/全唐诗/poet.tang.{}.json"  # 输入文件路径模板
output_file = "filtered_poems.json"  # 输出文件路径

# 定义函数检查句子格式
def is_five_character_line(line):
    """
    检查一个句子是否符合格式：逗号前是5个汉字，逗号后是6个汉字。
    """
    parts = line.split("，")
    if len(parts) != 2:  # 确保句子由一个逗号分隔为两部分
        return False
    return len(parts[0]) == 5 and len(parts[1]) == 6

def is_seven_character_line(line):
    """
    检查一个句子是否符合格式：逗号前是7个汉字，逗号后是7个汉字。
    """
    parts = line.split("，")
    if len(parts) != 2:  # 确保句子由一个逗号分隔为两部分
        return False
    return len(parts[0]) == 7 and len(parts[1]) == 8

# 检查输出文件是否存在
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
else:
    existing_data = []

# 遍历文件
for i in range(28):  # 从 0 到 19
    input_file = input_file_template.format(i*1000)  # 替换文件名中的数字
    if not os.path.exists(input_file):  # 如果文件不存在，跳过
        print(f"文件 {input_file} 不存在，跳过...")
        continue

    # 加载输入 JSON 数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 筛选符合条件的数据
    for entry in data:
        paragraphs = entry.get("paragraphs", [])
        if all(is_five_character_line(line) for line in paragraphs):
            if len(paragraphs) == 2:
                rhythmic = "五言绝句"
            elif len(paragraphs) == 4:
                rhythmic = "五言律诗"
            else:
                continue
        elif all(is_seven_character_line(line) for line in paragraphs):
            if len(paragraphs) == 2:
                rhythmic = "七言绝句"
            elif len(paragraphs) == 4:
                rhythmic = "七言律诗"
            else:
                continue
        else:
            continue

        # 合并段落为完整诗，每行后加换行符
        combined_paragraphs = "\n".join(paragraphs) + "\n"
        # 添加到分类结果
        existing_data.append({
            "rhythmic": rhythmic,
            "poem": combined_paragraphs
        })

# 写入更新后的数据到文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=4)

print(f"筛选完成，共分类了 {len(existing_data)} 首诗，结果已保存到 {output_file}")