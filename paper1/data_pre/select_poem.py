import json
import os

# 定义需要筛选的词牌名
target_rhythmics = {"菩萨蛮", "沁园春", "清平乐", "如梦令", "蝶恋花", "水调歌头", "卜算子", "减字木兰花", "满江红"}

# 定义文件名模板和输出文件
input_file_template = "./chinese-poetry/宋词/ci.song.{num}.json"  # 输入文件名模板
output_file = "filtered_poems.json"  # 替换为你的输出文件路径

all_num = 0
# 检查输出文件是否存在
if os.path.exists(output_file):
    # 如果文件存在，加载现有数据
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告：文件 {output_file} 内容无效，已重新初始化为空列表。")
            existing_data = []
else:
    # 如果文件不存在，初始化为空列表
    existing_data = []

# 遍历从 6 到 21 的文件名
for num in range(0, 22):
    input_file = input_file_template.format(num=num*1000)
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过...")
        continue

    # 加载当前输入 JSON 数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 筛选符合条件的数据并合并 paragraphs
    filtered_data = []
    for entry in data:
        if entry.get("rhythmic") in target_rhythmics:
            # 将 paragraphs 合并成一首完整的诗，每行后加换行符
            combined_paragraphs = "\n".join(entry.get("paragraphs", [])) + "\n"
            filtered_data.append({
                "rhythmic": entry["rhythmic"],
                "poem": combined_paragraphs
            })
    all_num = all_num + len(filtered_data)
    print(f"文件 {input_file} 筛选出 {len(filtered_data)} 条符合条件的数据。")
    # 合并新筛选的数据到现有数据
    existing_data.extend(filtered_data)

# 写入更新后的数据到文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=4)

print(f"所有文件处理完成，结果已保存到 {output_file}")

print(f"共有 {all_num}")