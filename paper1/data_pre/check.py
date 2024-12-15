import json
import re

# 定义韵律要求
masked_poem_dict = {
    "五言绝句": [
        ".{5}，.{5}。",
        ".{5}，.{5}。",
    ],
    "七言绝句": [
        ".{7}，.{7}。",
        ".{7}，.{7}。",
    ],
    "五言律诗": [
        ".{5}，.{5}。",
        ".{5}，.{5}。",
        ".{5}，.{5}。",
        ".{5}，.{5}。",
    ],
    "七言律诗": [
        ".{7}，.{7}。",
        ".{7}，.{7}。",
        ".{7}，.{7}。",
        ".{7}，.{7}。",
    ],
    "减字木兰花": [
        ".{4}。",
        ".{7}。",
        ".{4}。",
        ".{7}。",
        ".{4}。",
        ".{7}。",
        ".{4}。",
        ".{7}。",
    ],
    "满江红": [
        ".{4}，.{3}、.{4}。",
        ".{3}、.{4}，.{4}。",
        ".{7}，.{7}。",
        ".{3}、.{5}，.{3}。",
        ".{3}，.{3}。",
        ".{3}，.{3}。",
        ".{5}，.{4}。",
        ".{7}，.{7}。",
        ".{3}、.{5}，.{3}。",
    ],
    "水调歌头": [
        ".{5}，.{5}。",
        ".{4}，.{7}。",
        ".{6}，.{6}，.{5}。",
        ".{5}，.{5}。",
        ".{3}，.{3}，.{3}。",
        ".{4}，.{7}。",
        ".{6}，.{6}，.{5}。",
        ".{5}，.{5}。",
    ],
    "蝶恋花": [
        ".{7}。",
        ".{4}，.{5}。",
        ".{7}。",
        ".{7}。",
        ".{7}。",
        ".{4}，.{5}。",
        ".{7}。",
        ".{7}。",
    ],
    "如梦令": [
        ".{6}。",
        ".{6}。",
        ".{5}，.{6}。",
        ".{2}。",
        ".{2}。",
        ".{6}。",
    ],
    "清平乐": [
        ".{4}。",
        ".{5}。",
        ".{7}。",
        ".{6}。",
        ".{6}。",
        ".{6}。",
        ".{6}，.{6}。",
    ],
    "沁园春": [
        ".{4}，.{4}，.{4}。",
        ".{5}，.{4}，.{4}，.{4}。",
        ".{4}，.{4}，.{7}。",
        ".{3}，.{5}，.{4}。",
        ".{2}。",
        ".{4}。",
        ".{8}。",
        ".{5}，.{4}，.{4}，.{4}。",
        ".{4}，.{4}，.{7}。",
        ".{3}，.{5}，.{4}。",
    ],
    "菩萨蛮": [
        ".{7}。",
        ".{7}。",
        ".{5}。",
        ".{5}。",
        ".{5}。",
        ".{5}。",
        ".{5}。",
        ".{5}。",
    ]
}
# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 检查是否符合韵律格式
def check_rhythmic(poem, rhythmic):
    if rhythmic not in masked_poem_dict:
        return False, f"韵律 '{rhythmic}' 未定义"

    # 获取韵律的行模式
    rhythmic_patterns = masked_poem_dict[rhythmic]

    # 分割诗句为行，去除多余空白
    poem_lines = [line.strip() for line in poem.strip().split("\n") if line.strip()]
    if len(poem_lines) != len(rhythmic_patterns):
        return False, f"行数不匹配：期待 {len(rhythmic_patterns)} 行，实际 {len(poem_lines)} 行"

    # 按行匹配
    for i, (line, pattern) in enumerate(zip(poem_lines, rhythmic_patterns), 1):
        if not re.fullmatch(pattern, line):
            return False, f"第 {i} 行不匹配：'{line}' 不符合模式 '{pattern}'"

    return True, "符合韵律"

# 主函数
def main(file_path):
    data = load_json(file_path)
    results = []
    valid_poems = []

    for i, entry in enumerate(data):
        rhythmic = entry.get("rhythmic", "未知")
        poem = entry.get("poem", "")
        is_valid, message = check_rhythmic(poem, rhythmic)
        if is_valid:
            valid_poems.append(entry)  # 添加匹配成功的诗到列表中
        results.append({
            "id": entry.get("id", f"poem_{i}"),
            "rhythmic": rhythmic,
            "is_valid": is_valid,
            "message": message,
        })

    # 打印结果
    for result in results:
        print(f"ID: {result['id']}, 韵律: {result['rhythmic']}, 检查结果: {result['message']}")

    # 保存检查结果
    output_path = "rhythmic_check_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"检查结果已保存到 {output_path}")

    # 保存匹配成功的诗歌
    valid_output_path = "valid_poems.json"
    print(len(valid_poems))
    with open(valid_output_path, "w", encoding="utf-8") as f:
        json.dump(valid_poems, f, ensure_ascii=False, indent=4)
    print(f"匹配成功的诗歌已保存到 {valid_output_path}")

# 执行脚本
if __name__ == "__main__":
    # 替换为你的 JSON 文件路径
    file_path = "instructions_with_original.json"
    main(file_path)