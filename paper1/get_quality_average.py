import json

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

def process_json_data(input_file, output_file):
    """
    处理 JSON 数据，按 10 个为一组，计算 `total` 不为 0 的数据的 `score` 平均值。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []  # 存储每组的均值
    group = []  # 当前分组

    for idx, item in enumerate(data):
        if item['total'] != 0:
            group.append(item['total'])
        if (idx+1) % 10 == 0:
            results.append(sum(group)/len(group))
            print(rhythmic_formats[int((idx+1)/10)-1])
            print("内容质量平均分为：" + str(sum(group)/len(group)))
            group = []

    ans = sum(results) / len(results)

    print(f"处理完成，总平均分为 {ans}")


# 示例用法
input_file = 'score.json'  # 输入 JSON 文件
output_file = 'results.json'  # 输出 JSON 文件

process_json_data(input_file, output_file)