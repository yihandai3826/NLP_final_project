from datasets import load_dataset
import json

# 加载数据集
dataset = load_dataset("BelleGroup/train_1M_CN")

# 存储转换后的数据
converted_data = []
idx = 0

# 遍历数据集中的每条记录，转换格式
for data in dataset['train']:
    # 确保字段存在，避免缺失字段引发错误
    if "instruction" in data and "output" in data:
        conversation = [
            {
                "from": "user",
                "value": data['instruction']
            },
            {
                "from": "assistant",
                "value": data["output"]
            }
        ]
        converted_data.append({
            "id": f"identity_{idx}",
            "conversations": conversation
        })
    idx += 1

# 保存为 JSON 文件
output_file = "converted_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print(f"转换完成，保存到 {output_file}")