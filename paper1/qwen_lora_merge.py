from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Step 1: 设置路径
path_to_adapter = "/root/Qwen-main/output3_qwen/checkpoint-1000/"  # 微调后的 LoRA 模型路径
new_model_directory = "/root/Qwen-1_8B-ChatPoet"  # 保存合并模型的路径

# Step 2: 加载基础模型和 LoRA 微调权重
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter,  # LoRA 权重路径
    device_map="auto",  # 自动映射设备
    trust_remote_code=True  # 信任远程代码（用于自定义模型）
).eval()  # 设置模型为评估模式

# Step 3: 合并 LoRA 权重
merged_model = model.merge_and_unload()  # 将 LoRA 权重合并到基础模型

# Step 4: 保存合并后的模型
# max_shard_size 用于分片存储较大的模型文件，safe_serialization 启用 safetensors 格式
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)

# Step 5: 保存分词器
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter,  # 与微调模型同路径
    trust_remote_code=True  # 信任远程代码
)
tokenizer.save_pretrained(new_model_directory)  # 保存分词器

print(f"模型和分词器已保存到 {new_model_directory}")
