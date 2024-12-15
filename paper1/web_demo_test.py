import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained('./bestChatPoet', device_map='auto', trust_remote_code=True, quantization_config=quantization_config)
# model = AutoModelForCausalLM.from_pretrained('./CharPoet', device_map='auto', trust_remote_code=True, torch_dtype=torch.float16, quantization_config=None)
tokenizer = AutoTokenizer.from_pretrained('./bestChatPoet', trust_remote_code=True)

from bestChatPoet.utils import write_poem
import gradio as gr

demo = gr.Interface(
    fn = lambda prompt, rhythmic: write_poem(model, tokenizer, prompt, rhythmic),
    inputs=[gr.Textbox(label="提示/Prompt", value="请围绕'春暖花开'主题创作")
            ,gr.Radio(["五言绝句/Wuyanjueju", "七言绝句/Qiyanjueju", "五言律诗/Wuyanlvshi", "七言律诗/Qiyanlvshi", "如梦令/Rumengling", "减字木兰花/Jianzimulanhua", "清平乐/Qingpingyue", "蝶恋花/Dielianhua", "满江红/Manjianghong" , "沁园春/Qinyuanchun", "水调歌头/Shuidiaogetou", "菩萨蛮/Pusaman"], label="诗词种类/Format type", value='七言绝句/Qiyanjueju')
            ],
    outputs=gr.Textbox(label="诗歌创作结果/Result"),
    examples = [
        ["Write me a poem about Spring.", "如梦令/Rumengling"],
        ["Introduce New York city.", "沁园春/Qinyuanchun"]
    ],
    title="CharPoet",
    description="poem generator based on token-free LLM"
)

demo.launch(share=True)
