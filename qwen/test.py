from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 设置模型路径
model_path = "/data/lhc/model/Qwen/Qwen2.5-VL-7B-Instruct"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动选择设备
    trust_remote_code=True,
    torch_dtype=torch.float16  # 使用半精度以节省显存
)

# 设置模型为评估模式
model.eval()

# 示例对话函数
def chat(prompt, image_path=None):
    if image_path:
        # 如果有图片输入
        query = tokenizer.from_list_format([
            
            {'text': 你好}
        ])
    else:
        # 纯文本输入
        query = prompt
    
    response, history = model.chat(tokenizer, query, history=[])
    return response

# 使用示例
# 纯文本对话
response = chat("你好，请介绍一下你自己。")
print(response)

# 图文对话
# response = chat("这张图片里有什么？", "path/to/your/image.jpg")
# print(response)