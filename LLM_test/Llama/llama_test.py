import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/data/lhc/models/Llama-3.2-1B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 准备输入消息
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# 将消息格式化为模型输入
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

# 生成回复
outputs = model.generate(
    inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

# 解码并打印结果
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)