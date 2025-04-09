import json
import os
from transformers import AutoTokenizer
import numpy as np

# 加载分词器
tokenizer_path = "/data/lhc/models/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 加载测试数据
test_file = "/data/lhc/projects/LLaMA-Factory/data/edf1_100hz_30000ms_test.json"
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 计算token数量
instruction_tokens = []
input_tokens = []
total_tokens = []

for idx, item in enumerate(test_data):
    # 构建完整提示
    system = item.get('system', '')
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output = item.get('output', '')
    
    # 计算指令部分的token数量
    instruction_token_count = len(tokenizer.encode(instruction))
    instruction_tokens.append(instruction_token_count)
    
    # 计算输入部分的token数量
    input_token_count = len(tokenizer.encode(input_text))
    input_tokens.append(input_token_count)
    
    # 计算总token数量（包括系统提示）
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction + "\n" + input_text}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        prompt_text = instruction + "\n" + input_text
    
    total_token_count = len(tokenizer.encode(prompt_text))
    total_tokens.append(total_token_count)
    
    if idx < 3 or idx == len(test_data) - 1:
        print(f"样本 {idx+1}:")
        print(f"  指令部分token数: {instruction_token_count}")
        print(f"  输入部分token数: {input_token_count}")
        print(f"  总token数: {total_token_count}")
        print()

# 计算统计信息
print("统计信息:")
print(f"样本数量: {len(test_data)}")
print(f"指令部分平均token数: {np.mean(instruction_tokens):.2f} ± {np.std(instruction_tokens):.2f}")
print(f"输入部分平均token数: {np.mean(input_tokens):.2f} ± {np.std(input_tokens):.2f}")
print(f"总平均token数: {np.mean(total_tokens):.2f} ± {np.std(total_tokens):.2f}")
print(f"最大token数: {max(total_tokens)}")
print(f"最小token数: {min(total_tokens)}")

# 分析CUTOFF_LEN设置
cutoff_len = 40000
over_limit = sum(1 for t in total_tokens if t > cutoff_len)
print(f"\n超过CUTOFF_LEN ({cutoff_len})的样本数: {over_limit} ({over_limit/len(test_data)*100:.2f}%)")
