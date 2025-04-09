import os
os.environ['MODELSCOPE_CACHE'] = '/data/lhc/'

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径
model_path = "/data/lhc/model/Qwen/Qwen2.5-VL-7B-Instruct"

# 使用正确的模型类：AutoModelForVision2Seq 而不是 AutoModelForCausalLM
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 或使用 torch.bfloat16，取决于您的 GPU 支持
    trust_remote_code=True,
    device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "where are you from?"},
        ],
    }
]

# 准备推理
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 生成输出
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)