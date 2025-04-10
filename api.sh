#!/bin/bash

# 激活正确的conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llama_factory
cd /data/lhc/projects/LLaMA-Factory
# 运行API服务
API_PORT=8000 llamafactory-cli api \
    --model_name_or_path "/data/lhc/models_new/Llama-3.2-1B-Instruct_edf10_200hz_7500ms_tok12588_balanced_0.8_sqrt_inverse_train/final_20250410_173759" \
    --template llama3 \
    --infer_backend vllm \
    --trust_remote_code \
    --vllm_enforce_eager \
    --vllm_maxlen 65536 \
    --max_new_tokens 16384 \
    --repetition_penalty 1.0 \
    --num_beams 1 \
    --length_penalty 1.0 \
    --skip_special_tokens \
    --max_length 65536 \
