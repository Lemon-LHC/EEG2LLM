#!/bin/bash

# 激活正确的conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llama_factory
cd /data/lhc/projects/LLaMA-Factory
# 运行API服务
API_PORT=8001 llamafactory-cli api \
    --model_name_or_path "/data/lhc/models_new/Qwen3-0.6B_sleep_st_44_100hz_eeg10s-step10s_emo2.0s-step0.5s_win_all_tok9314_bal0.5_sqrt_inverse_202505071543_train/final_20250508_073419" \
    --template llama3 \
    --infer_backend vllm \
    --trust_remote_code \
    --vllm_enforce_eager \
    --vllm_maxlen 15000 \
    --max_new_tokens 16384 \
    --repetition_penalty 1.0 \
    --num_beams 1 \
    --length_penalty 1.0 \
    --skip_special_tokens \
    --max_length 15000\
