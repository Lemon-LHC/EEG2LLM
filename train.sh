#!/bin/bash
# 使用torchrun启动LLaMA-Factory分布式训练

# 配置路径和参数
MODEL_NAME="/data/lhc/models/Llama-3.2-1B-Instruct"
DATASET_DIR="/data/lhc/datasets_new/sleep"
TRAIN_DATASET="edf5_100hz_10000ms_tok8363_train"
TEST_DATASET="edf5_100hz_10000ms_tok8363_test"
OUTPUT_DIR="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/${TRAIN_DATASET}"
LLAMAFACTORY_PATH="/data/lhc/projects/LLaMA-Factory"
GPU_IDS="0,1,2,3"
PORT=6006
SERVER_IP="192.168.1.110"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# 使用torchrun启动训练
GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "正在使用torchrun启动分布式训练..."
echo "GPU数量: $GPU_COUNT"
echo "使用GPU: $GPU_IDS"

cd "$LLAMAFACTORY_PATH"

torchrun --nnodes 1 --nproc_per_node "$GPU_COUNT" --master_port 29500 \
    src/llamafactory/launcher.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$TRAIN_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --template alpaca \
    --flash_attn auto \
    --cutoff_len 500 \
    --learning_rate 5e-05 \
    --num_train_epochs 6.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --save_steps 20 \
    --eval_steps 20 \
    --eval_delay 0 \
    --warmup_steps 50 \
    --packing False \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --fp16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --val_size 0.11 \
    --eval_strategy steps \
    --per_device_eval_batch_size 1

echo "训练完成！"