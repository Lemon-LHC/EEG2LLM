#!/bin/bash
# 使用torchrun启动LLaMA-Factory训练

# 配置路径
LLAMA_FACTORY_PATH="/data/lhc/projects/LLaMA-Factory"
MODEL_PATH="/data/lhc/models_new/sota_llama_edf200_100hz_10000ms_train"
OUTPUT_DIR="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf200_100hz_10000ms_tok8363_train"
DATASET_DIR="/data/lhc/datasets_new/sleep"
DATASET_NAME="edf200_100hz_10000ms_tok8363_train"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量以避免内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 切换到LLaMA-Factory目录
cd "$LLAMA_FACTORY_PATH"

# 使用torchrun启动训练
torchrun --nnodes 1 --nproc_per_node 4 --master_port 29500 \
    src/llamafactory/launcher.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --stage sft \
    --do_train True \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template alpaca \
    --flash_attn auto \
    --cutoff_len 8363 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
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
    --per_device_eval_batch_size 1 \
    --sampling_strategy "balanced" \
    --balance_alpha 0.5 \
    --class_weight_method "sqrt_inverse" \
    --base_output_dir "$OUTPUT_DIR" \
    --num_epochs 2.0 \
    --train_batch_size 1 \
    --grad_accum_steps 4 \
    --test_interval 3000

echo "训练完成，请检查输出目录: $OUTPUT_DIR" 