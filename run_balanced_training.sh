#!/bin/bash
# 使用torchrun启动LLaMA-Factory训练 - 平衡样本版本

# 配置路径
LLAMA_FACTORY_PATH="/data/lhc/projects/LLaMA-Factory"
MODEL_PATH="/data/lhc/models/Llama-3.2-1B-Instruct"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf197_100hz_10000ms_balanced_${TIMESTAMP}"
DATASET_DIR="/data/lhc/datasets_new/sleep/train/"
DATASET_NAME="edf197_100hz_10000ms_tok8521_train"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/train.log"

# 设置环境变量以避免内存碎片和线程冲突
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU

echo "启动平衡训练，使用balanced平衡策略和sqrt_inverse权重计算方法..."
echo "输出目录: $OUTPUT_DIR"
echo "训练日志: $LOG_FILE"

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
    --cutoff_len 8600 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --eval_steps 1000 \
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
    --test_interval 3000 > "$LOG_FILE" 2>&1

EXITCODE=$?
if [ $EXITCODE -eq 0 ]; then
  echo "训练成功完成！结果保存在 ${OUTPUT_DIR}"
else
  echo "训练过程中出现错误(退出码: $EXITCODE)，请查看日志: ${LOG_FILE}"
fi 