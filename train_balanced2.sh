#!/bin/bash
# 使用torchrun启动LLaMA-Factory训练 - 平衡样本版本 (修复版)
# 此版本先通过dataset_balancer.py进行数据平衡处理，再使用LLaMA-Factory训练

# 配置路径
LLAMA_FACTORY_PATH="/data/lhc/projects/LLaMA-Factory"
MODEL_PATH="/data/lhc/models/Llama-3.2-1B-Instruct"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_DIR="/data/lhc/datasets_new/sleep"
ORIG_DATASET_PATH="${DATASET_DIR}/train/edf197_200hz_10000ms_tok16521_train.json"
ORIG_DATASET_NAME="edf197_200hz_10000ms_tok16521_train"
BALANCED_DATASET_NAME="${ORIG_DATASET_NAME}_balanced_${TIMESTAMP}"
BALANCED_DATASET_PATH="${DATASET_DIR}/balanced/${BALANCED_DATASET_NAME}.json"
OUTPUT_DIR="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf197_200hz_10000ms_balanced_${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "${DATASET_DIR}/balanced"
LOG_FILE="${OUTPUT_DIR}/train.log"

# 设置环境变量以避免内存碎片和线程冲突
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_THREADING_LAYER=GNU

# 设置可见的GPU设备 - 使用全部4个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "== 步骤1: 使用dataset_balancer.py预处理数据集 =="
echo "平衡策略: balanced, 平衡系数: 0.7, 权重方法: sqrt_inverse"
echo "训练日志将同时保存到 ${LOG_FILE} 并显示在终端"

# 切换到fine目录预处理数据
cd "/data/lhc/projects/fine"
python dataset_balancer.py \
  --input_file "${ORIG_DATASET_PATH}" \
  --output_file "${BALANCED_DATASET_PATH}" \
  --strategy "balanced" \
  --balance_alpha 0.7 \
  --weight_method "sqrt_inverse" \
  --update_config 2>&1 | tee -a "${LOG_FILE}"

BALANCER_EXIT=${PIPESTATUS[0]}
if [ $BALANCER_EXIT -ne 0 ]; then
  echo "数据平衡处理失败(退出码: $BALANCER_EXIT)，终止训练"
  exit $BALANCER_EXIT
fi

echo "== 步骤2: 使用LLaMA-Factory训练平衡后的数据集 =="
echo "输出目录: $OUTPUT_DIR"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 切换到LLaMA-Factory目录
cd "$LLAMA_FACTORY_PATH"

# 使用torchrun启动训练，使用tee同时输出到终端和日志文件
# 使用全部4个GPU，优化显存使用
torchrun --nnodes 1 --nproc_per_node 4 --master_port 29500 \
    src/llamafactory/launcher.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$BALANCED_DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --stage sft \
    --do_train True \
    --preprocessing_num_workers 4 \
    --finetuning_type lora \
    --template alpaca \
    --flash_attn auto \
    --cutoff_len 16521 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 40000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 10000 \
    --eval_steps 5000 \
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
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --val_size 0.11 \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    2>&1 | tee -a "${LOG_FILE}"

EXITCODE=${PIPESTATUS[0]}
if [ $EXITCODE -eq 0 ]; then
  echo "训练成功完成！结果保存在 ${OUTPUT_DIR}"
else
  echo "训练过程中出现错误(退出码: $EXITCODE)，日志已保存到: ${LOG_FILE}"
fi 