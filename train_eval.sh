#!/bin/bash

# 设置环境变量以避免内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置模型和数据集参数
MODEL_NAME="/data/lhc/models/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME=$(basename "$MODEL_NAME" | sed 's/-/_/g')  # 提取模型名称并将破折号替换为下划线

DATASET_DIR="data"
TRAIN_DATASET="edf200_100hz_15000ms_train"
TEST_DATASET="edf200_100hz_15000ms_test"

# 从数据集名称中提取信息
DATASET_INFO=$(echo "$TRAIN_DATASET" | sed 's/_train$//')  # 移除_train后缀

# 生成包含模型和数据集信息的输出目录名
RESULT_PREFIX="${MODEL_SHORT_NAME}_${DATASET_INFO}"
OUTPUT_DIR="saves/${MODEL_SHORT_NAME}/lora/${DATASET_INFO}"
TEST_DATA_PATH="/data/lhc/projects/LLaMA-Factory/data/${TEST_DATASET}.json"
EVAL_OUTPUT_DIR="/data/lhc/results/${RESULT_PREFIX}"
TENSORBOARD_BASE_DIR="${OUTPUT_DIR}/tensorboard"
TENSORBOARD_TRAIN_DIR="${TENSORBOARD_BASE_DIR}/train"  # 训练和验证集指标
TENSORBOARD_TEST_DIR="${TENSORBOARD_BASE_DIR}/test"   # 测试集指标

# 合并后的模型输出目录 - 设置为'none'以禁用导出
EXPORT_DIR="none"

# 设置训练参数
CUTOFF_LEN=12200  # 大幅减少截断长度以降低内存使用
LEARNING_RATE=5e-05
NUM_EPOCHS=3.0
MAX_SAMPLES=40000  # 减少样本数量以降低内存压力
TRAIN_BATCH_SIZE=1
GRAD_ACCUM_STEPS=16  # 进一步增加梯度累积步骤
WARMUP_STEPS=50
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
VAL_SIZE=0.11
GPU_COUNT=4
SAVE_STEPS=150  # 固定每150步保存一次

# 计算总步数和每个epoch的步数（用于信息显示）
TOTAL_SAMPLES=$(wc -l < "/data/lhc/projects/LLaMA-Factory/data/${TRAIN_DATASET}.json" || echo 1000)
EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS * GPU_COUNT))
TOTAL_STEPS=$((TOTAL_SAMPLES * NUM_EPOCHS / EFFECTIVE_BATCH_SIZE))
EPOCH_STEPS=$((TOTAL_SAMPLES / EFFECTIVE_BATCH_SIZE))

echo "训练总步数估计: $TOTAL_STEPS, 每个epoch步数: $EPOCH_STEPS"
echo "每${SAVE_STEPS}步评估一次测试集"

# 评估测试集的函数（使用原模型+LoRA权重）
test_with_lora() {
    CHECKPOINT_DIR=$1
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    echo "开始评估检查点: $CHECKPOINT_NAME (时间戳: $TIMESTAMP)..."
    
    # 创建评估结果目录
    TEST_EVAL_DIR="${EVAL_OUTPUT_DIR}/test_${CHECKPOINT_NAME}_${TIMESTAMP}"
    mkdir -p "$TEST_EVAL_DIR"
    
    # 检查checkpoint_dir是否存在adapter_config.json文件
    if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
        echo "警告: $CHECKPOINT_DIR 中没有找到adapter_config.json文件"
        # 尝试使用整个输出目录
        if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
            echo "使用 $OUTPUT_DIR 作为checkpoint目录进行评估"
            CHECKPOINT_DIR="$OUTPUT_DIR"
        else
            echo "错误: 无法找到有效的adapter_config.json文件，跳过此次评估"
            return 1
        fi
    fi
    
    # 使用LoRA权重进行评估（不合并模型）
    echo "使用LoRA权重评估测试集..."
    python eval_checkpoint.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --test_data "$TEST_DATA_PATH" \
        --output_dir "$TEST_EVAL_DIR" \
        --tensorboard_dir "${TENSORBOARD_TEST_DIR}" \
        --device cuda \
        --template alpaca
    
    echo "检查点 $CHECKPOINT_NAME 的测试集评估完成，结果保存在 $TEST_EVAL_DIR"
}

# 创建后台评估进程的函数
start_test_process() {
    echo "启动后台测试评估进程..."
    
    # 创建TensorBoard日志目录
    mkdir -p "${TENSORBOARD_TEST_DIR}"

    # 启动后台进程
    (
        LAST_PROCESSED_CHECKPOINT=""
        LAST_EVALUATED_STEP=0
        
        while true; do
            # 检查是否有新的检查点
            LATEST_CHECKPOINT=$(find ${OUTPUT_DIR}/checkpoint-* -type d 2>/dev/null | sort -V | tail -n 1)
            
            if [ -n "$LATEST_CHECKPOINT" ] && [ "$LATEST_CHECKPOINT" != "$LAST_PROCESSED_CHECKPOINT" ]; then
                CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
                CHECKPOINT_STEP=${CHECKPOINT_NAME#checkpoint-}
                
                # 如果达到保存间隔，则进行评估
                if (( $CHECKPOINT_STEP >= $LAST_EVALUATED_STEP + $SAVE_STEPS )); then
                    echo "当前步数: $CHECKPOINT_STEP, 开始评估"
                    
                    # 评估测试集
                    test_with_lora "$LATEST_CHECKPOINT"
                    
                    # 更新最后评估的步数
                    LAST_EVALUATED_STEP=$CHECKPOINT_STEP
                    
                    # 更新最后处理的检查点
                    LAST_PROCESSED_CHECKPOINT="$LATEST_CHECKPOINT"
                    
                    # 休息一下，避免GPU资源冲突
                    sleep 60
                else
                    echo "当前步数: $CHECKPOINT_STEP, 尚未达到保存间隔, 继续等待..."
                fi
            else
                echo "没有发现新的检查点, 继续等待..."
            fi
            
            # 等待下一次检查
            sleep 60
        done
    ) &
    TEST_PID=$!
    echo "后台测试评估进程已启动 (PID: $TEST_PID)"
}

# 合并模型并测试的函数（用于最终模型）
merge_and_test_final_model() {
    CHECKPOINT_DIR=$1
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    echo "开始处理最终模型..."
    
    # 如果EXPORT_DIR设置为'none'，跳过模型合并导出
    if [ "$EXPORT_DIR" = "none" ]; then
        echo "EXPORT_DIR设置为'none'，跳过模型合并导出"
        return 0
    fi
    
    # 检查checkpoint_dir是否存在adapter_config.json文件
    if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
        echo "警告: $CHECKPOINT_DIR 中没有找到adapter_config.json文件"
        # 尝试使用整个输出目录
        if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
            echo "使用 $OUTPUT_DIR 作为checkpoint目录进行合并"
            CHECKPOINT_DIR="$OUTPUT_DIR"
        else
            echo "错误: 无法找到有效的adapter_config.json文件，无法合并模型"
            return 1
        fi
    fi
    
    # 创建合并模型的输出目录
    FINAL_MERGED_DIR="${EXPORT_DIR}/final_${TIMESTAMP}"
    mkdir -p "$FINAL_MERGED_DIR"
    
    # 创建评估结果目录
    FINAL_EVAL_DIR="${EVAL_OUTPUT_DIR}/final_merged_${TIMESTAMP}"
    mkdir -p "$FINAL_EVAL_DIR"
    
    echo "合并LoRA权重和原模型到 $FINAL_MERGED_DIR..."
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
        --model_name_or_path "$MODEL_NAME" \
        --adapter_name_or_path "$CHECKPOINT_DIR" \
        --template alpaca \
        --finetuning_type lora \
        --export_dir "$FINAL_MERGED_DIR" \
        --export_size 2 \
        --export_device auto \
        --export_legacy_format False
    
    # 检查合并是否成功
    if [ $? -ne 0 ]; then
        echo "模型合并失败，跳过评估"
        return 1
    fi
    
    echo "合并完成，开始使用合并后的模型进行测试..."
    
    # 使用合并后的模型进行评估
    python eval_checkpoint.py \
        --checkpoint_dir "$FINAL_MERGED_DIR" \
        --test_data "$TEST_DATA_PATH" \
        --output_dir "$FINAL_EVAL_DIR" \
        --tensorboard_dir "${TENSORBOARD_TEST_DIR}" \
        --device cuda \
        --template alpaca
    
    echo "最终模型评估完成，结果保存在 $FINAL_EVAL_DIR"
}

# 激活conda环境
conda activate llama_factory
cd /data/lhc/projects/LLaMA-Factory

# 创建结果目录
mkdir -p "$EVAL_OUTPUT_DIR"
mkdir -p "${TENSORBOARD_BASE_DIR}"
mkdir -p "${TENSORBOARD_TRAIN_DIR}"
mkdir -p "${TENSORBOARD_TEST_DIR}"

# 启动TensorBoard服务
echo "启动TensorBoard服务..."
pkill -f tensorboard || true  # 关闭可能已经运行的TensorBoard进程
sleep 1
nohup tensorboard --logdir "${TENSORBOARD_BASE_DIR}" --bind_all --port 6006 > "${EVAL_OUTPUT_DIR}/tensorboard.log" 2>&1 &
echo "TensorBoard服务已启动，请访问 http://$(hostname -I | awk '{print $1}'):6006 查看训练和评估指标"
echo "  - train/: 训练过程和验证集指标（验证集从训练集中划分）"
echo "  - test/: 测试集评估指标（使用单独的测试集文件）"

# 开始后台测试评估进程
start_test_process

# 开始训练
echo "开始训练模型..."
torchrun --nnodes 1 --nproc_per_node $GPU_COUNT --master_port 29500 \
    src/llamafactory/launcher.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --dataset "$TRAIN_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --stage sft \
    --do_train True \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template alpaca \
    --flash_attn auto \
    --cutoff_len "$CUTOFF_LEN" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_samples "$MAX_SAMPLES" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps "$SAVE_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --packing False \
    --report_to tensorboard \
    --tensorboard_dir "${TENSORBOARD_TRAIN_DIR}" \
    --gradient_checkpointing True \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --val_size "$VAL_SIZE" \
    --eval_strategy steps \
    --eval_steps "$SAVE_STEPS" \
    --per_device_eval_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_checkpointing True

# 训练结束后，停止测试评估进程
if [ -n "$TEST_PID" ]; then
    echo "训练完成，停止后台测试评估进程 (PID: $TEST_PID)..."
    kill "$TEST_PID" 2>/dev/null || true
fi

# 模型训练完成后进行最终评估
echo "训练完成，进行最终评估..."

# 添加时间戳，避免覆盖之前的评估结果
FINAL_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_LORA_DIR="${OUTPUT_DIR}_final_${FINAL_TIMESTAMP}"
echo "保存最终LoRA权重到 $FINAL_LORA_DIR..."
cp -r "$OUTPUT_DIR" "$FINAL_LORA_DIR"

# 合并最终模型并评估
echo "使用最终LoRA权重合并模型并评估..."
merge_and_test_final_model "$OUTPUT_DIR"

echo "训练、评估和合并全部完成！"
