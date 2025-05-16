#!/bin/bash

# 睡眠阶段分类模型训练与评估一体化脚本
# 流程：训练 -> 启动API服务 -> 评估测试 -> 关闭API服务

# 设置错误时立即退出
set -e

# 颜色输出函数
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

print_blue() {
    echo -e "\033[34m$1\033[0m"
}

# 设置关键路径变量
SCRIPT_DIR="/data/lhc/projects/fine"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_old.py"
MODEL_NAME="/data/lhc/models_new/Llama-3.2-1B-Instruct_edf197_100hz_15000ms_tok12521_train_balanced_0.1/final_20250409_090750"
DATASET_DIR="/data/lhc/datasets_new/sleep"
TRAIN_DATASET="train/ST_edf44_100hz_15000ms_raw_clean_tok12588_bal0.5_sqrt_inverse_train"  # 完整的相对路径
TEST_DATASET="test/ST_edf44_100hz_15000ms_raw_clean_tok12588_bal0.5_sqrt_inverse_test"     # 完整的相对路径

# 为API和评估阶段准备变量
API_PORT=8000
API_LOG_FILE="/tmp/api_service_${API_PORT}.log"
API_PID_FILE="/tmp/api_service_${API_PORT}.pid"

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/data/lhc/results/${TIMESTAMP}_balanced_sqrt_inverse"
mkdir -p $RESULTS_DIR
LOG_FILE="${RESULTS_DIR}/train_test.log"

# 提取模型短名称和数据集信息，用于构建导出路径
MODEL_SHORT_NAME=$(basename $MODEL_NAME)
DATASET_INFO=$(basename ${TRAIN_DATASET})
# 构建标准化的导出路径
EXPORT_DIR="/data/lhc/models_new/${MODEL_SHORT_NAME}_${DATASET_INFO}"

# ==== 第一阶段：训练模型 ====
print_blue "===== 第一阶段：开始训练模型 ====="
print_yellow "使用训练集: ${TRAIN_DATASET}"
print_yellow "训练日志将保存到: ${LOG_FILE}"

# 激活LLaMA-Factory虚拟环境
source /home/lhc/anaconda3/bin/activate llama_factory

# 设置环境变量以避免线程冲突
export MKL_THREADING_LAYER=GNU

# 切换到脚本所在目录并运行
cd "${SCRIPT_DIR}"
print_yellow "开始训练模型..."

# 训练模型
python "${TRAIN_SCRIPT}" \
  --model_name $MODEL_NAME \
  --dataset_dir $DATASET_DIR \
  --train_dataset $TRAIN_DATASET \
  --sampling_strategy "original" \
  --balance_alpha 0.1 \
  --class_weight_method "sqrt_inverse" \
  --base_output_dir $RESULTS_DIR \
  --export_dir $EXPORT_DIR \
  --num_epochs 3 \
  --learning_rate 5e-05 \
  --train_batch_size 1 \
  --grad_accum_steps 4 \
  --lora_rank 8 \
  --save_steps 5000 \
  --cutoff_len 12588 2>&1 | tee -a "${LOG_FILE}"

# 检查训练是否成功
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
  print_red "训练失败，退出代码: $TRAIN_EXIT_CODE"
  exit $TRAIN_EXIT_CODE
fi

print_green "训练成功完成！"

# 查找导出的最终模型目录
FINAL_MODEL_DIR=$(find $EXPORT_DIR -type d -name "final_*" | sort -r | head -n 1)
if [ -z "$FINAL_MODEL_DIR" ]; then
  print_red "未找到导出的最终模型目录，请检查导出路径: $EXPORT_DIR"
  exit 1
fi

print_yellow "找到导出的最终模型: $FINAL_MODEL_DIR"

# ==== 第二阶段：启动API服务 ====
print_blue "===== 第二阶段：启动API服务 ====="

# 检查API端口是否已被占用，如果占用则结束相应进程
if lsof -i:${API_PORT} > /dev/null 2>&1; then
  print_yellow "端口 ${API_PORT} 已被占用，尝试关闭相关进程..."
  kill -9 $(lsof -t -i:${API_PORT}) || true
  sleep 2
fi

# 启动API服务
print_yellow "启动API服务，使用模型: ${FINAL_MODEL_DIR}"
cd /data/lhc/projects/LLaMA-Factory

# 启动API服务并保存PID
API_PORT=$API_PORT llamafactory-cli api \
    --model_name_or_path "${FINAL_MODEL_DIR}" \
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
    --max_length 65536 > "${API_LOG_FILE}" 2>&1 &

API_PID=$!
echo $API_PID > "${API_PID_FILE}"

# 等待API服务启动
print_yellow "等待API服务启动..."
max_attempts=30
attempt=0
api_ready=false

while [ $attempt -lt $max_attempts ]; do
  attempt=$((attempt+1))
  if curl -s "http://localhost:${API_PORT}/v1/models" > /dev/null 2>&1; then
    api_ready=true
    break
  fi
  print_yellow "等待API服务启动，尝试 $attempt/$max_attempts..."
  sleep 5
done

if [ "$api_ready" = false ]; then
  print_red "API服务启动失败或超时，查看日志: ${API_LOG_FILE}"
  exit 1
fi

print_green "API服务已成功启动，PID: ${API_PID}"

# ==== 第三阶段：进行模型评估 ====
print_blue "===== 第三阶段：开始模型评估 ====="

# 确保测试数据集路径完整
TEST_DATA_FULLPATH="${DATASET_DIR}/${TEST_DATASET}.json"

# 检查测试数据集是否存在
if [ ! -f "${TEST_DATA_FULLPATH}" ]; then
  print_red "测试数据集不存在: ${TEST_DATA_FULLPATH}"
  # 关闭API服务
  if [ -f "${API_PID_FILE}" ]; then
    kill -9 $(cat "${API_PID_FILE}") || true
    rm -f "${API_PID_FILE}"
  fi
  exit 1
fi

print_yellow "使用测试集: ${TEST_DATASET}"
print_yellow "开始评估模型..."

# 提取模型名和测试集名用于评估结果
MODEL_NAME_SHORT=$(basename "${FINAL_MODEL_DIR}" | sed 's/final_[0-9]*_[0-9]*/final/')
TEST_SET_NAME=$(basename "${TEST_DATASET}")

# 运行评估脚本
cd "${SCRIPT_DIR}"
API_PORT=${API_PORT} \
TEST_DATA_PATH="${TEST_DATA_FULLPATH}" \
MODEL_NAME="${MODEL_NAME_SHORT}" \
SAVE_DIR="${RESULTS_DIR}/evaluation" \
PRINT_INTERVAL=5 \
SAVE_INTERVAL=100 \
python "${SCRIPT_DIR}/test_api_llm.py" 2>&1 | tee -a "${LOG_FILE}"

# 检查评估是否成功
EVAL_EXIT_CODE=${PIPESTATUS[0]}
if [ $EVAL_EXIT_CODE -ne 0 ]; then
  print_red "评估失败，退出代码: $EVAL_EXIT_CODE"
  # 即使评估失败也要正常关闭API服务
fi

# ==== 第四阶段：关闭API服务 ====
print_blue "===== 第四阶段：关闭API服务 ====="

# 关闭API服务
if [ -f "${API_PID_FILE}" ]; then
  API_PID=$(cat "${API_PID_FILE}")
  print_yellow "关闭API服务 (PID: ${API_PID})..."
  kill -9 $API_PID || true
  rm -f "${API_PID_FILE}"
  print_green "API服务已关闭"
else
  print_yellow "未找到API服务PID文件，尝试关闭所有API端口进程..."
  kill -9 $(lsof -t -i:${API_PORT}) 2>/dev/null || true
fi

# 清理临时文件
rm -f "${API_LOG_FILE}" || true

# 结束流程
if [ $TRAIN_EXIT_CODE -eq 0 ] && [ $EVAL_EXIT_CODE -eq 0 ]; then
  print_green "===== 训练和评估流程全部成功完成！======"
  print_green "训练和评估日志: ${LOG_FILE}"
  print_green "评估结果目录: ${RESULTS_DIR}/evaluation"
  print_green "训练导出模型: ${FINAL_MODEL_DIR}"
  exit 0
else
  print_red "===== 训练和评估流程未完全成功 ======"
  print_red "训练退出码: ${TRAIN_EXIT_CODE}, 评估退出码: ${EVAL_EXIT_CODE}"
  print_red "检查日志: ${LOG_FILE}"
  exit 1
fi 