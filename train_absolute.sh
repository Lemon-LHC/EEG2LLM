#!/bin/bash

# 睡眠阶段分类模型训练脚本 - 使用绝对路径执行
# 使用推荐的balanced策略和sqrt_inverse权重计算方法

# 设置关键路径变量（全部使用绝对路径）
TRAIN_SCRIPT="/data/lhc/projects/fine/train_old.py"
MODEL_NAME="/data/lhc/models_new/llama_edf197_10000_balanced/"
DATASET_DIR="/data/lhc/datasets_new/sleep"
TRAIN_DATASET="edf197_100hz_10000ms_tok8521_train"
TEST_DATASET="edf197_100hz_10000ms_tok8521_test"

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/data/lhc/results/${TIMESTAMP}_balanced_sqrt_inverse"
mkdir -p $RESULTS_DIR
LOG_FILE="${RESULTS_DIR}/train.log"

echo "激活LLaMA-Factory虚拟环境..."
# 根据您的环境配置修改下面的路径和名称
# Anaconda环境
source /home/lhc/anaconda3/bin/activate llama_factory
# 或者其他虚拟环境
# source /path/to/your/venv/bin/activate

# 设置环境变量以避免线程冲突
export MKL_THREADING_LAYER=GNU

echo "启动训练，使用balanced平衡策略和sqrt_inverse权重计算方法..."
echo "使用训练脚本: ${TRAIN_SCRIPT}"
echo "训练日志将保存到: ${LOG_FILE} 并同时显示在终端"

# 直接使用绝对路径执行脚本，使用tee同时输出到终端和日志文件
python "${TRAIN_SCRIPT}" \
  --model_name "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --train_dataset "${TRAIN_DATASET}" \
  --test_dataset "${TEST_DATASET}" \
  --sampling_strategy "balanced" \
  --balance_alpha 0.5 \
  --class_weight_method "sqrt_inverse" \
  --base_output_dir "${RESULTS_DIR}" \
  --num_epochs 1.0 \
  --learning_rate 5e-05 \
  --train_batch_size 1 \
  --grad_accum_steps 4 \
  --lora_rank 8 \
  --save_steps 3000 \
  --test_interval 3000 2>&1 | tee "${LOG_FILE}"

EXITCODE=${PIPESTATUS[0]}  # 获取python命令的退出码，而不是tee的退出码
if [ $EXITCODE -eq 0 ]; then
  echo "训练成功完成！结果保存在 ${RESULTS_DIR}"
else
  echo "训练过程中出现错误(退出码: $EXITCODE)，日志已保存到: ${LOG_FILE}"
fi 