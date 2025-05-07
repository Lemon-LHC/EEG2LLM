#!/bin/bash

# 睡眠阶段分类模型训练脚本
# 使用推荐的平衡策略和权重计算方法

# 设置关键路径变量
SCRIPT_DIR="/data/lhc/projects/EEG2LLM"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_old.py"
MODEL_NAME="/data/lhc/models/Qwen/Qwen3-0.6B"
DATASET_DIR="/data/lhc/datasets_new/emotion"
TRAIN_DATASET="train/sleep_st_1_100hz_eeg15s-step15s_emo2.0s-step1s_win10_tok13112_bal0.2_sqrt_inverse_202504291924_train"  # 完整的相对路径
TEST_DATASET="train/sleep_st_1_100hz_eeg15s-step15s_emo2.0s-step1s_win10_tok13112_bal0.2_sqrt_inverse_202504291924_train"     # 完整的相对路径

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/data/lhc/results/${TIMESTAMP}_emotion"
mkdir -p $RESULTS_DIR
LOG_FILE="${RESULTS_DIR}/train.log"

# 提取模型短名称和数据集信息，用于构建导出路径
MODEL_SHORT_NAME=$(basename $MODEL_NAME)
DATASET_INFO=$(basename ${TRAIN_DATASET})
# 构建标准化的导出路径
EXPORT_DIR="/data/lhc/models_new/${MODEL_SHORT_NAME}_${DATASET_INFO}"

# echo "激活LLaMA-Factory虚拟环境..."
# 根据您的环境配置修改下面的路径和名称
# Anaconda环境
# source /home/lhc/anaconda3/bin/activate llama_factory
# 或者其他虚拟环境
# source /path/to/your/venv/bin/activate

# 设置环境变量以避免线程冲突
export MKL_THREADING_LAYER=GNU

echo "启动训练，使用balanced平衡策略和sqrt_inverse权重计算方法..."
echo "训练日志将保存到: ${LOG_FILE} 并同时显示在终端"
echo "模型导出路径: ${EXPORT_DIR}"

# 确保脚本使用绝对路径，不依赖工作目录
echo "使用训练脚本: ${TRAIN_SCRIPT}"
echo "切换到工作目录: ${SCRIPT_DIR}"

# 检查训练和测试数据集是否存在
if [ ! -f "${DATASET_DIR}/${TRAIN_DATASET}.json" ]; then
  echo "错误: 训练数据集文件不存在: ${DATASET_DIR}/${TRAIN_DATASET}.json"
  exit 1
fi

if [ ! -f "${DATASET_DIR}/${TEST_DATASET}.json" ]; then
  echo "错误: 测试数据集文件不存在: ${DATASET_DIR}/${TEST_DATASET}.json"
  exit 1
fi

# 切换到脚本所在目录并运行，使用tee同时输出到终端和日志文件
cd "${SCRIPT_DIR}"
python "${TRAIN_SCRIPT}" \
  --model_name $MODEL_NAME \
  --dataset_dir $DATASET_DIR \
  --train_dataset $TRAIN_DATASET \
  --test_dataset $TEST_DATASET \
  --sampling_strategy "original" \
  --balance_alpha 1 \
  --class_weight_method "sqrt_inverse" \
  --base_output_dir $RESULTS_DIR \
  --export_dir $EXPORT_DIR \
  --num_epochs 3 \
  --learning_rate 5e-05 \
  --train_batch_size 1 \
  --grad_accum_steps 8 \
  --lora_rank 4 \
  --save_steps 10000 \
  --cutoff_len 12500 \
  --test_interval 10000 2>&1 | tee "${LOG_FILE}"

EXITCODE=${PIPESTATUS[0]}  # 获取python命令的退出码，而不是tee的退出码
if [ $EXITCODE -eq 0 ]; then
  echo "训练成功完成！结果保存在 $RESULTS_DIR"
else
  echo "训练过程中出现错误(退出码: $EXITCODE)，日志已保存到: ${LOG_FILE}"
fi 