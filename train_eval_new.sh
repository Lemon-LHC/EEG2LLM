#!/bin/bash

# 激活conda环境
conda activate llama_factory

# 切换到fine目录
cd /data/lhc/projects/fine

# 运行Python脚本来管理训练和评估过程
python train_eval_manager.py "$@"
