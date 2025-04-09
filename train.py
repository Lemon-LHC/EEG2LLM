#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大语言模型微调训练脚本，支持数据集平衡功能
"""

import os
# 设置MKL线程层为GNU，解决与libgomp的兼容性问题
os.environ["MKL_THREADING_LAYER"] = "GNU"

# 确保numpy优先导入
import numpy as np

import sys
import argparse
import subprocess
from dataset_balancer import balance_dataset, update_dataset_config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练和评估大语言模型")
    
    # 模型和数据集参数
    parser.add_argument("--model_name", type=str, default="/data/lhc/models/Llama-3.2-1B-Instruct", 
                        help="模型路径")
    parser.add_argument("--dataset_dir", type=str, default="/data/lhc/datasets_new/sleep", 
                        help="数据集目录")
    parser.add_argument("--train_dataset", type=str, default="edf197_100hz_10000ms_tok8521_train", 
                        help="训练数据集名称或相对于dataset_dir的路径")
    parser.add_argument("--test_dataset", type=str, default="edf197_100hz_10000ms_tok8521_test", 
                        help="测试数据集名称或相对于dataset_dir的路径")
    
    # 训练参数
    parser.add_argument("--cutoff_len", type=int, default=8600, 
                        help="序列截断长度")
    parser.add_argument("--learning_rate", type=float, default=5e-05, 
                        help="学习率")
    parser.add_argument("--num_epochs", type=float, default=1.0, 
                        help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, default=1, 
                        help="训练批次大小")
    parser.add_argument("--grad_accum_steps", type=int, default=4, 
                        help="梯度累积步数")
    parser.add_argument("--warmup_steps", type=int, default=50, 
                        help="预热步数")
    parser.add_argument("--lora_rank", type=int, default=8, 
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout")
    parser.add_argument("--save_steps", type=int, default=3000, 
                        help="保存检查点的步数间隔")
    parser.add_argument("--test_interval", type=int, default=3000, 
                        help="测试集评估的步数间隔")
    parser.add_argument("--logging_steps", type=int, default=5, 
                        help="日志记录的步数间隔")
    
    # 数据平衡参数
    parser.add_argument("--sampling_strategy", type=str, default="original", 
                        choices=["original", "balanced", "weighted"],
                        help="数据采样策略: original=原始分布, balanced=平衡采样, weighted=加权采样")
    parser.add_argument("--balance_alpha", type=float, default=0.3, 
                        help="平衡系数，值越大对少数类的过采样程度越高，范围[0-1]")
    parser.add_argument("--class_weight_method", type=str, default="inverse", 
                        choices=["none", "inverse", "sqrt_inverse", "effective_samples"],
                        help="类别权重计算方法: none=不使用, inverse=反比例权重, sqrt_inverse=反比例平方根, effective_samples=有效样本数")
    
    # 输出目录
    parser.add_argument("--base_output_dir", type=str, default="/data/lhc/results", 
                        help="基础输出目录")
    parser.add_argument("--export_dir", type=str, default=None, 
                        help="导出合并模型的目录，默认为/data/lhc/models_new/{model}_{dataset}")
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 构建数据路径
    train_data_path = os.path.join(args.dataset_dir, "train", f"{args.train_dataset}.json")
    test_data_path = os.path.join(args.dataset_dir, "test", f"{args.test_dataset}.json")
    
    # 数据集平衡处理
    if args.sampling_strategy != "original":
        print(f"\n应用数据集平衡策略: {args.sampling_strategy}")
        
        # 创建平衡数据集目录
        balanced_dataset_dir = os.path.join(args.dataset_dir, "balanced")
        os.makedirs(balanced_dataset_dir, exist_ok=True)
        
        # 定义平衡后的数据集文件名
        train_dataset_name = os.path.basename(args.train_dataset).split('.')[0]
        balanced_train_dataset_name = f"{train_dataset_name}_{args.sampling_strategy}"
        balanced_train_data_path = os.path.join(balanced_dataset_dir, f"{balanced_train_dataset_name}.json")
        
        # 处理训练数据集
        balance_success = balance_dataset(
            input_file=train_data_path,
            output_file=balanced_train_data_path,
            strategy=args.sampling_strategy,
            balance_alpha=args.balance_alpha,
            weight_method=args.class_weight_method
        )
        
        if balance_success:
            print(f"使用平衡后的训练数据集: {balanced_train_dataset_name}")
            # 更新数据集名称和路径
            args.train_dataset = balanced_train_dataset_name
            train_data_path = balanced_train_data_path
            
            # 更新数据集配置
            dataset_info_path = "/data/lhc/projects/LLaMA-Factory/data/dataset_info.json"
            update_dataset_config(dataset_info_path, balanced_train_dataset_name, balanced_train_data_path)
        else:
            print(f"数据集平衡处理失败，将使用原始数据集")
    
    # 构造完整的train_old.py命令，将数据平衡的结果传递给原始训练脚本
    train_cmd = [
        "python", "train_old.py",
        "--model_name", args.model_name,
        "--train_dataset", args.train_dataset,
        "--test_dataset", args.test_dataset,
        "--cutoff_len", str(args.cutoff_len),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--train_batch_size", str(args.train_batch_size),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--save_steps", str(args.save_steps),
        "--test_interval", str(args.test_interval),
        "--logging_steps", str(args.logging_steps),
        "--base_output_dir", args.base_output_dir
    ]
    
    # 添加导出目录参数（如果指定）
    if args.export_dir:
        train_cmd.extend(["--export_dir", args.export_dir])
    
    # 输出并执行命令
    cmd_str = " ".join(train_cmd)
    print(f"执行训练命令: {cmd_str}")
    
    # 启动训练进程
    result = subprocess.run(train_cmd, check=True)
    
    # 返回训练结果
    return result.returncode == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
