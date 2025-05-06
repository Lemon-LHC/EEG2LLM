#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大语言模型微调训练脚本
"""

import os
# 设置MKL线程层为GNU，解决与libgomp的兼容性问题
os.environ["MKL_THREADING_LAYER"] = "GNU"

# 确保numpy优先导入
import numpy as np

import sys
import argparse
import subprocess
import shlex
from dataset_balancer import update_dataset_config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练和评估大语言模型")
    
    # 模型和数据集参数
    parser.add_argument("--model_name", type=str, default="/data/lhc/models/Llama-3.2-1B-Instruct", 
                        help="模型路径")
    parser.add_argument("--dataset_dir", type=str, default="/data/lhc/datasets_new/sleep", 
                        help="数据集目录")
    parser.add_argument("--train_dataset", type=str, 
                        default="/data/lhc/datasets_new/sleep/train/balanced/edf5_200hz_10000ms_tok16588_balanced_0.7_sqrt_inverse_train.json", 
                        help="训练数据集路径")
    parser.add_argument("--test_dataset", type=str, 
                        default="/data/lhc/datasets_new/sleep/test/balanced/edf5_200hz_10000ms_tok16588_balanced_0.7_sqrt_inverse_test.json", 
                        help="测试数据集路径")
    
    # 训练参数
    parser.add_argument("--cutoff_len", type=int, default=16588, 
                        help="序列截断长度")
    parser.add_argument("--learning_rate", type=float, default=5e-05, 
                        help="学习率")
    parser.add_argument("--num_epochs", type=float, default=3.0, 
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
    parser.add_argument("--save_steps", type=int, default=5000, 
                        help="保存检查点的步数间隔")
    parser.add_argument("--test_interval", type=int, default=5000, 
                        help="测试集评估的步数间隔")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="日志记录的步数间隔")
    
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
    
    # 构建数据路径 - 直接使用完整路径
    train_data_path = args.train_dataset
    test_data_path = args.test_dataset

    # 从完整路径中提取数据集名称用于配置更新
    train_dataset_name = os.path.basename(train_data_path).split('.')[0]
    test_dataset_name = os.path.basename(test_data_path).split('.')[0]
    
    # 更新数据集配置
    dataset_info_path = "/data/lhc/projects/LLaMA-Factory/data/dataset_info.json"
    update_dataset_config(dataset_info_path, train_dataset_name, train_data_path)
    update_dataset_config(dataset_info_path, test_dataset_name, test_data_path)
    
    # 确定输出目录
    model_name_short = os.path.basename(args.model_name)
    output_dir = os.path.join(args.base_output_dir, f"{model_name_short}_{train_dataset_name}")
    
    # 构建llamafactory-cli的train命令
    llama_factory_dir = "/data/lhc/projects/LLaMA-Factory"
    
    # 构建基本命令参数
    train_args = [
        "llamafactory-cli", "train",
        f"--model_name_or_path={args.model_name}",
        f"--dataset={train_dataset_name}",
        "--template=llama3",
        f"--finetuning_type=lora",
        f"--output_dir={output_dir}",
        "--overwrite_cache",
        "--overwrite_output_dir",
        f"--cutoff_len={args.cutoff_len}",
        f"--learning_rate={args.learning_rate}",
        f"--num_train_epochs={args.num_epochs}",
        f"--per_device_train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={args.grad_accum_steps}",
        f"--warmup_steps={args.warmup_steps}",
        f"--lora_rank={args.lora_rank}",
        f"--lora_alpha={args.lora_alpha}",
        f"--lora_dropout={args.lora_dropout}",
        f"--logging_steps={args.logging_steps}",
        f"--save_steps={args.save_steps}",
        "--do_train", # 显式指定执行训练
        "--report_to=none" # 防止wandb等报告工具干扰
    ]
    
    # 添加测试集评估
    if test_dataset_name:
        train_args.extend([
            f"--eval_dataset={test_dataset_name}",
            f"--eval_steps={args.test_interval}",
            "--do_eval" # 显式指定执行评估
        ])
    
    # 添加导出目录参数（如果指定）
    if args.export_dir:
        export_dir = args.export_dir if args.export_dir else f"/data/lhc/models_new/{model_name_short}_{train_dataset_name}"
        train_args.append(f"--export_dir={export_dir}")
    
    # 打印将要执行的命令
    print(f"将在目录 {llama_factory_dir} 中执行:")
    print(" ".join(train_args))
    
    try:
        # 使用Popen执行命令并实时输出结果
        process = subprocess.Popen(
            train_args,
            cwd=llama_factory_dir, # 设置工作目录，而不是使用cd命令
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        print("训练开始，输出实时日志...")
        for line in process.stdout:
            print(line, end='')  # 实时打印输出
        
        # 等待进程完成
        returncode = process.wait()
        
        if returncode == 0:
            print("训练成功完成!")
            return True
        else:
            print(f"训练失败，返回代码: {returncode}")
            return False
            
    except Exception as e:
        print(f"执行训练命令时发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
