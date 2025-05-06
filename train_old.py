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
import json
import time
import glob
import argparse
import subprocess
import datetime
import shutil
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import gzip
from tqdm import tqdm
import traceback

# 定义睡眠阶段标签，避免重复硬编码
SLEEP_STAGE_LABELS = [
    'Wake (W)', 
    'NREM Stage 1 (N1)', 
    'NREM Stage 2 (N2)', 
    'NREM Stage 3 (N3)', 
    'NREM Stage 4 (N4)', 
    'REM Sleep (R)'
]

def setup_environment():
    """设置训练环境和目录"""
    # 设置环境变量以避免内存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 获取可用的GPU数量
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"可用GPU数量: {gpu_count}")
    
    return gpu_count

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练和评估大语言模型")
    
    # 模型和数据集参数
    parser.add_argument("--model_name", type=str, default="/data/lhc/models/Llama-3.2-1B-Instruct", 
                        help="模型路径")
    parser.add_argument("--dataset_dir", type=str, default="/data/lhc/datasets_new/sleep", 
                        help="数据集目录")
    parser.add_argument("--train_dataset", type=str, default="/balanced/edf197_200hz_10000ms_tok16588_balanced_0.7_sqrt_inverse_train.json", 
                        help="训练数据集名称或相对于dataset_dir的路径")
    parser.add_argument("--test_dataset", type=str, default="/balanced/edf197_200hz_10000ms_tok16588_balanced_0.7_sqrt_inverse_test.json", 
                        help="测试数据集名称或相对于dataset_dir的路径")
    
    # 训练参数
    parser.add_argument("--cutoff_len", type=int, default=16588, 
                        help="序列截断长度")
    parser.add_argument("--learning_rate", type=float, default=5e-05, 
                        help="学习率")
    parser.add_argument("--num_epochs", type=float, default=3.0, 
                        help="训练轮数")
    parser.add_argument("--max_samples", type=int, default=40000, 
                        help="最大样本数")
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
    parser.add_argument("--val_size", type=float, default=0.11, 
                        help="验证集比例")
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
                        help="导出合并模型的目录，默认为/data/lhc/models_new/{model}_{dataset}，设置为'none'可禁用导出")
    
    # 添加 tensorboard 相关参数
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                       help="TensorBoard日志目录")
    
    args = parser.parse_args()
    return args

def setup_directories(args):
    """设置输出目录结构"""
    # 从模型名称中提取短名称
    model_short_name = os.path.basename(args.model_name.rstrip('/'))
    
    # 从数据集名称中提取信息
    dataset_info = os.path.basename(args.train_dataset).split('.')[0]
    result_prefix = f"{model_short_name}_{dataset_info}"
    
    # 生成包含模型和数据集信息的输出目录名
    # 使用绝对路径，确保输出到/data/lhc/saves/目录
    output_dir = f"/data/lhc/saves/{model_short_name}/lora/{dataset_info}"
    
    # 直接使用提供的路径构建完整数据集路径
    train_data_path = os.path.join(args.dataset_dir, f"{args.train_dataset}.json")
    test_data_path = os.path.join(args.dataset_dir, f"{args.test_dataset}.json")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 仅在实际需要使用时创建此目录，避免创建空目录
    eval_output_dir = f"{args.base_output_dir}/{result_prefix}_{timestamp}"
    
    # 直接使用LLaMA-Factory生成的runs目录作为主要TensorBoard日志目录
    tensorboard_base_dir = f"{output_dir}/runs"
    # 这些目录作为备用，不过实际训练日志会写入runs目录
    tensorboard_train_dir = f"{output_dir}/runs"
    tensorboard_test_dir = f"{output_dir}/runs"
    
    # 合并后的模型输出目录
    if args.export_dir is None:
        # 不会立即创建目录，只是指定默认路径，在最后执行导出时才创建
        # 使用传入的train_dataset参数而非硬编码值
        export_dir = f"/data/lhc/models_new/{model_short_name}_{dataset_info}"
    else:
        export_dir = args.export_dir
    
    # 仅创建实际需要的目录
    # 评估目录将在实际需要时创建，不要在这里创建
    os.makedirs(tensorboard_base_dir, exist_ok=True)
    
    directories = {
        "result_prefix": result_prefix,
        "output_dir": output_dir,
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "eval_output_dir": eval_output_dir,
        "tensorboard_base_dir": tensorboard_base_dir,
        "tensorboard_train_dir": tensorboard_train_dir,
        "tensorboard_test_dir": tensorboard_test_dir,
        "export_dir": export_dir
    }
    
    print(f"输出目录结构:")
    print(f"  - 训练输出: {output_dir}")
    print(f"  - 评估输出(仅在需要时创建): {eval_output_dir}")
    print(f"  - TensorBoard: {tensorboard_base_dir}")
    print(f"  - 合并模型输出(仅在导出时创建): {export_dir}")
    
    return directories

def update_dataset_info(args, directories):
    """更新LLaMA-Factory的数据集配置文件
    
    将当前使用的训练和测试数据集信息添加到LLaMA-Factory的dataset_info.json文件中，
    确保每次训练时都能正确识别数据集。
    
    Args:
        args: 命令行参数
        directories: 输出目录结构
    """
    dataset_info_path = "/data/lhc/projects/LLaMA-Factory/data/dataset_info.json"
    print(f"更新数据集配置: {dataset_info_path}")
    
    # 备份原始文件
    backup_path = f"{dataset_info_path}.bak"
    try:
        shutil.copy2(dataset_info_path, backup_path)
        print(f"已创建配置文件备份: {backup_path}")
    except Exception as e:
        print(f"创建备份文件时出错: {e}")
    
    try:
        # 读取现有配置
        with open(dataset_info_path, 'r') as f:
            data = json.load(f)
        
        # 提取数据集名称（不含扩展名和路径）
        train_dataset_name = os.path.basename(args.train_dataset).split('.')[0]
        test_dataset_name = os.path.basename(args.test_dataset).split('.')[0]
        
        # 添加或更新训练和测试数据集定义
        data[train_dataset_name] = {
            'file_name': directories['train_data_path'],
            'columns': {
                'prompt': 'instruction',
                'query': 'input',
                'response': 'output'
            }
        }
        
        data[test_dataset_name] = {
            'file_name': directories['test_data_path'],
            'columns': {
                'prompt': 'instruction',
                'query': 'input',
                'response': 'output'
            }
        }
        
        # 写入更新后的配置
        with open(dataset_info_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"成功更新数据集配置:")  
        print(f"  - 添加训练数据集: {train_dataset_name}")
        print(f"  - 添加测试数据集: {test_dataset_name}")
    except Exception as e:
        print(f"更新数据集配置时出错: {e}")
        # 如果更新失败，尝试恢复备份
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, dataset_info_path)
                print(f"已恢复配置文件备份")
            except Exception as e2:
                print(f"恢复备份失败: {e2}")

def check_adapter_configs(directories):
    """检查并修复所有检查点的adapter_config.json文件，确保不使用models_new路径
    
    Args:
        directories: 输出目录结构，包含output_dir等
    """
    print("检查adapter_config.json文件中的基础模型路径...")
    
    # 查找所有的adapter_config.json文件
    output_dir = directories["output_dir"]
    adapter_configs = []
    
    # 检查主目录的adapter_config.json
    main_config = os.path.join(output_dir, "adapter_config.json")
    if os.path.isfile(main_config):
        adapter_configs.append(main_config)
    
    # 检查所有检查点目录
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for cp_dir in checkpoint_dirs:
        cp_config = os.path.join(cp_dir, "adapter_config.json")
        if os.path.isfile(cp_config):
            adapter_configs.append(cp_config)
    
    # 获取当前使用的基础模型路径，作为替换models_new的安全路径
    safe_model_path = args.model_name
    
    # 修复所有包含models_new路径的配置
    fixed_count = 0
    for config_path in adapter_configs:
        try:
            # 读取配置
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 检查是否包含models_new路径
            if "base_model_name_or_path" in config and "/data/lhc/models_new/" in config["base_model_name_or_path"]:
                # 备份原始配置
                backup_path = config_path + ".bak"
                with open(backup_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # 创建修复后的配置
                fixed_config = config.copy()
                original_path = fixed_config["base_model_name_or_path"]
                
                # 使用当前指定的模型路径替代，而不是硬编码值
                fixed_config["base_model_name_or_path"] = safe_model_path
                
                # 验证替代路径是否存在
                if not os.path.exists(fixed_config["base_model_name_or_path"]):
                    # 尝试其他备用路径
                    for path in ["/data/lhc/models/Llama-3.2-1B-Instruct", "/data/lhc/models/Llama-3.1-8B-Instruct", "/data/lhc/models/Llama-2-7B-Chat-GPTQ"]:
                        if os.path.exists(path):
                            fixed_config["base_model_name_or_path"] = path
                            break
                
                # 写入修复后的配置
                with open(config_path, 'w') as f:
                    json.dump(fixed_config, f, indent=2)
                
                print(f"已修复配置文件: {config_path}")
                print(f"  - 原始路径: {original_path}")
                print(f"  - 修复路径: {fixed_config['base_model_name_or_path']}")
                fixed_count += 1
        except Exception as e:
            print(f"检查或修复配置文件 {config_path} 时出错: {e}")
    
    if fixed_count > 0:
        print(f"共修复了 {fixed_count} 个adapter_config.json文件")
    else:
        print("未发现需要修复的adapter_config.json文件")
    
    return fixed_count

def balance_dataset(input_file, output_file, strategy, balance_alpha=0.3, weight_method="inverse"):
    """平衡数据集
    
    根据指定的策略对数据集进行平衡处理：
    - original: 保持原始分布不变
    - balanced: 对少数类进行过采样，使得各类别样本数量更平衡
    - weighted: 为每个样本分配权重，反映在训练中的重要性
    
    Args:
        input_file: 输入数据集文件路径
        output_file: 输出数据集文件路径
        strategy: 采样策略 ("original", "balanced", "weighted")
        balance_alpha: 平衡系数，值越大对少数类的过采样程度越高，范围[0-1]
        weight_method: 类别权重计算方法
        
    Returns:
        bool: 处理是否成功
    """
    print(f"开始数据集平衡处理...")
    print(f"策略: {strategy}, 平衡系数: {balance_alpha}, 权重方法: {weight_method}")
    
    try:
        # 读取原始数据集
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是原始策略，直接返回
        if strategy == "original":
            print(f"使用原始分布，不进行平衡处理")
            return True
        
        # 统计各类别样本数量
        class_counts = {}
        for item in data:
            # 提取样本的类别（通常在输出字段中）
            output = item.get('output', '')
            # 在睡眠分期任务中，输出通常是睡眠阶段，如"Wake (W)"
            # 使用第一个出现的睡眠阶段标签作为类别
            label = None
            for stage in SLEEP_STAGE_LABELS:
                if stage in output:
                    label = stage
                    break
            
            if label:
                class_counts[label] = class_counts.get(label, 0) + 1
        
        # 如果无法识别类别，则无法进行平衡
        if not class_counts:
            print(f"警告: 无法识别数据集中的类别标签，保持原始数据不变")
            return True
        
        print(f"数据集类别分布:")
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}样本")
        
        # 计算类别权重
        class_weights = {}
        if weight_method == "none":
            # 所有类别权重相等
            for cls in class_counts:
                class_weights[cls] = 1.0
        elif weight_method == "inverse":
            # 反比例权重
            max_count = max(class_counts.values())
            for cls, count in class_counts.items():
                class_weights[cls] = max_count / count if count > 0 else 1.0
        elif weight_method == "sqrt_inverse":
            # 反比例平方根权重 - 更温和的平衡
            max_count = max(class_counts.values())
            for cls, count in class_counts.items():
                class_weights[cls] = np.sqrt(max_count / count) if count > 0 else 1.0
        elif weight_method == "effective_samples":
            # 有效样本数方法
            beta = balance_alpha
            for cls, count in class_counts.items():
                if count > 0:
                    class_weights[cls] = (1 - beta) / (1 - beta ** count)
                else:
                    class_weights[cls] = 1.0
        
        print(f"计算的类别权重:")
        for cls, weight in class_weights.items():
            print(f"  - {cls}: {weight:.4f}")
        
        # 根据策略处理数据集
        if strategy == "balanced":
            # 对各类别进行过采样或欠采样
            balanced_data = []
            # 计算目标样本数
            median_count = sorted(class_counts.values())[len(class_counts) // 2]  # 中位数
            max_count = max(class_counts.values())
            
            # 根据平衡系数计算每个类别的目标样本数
            # alpha=0意味着所有类别样本数等于中位数
            # alpha=1意味着所有类别样本数等于最大值
            target_counts = {}
            for cls, count in class_counts.items():
                target = int(median_count + balance_alpha * (max_count - median_count))
                target_counts[cls] = max(target, count)  # 确保不减少样本
            
            print(f"目标类别分布:")
            for cls, target in target_counts.items():
                print(f"  - {cls}: {target}样本 (当前: {class_counts[cls]})")
            
            # 创建每个类别的样本列表
            class_samples = {cls: [] for cls in class_counts}
            for item in data:
                output = item.get('output', '')
                label = None
                for stage in SLEEP_STAGE_LABELS:
                    if stage in output:
                        label = stage
                        break
                
                if label:
                    class_samples[label].append(item)
            
            # 进行过采样
            for cls, samples in class_samples.items():
                current_count = len(samples)
                target_count = target_counts[cls]
                
                if current_count == 0:
                    continue
                
                # 添加所有原始样本
                balanced_data.extend(samples)
                
                # 如果需要增加样本
                if target_count > current_count:
                    # 计算需要额外添加的样本数
                    extra_count = target_count - current_count
                    # 随机抽样添加
                    for _ in range(extra_count):
                        # 随机选择一个样本复制
                        idx = np.random.randint(0, current_count)
                        balanced_data.append(samples[idx])
            
            # 打乱数据集
            np.random.shuffle(balanced_data)
            
            print(f"平衡后的数据集大小: {len(balanced_data)} 样本")
            
            # 写入平衡后的数据集
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(balanced_data, f, ensure_ascii=False, indent=2)
                
        elif strategy == "weighted":
            # 为每个样本添加权重字段
            weighted_data = []
            for item in data:
                output = item.get('output', '')
                label = None
                for stage in SLEEP_STAGE_LABELS:
                    if stage in output:
                        label = stage
                        break
                
                # 创建带权重的样本
                weighted_item = item.copy()
                if label:
                    weighted_item['weight'] = class_weights.get(label, 1.0)
                else:
                    weighted_item['weight'] = 1.0
                
                weighted_data.append(weighted_item)
            
            print(f"添加权重后的数据集大小: {len(weighted_data)} 样本")
            
            # 写入带权重的数据集
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(weighted_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集平衡处理完成，输出到: {output_file}")
        return True
    except Exception as e:
        print(f"数据集平衡处理失败: {e}")
        return False

def start_tensorboard(tensorboard_dirs, eval_output_dir):
    """启动TensorBoard服务
    
    Args:
        tensorboard_dirs: 字典，包含多个日志目录，格式为 {名称: 路径}
        eval_output_dir: 评估输出目录，用于保存TensorBoard日志
    """
    print("启动TensorBoard服务...")
    
    # 优先使用当前训练的特定日志目录
    # 根据实际测试，直接指向特定日志目录能更可靠地显示数据
    specific_log_dir = None
    
    # 1. 首先查找当前训练的runs目录
    if "train" in tensorboard_dirs and os.path.exists(tensorboard_dirs["train"]):
        specific_log_dir = tensorboard_dirs["train"]
        print(f"使用当前训练的runs目录: {specific_log_dir}")
    
    # 2. 如果未找到当前训练目录，尝试使用已知的常用日志目录
    if not specific_log_dir:
        sota_dir = "/data/lhc/saves/sota_llama_edf200_100hz_10000ms_train/lora/edf5_100hz_10000ms_tok8363_train/runs"
        if os.path.exists(sota_dir):
            specific_log_dir = sota_dir
            print(f"使用已知的sota_llama训练日志目录: {specific_log_dir}")
    
    # 3. 如果仍然未找到，使用通用目录
    if not specific_log_dir:
        for general_dir in ["/data/lhc/saves", "/data/lhc/projects/LLaMA-Factory/saves"]:
            if os.path.exists(general_dir):
                specific_log_dir = general_dir
                print(f"使用通用目录: {specific_log_dir}")
                break
    
    # 4. 最后创建默认目录
    if not specific_log_dir:
        specific_log_dir = "/data/lhc/saves"
        os.makedirs(specific_log_dir, exist_ok=True)
        print(f"创建默认日志目录: {specific_log_dir}")
        
    # 检查6006端口是否已被占用
    try:
        # 检查端口是否占用
        port_check = subprocess.run(
            ["lsof", "-i", ":6006"], 
            capture_output=True, 
            text=True
        )
        
        if port_check.stdout:
            print("发现已有进程占用 TensorBoard 端口 (6006)，关闭该进程...")
            # 仅关闭端口6006的进程
            kill_cmd = "kill -9 $(lsof -t -i:6006)"
            subprocess.run(kill_cmd, shell=True, check=False)
            time.sleep(1)
    except Exception as e:
        print(f"检查TensorBoard端口占用时出错: {e}")
    
    # 启动新的TensorBoard进程
    try:
        # 确保在需要时创建评估目录，避免创建空目录
        if not os.path.exists(os.path.dirname(eval_output_dir)):
            os.makedirs(os.path.dirname(eval_output_dir), exist_ok=True)
            
        # 创建TensorBoard日志文件所在目录
        log_dir = os.path.dirname(eval_output_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/tensorboard.log"
        
        # 使用精简的命令行参数，直接指向特定目录
        # 此方法已被证明更可靠地显示数据
        cmd = [
            "tensorboard", 
            "--logdir", specific_log_dir,
            "--bind_all", 
            "--port", "6006", 
            "--reload_interval", "1"
        ]
        
        print(f"执行精简版TensorBoard命令: {' '.join(cmd)}")
        
        # 确保日志文件的父目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT
        )
        
        # 获取主机IP地址
        hostname_process = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
        ip_address = hostname_process.stdout.strip().split()[0]
        
        print(f"TensorBoard服务已启动，请访问 http://{ip_address}:6006 查看训练和评估指标")
        print(f"实际监控的日志目录: {specific_log_dir}")
            
        # 打印提示信息
        print(f"注意: 如果TensorBoard仍然没有显示数据，请尝试手动访问以下网址:")
        print(f"  - http://{ip_address}:6006/#scalars&_smoothingWeight=0")
        print(f"  - http://{ip_address}:6006/#text&_smoothingWeight=0")
        
        return process
    except Exception as e:
        print(f"启动TensorBoard时出错: {e}")
        return None

def test_checkpoint(checkpoint_dir, test_data_path, eval_output_dir, tensorboard_test_dir):
    """评估检查点性能"""
    checkpoint_name = os.path.basename(checkpoint_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"开始评估检查点: {checkpoint_name} (时间戳: {timestamp})...")
    
    # 确保评估输出目录存在
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)
        print(f"创建评估输出目录: {eval_output_dir}")
    
    # 创建检查点特定的评估结果目录
    test_eval_dir = f"{eval_output_dir}/test_{checkpoint_name}_{timestamp}"
    os.makedirs(test_eval_dir, exist_ok=True)
    
    # 检查checkpoint_dir是否存在adapter_config.json文件
    adapter_config_path = f"{checkpoint_dir}/adapter_config.json"
    if not os.path.isfile(adapter_config_path):
        print(f"警告: {checkpoint_dir} 中没有找到adapter_config.json文件")
        # 尝试使用整个输出目录
        parent_dir = os.path.dirname(checkpoint_dir)
        adapter_config_path = f"{parent_dir}/adapter_config.json"
        if os.path.isfile(adapter_config_path):
            print(f"使用 {parent_dir} 作为checkpoint目录进行评估")
            checkpoint_dir = parent_dir
        else:
            print(f"错误: 无法找到有效的adapter_config.json文件，跳过此次评估")
            return False
    else:
        adapter_config_path = f"{checkpoint_dir}/adapter_config.json"
    
    # 对adapter_config.json进行临时修改，防止使用models_new路径
    original_config = None
    if os.path.exists(adapter_config_path):
        try:
            # 读取原始配置
            with open(adapter_config_path, 'r') as f:
                original_config = json.load(f)
            
            # 检查是否包含models_new路径
            if "base_model_name_or_path" in original_config and "/data/lhc/models_new/" in original_config["base_model_name_or_path"]:
                # 创建临时配置，替换models_new路径
                temp_config = original_config.copy()
                # 使用当前模型路径替代，而不是硬编码值
                temp_config["base_model_name_or_path"] = args.model_name
                
                # 验证替代路径是否存在
                if not os.path.exists(temp_config["base_model_name_or_path"]):
                    # 尝试其他备用路径
                    for path in ["/data/lhc/models/Llama-3.2-1B-Instruct", "/data/lhc/models/Llama-3.1-8B-Instruct", "/data/lhc/models/Llama-2-7B-Chat-GPTQ"]:
                        if os.path.exists(path):
                            temp_config["base_model_name_or_path"] = path
                            break
                
                print(f"临时替换adapter_config.json中的路径: '{original_config['base_model_name_or_path']}' -> '{temp_config['base_model_name_or_path']}'")
                
                # 写入临时配置
                with open(adapter_config_path, 'w') as f:
                    json.dump(temp_config, f, indent=2)
        except Exception as e:
            print(f"修改adapter_config.json时出错: {e}")
    
    # 使用eval_checkpoint.py进行评估
    try:
        # 获取eval_checkpoint.py脚本的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        eval_script_path = os.path.join(script_dir, "eval_checkpoint.py")
        
        cmd = [
            "python", eval_script_path,
            "--checkpoint_dir", checkpoint_dir,
            "--test_data", test_data_path,
            "--output_dir", test_eval_dir,
            "--tensorboard_dir", tensorboard_test_dir,
            "--device", "cuda",
            "--template", "alpaca",
            "--calculate_metrics", "True",  # 添加计算指标的参数
            "--verbose", "True"  # 添加详细输出参数
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # 恢复原始配置
        if original_config and "base_model_name_or_path" in original_config and "/data/lhc/models_new/" in original_config["base_model_name_or_path"]:
            try:
                with open(adapter_config_path, 'w') as f:
                    json.dump(original_config, f, indent=2)
                print("已恢复原始adapter_config.json")
            except Exception as e:
                print(f"恢复原始adapter_config.json时出错: {e}")
        
        # 解析评估结果中的指标
        metrics_file = os.path.join(test_eval_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                
            # 打印每个睡眠阶段的准确率
            print("\n每个睡眠阶段的准确率:")
            if 'class_accuracies' in metrics:
                for stage_idx, accuracy in metrics['class_accuracies'].items():
                    stage_name = SLEEP_STAGE_LABELS[int(stage_idx)]
                    print(f"  - {stage_name}: 准确率 {accuracy:.4f}")
            
            # 打印整体指标
            print("\n整体指标:")
            if 'accuracy' in metrics:
                print(f"  - 整体准确率: {metrics['accuracy']:.4f}")
            if 'f1_macro' in metrics:
                print(f"  - 整体F1分数: {metrics['f1_macro']:.4f}")
            if 'avg_inference_time' in metrics:
                print(f"  - 平均推理时间: {metrics['avg_inference_time']:.4f} 秒")
            
            # 将指标写入TensorBoard
            # 直接使用检查点所在目录中的runs目录作为TensorBoard日志目录
            # 先提取检查点路径的基本结构
            checkpoint_base = os.path.dirname(checkpoint_dir)
            if "/checkpoint-" in checkpoint_base:
                # 如果是常规检查点目录，往上一级
                checkpoint_base = os.path.dirname(checkpoint_base)
            
            # 构建日志目录
            actual_tensorboard_dir = os.path.join(checkpoint_base, "runs")
            
            # 如果这个目录不存在，创建它
            if not os.path.exists(actual_tensorboard_dir):
                os.makedirs(actual_tensorboard_dir, exist_ok=True)
                
            print(f"测试集指标将写入到: {actual_tensorboard_dir}")
            
            writer = SummaryWriter(actual_tensorboard_dir)
            step = int(checkpoint_name.split('-')[-1]) if checkpoint_name.startswith('checkpoint-') else 0
            
            # 写入整体指标
            if 'accuracy' in metrics:
                writer.add_scalar('test/accuracy', metrics['accuracy'], step)
            if 'f1_macro' in metrics:
                writer.add_scalar('test/f1_macro', metrics['f1_macro'], step)
            if 'avg_inference_time' in metrics:
                writer.add_scalar('test/avg_inference_time', metrics['avg_inference_time'], step)
            
            # 写入每个睡眠阶段的准确率
            if 'class_accuracies' in metrics:
                for stage_idx, accuracy in metrics['class_accuracies'].items():
                    writer.add_scalar(f'test/stage_{stage_idx}_accuracy', accuracy, step)
            
            # 如果有混淆矩阵，将其可视化
            if 'confusion_matrix' in metrics:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                plt.figure(figsize=(10, 8))
                cm = np.array(metrics['confusion_matrix'])
                # 使用简化的标签用于混淆矩阵
                short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=short_labels,
                            yticklabels=short_labels)
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                plt.title(f'混淆矩阵 (检查点: {checkpoint_name})')
                
                cm_image_path = os.path.join(test_eval_dir, "confusion_matrix.png")
                plt.savefig(cm_image_path)
                plt.close()
                
                # 将混淆矩阵图添加到TensorBoard
                from PIL import Image
                import io
                import torchvision
                
                img = Image.open(cm_image_path)
                img_tensor = torchvision.transforms.ToTensor()(img)
                writer.add_image('test/confusion_matrix', img_tensor, step)
            
            writer.close()
        
        if result.returncode == 0:
            print(f"检查点 {checkpoint_name} 的测试集评估完成，结果保存在 {test_eval_dir}")
            return True
        else:
            print(f"评估失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"评估过程中出错: {e}")
        return False

def merge_and_export_final_model(checkpoint_dir, export_dir):
    """合并模型并导出为单一权重文件"""
    # 如果export_dir明确设置为'none'，则不执行合并操作
    if export_dir and export_dir.lower() == 'none':
        print("export_dir设置为'none'，跳过模型合并导出")
        return False
    
    # 检查路径是否包含特定敏感字符串（避免创建不必要的目录）
    if export_dir and "edf200_100hz_15000ms" in export_dir:
        print(f"检测到敏感路径(edf200_100hz_15000ms)，为避免创建不必要目录，跳过模型合并导出")
        return False
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("开始合并最终模型...")
    
    # 检查checkpoint_dir是否存在adapter_config.json文件
    if not os.path.isfile(f"{checkpoint_dir}/adapter_config.json"):
        print(f"警告: {checkpoint_dir} 中没有找到adapter_config.json文件")
        # 尝试使用整个输出目录
        parent_dir = os.path.dirname(checkpoint_dir)
        if os.path.isfile(f"{parent_dir}/adapter_config.json"):
            print(f"使用 {parent_dir} 作为checkpoint目录进行合并")
            checkpoint_dir = parent_dir
        else:
            print(f"错误: 无法找到有效的adapter_config.json文件，无法合并模型")
            return False
    
    # 创建合并模型的输出目录
    final_merged_dir = f"{export_dir}/final_{timestamp}"
    os.makedirs(final_merged_dir, exist_ok=True)
    
    # 合并LoRA权重和原模型
    print(f"合并LoRA权重和原模型到 {final_merged_dir}...")
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        # 使用export_size=1参数确保导出单一权重文件
        merge_cmd = [
            "llamafactory-cli", "export",
            "--model_name_or_path", args.model_name,
            "--adapter_name_or_path", checkpoint_dir,
            "--template", "alpaca",
            "--finetuning_type", "lora",
            "--export_dir", final_merged_dir,
            "--export_size", "1",  # 确保导出单一权重文件
            "--export_device", "auto",
            "--export_legacy_format", "False"
        ]
        
        merge_result = subprocess.run(merge_cmd, env=env, check=True, capture_output=True, text=True)
        print(merge_result.stdout)
        
        if merge_result.returncode != 0:
            print(f"模型合并失败: {merge_result.stderr}")
            return False
        
        print(f"合并完成，模型已导出到: {final_merged_dir}")
        
        # 检查合并目录的内容
        merged_files = os.listdir(final_merged_dir)
        print(f"合并目录内容: {merged_files}")
        
        return True
    
    except Exception as e:
        print(f"合并过程中出错: {e}")
        return False

def load_data_in_batches(file_path, batch_size=1000, max_samples=None, use_compressed=False, cache_dir=None):
    """分批加载数据，支持压缩格式和缓存
    
    Args:
        file_path: 数据文件路径
        batch_size: 每批加载的样本数
        max_samples: 最大加载样本数，None表示加载全部
        use_compressed: 是否使用压缩格式
        cache_dir: 缓存目录路径
    """
    try:
        # 检查缓存
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, os.path.basename(file_path) + ".cache")
            if os.path.exists(cache_file):
                print(f"从缓存加载数据: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        print(f"文件大小: {file_size / 1024 / 1024:.2f}MB")
        
        # 计算合适的线程数
        cpu_count = os.cpu_count()
        thread_count = min(cpu_count, 16)  # 增加最大线程数到16
        print(f"使用 {thread_count} 个线程加载数据")
        
        # 计算每个线程处理的数据量
        samples_per_thread = batch_size // thread_count
        if samples_per_thread < 1:
            samples_per_thread = 1
        
        # 创建线程池
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        
        data_queue = Queue()
        total_samples = 0
        start_time = time.time()
        
        def load_batch(thread_id):
            try:
                local_data = []
                with gzip.open(file_path, 'rt', encoding='utf-8') if use_compressed else open(file_path, 'r', encoding='utf-8') as f:
                    # 跳过其他线程的数据
                    for _ in range(thread_id * samples_per_thread):
                        next(f, None)
                    
                    # 读取本线程的数据
                    for _ in range(samples_per_thread):
                        line = next(f, None)
                        if line is None:
                            break
                        try:
                            sample = json.loads(line)
                            local_data.append(sample)
                        except json.JSONDecodeError:
                            continue
                
                data_queue.put(local_data)
            except Exception as e:
                print(f"线程 {thread_id} 加载数据失败: {e}")
        
        # 启动线程
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(load_batch, i) for i in range(thread_count)]
            
            # 监控内存使用
            import psutil
            process = psutil.Process()
            
            # 收集数据
            all_data = []
            with tqdm(total=batch_size, desc="加载数据") as pbar:
                while len(all_data) < batch_size and (max_samples is None or total_samples < max_samples):
                    try:
                        batch_data = data_queue.get(timeout=5)
                        all_data.extend(batch_data)
                        total_samples += len(batch_data)
                        pbar.update(len(batch_data))
                        
                        # 监控内存使用
                        mem_info = process.memory_info()
                        if mem_info.rss > 20 * 1024 * 1024 * 1024:  # 增加内存警告阈值到20GB
                            print(f"\n内存使用警告: {mem_info.rss / 1024 / 1024 / 1024:.2f}GB")
                            # 如果内存使用过高，减少线程数
                            if thread_count > 4:  # 保持最小线程数为4
                                thread_count -= 2  # 每次减少2个线程
                                print(f"减少线程数到 {thread_count}")
                    except Exception as e:
                        print(f"获取数据失败: {e}")
                        break
        
        # 如果指定了最大样本数，随机选择样本
        if max_samples is not None and len(all_data) > max_samples:
            import random
            all_data = random.sample(all_data, max_samples)
        
        # 保存到缓存
        if cache_dir and all_data:
            print(f"保存数据到缓存: {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f)
        
        end_time = time.time()
        print(f"\n数据加载完成:")
        print(f"- 总样本数: {len(all_data)}")
        print(f"- 加载时间: {end_time - start_time:.2f}秒")
        print(f"- 平均加载速度: {len(all_data) / (end_time - start_time):.2f}样本/秒")
        
        return all_data
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def train_model(args, directories, gpu_count):
    """启动模型训练
    
    Args:
        args: 命令行参数
        directories: 输出目录结构
        gpu_count: 可用GPU数量
    """
    print("\n开始训练模型...")
    
    # 更新LLaMA-Factory的数据集配置文件
    update_dataset_info(args, directories)
    
    # 提取训练和测试数据集名称（不含路径和扩展名）
    train_dataset_name = os.path.basename(args.train_dataset).split('.')[0]
    test_dataset_name = os.path.basename(args.test_dataset).split('.')[0]
    
    # 设置环境变量以避免内存碎片和确保分布式训练正确关闭
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 设置NCCL超时时间，防止分布式训练卡住
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # 设置 TensorBoard writer
    if args.tensorboard_dir:
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
    
    # 构建训练命令
    cmd = [
        "torchrun", "--nnodes", "1", f"--nproc_per_node={gpu_count}", "--master_port", "29500",
        "/data/lhc/projects/LLaMA-Factory/src/llamafactory/launcher.py",
        "--model_name_or_path", args.model_name,
        "--dataset", train_dataset_name,
        "--val_size", "0.1",
        "--dataset_dir", "/data/lhc/projects/LLaMA-Factory/data",
        "--output_dir", directories["output_dir"],
        "--stage", "sft",
        "--do_train", "True",
        "--cutoff_len", str(args.cutoff_len),
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_epochs),
        "--per_device_train_batch_size", str(args.train_batch_size),
        "--logging_steps", str(args.logging_steps),
        "--save_steps", str(args.save_steps),
        "--warmup_steps", str(args.warmup_steps),
        "--packing", "False",
        "--report_to", "tensorboard",
        "--gradient_checkpointing", "True",
        "--bf16", "True",
        "--plot_loss", "True",
        "--trust_remote_code", "True",
        "--ddp_timeout", "180000000",
        "--include_num_input_tokens_seen", "True",
        "--optim", "adamw_torch",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--lora_target", "q_proj,k_proj,v_proj,o_proj",
        "--eval_strategy", "steps",
        "--eval_steps", str(args.test_interval),
        "--per_device_eval_batch_size", str(args.train_batch_size),
        "--gradient_checkpointing", "True",
        "--ddp_find_unused_parameters", "False",
        "--overwrite_output_dir", "True",
        "--template", "alpaca"
    ]
    
    print(f"\n训练命令: {' '.join(cmd)}")
    print("\n开始训练...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 启动训练进程
    train_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 初始化变量来跟踪检查点
    last_checkpoint_step = 0
    last_test_step = 0
    test_interval = args.test_interval  # 使用参数中设置的测试间隔
    
    # 实时输出训练日志并检测检查点
    for line in train_process.stdout:
        print(line, end='')
        
        # 检测新的检查点
        if "Saving model checkpoint" in line:
            try:
                # 先获取检查点路径
                checkpoint_path = line.strip().split("Saving model checkpoint to ")[-1].strip()
                
                # 如果检查点路径中包含"checkpoint-"，尝试提取步数
                if "checkpoint-" in checkpoint_path:
                    # 搜索checkpoint-后面的数字
                    import re
                    checkpoint_match = re.search(r'checkpoint-(\d+)', checkpoint_path)
                    if checkpoint_match:
                        checkpoint_step = int(checkpoint_match.group(1))
                        checkpoint_dir = os.path.join(directories["output_dir"], f"checkpoint-{checkpoint_step}")
                    else:
                        # 如果找不到数字，使用完整路径
                        checkpoint_dir = checkpoint_path
                        checkpoint_step = time.time()  # 使用当前时间作为唯一标识
                else:
                    # 如果路径中没有checkpoint-格式，直接使用该路径
                    checkpoint_dir = checkpoint_path
                    checkpoint_step = time.time()  # 使用当前时间作为唯一标识
                
                # 检查是否应该运行测试集评估
                # 检查是否在训练过程中（而不是训练结束时）
                if "Training completed" not in line and "train_runtime" not in line:
                    # 检查是否达到测试间隔
                    if checkpoint_step >= last_test_step + test_interval:
                        # 确保至少间隔一定时间再进行评估，避免过于频繁的评估
                        current_time = time.time()
                        time_since_last_test = current_time - getattr(train_model, 'last_test_time', 0)
                        
                        # 至少间隔300秒，避免过快进行评估
                        if time_since_last_test > 300:  # 5分钟
                            print(f"\n\n到达测试间隔 ({test_interval} 步)，开始评估测试集...")
                            # 修改调用方式，确保不会创建不必要的目录
                            # 使用checkpoint_dir所在目录中的runs子目录作为tensorboard_dir
                            tb_dir = os.path.join(os.path.dirname(checkpoint_dir), "runs")
                            test_checkpoint(checkpoint_dir, directories["test_data_path"], 
                                          directories["eval_output_dir"], tb_dir)
                            last_test_step = checkpoint_step
                            # 记录本次评估时间
                            train_model.last_test_time = current_time
                        else:
                            print(f"\n距离上次评估时间过短 ({time_since_last_test:.1f} 秒), 跳过本次评估")
                
                last_checkpoint_step = checkpoint_step
            except Exception as e:
                print(f"\n解析检查点信息时出错: {e}")
    
    # 等待训练完成
    return_code = train_process.wait()
    
    # 确保分布式进程组正确关闭
    # 这里我们运行一个简单的脚本来确保分布式环境被正确清理
    cleanup_cmd = [
        "python", "-c", 
        "import torch.distributed as dist\ntry:\n    if dist.is_initialized():\n        dist.destroy_process_group()\n        print('分布式进程组已正确关闭')\nexcept:\n    pass"
    ]
    try:
        subprocess.run(cleanup_cmd, check=False, timeout=10)
        print("已尝试清理分布式训练环境")
    except Exception as e:
        print(f"清理分布式环境时出错: {e}")
    
    # 记录结束时间并计算总训练时间
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 检查训练是否成功
    if return_code != 0 or last_checkpoint_step == 0:
        print(f"\n训练失败或未生成任何检查点! 总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        return False  # 返回失败状态
    
    # 在训练结束后，确保对最后一个检查点进行评估
    if last_checkpoint_step > last_test_step:
        print(f"\n\n训练结束，对最后一个检查点 (checkpoint-{last_checkpoint_step}) 进行评估...")
        # 修改调用方式，确保不会创建不必要的目录
        checkpoint_dir = os.path.join(directories["output_dir"], f"checkpoint-{last_checkpoint_step}")
        tb_dir = os.path.join(os.path.dirname(checkpoint_dir), "runs")
        test_checkpoint(
            checkpoint_dir,
            directories["test_data_path"],
            directories["eval_output_dir"],
            tb_dir
        )
    
    print(f"\n训练完成! 总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 在训练循环中记录指标
    for epoch in range(int(args.num_epochs)):
        # ... 训练代码 ...
        if args.tensorboard_dir:
            writer.add_scalar('Loss/train', train_loss, global_step)
            writer.add_scalar('Accuracy/train', train_accuracy, global_step)
            
        # 验证阶段
        if args.tensorboard_dir:
            writer.add_scalar('Loss/eval', eval_loss, global_step)
            writer.add_scalar('Accuracy/eval', eval_accuracy, global_step)
    
    # 关闭 writer
    if args.tensorboard_dir:
        writer.close()
    
    return True  # 返回成功状态

def main():
    # 设置环境
    gpu_count = setup_environment()
    if gpu_count == 0:
        print("错误: 没有可用的GPU，无法继续")
        return
    
    # 解析参数
    global args
    args = parse_arguments()
    
    # 设置目录
    directories = setup_directories(args)
    
    # 检查并修复已存在的adapter_config.json文件
    check_adapter_configs(directories)
    
    # 启动TensorBoard
    # 检查LLaMA-Factory输出目录中是否存在runs目录
    runs_dir = os.path.join(directories["output_dir"], "runs")
    
    # 初始化TensorBoard目录字典
    tensorboard_dirs = {
        "train": directories["tensorboard_train_dir"],
        "test": directories["tensorboard_test_dir"]
    }
    
    # 添加LLaMA-Factory自动生成的日志目录（如果存在）
    if os.path.exists(runs_dir):
        print(f"检测到LLaMA-Factory生成的日志目录: {runs_dir}")
        tensorboard_dirs["runs"] = runs_dir
    tensorboard_process = start_tensorboard(tensorboard_dirs, directories["eval_output_dir"])
    
    try:
        # 训练模型
        train_success = train_model(args, directories, gpu_count)
        
        if train_success:
            # 训练完成后，合并导出最终模型
            if directories["export_dir"] and directories["export_dir"].lower() == 'none':
                print("\n\n训练完成，export_dir设置为'none'，跳过模型合并导出")
            else:
                print("\n\n训练完成，开始合并导出最终模型...")
                
                # 找到最后一个检查点目录
                last_checkpoint_dirs = glob.glob(os.path.join(directories["output_dir"], "checkpoint-*"))
                if last_checkpoint_dirs:
                    # 按照检查点号排序，选择最后一个
                    last_checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                    last_checkpoint_dir = last_checkpoint_dirs[-1]
                    
                    print(f"使用最终检查点 {last_checkpoint_dir} 合并导出模型...")
                    
                    # 合并模型到单一权重文件
                    merge_success = merge_and_export_final_model(
                        last_checkpoint_dir,
                        directories["export_dir"]
                    )
                    
                    if merge_success:
                        print("模型合并导出成功！最终模型路径: " + directories["export_dir"])
                    else:
                        print("模型合并导出失败！")
                else:
                    print("未找到任何检查点，无法合并导出模型")
        else:
            print("训练过程未成功完成，跳过模型合并")
    
    finally:
        # 保持TensorBoard进程运行，不在程序结束时关闭
        pass

if __name__ == "__main__":
    main()
