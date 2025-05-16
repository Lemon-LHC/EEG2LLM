#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集平衡处理模块
提供睡眠阶段数据集的平衡处理功能，包括过采样和加权采样策略
"""

import os
# 设置MKL线程层为GNU，解决与libgomp的兼容性问题
os.environ["MKL_THREADING_LAYER"] = "GNU"

# 确保numpy优先导入
import numpy as np
import json

# 定义睡眠阶段标签，与主训练脚本保持一致
SLEEP_STAGE_LABELS = [
    'Wake (W)', 
    'NREM Stage 1 (N1)', 
    'NREM Stage 2 (N2)', 
    'NREM Stage 3 (N3)', 
    'NREM Stage 4 (N4)', 
    'REM Sleep (R)'
]

def balance_dataset(input_file, output_file, strategy="balanced", balance_alpha=0.3, weight_method="inverse"):
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

def update_dataset_config(dataset_info_path, dataset_name, file_path):
    """更新LLaMA-Factory的数据集配置
    
    将平衡处理后的数据集添加到LLaMA-Factory的配置文件中
    
    Args:
        dataset_info_path: 数据集配置文件路径
        dataset_name: 数据集名称
        file_path: 数据集文件路径
        
    Returns:
        bool: 更新是否成功
    """
    try:
        # 读取现有配置
        with open(dataset_info_path, 'r') as f:
            data = json.load(f)
        
        # 添加或更新数据集定义
        data[dataset_name] = {
            'file_name': file_path,
            'columns': {
                'prompt': 'instruction',
                'query': 'input',
                'response': 'output'
            }
        }
        
        # 写入更新后的配置
        with open(dataset_info_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"已更新数据集配置，添加平衡数据集: {dataset_name}")
        return True
    except Exception as e:
        print(f"更新数据集配置失败: {e}")
        return False

if __name__ == "__main__":
    # 如果直接运行此脚本，提供简单的命令行接口
    import argparse
    
    parser = argparse.ArgumentParser(description="睡眠阶段数据集平衡处理工具")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入数据集文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出数据集文件路径")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["original", "balanced", "weighted"],
                        help="采样策略")
    parser.add_argument("--balance_alpha", type=float, default=0.3,
                        help="平衡系数，范围[0-1]")
    parser.add_argument("--weight_method", type=str, default="inverse",
                        choices=["none", "inverse", "sqrt_inverse", "effective_samples"],
                        help="权重计算方法")
    parser.add_argument("--update_config", action="store_true",
                        help="是否更新LLaMA-Factory的数据集配置")
    parser.add_argument("--config_path", type=str,
                        default="/data/lhc/projects/LLaMA-Factory/data/dataset_info.json",
                        help="LLaMA-Factory数据集配置文件路径")
    
    args = parser.parse_args()
    
    # 执行平衡处理
    success = balance_dataset(
        args.input_file,
        args.output_file,
        args.strategy,
        args.balance_alpha,
        args.weight_method
    )
    
    if success and args.update_config:
        # 提取数据集名称
        dataset_name = os.path.basename(args.output_file).split('.')[0]
        update_dataset_config(args.config_path, dataset_name, args.output_file) 