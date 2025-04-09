#!/usr/bin/env python3
"""
数据集平衡工具
用于平衡睡眠阶段分类数据集中各阶段的样本数量
支持过采样少数类和欠采样多数类
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# 定义睡眠阶段标签
SLEEP_STAGE_LABELS = [
    'Wake (W)', 
    'NREM Stage 1 (N1)', 
    'NREM Stage 2 (N2)', 
    'NREM Stage 3 (N3)', 
    'NREM Stage 4 (N4)', 
    'REM Sleep (R)'
]

# 睡眠阶段对应的简短标签，用于显示
SHORT_LABELS = ['W', 'N1', 'N2', 'N3', 'N4', 'R']

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="平衡睡眠阶段分类数据集")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入数据集文件路径 (.json)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出数据集文件路径 (.json)")
    parser.add_argument("--method", type=str, choices=["oversample", "undersample", "hybrid"],
                        default="hybrid", help="平衡方法: 过采样/欠采样/混合")
    parser.add_argument("--target_ratio", type=float, default=1.0,
                        help="目标类别比例 (相对于平均样本数)")
    parser.add_argument("--rem_boost", type=float, default=1.5,
                        help="REM阶段样本数量的额外提升倍数")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--visualize", action="store_true",
                        help="生成数据分布可视化图表")
    
    args = parser.parse_args()
    return args

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        数据集样本列表
    """
    print(f"加载数据集: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"成功加载 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)

def extract_class_from_output(output: str) -> Optional[int]:
    """从输出文本中提取睡眠阶段类别
    
    Args:
        output: 输出文本，通常包含睡眠阶段标签
        
    Returns:
        睡眠阶段类别索引，如果无法提取则返回None
    """
    # 尝试直接匹配睡眠阶段标签
    for i, label in enumerate(SLEEP_STAGE_LABELS):
        if label in output:
            return i
    
    # 尝试匹配简短标签
    for i, label in enumerate(SHORT_LABELS):
        if f" {label}" in output or f"({label})" in output or output.strip() == label:
            return i
    
    # 尝试直接匹配数字（0-5对应6个阶段）
    if output.strip().isdigit():
        stage = int(output.strip())
        if 0 <= stage <= 5:
            return stage
    
    # 无法识别睡眠阶段
    return None

def group_samples_by_class(dataset: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按类别分组样本
    
    Args:
        dataset: 数据集样本列表
        
    Returns:
        按类别索引分组的样本字典
    """
    class_samples = defaultdict(list)
    unclassified = []
    
    for sample in dataset:
        # 尝试从输出中提取类别
        if 'output' in sample:
            class_idx = extract_class_from_output(sample['output'])
            if class_idx is not None:
                class_samples[class_idx].append(sample)
            else:
                unclassified.append(sample)
        else:
            unclassified.append(sample)
    
    # 输出各类别的样本数量
    print("\n各睡眠阶段样本数量:")
    for class_idx, samples in sorted(class_samples.items()):
        if class_idx < len(SLEEP_STAGE_LABELS):
            print(f"  - {SLEEP_STAGE_LABELS[class_idx]}: {len(samples)} 个样本")
    
    if unclassified:
        print(f"  - 未分类样本: {len(unclassified)} 个")
    
    return class_samples

def balance_dataset(
    class_samples: Dict[int, List[Dict[str, Any]]],
    method: str = "hybrid",
    target_ratio: float = 1.0,
    rem_boost: float = 1.5,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """平衡数据集中各类别的样本数量
    
    Args:
        class_samples: 按类别分组的样本
        method: 平衡方法 (oversample/undersample/hybrid)
        target_ratio: 目标类别比例
        rem_boost: REM阶段样本数量的额外提升倍数
        random_seed: 随机种子
        
    Returns:
        平衡后的数据集样本列表
    """
    random.seed(random_seed)
    
    # 计算平均样本数
    sample_counts = [len(samples) for samples in class_samples.values()]
    avg_samples = sum(sample_counts) / len(sample_counts) if sample_counts else 0
    
    # 设置各类别的目标样本数
    target_counts = {}
    for class_idx in class_samples.keys():
        # 对REM阶段(索引为5)应用额外的提升
        if class_idx == 5:  # REM阶段
            target_counts[class_idx] = int(avg_samples * target_ratio * rem_boost)
        else:
            target_counts[class_idx] = int(avg_samples * target_ratio)
    
    # 平衡各类别样本
    balanced_samples = []
    
    for class_idx, samples in class_samples.items():
        current_count = len(samples)
        target_count = target_counts[class_idx]
        
        if method == "oversample" or (method == "hybrid" and current_count < target_count):
            # 过采样：随机复制样本直到达到目标数量
            if current_count > 0:
                # 计算需要复制的样本数
                num_to_add = target_count - current_count
                if num_to_add > 0:
                    # 随机选择样本进行复制
                    additional_samples = random.choices(samples, k=num_to_add)
                    balanced_samples.extend(samples + additional_samples)
                else:
                    balanced_samples.extend(samples)
            
        elif method == "undersample" or (method == "hybrid" and current_count > target_count):
            # 欠采样：随机选择样本减少到目标数量
            if target_count > 0:
                # 随机选择目标数量的样本
                selected_samples = random.sample(samples, target_count)
                balanced_samples.extend(selected_samples)
            
        else:
            # 保持不变
            balanced_samples.extend(samples)
    
    # 打乱样本顺序
    random.shuffle(balanced_samples)
    
    # 输出平衡后的样本数量
    print("\n平衡后各睡眠阶段样本数量:")
    class_counts = defaultdict(int)
    for sample in balanced_samples:
        class_idx = extract_class_from_output(sample['output'])
        if class_idx is not None:
            class_counts[class_idx] += 1
    
    for class_idx, count in sorted(class_counts.items()):
        if class_idx < len(SLEEP_STAGE_LABELS):
            print(f"  - {SLEEP_STAGE_LABELS[class_idx]}: {count} 个样本")
    
    return balanced_samples

def visualize_distribution(
    original_class_samples: Dict[int, List[Dict[str, Any]]],
    balanced_class_samples: Dict[int, List[Dict[str, Any]]],
    output_path: str
):
    """可视化数据分布
    
    Args:
        original_class_samples: 原始数据集按类别分组的样本
        balanced_class_samples: 平衡后数据集按类别分组的样本
        output_path: 输出图表路径
    """
    # 准备数据
    original_counts = [len(original_class_samples.get(i, [])) for i in range(len(SLEEP_STAGE_LABELS))]
    
    # 重新分组平衡后的样本
    balanced_grouped = defaultdict(list)
    for sample in balanced_class_samples:
        class_idx = extract_class_from_output(sample['output'])
        if class_idx is not None:
            balanced_grouped[class_idx].append(sample)
    
    balanced_counts = [len(balanced_grouped.get(i, [])) for i in range(len(SLEEP_STAGE_LABELS))]
    
    # 创建柱状图
    x = np.arange(len(SHORT_LABELS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, original_counts, width, label='原始数据集')
    rects2 = ax.bar(x + width/2, balanced_counts, width, label='平衡后数据集')
    
    # 添加标签和标题
    ax.set_ylabel('样本数量')
    ax.set_title('睡眠阶段样本分布对比')
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"数据分布可视化已保存至: {output_path}")

def save_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """保存数据集
    
    Args:
        dataset: 数据集样本列表
        output_path: 输出文件路径
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"已保存平衡后的数据集: {output_path} ({len(dataset)} 个样本)")
    except Exception as e:
        print(f"保存数据集失败: {e}")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 加载数据集
    dataset = load_dataset(args.input_file)
    
    # 按类别分组样本
    class_samples = group_samples_by_class(dataset)
    
    # 平衡数据集
    balanced_dataset = balance_dataset(
        class_samples,
        method=args.method,
        target_ratio=args.target_ratio,
        rem_boost=args.rem_boost,
        random_seed=args.random_seed
    )
    
    # 可视化数据分布
    if args.visualize:
        # 重新分组平衡后的样本
        output_dir = os.path.dirname(args.output_file)
        vis_path = os.path.join(output_dir, "class_distribution.png")
        visualize_distribution(class_samples, balanced_dataset, vis_path)
    
    # 保存平衡后的数据集
    save_dataset(balanced_dataset, args.output_file)
    
    print("\n数据集平衡完成!")
    print(f"原始样本数: {len(dataset)}")
    print(f"平衡后样本数: {len(balanced_dataset)}")
    print(f"REM阶段提升倍数: {args.rem_boost}")

if __name__ == "__main__":
    main() 