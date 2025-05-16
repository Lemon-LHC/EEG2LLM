"""
数据处理工具集
包含各种处理数据集的常规操作函数，如数据集窗口处理、数据生成格式化、统计信息计算、
数据集平衡策略和文件操作等。

使用方法:
from tool import *
"""

import os
import json
import numpy as np
import pandas as pd
import glob
import re
import warnings
import shutil
from tqdm import tqdm
from collections import Counter
import datetime
import openpyxl
from scipy import interpolate
from scipy.signal import resample
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import concurrent.futures
import multiprocessing

# 文件操作相关函数
def clean_output_directories(base_dir):
    """
    清理输出目录
    
    参数:
        base_dir: 基础目录路径
    """
    if os.path.exists(base_dir):
        try:
            shutil.rmtree(base_dir)
            print(f"已清理目录: {base_dir}")
        except Exception as e:
            print(f"清理目录失败: {e}")
    
def prepare_output_directories(base_dir):
    """
    准备输出目录结构
    
    参数:
        base_dir: 基础目录路径
        
    返回:
        dict: 包含各个子目录路径的字典
    """
    # 创建主目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = {
        'train': os.path.join(base_dir, 'train'),
        'test': os.path.join(base_dir, 'test'),
        'metadata': os.path.join(base_dir, 'metadata'),
        'temp': os.path.join(base_dir, 'temp')
    }
    
    for name, path in subdirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"已创建目录: {path}")
    
    return subdirs

def find_annotation_file(edf_path):
    """
    查找与EDF文件对应的注释文件
    
    参数:
        edf_path: EDF文件路径
        
    返回:
        str: 注释文件路径，如果没有找到则返回None
    """
    base_path = os.path.splitext(edf_path)[0]
    possible_extensions = ['.txt', '.csv', '.tsv', '.json', '.xml']
    
    for ext in possible_extensions:
        ann_path = base_path + ext
        if os.path.exists(ann_path):
            return ann_path
    
    # 尝试查找同一目录下的同名文件但扩展名不同
    dirname = os.path.dirname(edf_path)
    basename = os.path.basename(os.path.splitext(edf_path)[0])
    
    for filename in os.listdir(dirname):
        if filename.startswith(basename) and any(filename.endswith(ext) for ext in possible_extensions):
            return os.path.join(dirname, filename)
    
    return None

# 信号处理相关函数
def interpolate_signal(signal, original_sfreq, target_sfreq):
    """
    将信号从原始采样率重采样到目标采样率
    
    参数:
        signal: 原始信号数据
        original_sfreq: 原始采样率
        target_sfreq: 目标采样率
        
    返回:
        numpy.ndarray: 重采样后的信号
    """
    if original_sfreq == target_sfreq:
        return signal
    
    # 获取时间点
    n_samples = signal.shape[0]
    old_times = np.arange(n_samples) / original_sfreq
    
    # 计算新的采样点数
    new_n_samples = int(n_samples * target_sfreq / original_sfreq)
    new_times = np.arange(new_n_samples) / target_sfreq
    
    # 对每个通道进行插值
    resampled_signal = np.zeros((new_n_samples, signal.shape[1]))
    
    for i in range(signal.shape[1]):
        if np.all(np.isnan(signal[:, i])):
            resampled_signal[:, i] = np.nan
        else:
            # 使用线性插值
            not_nan_indices = ~np.isnan(signal[:, i])
            if np.sum(not_nan_indices) > 1:  # 至少需要两个点进行插值
                f = interpolate.interp1d(
                    old_times[not_nan_indices], 
                    signal[not_nan_indices, i],
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                resampled_signal[:, i] = f(new_times)
            elif np.sum(not_nan_indices) == 1:  # 只有一个有效点
                resampled_signal[:, i] = signal[not_nan_indices, i][0]
            else:  # 全部无效
                resampled_signal[:, i] = np.nan
    
    return resampled_signal

def format_signal_data(signal, channel_names, sfreq=None):
    """
    格式化信号数据，生成包含信号、通道名和采样率的字典
    
    参数:
        signal: 信号数据
        channel_names: 通道名列表
        sfreq: 采样率
        
    返回:
        dict: 格式化的信号数据字典
    """
    formatted_data = {
        "data": signal.tolist(),
        "channels": channel_names
    }
    
    if sfreq is not None:
        formatted_data["sampling_rate"] = float(sfreq)
    
    return formatted_data

def read_edf_file(edf_path, target_sfreq=100):
    """
    读取EDF文件，并根据需要进行重采样
    
    参数:
        edf_path: EDF文件路径
        target_sfreq: 目标采样率
        
    返回:
        tuple: (raw_data, channel_names, sfreq)
            raw_data: MNE Raw对象
            channel_names: 通道名列表
            sfreq: 采样率
    """
    try:
        # 读取EDF文件
        raw_data = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
        
        # 获取通道名
        channel_names = raw_data.ch_names
        
        # 获取原始采样率
        original_sfreq = raw_data.info['sfreq']
        
        # 如果需要，进行重采样
        if original_sfreq != target_sfreq:
            raw_data = raw_data.resample(target_sfreq)
        
        return raw_data, channel_names, target_sfreq
    
    except Exception as e:
        print(f"读取EDF文件失败: {edf_path}")
        print(f"错误信息: {e}")
        return None, None, None

def extract_stages_from_annotations(annotations):
    """
    从MNE Raw对象的注释中提取睡眠阶段
    
    参数:
        annotations: MNE Annotations对象
        
    返回:
        list: 包含睡眠阶段信息的列表，每个元素是(开始时间, 持续时间, 阶段)的元组
    """
    stages = []
    
    # 定义睡眠阶段的映射关系
    stage_mapping = {
        '0': 'W', 'W': 'W', 'SLEEP-WAKE': 'W', 'Sleep stage W': 'W', 'wakefulness': 'W', 'wake': 'W',
        '1': 'N1', 'N1': 'N1', 'SLEEP-S1': 'N1', 'Sleep stage 1': 'N1', 'S1': 'N1', 'stage 1': 'N1',
        '2': 'N2', 'N2': 'N2', 'SLEEP-S2': 'N2', 'Sleep stage 2': 'N2', 'S2': 'N2', 'stage 2': 'N2',
        '3': 'N3', 'N3': 'N3', 'SLEEP-S3': 'N3', 'Sleep stage 3': 'N3', 'S3': 'N3', 'stage 3': 'N3',
        '4': 'N3', 'N4': 'N3', 'SLEEP-S4': 'N3', 'Sleep stage 4': 'N3', 'S4': 'N3', 'stage 4': 'N3',
        'R': 'REM', 'REM': 'REM', 'SLEEP-REM': 'REM', 'Sleep stage R': 'REM', 'stage R': 'REM',
        'M': 'MOVEMENT', 'MOVEMENT': 'MOVEMENT'
    }
    
    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        # 尝试解析睡眠阶段
        stage = None
        
        # 直接查找映射
        if description in stage_mapping:
            stage = stage_mapping[description]
        else:
            # 尝试从描述中提取阶段信息
            for key, value in stage_mapping.items():
                if key in description:
                    stage = value
                    break
        
        if stage:
            stages.append((onset, duration, stage))
    
    return stages

def extract_stage_windows(raw_data, annotations, channel_names, add_noise=False, max_windows=None, eeg_window_size=15000):
    """
    从原始数据中提取睡眠阶段窗口
    
    参数:
        raw_data: MNE Raw对象
        annotations: 注释信息
        channel_names: 通道名列表
        add_noise: 是否添加噪声
        max_windows: 每个阶段最大窗口数
        eeg_window_size: EEG窗口大小
        
    返回:
        list: 包含窗口数据的列表，每个元素是字典
    """
    # 提取睡眠阶段
    stages = extract_stages_from_annotations(annotations)
    if not stages:
        return []
    
    # 获取采样率
    sfreq = raw_data.info['sfreq']
    
    # 从Raw对象获取数据
    data = raw_data.get_data()
    data = data.T  # 转置为(n_samples, n_channels)
    
    # 初始化结果列表和计数器
    windows = []
    stage_counters = Counter()
    
    # 遍历所有睡眠阶段
    for onset, duration, stage in stages:
        # 跳过不感兴趣的阶段
        if stage not in ['W', 'N1', 'N2', 'N3', 'REM']:
            continue
            
        # 检查是否已达到该阶段的最大窗口数
        if max_windows and stage_counters[stage] >= max_windows:
            continue
            
        # 计算起始和结束样本索引
        start_idx = int(onset * sfreq)
        end_idx = int((onset + duration) * sfreq)
        
        # 确保结束索引不超出数据长度
        end_idx = min(end_idx, data.shape[0])
        
        # 长度不足的睡眠阶段
        if end_idx - start_idx < eeg_window_size:
            continue
            
        # 从这个睡眠阶段中提取窗口
        for window_start in range(start_idx, end_idx - eeg_window_size + 1, eeg_window_size // 2):  # 50% 重叠
            window_end = window_start + eeg_window_size
            
            # 如果窗口超出了这个阶段的结束，就跳过
            if window_end > end_idx:
                continue
                
            # 提取窗口数据
            window_data = data[window_start:window_end].copy()
            
            # 可选：添加噪声
            if add_noise:
                noise_level = 0.01
                noise = np.random.normal(0, noise_level, window_data.shape)
                window_data += noise
            
            # 格式化窗口数据
            formatted_signal = format_signal_data(window_data, channel_names, sfreq)
            
            # 创建窗口字典
            window_dict = {
                "signal": formatted_signal,
                "stage": stage,
                "start_time": float(window_start / sfreq),
                "end_time": float(window_end / sfreq),
                "duration": float(eeg_window_size / sfreq)
            }
            
            windows.append(window_dict)
            stage_counters[stage] += 1
            
            # 检查是否已达到该阶段的最大窗口数
            if max_windows and stage_counters[stage] >= max_windows:
                break
    
    return windows

# 数据集生成和格式化相关函数
def create_llm_sample(signal, stage, channel_names, sfreq):
    """
    创建用于LLM的样本
    
    参数:
        signal: 信号数据
        stage: 睡眠阶段
        channel_names: 通道名列表
        sfreq: 采样率
        
    返回:
        dict: 格式化的LLM样本
    """
    # 将睡眠阶段映射到全称
    stage_mapping = {
        'W': '清醒状态',
        'N1': '非快速眼动睡眠第一阶段',
        'N2': '非快速眼动睡眠第二阶段',
        'N3': '非快速眼动睡眠第三阶段',
        'REM': '快速眼动睡眠阶段'
    }
    
    # 格式化信号数据
    formatted_signal = format_signal_data(signal, channel_names, sfreq)
    
    # 创建样本
    sample = {
        "signal": formatted_signal,
        "stage": stage,
        "stage_fullname": stage_mapping.get(stage, stage),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return sample

# 数据集统计相关函数
def count_stages(data):
    """
    统计数据集中的睡眠阶段数量
    
    参数:
        data: 数据列表，每个元素是一个包含'stage'键的字典
        
    返回:
        Counter: 各睡眠阶段的计数
    """
    stages = [item['stage'] for item in data]
    return Counter(stages)

def print_stage_statistics(all_data, train_data, test_data, file_types=None):
    """
    打印数据集睡眠阶段的统计信息
    
    参数:
        all_data: 所有数据
        train_data: 训练数据
        test_data: 测试数据
        file_types: 文件类型列表
    """
    print("\n" + "="*50)
    print("睡眠阶段统计信息")
    print("="*50)
    
    all_stages = count_stages(all_data)
    train_stages = count_stages(train_data)
    test_stages = count_stages(test_data)
    
    print("\n所有数据集:")
    for stage, count in sorted(all_stages.items()):
        print(f"  {stage}: {count} 个样本 ({count/sum(all_stages.values())*100:.2f}%)")
    print(f"  总计: {sum(all_stages.values())} 个样本")
    
    print("\n训练集:")
    for stage, count in sorted(train_stages.items()):
        print(f"  {stage}: {count} 个样本 ({count/sum(train_stages.values())*100:.2f}%)")
    print(f"  总计: {sum(train_stages.values())} 个样本")
    
    print("\n测试集:")
    for stage, count in sorted(test_stages.items()):
        print(f"  {stage}: {count} 个样本 ({count/sum(test_stages.values())*100:.2f}%)")
    print(f"  总计: {sum(test_stages.values())} 个样本")
    
    if file_types:
        print("\n按文件类型统计:")
        for file_type in file_types:
            file_type_data = [item for item in all_data if item.get('file_type') == file_type]
            if file_type_data:
                file_type_stages = count_stages(file_type_data)
                print(f"\n{file_type}:")
                for stage, count in sorted(file_type_stages.items()):
                    print(f"  {stage}: {count} 个样本 ({count/sum(file_type_stages.values())*100:.2f}%)")
                print(f"  总计: {sum(file_type_stages.values())} 个样本")
    
    print("="*50)

def save_metrics_to_excel(dataset_info, output_path):
    """
    将数据集指标保存到Excel文件
    
    参数:
        dataset_info: 包含数据集信息的字典
        output_path: 输出Excel文件路径
    """
    # 创建DataFrame
    df = pd.DataFrame([dataset_info])
    
    # 保存到Excel
    try:
        df.to_excel(output_path, index=False)
        print(f"指标已保存到: {output_path}")
    except Exception as e:
        print(f"保存指标失败: {e}")

def update_dataset_info(train_filename, output_dir):
    """
    更新和保存数据集信息
    
    参数:
        train_filename: 训练数据文件名
        output_dir: 输出目录
        
    返回:
        dict: 更新后的数据集信息
    """
    # 读取训练数据集
    train_path = os.path.join(output_dir, 'train', train_filename)
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 读取测试数据集
    test_filename = train_filename.replace('train', 'test')
    test_path = os.path.join(output_dir, 'test', test_filename)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 统计睡眠阶段
    train_stages = count_stages(train_data)
    test_stages = count_stages(test_data)
    all_stages = train_stages + test_stages
    
    # 计算各阶段的比例
    total_train = sum(train_stages.values())
    total_test = sum(test_stages.values())
    total_all = sum(all_stages.values())
    
    stage_percents = {stage: f"{count/total_all*100:.2f}%" for stage, count in all_stages.items()}
    train_percents = {stage: f"{count/total_train*100:.2f}%" for stage, count in train_stages.items()}
    test_percents = {stage: f"{count/total_test*100:.2f}%" for stage, count in test_stages.items()}
    
    # 创建数据集信息字典
    dataset_info = {
        "dataset_name": os.path.splitext(train_filename)[0],
        "created_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total_all,
        "train_samples": total_train,
        "test_samples": total_test,
        "train_test_ratio": f"{total_train/total_all*100:.2f}%/{total_test/total_all*100:.2f}%"
    }
    
    # 添加各阶段的统计信息
    for stage in sorted(all_stages.keys()):
        dataset_info[f"{stage}_total"] = all_stages[stage]
        dataset_info[f"{stage}_percent"] = stage_percents[stage]
        dataset_info[f"{stage}_train"] = train_stages[stage]
        dataset_info[f"{stage}_train_percent"] = train_percents[stage]
        dataset_info[f"{stage}_test"] = test_stages[stage]
        dataset_info[f"{stage}_test_percent"] = test_percents[stage]
    
    # 保存数据集信息
    info_path = os.path.join(output_dir, 'metadata', f"{os.path.splitext(train_filename)[0]}_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 保存到Excel
    excel_path = os.path.join(output_dir, 'metadata', f"{os.path.splitext(train_filename)[0]}_info.xlsx")
    save_metrics_to_excel(dataset_info, excel_path)
    
    return dataset_info

# 数据集平衡策略
def balance_dataset(data, strategy="balanced", balance_alpha=0.7, weight_method="sqrt_inverse"):
    """
    对数据集进行平衡
    
    参数:
        data: 数据列表，每个元素是一个包含'stage'键的字典
        strategy: 平衡策略，可选值为"none"(不平衡), "balanced"(完全平衡), "natural"(保持自然分布)
        balance_alpha: 平衡系数，用于"natural"策略
        weight_method: 权重计算方法，可选值为"inverse"(反比例), "sqrt_inverse"(反比例的平方根)
        
    返回:
        list: 平衡后的数据集
    """
    if strategy == "none":
        return data
    
    # 统计各睡眠阶段的数量
    stage_counts = count_stages(data)
    print(f"原始数据集统计: {dict(stage_counts)}")
    
    # 根据策略计算目标数量
    if strategy == "balanced":
        # 所有阶段数量相等
        target_count = min(stage_counts.values())
        target_counts = {stage: target_count for stage in stage_counts.keys()}
    else:  # natural
        # 计算权重
        total = sum(stage_counts.values())
        if weight_method == "inverse":
            weights = {stage: 1 / (count / total) for stage, count in stage_counts.items()}
        elif weight_method == "sqrt_inverse":
            weights = {stage: 1 / np.sqrt(count / total) for stage, count in stage_counts.items()}
        else:
            weights = {stage: 1 for stage in stage_counts.keys()}
            
        # 归一化权重
        weight_sum = sum(weights.values())
        normalized_weights = {stage: w / weight_sum for stage, w in weights.items()}
        
        # 计算目标分布
        natural_distrib = {stage: count / total for stage, count in stage_counts.items()}
        target_distrib = {}
        
        for stage in stage_counts.keys():
            # 在自然分布和均匀分布之间插值
            uniform_prob = 1 / len(stage_counts)
            natural_prob = natural_distrib[stage]
            target_distrib[stage] = balance_alpha * uniform_prob + (1 - balance_alpha) * natural_prob
        
        # 调整目标分布使总和为1
        distrib_sum = sum(target_distrib.values())
        target_distrib = {stage: p / distrib_sum for stage, p in target_distrib.items()}
        
        # 计算目标数量
        min_stage = min(stage_counts.items(), key=lambda x: x[1])[0]
        min_target_prob = target_distrib[min_stage]
        min_count = stage_counts[min_stage]
        
        # 计算其他阶段的目标数量
        target_counts = {stage: int(min_count * target_distrib[stage] / min_target_prob) for stage in stage_counts.keys()}
    
    # 对数据集进行采样
    balanced_data = []
    for stage, target_count in target_counts.items():
        stage_data = [item for item in data if item['stage'] == stage]
        current_count = len(stage_data)
        
        if current_count <= target_count:
            # 如果当前数量小于目标数量，全部保留，并进行重复采样
            balanced_data.extend(stage_data)
            # 重复采样直到达到目标数量
            if current_count < target_count:
                # 计算需要重复的次数
                repeats_needed = target_count - current_count
                # 随机选择要重复的样本
                repeat_samples = np.random.choice(stage_data, size=repeats_needed, replace=True)
                balanced_data.extend(repeat_samples)
        else:
            # 如果当前数量大于目标数量，随机采样
            sampled_data = np.random.choice(stage_data, size=target_count, replace=False)
            balanced_data.extend(sampled_data)
    
    # 随机打乱数据
    np.random.shuffle(balanced_data)
    
    print(f"平衡后数据集统计: {dict(count_stages(balanced_data))}")
    return balanced_data

# 文本处理相关函数
def get_tokenizer():
    """
    获取用于计算token长度的tokenizer
    
    返回:
        transformers.AutoTokenizer: tokenizer对象
    """
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        return None

def safe_calculate_token_length(text):
    """
    安全计算文本的token长度
    
    参数:
        text: 要计算长度的文本
        
    返回:
        int: token长度，如果失败则返回估计长度
    """
    try:
        tokenizer = get_tokenizer()
        if tokenizer:
            tokens = tokenizer.encode(text)
            return len(tokens)
    except Exception as e:
        print(f"计算token长度失败: {e}")
    
    # 如果失败，使用简单估计
    return len(text.split()) * 1.3

def calculate_tokens(sample):
    """
    计算样本中文本的token长度
    
    参数:
        sample: 样本字典
        
    返回:
        int: token长度
    """
    if not isinstance(sample, dict):
        return 0
    
    total_tokens = 0
    
    # 计算文本字段的token长度
    text_fields = ['stage', 'stage_fullname', 'description', 'summary']
    for field in text_fields:
        if field in sample and isinstance(sample[field], str):
            total_tokens += safe_calculate_token_length(sample[field])
    
    return total_tokens

# 辅助函数
def safe_print(*args, **kwargs):
    """
    安全打印函数，捕获并处理异常
    """
    try:
        print(*args, **kwargs)
    except Exception as e:
        try:
            print(f"打印错误: {str(e)}")
        except:
            pass 