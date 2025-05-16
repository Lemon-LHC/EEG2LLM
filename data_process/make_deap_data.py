#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import json
import numpy as np
import argparse
from tqdm import tqdm
import warnings
from scipy import signal
from transformers import AutoTokenizer
import pandas as pd
from datetime import datetime
import re
import shutil

# 过滤警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def bin_power(x, bands, fs):
    """计算信号在各频段的功率谱
    
    参数:
        x: 信号数据
        bands: 频段边界列表
        fs: 采样频率
    
    返回:
        power_ratio: 各频段的相对功率
        power: 各频段的绝对功率
    """
    n_samples = len(x)
    
    # 快速傅里叶变换
    fft_data = np.abs(np.fft.rfft(x))
    
    # 计算频率分辨率
    freq = np.fft.rfftfreq(n_samples, 1.0/fs)
    
    # 计算各频段的功率
    power = np.zeros(len(bands) - 1)
    for band_idx in range(len(bands) - 1):
        freq_mask = (freq >= bands[band_idx]) & (freq < bands[band_idx + 1])
        power[band_idx] = np.sum(fft_data[freq_mask] ** 2)
    
    # 计算相对功率
    total_power = np.sum(power)
    if total_power == 0:
        power_ratio = np.zeros_like(power)
    else:
        power_ratio = power / total_power
    
    return power_ratio, power

def format_signal_data(signal, channel_names, sfreq=None):
    """将信号数据转换为文本格式，保留原始微伏级别数据且使用4位有效数字，使用>分隔数据点
    
    参数:
        signal: 信号数据数组 (微伏级别)
        channel_names: 通道名称列表
        sfreq: 采样频率（Hz），默认为None，则使用128Hz
    """
    # 如果没有提供采样频率，默认使用128Hz
    if sfreq is None:
        sfreq = 128  
    
    # 构建数据点字符串列表
    data_points = []
    for t in range(signal.shape[1]):
        # 直接使用原始值，不使用科学计数法
        ch1_val = float(signal[0, t])
        ch2_val = float(signal[1, t]) if signal.shape[0] > 1 else 0.0
        
        # 使用普通浮点数表示，保留4位有效数字
        point_data = f"{ch1_val:.4f},{ch2_val:.4f}"
        data_points.append(point_data)
    
    # 使用 > 分隔数据点，并在开头添加通道名称
    return f"{', '.join(channel_names)}>{'>'.join(data_points)}>"

def calculate_spectral_features(signal_data, sfreq=128, bands=None):
    """计算信号的频谱特征
    
    参数:
        signal_data: 2D信号数据数组，形状为 [channels, samples]
        sfreq: 采样频率，默认为128Hz
        bands: 频段边界列表，默认为[4, 8, 12, 16, 25, 45]
    
    返回:
        字典，包含各通道各频段的功率谱特征
    """
    # 定义频段
    if bands is None:
        bands = [4, 8, 12, 16, 25, 45]  # 5个频段：theta, alpha, low beta, high beta, gamma
    
    # 从频段边界确定频段名称，确保名称与频段数量匹配
    if len(bands) == 6:  # 对于默认的5个频段（6个边界）
        band_names = ["theta", "alpha", "low_beta", "high_beta", "gamma"]
    else:
        # 对于自定义频段，生成通用名称
        band_names = [f"band_{i}" for i in range(len(bands) - 1)]
    
    # 存储所有通道的频谱特征
    features = {}
    
    # 计算每个通道的频谱特征
    for ch_idx in range(signal_data.shape[0]):
        ch_features, ch_power = bin_power(signal_data[ch_idx], bands, sfreq)
        
        # 将特征存储为字典，键为频段名称，值为相对功率
        ch_feature_dict = {band_names[i]: float(ch_features[i]) for i in range(len(band_names))}
        
        # 将通道特征添加到总特征字典中
        features[f"channel_{ch_idx}"] = ch_feature_dict
    
    return features

def format_spectral_features(features, channel_names):
    """格式化频谱特征为文本"""
    formatted_text = "Spectral Features:"
    
    for ch_idx, ch_name in enumerate(channel_names):
        ch_key = f"channel_{ch_idx}"
        if ch_key in features:
            formatted_text += f"\n{ch_name}: "
            ch_features = features[ch_key]
            
            feature_texts = []
            for band_name, power in ch_features.items():
                feature_texts.append(f"{band_name}={power:.4f}")
            
            formatted_text += ", ".join(feature_texts)
    
    return formatted_text

def normalize_spectral_features(features):
    """对功率谱密度进行归一化处理
    
    参数:
        features: 频谱特征字典
    
    返回:
        归一化后的频谱特征字典
    """
    # 将所有通道的特征值转换为numpy数组
    all_values = []
    for ch_features in features.values():
        all_values.extend(list(ch_features.values()))
    all_values = np.array(all_values)
    
    # 计算均值和标准差
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    
    # 归一化处理
    normalized_features = {}
    for ch_key, ch_features in features.items():
        normalized_features[ch_key] = {
            band: (value - mean_val) / std_val 
            for band, value in ch_features.items()
        }
    
    return normalized_features

def format_eeg_signal(signal):
    """格式化EEG信号数据，保留4位有效数字
    
    参数:
        signal: 原始信号数据
    
    返回:
        格式化后的信号数据
    """
    return np.around(signal, decimals=4)

def get_prompt_by_type(input_type):
    """根据输入类型获取相应的提示词"""
    # 设计三种不同的提示词基础模板，避免重复
    if input_type == 1:
        # 仅频谱特征模式 - 保持不变
        base_prompt = """You are a neurobiologist specializing in EEG frequency analysis. Your task is to analyze the provided normalized spectral features and determine the participant's emotional state based on the following classification:

0: Low Arousal Low Valence (LALV)
1: Low Arousal High Valence (LAHV) 
2: High Arousal Low Valence (HALV)
3: High Arousal High Valence (HAHV)

The data contains normalized relative power values for five frequency bands:
- Theta (4-8Hz): Associated with drowsiness, creativity, and emotional processing
- Alpha (8-12Hz): Linked to relaxation, meditation, and attentional focus
- Low Beta (12-16Hz): Connected to conscious thought and light concentration
- High Beta (16-25Hz): Related to active thinking, alertness, and anxiety
- Gamma (25-45Hz): Involved in higher cognitive processing and peak concentration

When analyzing spectral features, consider these emotional indicators:
- LALV (0): Emotions like sadness and depression typically show enhanced theta waves with reduced beta activity, reflecting low mental energy with negative affect
- LAHV (1): Emotions like contentment and relaxation show elevated alpha power, indicating a calm state with positive emotional tone
- HALV (2): Emotions like fear and anger display increased beta and reduced alpha, showing high arousal with negative valence
- HAHV (3): Emotions like excitement and happiness exhibit balanced beta/gamma activity with moderate alpha, indicating engaged positive states

Your response must be a single digit (0, 1, 2, or 3) corresponding to the above emotional categories."""
        
    elif input_type == 2:
        # 仅信号模式
        base_prompt = """You are a neurobiologist specializing in EEG time-series analysis. Your task is to examine the provided raw EEG signals and classify the participant's emotional state into one of four categories:

0: Low Arousal Low Valence (LALV)
1: Low Arousal High Valence (LAHV)
2: High Arousal Low Valence (HALV)
3: High Arousal High Valence (HAHV)

The data contains voltage values from Fpz-Cz and Pz-Oz channels in microvolts (μV), expressed as decimal numbers with 4 significant digits. When interpreting time-domain EEG signals:
- Low frequency waves with higher amplitude often indicate relaxed or drowsy states
- Fast, low-amplitude patterns typically reflect active mental processing
- Frontal asymmetry (differences between hemispheres) can indicate emotional valence
- Signal complexity and variability can reflect arousal levels

Key signal characteristics for each emotional state:
- LALV (0): Sadness/depression shows slow, higher amplitude waves with greater frontal right activity
- LAHV (1): Relaxation/contentment displays moderate, rhythmic waveforms with balanced frontal activity
- HALV (2): Anger/fear produces fast, irregular patterns with frontal asymmetry favoring the right hemisphere
- HAHV (3): Excitement/joy exhibits moderately fast, dynamic patterns with greater left frontal activation

Your response must be a single digit (0, 1, 2, or 3) corresponding to the above emotional categories."""
        
    else:
        # 综合模式
        base_prompt = """You are a neurobiologist specializing in multi-modal EEG analysis. Your task is to integrate spectral and time-domain EEG data to determine the participant's emotional state according to this framework:

0: Low Arousal Low Valence (LALV)
1: Low Arousal High Valence (LAHV)
2: High Arousal Low Valence (HALV)
3: High Arousal High Valence (HAHV)

The data contains two complementary components:
1. Spectral features: Normalized power distribution across frequency bands (theta, alpha, low_beta, high_beta, gamma)
2. Time-domain signals: Raw voltage readings from Fpz-Cz and Pz-Oz channels in microvolts (μV), expressed as decimal numbers with 4 significant digits

Brain wave patterns correlate with emotional states as follows:
- Theta waves (4-8Hz): Enhanced during emotional processing and creative visualization
- Alpha waves (8-12Hz): Dominant during relaxed, meditative states with focused attention
- Beta waves (12-25Hz): Prominent during alert, engaged mental activity and stress responses
- Gamma waves (25-45Hz): Present during peak cognitive processing and intense emotional states

Comprehensive emotional state indicators:
- LALV (0): Negative low-energy emotions (sadness, depression) manifest as increased theta activity, reduced beta power, and slow-wave dominant EEG patterns
- LAHV (1): Positive low-energy emotions (contentment, serenity) show enhanced alpha rhythms, moderate theta, and smooth, regular EEG oscillations
- HALV (2): Negative high-energy emotions (anger, fear) display elevated beta/high-beta activity, suppressed alpha, and rapid, asymmetric EEG patterns
- HAHV (3): Positive high-energy emotions (joy, excitement) exhibit balanced fast-wave activity, moderate alpha, and dynamic, harmonious EEG signals

Your response must be a single digit (0, 1, 2, or 3) corresponding to the above emotional categories."""
    
    return base_prompt

def generate_output_filename(param_mapping, num_files, avg_tokens):
    """生成输出文件名
    
    参数:
        param_mapping: 参数映射字典
        num_files: 处理的文件数量
        avg_tokens: 前10条数据的平均token数
    """
    # 构建基本文件名
    base_filename = f"deap_{num_files}_type{param_mapping['input_type']}_{param_mapping['sampling_rate']}hz_" \
                   f"{int(param_mapping['window_length_sec'] * 1000)}ms_" \
                   f"step{int(param_mapping['window_step_sec'] * 1000)}ms_" \
                   f"tok{int(avg_tokens)}_" \
                   f"bal{param_mapping['balance_ratio']}_{param_mapping['balance_strategy']}"
    
    # 如果需要添加时间戳
    if param_mapping.get('append_timestamp', True):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{base_filename}_{timestamp}"
    else:
        filename = base_filename
    
    return filename

def create_llm_sample(signal, labels, channel_names, sfreq, input_type=3, include_spectral=True, spectral_bands=None):
    """创建用于大模型训练的样本
    
    参数:
        signal: 信号数据数组
        labels: 情绪标签
        channel_names: 通道名称列表
        sfreq: 采样频率
        input_type: 输入类型 (1: 仅频谱, 2: 仅信号, 3: 两者都包含)
        include_spectral: 是否包含频谱特征
        spectral_bands: 频谱分析的频段边界
    """
    # 获取情绪类别
    emotion_category = get_emotion_category(labels)
    
    # 获取对应的提示词
    instruction = get_prompt_by_type(input_type)
    
    # 根据input_type准备输入数据
    if input_type == 1:
        # 仅包含归一化的频谱特征
        spectral_features = calculate_spectral_features(signal, sfreq, bands=spectral_bands)
        normalized_features = normalize_spectral_features(spectral_features)
        input_data = format_spectral_features(normalized_features, channel_names)
    
    elif input_type == 2:
        # 仅包含处理后的EEG信号
        formatted_signal = format_eeg_signal(signal)
        input_data = format_signal_data(formatted_signal, channel_names, sfreq)
    
    else:
        # 同时包含两种信息
        spectral_features = calculate_spectral_features(signal, sfreq, bands=spectral_bands)
        normalized_features = normalize_spectral_features(spectral_features)
        formatted_signal = format_eeg_signal(signal)
        
        input_data = f"{format_spectral_features(normalized_features, channel_names)}\n\n" \
                    f"Raw EEG Data: {format_signal_data(formatted_signal, channel_names, sfreq)}"
    
    # 创建样本
    sample = {
        "instruction": instruction,
        "input": input_data,
        "output": f"{emotion_category}",
        "system": "You are a neuroscience expert specializing in EEG and emotion analysis."
    }
    
    return sample

def get_emotion_category(labels):
    """
    根据情绪标签确定情绪类别
    
    参数:
        labels: [valence, arousal, dominance, liking]
    
    返回:
        情绪类别：0-LALV, 1-LAHV, 2-HALV, 3-HAHV
    """
    valence = labels[0]
    arousal = labels[1]
    
    # 阈值划分，通常在情感研究中使用5作为阈值（1-9分量表）
    if valence <= 5 and arousal <= 5:
        return 0  # LALV
    elif valence > 5 and arousal <= 5:
        return 1  # LAHV
    elif valence <= 5 and arousal > 5:
        return 2  # HALV
    else:  # valence > 5 and arousal > 5
        return 3  # HAHV

def clean_output_directories(base_dir):
    """清理输出目录中的所有json文件
    
    参数:
        base_dir: 基础输出目录路径
    """
    try:
        if not os.path.exists(base_dir):
            print(f"目录不存在，将创建: {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            return
        
        print("清理之前生成的json文件...")
        # 遍历所有子目录
        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)
                    try:
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {str(e)}")
                        
    except Exception as e:
        print(f"清理目录时出错: {str(e)}")
        raise

def create_excel_report(param_mapping, dataset_info, output_dir):
    """创建Excel报告，记录数据处理结果和参数信息"""
    try:
        # 生成与JSON文件相同格式的文件名
        base_filename = generate_output_filename(
            param_mapping,
            dataset_info['files_processed'],
            dataset_info['avg_tokens']
        )
        
        # 创建info目录
        info_dir = os.path.join(output_dir, 'info')
        os.makedirs(info_dir, exist_ok=True)
        
        # 创建Excel writer对象
        excel_file = os.path.join(info_dir, f"{base_filename}.xlsx")
        
        try:
            writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
        except ImportError:
            print("警告: xlsxwriter未安装，尝试使用openpyxl引擎...")
            writer = pd.ExcelWriter(excel_file, engine='openpyxl')
        
        # 创建参数信息表
        param_df = pd.DataFrame(list(param_mapping.items()), columns=['参数名称', '参数值'])
        param_df.to_excel(writer, sheet_name='参数配置', index=False)
        
        # 创建数据集统计信息表
        stats_data = {
            '统计指标': [
                '样本总数',
                '训练样本数',
                '测试样本数',
                '平均Token数',
                '处理时间(秒)',
                '处理文件数'
            ],
            '数值': [
                dataset_info.get('total_samples', 0),
                dataset_info.get('train_samples', 0),
                dataset_info.get('test_samples', 0),
                dataset_info.get('avg_tokens', 0),
                dataset_info.get('processing_time', 0),
                dataset_info.get('files_processed', 0)
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='数据统计', index=False)
        
        # 创建情绪分布表
        if 'distributions' in dataset_info:
            dist = dataset_info['distributions']
            
            # 训练集分布
            emotion_categories = {
                0: "低唤醒低效价(LALV)",
                1: "低唤醒高效价(LAHV)",
                2: "高唤醒低效价(HALV)",
                3: "高唤醒高效价(HAHV)"
            }
            
            # 训练集分布
            categories = sorted(list(dist['train_before'].keys()))
            train_dist_data = {
                '情绪类别': [emotion_categories.get(k, f"类别{k}") for k in categories],
                '平衡前数量': [dist['train_before'][k] for k in categories],
                '平衡前占比(%)': [dist['train_before'][k]/sum(dist['train_before'].values())*100 for k in categories],
                '平衡后数量': [dist['train_after'][k] for k in categories],
                '平衡后占比(%)': [dist['train_after'][k]/sum(dist['train_after'].values())*100 for k in categories],
                '变化率(%)': [(dist['train_after'][k]/sum(dist['train_after'].values())*100 - 
                           dist['train_before'][k]/sum(dist['train_before'].values())*100) 
                           for k in categories]
            }
            train_dist_df = pd.DataFrame(train_dist_data)
            train_dist_df.to_excel(writer, sheet_name='训练集分布', index=False)
            
            # 测试集分布
            test_categories = sorted(list(dist['test_before'].keys()))
            test_dist_data = {
                '情绪类别': [emotion_categories.get(k, f"类别{k}") for k in test_categories],
                '样本数量': [dist['test_before'][k] for k in test_categories],
                '占比(%)': [dist['test_before'][k]/sum(dist['test_before'].values())*100 for k in test_categories]
            }
            test_dist_df = pd.DataFrame(test_dist_data)
            test_dist_df.to_excel(writer, sheet_name='测试集分布', index=False)
        
        # 保存Excel文件
        writer.close()
        
        if param_mapping['verbose']:
            print(f"\nExcel报告已保存到: {excel_file}")
            
            # 打印数据分布情况
            print("\n" + "="*80)
            print("数据集分布情况统计")
            print("="*80)
            
            # 打印训练集分布
            print("\n训练集分布:")
            print("-"*100)
            print(f"{'情绪类别':<20}{'平衡前数量':<12}{'平衡前占比':<12}{'平衡后数量':<12}{'平衡后占比':<12}{'变化率':<12}")
            print("-"*100)
            
            train_total_before = sum(dist['train_before'].values())
            train_total_after = sum(dist['train_after'].values())
            
            for category in sorted(dist['train_before'].keys()):
                before_count = dist['train_before'][category]
                after_count = dist['train_after'][category]
                before_percent = before_count/train_total_before*100
                after_percent = after_count/train_total_after*100
                change_percent = after_percent - before_percent
                
                print(f"{emotion_categories[category]:<20}"
                      f"{before_count:<12}"
                      f"{before_percent:.2f}%{'':<6}"
                      f"{after_count:<12}"
                      f"{after_percent:.2f}%{'':<6}"
                      f"{change_percent:+.2f}%")
            
            print("-"*100)
            print(f"{'总计':<20}{train_total_before:<12}{'100.00%':<12}"
                  f"{train_total_after:<12}{'100.00%':<12}")
            
            # 打印测试集分布
            print("\n测试集分布:")
            print("-"*60)
            print(f"{'情绪类别':<20}{'样本数量':<12}{'占比':<12}")
            print("-"*60)
            
            test_total = sum(dist['test_before'].values())
            
            for category in sorted(dist['test_before'].keys()):
                count = dist['test_before'][category]
                percent = count/test_total*100 if test_total > 0 else 0
                
                print(f"{emotion_categories[category]:<20}"
                      f"{count:<12}"
                      f"{percent:.2f}%")
            
            print("-"*60)
            print(f"{'总计':<20}{test_total:<12}{'100.00%':<12}")
            print("="*80 + "\n")
            
    except Exception as e:
        print(f"警告: 创建Excel报告时出错: {str(e)}")
        print("继续处理其他任务...")

def balance_samples(samples, strategy='sqrt_inverse', balance_ratio=0.5):
    """对样本进行平衡处理
    
    参数:
        samples: 样本列表
        strategy: 平衡策略 ('sqrt_inverse'/'linear_inverse'/'none')
        balance_ratio: 平衡系数 (0-1)
    
    返回:
        平衡后的样本列表
    """
    # 如果不需要平衡，直接返回原始样本
    if strategy == 'none' or balance_ratio == 0:
        return samples
    
    # 统计各类别样本数量
    category_counts = {}
    for sample in samples:
        category = int(sample['output'])
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # 计算各类别的权重
    max_count = max(category_counts.values())
    weights = {}
    
    for category, count in category_counts.items():
        if strategy == 'sqrt_inverse':
            weight = np.sqrt(max_count / count)
        else:  # linear_inverse
            weight = max_count / count
        
        # 应用平衡系数
        weights[category] = 1 + (weight - 1) * balance_ratio
    
    # 根据权重复制样本
    balanced_samples = []
    for category in category_counts:
        category_samples = [s for s in samples if int(s['output']) == category]
        repeat_times = int(weights[category])
        balanced_samples.extend(category_samples * repeat_times)
        
        # 处理小数部分
        fraction = weights[category] - int(weights[category])
        if fraction > 0:
            additional_samples = int(len(category_samples) * fraction)
            if additional_samples > 0:
                np.random.shuffle(category_samples)
                balanced_samples.extend(category_samples[:additional_samples])
    
    # 随机打乱平衡后的样本
    np.random.shuffle(balanced_samples)
    return balanced_samples

def prepare_output_directories(base_dir, clean_files=False):
    """准备输出目录，返回三个主要数据集目录列表
    
    参数:
        base_dir: 基础输出目录路径 (/data/lhc/datasets_new/deap)
        clean_files: 是否清理目录中现有的json文件
    
    返回:
        [train_dir, test_dir, all_dir]: 包含三个子目录路径的列表
    """
    try:
        # 确保基础目录存在
        if not os.path.exists(base_dir):
            print(f"创建基础输出目录: {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
        
        # 创建info目录
        info_dir = os.path.join(base_dir, 'info')
        if not os.path.exists(info_dir):
            print(f"创建info目录: {info_dir}")
            os.makedirs(info_dir, exist_ok=True)
        
        # 创建train、test和all子目录
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')
        all_dir = os.path.join(base_dir, 'all')
        
        # 创建所有必要的子目录
        for dir_path in [train_dir, test_dir, all_dir]:
            if not os.path.exists(dir_path):
                print(f"创建目录: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
        
        # 如果指定了清理选项，则清理现有的json文件
        if clean_files:
            clean_output_directories(base_dir)
            print("已清理现有JSON文件")
        else:
            print("保留现有JSON文件")
        
        return [train_dir, test_dir, all_dir]
        
    except Exception as e:
        print(f"创建目录结构时出错: {str(e)}")
        raise

def calculate_average_tokens(samples):
    """计算样本的平均token数量
    
    参数:
        samples: 样本列表，每个样本包含instruction和input字段
    
    返回:
        平均token数量
    """
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained("/data/lhc/models/Llama-3.2-1B-Instruct", trust_remote_code=True)
        
        # 计算每个样本的token数量
        token_counts = []
        for sample in samples:
            # 合并instruction和input文本
            text = f"{sample['instruction']}\n\n{sample['input']}"
            
            # 计算token数量
            tokens = tokenizer(text, return_tensors="pt")
            token_count = len(tokens['input_ids'][0])
            token_counts.append(token_count)
        
        # 计算平均值
        avg_tokens = np.mean(token_counts)
        return avg_tokens
        
    except Exception as e:
        print(f"计算token数量时出错: {str(e)}")
        # 如果出错，返回一个默认值
        return 1000

def create_default_param_mapping():
    """创建默认参数映射字典"""
    return {
        # 数据处理相关参数
        'input_dir': '/data/lhc/datasets/DEAP/data_preprocessed_python',  # 输入目录，DEAP数据集位置
        'output_dir': '/data/lhc/datasets_new/deap',  # 输出目录，用于保存生成的JSON文件
        'max_files': None,  # 最大处理文件数，None表示处理所有文件
        'file_pattern': '*.dat',  # 文件匹配模式
        
        # EEG信号处理参数
        'sampling_rate': 128,  # 采样率(Hz)，DEAP数据集默认128Hz
        'window_length_sec': 2,  # 窗口长度(秒)
        'window_step_sec': 0.1,  # 窗口移动步长(秒)，默认为0.5秒
        'channel_indices': [13, 16],  # 使用的通道索引，默认使用13(PO3)和16(Pz)
        'spectral_bands': [4, 8, 12, 16, 25, 45],  # 频段边界
        
        # 数据增强参数
        'add_noise': False,  # 是否添加噪声增强数据
        'noise_level': 0.05,  # 噪声水平
        
        # 特征提取参数
        'input_type': 1,  # 输入数据类型 (1:仅频谱, 2:仅信号, 3:两者都包含)
        'include_spectral': True,  # 是否包含频谱特征
        'normalize_features': True,  # 是否对频谱特征进行标准化处理
        
        # 数据集划分和平衡参数
        'test_ratio': 0.1,  # 测试集比例
        'balance_strategy': 'sqrt_inverse',  # 数据平衡策略 (sqrt_inverse/linear_inverse/none)
        'balance_ratio': 0.5,  # 平衡系数 (0-1)
        
        # 运行配置参数
        'n_jobs': 12,  # 并行处理的进程数，-1表示使用所有核心
        'verbose': True,  # 是否显示详细处理信息
        'device': 'cuda',  # 运行设备，可选cpu、cuda
        
        # 模型相关参数
        'tokenizer_path': '/data/lhc/models/Llama-3.2-1B-Instruct',  # 分词器路径
        
        # 输出控制参数
        'clean_output': False,   # 是否清理之前的输出文件
        'overwrite': False,      # 是否覆盖同名文件
        'append_timestamp': True, # 是否在文件名后添加时间戳
    }

def process_deap_data(input_dir, output_dir, param_mapping):
    """处理DEAP数据集并生成JSON格式的数据"""
    # 从参数映射中获取参数
    window_length_sec = param_mapping.get('window_length_sec', 15)
    window_step_sec = param_mapping.get('window_step_sec', 0.5)  # 获取窗口步长参数
    channel_indices = param_mapping.get('channel_indices', [13, 16])
    test_ratio = param_mapping.get('test_ratio', 0.25)
    max_files = param_mapping.get('max_files', None)
    sampling_rate = param_mapping.get('sampling_rate', 128)
    include_spectral = param_mapping.get('include_spectral', True)
    spectral_bands = param_mapping.get('spectral_bands', [4, 8, 12, 16, 25, 45])
    add_noise = param_mapping.get('add_noise', False)
    noise_level = param_mapping.get('noise_level', 0.05)
    balance_strategy = param_mapping.get('balance_strategy', 'sqrt_inverse')
    verbose = param_mapping.get('verbose', True)
    input_type = param_mapping.get('input_type', 3)
    balance_ratio = param_mapping.get('balance_ratio', 0.5)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有数据文件
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
    all_files.sort()  # 确保按顺序处理
    
    # 如果指定了最大文件数，限制处理的文件数量
    if max_files is not None and max_files > 0:
        all_files = all_files[:max_files]
        if verbose:
            print(f"已限制处理文件数量为 {max_files} 个")
    
    if verbose:
        print(f"测试集比例: {test_ratio}")
    
    all_samples = []
    emotion_samples = {0: [], 1: [], 2: [], 3: []}  # 按情绪类别存储样本
    
    if verbose:
        print(f"开始处理DEAP数据文件，共{len(all_files)}个文件...")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 首先处理所有数据
    for file_name in tqdm(all_files, disable=not verbose):
        file_path = os.path.join(input_dir, file_name)
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            if verbose:
                print(f"无法加载文件 {file_path}: {str(e)}")
            continue
        
        # 获取数据和标签
        eeg_data = data['data']
        labels = data['labels']
        
        # 处理每个试验
        for trial_idx in range(eeg_data.shape[0]):
            trial_data = eeg_data[trial_idx]
            channel_data = trial_data[channel_indices, :]
            trial_labels = labels[trial_idx]
            
            # 计算窗口参数
            segment_length = int(window_length_sec * sampling_rate)
            step_length = int(window_step_sec * sampling_rate)
            total_samples = trial_data.shape[1]
            useful_samples = total_samples - (3 * sampling_rate)  # 去掉最后3秒
            
            # 使用移动窗口而不是连续窗口
            # 计算可能的起始位置数量
            num_starts = (useful_samples - segment_length) // step_length + 1
            
            # 处理每个片段
            for seg_idx in range(num_starts):
                start_idx = seg_idx * step_length
                end_idx = start_idx + segment_length
                
                # 确保不超过有效样本范围
                if end_idx > useful_samples:
                    break
                
                # 提取片段
                segment_data = channel_data[:, start_idx:end_idx]
                
                # 添加噪声（如果需要）
                if add_noise:
                    noise = np.random.normal(0, noise_level, segment_data.shape)
                    segment_data = segment_data + noise
                
                # 创建样本
                sample = create_llm_sample(
                    segment_data, 
                    trial_labels, 
                    ["EEG Fpz-Cz", "EEG Pz-Oz"], 
                    sampling_rate,
                    input_type=input_type,
                    include_spectral=include_spectral,
                    spectral_bands=spectral_bands
                )
                
                # 将样本添加到对应情绪类别的列表中
                emotion_category = int(sample['output'])
                emotion_samples[emotion_category].append(sample)
                all_samples.append(sample)
    
    # 分层划分训练集和测试集
    train_samples = []
    test_samples = []
    
    # 记录训练集平衡前的分布
    train_dist_before = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for emotion_category in emotion_samples:
        samples = emotion_samples[emotion_category]
        n_samples = len(samples)
        n_test = int(n_samples * test_ratio)
        
        # 随机打乱该类别的样本
        np.random.shuffle(samples)
        
        # 划分训练集和测试集
        test_samples.extend(samples[:n_test])
        category_train_samples = samples[n_test:]
        train_samples.extend(category_train_samples)
        
        # 记录训练集平衡前的分布
        train_dist_before[emotion_category] = len(category_train_samples)
    
    # 对训练集进行平衡处理
    balanced_train_samples = balance_samples(
        train_samples,
        strategy=balance_strategy,
        balance_ratio=balance_ratio
    )
    
    # 记录平衡后的分布
    train_dist_after = {0: 0, 1: 0, 2: 0, 3: 0}
    for sample in balanced_train_samples:
        category = int(sample['output'])
        train_dist_after[category] += 1
    
    if verbose:
        print(f"\n数据集划分结果:")
        print(f"总样本数: {len(all_samples)}")
        print(f"训练集样本数: {len(balanced_train_samples)} ({len(balanced_train_samples)/len(all_samples)*100:.2f}%)")
        print(f"测试集样本数: {len(test_samples)} ({len(test_samples)/len(all_samples)*100:.2f}%)")
        
        # 打印每个类别的分布情况
        print("\n各情绪类别的样本分布:")
        for emotion in sorted(emotion_samples.keys()):
            total = len(emotion_samples[emotion])
            train_count = sum(1 for s in balanced_train_samples if int(s['output']) == emotion)
            test_count = sum(1 for s in test_samples if int(s['output']) == emotion)
            print(f"类别 {emotion}: 总数={total}, 训练集={train_count}, 测试集={test_count}")
    
    # 计算前10条数据的平均token数
    avg_tokens = calculate_average_tokens(all_samples[:10] if len(all_samples) >= 10 else all_samples)
    
    # 计算处理时间
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # 更新数据集信息
    dataset_info = {
        'total_samples': len(all_samples),
        'train_samples': len(balanced_train_samples),
        'test_samples': len(test_samples),
        'avg_tokens': avg_tokens,
        'processing_time': processing_time,
        'files_processed': len(all_files),
        'distributions': {
            'train_before': train_dist_before,
            'train_after': train_dist_after,
            'test_before': {k: sum(1 for s in test_samples if int(s['output']) == k) for k in range(4)},
            'test_after': {k: sum(1 for s in test_samples if int(s['output']) == k) for k in range(4)}
        }
    }
    
    # 准备输出目录，传入清理文件的选项
    train_dir, test_dir, all_dir = prepare_output_directories(
        output_dir, 
        clean_files=param_mapping.get('clean_output', False)
    )
    
    # 生成输出文件名
    base_filename = generate_output_filename(
        param_mapping,
        len(all_files) if max_files is None else max_files,
        avg_tokens
    )
    
    # 保存到相应目录
    train_file = os.path.join(train_dir, f"{base_filename}_train.json")
    test_file = os.path.join(test_dir, f"{base_filename}_test.json")
    all_file = os.path.join(all_dir, f"{base_filename}_all.json")
    
    # 检查是否需要覆盖文件
    if param_mapping.get('overwrite', False) or not os.path.exists(train_file):
        # 保存JSON文件
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_train_samples, f, ensure_ascii=False, indent=2)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"数据已保存到:")
            print(f"  训练文件: {train_file}")
            print(f"  测试文件: {test_file}")
            print(f"  全部数据: {all_file}")
    else:
        if verbose:
            print(f"文件已存在，未覆盖: {train_file}")
            print(f"如需覆盖，请使用 --overwrite 参数")
    
    # 更新dataset_info.json
    train_filename = os.path.basename(train_file)
    update_dataset_info(train_filename, output_dir)
    
    # 创建Excel报告
    create_excel_report(param_mapping, dataset_info, output_dir)
    
    return all_samples, balanced_train_samples, test_samples

def update_dataset_info(train_filename, output_dir):
    """更新dataset_info.json文件中的DEAP数据集信息
    
    参数:
        train_filename: 训练集文件名
        output_dir: 输出目录
    """
    dataset_info_path = "/data/lhc/projects/LLaMA-Factory/data/dataset_info.json"
    backup_path = dataset_info_path + ".bak"
    
    # 备份原始文件
    try:
        shutil.copy2(dataset_info_path, backup_path)
        print(f"已创建备份文件: {backup_path}")
    except Exception as e:
        print(f"创建备份文件失败: {str(e)}")
    
    # 创建新的dataset_info数据
    train_key = train_filename.replace('.json', '')
    
    # 确定相对路径
    relative_path = os.path.relpath(output_dir, "/data/lhc/projects/LLaMA-Factory/data")
    
    # 新数据集信息（简化格式）
    new_info = {
        "file_name": os.path.join(relative_path, "train", train_filename),
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    
    # 尝试读取并更新现有文件
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            dataset_info = json.loads(content)
            dataset_info[train_key] = new_info
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            print(f"更新dataset_info.json文件成功，添加了 {train_key} 数据集")
            return
        except json.JSONDecodeError:
            print("JSON解析失败，将创建新文件")
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
    
    # 如果上述步骤失败，创建新文件
    try:
        dataset_info = {train_key: new_info}
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"已创建新的dataset_info.json文件，包含 {train_key} 数据集")
    except Exception as e:
        print(f"创建新文件失败: {str(e)}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='处理DEAP数据集并生成JSON格式的数据')
    
    # 获取默认参数映射
    default_params = create_default_param_mapping()
    
    # 添加命令行参数
    parser.add_argument('--input_dir', type=str, default=default_params['input_dir'],
                        help='DEAP数据集所在目录')
    parser.add_argument('--output_dir', type=str, default=default_params['output_dir'],
                        help='输出目录，用于保存JSON文件')
    parser.add_argument('--window_length_sec', type=int, default=default_params['window_length_sec'],
                        help='窗口长度(秒)')
    parser.add_argument('--window_step_sec', type=float, default=default_params['window_step_sec'],
                        help='窗口移动步长(秒)，控制窗口重叠程度')
    parser.add_argument('--test_ratio', type=float, default=default_params['test_ratio'],
                        help='测试集比例')
    parser.add_argument('--channel_indices', type=str, default='13,16',
                        help='要使用的通道索引，以逗号分隔')
    parser.add_argument('--sampling_rate', type=int, default=default_params['sampling_rate'],
                        help='采样率(Hz)')
    parser.add_argument('--add_noise', type=bool, default=default_params['add_noise'],
                        help='是否添加噪声增强数据')
    parser.add_argument('--noise_level', type=float, default=default_params['noise_level'],
                        help='添加的噪声水平')
    parser.add_argument('--include_spectral', type=bool, default=default_params['include_spectral'],
                        help='是否包含频谱特征')
    parser.add_argument('--spectral_bands', type=str, default='4,8,12,16,25,45',
                        help='频谱分析的频段边界，以逗号分隔')
    parser.add_argument('--clean_output', action='store_true',
                        help='清理输出目录中现有的json文件')
    parser.add_argument('--max_files', type=int, default=default_params['max_files'],
                        help='最大处理文件数')
    parser.add_argument('--balance_strategy', type=str, default=default_params['balance_strategy'],
                        choices=['sqrt_inverse', 'linear_inverse', 'none'],
                        help='数据集平衡策略')
    parser.add_argument('--balance_ratio', type=float, default=default_params['balance_ratio'],
                        help='数据平衡系数 (0-1)')
    parser.add_argument('--input_type', type=int, default=default_params['input_type'],
                        choices=[1, 2, 3],
                        help='输入数据类型 (1:仅频谱, 2:仅信号, 3:两者都包含)')
    parser.add_argument('--n_jobs', type=int, default=default_params['n_jobs'],
                        help='并行处理的进程数')
    parser.add_argument('--device', type=str, default=default_params['device'],
                        choices=['cpu', 'cuda'],
                        help='运行设备')
    parser.add_argument('--verbose', type=bool, default=default_params['verbose'],
                        help='是否显示详细处理信息')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖同名输出文件')
    parser.add_argument('--append_timestamp', action='store_true', default=True,
                        help='在输出文件名后添加时间戳以避免覆盖')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建参数映射字典
    param_mapping = default_params.copy()
    
    # 更新参数映射
    for key, value in vars(args).items():
        if key in param_mapping:
            param_mapping[key] = value
    
    # 特别处理布尔型参数
    param_mapping['clean_output'] = args.clean_output
    param_mapping['overwrite'] = args.overwrite
    param_mapping['append_timestamp'] = args.append_timestamp
    
    # 处理特殊参数
    try:
        param_mapping['channel_indices'] = [int(idx) for idx in args.channel_indices.split(',')]
    except ValueError:
        print(f"警告: 通道索引格式错误，使用默认值{default_params['channel_indices']}")
    
    try:
        param_mapping['spectral_bands'] = [float(band) for band in args.spectral_bands.split(',')]
    except ValueError:
        print(f"警告: 频谱边界格式错误，使用默认值{default_params['spectral_bands']}")
    
    # 处理DEAP数据
    all_samples, train_samples, test_samples = process_deap_data(
        input_dir=param_mapping['input_dir'],
        output_dir=param_mapping['output_dir'],
        param_mapping=param_mapping
    )

if __name__ == "__main__":
    main()
