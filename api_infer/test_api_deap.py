#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI
from transformers.utils.versions import require_version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import re

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

# 定义睡眠阶段标签，避免重复硬编码
SLEEP_STAGE_LABELS = [
    'Wake (W)', 
    'NREM Stage 1 (N1)', 
    'NREM Stage 2 (N2)', 
    'NREM Stage 3 (N3)', 
    'NREM Stage 4 (N4)', 
    'REM Sleep (R)'
]

# 添加DEAP情绪标签定义
DEAP_EMOTION_LABELS = [
    'LALV (低唤醒低效价)', 
    'LAHV (低唤醒高效价)', 
    'HALV (高唤醒低效价)', 
    'HAHV (高唤醒高效价)'
]

# DEAP情绪简短标签（用于混淆矩阵等）
DEAP_EMOTION_SHORT_LABELS = ['LALV', 'LAHV', 'HALV', 'HAHV']

# 添加命令行参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description='评估大语言模型在脑电数据分析任务上的表现')
    
    # 添加数据集类型参数
    parser.add_argument('--dataset_type', type=str, choices=['sleep', 'deap'], default='sleep',
                       help='测试数据集类型: sleep(睡眠阶段分类)或deap(情绪状态分类)')
    
    # 添加输入输出路径参数
    parser.add_argument('--input_path', type=str, default='/data/lhc/datasets_new/emotion/test/sleep_st_44_100hz_eeg15s-step15s_emo2.0s-step1s_win_all_tok13101_bal0.5_sqrt_inverse_202504272332_test.json',
                       help='测试数据集的路径')
    
    parser.add_argument('--output_dir', type=str, default='/data/lhc/results',
                       help='结果保存目录')
    
    # 其他可选参数
    parser.add_argument('--model_name', type=str, default='emotion_st44',
                       help='被测试的模型名称')
    
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大测试样本数量，默认测试全部样本')
    
    parser.add_argument('--print_interval', type=int, default=10,
                       help='打印中间结果的间隔')
    
    parser.add_argument('--save_interval', type=int, default=100,
                       help='保存中间结果的间隔')
    
    # 添加最大token数量参数
    parser.add_argument('--max_tokens', type=int, default=None,
                       help='API请求的最大token数量，默认根据数据集类型自动设置: sleep=13500, deap=400')
    
    return parser.parse_args()

def load_test_data(file_path, max_samples=None):
    """加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"已加载{len(test_data)}条测试数据")
        
        # 如果设置了最大样本数，随机选择样本
        if max_samples and max_samples < len(test_data):
            indices = np.random.choice(len(test_data), max_samples, replace=False)
            test_data = [test_data[i] for i in indices]
            print(f"随机选择了{len(test_data)}条数据用于测试")
        
        return test_data
    
    except Exception as e:
        print(f"加载测试数据时出错: {str(e)}")
        traceback.print_exc()
        return []


def get_api_prediction(client, messages, max_retries=5, retry_delay=2, dataset_type='deap', max_tokens=None):
    """通过API获取模型预测，包含重试机制"""
    # 如果未指定max_tokens，根据数据集类型设置默认值
    if max_tokens is None:
        max_tokens = 13500 if dataset_type == 'sleep' else 400
        
    for attempt in range(max_retries):
        try:
            # 直接使用传入的messages列表
            start_time = time.time()
            result = client.chat.completions.create(
                messages=messages, # 直接使用传入的 messages
                model="test",
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=30  # 设置超时时间为30秒
            )
            end_time = time.time()
            
            # 获取响应
            response = result.choices[0].message.content
            
            # 提取数字，根据数据集类型决定有效范围
            prediction = None
            valid_range = [0, 1, 2, 3] if dataset_type == 'deap' else [0, 1, 2, 3, 4, 5]
            # 使用正则表达式查找第一个符合条件的数字
            match = re.search(r'\b([0-{}])\b'.format(valid_range[-1]), response)
            if match:
                prediction = int(match.group(1))
            else:
                # 如果正则没匹配到，尝试原来的字符遍历方法作为后备
                for char in response:
                    if char.isdigit() and int(char) in valid_range:
                        prediction = int(char)
                        break

            return response, prediction, end_time - start_time
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API调用出错 (尝试 {attempt+1}/{max_retries}): {e}，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"API调用失败，已达到最大重试次数 ({max_retries}): {e}")
                return "API调用失败", None, 0


def evaluate_model(client, test_data, print_interval=10, save_interval=500, save_dir='/data/lhc/results', 
                  model_name="unknown_model", test_set_name="unknown_dataset", dataset_type='sleep', max_tokens=None):
    """评估模型性能
    
    Args:
        client: OpenAI客户端
        test_data: 测试数据
        print_interval: 打印中间结果的间隔
        save_interval: 保存中间结果的间隔
        save_dir: 保存结果的目录
        model_name: 模型名称
        test_set_name: 测试集名称
        dataset_type: 数据集类型，'sleep'或'deap'
        max_tokens: API请求的最大token数量
    """
    # 创建一个包含模型名称和测试集名称的文件夹来保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(save_dir, f"{model_name}_{test_set_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建中间结果保存函数
    def save_intermediate_results(current_idx, current_metrics, current_results):
        """保存中间结果
        
        Args:
            current_idx: 当前处理的样本索引
            current_metrics: 当前的评估指标
            current_results: 当前的详细结果
        """
        intermediate_dir = os.path.join(result_dir, f"intermediate_{current_idx}")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 保存模型和测试集信息
        with open(os.path.join(intermediate_dir, "info.json"), "w") as f:
            json.dump({
                "model_name": model_name,
                "test_set_name": test_set_name,
                "samples_processed": current_idx,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)
        
        # 保存指标 - 确保ndarray转换为list
        metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in current_metrics.items()}
        with open(os.path.join(intermediate_dir, "metrics.json"), "w") as f:
             json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

        # 保存结果
        with open(os.path.join(intermediate_dir, "results.json"), "w") as f:
            json.dump(current_results, f, indent=4)
        
        # 保存混淆矩阵
        if "confusion_matrix" in current_metrics and current_metrics["confusion_matrix"]: # Check if exists and not empty
             plt.figure(figsize=(10, 8))
             cm = np.array(current_metrics['confusion_matrix']) # Convert list back to numpy array for heatmap
             
             # 使用 metrics 中保存的标签
             labels = current_metrics.get("confusion_matrix_labels", [])
             if labels: # Only plot if labels are available
                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                             xticklabels=labels,
                             yticklabels=labels)
                 plt.xlabel('Predicted Label')
                 plt.ylabel('True Label')
                 plt.title(f'Intermediate Confusion Matrix (Samples: {current_idx})')
                 plt.savefig(os.path.join(intermediate_dir, "confusion_matrix.png"))
             plt.close()

        # 保存Excel文件 (注意：save_results_to_excel 也需要能处理新的 metrics 结构)
        save_results_to_excel(current_results, current_metrics, test_data[:current_idx], intermediate_dir, model_name, test_set_name, dataset_type)
        
        print(f"\n中间结果已保存至: {intermediate_dir}")
    
    print(f"开始评估模型性能... 数据集类型: {dataset_type}")
    results = []
    true_labels = []
    pred_labels = []
    responses = []
    
    # 根据数据集类型显示不同的标签信息
    if dataset_type == 'sleep':
        print("睡眠阶段标签顺序:", SLEEP_STAGE_LABELS)
    else:  # deap
        print("情绪标签顺序:", DEAP_EMOTION_LABELS)
    
    # 初始化指标 (不再需要在这里初始化，calculate_metrics会处理)
    
    # 使用tqdm创建进度条
    pbar = tqdm(test_data, desc="Test Samples", ncols=100)
    
    # 用于实时显示性能指标
    correct_count = 0
    total_count = 0 # Use total_count for pbar display based on processed samples
    total_time = 0.0
    
    for idx, data in enumerate(pbar):
        true_label = int(data["output"])
        
        try:
            # 从data中获取 system, instruction, input 内容
            system_content = data.get("system", "")
            instruction_content = data.get("instruction", "")
            input_content = data.get("input", "") # input 字段通常是必须的

            # 构建 user 角色的 content
            user_content = f"{instruction_content}\n\n{input_content}" if instruction_content else input_content

            # 构建 messages 列表
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})
            
            # 获取预测，传入构建好的 messages
            response, prediction, infer_time = get_api_prediction(
                client, 
                messages,  # 传入 messages 列表
                dataset_type=dataset_type, 
                max_tokens=max_tokens
            )
            
            total_time += infer_time
            
            # 解析预测结果，根据数据集类型有不同的处理方式
            # Note: prediction is now the extracted int or None
            correct = False # Default to False
            if prediction is not None: # Check if prediction was successfully extracted
                 valid_range = [0, 1, 2, 3] if dataset_type == 'deap' else [0, 1, 2, 3, 4, 5]
                 if prediction in valid_range:
                     correct = (prediction == true_label)
                     if correct:
                         correct_count += 1
                 else:
                     prediction = -1 # Mark as invalid if extracted but out of range
            else:
                 prediction = -1 # Mark as invalid if extraction failed (returned None)

            total_count += 1 # Increment processed samples count
            
            # 更新进度条状态
            current_acc = correct_count / total_count if total_count > 0 else 0
            avg_time = total_time / total_count if total_count > 0 else 0
            
            # 设置进度条描述，实时显示准确率和平均推理时间
            pbar.set_postfix({
                'acc': f'{current_acc:.2%}',
                'avg_time': f'{avg_time:.2f}s',
                '当前预测': f'{prediction}', # Shows -1 if invalid
                '真实标签': f'{true_label}',
                '正确': correct
            })
            
            result = {
                "id": idx,
                "true_label": true_label,
                "prediction": prediction, # Records the actual prediction (-1 if invalid)
                "response": response,
                "correct": correct,
                "inference_time": infer_time
            }
            
            results.append(result)
            true_labels.append(true_label)
            # 注意：如果prediction是-1，表示提取失败，计算指标时需要处理
            pred_labels.append(prediction) # 直接记录预测值，包括-1
            responses.append(response)
            
            # 更新指标 (更新逻辑移至 calculate_metrics 函数)
            
            # 每隔print_interval次测试打印一次相关的测试数据
            if (idx + 1) % print_interval == 0 or (idx + 1) == len(test_data):
                # 注意：传入原始的 pred_labels (包含-1)
                current_metrics = calculate_metrics(true_labels, pred_labels, total_time, dataset_type) 
                print(f"\n已处理{idx + 1}个样本")
                print(f"当前总体准确率 (基于有效预测): {current_metrics['accuracy']:.4f}")
                print(f"当前宏平均精确率: {current_metrics['precision_macro']:.4f}")
                print(f"当前宏平均召回率: {current_metrics['recall_macro']:.4f}")
                print(f"当前宏平均F1分数: {current_metrics['f1_macro']:.4f}")
                print(f"当前未知预测比例: {current_metrics['unknown_ratio']:.4f}")
                print(f"当前错误预测比例: {current_metrics['error_ratio']:.4f}") # Added error ratio print
                print(f"当前平均推理时间 (所有样本): {current_metrics['avg_time_per_sample']:.4f} 秒/样本")

                # 显示各类别指标
                if dataset_type == 'sleep':
                    print("各睡眠阶段详细指标:")
                    class_labels_full = SLEEP_STAGE_LABELS
                    short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
                else: # deap
                    print("各情绪类别详细指标:")
                    class_labels_full = DEAP_EMOTION_LABELS
                    short_labels = DEAP_EMOTION_SHORT_LABELS

                for class_idx, label_name in enumerate(class_labels_full):
                     metric_key = short_labels[class_idx]
                     class_metric = current_metrics['class_metrics'].get(metric_key, {})
                     # Use class_accuracies calculated in metrics
                     class_acc = current_metrics['class_accuracies'].get(class_idx, 0.0)
                     print(f"  {label_name}: 准确率={class_acc:.4f}, "
                           f"精确率={class_metric.get('precision', 0.0):.4f}, "
                           f"召回率={class_metric.get('recall', 0.0):.4f}, "
                           f"F1分数={class_metric.get('f1', 0.0):.4f}")

                print("混淆矩阵 (基于有效预测):")
                # Print confusion matrix using labels from metrics
                cm_labels = current_metrics.get("confusion_matrix_labels", [])
                cm_data = current_metrics.get("confusion_matrix", [])
                if cm_labels and cm_data:
                    print("    " + " ".join(f"{label:^5s}" for label in cm_labels))
                    for i, label in enumerate(cm_labels):
                        print(f"{label:^5s}" + " ".join(f"{cm_data[i][j]:^5d}" for j in range(len(cm_labels))))
                else:
                    print("  (无有效预测，无法生成混淆矩阵)")

                
            # 每隔save_interval个样本保存一次中间结果
            if (idx + 1) % save_interval == 0:
                 # 注意：传入原始的 pred_labels (包含-1)
                current_metrics = calculate_metrics(true_labels, pred_labels, total_time, dataset_type)
                save_intermediate_results(idx + 1, current_metrics, results)
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            traceback.print_exc()
            # 记录失败样本信息，避免丢失标签对应关系
            results.append({
                "id": idx,
                "true_label": true_label,
                "prediction": -2, # 使用-2表示处理出错
                "response": f"Error: {e}",
                "correct": False,
                "inference_time": 0
            })
            true_labels.append(true_label)
            pred_labels.append(-2) # 记录错误标记
            responses.append(f"Error: {e}")
            continue # 继续处理下一个样本
    
    # 计算最终指标 (传入包含-1和-2的pred_labels)
    final_metrics = calculate_metrics(true_labels, pred_labels, total_time, dataset_type)
    
    # 打印最终结果
    print("\n" + "=" * 50)
    if dataset_type == 'sleep':
        print("睡眠阶段分类测试结果")
        class_labels_full = SLEEP_STAGE_LABELS
        short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
    else:
        print("情绪状态分类测试结果")
        class_labels_full = DEAP_EMOTION_LABELS
        short_labels = DEAP_EMOTION_SHORT_LABELS

    print("=" * 50)
    print(f"总样本数: {final_metrics['total_samples']}")
    print(f"有效预测样本数: {final_metrics['valid_predictions']}")
    print(f"提取失败样本数: {final_metrics['unknown_predictions']} ({final_metrics['unknown_ratio']:.2%})")
    print(f"处理错误样本数: {final_metrics['error_predictions']} ({final_metrics['error_ratio']:.2%})") # Added error count print
    print(f"总体准确率 (基于有效预测): {final_metrics['accuracy']:.2%}")
    print(f"宏平均精确率: {final_metrics['precision_macro']:.2%}")
    print(f"宏平均召回率: {final_metrics['recall_macro']:.2%}")
    print(f"宏平均F1分数: {final_metrics['f1_macro']:.2%}")
    print(f"平均推理时间 (所有样本): {final_metrics['avg_time_per_sample']:.4f}秒/样本")

    # 打印每个标签的详细指标
    print(f"\n各{dataset_type}类别详细指标 (基于有效预测):")
    if class_labels_full:
        for class_idx, label_name in enumerate(class_labels_full):
            metric_key = short_labels[class_idx]
            class_metric = final_metrics['class_metrics'].get(metric_key, {})
            class_acc = final_metrics['class_accuracies'].get(class_idx, 0.0)
            
            print(f"  {label_name}: 准确率={class_acc:.2%}, "
                  f"精确率={class_metric.get('precision', 0.0):.2%}, "
                  f"召回率={class_metric.get('recall', 0.0):.2%}, "
                  f"F1分数={class_metric.get('f1', 0.0):.2%}")
    else:
        print("  无类别数据")

    # 打印混淆矩阵
    print("\n混淆矩阵 (基于有效预测):")
    cm_labels = final_metrics.get("confusion_matrix_labels", [])
    cm_data = final_metrics.get("confusion_matrix", [])
    if cm_labels and cm_data:
        print("    " + " ".join(f"{label:^5s}" for label in cm_labels))
        for i, label in enumerate(cm_labels):
            print(f"{label:^5s}" + " ".join(f"{cm_data[i][j]:^5d}" for j in range(len(cm_labels))))
    else:
         print("  (无有效预测，无法生成混淆矩阵)")
    
    print("=" * 50)
    
    # 保存结果到指定目录
    save_results(results, final_metrics, test_data, save_dir, model_name=model_name, test_set_name=test_set_name, dataset_type=dataset_type)
    
    return final_metrics


def save_results(results, metrics, test_data, save_dir, model_name="unknown_model", 
                test_set_name="unknown_dataset", dataset_type='sleep'):
    """保存测试结果
    
    参数:
        results: 测试结果列表
        metrics: 评估指标字典
        test_data: 测试数据列表
        save_dir: 结果保存目录
        model_name: 模型名称
        test_set_name: 测试集名称
        dataset_type: 数据集类型，'sleep'或'deap'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建包含数据集类型的结果目录
    result_dir = os.path.join(save_dir, f"{dataset_type}_{model_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存原始预测结果
    results_file = os.path.join(result_dir, f"{dataset_type}_{test_set_name}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存评估指标
    metrics_file = os.path.join(result_dir, f"{dataset_type}_{test_set_name}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 创建并保存混淆矩阵可视化
    if "confusion_matrix" in metrics and len(metrics["confusion_matrix"]) > 0:
        confusion_mat = np.array(metrics["confusion_matrix"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=metrics["confusion_matrix_labels"],
                   yticklabels=metrics["confusion_matrix_labels"])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{dataset_type.upper()} 混淆矩阵')
        plt.tight_layout()
        cm_path = os.path.join(result_dir, f"{dataset_type}_{test_set_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
    
    # 保存Excel格式的详细结果
    save_results_to_excel(results, metrics, test_data, result_dir, model_name, test_set_name, dataset_type)
    
    # 更新汇总Excel文件
    update_summary_excel(results, metrics, test_data, save_dir, model_name, test_set_name, timestamp, dataset_type)
    
    print(f"结果已保存到: {result_dir}")
    return result_dir


def update_summary_excel(results, metrics, test_data, save_dir, model_name, test_set_name, timestamp, dataset_type='sleep'):
    """更新汇总Excel文件 (需要更新以匹配新的metrics结构)
    
    参数:
        results: 测试结果列表 (可能不需要，metrics包含了汇总信息)
        metrics: 评估指标字典 (新结构)
        test_data: 测试数据列表 (用于获取总样本数)
        save_dir: 结果保存目录
        model_name: 模型名称
        test_set_name: 测试集名称
        timestamp: 时间戳
        dataset_type: 数据集类型，'sleep'或'deap'
    """
    # 创建info目录
    info_dir = os.path.join(save_dir, 'info')
    os.makedirs(info_dir, exist_ok=True)
    
    # 汇总文件路径
    summary_file = os.path.join(info_dir, f"{dataset_type}_test_summary.xlsx")
    
    # 准备新的汇总行数据 (使用新指标)
    new_row = {
        '测试ID': timestamp,
        '模型名称': model_name,
        '测试集名称': test_set_name,
        '数据集类型': dataset_type,
        '样本数量': metrics.get("total_samples", 'N/A'),
        '有效预测数': metrics.get("valid_predictions", 'N/A'),
        '提取失败数': metrics.get("unknown_predictions", 'N/A'),
        '处理错误数': metrics.get("error_predictions", 'N/A'), # Added error count
        '准确率': metrics.get("accuracy", 'N/A'),
        '宏平均精确率': metrics.get("precision_macro", 'N/A'),
        '宏平均召回率': metrics.get("recall_macro", 'N/A'),
        '宏平均F1分数': metrics.get("f1_macro", 'N/A'),
        '未知预测比例': metrics.get("unknown_ratio", 'N/A'),
        '错误预测比例': metrics.get("error_ratio", 'N/A'), # Added error ratio
        '总预测时间(秒)': metrics.get("total_time", 'N/A'),
        '平均预测时间(秒/样本)': metrics.get("avg_time_per_sample", 'N/A'),
        '测试时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 添加类别指标 (使用新指标)
    class_metrics_dict = metrics.get('class_metrics', {})
    for class_name, class_metrics_values in class_metrics_dict.items():
        prefix = f"{class_name}_"
        new_row[f"{prefix}精确率"] = class_metrics_values.get('precision', 'N/A')
        new_row[f"{prefix}召回率"] = class_metrics_values.get('recall', 'N/A')
        new_row[f"{prefix}F1分数"] = class_metrics_values.get('f1', 'N/A')

    # 添加每个类别的准确率 (使用新指标)
    class_accuracies = metrics.get('class_accuracies', {})
    if dataset_type == 'sleep':
        short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
    else:
        short_labels = DEAP_EMOTION_SHORT_LABELS

    for class_idx, label_name in enumerate(short_labels):
         new_row[f"{label_name}_准确率"] = class_accuracies.get(class_idx, 'N/A')

    # 尝试加载现有的汇总文件
    try:
        # Specify engine if needed, e.g., engine='openpyxl'
        summary_df = pd.read_excel(summary_file) 
    except FileNotFoundError:
        # 如果文件不存在，创建新的DataFrame
        summary_df = pd.DataFrame()
    except Exception as e:
        print(f"读取汇总文件 {summary_file} 出错: {e}, 将创建新文件。")
        summary_df = pd.DataFrame() # Fallback to new DataFrame on other errors

    # 使用 pd.concat 添加新行
    new_row_df = pd.DataFrame([new_row])
    # Ensure columns match, handling new columns added over time
    summary_df = pd.concat([summary_df, new_row_df], ignore_index=True) 
    
    # 保存更新后的汇总文件
    try:
        # Specify engine if needed, e.g., engine='openpyxl'
        summary_df.to_excel(summary_file, index=False) 
        print(f"测试汇总已更新: {summary_file}")
    except Exception as e:
        print(f"保存汇总文件 {summary_file} 出错: {e}")


def calculate_metrics(true_labels, pred_labels, total_time, dataset_type='sleep'):
    """计算评估指标
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表 (可能包含 -1 表示提取失败, -2 表示处理错误)
        total_time: 总推理时间
        dataset_type: 数据集类型 ('sleep' 或 'deap')
    """
    # 过滤掉无效预测 (-1 和 -2)
    valid_indices = [i for i, pred in enumerate(pred_labels) if pred >= 0]
    
    # 计算未知预测的数量（提取失败）
    unknown_predictions = sum(1 for pred in pred_labels if pred == -1)
    # 计算处理错误的数量
    error_predictions = sum(1 for pred in pred_labels if pred == -2)
    
    num_total_samples = len(true_labels)
    
    # 确定分类数量
    num_classes = 6 if dataset_type == 'sleep' else 4
    
    # 确定标签名称
    if dataset_type == 'sleep':
        short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
    else: # deap
        short_labels = DEAP_EMOTION_SHORT_LABELS
    
    # 处理没有有效预测的情况
    if not valid_indices:
        class_metrics_empty = {label: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for label in short_labels}
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'avg_inference_time': total_time / num_total_samples if num_total_samples > 0 else 0.0, # Kept for backward compatibility maybe?
            'confusion_matrix': np.zeros((num_classes, num_classes), dtype=int).tolist(), # 返回列表以便JSON序列化
            'class_accuracies': {i: 0.0 for i in range(num_classes)},
            'class_f1s': {i: 0.0 for i in range(num_classes)}, # Kept for backward compatibility maybe?
            'class_metrics': class_metrics_empty,
            'confusion_matrix_labels': short_labels,
            'total_time': total_time,
            'total_samples': num_total_samples,
            'valid_predictions': 0,
            'unknown_predictions': unknown_predictions, # 提取失败的数量
            'error_predictions': error_predictions, # 处理错误的数量
            'unknown_ratio': unknown_predictions / num_total_samples if num_total_samples > 0 else 0.0,
            'error_ratio': error_predictions / num_total_samples if num_total_samples > 0 else 0.0,
            'avg_time_per_sample': total_time / num_total_samples if num_total_samples > 0 else 0.0
        }
    
    # 提取有效的真实标签和预测标签
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [pred_labels[i] for i in valid_indices]
    
    # 计算总体指标 (基于有效预测)
    accuracy = accuracy_score(valid_true, valid_pred)
    precision_macro = precision_score(valid_true, valid_pred, average='macro', zero_division=0)
    recall_macro = recall_score(valid_true, valid_pred, average='macro', zero_division=0)
    f1_macro = f1_score(valid_true, valid_pred, average='macro', zero_division=0)
    avg_inference_time = total_time / num_total_samples if num_total_samples > 0 else 0 # 平均时间基于总样本数
    confusion_matrix_result = confusion_matrix(valid_true, valid_pred, labels=range(num_classes))
    
    # 计算每个类别的准确率 (基于有效预测)
    class_accuracies = {}
    for class_idx in range(num_classes):
        # 找到所有真实标签为 class_idx 的有效样本
        class_true_indices = [i for i, label in enumerate(valid_true) if label == class_idx]
        if class_true_indices:
            # 在这些样本中，计算预测也为 class_idx 的比例
            correct_in_class = sum(1 for i in class_true_indices if valid_pred[i] == class_idx)
            class_accuracies[class_idx] = correct_in_class / len(class_true_indices)
        else:
             # 如果该类别在有效真实标签中没有出现，则准确率为0 (或 NaN, 但 0 更安全)
            class_accuracies[class_idx] = 0.0 

    # 计算每个类别的F1分数 (基于有效预测) - Redundant if class_metrics includes F1
    class_f1_arr = f1_score(valid_true, valid_pred, labels=range(num_classes), average=None, zero_division=0)
    class_f1s = {i: float(class_f1_arr[i]) for i in range(num_classes)}
    
    # 计算分类指标 (基于有效预测)
    class_metrics = {}
    labels_for_metrics = range(num_classes) # 使用数字标签 0, 1, 2...

    # Use average=None to get per-class scores directly
    precision_per_class = precision_score(valid_true, valid_pred, labels=labels_for_metrics, average=None, zero_division=0)
    recall_per_class = recall_score(valid_true, valid_pred, labels=labels_for_metrics, average=None, zero_division=0)
    f1_per_class = f1_score(valid_true, valid_pred, labels=labels_for_metrics, average=None, zero_division=0)

    for i, label_name in enumerate(short_labels):
        class_metrics[label_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i])
        }
        
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'avg_inference_time': avg_inference_time, # Based on total samples, maybe remove?
        'total_time': total_time,
        'confusion_matrix': confusion_matrix_result.tolist(), # 转为列表以便JSON序列化
        'class_accuracies': class_accuracies, # Per-class accuracy
        'class_f1s': class_f1s, # Per-class F1, redundant with class_metrics['f1']
        'class_metrics': class_metrics, # Contains precision, recall, f1 per class
        'confusion_matrix_labels': short_labels,
        'total_samples': num_total_samples,
        'valid_predictions': len(valid_indices),
        'unknown_predictions': unknown_predictions,
        'error_predictions': error_predictions,
        'unknown_ratio': unknown_predictions / num_total_samples if num_total_samples > 0 else 0.0,
        'error_ratio': error_predictions / num_total_samples if num_total_samples > 0 else 0.0,
        'avg_time_per_sample': total_time / num_total_samples if num_total_samples > 0 else 0.0 # Based on total samples
    }


def save_results_to_excel(results, metrics, test_data, save_dir, model_name="unknown_model", 
                          test_set_name="unknown_dataset", dataset_type='sleep'):
    """将结果保存为Excel格式 (需要更新以匹配新的metrics结构)
    
    参数:
        results: 测试结果列表
        metrics: 评估指标字典 (新结构)
        test_data: 测试数据列表 (注意: 中间保存时可能不是完整列表)
        save_dir: 结果保存目录
        model_name: 模型名称
        test_set_name: 测试集名称
        dataset_type: 数据集类型，'sleep'或'deap'
    """
    # 创建预测结果DataFrame
    results_data = []
    for r in results:
        results_data.append({
            "样本ID": r["id"],
            "真实标签": r["true_label"],
            "预测标签": r["prediction"], # 可能为 -1 或 -2
            "是否正确": r["correct"],
            "预测时间(秒)": r["inference_time"],
            "预测文本": r["response"]
        })
    
    results_df = pd.DataFrame(results_data)
    
    # 创建指标摘要DataFrame (使用新指标)
    metrics_data = {
        "指标": [
            "测试样本总数",
            "有效预测数",
            "提取失败数 (-1)",
            "处理错误数 (-2)",
            "未知预测比例 (-1)",
            "错误预测比例 (-2)",
            "准确率 (基于有效预测)",
            "宏平均精确率",
            "宏平均召回率",
            "宏平均F1分数",
            "总预测时间(秒)",
            "平均每样本预测时间(秒)"
        ],
        "值": [
            metrics.get("total_samples", 'N/A'),
            metrics.get("valid_predictions", 'N/A'),
            metrics.get("unknown_predictions", 'N/A'),
            metrics.get("error_predictions", 'N/A'), # Added error count
            f"{metrics.get('unknown_ratio', 0.0):.4f}",
            f"{metrics.get('error_ratio', 0.0):.4f}", # Added error ratio
            f"{metrics.get('accuracy', 0.0):.4f}",
            f"{metrics.get('precision_macro', 0.0):.4f}",
            f"{metrics.get('recall_macro', 0.0):.4f}",
            f"{metrics.get('f1_macro', 0.0):.4f}",
            f"{metrics.get('total_time', 0.0):.2f}",
            f"{metrics.get('avg_time_per_sample', 0.0):.4f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 创建详细类别指标DataFrame (使用新指标和标签)
    detailed_class_metrics_data = []
    
    # 根据数据集类型获取类别标签
    if dataset_type == 'sleep':
        class_labels_full = SLEEP_STAGE_LABELS
        short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
    else:
        class_labels_full = DEAP_EMOTION_LABELS
        short_labels = DEAP_EMOTION_SHORT_LABELS
    
    # 为每个类别生成详细指标
    class_accuracies = metrics.get('class_accuracies', {})
    class_metrics_dict = metrics.get('class_metrics', {})
    
    for class_idx, label_name in enumerate(class_labels_full):
        metric_key = short_labels[class_idx]
        class_metric = class_metrics_dict.get(metric_key, {})
        class_acc = class_accuracies.get(class_idx, 0.0)
            
        detailed_class_metrics_data.append({
            "类别": label_name,
            "准确率": f"{class_acc:.4f}",
            "精确率": f"{class_metric.get('precision', 0.0):.4f}",
            "召回率": f"{class_metric.get('recall', 0.0):.4f}",
            "F1分数": f"{class_metric.get('f1', 0.0):.4f}"
        })
    
    detailed_class_metrics_df = pd.DataFrame(detailed_class_metrics_data)
    
    # 简化版类别指标DataFrame (兼容旧版本) - 使用新指标
    class_metrics_data = []
    for class_name, class_metrics_values in class_metrics_dict.items():
        class_metrics_data.append({
            "类别": class_name,
            "精确率": f"{class_metrics_values.get('precision', 0.0):.4f}",
            "召回率": f"{class_metrics_values.get('recall', 0.0):.4f}",
            "F1分数": f"{class_metrics_values.get('f1', 0.0):.4f}"
        })
    
    class_metrics_df = pd.DataFrame(class_metrics_data)
    
    # 创建混淆矩阵DataFrame (使用新指标)
    confusion_matrix_list = metrics.get("confusion_matrix", [])
    confusion_matrix_labels = metrics.get("confusion_matrix_labels", [])
    if confusion_matrix_list and confusion_matrix_labels:
        confusion_df = pd.DataFrame(
            confusion_matrix_list,
            index=confusion_matrix_labels,
            columns=confusion_matrix_labels
        )
    else:
        confusion_df = pd.DataFrame() # Create empty if no valid predictions
    
    # 创建Excel文件
    # Ensure filename uses dataset_type from args/env, not just 'dataset_type' string
    excel_file = os.path.join(save_dir, f"{dataset_type}_{test_set_name}_detailed_results.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # 写入各个工作表
        results_df.to_excel(writer, sheet_name='预测结果', index=False)
        metrics_df.to_excel(writer, sheet_name='总体指标', index=False)
        detailed_class_metrics_df.to_excel(writer, sheet_name='详细类别指标', index=False)
        # class_metrics_df is somewhat redundant with detailed_class_metrics_df, decide if needed
        # class_metrics_df.to_excel(writer, sheet_name='类别指标', index=False) 
        
        if not confusion_df.empty:
            confusion_df.to_excel(writer, sheet_name='混淆矩阵')
        else:
             # Write a placeholder if empty
             pd.DataFrame({"状态": ["无有效预测数据"]}).to_excel(writer, sheet_name='混淆矩阵', index=False)
        
        # 添加测试信息工作表
        # Use len(results) for current sample count in intermediate saves
        info_data = {
            "项目": [
                "模型名称",
                "测试集名称",
                "数据集类型",
                "测试时间",
                "测试样本数 (当前)" # Clarify this might be intermediate count
            ],
            "值": [
                model_name,
                test_set_name,
                dataset_type,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(results) # Use length of results list for current count
            ]
        }
        
        pd.DataFrame(info_data).to_excel(writer, sheet_name='测试信息', index=False)
    
    print(f"详细结果已保存至: {excel_file}")


def main():
    """主函数，处理命令行参数并执行测试"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置API端口
    port = os.environ.get("API_PORT", 8000)
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="0",
        base_url=f"http://localhost:{port}/v1",
    )
    
    # 获取测试数据路径和模型名称
    test_data_path = os.environ.get("TEST_DATA_PATH", args.input_path)
    
    # 确定数据集类型，优先使用命令行参数或环境变量
    dataset_type = os.environ.get("DATASET_TYPE", args.dataset_type)
    print(f"使用数据集类型: {dataset_type}")
    
    # 设置最大token数量，如果没有指定则根据数据集类型自动设置
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 13500 if dataset_type == 'sleep' else 400
    print(f"API最大token数量: {max_tokens}")
    
    # 加载测试数据
    max_samples = args.max_samples # Use max_samples from args
    test_data = load_test_data(test_data_path, max_samples=max_samples) # Pass max_samples here
    
    if not test_data: # Exit if data loading failed
        print("无法加载测试数据，程序退出。")
        return

    # 从测试数据路径中提取测试集名称，并去掉数据量信息
    file_basename = os.path.basename(test_data_path).split('.')[0]
    # 改进正则以移除更多可能的后缀，如日期、版本等
    # This regex removes _n<digits>, _<date><time>, _tok<digits>, etc. Adjust as needed.
    test_set_name = re.sub(r'(_n\d+|_?\d{8,}(?:_\d{4,})?|_tok\d+|_bal[\d.]+|_sqrt_inverse)', '', file_basename)

    # 尝试从环境变量或命令行参数获取模型名称
    model_name = os.environ.get("MODEL_NAME", args.model_name)
    
    print(f"模型: {model_name}")
    print(f"测试集: {test_set_name}")
    print(f"加载并准备测试 {len(test_data)} 个样本") # Changed message slightly
    
    # 创建保存结果的目录
    save_dir = os.environ.get("SAVE_DIR", args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置打印和保存的间隔
    print_interval = int(os.environ.get("PRINT_INTERVAL", args.print_interval))
    # 根据数据集长度动态设置保存步长，默认为总长度的1/10，最小为1, 不超过500
    default_save_interval = max(1, min(len(test_data) // 10, 500)) # Added upper limit
    # Use provided save_interval if it exists and is valid, otherwise use default
    save_interval_arg = args.save_interval
    save_interval = int(os.environ.get("SAVE_INTERVAL", save_interval_arg if save_interval_arg and save_interval_arg > 0 else default_save_interval))
    print(f"打印间隔: {print_interval}, 保存间隔: {save_interval}") # Print intervals used

    
    # 评估模型
    metrics = evaluate_model(
        client, 
        test_data, 
        print_interval=print_interval, 
        save_interval=save_interval, 
        save_dir=save_dir, 
        model_name=model_name, 
        test_set_name=test_set_name,
        dataset_type=dataset_type,
        max_tokens=max_tokens
    )
    
    print("测试完成!")

if __name__ == "__main__":
    main()