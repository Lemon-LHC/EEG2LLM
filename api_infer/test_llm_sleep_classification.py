import os
import json
import glob
import argparse
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import threading
import requests

# 添加OpenAI API支持
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: 未找到OpenAI库，若要使用API模式，请先安装: pip install openai")

# 添加当前目录到路径，以便导入Qwen模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen_vl_utils import process_vision_info

# 在文件顶部添加API处理类
class DeepSeekAPI:
    """DeepSeek API处理器"""
    
    def __init__(self, api_key: str = None):
        self.base_url = "https://api.siliconflow.cn/v1"
        # 从环境变量获取密钥，如果未提供
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未提供DeepSeek API密钥，请设置DEEPSEEK_API_KEY环境变量")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, messages, **kwargs):
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": 1,
            "response_format": {"type": ""}
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_with_retry(self, messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self.generate(messages)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"请求失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
        return ""

class OpenAIAPI:
    """OpenAI API处理器"""
    
    def __init__(self, api_key: str):
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, messages, **kwargs):
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 0.7)
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def load_test_data(data_dir, max_samples=None):
    """加载测试数据（添加提前终止功能）"""
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    test_data = []
    
    # 如果设置了最大样本数，提前过滤文件列表
    if max_samples and max_samples < len(json_files):
        json_files = np.random.choice(json_files, max_samples, replace=False)
    
    for file_path in tqdm(json_files, desc="加载测试数据"):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                test_data.append(data)
                # 提前终止条件
                if max_samples and len(test_data) >= max_samples:
                    break
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {str(e)}")
    
    return test_data

def initialize_model(model_path, model_type="qwen"):
    """初始化大模型，支持Qwen和Llama模型
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，可选 "qwen" 或 "llama"
    """
    print(f"正在加载模型: {model_path}，类型: {model_type}")
    
    # 显示加载进度的装饰器
    def progress_indicator(func):
        def wrapper(*args, **kwargs):
            print(f"开始加载模型组件...")
            t_start = time.time()
            try:
                # 创建进度条
                pbar = tqdm(total=100, desc="模型加载进度")
                
                # 定义进度更新函数
                def update_progress(progress):
                    pbar.update(progress - pbar.n)
                
                # 每5秒更新一次进度，模拟加载过程
                update_thread = threading.Thread(target=lambda: [
                    update_progress(min(i, 95)) for i in range(0, 100, 5) 
                    if pbar.n < 95 and (time.sleep(0.5) or True)
                ])
                update_thread.daemon = True
                update_thread.start()
                
                # 执行原函数
                result = func(*args, **kwargs)
                
                # 完成进度条
                pbar.update(100 - pbar.n)
                pbar.close()
                
                t_end = time.time()
                load_time = t_end - t_start
                print(f"模型加载完成，耗时: {load_time:.2f}秒")
                return result
            except Exception as e:
                if 'pbar' in locals():
                    pbar.close()
                raise e
        return wrapper
    
    # 使用进度条装饰器
    @progress_indicator
    def load_model():
        if model_type.lower() == "llama":
            try:
                # 使用transformers加载Llama模型
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print("正在加载Llama模型...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Llama通常使用bfloat16
                    device_map="auto",
                    trust_remote_code=True
                ).eval()
                
                print("成功加载Llama模型")
                return model, tokenizer, model_type
            except Exception as e:
                print(f"加载Llama模型失败: {e}")
                raise ValueError(f"无法加载Llama模型，请检查模型路径和配置: {e}")
        else:
            # 默认加载Qwen模型
            try:
                # 使用transformers中的AutoProcessor和AutoModelForVision2Seq加载模型
                from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
                
                print("正在使用transformers加载Qwen2.5-VL模型...")
                
                # 加载处理器
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                
                # 加载模型
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto"
                )
                
                print("成功加载Qwen2.5-VL模型")
                return model, processor, model_type
            except Exception as e:
                print(f"使用transformers的AutoModelForVision2Seq加载失败: {e}")
                
                try:
                    # 尝试使用AutoModelForCausalLM加载
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    print("尝试使用AutoModelForCausalLM加载...")
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    ).eval()
                    print("成功使用AutoModelForCausalLM加载模型")
                    return model, tokenizer, "causal"
                except Exception as e2:
                    print(f"使用AutoModelForCausalLM加载失败: {e2}")
                    
                    # 尝试使用modelscope加载
                    try:
                        from modelscope import AutoModelForCausalLM as MSAutoModelForCausalLM
                        from modelscope import AutoTokenizer as MSAutoTokenizer
                        
                        print("正在使用modelscope加载模型...")
                        tokenizer = MSAutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                        model = MSAutoModelForCausalLM.from_pretrained(
                            model_path,
                            device_map="auto",
                            trust_remote_code=True
                        ).eval()
                        print("成功使用modelscope加载模型")
                        return model, tokenizer, "modelscope"
                    except Exception as e3:
                        print(f"使用modelscope加载也失败: {e3}")
                        
                        # 最后的尝试，打印模型目录下的文件
                        try:
                            files = os.listdir(model_path)
                            print(f"模型目录下的文件: {files}")
                        except Exception:
                            pass
                        
                        raise ValueError(f"无法加载模型，请检查模型路径和配置: {e}")
    
    return load_model()

def get_api_prediction(prompt, api_settings):
    """通过API获取模型预测（支持DeepSeek和OpenAI）"""
    try:
        # 根据模型类型选择API处理器
        if "deepseek" in api_settings['model'].lower():
            client = DeepSeekAPI(api_key=api_settings['api_key'])
        else:  # 默认为OpenAI格式
            client = OpenAIAPI(api_key=api_settings['api_key'])
        
        # 构建系统提示
        system_prompt = (
            "你是一个医学助手，擅长睡眠分期。请根据描述判断这是哪种睡眠阶段（0-5）："
            "0表示清醒(W)，1表示非快眼动1期(N1)，2表示非快眼动2期(N2)，"
            "3表示非快眼动3期(N3)，4表示非快眼动4期(N4)，5表示快速眼动(REM)。"
            "只需回答对应的数字。"
        )
        
        # 调用API
        start_time = time.time()
        response = client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        end_time = time.time()
        
        # 提取数字
        prediction = None
        for char in response:
            if char.isdigit() and int(char) in [0, 1, 2, 3, 4, 5]:
                prediction = int(char)
                break
        
        return response, prediction, end_time - start_time
    except Exception as e:
        print(f"API调用出错: {e}")
        return "API调用失败", None, 0

def get_llm_prediction(model, tokenizer_or_processor, prompt, model_type="qwen", use_api=False, api_settings=None):
    """获取大模型对提示的回答，支持Qwen和Llama模型或API调用
    
    Args:
        model: 模型对象
        tokenizer_or_processor: tokenizer或processor对象
        prompt: 提示文本
        model_type: 模型类型，可选 "qwen"、"llama"、"causal"、"modelscope"
        use_api: 是否使用API调用
        api_settings: API设置（使用API模式时）
        
    Returns:
        tuple: (回答文本, 预测结果, 推理耗时)
    """
    # API模式
    if use_api:
        return get_api_prediction(prompt, api_settings)
    
    # 本地模型
    response = ""
    start_time = time.time()
    
    if model_type.lower() == "llama":
        try:
            # 构建Llama模型的消息格式
            messages = [
                {"role": "system", "content": "你是一个医学助手，擅长睡眠分期。请根据描述判断这是哪种睡眠阶段（0-5）：0表示清醒(W)，1表示非快眼动1期(N1)，2表示非快眼动2期(N2)，3表示非快眼动3期(N3)，4表示非快眼动4期(N4)，5表示快速眼动(REM)。只需回答对应的数字。"},
                {"role": "user", "content": prompt}
            ]
            
            # 使用chat_template格式化输入
            input_text = tokenizer_or_processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 模型输入处理
            inputs = tokenizer_or_processor(input_text, return_tensors="pt").to(model.device)
            
            # 生成输出
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False, 
                temperature=0.1
            )
            
            # 解码输出
            response = tokenizer_or_processor.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
        except Exception as e:
            print(f"Llama生成出错: {e}")
            response = "无法生成回答"
    
    elif model_type.lower() == "qwen":
        try:
            # 构建消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 准备推理
            text = tokenizer_or_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = tokenizer_or_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 生成输出
            generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.1)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = tokenizer_or_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
                
        except Exception as e:
            print(f"Qwen框架生成出错: {e}")
            response = "无法生成回答"
    
    else:  # causal或modelscope类型
        try:
            # 添加系统提示以改善回答质量
            system_prompt = "You are a neurobiological expert specializing in EEG data analysis and sleep stage classification. Your task is to analyze the provided EEG data (including voltage values from the Fpz-Cz and Pz-Oz channels) and determine the current sleep stage of the volunteer based on the following classification criteria:\n0: Wakefulness (W)\n1: Non-rapid eye movement sleep stage 1 (N1)\n2: Non-rapid eye movement sleep stage 2 (N2)\n3: Non-rapid eye movement sleep stage 3 (N3)\n4: Non-rapid eye movement sleep stage 4 (N4)\n5: Rapid eye movement sleep stage (R)\nThe EEG data is provided in the format (time in milliseconds, Fpz-Cz voltage in μV, Pz-Oz voltage in μV). The data spans 1000ms with a sampling interval of 5ms. In your analysis, pay attention to the following characteristics of each sleep stage:\n- Wakefulness (W): High-frequency, low-amplitude waves.\n- N1: Low-amplitude, mixed-frequency waves.\n- N2: Sleep spindles and K-complexes.\n- N3: High-amplitude, low-frequency delta waves.\n- N4: Dominant delta waves.\n- REM (R): Rapid eye movements and low muscle tone.\nYour response must be a single number (0, 1, 2, 3, 4, or 5) corresponding to the sleep stage. Do not include any additional text, punctuation, or explanations. "
            full_prompt = system_prompt + prompt
            
            # 尝试标准的tokenizer方法
            inputs = tokenizer_or_processor(full_prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,
                temperature=0.1, 
                do_sample=False  
            )
            
            if hasattr(tokenizer_or_processor, 'batch_decode'):
                response = tokenizer_or_processor.batch_decode(
                    [outputs[0][inputs.input_ids.shape[1]:]], 
                    skip_special_tokens=True
                )[0]
            else:
                response = tokenizer_or_processor.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                
            # 如果回复包含原始提示，去除它
            if prompt in response:
                response = response[len(prompt):].strip()
        except Exception as e2:
            print(f"标准生成方法也失败: {e2}")
            try:
                # 尝试chat方法（如果存在）
                if hasattr(model, 'chat'):
                    messages = [{"role": "user", "content": prompt}]
                    response, _ = model.chat(tokenizer_or_processor, messages)
                else:
                    response = "无法生成回答"
            except Exception as e3:
                print(f"chat方法也失败: {e3}")
                response = "无法生成回答"
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 尝试从回答中提取数字
    prediction = None
    for char in response:
        if char.isdigit() and int(char) in [0, 1, 2, 3, 4, 5]:
            prediction = int(char)
            break
    
    return response, prediction, inference_time

def evaluate_model(model, tokenizer_or_processor, test_data, output_dir, model_type="qwen", use_api=True, api_settings=None):
    """评估模型性能"""
    print("开始评估模型性能...")
    results = []
    true_labels = []
    pred_labels = []
    responses = []
    
    # 创建存储详细结果的文件
    os.makedirs(output_dir, exist_ok=True)
    detail_file = os.path.join(output_dir, "detailed_results.txt")
    
    # 在评估函数中添加API性能跟踪
    api_metrics = {
        "total_requests": 0,
        "failed_requests": 0,
        "average_latency": 0
    }
    
    # 初始化指标时添加默认值
    metrics = {
        'total_samples': 0,
        'accuracy': 0.0,
        'f1_macro': 0.0,
        'avg_inference_time': 0.0,
        'confusion_matrix': np.zeros((6, 6), dtype=int)
    }
    
    with open(detail_file, 'w', encoding='utf-8') as detail_f:
        # 使用tqdm创建进度条，包含更多信息
        pbar = tqdm(test_data, desc="测试样本", ncols=100)
        
        # 用于实时显示性能指标
        correct_count = 0
        total_count = 0
        total_time = 0.0
        
        for idx, data in enumerate(pbar):
            human_prompt = data["conversations"][0]["value"]
            true_label = int(data["conversations"][1]["value"])
            
            try:
                # 获取模型预测：本地模型或API调用
                response, prediction, inference_time = get_llm_prediction(
                    model, tokenizer_or_processor, human_prompt, model_type, use_api, api_settings
                )
                
                total_time += inference_time
                
                if prediction is not None:
                    correct = prediction == true_label
                    if correct:
                        correct_count += 1
                else:
                    correct = False
                    prediction = -1  # 表示无法从回答中提取预测值
                
                total_count += 1
                
                # 更新进度条状态
                current_acc = correct_count / total_count if total_count > 0 else 0
                avg_time = total_time / total_count if total_count > 0 else 0
                
                # 设置进度条描述，实时显示准确率和平均推理时间
                pbar.set_postfix({
                    'acc': f'{current_acc:.4f}',
                    'avg_time': f'{avg_time:.2f}s',
                    '当前预测': f'{prediction}',
                    '真实标签': f'{true_label}',
                    '正确': correct
                })
                
                result = {
                    "id": idx,
                    "true_label": true_label,
                    "prediction": prediction,
                    "response": response,
                    "correct": correct,
                    "inference_time": inference_time
                }
                
                results.append(result)
                true_labels.append(true_label)
                pred_labels.append(prediction if prediction != -1 else 0)  # 对于无法提取预测的情况，默认为0
                responses.append(response)
                
                # 写入详细结果
                detail_f.write(f"样本ID: {idx}\n")
                detail_f.write(f"真实标签: {true_label}\n")
                detail_f.write(f"预测标签: {prediction}\n")
                detail_f.write(f"模型回答: {response}\n")
                detail_f.write(f"正确: {correct}\n")
                detail_f.write(f"推理时间: {inference_time:.4f}秒\n")
                detail_f.write("-" * 80 + "\n")
                
                # 在每次API调用后更新指标
                api_metrics["total_requests"] += 1
                if prediction is None:
                    api_metrics["failed_requests"] += 1
                api_metrics["average_latency"] = (
                    (api_metrics["average_latency"] * (api_metrics["total_requests"] - 1) + inference_time) 
                    / api_metrics["total_requests"]
                )
                
                # 更新指标
                metrics['total_samples'] += 1  # 确保总样本数统计
                if correct:
                    metrics['accuracy'] += 1
                if prediction is not None:
                    metrics['confusion_matrix'][true_label, prediction] += 1
                
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {str(e)}")
                continue  # 跳过失败样本
        
        # 计算前检查有效样本
        valid_samples = len([i for i, pred in enumerate(pred_labels) if pred != -1])
        if valid_samples > 0:
            metrics['accuracy'] = metrics['accuracy'] / valid_samples
            metrics['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            metrics['avg_inference_time'] = total_time / valid_samples
            
            # 计算混淆矩阵
            cm = metrics['confusion_matrix']
            
            # 保存混淆矩阵图 - 使用英文标签避免中文字体问题
            try:
                # 捕获pyplot的警告并忽略
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['W', 'N1', 'N2', 'N3', 'N4', 'REM'],
                                yticklabels=['W', 'N1', 'N2', 'N3', 'N4', 'REM'])
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title('Sleep Stage Classification Confusion Matrix')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
                    plt.close()  # 关闭图形，释放内存
                    print("成功保存混淆矩阵图")
            except Exception as e:
                print(f"保存混淆矩阵图失败: {e}")
            
            # 统计各类样本数量
            class_counts = {}
            for label in true_labels:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
            
            # 保存评估结果
            summary_file = os.path.join(output_dir, "performance_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\n")
                f.write(f"睡眠分期大模型({model_type})性能评估报告\n")
                f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                f.write("数据集统计:\n")
                f.write(f"总样本数: {len(test_data)}\n")
                f.write(f"有效预测样本数: {valid_samples}\n")
                f.write(f"无法识别预测的样本数: {len(test_data) - valid_samples}\n\n")
                
                f.write("各睡眠阶段样本分布:\n")
                for i in range(6):
                    count = class_counts.get(i, 0)
                    percentage = (count / len(test_data)) * 100
                    label_name = ["清醒(W)", "非快眼动1期(N1)", "非快眼动2期(N2)", "非快眼动3期(N3)", "非快眼动4期(N4)", "快速眼动(REM)"][i]
                    f.write(f"阶段 {i} ({label_name}): {count} 样本 ({percentage:.2f}%)\n")
                f.write("\n")
                
                f.write("总体性能指标:\n")
                f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
                f.write(f"宏平均F1分数 (Macro F1): {metrics['f1_macro']:.4f}\n\n")
                
                f.write("平均推理时间: {:.4f}秒/样本\n".format(metrics['avg_inference_time']))
            
            print(f"评估完成，结果已保存到: {output_dir}")
            
            # 返回主要指标
            return {
                "total_samples": metrics['total_samples'],
                "accuracy": metrics['accuracy'],
                "f1_macro": metrics['f1_macro'],
                "avg_inference_time": metrics['avg_inference_time'],
                "confusion_matrix": metrics['confusion_matrix'],
                "api_metrics": api_metrics
            }
        else:
            print("警告：所有样本处理失败，无法计算准确率")
            return None

def main():
    parser = argparse.ArgumentParser(description="评估大模型在睡眠分类任务上的性能")
    parser.add_argument("--data_dir", type=str, default="/data/lhc/datasets/sleep-edfx/test_200hz/processed_1/processed_test/", help="测试数据目录路径")
    parser.add_argument("--model_path", type=str, default="/data/lhc/models/Llama-3.1-8B-Instruct", help="模型路径")
    parser.add_argument("--model_type", type=str, default="llama", choices=["qwen", "llama"], help="模型类型，可选qwen或llama")
    parser.add_argument("--output_dir", type=str, default="/data/lhc/results", help="结果输出目录")
    parser.add_argument("--max_samples", type=int, default=100, help="最大测试样本数，默认为100")
    
    # 修改API相关参数，移除默认值
    parser.add_argument("--use_api", action="store_true", help="使用API调用而非本地模型")
    parser.add_argument("--api_base_url", type=str, help="API基础URL")
    parser.add_argument("--api_key", type=str, help="API密钥")
    parser.add_argument("--api_model", type=str, default="gpt-4o", help="API模型名称")
    
    args = parser.parse_args()
    
    # API设置
    use_api = args.use_api
    api_settings = {
        "provider": "deepseek" if "deepseek" in args.api_model.lower() else "openai",
        "api_key": args.api_key,
        "model": args.api_model,
        "base_url": args.api_base_url
    }
    
    # 检查API设置
    if use_api:
        if not OPENAI_AVAILABLE:
            print("错误: 使用API模式需要安装OpenAI库: pip install openai")
            return
        if not api_settings["api_key"]:
            print("错误: 使用API模式需要提供API密钥")
            return
        print(f"使用API模式，模型: {api_settings['model']}")
        # API模式下使用API模型名作为模型类型
        model_type = f"api_{api_settings['model']}"
    else:
        # 根据模型类型设置正确的模型路径
        if args.model_type.lower() == "qwen":
            args.model_path = "/data/lhc/models/Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            args.model_path = "/data/lhc/models/Llama-3.1-8B-Instruct"
        model_type = args.model_type
    
    # 加载测试数据
    test_data = load_test_data(args.data_dir, args.max_samples)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 如果设置了最大样本数，则只使用部分样本
    if args.max_samples and args.max_samples < len(test_data):
        np.random.seed(42)  # 固定随机种子以确保可重复性
        indices = np.random.choice(len(test_data), args.max_samples, replace=False)
        test_data = [test_data[i] for i in indices]
        print(f"随机选择了 {args.max_samples} 个样本进行测试")
    
    # 初始化模型（仅在非API模式下）
    model = None
    tokenizer_or_processor = None
    
    if not use_api:
        model, tokenizer_or_processor, _ = initialize_model(args.model_path, args.model_type)
    
    # 创建输出目录，处理权限问题
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{model_type}_{timestamp}")
    
    try:
        # 首先确保父目录存在
        parent_dir = os.path.dirname(output_dir)
        if not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
                print(f"已创建父目录: {parent_dir}")
            except PermissionError:
                # 如果没有权限，则切换到临时目录
                parent_dir = "/tmp/llm_sleep_results"
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                output_dir = os.path.join(parent_dir, f"eval_{model_type}_{timestamp}")
                print(f"没有权限访问原目录，切换到: {output_dir}")
        
        # 创建实际的输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"已创建输出目录: {output_dir}")
    except Exception as e:
        # 如果仍然出错，使用绝对的临时目录路径
        print(f"创建目录出错: {e}")
        output_dir = f"/tmp/llm_sleep_results/eval_{model_type}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用临时目录: {output_dir}")
    
    # 评估模型
    metrics = evaluate_model(model, tokenizer_or_processor, test_data, output_dir, model_type, use_api, api_settings)
    
    if metrics:
        print("\n主要性能指标:")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"宏平均F1分数 (Macro F1): {metrics['f1_macro']:.4f}")
        print(f"\n详细结果已保存至: {output_dir}")
        
        # 打印混淆矩阵
        print("\n混淆矩阵:")
        cm = metrics["confusion_matrix"]
        for i, row in enumerate(cm):
            stage_name = ["W", "N1", "N2", "N3", "N4", "REM"][i]
            print(f"{stage_name}: {row}")

# 2. 添加一个新函数，允许直接通过代码调用
def run_deepseek_test(
    data_dir="/data/lhc/datasets/sleep-edfx/test_200hz/processed_1/processed_test/",
    output_dir="/data/lhc/results",
    max_samples=50,  # 默认改为50
    api_key=None,
    model_name="deepseek-ai/DeepSeek-R1"
):
    # 添加环境检查
    try:
        import torch
    except ImportError:
        raise RuntimeError("需要安装PyTorch，请执行: pip install torch")
    
    # 修改数据加载调用
    test_data = load_test_data(data_dir, max_samples=max_samples)  # 提前过滤
    
    # 移除后续的随机采样
    print(f"已加载 {len(test_data)} 个样本（最大限制 {max_samples}）")
    
    # 设置API参数
    api_settings = {
        "provider": "deepseek",
        "api_key": api_key,  # 如果为None，DeepSeekAPI会从环境变量获取
        "model": model_name,
        "base_url": "https://api.siliconflow.cn/v1"
    }
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = f"api_{model_name}"
    output_dir = os.path.join(output_dir, f"eval_{model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估模型
    metrics = evaluate_model(
        model=None, 
        tokenizer_or_processor=None, 
        test_data=test_data, 
        output_dir=output_dir, 
        model_type=model_type, 
        use_api=True, 
        api_settings=api_settings
    )
    
    return metrics

if __name__ == "__main__":
    # 命令行方式调用
    main()
    
    # 直接代码调用示例（取消注释使用）
    """
    # 直接通过代码调用DeepSeek测试
    metrics = run_deepseek_test(
        data_dir="/data/lhc/datasets/sleep-edfx/test_200hz/processed_1/processed_test/",
        output_dir="/data/lhc/results",
        max_samples=50,
        # api_key="your_key_here",  # 可选，默认从环境变量获取
        model_name="deepseek-ai/DeepSeek-R1"
    )
    
    if metrics:
        print("\n主要性能指标:")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"宏平均F1分数 (Macro F1): {metrics['f1_macro']:.4f}")
    """ 