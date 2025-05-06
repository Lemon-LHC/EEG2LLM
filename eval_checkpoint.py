import os
import json
import time
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import gc
import traceback
from accelerate import init_empty_weights
from accelerate.utils import get_max_memory
from peft import PeftModel, PeftConfig

# 定义睡眠阶段标签，避免重复硬编码
SLEEP_STAGE_LABELS = [
    'Wake (W)', 
    'NREM Stage 1 (N1)', 
    'NREM Stage 2 (N2)', 
    'NREM Stage 3 (N3)', 
    'NREM Stage 4 (N4)', 
    'REM Sleep (R)'
]

def load_test_data(file_path, max_samples=None):
    """加载测试数据"""
    print(f"开始加载测试数据: {file_path}")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    load_time = time.time() - start_time
    print(f"加载完成，原始样本数: {len(test_data)}，耗时: {load_time:.2f}秒")
    
    # 如果设置了最大样本数，随机选择样本
    if max_samples and max_samples < len(test_data):
        print(f"限制最大样本数为: {max_samples}")
        indices = np.random.choice(len(test_data), max_samples, replace=False)
        test_data = [test_data[i] for i in indices]
    
    return test_data

def get_model_prediction(model, tokenizer, prompt, system_prompt=None, max_retries=3, retry_delay=2, device="cuda"):
    """使用本地模型获取预测，包含重试机制"""
    for attempt in range(max_retries):
        try:
            # 构建完整提示
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt_text = prompt
            
            # 编码输入
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            
            # 记录推理时间
            start_time = time.time()
            
            # 生成输出
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # 对于分类任务，只需要短输出
                    do_sample=False
                )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 解码输出
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # 提取数字
            prediction = None
            for char in response:
                if char.isdigit() and int(char) in [0, 1, 2, 3, 4, 5]:
                    prediction = int(char)
                    break
            
            return response, prediction, inference_time
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"预测出错 (尝试 {attempt+1}/{max_retries}): {e}，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"预测失败，已达到最大重试次数 ({max_retries}): {e}")
                return "预测失败", None, 0

def calculate_metrics(true_labels, pred_labels, writer=None, step=0):
    """计算评估指标并记录到TensorBoard
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        writer: TensorBoard的SummaryWriter对象
        step: 当前步数
    """
    # 计算总体指标
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=range(6))
    
    # 记录总体指标到TensorBoard
    if writer is not None:
        writer.add_scalar('test/accuracy', accuracy, step)
        writer.add_scalar('test/f1', f1_macro, step)
        writer.add_scalar('test/precision', precision_macro, step)
        writer.add_scalar('test/recall', recall_macro, step)
        writer.add_scalar('test/loss', 0.0, step)  # 添加loss占位符
    
    # 计算每个类别的指标
    class_metrics = {}
    for class_idx in range(6):
        # 计算该类别的准确率
        class_true = [1 if label == class_idx else 0 for label in true_labels]
        class_pred = [1 if label == class_idx else 0 for label in pred_labels]
        class_accuracy = accuracy_score(class_true, class_pred)
        class_f1 = f1_score(class_true, class_pred, zero_division=0)
        class_precision = precision_score(class_true, class_pred, zero_division=0)
        class_recall = recall_score(class_true, class_pred, zero_division=0)
        
        # 获取类别名称（简短版本）
        stage_name = SLEEP_STAGE_LABELS[class_idx].split(' ')[0]  # 只取简短名称，如'W', 'N1'等
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar(f'test/Class_{stage_name}/Accuracy', class_accuracy, step)
            writer.add_scalar(f'test/Class_{stage_name}/F1', class_f1, step)
            writer.add_scalar(f'test/Class_{stage_name}/Precision', class_precision, step)
            writer.add_scalar(f'test/Class_{stage_name}/Recall', class_recall, step)
        
        class_metrics[f'Class_{stage_name}'] = {
            'accuracy': class_accuracy,
            'f1': class_f1,
            'precision': class_precision,
            'recall': class_recall
        }
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'class_metrics': class_metrics,
        'confusion_matrix': cm
    }
    
    return metrics

def save_results(results, metrics, checkpoint_name, save_dir, test_set_name="unknown_dataset"):
    """保存测试结果到指定目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(save_dir, f"{checkpoint_name}_{test_set_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存模型名称和测试集信息
    with open(os.path.join(result_dir, "info.json"), "w") as f:
        json.dump({
            "checkpoint_name": checkpoint_name,
            "test_set_name": test_set_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)
    
    # 保存指标
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                 for k, v in metrics.items()}, f, indent=4)
    
    # 保存结果
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # 保存混淆矩阵图像
    plt.figure(figsize=(10, 8))
    cm = metrics['confusion_matrix']
    # 使用简化的标签
    short_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_labels,
                yticklabels=short_labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 - {checkpoint_name}')
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()
    
    # 打印结果路径
    print(f"\n结果已保存至: {result_dir}")

def evaluate_checkpoint(checkpoint_path, test_data_path, save_dir, max_samples=None, device="cuda", template="alpaca", tensorboard_dir=None, verbose=False, precision="bf16", use_cpu_offload=True):
    """评估特定检查点的模型性能
    
    Args:
        checkpoint_path (str): 检查点路径
        test_data_path (str): 测试数据文件路径
        save_dir (str): 保存结果的目录
        max_samples (int, optional): 最大评估样本数. Defaults to None.
        device (str, optional): 运行设备. Defaults to "cuda".
        template (str, optional): 对话模板. Defaults to "alpaca".
        tensorboard_dir (str, optional): TensorBoard日志目录. Defaults to None.
        verbose (bool, optional): 是否输出详细信息. Defaults to False.
        precision (str, optional): 模型精度. Defaults to "bf16".
        use_cpu_offload (bool, optional): 是否启用CPU卸载. Defaults to True.
    
    Returns:
        dict: 评估结果
    """
    # 提取检查点名称和步数
    checkpoint_name = checkpoint_path.split('/')[-1]
    checkpoint_step = int(checkpoint_name.split('-')[-1]) if checkpoint_name.startswith('checkpoint-') else 0
    
    # 初始化TensorBoard记录器（如果指定了目录）
    tb_writer = None
    if tensorboard_dir:
        try:
            # 使用传入的tensorboard_dir作为根目录
            print(f"准备使用TensorBoard日志目录: {tensorboard_dir}")
            
            # 从测试数据路径中提取测试集名称，用于创建子目录
            test_set_name = os.path.basename(test_data_path).split('.')[0]
            
            # 创建特定于此次评估的子目录
            eval_tb_dir = os.path.join(tensorboard_dir, f"test_{test_set_name}_{checkpoint_name}")
            os.makedirs(eval_tb_dir, exist_ok=True)
            print(f"创建测试集TensorBoard子目录: {eval_tb_dir}")
            
            # 确保目录权限正确
            os.system(f"chmod -R 755 {eval_tb_dir}")
            
            # 初始化TensorBoard写入器 - 注意使用子目录
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=eval_tb_dir)
            print(f"成功创建TensorBoard写入器，指向目录: {eval_tb_dir}")
            
            # 记录基本信息
            import socket
            host_info = f"{socket.gethostname()}_{os.getpid()}"
            tb_writer.add_text('eval_info/checkpoint', checkpoint_path, 0)
            tb_writer.add_text('eval_info/test_data', test_data_path, 0)
            tb_writer.add_text('eval_info/host', host_info, 0)
            tb_writer.add_text('eval_info/time', str(datetime.datetime.now()), 0)
            
            # 确保这些元数据被写入
            tb_writer.flush()
            
            # 列出目录内容，确认写入器创建了事件文件
            event_files = glob.glob(os.path.join(eval_tb_dir, "events.out.tfevents*"))
            if event_files:
                print(f"在TensorBoard目录找到事件文件: {[os.path.basename(f) for f in event_files]}")
            else:
                print(f"警告: 在TensorBoard目录未找到事件文件，可能存在权限或路径问题")
                
        except Exception as e:
            print(f"初始化TensorBoard记录器时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            print("将继续评估，但不记录TensorBoard数据")
    
    # 从测试数据路径中提取测试集名称
    test_set_name = os.path.basename(test_data_path).split('.')[0]
    
    print(f"加载检查点: {checkpoint_name}")
    print(f"测试集: {test_set_name}")
    
    # 加载模型和分词器
    try:
        print(f"开始加载检查点: {checkpoint_path}...")
        model_load_start = time.time()
        
        # 首先检查是否为LoRA检查点
        is_lora = False
        if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")) or \
           os.path.isfile(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
           os.path.isfile(os.path.join(checkpoint_path, "adapter_model.bin")):
            is_lora = True
            print("检测到LoRA检查点")
        
        # 检查是否为safetensors格式的分片模型
        is_sharded_safetensors = False
        safetensors_files = [f for f in os.listdir(checkpoint_path) if f.startswith("model-") and f.endswith(".safetensors")]
        if safetensors_files:
            is_sharded_safetensors = True
            print(f"检测到分片safetensors模型文件: {len(safetensors_files)}个分片")
        
        # 检查是否需要加载基础模型
        base_model_path = None
        if is_lora:
            try:
                # 尝试从adapter_config.json读取基础模型路径
                adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                print(f"尝试读取adapter配置: {adapter_config_path}")
                if os.path.isfile(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        if "base_model_name_or_path" in adapter_config:
                            base_model_path = adapter_config["base_model_name_or_path"]
                            print(f"从adapter_config读取到基础模型路径: {base_model_path}")
                            
                            # 检查基础模型路径是否存在
                            if not os.path.exists(base_model_path):
                                print(f"警告: 配置文件指定的基础模型路径 {base_model_path} 不存在")
                                base_model_path = None
            except Exception as e:
                print(f"读取adapter_config.json时出错: {e}")
                print(f"错误详情: {traceback.format_exc()}")
            
            # 如果无法从config获取，尝试从checkpoint_path推断
            if not base_model_path:
                print("无法从adapter_config.json获取有效的基础模型路径，尝试推断...")
                # 提取检查点所在目录的父目录路径
                checkpoint_dir = os.path.dirname(checkpoint_path)
                parent_dir = os.path.dirname(checkpoint_dir)
                grandparent_dir = os.path.dirname(parent_dir)
                
                print(f"检查点目录: {checkpoint_dir}")
                print(f"父目录: {parent_dir}")
                print(f"祖父目录: {grandparent_dir}")
                
                # 如果父目录或祖父目录包含model.safetensors或pytorch_model.bin，可能是基础模型
                for dir_to_check in [parent_dir, grandparent_dir]:
                    if os.path.isfile(os.path.join(dir_to_check, "model.safetensors")) or \
                       os.path.isfile(os.path.join(dir_to_check, "pytorch_model.bin")) or \
                       os.path.isfile(os.path.join(dir_to_check, "config.json")):
                        base_model_path = dir_to_check
                        print(f"从目录结构推断基础模型路径: {base_model_path}")
                        break
            
            # 如果还是无法确定基础模型，尝试使用环境变量或默认值
            if not base_model_path:
                # 不再使用/data/lhc/models_new/路径
                base_model_path = os.environ.get("BASE_MODEL_PATH", "/data/lhc/models/Llama-3.2-1B-Instruct")
                print(f"无法确定基础模型路径，使用默认值: {base_model_path}")
                
                # 验证默认路径是否存在
                if not os.path.exists(base_model_path):
                    print(f"警告: 默认基础模型路径 {base_model_path} 不存在")
                    # 尝试备用路径，不使用/data/lhc/models_new/
                    alternate_paths = [
                        "/data/lhc/models/Llama-3.2-1B-Instruct",
                        "/data/lhc/models/Llama-3.1-8B-Instruct",
                        "/data/lhc/models/Llama-2-7B-Chat-GPTQ"
                    ]
                    for path in alternate_paths:
                        if os.path.exists(path):
                            base_model_path = path
                            print(f"使用备用基础模型路径: {base_model_path}")
                            break
                            
        # 最后安全检查，确保不会使用/data/lhc/models_new/目录
        if base_model_path and ('/data/lhc/models_new/' in base_model_path or 'edf200_100hz_15000ms' in base_model_path):
            print(f"警告: 检测到base_model_path指向了models_new目录或含有edf200_100hz_15000ms: {base_model_path}")
            
            # 尝试从checkpoint_path中提取适配器配置信息
            adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
            original_model_path = None
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        if "base_model_name_or_path" in adapter_config:
                            original_model_path = adapter_config["base_model_name_or_path"]
                except Exception:
                    pass
            
            print(f"为避免创建不必要的目录，将使用安全的替代路径")
            
            # 首先尝试使用命令行参数中指定的基础模型路径
            if hasattr(args, 'model_name_or_path') and args.model_name_or_path:
                base_model_path = args.model_name_or_path
                print(f"使用命令行参数指定的模型路径: {base_model_path}")
            # 然后尝试使用预定义的模型路径
            else:
                base_model_path = "/data/lhc/models/Llama-3.2-1B-Instruct"
                print(f"使用默认安全模型路径: {base_model_path}")
            
            # 检查所选路径是否存在
            if not os.path.exists(base_model_path):
                # 尝试其他备用路径
                for path in ["/data/lhc/models/Llama-3.2-1B-Instruct", "/data/lhc/models/Llama-3.1-8B-Instruct", "/data/lhc/models/Llama-2-7B-Chat-GPTQ"]:
                    if os.path.exists(path):
                        base_model_path = path
                        print(f"使用备用安全模型路径: {base_model_path}")
                        break
            
        # 加载分词器
        print("开始加载分词器...")
        tokenizer_start = time.time()
        if is_lora and base_model_path:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            print(f"从基础模型加载分词器完成，耗时: {time.time() - tokenizer_start:.2f}秒")
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            print(f"从检查点加载分词器完成，耗时: {time.time() - tokenizer_start:.2f}秒")

        # 设置设备
        if device == "cuda":
            if torch.cuda.is_available():
                # 如果指定了GPU ID，则只使用该GPU
                device = "cuda"
            else:
                print("CUDA不可用，使用CPU")
                device = "cpu"
        else:
            device = "cpu"
        
        # 控制精度以降低内存占用
        torch_dtype = torch.bfloat16
        if precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "fp32":
            torch_dtype = torch.float32
        
        print(f"使用精度: {precision}")
        
        # 设置模型加载配置
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        
        # 启用CPU卸载以节省GPU内存
        if use_cpu_offload and device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = "offload"
            model_kwargs["offload_state_dict"] = True
            print("启用CPU卸载功能以减少GPU内存使用")
        else:
            model_kwargs["device_map"] = device
        
        # 加载模型
        print("开始加载模型...")
        model_start = time.time()
        if is_lora and base_model_path:
            # 加载基础模型
            print(f"加载基础模型: {base_model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs
            )
            print(f"基础模型加载完成，耗时: {time.time() - model_start:.2f}秒")
            
            # 加载LoRA适配器
            print(f"加载LoRA适配器: {checkpoint_path}...")
            adapter_start = time.time()
            try:
                from peft import PeftModel
                # 在加载前先修改adapter_config.json
                adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                original_config = None
                if os.path.exists(adapter_config_path):
                    try:
                        # 备份原始配置
                        with open(adapter_config_path, 'r') as f:
                            original_config = json.load(f)
                        
                        # 检查是否包含models_new路径
                        if original_config and "base_model_name_or_path" in original_config and ("/data/lhc/models_new/" in original_config["base_model_name_or_path"] or "edf200_100hz_15000ms" in original_config["base_model_name_or_path"]):
                            # 创建临时配置，替换models_new路径
                            temp_config = original_config.copy()
                            # 使用当前模型路径替代
                            temp_config["base_model_name_or_path"] = base_model_path
                            
                            print(f"临时替换adapter_config.json中的路径: '{original_config['base_model_name_or_path']}' -> '{base_model_path}'")
                            
                            # 写入临时配置
                            with open(adapter_config_path, 'w') as f:
                                json.dump(temp_config, f, indent=2)
                    except Exception as e:
                        print(f"修改adapter_config.json时出错: {e}")
                
                # 加载适配器
                model = PeftModel.from_pretrained(
                    model,
                    checkpoint_path,
                    device_map="auto" if device.startswith("cuda") else device
                )
                
                # 恢复原始配置
                if original_config and os.path.exists(adapter_config_path) and "base_model_name_or_path" in original_config and ("/data/lhc/models_new/" in original_config["base_model_name_or_path"] or "edf200_100hz_15000ms" in original_config["base_model_name_or_path"]):
                    try:
                        with open(adapter_config_path, 'w') as f:
                            json.dump(original_config, f, indent=2)
                        print("已恢复原始adapter_config.json")
                    except Exception as e:
                        print(f"恢复原始adapter_config.json时出错: {e}")
                
                print(f"LoRA适配器加载完成，耗时: {time.time() - adapter_start:.2f}秒")
            except Exception as e:
                print(f"加载LoRA适配器失败: {e}")
                print(f"错误详情: {traceback.format_exc()}")
                print("尝试直接使用基础模型继续评估...")
        elif is_sharded_safetensors:
            # 加载分片safetensors模型
            print(f"加载分片safetensors模型: {checkpoint_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                **model_kwargs
            )
            print(f"分片safetensors模型加载完成，耗时: {time.time() - model_start:.2f}秒")
        else:
            # 直接加载完整模型
            print(f"加载完整模型...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                **model_kwargs
            )
            print(f"完整模型加载完成，耗时: {time.time() - model_start:.2f}秒")
        
        total_load_time = time.time() - model_load_start
        print(f"模型和分词器加载完成，总耗时: {total_load_time:.2f}秒")
        
        # 调试信息 - 列出检查点目录内容
        try:
            print(f"检查点目录内容：")
            for item in os.listdir(checkpoint_path):
                item_path = os.path.join(checkpoint_path, item)
                if os.path.isfile(item_path):
                    print(f"  文件: {item} ({os.path.getsize(item_path)} 字节)")
                else:
                    print(f"  目录: {item}")
        except Exception as e:
            print(f"无法列出目录内容: {e}")
            
    except Exception as e:
        print(f"加载模型出错: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        # 尝试清理显存
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return None
    
    # 加载测试数据
    test_data = load_test_data(test_data_path, max_samples)
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 系统提示
    print("准备评估数据...")
    system_prompt = (
        "You are a neurobiological expert specializing in EEG data analysis and sleep stage classification. "
        "Your task is to analyze the provided EEG data (including voltage values from the Fpz-Cz and Pz-Oz channels) "
        "and determine the current sleep stage of the volunteer based on the following classification criteria:\n"
        "0: Wakefulness (W)\n"
        "1: Non-rapid eye movement sleep stage 1 (N1)\n"
        "2: Non-rapid eye movement sleep stage 2 (N2)\n"
        "3: Non-rapid eye movement sleep stage 3 (N3)\n"
        "4: Non-rapid eye movement sleep stage 4 (N4)\n"
        "5: Rapid eye movement sleep stage (R)\n"
        "The EEG data is provided in the format (time in milliseconds, Fpz-Cz voltage in μV, Pz-Oz voltage in μV). "
        "The data spans 30000ms with a sampling interval of 5ms. In your analysis, pay attention to the following "
        "characteristics of each sleep stage:\n"
        "- Wakefulness (W): High-frequency, low-amplitude waves.\n"
        "- N1: Low-amplitude, mixed-frequency waves.\n"
        "- N2: Sleep spindles and K-complexes.\n"
        "- N3: High-amplitude, low-frequency delta waves.\n"
        "- N4: Dominant delta waves.\n"
        "- REM (R): Rapid eye movements and low muscle tone.\n"
        "Your response must be a single number (0, 1, 2, 3, 4, or 5) corresponding to the sleep stage. "
        "Do not include any additional text, punctuation, or explanations."
    )
    
    # 用于收集结果和标签
    results = []
    true_labels = []
    pred_labels = []
    total_time = 0.0
    
    # 创建进度条
    pbar = tqdm(test_data, desc=f"评估 {checkpoint_name}", ncols=100)
    
    # 评估每个样本
    for idx, data in enumerate(pbar):
        human_prompt = data["input"]
        true_label = int(data["output"])
        
        # 获取模型预测
        response, prediction, inference_time = get_model_prediction(
            model, tokenizer, human_prompt, system_prompt, device=device
        )
        
        # 更新结果
        true_labels.append(true_label)
        pred_labels.append(prediction)
        total_time += inference_time
        
        # 收集详细结果
        results.append({
            "sample_idx": idx,
            "true_label": true_label,
            "predicted_label": prediction,
            "response": response,
            "inference_time": inference_time
        })
        
        # 更新进度条
        current_accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if p is not None and t == p) / len(true_labels)
        pbar.set_postfix({"Accuracy": f"{current_accuracy:.4f}"})
        
        # 每10个样本打印一次中间结果
        if (idx + 1) % 10 == 0:
            # 计算当前的中间指标
            current_metrics = calculate_metrics(true_labels, pred_labels)
            
            print(f"\n--- 评估进度: {idx + 1}/{len(test_data)} 样本 ---")
            print(f"总体准确率: {current_metrics['accuracy']:.4f}  |  总体F1分数: {current_metrics['f1']:.4f}")
            
            # 打印各个睡眠阶段的中间准确率和F1分数
            print("各睡眠阶段指标:")
            print("{:<20} {:<10} {:<10}".format("睡眠阶段", "准确率", "F1分数"))
            print("-" * 40)
            
            for i in range(6):
                # 计算这个阶段的样本数
                true_class_count = sum(1 for label in true_labels if label == i)
                if true_class_count > 0:  # 只打印有样本的阶段
                    print("{:<20} {:<10.4f} {:<10.4f}  (样本数: {})".format(
                        SLEEP_STAGE_LABELS[i], 
                        current_metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}' ]['accuracy'],
                        current_metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}' ]['f1'],
                        true_class_count
                    ))
            
            # 如果提供了tensorboard，每10步更新一次
            if tb_writer:
                step = idx + 1
                tb_writer.add_scalar(f"{test_set_name}/accuracy_progress", current_metrics['accuracy'], step)
                tb_writer.add_scalar(f"{test_set_name}/f1_progress", current_metrics['f1_macro'], step)
                for i in range(6):
                    tb_writer.add_scalar(f"{test_set_name}/class_{i}_accuracy", current_metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['accuracy'], step)
                    tb_writer.add_scalar(f"{test_set_name}/class_{i}_f1", current_metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['f1'], step)
            
            # 打印一个分隔线
            print("-------------------------------------")
    
    # 计算最终指标
    metrics = calculate_metrics(true_labels, pred_labels)
    
    # 打印最终指标
    print("\n" + "="*50)
    print(f"模型评估结果汇总: {checkpoint_name}")
    print("="*50)
    
    # 总体指标
    print("\n[总体指标]")
    print(f"\u2022 总体准确率:  {metrics['accuracy']:.4f}")
    print(f"\u2022 宏观F1分数:   {metrics['f1_macro']:.4f}")
    print(f"\u2022 宏观精确率:   {metrics['precision_macro']:.4f}")
    print(f"\u2022 宏观召回率:   {metrics['recall_macro']:.4f}")
    print(f"\u2022 平均推理时间: {total_time/len(test_data):.4f} 秒/样本")
    print(f"\u2022 评估样本总数: {len(true_labels)}")
    
    # 打印各个睡眠阶段的详细指标
    print("\n[各睡眠阶段指标]")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("睡眠阶段", "准确率", "精确率", "召回率", "F1分数"))
    print("-"*60)
    
    for i in range(6):
        # 计算描述统计
        true_class_count = sum(1 for label in true_labels if label == i)
        pred_class_count = sum(1 for label in pred_labels if label == i)
        
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}  (样本数: {})".format(
            SLEEP_STAGE_LABELS[i], 
            metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['accuracy'],
            metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['precision'],
            metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['recall'],
            metrics['class_metrics'][f'Class_{SLEEP_STAGE_LABELS[i].split(" ")[0]}']['f1'],
            true_class_count
        ))
    
    # 深入分析
    if verbose:
        # 混淆矩阵分析
        print("\n[混淆矩阵分析]")
        cm = metrics['confusion_matrix']
        
        # 打印混淆矩阵标题
        header = "     " + "".join([f"{label.split(' ')[0]:<8}" for label in SLEEP_STAGE_LABELS])
        print(header)
        
        # 打印混淆矩阵内容
        for i in range(6):
            row = f"{SLEEP_STAGE_LABELS[i].split(' ')[0]:<5}" 
            for j in range(6):
                row += f"{cm[i][j]:<8}"
            print(row)
            
        # 错误分析
        print("\n[常见错误模式分析]")
        for i in range(6):
            for j in range(6):
                if i != j and cm[i][j] > 0:  # 只关注错误预测
                    error_rate = cm[i][j] / sum(cm[i]) if sum(cm[i]) > 0 else 0
                    if error_rate > 0.1:  # 只显示较重要的错误(>10%)
                        print(f"\u2022 {SLEEP_STAGE_LABELS[i]} 经常被错误预测为 {SLEEP_STAGE_LABELS[j]}: {error_rate:.2%}")
    
    print("\n" + "-"*50)
    
    # 记录到TensorBoard
    if tb_writer:
        try:
            # 先计算指标
            metrics = calculate_metrics(true_labels, pred_labels)
            
            # 手动写入TensorBoard
            print("正在写入测试集指标到TensorBoard...")
            tb_writer.add_scalar('test/accuracy', metrics['accuracy'], checkpoint_step)
            tb_writer.add_scalar('test/f1', metrics['f1_macro'], checkpoint_step)
            tb_writer.add_scalar('test/precision', metrics['precision_macro'], checkpoint_step)
            tb_writer.add_scalar('test/recall', metrics['recall_macro'], checkpoint_step)
            tb_writer.add_scalar('test/loss', 0.0, checkpoint_step)  # 添加loss占位符
            
            # 记录各个类别的指标
            for class_idx in range(6):
                stage_name = SLEEP_STAGE_LABELS[class_idx].split(' ')[0]  # 例如: 'Wake' -> 'W'
                class_metrics = metrics['class_metrics'][f'Class_{stage_name}']
                
                tb_writer.add_scalar(f'test/Class_{stage_name}/Accuracy', class_metrics['accuracy'], checkpoint_step)
                tb_writer.add_scalar(f'test/Class_{stage_name}/F1', class_metrics['f1'], checkpoint_step)
                tb_writer.add_scalar(f'test/Class_{stage_name}/Precision', class_metrics['precision'], checkpoint_step)
                tb_writer.add_scalar(f'test/Class_{stage_name}/Recall', class_metrics['recall'], checkpoint_step)
            
            # 添加混淆矩阵作为图像
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=[label.split(' ')[0] for label in SLEEP_STAGE_LABELS],
                        yticklabels=[label.split(' ')[0] for label in SLEEP_STAGE_LABELS])
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'混淆矩阵 - {test_set_name} - {checkpoint_name}')
            tb_writer.add_figure('test/ConfusionMatrix', fig, checkpoint_step)
            
            # 添加每个睡眠阶段的样本数量分布
            stage_samples = [sum(1 for label in true_labels if label == i) for i in range(6)]
            fig = plt.figure(figsize=(10, 6))
            plt.bar([label.split(' ')[0] for label in SLEEP_STAGE_LABELS], stage_samples)
            plt.xlabel('睡眠阶段')
            plt.ylabel('样本数量')
            plt.title(f'测试集样本分布 - {test_set_name}')
            tb_writer.add_figure('test/SampleDistribution', fig, checkpoint_step)
            
            # 确保数据被写入磁盘
            tb_writer.flush()
            
            # 验证写入的事件文件
            event_files = glob.glob(os.path.join(os.path.dirname(tb_writer.log_dir), "events.out.tfevents*"))
            if event_files:
                print(f"成功写入TensorBoard数据，事件文件: {[os.path.basename(f) for f in event_files]}")
                print(f"TensorBoard日志目录: {tb_writer.log_dir}")
            else:
                print(f"警告: 数据可能未被正确写入TensorBoard")
                
        except Exception as e:
            print(f"写入TensorBoard数据时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")
    
    # 保存结果
    save_results(results, metrics, checkpoint_name, save_dir, test_set_name)
    
    # 关闭TensorBoard写入器（确保在刷新后关闭）
    if tb_writer:
        tb_writer.close()
        print(f"已关闭TensorBoard写入器")
    
    # 返回评估结果
    return metrics

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点
    
    会处理两种情况：
    1. 目录中包含 checkpoint-* 格式的子目录（训练过程中生成的检查点）
    2. 目录本身就是一个已合并的模型目录（合并后的最终模型）
    """
    # 先检查是否有标准检查点格式
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if checkpoints:
        # 按修改时间排序，获取最新的检查点
        return max(checkpoints, key=os.path.getmtime)
    
    # 检查是否是合并后的模型目录
    # 判断目录中是否直接包含模型文件
    model_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "pytorch_model.bin",
        "generation_config.json"
    ]
    
    # 检查上述文件中的任一个是否存在
    for model_file in model_files:
        if os.path.exists(os.path.join(checkpoint_dir, model_file)):
            # 目录本身就是一个合并后的模型目录
            print(f"\n检测到合并模型目录: {checkpoint_dir}")
            return checkpoint_dir
    
    # 如果都不存在，返回None
    return None

def main():
    parser = argparse.ArgumentParser(description="评估大模型检查点在睡眠阶段分类任务上的性能")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="检查点目录，将评估最新的检查点")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="评估结果保存目录")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="TensorBoard日志目录，默认为None（使用checkpoint_dir中的runs目录）")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数，默认使用所有样本")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--template", type=str, default="alpaca", help="对话模板")
    parser.add_argument("--calculate_metrics", type=str, default="True", choices=["True", "False"], help="是否计算指标")
    parser.add_argument("--verbose", type=str, default="False", choices=["True", "False"], help="是否输出详细信息")
    parser.add_argument("--gpu_id", type=str, default="0", help="指定使用的GPU ID，例如'0,1,2,3'使用全部4张GPU")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], 
                     help="模型精度，bf16/fp16可降低显存占用")
    parser.add_argument("--use_cpu_offload", type=str, default="True", choices=["True", "False"], help="是否启用CPU卸载以节省GPU内存")
    
    args = parser.parse_args()
    
    # 查找最新的检查点
    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint_path:
        print(f"在目录 {args.checkpoint_dir} 中未找到检查点")
        return
    
    # 转换字符串参数为布尔值
    calculate_metrics = args.calculate_metrics.lower() == "true"
    verbose = args.verbose.lower() == "true"
    use_cpu_offload = args.use_cpu_offload.lower() == "true"
    
    # 设置CUDA_VISIBLE_DEVICES环境变量
    if args.device == "cuda" and args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"设置使用GPU: {args.gpu_id}")
    
    # 从测试数据路径中提取测试集名称
    test_set_name = os.path.basename(args.test_data).split('.')[0]
    
    # 设置TensorBoard目录
    tensorboard_dir = None
    if args.tensorboard_dir:
        # 如果明确指定了tensorboard_dir，则使用它
        tensorboard_dir = args.tensorboard_dir
    else:
        # 否则尝试使用checkpoint_dir所在目录的runs子目录
        checkpoint_base = os.path.dirname(args.checkpoint_dir)
        if os.path.exists(os.path.join(checkpoint_base, "runs")):
            tensorboard_dir = os.path.join(checkpoint_base, "runs")
            print(f"使用检查点目录中的runs目录: {tensorboard_dir}")
    
    if tensorboard_dir:
        print(f"使用TensorBoard日志目录: {tensorboard_dir}")
    else:
        print("未指定TensorBoard日志目录，不会记录TensorBoard日志")
    
    # 评估检查点
    evaluate_checkpoint(
        checkpoint_path,
        args.test_data,
        args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        template=args.template,
        tensorboard_dir=tensorboard_dir,
        verbose=verbose,
        precision=args.precision,
        use_cpu_offload=use_cpu_offload
    )

if __name__ == "__main__":
    main()
