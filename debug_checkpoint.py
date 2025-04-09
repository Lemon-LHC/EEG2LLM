#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="检查点调试工具")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="检查点目录路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备，cuda或cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    # 步骤1: 检查检查点目录和相关文件
    print(f"[1/5] 检查文件和目录: {args.checkpoint_dir}")
    checkpoint_path = args.checkpoint_dir
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点路径不存在: {checkpoint_path}")
        return

    print(f"检查点路径存在，列出目录内容:")
    try:
        files = os.listdir(checkpoint_path)
        for file in files:
            file_path = os.path.join(checkpoint_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
                print(f"  - {file} ({size:.2f} MB)")
            else:
                print(f"  - {file}/ (目录)")
    except Exception as e:
        print(f"列出目录内容失败: {e}")
    
    # 步骤2: 检查是否为LoRA检查点
    print(f"\n[2/5] 检查是否为LoRA检查点")
    is_lora = False
    if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
        print("发现adapter_config.json文件，确认为LoRA检查点")
        is_lora = True
        
        # 尝试读取adapter_config.json
        try:
            with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                print(f"adapter_config内容:\n{json.dumps(adapter_config, indent=2, ensure_ascii=False)}")
                
                if "base_model_name_or_path" in adapter_config:
                    base_model_path = adapter_config["base_model_name_or_path"]
                    print(f"基础模型路径: {base_model_path}")
                    
                    # 检查基础模型路径是否存在
                    if os.path.exists(base_model_path):
                        print(f"基础模型路径存在")
                    else:
                        print(f"警告: 基础模型路径不存在: {base_model_path}")
        except Exception as e:
            print(f"读取adapter_config.json失败: {e}")
    else:
        print("未发现adapter_config.json文件")
        
        # 检查是否有adapter_model文件
        if os.path.isfile(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
           os.path.isfile(os.path.join(checkpoint_path, "adapter_model.bin")):
            print("发现adapter_model文件，确认为LoRA检查点")
            is_lora = True
        else:
            print("未发现adapter_model文件，可能不是LoRA检查点")
    
    # 步骤3: 检查测试数据
    print(f"\n[3/5] 检查测试数据: {args.test_data}")
    if not os.path.exists(args.test_data):
        print(f"错误: 测试数据文件不存在: {args.test_data}")
        return
    
    # 尝试加载测试数据开头的几个样本
    try:
        data_load_start = time.time()
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        data_load_time = time.time() - data_load_start
        print(f"测试数据加载成功，共 {len(test_data)} 个样本，耗时: {data_load_time:.2f}秒")
        
        if len(test_data) > 0:
            print(f"首个样本示例:")
            sample = test_data[0]
            print(f"  - 输入长度: {len(sample.get('input', ''))} 字符")
            print(f"  - 输出: {sample.get('output', '')}")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
    
    # 步骤4: 加载分词器
    print(f"\n[4/5] 尝试加载分词器")
    tokenizer = None
    try:
        tokenizer_start = time.time()
        # 如果是LoRA检查点，尝试从adapter_config中获取基础模型路径
        if is_lora:
            base_model_path = None
            try:
                with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                    if "base_model_name_or_path" in adapter_config:
                        base_model_path = adapter_config["base_model_name_or_path"]
            except:
                pass
            
            if base_model_path and os.path.exists(base_model_path):
                print(f"从基础模型加载分词器: {base_model_path}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            else:
                print(f"无法确定基础模型路径，尝试直接从检查点加载分词器")
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        else:
            print(f"从检查点加载分词器")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        tokenizer_time = time.time() - tokenizer_start
        print(f"分词器加载成功，耗时: {tokenizer_time:.2f}秒")
        
        # 输出vocab大小等信息
        print(f"分词器信息:")
        print(f"  - vocab大小: {len(tokenizer)}")
        print(f"  - 模型最大长度: {tokenizer.model_max_length}")
    except Exception as e:
        print(f"加载分词器失败: {e}")
    
    # 步骤5: 尝试加载模型(限时尝试)
    print(f"\n[5/5] 尝试加载模型 (限时30秒)")
    model = None
    try:
        model_start = time.time()
        
        # 设置超时时间(30秒)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("模型加载超时")
        
        # 注册超时处理函数
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30秒超时
        
        try:
            if is_lora and base_model_path and os.path.exists(base_model_path):
                print(f"从基础模型加载模型: {base_model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto" if args.device == "cuda" else "cpu"
                )
                
                print(f"加载LoRA适配器: {checkpoint_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    checkpoint_path,
                    device_map="auto" if args.device == "cuda" else "cpu"
                )
            else:
                print(f"直接加载模型: {checkpoint_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto" if args.device == "cuda" else "cpu"
                )
            
            model_time = time.time() - model_start
            print(f"模型加载成功，耗时: {model_time:.2f}秒")
            
            # 打印模型信息
            params = sum(p.numel() for p in model.parameters())
            print(f"模型信息:")
            print(f"  - 参数量: {params / 1e6:.2f}M")
        except TimeoutError:
            print("模型加载超时，这可能是问题所在")
        finally:
            # 取消超时
            signal.alarm(0)
    except Exception as e:
        print(f"加载模型失败: {e}")
    
    # 总结
    total_time = time.time() - start_time
    print(f"\n调试完成，总耗时: {total_time:.2f}秒")
    
    if model is None:
        print("问题诊断: 模型加载失败或超时，这可能是eval_checkpoint.py脚本卡住的原因")
        print("建议解决方案:")
        print("1. 检查GPU内存是否足够")
        print("2. 尝试使用更小的模型或减少批次大小")
        print("3. 确保基础模型路径正确存在")
    else:
        print("模型加载成功，问题可能出在评估阶段")

if __name__ == "__main__":
    main()
