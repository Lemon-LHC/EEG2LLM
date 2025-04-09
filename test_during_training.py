#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试评估脚本，用于评估检查点性能并写入TensorBoard
"""

import os
import sys
import json
import time
import datetime
import argparse
import traceback
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_from_disk, DatasetDict

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试评估脚本")
    
    # 基本参数
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="包含检查点的目录")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="测试数据集路径")
    parser.add_argument("--tensorboard_dir", type=str, required=True,
                        help="TensorBoard日志目录")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="保存评估结果的目录")
    parser.add_argument("--global_step", type=int, default=None,
                        help="全局步数，用于TensorBoard记录")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="评估的批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                        help="生成的最大长度")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="评估超时时间（秒）")
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称或路径，如果不使用检查点")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    parser.add_argument("--no_write_tensorboard", action="store_true",
                        help="不写入TensorBoard（仅在控制台打印结果）")
    
    args = parser.parse_args()
    return args

def extract_step_from_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """从检查点目录名提取步数"""
    try:
        # 尝试从目录名称中提取步数
        if "checkpoint-" in checkpoint_dir:
            step = int(os.path.basename(checkpoint_dir).split("-")[-1])
            return step
        
        # 检查adapter_config.json中是否有步数信息
        config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if "step" in config:
                    return config["step"]
    except Exception as e:
        logger.warning(f"从检查点提取步数失败: {e}")
    
    return None

def load_test_data(data_path: str) -> DatasetDict:
    """加载测试数据集"""
    logger.info(f"加载测试数据: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        if isinstance(dataset, DatasetDict):
            # 如果是DatasetDict，尝试获取测试集
            if "test" in dataset:
                return dataset["test"]
            # 否则使用所有数据
            logger.warning(f"数据集中未找到'test'拆分，使用所有可用数据")
            return next(iter(dataset.values()))
        return dataset
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        raise

def load_model_and_tokenizer(checkpoint_dir: str, model_name: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和分词器"""
    logger.info(f"从{checkpoint_dir}加载模型和分词器")
    
    # 如果未提供model_name，尝试从adapter_config.json读取
    if model_name is None:
        config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                model_name = config.get("base_model_name_or_path")
                logger.info(f"从adapter_config.json中获取基础模型: {model_name}")
        
        # 如果仍然找不到，则使用默认值
        if model_name is None:
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            logger.warning(f"未找到基础模型名称，使用默认值: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型（包括LoRA适配器）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        logger.info(f"成功加载LoRA适配器")
    except Exception as e:
        logger.error(f"加载LoRA适配器失败，将使用基础模型: {e}")
    
    # 设置生成配置
    generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config = generation_config
    
    return model, tokenizer

def prepare_batches(dataset, tokenizer, batch_size: int = 4) -> List[Dict[str, torch.Tensor]]:
    """准备数据批次"""
    batches = []
    current_batch = {"input_ids": [], "attention_mask": [], "labels": []}
    batch_count = 0
    
    for item in dataset:
        # 假设数据集中有'input_ids', 'attention_mask', 'labels'字段
        # 如果没有，需要根据实际情况调整
        if "input_ids" not in item or "attention_mask" not in item:
            logger.warning(f"跳过格式不符的数据项: {item.keys()}")
            continue
        
        current_batch["input_ids"].append(torch.tensor(item["input_ids"]))
        current_batch["attention_mask"].append(torch.tensor(item["attention_mask"]))
        
        if "labels" in item:
            current_batch["labels"].append(torch.tensor(item["labels"]))
        
        batch_count += 1
        
        if batch_count == batch_size:
            # 转换为张量并添加到批次列表
            processed_batch = {
                "input_ids": torch.stack(current_batch["input_ids"]),
                "attention_mask": torch.stack(current_batch["attention_mask"])
            }
            
            if current_batch["labels"]:
                processed_batch["labels"] = torch.stack(current_batch["labels"])
            
            batches.append(processed_batch)
            
            # 重置批次
            current_batch = {"input_ids": [], "attention_mask": [], "labels": []}
            batch_count = 0
    
    # 处理剩余数据
    if batch_count > 0:
        processed_batch = {
            "input_ids": torch.stack(current_batch["input_ids"]),
            "attention_mask": torch.stack(current_batch["attention_mask"])
        }
        
        if current_batch["labels"] and len(current_batch["labels"]) == batch_count:
            processed_batch["labels"] = torch.stack(current_batch["labels"])
        
        batches.append(processed_batch)
    
    logger.info(f"准备了{len(batches)}个批次进行评估")
    return batches

def evaluate_model(model, batches, tokenizer, max_length: int = 512) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(batches):
            # 将批次移到设备上
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 模型推理
            outputs = model(**batch)
            
            # 收集损失
            if "labels" in batch:
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
            
            # 生成预测
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                do_sample=False
            )
            
            # 解码预测和标签
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(predictions)
            
            if "labels" in batch:
                # 处理标签（忽略填充标记-100）
                label_ids = batch["labels"].clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                all_labels.extend(labels)
            
            if i < 2:  # 仅显示前两个批次的示例
                logger.info(f"批次 {i+1} 示例:")
                for j in range(min(2, len(predictions))):
                    logger.info(f"  预测: {predictions[j][:100]}...")
                    if "labels" in batch and j < len(labels):
                        logger.info(f"  标签: {labels[j][:100]}...")
                    logger.info("---")
    
    # 计算指标
    metrics = {}
    
    if total_samples > 0:
        metrics["loss"] = total_loss / total_samples
    
    # 计算准确率等指标（需根据实际任务调整）
    if all_labels:
        correct = 0
        for pred, label in zip(all_predictions, all_labels):
            if pred.strip() == label.strip():
                correct += 1
        
        metrics["accuracy"] = correct / len(all_predictions) if all_predictions else 0
        
        # 可添加其他指标如F1、精确率等
        metrics["samples"] = len(all_predictions)
    
    return metrics

def write_to_tensorboard(writer, metrics: Dict[str, float], global_step: int, prefix: str = "test"):
    """将指标写入TensorBoard"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 将指标写入TensorBoard（同时使用标准和大写标签）
    for metric_name, value in metrics.items():
        # 使用标准标签
        writer.add_scalar(f"{prefix}/{metric_name}", value, global_step)
        # 使用大写标签（便于在TensorBoard中快速识别）
        writer.add_scalar(f"{prefix.upper()}/{metric_name}", value, global_step)
    
    # 添加时间戳和步数信息
    writer.add_text(f"{prefix}/debug_info", f"步数: {global_step}, 时间: {timestamp}", global_step)
    writer.add_text(f"{prefix.upper()}/debug_info", f"步数: {global_step}, 时间: {timestamp}", global_step)
    
    # 添加摘要文本
    summary_text = f"步数 {global_step} 的测试结果 ({timestamp}):\n"
    for metric_name, value in metrics.items():
        summary_text += f"- {metric_name}: {value:.4f}\n"
    
    writer.add_text(f"{prefix}/summary", summary_text, global_step)
    writer.add_text(f"{prefix.upper()}/summary", summary_text, global_step)
    
    # 添加超参数
    writer.add_hparams(
        {"step": global_step, "timestamp": timestamp},
        {f"hparam/{k}": v for k, v in metrics.items()}
    )
    
    # 确保写入完成
    writer.flush()

def find_root_tensorboard_dir(checkpoint_dir: str) -> Optional[str]:
    """
    尝试找到与训练和验证数据共享的根TensorBoard目录
    
    策略:
    1. 如果checkpoint_dir包含'checkpoint-'，查找其父目录下的tensorboard
    2. 查找checkpoint_dir父目录下的runs目录（LLaMA-Factory常用）
    3. 向上级目录搜索，直到找到tensorboard目录或runs目录
    
    Args:
        checkpoint_dir: 检查点目录路径
    
    Returns:
        找到的TensorBoard目录路径，或None表示未找到
    """
    logger.info(f"尝试为检查点找到根TensorBoard目录: {checkpoint_dir}")
    
    # 检查当前checkpoint_dir是否包含'checkpoint-'
    if "checkpoint-" in checkpoint_dir:
        # 获取父目录
        parent_dir = os.path.dirname(checkpoint_dir)
        
        # 检查是否存在tensorboard目录
        tensorboard_dir = os.path.join(parent_dir, "tensorboard")
        if os.path.exists(tensorboard_dir):
            logger.info(f"找到相对路径tensorboard目录: {tensorboard_dir}")
            return tensorboard_dir
        
        # 检查是否存在runs目录
        runs_dir = os.path.join(parent_dir, "runs")
        if os.path.exists(runs_dir):
            logger.info(f"找到LLaMA-Factory runs目录: {runs_dir}")
            return runs_dir
    
    # 向上级目录查找
    current_dir = checkpoint_dir
    max_depth = 3  # 最多向上查找3级目录
    
    for _ in range(max_depth):
        current_dir = os.path.dirname(current_dir)
        
        # 检查当前目录下是否有tensorboard或runs
        tensorboard_dir = os.path.join(current_dir, "tensorboard")
        if os.path.exists(tensorboard_dir):
            logger.info(f"在上级目录找到tensorboard目录: {tensorboard_dir}")
            return tensorboard_dir
        
        runs_dir = os.path.join(current_dir, "runs")
        if os.path.exists(runs_dir):
            logger.info(f"在上级目录找到runs目录: {runs_dir}")
            return runs_dir
    
    logger.warning("未找到现有的TensorBoard目录")
    return None

def backup_results(results: Dict[str, Any], results_dir: str):
    """备份评估结果"""
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存为JSON
    results_path = os.path.join(results_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存为文本文件（便于快速查看）
    txt_path = os.path.join(results_dir, "test_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"测试评估结果 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("-" * 50 + "\n")
        
        # 写入指标
        f.write("评估指标:\n")
        for metric_name, value in results["metrics"].items():
            f.write(f"{metric_name}: {value:.6f}\n")
        
        # 写入配置
        f.write("\n配置:\n")
        for key, value in results["config"].items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"结果已保存到: {results_path} 和 {txt_path}")
    
    return results_path, txt_path

def main():
    """主函数"""
    start_time = time.time()
    args = parse_arguments()
    
    # 确保目录存在
    if args.results_dir is None:
        args.results_dir = os.path.join(args.checkpoint_dir, "test_results")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(args.results_dir, "test_evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"开始测试评估，参数: {args}")
    
    try:
        # 提取步数
        global_step = args.global_step
        if global_step is None:
            global_step = extract_step_from_checkpoint(args.checkpoint_dir)
            if global_step is None:
                global_step = 0
                logger.warning(f"无法从检查点提取步数，使用默认值0")
        
        # 处理TensorBoard目录，优先使用已有的目录
        if not os.path.exists(args.tensorboard_dir):
            logger.warning(f"指定的TensorBoard目录不存在: {args.tensorboard_dir}")
            # 尝试查找与训练共享的tensorboard目录
            shared_tb_dir = find_root_tensorboard_dir(args.checkpoint_dir)
            if shared_tb_dir:
                logger.info(f"将使用共享的TensorBoard目录: {shared_tb_dir}")
                args.tensorboard_dir = shared_tb_dir
        
        # 确保tensorboard目录存在
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        logger.info(f"TensorBoard日志将写入: {args.tensorboard_dir}")
        
        # 加载测试数据
        test_dataset = load_test_data(args.test_data_path)
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(args.checkpoint_dir, args.model_name)
        
        # 准备批次
        batches = prepare_batches(test_dataset, tokenizer, args.batch_size)
        
        # 评估模型
        metrics = evaluate_model(model, batches, tokenizer, args.max_length)
        
        # 记录结果
        logger.info(f"评估指标:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.6f}")
        
        # 准备结果
        results = {
            "checkpoint_dir": args.checkpoint_dir,
            "global_step": global_step,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "config": vars(args)
        }
        
        # 备份结果
        results_path, txt_path = backup_results(results, args.results_dir)
        
        # 写入TensorBoard（除非明确禁用）
        if not args.no_write_tensorboard:
            try:
                logger.info(f"将结果写入TensorBoard: {args.tensorboard_dir}")
                writer = SummaryWriter(args.tensorboard_dir)
                write_to_tensorboard(writer, metrics, global_step)
                writer.close()
                
                # 添加备份TensorBoard目录
                backup_tb_dir = os.path.join(args.results_dir, "tensorboard")
                os.makedirs(backup_tb_dir, exist_ok=True)
                backup_writer = SummaryWriter(backup_tb_dir)
                write_to_tensorboard(backup_writer, metrics, global_step, prefix="test_backup")
                backup_writer.close()
                
                logger.info(f"TensorBoard写入完成")
            except Exception as e:
                logger.error(f"写入TensorBoard失败: {e}")
                logger.error(traceback.format_exc())
        
        # 计算总运行时间
        elapsed_time = time.time() - start_time
        logger.info(f"测试评估完成，总用时: {elapsed_time:.2f}秒")
        
        return 0
    
    except Exception as e:
        logger.error(f"测试评估失败: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 