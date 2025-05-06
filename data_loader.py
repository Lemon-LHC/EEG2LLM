import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from transformers import AutoTokenizer

class SleepDataset(Dataset):
    """睡眠分期数据集"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取输入和标签
        input_text = item.get("input", "")
        if not input_text and "conversations" in item and len(item["conversations"]) > 0:
            input_text = item["conversations"][0]["value"]
            
        # 获取标签
        label = None
        if "output" in item:
            label = int(item["output"])
        elif "conversations" in item and len(item["conversations"]) > 1:
            label = int(item["conversations"][1]["value"])
        
        if label is None:
            label = 0  # 默认标签，避免出错
        
        # 获取阶段信息（如果有）
        stage = item.get("stage", "unknown")
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 移除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 添加labels和stage
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        encoding["stage"] = stage
        
        return encoding

def load_test_data(file_path, max_samples=None):
    """加载测试数据
    
    Args:
        file_path: 测试数据文件路径
        max_samples: 最大样本数，如果设置则随机选择指定数量的样本
        
    Returns:
        测试数据列表
    """
    print(f"加载测试数据: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 如果设置了最大样本数，随机选择样本
        if max_samples and max_samples < len(test_data):
            np.random.seed(42)  # 固定随机种子以确保可重复性
            indices = np.random.choice(len(test_data), max_samples, replace=False)
            test_data = [test_data[i] for i in indices]
            print(f"随机选择了 {max_samples} 个样本进行测试")
        else:
            print(f"加载了 {len(test_data)} 个测试样本")
    except Exception as e:
        print(f"加载测试数据时出错: {str(e)}")
        test_data = []
    
    return test_data

def prepare_test_dataloader(tokenizer, test_data, batch_size=8):
    """准备测试数据加载器
    
    Args:
        tokenizer: 分词器对象
        test_data: 测试数据列表
        batch_size: 批处理大小
        
    Returns:
        test_dataloader: 测试数据加载器
    """
    if not test_data:
        print("警告: 测试数据为空")
        return None
        
    # 确保每个样本都包含stage信息
    stage_names = ["Wake", "N1", "N2", "N3", "N4", "REM"]
    
    for sample in test_data:
        # 如果没有stage字段，根据output/label设置对应的睡眠阶段
        if "stage" not in sample:
            label = None
            if "output" in sample:
                label = int(sample["output"])
            elif "conversations" in sample and len(sample["conversations"]) > 1:
                label = int(sample["conversations"][1]["value"])
                
            if label is not None and 0 <= label < len(stage_names):
                sample["stage"] = stage_names[label]
            else:
                sample["stage"] = "unknown"
    
    # 创建数据集
    test_dataset = SleepDataset(test_data, tokenizer)
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "stage": [item["stage"] for item in batch]
        }
    )
    
    return test_dataloader 