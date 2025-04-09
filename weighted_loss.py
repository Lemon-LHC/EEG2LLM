import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数，用于处理类别不平衡问题
    特别适用于睡眠阶段分类中REM阶段识别率低的情况
    """
    def __init__(
        self, 
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean"
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        if class_weights is not None:
            # 将权重列表转换为tensor
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None
    
    def forward(self, logits, labels):
        """
        计算加权交叉熵损失
        
        Args:
            logits: 模型输出的预测logits，形状为 [batch_size, seq_len, vocab_size]
            labels: 目标标签，形状为 [batch_size, seq_len]
            
        Returns:
            加权交叉熵损失
        """
        # 调整权重设备以匹配输入
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        
        # 计算交叉熵损失，使用指定的类别权重
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )

def parse_class_weights(weights_str: str) -> List[float]:
    """
    解析类别权重字符串为浮点数列表
    
    Args:
        weights_str: 以逗号分隔的浮点数字符串，例如 "1.0,1.0,1.0,1.0,1.0,3.0"
        
    Returns:
        浮点数列表，例如 [1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
    """
    if not weights_str:
        return None
    
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        print(f"解析类别权重: {weights}")
        return weights
    except ValueError as e:
        print(f"解析类别权重出错: {e}")
        print(f"权重字符串格式错误: {weights_str}")
        print(f"请使用逗号分隔的浮点数，例如 '1.0,1.0,1.0,1.0,1.0,3.0'")
        return None

def create_weighted_loss_fn(weights_str: Optional[str] = None):
    """
    创建加权损失函数
    
    Args:
        weights_str: 以逗号分隔的浮点数字符串，表示各类别权重
        
    Returns:
        加权交叉熵损失函数实例
    """
    if weights_str:
        weights = parse_class_weights(weights_str)
        return WeightedCrossEntropyLoss(class_weights=weights)
    else:
        return nn.CrossEntropyLoss()

# 使用示例
# weights_str = "1.0,1.0,1.0,1.0,1.0,3.0"  # 给REM阶段3倍的权重
# loss_fn = create_weighted_loss_fn(weights_str) 