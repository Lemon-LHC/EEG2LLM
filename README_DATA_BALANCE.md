# 睡眠阶段分类数据集平衡工具

本工具提供了睡眠阶段分类任务的数据集平衡功能，可以通过过采样和加权采样来解决类别不平衡问题。

## 实现的平衡策略

- **原始分布 (original)**: 保持数据集原始分布不变
- **平衡采样 (balanced)**: 对少数类进行过采样，使各类别样本数量更加平衡
- **加权采样 (weighted)**: 为每个样本分配权重，反映在训练中的重要性

## 类别权重计算方法

- **none**: 所有类别权重相等
- **inverse**: 反比例权重，类别权重与样本数量成反比
- **sqrt_inverse**: 反比例平方根权重，更温和的平衡方式
- **effective_samples**: 有效样本数方法，通过平衡系数调整权重

## 实际应用示例

### 示例1：典型不平衡分布

对于以下不平衡数据集分布：
- Wake (W): 21.6%
- NREM Stage 1 (N1): 19.9%
- NREM Stage 2 (N2): 21.6%
- NREM Stage 3 (N3): 9%
- NREM Stage 4 (N4): 4.4%
- REM Sleep (R): 23.5%

#### 推荐配置
- **平衡策略**: `balanced`
- **平衡系数**: `0.5`
- **权重计算方法**: `sqrt_inverse`

#### 平衡前后效果对比
```
平衡前:
- Wake (W):     21.6% (1080 samples)
- NREM Stage 1: 19.9% (995 samples)
- NREM Stage 2: 21.6% (1080 samples)
- NREM Stage 3:  9.0% (450 samples)
- NREM Stage 4:  4.4% (220 samples)
- REM Sleep:    23.5% (1175 samples)

平衡后 (balanced, alpha=0.5):
- Wake (W):     16.7% (1080 samples)
- NREM Stage 1: 15.4% (995 samples)
- NREM Stage 2: 16.7% (1080 samples)
- NREM Stage 3: 15.5% (1000 samples) ↑
- NREM Stage 4: 15.5% (1000 samples) ↑
- REM Sleep:    20.3% (1310 samples)

类别权重 (sqrt_inverse):
- Wake (W):     1.00
- NREM Stage 1: 1.04
- NREM Stage 2: 1.00
- NREM Stage 3: 1.54
- NREM Stage 4: 2.20
- REM Sleep:    0.97
```

通过这种配置，我们保持了相对均衡的分布，同时对少数类(N3和N4)进行了适度的补偿，避免模型过度偏向多数类。

## 使用方法

### 选项1：使用新的训练脚本

直接使用新的训练脚本，它已经集成了数据平衡功能：

```bash
python train.py \
  --model_name /data/lhc/models/Llama-3.2-1B-Instruct \
  --train_dataset edf197_100hz_10000ms_tok8521_train \
  --test_dataset edf197_100hz_10000ms_tok8521_test \
  --sampling_strategy balanced \
  --balance_alpha 0.5 \
  --class_weight_method inverse
```

### 选项2：单独使用数据平衡工具

您也可以单独使用数据平衡工具来处理数据集：

```bash
python dataset_balancer.py \
  --input_file /data/lhc/datasets_new/sleep/train/edf197_100hz_10000ms_tok8521_train.json \
  --output_file /data/lhc/datasets_new/sleep/balanced/edf197_100hz_10000ms_tok8521_train_balanced.json \
  --strategy balanced \
  --balance_alpha 0.5 \
  --weight_method inverse \
  --update_config
```

然后使用平衡后的数据集进行训练：

```bash
python train_old.py \
  --model_name /data/lhc/models/Llama-3.2-1B-Instruct \
  --train_dataset edf197_100hz_10000ms_tok8521_train_balanced \
  --test_dataset edf197_100hz_10000ms_tok8521_test
```

## 参数说明

### 数据平衡相关参数

- `--sampling_strategy`: 数据采样策略，可选 "original"、"balanced" 或 "weighted"
- `--balance_alpha`: 平衡系数，范围 [0-1]，值越大对少数类的过采样程度越高
- `--class_weight_method`: 类别权重计算方法，可选 "none"、"inverse"、"sqrt_inverse" 或 "effective_samples"

### 一般训练参数

- `--model_name`: 模型路径
- `--train_dataset`: 训练数据集名称
- `--test_dataset`: 测试数据集名称
- `--cutoff_len`: 序列截断长度
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数
- `--train_batch_size`: 训练批次大小
- 更多参数请参考完整的帮助文档 (`python train.py --help`)

## 平衡效果示例

假设原始数据集中各睡眠阶段样本分布如下：

```
Wake (W): 5000样本
NREM Stage 1 (N1): 2000样本
NREM Stage 2 (N2): 10000样本
NREM Stage 3 (N3): 3000样本
NREM Stage 4 (N4): 1500样本
REM Sleep (R): 800样本
```

使用平衡采样策略 (balance_alpha=0.5) 后，样本分布可能变为：

```
Wake (W): 5000样本
NREM Stage 1 (N1): 5000样本
NREM Stage 2 (N2): 10000样本
NREM Stage 3 (N3): 5000样本
NREM Stage 4 (N4): 5000样本
REM Sleep (R): 5000样本
```

## 注意事项

1. 平衡处理会创建新的数据集文件，不会修改原始数据集
2. 平衡后的数据集会保存在 `datasets_new/sleep/balanced/` 目录下
3. 平衡后的数据集会自动添加到LLaMA-Factory的数据集配置中
4. 使用加权采样策略时，样本数量不变，但每个样本会添加权重字段
5. 为了进行更公平的模型评估，建议在测试集上保持原始数据分布，只对训练集进行平衡处理
6. 在训练过程中观察各个类别的性能指标，必要时调整平衡策略和参数
7. 平衡处理可能会增加训练集大小，相应地增加训练时间

## 相关文档

- [数据平衡功能实现细节](./balancing_explanation.md)
- [训练脚本使用说明](./README_TRAINING_SCRIPTS.md)
- [模型评估指南](./README_EVALUATION.md)（如有） 