# 睡眠阶段分类训练脚本说明

本文档详细介绍了用于睡眠阶段分类模型训练的三个脚本，它们都集成了数据平衡功能，专门针对睡眠阶段数据的"四高两低"分布特点进行了优化。

## 脚本概述

我们提供了以下训练脚本，它们使用相同的平衡策略和参数，但执行方式略有不同：

1. **train_balanced.sh** - 基于train_old.py的标准训练脚本
2. **train_absolute.sh** - 使用全绝对路径的健壮版训练脚本
3. **run_balanced_training.sh** - 基于LLaMA-Factory的分布式训练脚本（注意：该脚本存在参数兼容性问题）
4. **run_balanced_training_fixed.sh** - 修复版LLaMA-Factory训练脚本，二步法实现数据平衡

所有脚本都配置了相同的数据平衡策略：
- 平衡策略：`balanced`（对少数类进行过采样）
- 平衡系数：`0.5`（中等强度的平衡）
- 权重计算方法：`sqrt_inverse`（反比例平方根权重）

## 脚本兼容性说明

**重要提示**：原始的`run_balanced_training.sh`脚本可能会遇到如下错误：

```
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--sampling_strategy', 'balanced', '--balance_alpha', '0.5', '--class_weight_method', 'sqrt_inverse', '--test_interval', '3000']
```

这是因为LLaMA-Factory不支持我们自定义的数据平衡参数。推荐使用以下两种方法之一：

1. 使用`train_absolute.sh`或`train_balanced.sh`脚本（基于我们自己的训练代码）
2. 使用修复版的`run_balanced_training_fixed.sh`脚本（先平衡数据，再训练）

## 脚本详细说明

### 1. train_balanced.sh

```bash
#!/bin/bash
# 睡眠阶段分类模型训练脚本
# 使用推荐的平衡策略和权重计算方法
```

**特点**：
- 使用项目中现有的`train_old.py`脚本
- 切换到脚本所在目录后执行
- 日志输出到结果目录

**适用场景**：
- 单GPU训练
- 标准训练流程
- 需要与现有train_old.py保持兼容

### 2. train_absolute.sh

```bash
#!/bin/bash
# 睡眠阶段分类模型训练脚本 - 使用绝对路径执行
# 使用推荐的balanced策略和sqrt_inverse权重计算方法
```

**特点**：
- 使用完全绝对路径执行，不依赖工作目录
- 明确指定python解释器路径
- 增加了错误处理和退出码检查
- 全部使用双引号包含变量，更健壮

**适用场景**：
- 从任意目录执行
- 需要更可靠的执行和错误处理
- 通过cron或其他自动化工具执行

### 3. run_balanced_training.sh

```bash
#!/bin/bash
# 使用torchrun启动LLaMA-Factory训练 - 平衡样本版本
```

**特点**：
- 使用LLaMA-Factory的训练框架
- 配置了分布式训练（4个GPU）
- 使用torchrun进行启动
- 包含更多高级训练参数

**适用场景**：
- 多GPU训练
- 需要LLaMA-Factory特有功能
- 大规模数据集训练

### 4. run_balanced_training_fixed.sh

```bash
#!/bin/bash
# 使用torchrun启动LLaMA-Factory训练 - 平衡样本版本 (修复版)
# 此版本先通过dataset_balancer.py进行数据平衡处理，再使用LLaMA-Factory训练
```

**特点**：
- 采用二步法解决LLaMA-Factory参数兼容性问题
- 第一步：使用`dataset_balancer.py`预处理数据集
- 第二步：使用LLaMA-Factory训练平衡后的数据集
- 保留分布式训练和多GPU支持
- 自动更新数据集配置

**适用场景**：
- 需要LLaMA-Factory特有功能
- 需要多GPU训练
- 需要更高级的训练参数

## 日志输出说明

所有脚本都配置为同时将训练日志：
1. 显示在终端（实时查看）
2. 保存到结果目录中的train.log文件

这是通过`tee`命令实现的，该命令会读取标准输入，然后写入到文件的同时输出到标准输出。这样您可以：
- 实时监控训练进度
- 在训练结束后查看完整日志记录
- 在训练过程中中断并继续，不会丢失日志

如果您希望只将日志输出到文件而不显示在终端，可以将以下行：
```bash
command 2>&1 | tee "${LOG_FILE}"
```
修改为：
```bash
command > "${LOG_FILE}" 2>&1
```

## 使用方法

### 准备工作

1. 确保脚本具有执行权限：

```bash
chmod +x /data/lhc/projects/fine/train_balanced.sh
chmod +x /data/lhc/projects/fine/train_absolute.sh
chmod +x /data/lhc/projects/fine/run_balanced_training.sh
chmod +x /data/lhc/projects/fine/run_balanced_training_fixed.sh
```

2. 确保脚本中的虚拟环境路径正确：

所有脚本现在都包含了激活虚拟环境的步骤。请根据您的环境配置修改以下行：

```bash
# 默认配置（根据您的环境可能需要修改）
source /home/lhc/anaconda3/bin/activate llama_factory
```

如果您使用的是不同的环境路径或名称，请在执行脚本前进行修改。

### 执行脚本

#### 方案1：使用train_balanced.sh

```bash
cd /data/lhc/projects/fine
./train_balanced.sh
```

#### 方案2：使用train_absolute.sh

```bash
# 可从任意目录执行
/data/lhc/projects/fine/train_absolute.sh
```

#### 方案3：使用run_balanced_training.sh

```bash
cd /data/lhc/projects/fine
./run_balanced_training.sh
```

#### 方案4：使用run_balanced_training_fixed.sh

```bash
cd /data/lhc/projects/fine
chmod +x run_balanced_training_fixed.sh
./run_balanced_training_fixed.sh
```

## 数据平衡策略说明

这些脚本针对睡眠阶段数据分布特点（W、N1、N2、R较均衡，N3、N4显著偏低）进行了优化，采用了以下平衡策略：

### 平衡策略：balanced

- **原理**：对少数类（N3、N4）进行过采样，使各类别样本数量更加平衡
- **效果**：增加N3和N4的样本数量，保持其他类别样本数不变
- **参数控制**：balance_alpha=0.5提供适中的平衡程度

### 权重计算方法：sqrt_inverse

- **原理**：使用反比例平方根计算类别权重，避免极端权重值
- **公式**：W(c) = √(max_count / count(c))
- **优势**：比inverse更温和，避免过度补偿少数类

对于典型的睡眠阶段分布：
```
- Wake (W):     21.6% (1080 samples)
- NREM Stage 1: 19.9% (995 samples)
- NREM Stage 2: 21.6% (1080 samples)
- NREM Stage 3:  9.0% (450 samples)
- NREM Stage 4:  4.4% (220 samples)
- REM Sleep:    23.5% (1175 samples)
```

平衡后的分布示例：
```
- Wake (W):     16.7% (1080 samples)
- NREM Stage 1: 15.4% (995 samples)
- NREM Stage 2: 16.7% (1080 samples)
- NREM Stage 3: 15.5% (1000 samples) ↑
- NREM Stage 4: 15.5% (1000 samples) ↑
- REM Sleep:    20.3% (1310 samples)
```

## 重要参数说明

所有脚本共享的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| sampling_strategy | balanced | 使用平衡采样策略，对少数类进行过采样 |
| balance_alpha | 0.5 | 平衡系数，控制过采样强度 |
| class_weight_method | sqrt_inverse | 使用反比例平方根计算类别权重 |
| num_epochs | 2.0-3.0 | 训练轮数 |
| learning_rate | 5e-05 | 学习率 |
| lora_rank | 8 | LoRA秩 |

## 输出结果

所有脚本都会创建带有时间戳的输出目录：

- **train_balanced.sh** 和 **train_absolute.sh**：
  ```
  /data/lhc/results/{TIMESTAMP}_balanced_sqrt_inverse/
  ```

- **run_balanced_training.sh**：
  ```
  /data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf200_100hz_10000ms_balanced_{TIMESTAMP}/
  ```

- **run_balanced_training_fixed.sh**：
  ```
  /data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf200_100hz_10000ms_balanced_{TIMESTAMP}/
  ```

输出目录包含：
- 训练日志
- 模型检查点
- TensorBoard日志
- 评估结果（如果启用）

## 查看训练进度

训练期间和训练后，可以通过以下方式查看进度和结果：

```bash
# 查看训练日志
tail -f /data/lhc/results/{TIMESTAMP}_balanced_sqrt_inverse/train.log

# 启动TensorBoard查看训练指标
tensorboard --logdir=/data/lhc/results/{TIMESTAMP}_balanced_sqrt_inverse
``` 