# EEG2LLM: 基于大语言模型的脑电图睡眠阶段分类

## 项目简介

本项目（EEG2LLM）使用大语言模型(LLM)进行睡眠阶段分类。通过对LLM进行微调，使其能够根据脑电图(EEG)数据识别不同的睡眠阶段，包括清醒(W)、NREM睡眠(N1-N4)和REM睡眠(R)。

## 主要功能

- 数据预处理与平衡：支持多种数据平衡策略，解决睡眠阶段分布不均衡问题
- 模型训练与微调：基于Llama模型架构，使用LoRA进行高效微调
- 模型评估与测试：提供详细的评估指标和可视化工具
- API服务部署：支持模型部署为API服务进行在线推理

## 技术特点

- 高效数据加载：支持批量加载、多线程处理和数据缓存
- 内存优化：动态调整线程数和批处理大小，避免内存溢出
- 训练优化：支持权重衰减、学习率调度和梯度累积
- 评估全面：提供混淆矩阵、F1分数、准确率等多种评估指标

## 部署环境

### 系统要求
- 操作系统：Ubuntu 20.04.6 LTS (Focal Fossa)
- GPU：NVIDIA GeForce RTX 3090 (24GB显存)
- CUDA版本：12.2
- NVIDIA驱动版本：535.183.01

### Python环境
- Python版本：3.12.0
- 虚拟环境管理：Anaconda/Conda (环境名：fine)

### 核心依赖库
- PyTorch：2.6.0+cu126
- Transformers：4.49.0
- PEFT：0.12.0
- Accelerate：1.4.0
- Datasets：3.3.2
- BitsAndBytes：0.45.4
- TRL：0.9.6
- LLaMA-Factory：0.9.3.dev0
- TensorFlow：2.19.0
- scikit-learn：1.6.1
- MNE：1.9.0 (脑电数据处理)
- NumPy：1.26.4
- pandas：2.2.3
- matplotlib：3.10.1

## 使用方法

### 数据准备
```bash
python make_SCsleep_edfx.py --input_dir /path/to/data --output_dir /path/to/output
```

### 模型训练
```bash
bash train_balanced.sh
```

### 评估模型
```bash
python test_api_llm.py --model_path /path/to/model --test_data /path/to/test_data
```

### 部署API服务
```bash
bash api.sh
```

## 环境配置

### 创建和配置环境
```bash
# 创建conda环境
conda create -n fine python=3.12
conda activate fine

# 安装PyTorch
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126

# 安装其他依赖
pip install transformers==4.49.0 peft==0.12.0 accelerate==1.4.0 datasets==3.3.2
pip install bitsandbytes==0.45.4 trl==0.9.6 scikit-learn==1.6.1
pip install mne==1.9.0 numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.1
```

### LLaMA-Factory安装
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```
