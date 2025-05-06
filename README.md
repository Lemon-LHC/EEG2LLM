# EEG2LLM: 基于大语言模型的脑电信号睡眠阶段分类

## 项目简介

本项目（EEG2LLM）使用大语言模型(LLM)进行睡眠阶段分类。通过对LLM进行微调，使其能够根据脑电图(EEG)数据识别不同的睡眠阶段，包括清醒(W)、NREM睡眠(N1-N4)和REM睡眠(R)。项目还支持情绪特征提取与分析功能，可结合脑电信号进行多维度分析。

## 主要功能

- 数据预处理与平衡：支持多种数据平衡策略，解决睡眠阶段分布不均衡问题
- 情绪特征提取：使用预训练情绪模型从脑电信号中提取情绪特征
- 模型训练与微调：基于Llama模型架构，使用LoRA进行高效微调
- 模型评估与测试：提供详细的评估指标和可视化工具
- API服务部署：支持模型部署为API服务进行在线推理

## 技术特点

- 高效数据加载：支持批量加载、多线程处理和数据缓存
- 内存优化：动态调整线程数和批处理大小，避免内存溢出
- 训练优化：支持权重衰减、学习率调度和梯度累积
- 情绪分析：多模型集成的情绪特征提取（HVHA, HVLA, LVHA, LVLA）
- 评估全面：提供混淆矩阵、F1分数、准确率等多种评估指标

## 数据处理流程

### 1. 数据加载
- 读取EDF格式的脑电图数据和相应的注释文件
- 根据文件类型（ST/SC）和模式过滤文件
- 对每个EDF文件查找对应的注释文件

### 2. 信号处理
- 对信号进行重采样以达到目标采样率
- 从注释中提取睡眠阶段
- 按照睡眠阶段分段提取信号窗口

### 3. 情绪特征提取
- 加载预训练的情绪模型（HVHA, HVLA, LVHA, LVLA）
- 使用滑动窗口方法提取情绪特征
- 生成情绪编码序列，可选择解决情绪预测矛盾

### 4. 样本生成
- 为每个窗口创建指令（英文）
- 格式化信号数据为输入文本
- 添加情绪信息（如果启用）
- 生成包含指令、输入和输出的样本

### 5. 数据集处理
- 将所有样本合并到一个数据集
- 训练集和测试集划分
- 对训练集进行平衡处理（根据策略）
- 生成数据统计信息

## 部署环境

### 系统要求
- 操作系统：Ubuntu 20.04.6 LTS (Focal Fossa)
- GPU：4*NVIDIA GeForce RTX 3090 (24GB显存)
- CUDA版本：12.2
- NVIDIA驱动版本：535.183.01

### Python环境
- Python版本：3.12.0
- 虚拟环境管理：Anaconda/Conda (环境名：EEG2LLM)

### 核心依赖库
- PyTorch：2.6.0+cu126
- Transformers：4.49.0
- PEFT：0.12.0
- Accelerate：1.4.0
- Datasets：3.3.2
- BitsAndBytes：0.45.4
- TRL：0.9.6
- LLaMA-Factory：0.9.3.dev0
- TensorFlow：2.19.0（情绪模型使用）
- scikit-learn：1.6.1
- MNE：1.9.0 (脑电数据处理)
- PyEEG：直接从GitHub安装 (脑电信号特征提取)
- NumPy：1.26.4
- pandas：2.2.3
- matplotlib：3.10.1

## 使用方法

### 数据准备 - 基础版
```bash
python make_sleep_data_origin.py --input_dir /path/to/data --output_dir /path/to/output
```

### 数据准备 - 带情绪分析
```bash
python make_emo_data.py --input_dir /path/to/data --output_dir /path/to/output --include_emotion --emotion_model_dir /path/to/models
```

### 数据准备 - 完整参数示例
```bash
python make_emo_data.py \
    --input_dir /data/sleep-edfx \
    --output_dir /data/output \
    --max_files 44 \
    --n_jobs 22 \
    --target_sfreq 100 \
    --file_type st \
    --include_emotion \
    --emotion_model_dir /data/models/emotion \
    --emotion_window_length 2.0 \
    --emotion_step_size 0.5 \
    --balance_strategy balanced \
    --balance_alpha 0.5 \
    --weight_method sqrt_inverse \
    --max_windows 0
```

### 模型训练
```bash
bash train_balanced.sh
```

### 使用情绪模型训练
```bash
bash train_emotion.sh
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
conda create -n EEG2LLM python=3.12
conda activate EEG2LLM

# 安装PyTorch
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126

# 安装其他依赖
pip install transformers==4.49.0 peft==0.12.0 accelerate==1.4.0 datasets==3.3.2
pip install bitsandbytes==0.45.4 trl==0.9.6 scikit-learn==1.6.1
pip install mne==1.9.0 numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.1

# 安装PyEEG（从GitHub安装最新版本）
pip install git+https://github.com/forrestbao/pyeeg.git
```

### LLaMA-Factory安装
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 常见问题与故障排除

### TensorFlow不可用
确保在`EEG2LLM`环境中运行脚本：
```bash
conda run -n EEG2LLM python make_emo_data.py [参数]
```
如果仍有问题，尝试重新安装TensorFlow：
```bash
conda install -n EEG2LLM tensorflow=2.10.0
```

### 情绪模型加载失败
- 确认情绪模型目录包含四个必要模型文件（HVHA, HVLA, LVHA, LVLA）
- 检查模型格式是否兼容
- 尝试增加系统内存或使用CPU模式加载较大模型

### 数据集样本不足
- 检查设置的最大窗口数，设为0表示不限制
- 确认数据文件中包含有效的睡眠阶段注释
- 检查数据过滤条件是否过于严格
