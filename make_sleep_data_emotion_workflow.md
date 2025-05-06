# make_sleep_data_emotion.py 工作流程文档

## 1. 概述

`make_sleep_data_emotion.py`是一个专门为睡眠数据处理和微调LLM模型设计的脚本。它将EEG（脑电图）睡眠数据转换为带有情绪特征标注的训练样本，通过分析EEG信号来分类睡眠阶段并提取情绪信息。

本脚本的主要功能：
- 读取EDF格式的脑电图数据和相应的注释文件
- 提取睡眠阶段信息（W, N1, N2, N3/N4, REM）
- 使用训练好的情绪模型提取情绪特征
- 生成带标注的样本用于LLM模型微调
- 对数据集进行平衡处理以提高训练质量
- 保存处理后的数据集并提供统计信息

## 2. 环境要求

### 依赖项

- Python 3.8+
- TensorFlow 2.10.0（用于情绪模型）
- MNE-Python（处理EEG数据）
- PyEEG（脑电信号特征提取，直接从GitHub安装）
- SciPy（信号处理）
- NumPy（数值计算）
- Pandas（数据分析）
- openpyxl（Excel文件生成）
- tqdm（进度显示）

### 安装方法

推荐使用Conda环境：

```bash
conda create -n EEG2LLM python=3.12
conda activate EEG2LLM
conda install tensorflow=2.10.0
pip install mne scipy numpy pandas openpyxl tqdm
pip install git+https://github.com/forrestbao/pyeeg.git
```

## 3. 工作流程图

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  输入EDF文件    │────▶│  加载情绪模型   │────▶│  数据预处理     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  数据集平衡处理 │◀────│  睡眠阶段提取   │◀────│  情绪特征提取   │
│                 │     │                 │     │                 │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  生成LLM样本    │────▶│  训练/测试集分割│────▶│  保存处理结果   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 4. 主要组件和函数

### 核心功能组件

#### 4.1 数据加载与预处理
- `read_edf_file`: 读取EDF格式的脑电图数据
- `find_annotation_file`: 查找对应的注释文件
- `interpolate_signal`: 对信号进行插值以达到目标采样率
- `extract_stages_from_annotations`: 从注释中提取睡眠阶段

#### 4.2 情绪特征提取
- `load_emotion_models`: 加载预训练的情绪模型
- `extract_emotion_features`: 从信号中提取情绪特征
- `extract_emotion_features_sliding_window`: 使用滑动窗口提取情绪特征
- `predict_emotions_multi_model`: 使用多个模型预测情绪
- `resolve_prediction_contradiction`: 解决情绪预测矛盾

#### 4.3 样本生成
- `create_llm_sample`: 创建用于LLM微调的样本
- `format_signal_data`: 将信号数据转换为文本格式
- `extract_stage_windows`: 提取各睡眠阶段的窗口

#### 4.4 数据集处理
- `balance_dataset`: 对数据集进行平衡处理
- `process_directory`: 处理目录中的所有EDF文件
- `process_and_save_direct`: 处理单个文件并直接保存结果

#### 4.5 工具函数
- `custom_load_model`: 加载TensorFlow模型
- `safe_calculate_token_length`: 安全计算文本的token长度
- `calculate_tokens`: 计算样本的token数量
- `safe_print`: 线程安全的打印函数

## 5. 数据处理流程

### 5.1 数据加载阶段
1. 扫描输入目录查找EDF文件
2. 根据文件类型（ST/SC）和模式过滤文件
3. 对每个EDF文件查找对应的注释文件
4. 读取EDF文件并提取信号数据

### 5.2 信号处理阶段
1. 对信号进行重采样以达到目标采样率
2. 从注释中提取睡眠阶段
3. 按照睡眠阶段分段提取信号窗口
4. 可选：添加随机噪声增强特征

### 5.3 情绪特征提取
1. 加载预训练的情绪模型（HVHA, HVLA, LVHA, LVLA）
2. 使用滑动窗口方法提取情绪特征
3. 生成情绪编码序列
4. 可选：解决情绪预测矛盾

### 5.4 样本生成
1. 为每个窗口创建指令（英文）
2. 格式化信号数据为输入文本
3. 添加情绪信息（如果启用）
4. 生成包含指令、输入和输出的样本

### 5.5 数据集处理
1. 将所有样本合并到一个数据集
2. 使用sklearn进行训练集和测试集划分
3. 对训练集进行平衡处理（根据策略）
4. 生成数据统计信息

### 5.6 结果保存
1. 保存训练集、测试集和全部数据集为JSON文件
2. 生成并保存数据集统计信息（Excel和JSON格式）
3. 记录处理日志

## 6. 参数说明

### 主要参数
- `input_dir`: 输入目录，存放原始EDF睡眠数据文件
- `output_dir`: 输出目录，用于保存生成的训练数据
- `max_files`: 最大处理文件数
- `n_jobs`: 并行处理的作业数
- `target_sfreq`: 目标采样率（Hz）
- `file_type`: 文件类型过滤（all/sc/st）

### 情绪相关参数
- `include_emotion`: 是否包含情绪信息
- `emotion_model_dir`: 情绪模型目录
- `emotion_window_length`: 情绪编码窗口长度（秒）
- `emotion_step_size`: 情绪编码步长（秒）
- `resolve_emotion_conflict`: 是否解决情绪矛盾

### 数据集平衡参数
- `balance_strategy`: 平衡策略（balanced/original/none）
- `balance_alpha`: 平衡因子（0为完全均衡，1为原始分布）
- `weight_method`: 权重计算方法（inverse/sqrt_inverse/log_inverse）

## 7. 输出文件

### 数据文件
- `{file_type}_edf{max_files}_{sfreq}hz_{window_length}ms_tok{token_count}_{emotion_flag}_bal{alpha}_{weight_method}_train.json`: 训练集
- `{file_type}_edf{max_files}_{sfreq}hz_{window_length}ms_tok{token_count}_{emotion_flag}_bal{alpha}_{weight_method}_test.json`: 测试集
- `{file_type}_edf{max_files}_{sfreq}hz_{window_length}ms_tok{token_count}_{emotion_flag}_bal{alpha}_{weight_method}_all.json`: 全部数据

### 统计信息
- `{file_prefix}_stats.xlsx`: Excel格式的统计信息
- `{file_prefix}_stats.json`: JSON格式的统计信息

## 8. 使用示例

### 基本用法
```bash
conda run -n EEG2LLM python make_sleep_data_emotion.py --input_dir /path/to/edf/files --output_dir /path/to/output
```

### 带情绪特征的处理
```bash
conda run -n EEG2LLM python make_sleep_data_emotion.py --input_dir /path/to/edf/files --output_dir /path/to/output --include_emotion --emotion_model_dir /path/to/models
```

### 完整示例
```bash
conda run -n EEG2LLM python make_sleep_data_emotion.py \
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

## 9. 常见问题和故障排除

### TensorFlow不可用
确保在`EEG2LLM`环境中运行脚本：
```bash
conda run -n EEG2LLM python make_sleep_data_emotion.py [参数]
```
如果仍有问题，尝试重新安装TensorFlow：
```bash
conda install -n EEG2LLM tensorflow=2.10.0
```

### 找不到EDF文件
- 检查`input_dir`参数是否正确
- 确认EDF文件命名格式符合预期（如ST/SC前缀）
- 验证文件是否有正确的扩展名`.edf`

### 情绪模型加载失败
- 确认`emotion_model_dir`包含所需的四个模型
- 检查模型格式是否兼容
- 增加内存或使用CPU模式尝试加载较大模型

### 数据集为空或样本数量过少
- 检查`max_windows`参数，设为0以不限制窗口数量
- 确认文件中包含有效的睡眠阶段注释
- 检查是否过滤条件过于严格（如文件类型过滤）

## 10. 维护和扩展指南

### 添加新的情绪模型
1. 准备与现有模型格式兼容的新模型
2. 将模型文件放入emotion_model_dir目录
3. 修改`load_emotion_models`函数以支持新模型
4. 更新情绪编码映射表以反映新的情绪类别

### 支持新的EEG数据格式
1. 创建新的数据加载函数（类似于`read_edf_file`）
2. 确保新函数返回与现有格式兼容的信号数据
3. 在`process_file`函数中添加对新格式的支持

### 添加新的数据集平衡策略
1. 在`balance_dataset`函数中添加新策略
2. 实现计算目标分布的算法
3. 更新命令行参数和帮助文档

### 性能优化
1. 增加缓存机制以避免重复处理
2. 优化滑动窗口算法以减少计算开销
3. 使用更高效的数据结构减少内存使用
4. 调整并行处理参数以适应不同硬件

### 添加新的输出格式
1. 创建处理和转换JSON数据的函数
2. 实现数据保存到新格式的逻辑
3. 更新命令行参数和帮助文档

## 11. 代码维护最佳实践

- 保持文档和代码同步更新
- 对性能关键部分添加缓存机制
- 使用版本控制跟踪代码更改
- 为新功能编写测试用例
- 定期检查和更新依赖项
- 关注情绪模型的更新和改进 