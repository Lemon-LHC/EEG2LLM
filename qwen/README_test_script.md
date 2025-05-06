# 睡眠阶段分类大模型性能测试脚本

这个测试脚本用于评估大模型在睡眠阶段分类任务上的性能。它能够测试模型对脑电图数据进行睡眠分期的准确性，并生成详细的性能报告。

## 功能特点

- 测试大模型对睡眠阶段分类任务的原始性能（无微调）
- 计算多种评估指标：准确率、精确率、召回率、F1分数等
- 生成混淆矩阵与可视化
- 分别评估对各个睡眠阶段分类的性能
- 详细记录每个样本的预测结果
- 支持选择性测试部分样本

## 安装依赖

```bash
pip install numpy tqdm scikit-learn matplotlib seaborn transformers torch
```

## 使用方法

### 基本使用

```bash
python test_llm_sleep_classification.py --data_dir /path/to/test/data --model_path ./Qwen
```

### 完整参数

```bash
python test_llm_sleep_classification.py \
  --data_dir /path/to/test/data \
  --model_path ./Qwen \
  --output_dir ./results \
  --max_samples 100
```

### 参数说明

- `--data_dir`：必需参数，测试数据目录路径，包含json格式的测试样本
- `--model_path`：模型路径，默认为`./Qwen`
- `--output_dir`：结果输出目录，默认为`./results`
- `--max_samples`：最大测试样本数，默认为全部

## 输出结果

脚本会在输出目录创建一个带有时间戳的子目录（如`eval_20230525_123456`），包含以下文件：

1. `performance_summary.txt`：包含模型性能的详细报告
2. `detailed_results.txt`：包含每个样本的详细预测结果
3. `confusion_matrix.png`：睡眠阶段分类的混淆矩阵可视化图

### 性能报告内容

性能报告包括：

- 数据集统计信息
- 各睡眠阶段样本分布
- 总体性能指标（准确率、精确率、召回率、F1分数）
- 各睡眠阶段的单独性能指标
- 混淆矩阵
- 平均推理时间

## 使用示例

测试所有测试数据：

```bash
python test_llm_sleep_classification.py --data_dir /data/lhc/datasets/sleep-edfx/processed_1/processed_test
```

测试少量样本作为快速验证：

```bash
python test_llm_sleep_classification.py --data_dir /data/lhc/datasets/sleep-edfx/processed_1/processed_test --max_samples 20
```

使用特定路径的模型：

```bash
python test_llm_sleep_classification.py --data_dir /data/lhc/datasets/sleep-edfx/processed_1/processed_test --model_path /path/to/custom/model
```

## 注意事项

1. 测试前确保已经生成了包含睡眠阶段分类测试数据的JSON文件
2. 确保已经下载并准备好Qwen模型
3. 对于大量测试样本，测试过程可能需要较长时间
4. 如果GPU内存有限，可以考虑减少`max_samples`参数的值 