# EEG2LLM: 基于大语言模型的脑电信号睡眠阶段分类

## 项目简介

本项目（EEG2LLM）旨在使用大语言模型 (LLM) 对脑电图 (EEG) 信号进行精细化的睡眠阶段分类。通过对LLM进行微调，使其能够准确识别不同的睡眠阶段，包括清醒期 (W)、非快速眼动睡眠期 (N1-N4) 和快速眼动睡眠期 (REM)。项目集成了先进的数据预处理流程，支持情绪特征的提取与融合，允许模型从多维度理解EEG信号。核心的数据处理脚本 `make_emo_data.py` 引入了 `tokenizer_id` 参数，该参数统一管理分词器的选择（如Llama, Qwen）及其关联的EEG信号和情绪特征提取的窗口与步长配置，提供了高度的灵活性和可配置性。

## 主要功能

-   **高级数据预处理**: 
    -   支持EDF/EDF+数据格式的读取与解析。
    -   通过 `tokenizer_id` 参数灵活配置EEG窗口长度、滑动步长及情绪特征提取的步长。
    -   包含信号重采样、滤波（需在代码中配置）和噪声添加选项。
-   **多维情绪特征提取**: 
    -   集成预训练的四分类情绪模型 (HVHA, HVLA, LVHA, LVLA)。
    -   支持滑动窗口提取情绪特征序列，并可选择是否解决预测矛盾。
-   **LLM样本生成**: 
    -   为每个EEG数据窗口生成结构化的训练样本，包含明确的指令 (instruction)、格式化的EEG输入 (input) 和期望的睡眠阶段输出 (output)。
    -   可选择性地将情绪特征序列和统计信息融入LLM的输入或指令中。
-   **数据集管理与平衡**: 
    -   支持多种数据集平衡策略（如基于类别权重的过采样/欠采样），以应对睡眠阶段数据天然的不均衡性。
    -   自动进行训练集/测试集划分。
    -   生成详细的数据集统计报告 (Excel和JSON格式)。
-   **模型训练与微调 (Train)**:
    -   （推测）提供基于 `LLaMA-Factory` 的高效微调脚本 (如 `train_balanced.sh`, `train_emotion.sh`)。
    -   （推测）支持使用LoRA (Low-Rank Adaptation) 等参数高效微调技术。
    -   （推测）可能包含针对不同数据集（如基础版、情绪增强版）的训练配置文件。
-   **模型评估与测试 (`api_infer` 目录)**:
    -   提供脚本进行模型性能的全面评估，计算准确率、F1分数、召回率、精确率以及混淆矩阵等关键指标。
    -   支持对不同模型检查点或不同实验设置下的模型进行基准测试和比较。
-   **评估结果可视化 (`utils` 目录)**:
    -   提供工具脚本对 `api_infer` 生成的评估结果（如混淆矩阵、性能指标）进行可视化分析，辅助模型调优。
-   **API服务部署 (Utils/Root)**:
    -   （推测）提供便捷的API服务部署脚本 (`api.sh` 或类似脚本)，用于将微调后的LLM模型部署为在线推理服务。

## 技术特点

-   **参数化数据处理**: 通过 `tokenizer_id` 集中控制关键的数据处理参数，简化配置并确保实验可复现性。
-   **高效数据加载与处理**: 
    -   利用多进程 (`multiprocessing`) 加速EDF文件的批量处理。
    -   训练流程中可能包含高效的数据加载器 (如 `torch.utils.data.DataLoader`)。
-   **内存优化**: 在数据处理脚本中，通过控制并行任务数和合理的窗口处理，管理内存使用。
-   **训练优化**: 
    -   支持LoRA、QLoRA等参数高效微调方法。
    -   可能集成学习率调度、权重衰减、梯度累积等高级训练策略。
-   **全面的情绪分析**: 集成经过验证的多模型情绪特征提取框架。
-   **详细的评估与可视化**: 通过 `api_infer` 和 `utils` 目录下的脚本，提供从原始指标计算到图形化结果展示的完整评估链条。
-   **可扩展性**: 代码结构设计考虑了未来添加新数据格式、新特征提取方法或新模型的可能性。

## 数据处理流程 (`make_emo_data.py`)

1.  **初始化与参数解析**: 
    *   脚本启动，解析命令行参数，核心是 `--tokenizer-id`。
    *   根据 `tokenizer_id` 确定所使用的分词器模型路径 (Llama或Qwen) 以及关联的EEG窗口/步长、情绪步长配置。
2.  **文件发现与过滤**: 
    *   扫描指定的 `--input_dir` 目录，查找EEG数据文件 (通常是 `-PSG.edf` 后缀)。
    *   根据 `--file-type` (sc/st/all) 和可选的 `--file-pattern` 进一步筛选文件。
    *   为每个数据文件自动查找对应的睡眠阶段注释文件 (通常是 `-Hypnogram.edf` 后缀)。
3.  **并行文件处理**: 
    *   使用 `ProcessPoolExecutor` 对筛选后的文件列表进行并行处理，任务数量由 `--n_jobs` 控制。
    *   每个子进程独立处理一个EEG文件及其注释：
        1.  **数据加载**: 使用MNE库读取EEG数据和注释。
        2.  **信号预处理**: 将信号重采样至 `--target_sfreq` (默认100Hz)。根据选项添加噪声 (`--add_noise`)。
        3.  **窗口提取**: 根据 `tokenizer_id` 决定的EEG窗口长度和步长，以及睡眠阶段注释，提取有效的EEG数据窗口。受 `--max_windows` 参数限制。
        4.  **情绪特征提取** (如果 `--include_emotion` 为True):
            *   加载预训练的四分类情绪模型 (来自 `--emotion_model_dir`)。
            *   对每个EEG窗口，使用 `--emotion_window_length` 作为子窗口，并根据 `tokenizer_id` 决定的情绪步长，提取情绪特征序列。
            *   根据 `--resolve_emotion_conflict` 选项处理预测矛盾。
        5.  **LLM样本构建**: 
            *   为每个处理过的EEG窗口（及其对应的情绪信息）构建一个LLM训练样本，包含`instruction`, `input` (格式化的EEG数据和情绪前缀), `output` (睡眠阶段标签), 和 `system` 提示。
            *   计算样本的Token长度。
4.  **数据聚合与划分**: 
    *   收集所有子进程生成的LLM样本。
    *   将聚合后的样本随机打乱，并按9:1的比例划分为原始训练集和测试集。
5.  **训练集平衡**: 
    *   根据 `--balance_strategy` (默认为`balanced`), `--balance_alpha` (默认0.5), 和 `--weight_method` (默认`sqrt_inverse`) 对原始训练集进行平衡处理，生成最终的平衡训练集。
6.  **结果保存与统计**: 
    *   将平衡后的训练集和原始测试集分别保存为JSON文件。
    *   文件名将详细反映所使用的配置 (包括`tokenizer_id`相关的分词器名称、EEG/情绪窗口/步长等)。
    *   生成包含详细处理参数和数据集分布的统计文件 (Excel `.xlsx` 和 JSON `.json` 格式)。

## 部署环境

### 系统要求
-   操作系统：Ubuntu 20.04.6 LTS (Focal Fossa) 或兼容的Linux发行版
-   GPU：推荐 4*NVIDIA GeForce RTX 3090 (24GB显存) 或更高级别，至少支持CUDA 12.1
-   CUDA版本：12.1 或 12.2 (根据PyTorch版本选择)
-   NVIDIA驱动版本：与CUDA版本兼容的最新驱动 (例如 535.x 或更高)

### Python环境
-   Python版本：3.12.0
-   虚拟环境管理：Anaconda/Conda (推荐环境名：`EEG2LLM`)

### 核心依赖库 (版本参考)
-   **LLM与训练框架:**
    -   `torch`: 2.3.0+cu121 (或更高版本，与CUDA匹配)
    -   `transformers`: 4.40.0 (或更高版本)
    -   `peft`: 0.10.0 (或更高版本)
    -   `accelerate`: 0.29.0 (或更高版本)
    -   `datasets`: 2.18.0 (或更高版本)
    -   `bitsandbytes`: 0.43.0 (Linux, 用于QLoRA等)
    -   `trl`: 0.8.6 (或更高版本)
    -   `llama-factory`: 0.7.0 (或更高版本，根据实际使用的分支)
-   **数据处理与科学计算:**
    -   `tensorflow`: 2.10.0 (或兼容版本，用于加载情绪模型 `.h5` 文件)
    -   `scikit-learn`: 1.3.0 (或更高版本)
    -   `mne`: 1.6.0 (或更高版本，用于EEG数据处理)
    -   `pyeeg`: (直接从GitHub安装: `pip install git+https://github.com/forrestbao/pyeeg.git`)
    -   `numpy`: 1.26.0 (或更高版本)
    -   `pandas`: 2.2.0 (或更高版本)
    -   `matplotlib`: 3.8.0 (或更高版本，用于可视化)
    -   `openpyxl`: (用于Excel文件读写)
    -   `tqdm`: (用于进度条显示)

## 使用方法

### 1. 数据准备与预处理 (`make_emo_data.py`)

核心脚本 `make_emo_data.py` 用于从原始EEG数据生成LLM微调所需的训练和测试样本。

**基本命令结构:**
```bash
conda run -n EEG2LLM python projects/EEG2LLM/data_process/make_emo_data.py --input_dir <原始数据路径> [其他参数]
```

**示例 A: 使用Llama配置 (tokenizer_id=1, 默认)**
```bash
conda run -n EEG2LLM python projects/EEG2LLM/data_process/make_emo_data.py \
    --input_dir /data/lhc/datasets/sleep-edfx \
    --tokenizer-id 1 \
    --include-emotion \
    --emotion_model_dir /data/lhc/models/emotion \
    --max_files 10 \
    --n_jobs 4 
```
*   使用Llama分词器及关联的EEG/情绪参数配置。
*   包含情绪特征，处理最多10个文件，使用4个CPU核心。

**示例 B: 使用Qwen配置 (tokenizer_id=2) 并自定义平衡策略**
```bash
conda run -n EEG2LLM python projects/EEG2LLM/data_process/make_emo_data.py \
    --input_dir /data/lhc/datasets/sleep-edfx \
    --tokenizer-id 2 \
    --file-type sc \
    --include-emotion \
    --emotion_model_dir /data/lhc/models/emotion \
    --balance_strategy balanced \
    --balance_alpha 0.3 \
    --max_files 20 \
    --n_jobs 8
```
*   使用Qwen分词器及关联的EEG/情绪参数配置。
*   仅处理`SC`类型文件，包含情绪特征，自定义了平衡alpha值。

*详细参数说明请参考 `projects/EEG2LLM/data_process/make_sleep_data_emotion_workflow.md` 文档。*

### 2. 模型训练 (`train` 目录)

假设 `train` 目录下存放了基于 `LLaMA-Factory` 的训练脚本和配置文件。

**示例训练命令 (请根据实际脚本调整):**

*   **训练基础模型 (不含情绪特征的数据集):**
    ```bash
    # 假设 train_balanced.sh 用于训练在平衡后的非情绪数据集上
    # 可能需要指定数据集路径、模型输出路径、LoRA配置等
    cd projects/EEG2LLM/train
    bash train_balanced.sh \
        --model_name_or_path <基础LLM路径, 如meta-llama/Llama-2-7b-hf> \
        --dataset_dir <make_emo_data.py输出的train数据目录> \
        --output_dir <模型保存路径> \
        --lora_target q_proj,v_proj 
        # ... 其他LLaMA-Factory参数 ...
    ```

*   **训练带情绪特征的模型:**
    ```bash
    # 假设 train_emotion.sh 用于训练在包含情绪特征的数据集上
    cd projects/EEG2LLM/train
    bash train_emotion.sh \
        --model_name_or_path <基础LLM路径> \
        --dataset_dir <make_emo_data.py输出的含情绪特征的train数据目录> \
        --output_dir <情绪模型保存路径> \
        # ... 其他LLaMA-Factory参数 ...
    ```
*请参考 `train` 目录下的具体脚本和 `LLaMA-Factory` 文档获取详细的训练参数。*

### 3. 模型评估与测试 (`api_infer` 目录)

模型评估和测试相关脚本位于 `/data/lhc/projects/EEG2LLM/api_infer` 目录下。

**示例评估命令 (推测性，请根据实际脚本调整):**
```bash
conda run -n EEG2LLM python /data/lhc/projects/EEG2LLM/api_infer/evaluate_model.py \
    --model_checkpoint_path <微调后模型的检查点路径> \
    --test_data_file <make_emo_data.py输出的test数据文件路径> \
    --output_dir <评估结果保存目录> \
    --batch_size 32
```
*   该脚本预计会加载指定的模型检查点和测试数据，执行推理，并计算各种性能指标，如准确率、F1分数、混淆矩阵等，并将结果保存到输出目录。

### 4. 评估结果可视化 (`utils` 目录)

对 `api_infer` 目录生成的评估结果进行可视化分析的脚本位于 `/data/lhc/projects/EEG2LLM/utils` 目录下。

**示例可视化命令 (请根据实际脚本调整):**
```bash
conda run -n EEG2LLM python /data/lhc/projects/EEG2LLM/utils/visualize_results.py \
    --evaluation_results_file <api_infer生成的结果文件路径，如metrics.json或confusion_matrix.csv> \
    --output_plot_path <可视化图表保存路径>
```
*   该脚本可能用于生成混淆矩阵图、PR曲线、ROC曲线等，帮助分析模型性能。

### 5. API服务部署 (`api.sh` 或 `utils` 目录)

**示例部署命令 :**
```bash
# 假设根目录下有 api.sh 脚本用于启动基于 LLaMA-Factory 的API服务
cd projects/EEG2LLM # 或项目根目录
bash api.sh \
    --model_name_or_path <微调后模型的路径> \
    --template default # 或适合您模型的对话模板
    # ... 其他API服务参数 ... 
```
*   这将启动一个HTTP API服务，可以通过发送请求来进行在线睡眠阶段分类。

## 环境配置

### 创建和配置Conda环境 (`EEG2LLM`)
```bash
# 1. 创建conda环境
conda create -n EEG2LLM python=3.12 -y
conda activate EEG2LLM

# 2. 安装PyTorch (请根据您的CUDA版本从PyTorch官网选择合适的命令)
# 例如 CUDA 12.1:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装LLaMA-Factory核心依赖 (版本请参考 LLaMA-Factory 官方文档或项目需求)
# 强烈建议您查阅 LLaMA-Factory 的官方 GitHub 仓库和文档，以获取最新和最准确的环境搭建指南、依赖版本和可选组件信息。
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# 下面的 extras (如 torch, metrics, deepspeed, qlora) 请根据 LLaMA-Factory 官方文档和您的具体需求进行选择和调整。
pip install -e .[torch,metrics,deepspeed,qlora] # 根据需要选择 extras
cd ..

# 4. 安装本项目特定依赖
pip install tensorflow==2.10.0 # 用于情绪模型加载
pip install scikit-learn mne pandas matplotlib openpyxl tqdm
pip install git+https://github.com/forrestbao/pyeeg.git

# 5. （可选）bitsandbytes (Linux，用于QLoRA等8位/4位量化训练)
# pip install bitsandbytes
```

## 常见问题与故障排除

### TensorFlow/Keras模型加载问题 (`.h5`)
-   **错误**: `ValueError: Unknown layer: ...` 或类似错误。
-   **原因**: TensorFlow版本不兼容或自定义对象未在加载时提供。
-   **解决**: 
    -   确保安装的TensorFlow版本 (推荐2.10.0) 与情绪模型训练时使用的版本兼容。
    -   `make_emo_data.py` 中的 `custom_load_model` 函数已尝试处理此问题，但如果模型包含非常特殊的自定义层，可能需要进一步调整。

### `make_emo_data.py` 常见问题
-   **找不到EDF文件或注释文件**: 
    -   仔细检查 `--input_dir` 路径是否正确。
    -   确认文件名符合 `-PSG.edf` (数据) 和 `-Hypnogram.edf` (注释) 的命名约定，并且两者前缀匹配。
-   **Tokenizer加载失败**: 
    -   检查 `tokenizer_id` 是否正确，并确认对应的分词器模型路径下文件完整。
-   **数据集样本过少**: 
    -   尝试将 `--max_windows` 设置为 `0` (不限制)。
    -   检查 `tokenizer_id` 对应的EEG窗口/步长配置是否适合您的数据。
    -   确认睡眠阶段注释是否有效且丰富。

### LLaMA-Factory 训练问题
-   **CUDA OOM (Out of Memory)**: 
    -   减小训练时的 `per_device_train_batch_size`。
    -   使用梯度累积 (`gradient_accumulation_steps`)。
    -   尝试QLoRA等量化技术以减少显存占用。
    -   确保 `LLaMA-Factory` 和 `bitsandbytes` (如果使用) 安装正确。
-   **依赖冲突**: 
    -   严格按照 `LLaMA-Factory` 的推荐版本安装依赖。在一个干净的Conda环境中安装通常能避免很多问题。

*其他问题请参考 `LLaMA-Factory` 的官方文档和GitHub Issues。*
