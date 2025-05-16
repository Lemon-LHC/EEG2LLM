# make_sleep_data_emotion.py 工作流程文档

## 1. 概述

`make_sleep_data_emotion.py`是一个专门为睡眠数据处理和微调LLM模型设计的脚本。它将EEG（脑电图）睡眠数据转换为带有情绪特征标注的训练样本，通过分析EEG信号来分类睡眠阶段并提取情绪信息。核心改动包括引入`tokenizer_id`参数，该参数统一管理分词器选择及其相关的EEG和情绪处理窗口/步长设置。

本脚本的主要功能：
- 读取EDF格式的脑电图数据和相应的注释文件
- 根据选择的`tokenizer_id`配置（如Llama或Qwen）及其关联参数（EEG窗口/步长，情绪步长）进行处理
- 提取睡眠阶段信息（W, N1, N2, N3/N4, REM）
- 使用训练好的情绪模型提取情绪特征
- 生成带标注的样本用于LLM模型微调
- 对数据集进行平衡处理以提高训练质量
- 保存处理后的数据集并提供统计信息，文件名将反映所选配置

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
- transformers (用于分词器)

### 安装方法

推荐使用Conda环境：

```bash
conda create -n EEG2LLM python=3.12
conda activate EEG2LLM
conda install tensorflow=2.10.0 # 根据需要选择CPU或GPU版本
pip install mne scipy numpy pandas openpyxl tqdm transformers
pip install git+https://github.com/forrestbao/pyeeg.git
```

## 3. 工作流程图

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │  `tokenizer_id` │     │                 │
│  输入EDF文件    │────▶│   参数解析      │────▶│  加载情绪模型   │
│                 │     │ (决定分词器及配置)│     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                   │                      │
                                   └──────────────────────┼────────┐
                                                          │        │
                                                          ▼        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │  EEG及情绪参数  │
│  数据集平衡处理 │◀────│  睡眠阶段提取   │◀────│  基于`tokenizer_id`配置的情绪特征提取   │
│                 │     │                 │     │ (窗口/步长调整) │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  生成LLM样本    │────▶│  训练/测试集分割│────▶│  保存处理结果   │
│ (文件名体现配置)│     │                 │     │                 │
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
- `extract_emotion_features`: 从信号中提取情绪特征（其内部窗口和步长现在由`tokenizer_id`关联的配置决定）
- `extract_emotion_features_sliding_window`: 使用滑动窗口提取情绪特征
- `predict_emotions_multi_model`: 使用多个模型预测情绪
- `resolve_prediction_contradiction`: 解决情绪预测矛盾

#### 4.3 样本生成
- `create_llm_sample`: 创建用于LLM微调的样本 (EEG窗口长度由`tokenizer_id`关联的配置决定)
- `format_signal_data`: 将信号数据转换为文本格式
- `extract_stage_windows`: 提取各睡眠阶段的窗口 (EEG窗口和步长由`tokenizer_id`关联的配置决定)

#### 4.4 数据集处理
- `balance_dataset`: 对数据集进行平衡处理
- `process_directory`: 处理目录中的所有EDF文件
- `process_and_save_direct`: 处理单个文件并直接保存结果

#### 4.5 工具函数
- `custom_load_model`: 加载TensorFlow模型
- `get_tokenizer`: 根据 `tokenizer_path` (由 `tokenizer_id` 决定) 加载分词器
- `safe_calculate_token_length`: 安全计算文本的token长度
- `calculate_tokens`: 计算样本的token数量
- `safe_print`: 线程安全的打印函数

## 5. 数据处理流程

### 5.1 数据加载阶段
1. 解析命令行参数，特别是 `tokenizer_id`，以确定分词器及相关处理配置。
2. 扫描输入目录查找EDF文件。
3. 根据文件类型（ST/SC）和模式过滤文件。
4. 对每个EDF文件查找对应的注释文件。
5. 读取EDF文件并提取信号数据。

### 5.2 信号处理阶段
1. 对信号进行重采样以达到目标采样率。
2. 从注释中提取睡眠阶段。
3. **根据 `tokenizer_id` 选定的配置 (EEG窗口长度和EEG滑动步长)**，分段提取信号窗口。
4. 可选：添加随机噪声增强特征。

### 5.3 情绪特征提取
1. 加载预训练的情绪模型（HVHA, HVLA, LVHA, LVLA）。
2. **根据 `tokenizer_id` 选定的配置 (情绪特征提取步长)**，结合可配置的 `emotion_window_length`，使用滑动窗口方法提取情绪特征。
3. 生成情绪编码序列。
4. 可选：解决情绪预测矛盾。

### 5.4 样本生成
1. 为每个窗口创建指令（英文）。
2. 格式化信号数据为输入文本。
3. 添加情绪信息（如果启用）。
4. 生成包含指令、输入和输出的样本。

### 5.5 数据集处理
1. 将所有样本合并到一个数据集。
2. 使用sklearn进行训练集和测试集划分。
3. 对训练集进行平衡处理（根据策略）。
4. 生成数据统计信息。

### 5.6 结果保存
1. 保存训练集、测试集和全部数据集为JSON文件，**文件名将清晰反映所选的 `tokenizer_id` 配置**。
2. 生成并保存数据集统计信息（Excel和JSON格式）。
3. 记录处理日志。

## 6. 参数说明

### 主要参数
- `input_dir`: 输入目录，存放原始EDF睡眠数据文件。
- `max_files`: 最大处理文件数。
- `n_jobs`: 并行处理的作业数。
- `target_sfreq`: 目标采样率（Hz）。
- `file_type`: 文件类型过滤（`all`/`sc`/`st`）。
- `tokenizer_id`: **核心参数**。此ID决定：
    1.  使用的分词器模型及其路径。
    2.  EEG数据处理的窗口长度 (`eeg_window_sec`) 和滑动步长 (`eeg_step_sec`)。
    3.  情绪特征提取的滑动步长 (`emotion_step_size`)。
    *   **ID `1` (Llama 配置):**
        *   分词器路径: `/data/lhc/models/Llama-3.2-1B-Instruct`
        *   `emotion_step_size`: 0.5 秒
        *   `eeg_window_sec`: 15.0 秒
        *   `eeg_step_sec`: 15.0 秒
    *   **ID `2` (Qwen 配置):**
        *   分词器路径: `/data/lhc/models/Qwen/Qwen3-0.6B`
        *   `emotion_step_size`: 0.25 秒
        *   `eeg_window_sec`: 7.5 秒
        *   `eeg_step_sec`: 7.5 秒
    *   *注意: 脚本内 `output_dir` 固定为 `/data/lhc/datasets_new/emotion`，包含 `train`, `test`, `all` 子目录。*
- `max_windows`: 每个睡眠阶段最多处理的EEG窗口数。`0` 或 `None` 表示不限制。

### 情绪相关参数
- `include_emotion`: 布尔值，是否在样本中包含情绪信息。
- `emotion_model_dir`: 情绪模型所在的目录路径。
- `emotion_window_length`: 浮点数，情绪特征提取时每个子窗口的长度（秒）。此参数仍可独立设置，与 `tokenizer_id` 控制的 `emotion_step_size` 共同作用。
- `resolve_emotion_conflict`: 布尔值，是否尝试解决多个情绪模型预测结果之间的矛盾。
- `add_noise`: 布尔值，是否在EEG数据处理时添加少量随机噪声以增强数据。
- `normalize_features`: 布尔值，是否对提取的情绪特征进行标准化。

### 数据集平衡参数
- `balance_strategy`: 平衡策略，可选值：`balanced` (尝试使各类别样本数接近目标分布)、`original` (保持原始样本分布)、`none` (不进行平衡，等同于`original`)。
- `balance_alpha`: 浮点数，平衡因子，仅在 `balance_strategy='balanced'` 时生效。取值范围 [0, 1]，`0` 表示目标分布为完全均匀，`1` 表示目标分布为原始样本分布。脚本内默认为 `0.5`。
- `weight_method`: 权重计算方法，仅在 `balance_strategy='balanced'` 时生效。可选值：`inverse` (频率倒数)、`sqrt_inverse` (频率平方根的倒数)、`log_inverse` (频率对数的倒数)。

### 已移除或由 `tokenizer_id` 间接控制的旧参数
以下参数已从命令行移除，其功能和值现在由 `tokenizer_id` 统一管理：
- `--emotion-step-size`
- `--eeg-window-sec`
- `--eeg-step-sec`
- `--tokenizer-path` (现在由 `tokenizer_id` 映射得到)
- `--output-dir` (脚本内部固定)

## 7. 输出文件

### 数据文件
生成的数据集文件名将采用以下格式，以清晰反映所使用的主要配置：
`sleep_{file_type}_{max_files_proc}_{sfreq}hz_eeg{EEG_WIN}s-step{EEG_STEP}s[_emo{EMO_WIN}s-step{EMO_STEP}s]_{max_win_limit}_tokenizer_{tokenizer_name}_tok{avg_tok_count}[_bal{bal_alpha}_{bal_method}]_{timestamp}_{train|test}.json`

**关键组成部分解释:**
- `{file_type}`: 文件类型 (`sc`, `st`, `all`)。
- `{max_files_proc}`: 实际处理的文件数量。
- `{sfreq}hz`: 目标采样频率 (例如 `100hz`)。
- `eeg{EEG_WIN}s-step{EEG_STEP}s`: EEG窗口长度和滑动步长（秒），由 `tokenizer_id` 决定 (例如 `eeg15.0s-step15.0s` 或 `eeg7.5s-step7.5s`)。
- `_emo{EMO_WIN}s-step{EMO_STEP}s`: (如果 `include_emotion=True`) 情绪特征提取的窗口长度 (`EMO_WIN`，来自 `--emotion-window-length` 参数) 和滑动步长 (`EMO_STEP`，由 `tokenizer_id` 决定，例如 `_emo2.0s-step0.5s` 或 `_emo2.0s-step0.25s`)。
- `{max_win_limit}`: `winX` (X为`--max-windows`的值) 或 `win_all` (如果`--max-windows`为0或None)。
- `tokenizer_{tokenizer_name}`: 使用的分词器名称 (例如 `tokenizer_llama3.2-1b`, `tokenizer_qwen0.6b`)，由 `tokenizer_id` 决定。
- `tok{avg_tok_count}`: 数据集样本的平均Token数量。
- `_bal{bal_alpha}_{bal_method}`: (如果 `balance_strategy='balanced'`) 平衡策略的alpha值和权重方法。
- `{timestamp}`: 文件生成时的时间戳 (例如 `202310281530`)。
- `{train|test}`: 标记文件是训练集 (`train`) 还是测试集 (`test`)。

**文件名示例 (假设 `tokenizer_id=1`, 包含情绪, 进行了平衡):**
`sleep_sc_44_100hz_eeg15.0s-step15.0s_emo2.0s-step0.5s_win0_tokenizer_llama3.2-1b_tok350_bal0.5_sqrt_inverse_202310281530_train.json`

### 统计信息
- `{file_prefix}_stats.xlsx`: Excel格式的详细统计信息。`{file_prefix}` 与上述数据文件名类似，但不包含 `_train` 或 `_test` 后缀。
- `{file_prefix}_stats.json`: JSON格式的统计信息，内容与Excel文件对应。

## 8. 使用示例

### A. 最简用法：使用默认Llama配置 (tokenizer_id=1)
```bash
conda run -n EEG2LLM python make_emo_data.py --input_dir /path/to/your/edf_files
```
*   这将使用 `tokenizer_id=1` (Llama) 的所有默认关联设置：
    *   分词器: Llama-3.2-1B-Instruct
    *   EEG窗口: 15s, EEG步长: 15s
    *   情绪步长: 0.5s (如果后续通过 `--include_emotion` 启用情绪特征)
*   其他参数如 `max_files`, `n_jobs` 等将使用脚本内定义的默认值。

### B. 使用Qwen配置 (tokenizer_id=2) 并指定文件类型
```bash
conda run -n EEG2LLM python make_emo_data.py --input_dir /path/to/your/edf_files --tokenizer-id 2 --file-type sc
```
*   这将使用 `tokenizer_id=2` (Qwen) 的所有默认关联设置：
    *   分词器: Qwen3-0.6B
    *   EEG窗口: 7.5s, EEG步长: 7.5s
    *   情绪步长: 0.25s (如果后续通过 `--include_emotion` 启用情绪特征)
*   仅处理 `SC` 类型的文件。

### C. 包含情绪特征，并自定义部分情绪参数 (使用Llama, tokenizer_id=1)
```bash
conda run -n EEG2LLM python make_emo_data.py \\
    --input_dir /path/to/your/edf_files \\
    --tokenizer-id 1 \\
    --include-emotion \\
    --emotion_model_dir /path/to/your/emotion_models \\
    --emotion_window_length 2.5
```
*   使用 `tokenizer_id=1` (Llama) 配置。EEG窗口/步长和情绪步长将自动设为Llama的预设值。
*   启用了情绪特征提取，并指定了情绪模型的路径。
*   自定义了情绪特征提取的子窗口长度为 `2.5` 秒。

### D. 较完整的示例：处理特定数量文件，多核并行，自定义平衡策略 (使用Qwen, tokenizer_id=2)
```bash
conda run -n EEG2LLM python make_emo_data.py \\
    --input_dir /data/lhc/datasets/sleep-edfx \\
    --max_files 50 \\
    --n_jobs 16 \\
    --target_sfreq 100 \\
    --file_type st \\
    --tokenizer-id 2 \\
    --include-emotion \\
    --emotion_model_dir /data/lhc/models/emotion \\
    --emotion_window_length 2.0 \\
    --balance_strategy balanced \\
    --balance_alpha 0.3 \\
    --weight_method inverse \\
    --max_windows 100
```
*   使用 `tokenizer_id=2` (Qwen) 配置。
*   处理 `ST` 类型文件，最多50个，使用16个CPU核心。
*   启用了情绪特征，情绪子窗口长度为2.0秒。
*   使用平衡策略，alpha为0.3，权重方法为inverse。
*   每个睡眠阶段最多提取100个EEG窗口。

**重要提示**:
*   `--output-dir` 参数已移除，输出目录固定为脚本内定义的路径 (`/data/lhc/datasets_new/emotion` 下的 `train`, `test`, `all` 子目录)。
*   参数如 `--eeg-window-sec`, `--eeg-step-sec`, 和 `--emotion-step-size` 已被移除，它们的值现在由 `--tokenizer-id` 统一控制。

## 9. 常见问题和故障排除

### TensorFlow不可用
确保在`EEG2LLM`环境中运行脚本：
```bash
conda run -n EEG2LLM python make_emo_data.py [参数]
```
如果仍有问题，尝试重新安装TensorFlow（根据需要选择CPU或GPU版本）：
```bash
conda install -n EEG2LLM tensorflow=2.10.0
```

### 找不到EDF文件
- 检查`input_dir`参数是否正确。
- 确认EDF文件命名格式符合预期（如ST/SC前缀）。
- 验证文件是否有正确的扩展名`.edf`。

### 情绪模型加载失败
- 确认`emotion_model_dir`包含所需的四个模型 (`.h5` 文件)。
- 检查模型格式是否兼容 (TensorFlow/Keras)。
- 脚本默认在CPU上加载和运行模型，如果模型较大或复杂，请确保有足够内存。

### 数据集为空或样本数量过少
- 检查`max_windows`参数，设为`0`以不限制每个睡眠阶段提取的EEG窗口数量。
- 确认输入文件中包含有效的睡眠阶段注释，并且注释描述与脚本内定义的阶段映射一致。
- 检查过滤条件是否过于严格（如`file_type`过滤或`max_files`过小）。
- 检查所选 `tokenizer_id` 关联的EEG窗口/步长设置是否合理，过大或过小的窗口/步长可能导致无法有效提取样本。

### Tokenizer 加载失败
- 确认 `tokenizer_id` 指定的路径下 (`/data/lhc/models/Llama-3.2-1B-Instruct` 或 `/data/lhc/models/Qwen/Qwen3-0.6B`) 存在完整的分词器文件 (例如 `tokenizer.json`, `tokenizer_config.json` 等)。
- 确保 `transformers` 库已正确安装。

## 10. 维护和扩展指南

### 添加新的 `tokenizer_id` 配置
1.  **准备分词器**: 将新的分词器模型文件下载或放置到可访问的路径。
2.  **修改脚本 `make_emo_data.py`**:
    *   在 `main` 函数内的 `tokenizer_id_map` 字典中添加新的ID和对应的分词器模型路径。
    *   在 `tokenizer_specific_configs` 字典中为新的ID添加对应的 `emotion_step_size`, `eeg_window_sec`, 和 `eeg_step_sec` 配置。
    *   在 `main` 函数内推断 `tokenizer_name` 的逻辑中，为新的分词器路径添加一个简洁的名称映射，用于文件名生成。
3.  **更新本文档**: 在第6节参数说明中补充新的 `tokenizer_id` 及其完整配置。

### 添加新的情绪模型 (指四分类模型)
1. 准备与现有模型格式兼容的新模型（通常是 `.h5` 格式的Keras模型）。
2. 将模型文件放入`emotion_model_dir`目录 (或您指定的其他目录，并通过参数传入)。
3. 如果模型文件名或数量有变，可能需要修改`load_emotion_models`函数以正确加载新模型。
4. 如果新的情绪模型改变了情绪类别的含义或数量，需要更新`EMOTION_MAPPINGS`字典（如果脚本中有此类全局映射）和相关的指令文本（`create_llm_sample`函数中的`instruction`部分）。

### 支持新的EEG数据格式
1. 创建新的数据加载函数（类似于`read_edf_file`），该函数应能解析新格式并返回 MNE Raw 对象或类似的、包含信号数据和采样率的结构。
2. 确保新函数能处理通道选择和重采样。
3. 在`process_file`或`process_and_save_direct`函数中添加逻辑，以识别和调用新的数据加载函数（可能基于文件扩展名或其他元数据）。

### 添加新的数据集平衡策略
1. 在`balance_dataset`函数中添加新的策略逻辑（例如一个新的 `if/elif` 分支对应新的策略名称）。
2. 实现计算新策略下目标样本分布的算法。
3. 更新命令行参数的 `choices`（如果适用）和本文档中的参数说明。

### 性能优化
1. **增加缓存机制**: 对于一些固定输入的、计算密集型的预处理步骤（如特定文件的特征提取结果），可以考虑加入基于文件哈希或修改时间的缓存，避免重复计算。 (目前脚本似乎没有显式缓存)
2. **优化滑动窗口算法**: 检查 `extract_stage_windows` 和 `extract_emotion_features` 中的循环和切片操作，确保它们尽可能高效。对于非常大的数据集，可以考虑使用 `numpy` 的向量化操作替代部分循环。
3. **内存使用**: 对于大型EDF文件，`mne.io.read_raw_edf(..., preload=True)` 会将整个文件加载到内存。如果内存成为瓶颈，可以研究 `preload=False` 并按需读取数据块，但这会增加代码复杂性。
4. **并行处理参数**: 根据实际硬件调整 `--n_jobs`。过多的进程可能导致上下文切换开销过大。监控CPU和内存使用情况以找到最佳值。
5. **TensorFlow/Keras性能**: 确保情绪模型预测时使用了合适的批处理大小 (`batch_size` 在 `predict_emotions_multi_model` 中已有使用)。

### 添加新的输出格式
1. 确定新输出格式的规范 (例如 Parquet, TFRecord 等)。
2. 创建一个新的保存函数，该函数接收处理后的样本列表 (例如 `balanced_train_data` 或 `raw_test_data`)，并将其转换为新格式并保存。
3. 可能需要在 `process_directory` 函数末尾调用这个新的保存函数。
4. 如果需要，添加新的命令行参数来启用或配置新输出格式的生成，并更新本文档。

## 11. 代码维护最佳实践

- **文档同步**: 在修改代码（特别是参数、核心逻辑、输入/输出格式）后，立即更新此 `workflow.md` 文档。
- **版本控制**: 使用 Git 等版本控制系统跟踪所有代码和文档的更改。为重要的功能添加或修改创建分支，并通过 Pull Request 进行合并。
- **代码风格和注释**: 遵循一致的代码风格 (如 PEP8 for Python)。对复杂逻辑或重要假设添加清晰的注释。
- **模块化**: 保持函数和类的职责单一，便于理解、测试和维护。
- **错误处理**: 增强错误处理和日志记录，特别是在文件I/O、外部库调用和并行处理部分。
- **测试**:
    - **单元测试**: 为核心函数（如特征提取、样本格式化、平衡算法）编写单元测试。
    - **集成测试**: 使用小型的、固定的数据集测试整个处理流程，验证输出是否符合预期。
- **依赖管理**: 定期检查并更新依赖库到稳定版本，注意版本兼容性问题。在 `requirements.txt` 或 Conda `environment.yml` 文件中明确依赖版本。
- **配置管理**: 对于更复杂的项目，可以考虑将一些固定配置（如模型路径、默认参数）移到专门的配置文件中（如 YAML 或 JSON格式），而不是硬编码在脚本里。当前脚本通过 `param_mapping` 和 `tokenizer_id_map` 实现了一定程度的配置管理。
- **日志**: 使用 `logging` 模块替代 `print` 输出，以便更好地控制日志级别、格式和输出目标（文件、控制台等）。脚本中已使用 `safe_print`，可以考虑将其接入 `logging`。 