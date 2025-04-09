# 训练过程测试评估回调

本文档介绍了如何使用测试回调功能在LLaMA-Factory训练过程中自动评估测试集。

## 主要功能

1. 在验证集评估完成后自动触发测试集评估
2. 测试结果实时记录到TensorBoard
3. 支持参数化配置，如测试间隔、批处理大小、精度等
4. 完整的日志记录，便于调试

## 实现原理

我们通过以下方式实现自动测试评估：

1. 创建`projects/fine/llm_patch`目录，包含Python站点包补丁
2. 通过`sitecustomize.py`在Python启动时自动加载补丁
3. 在LLaMA-Factory的SFT训练流程中注入自定义回调
4. 使用原有的测试评估脚本执行实际评估

## 文件结构

```
projects/fine/
├── llm_patch/
│   ├── __init__.py         # 补丁主入口，包含SFT训练流程的patch
│   └── sitecustomize.py    # Python站点包补丁，自动加载patch
├── test_callback.py        # 测试回调类
├── train.py                # 主训练脚本(已修改)
└── test_during_training.py # 测试评估脚本
```

## 环境变量配置

通过以下环境变量控制测试回调：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| ENABLE_TEST_CALLBACK | 是否启用测试回调 | false |
| TEST_INTERVAL | 测试间隔(步) | 20 |
| TEST_DATASET_PATH | 测试数据集路径 | 必须设置 |
| TENSORBOARD_DIR | TensorBoard日志目录 | 必须设置 |
| TEST_BATCH_SIZE | 测试批处理大小 | 1 |
| EVAL_PRECISION | 评估精度 | fp16 |

## 使用方法

### 1. 调整现有的训练命令

在训练命令中添加以下参数：

```bash
--eval_delay 0
```

这确保验证评估后立即进行测试评估，不会延迟。

### 2. 确保训练脚本设置了正确的环境变量

在`train_model`函数中，需要设置以下环境变量：

```python
env["PYTHONPATH"] = f"{patch_dir}:{env.get('PYTHONPATH', '')}"
env["ENABLE_TEST_CALLBACK"] = "true"
env["TEST_INTERVAL"] = str(args.test_interval)
env["TEST_DATASET_PATH"] = os.path.join(args.dataset_dir, "test", f"{args.test_dataset}.json")
env["TENSORBOARD_DIR"] = tensorboard_dir
env["TEST_BATCH_SIZE"] = str(args.train_batch_size)
env["EVAL_PRECISION"] = args.eval_precision
```

### 3. 启动训练

正常启动训练，测试回调会自动加载并在每次验证后执行测试评估。

## 调试技巧

1. 检查日志输出，查看是否有"[测试回调]"前缀的消息
2. 验证TensorBoard是否包含测试集指标
3. 如果测试回调未启动，检查环境变量是否正确设置

## 注意事项

1. 测试回调会增加训练时间，因为每次验证后都会进行测试评估
2. 确保测试数据集路径正确，否则回调会被跳过
3. 半精度推理可能导致精度略有下降，但会显著提高评估速度 