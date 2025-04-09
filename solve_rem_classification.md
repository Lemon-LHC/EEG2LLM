# 解决REM睡眠阶段识别率低的问题

本指南提供了解决REM睡眠阶段(R)被错误识别为其他阶段的完整解决方案。

## 问题分析

通过查看混淆矩阵和评估指标，我们发现REM阶段的识别准确率明显低于其他睡眠阶段。这种情况通常有以下原因：

1. **类别不平衡**：REM阶段在数据集中的占比较低
2. **特征相似性**：REM阶段与某些NREM阶段(特别是N1)的特征相似
3. **模型倾向性**：模型倾向于预测更常见的类别

## 解决方案：加权损失函数

我们提供了两种方法来解决这个问题：

### 方法一：修改LLaMA-Factory框架（推荐）

这种方法通过修改LLaMA-Factory框架，为其添加类别权重支持。

#### 步骤1：准备补丁脚本

```bash
# 确保所有脚本具有执行权限
chmod +x lhc/projects/fine/modify_loss.py
```

#### 步骤2：应用补丁到LLaMA-Factory

```bash
# 应用补丁
python lhc/projects/fine/modify_loss.py --apply_patch
```

这将修改LLaMA-Factory的损失函数计算代码，添加对类别权重的支持。

#### 步骤3：使用加权损失函数进行训练

```bash
# 使用加权损失函数训练模型
python lhc/projects/fine/train_old.py \
  --model_name /data/lhc/models/Llama-3.2-1B-Instruct \
  --train_dataset edf197_100hz_10000ms_tok8521_train \
  --test_dataset edf197_100hz_10000ms_tok8521_test \
  --use_weighted_loss True \
  --class_weights 1.0,1.0,1.0,1.0,1.0,3.0 \
  --num_epochs 3
```

这里我们给REM阶段赋予了3倍的权重。

### 方法二：使用自定义损失函数脚本

如果无法直接修改LLaMA-Factory框架，可以使用我们提供的自定义损失函数。

```bash
# 将自定义损失函数脚本复制到LLaMA-Factory目录
cp lhc/projects/fine/weighted_loss.py /data/lhc/projects/LLaMA-Factory/src/llamafactory/train/
```

然后需要手动修改训练脚本，导入并使用这个自定义损失函数。

## 推荐的权重设置

根据REM识别问题的严重程度，我们推荐以下几种权重配置：

1. **轻度调整**：`1.0,1.0,1.0,1.0,1.0,2.0`
   - 适用于REM阶段识别率稍低的情况
   - 将REM阶段权重设为其他阶段的2倍

2. **中度调整**（推荐）：`1.0,1.0,1.0,1.0,1.0,3.0`
   - 适用于大多数情况
   - 将REM阶段权重设为其他阶段的3倍

3. **强力调整**：`1.0,1.0,1.0,1.0,1.0,5.0`
   - 适用于REM阶段识别率极低的情况
   - 将REM阶段权重设为其他阶段的5倍

4. **多阶段调整**：`1.0,2.0,1.0,1.0,1.0,3.0`
   - 如果N1和REM阶段都有识别问题
   - 同时提高N1和REM阶段的权重

## 评估与调优

训练完成后，使用以下步骤评估模型性能：

1. 检查混淆矩阵，特别关注REM行和列
2. 查看REM阶段的精确率、召回率和F1分数
3. 根据评估结果调整权重设置
4. 如果REM识别率仍然不理想，可以考虑增加模型参数或使用数据增强

## 恢复原始代码（如需）

如果需要恢复LLaMA-Factory的原始代码：

```bash
# 还原LLaMA-Factory代码
python lhc/projects/fine/modify_loss.py --revert_patch
```

## 故障排除

如果遇到以下问题：

1. **权重未生效**：检查补丁是否成功应用，并确认参数名称正确
2. **训练崩溃**：尝试减小权重差异，例如使用`1.0,1.0,1.0,1.0,1.0,2.0`
3. **过拟合**：适当减小REM阶段权重，或增加正则化

## 补充资源

- 查看`lhc/projects/fine/README_weighted_training.md`获取更多信息
- 阅读`lhc/projects/fine/weighted_loss.py`了解损失函数实现细节 