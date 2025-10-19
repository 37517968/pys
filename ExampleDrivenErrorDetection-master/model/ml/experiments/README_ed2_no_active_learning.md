# ED2无主动清洗错误检测脚本使用说明

## 概述

这个脚本使用ED2（Example-Driven Error Detection）算法进行错误检测，完全移除了主动清洗部分，直接使用提供的正负样本训练数据进行训练，然后对待检测的测试数据进行错误检测，并返回F1分数、精确率和召回率。

## 文件说明

- `ed2_no_active_learning.py`: 主脚本文件，包含ED2错误检测的主要逻辑（无主动清洗）
- `README_ed2_no_active_learning.md`: 本说明文档

## 使用方法

### 基本用法

```bash
python ed2_no_active_learning.py
```

### 配置方法

在使用脚本前，需要修改脚本中的文件路径：

```python
# 在这里设置文件路径
train_dirty_data_path = "/path/to/train_dirty_data.csv"  # 训练脏数据文件路径
train_clean_data_path = "/path/to/train_clean_data.csv"  # 训练干净数据文件路径
test_dirty_data_path = "/path/to/test_dirty_data.csv"    # 测试脏数据文件路径
test_clean_data_path = "/path/to/test_clean_data.csv"    # 测试干净数据文件路径
output_dir = "/path/to/output"                          # 输出目录
model_path = os.path.join(output_dir, "ed2_model.pkl")   # 模型保存路径
```

将上述路径修改为您实际的文件路径，然后运行脚本即可。

## 数据格式要求

### 训练/测试数据格式

脏数据和干净数据文件都应为CSV格式，包含表头，且具有相同的结构。例如：

**训练脏数据 (train_dirty_data.csv):**
```csv
id,name,age,city
1,John,25,New Yrok
2,Jane,30,Los Angeles
3,Bob,35,Chicago
```

**训练干净数据 (train_clean_data.csv):**
```csv
id,name,age,city
1,John,25,New York
2,Jane,30,Los Angeles
3,Bob,35,Chicago
```

**测试脏数据 (test_dirty_data.csv):**
```csv
id,name,age,city
4,Alice,28,Boston
5,Charlie,32,Seattle
6,Diana,27,Miami
```

**测试干净数据 (test_clean_data.csv):**
```csv
id,name,age,city
4,Alice,28,Boston
5,Charlie,32,Seattle
6,Diana,27,Miami
```

脚本会自动比较脏数据和干净数据，生成错误标签。如果脏数据中的某个单元格与干净数据中的对应单元格不同，则标记为错误（1），否则标记为正确（0）。

## 输出结果

脚本会在指定的输出目录中生成以下文件：

1. `overall_metrics.txt`: 整体评估指标，包括：
   - F1 Score
   - Precision
   - Recall
   - Training Time
   - Total Train Samples
   - Total Test Samples
   - True Negatives
   - False Positives
   - False Negatives
   - True Positives

2. `per_column_metrics.csv`: 按列的评估指标，包括每列的：
   - F1 Score
   - Precision
   - Recall
   - Error Count
   - Error Rate

3. `predictions.npy`: 预测结果（NumPy数组格式）

4. `ground_truth.npy`: 真实标签（NumPy数组格式）

## 技术特点

1. **完全移除主动清洗**：脚本不使用任何主动清洗策略，直接使用提供的训练数据进行训练。
2. **使用ED2特征提取**：保留了ED2的特征提取方法，包括基础特征、元数据特征和Word2Vec特征。
3. **直接训练和预测**：简化了训练流程，直接使用所有标记数据进行训练，然后进行预测。
4. **全面的评估指标**：提供整体和按列的评估指标，包括F1分数、精确率和召回率。

## 注意事项

1. 确保脏数据文件和干净数据文件的行数和列数一致。
2. 脚本会自动比较脏数据和干净数据，生成错误标签，无需单独提供标签文件。
3. 脚本会自动处理缺失值，将它们填充为空字符串。
4. 脚本使用ED2原有的特征提取方法，但完全移除了主动清洗部分。
5. 脚本需要访问ED2库，请确保ED2库路径正确。
6. 在运行脚本前，请确保已修改脚本中的文件路径为您实际的文件路径。

## 依赖项

- Python 2.7（ED2原始代码使用Python 2.7）
- NumPy
- Pandas
- scikit-learn
- XGBoost
- SciPy

## 故障排除

如果遇到以下错误：

1. "错误：无法导入ED2库。"
   - 确保ED2库路径正确
   - 检查是否安装了所有依赖项

2. "错误：文件不存在:"
   - 检查提供的文件路径是否正确
   - 确保文件存在

3. "Warning: No features generated, creating default feature matrix"
   - 这通常是由于数据格式问题导致的，脚本会自动创建默认特征矩阵
   - 检查数据文件是否为空或格式是否正确

## 与原版ED2的区别

1. **移除主动清洗**：原版ED2使用主动学习策略逐步选择样本进行标记，本脚本直接使用所有提供的标记样本。
2. **简化训练流程**：原版ED2的训练过程涉及多轮迭代和列选择，本脚本直接进行一次性训练。
3. **直接评估**：原版ED2在训练过程中进行评估，本脚本在训练完成后直接在测试集上进行评估。
4. **保留特征提取**：保留了ED2的特征提取方法，包括基础特征、元数据特征和Word2Vec特征。

## 性能提示

1. 对于大型数据集，训练可能需要较长时间。
2. 可以通过减少`w2v_size`来加快训练速度，但可能会影响性能。
3. 如果内存不足，可以尝试减少数据集的大小。
4. 脚本会自动将数据分割为训练集和测试集（如果没有提供单独的测试集），默认比例为80%训练，20%测试。