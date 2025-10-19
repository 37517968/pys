# 简化版ED2 (Example-Driven Error Detection)

## 概述

简化版ED2是基于原始ED2系统的改进版本，移除了主动学习模块，改为使用提供的训练数据进行特征提取和模型训练，然后对待检测数据进行错误检测。

## 主要改进

1. **移除主动学习模块**：不再需要迭代式地选择样本进行标注，直接使用提供的训练数据。
2. **简化训练流程**：一次性训练模型，无需多轮迭代。
3. **保留特征提取功能**：保留了原始ED2中的所有特征提取方法，包括基础特征、元数据特征、Word2Vec特征、LSTM特征等。
4. **改进的分类器**：提供了简化的XGBoost分类器，支持更直接的训练和预测接口。

## 文件结构

```
ExampleDrivenErrorDetection-master/
├── model/
│   ├── ml/
│   │   ├── classes/
│   │   │   ├── simplified_error_detection_lib.py  # 主要的简化版错误检测库
│   │   │   └── machine_learning_detection.py      # 机器学习检测相关功能
│   │   ├── active_learning/
│   │   │   └── classifier/
│   │   │       ├── SimplifiedXGBoostClassifier.py  # 简化的XGBoost分类器
│   │   │       └── XGBoostClassifier.py           # 原始XGBoost分类器
│   │   ├── experiments/
│   │   │   ├── simplified_features_experiment.py   # 简化版实验脚本
│   │   │   └── features_experiment_multi.py       # 原始实验脚本
│   │   ├── examples/
│   │   │   └── simple_error_detection_example.py  # 使用示例
│   │   └── features/                              # 特征提取模块
│   │       ├── ActiveCleanFeatures.py
│   │       ├── ValueCorrelationFeatures.py
│   │       └── BoostCleanMetaFeatures.py
│   └── datasets/                                   # 数据集模块
└── README_simplified_ED2.md                       # 本文档
```

## 主要组件

### 1. 简化版错误检测库 (`simplified_error_detection_lib.py`)

这是核心库，提供了以下主要功能：

- `prepare_data_for_training()`: 准备训练数据
- `train_and_evaluate_error_detection_model()`: 训练和评估错误检测模型
- `run_error_detection_experiment()`: 运行错误检测实验
- `detect_errors_in_new_data()`: 在新数据上进行错误检测

### 2. 简化的XGBoost分类器 (`SimplifiedXGBoostClassifier.py`)

提供了简化的XGBoost分类器，主要功能包括：

- `train()`: 训练模型
- `predict()`: 预测类别
- `predict_proba()`: 预测概率
- `evaluate()`: 评估模型性能
- `save_model()` / `load_model()`: 保存/加载模型
- `get_feature_importance()`: 获取特征重要性

### 3. 简化版实验脚本 (`simplified_features_experiment.py`)

用于运行不同特征组合和分类器的实验，支持：

- 多进程并行运行实验
- 自动保存实验结果
- 结果分析和汇总

### 4. 使用示例 (`simple_error_detection_example.py`)

提供了三个使用示例：

1. 训练和评估模型
2. 在新数据上进行错误检测
3. 查看特征重要性

## 使用方法

### 1. 基本使用

```python
from ml.classes.simplified_error_detection_lib import train_and_evaluate_error_detection_model
from ml.datasets.adult.Adult import Adult
from ml.active_learning.classifier.SimplifiedXGBoostClassifier import SimplifiedXGBoostClassifier

# 加载数据集
train_data = Adult()
test_data = Adult()

# 训练和评估模型
result = train_and_evaluate_error_detection_model(
    train_dataSet=train_data,
    test_dataSet=test_data,
    classifier_model=SimplifiedXGBoostClassifier,
    use_metadata=True,
    use_word2vec=True,
    w2v_size=100,
    model_save_path="models/error_detection_model.pkl"
)

# 打印结果
print(f"F1 Score: {result['overall']['f1']:.4f}")
print(f"Precision: {result['overall']['precision']:.4f}")
print(f"Recall: {result['overall']['recall']:.4f}")
```

### 2. 在新数据上进行错误检测

```python
from ml.classes.simplified_error_detection_lib import detect_errors_in_new_data
from ml.datasets.adult.Adult import Adult

# 加载新数据
new_data = Adult()

# 使用已训练的模型进行错误检测
detection_result = detect_errors_in_new_data(
    model_path="models/error_detection_model.pkl",
    new_dataSet=new_data
)

# 打印结果
print(f"错误率: {detection_result['error_rate']:.4f}")
print(f"总错误数: {detection_result['total_errors']}")
```

### 3. 运行实验

```python
from ml.experiments.simplified_features_experiment import main

# 运行所有实验
results = main()
```

### 4. 查看特征重要性

```python
# 训练模型后
feature_importance = result['feature_importance']

# 打印前10个最重要的特征
for i, (feature_name, importance) in enumerate(feature_importance[:10]):
    print(f"{i+1}. {feature_name}: {importance:.4f}")
```

## 特征选项

简化版ED2支持以下特征类型：

- **基础特征**：
  - `ngrams`: N-gram特征的大小
  - `is_word`: 是否使用词级别的N-gram
  - `use_tf_idf`: 是否使用TF-IDF

- **元数据特征**：
  - `use_metadata`: 是否使用元数据特征
  - `use_metadata_only`: 是否仅使用元数据特征

- **Word2Vec特征**：
  - `use_word2vec`: 是否使用Word2Vec特征
  - `use_word2vec_only`: 是否仅使用Word2Vec特征
  - `w2v_size`: Word2Vec向量大小

- **LSTM特征**：
  - `use_lstm`: 是否使用LSTM特征
  - `use_lstm_only`: 是否仅使用LSTM特征

- **ActiveClean特征**：
  - `use_active_clean`: 是否使用ActiveClean特征
  - `use_activeclean_only`: 是否仅使用ActiveClean特征

- **条件概率特征**：
  - `use_cond_prob`: 是否使用条件概率特征
  - `use_cond_prob_only`: 是否仅使用条件概率特征

- **BoostClean元数据特征**：
  - `use_boostclean_metadata`: 是否使用BoostClean元数据特征
  - `use_boostclean_metadata_only`: 是否仅使用BoostClean元数据特征

## 与原始ED2的对比

| 特性 | 原始ED2 | 简化版ED2 |
|------|---------|-----------|
| 主动学习 | 支持 | 不支持 |
| 训练方式 | 迭代式，逐步添加标注数据 | 一次性使用所有训练数据 |
| 列选择 | 智能选择下一列处理 | 处理所有列 |
| 用户交互 | 需要用户参与标注 | 无需用户参与 |
| 训练时间 | 较长（多轮迭代） | 较短（一次性训练） |
| 特征提取 | 支持所有特征类型 | 支持所有特征类型 |
| 模型性能 | 相同特征下性能相近 | 相同特征下性能相近 |

## 适用场景

简化版ED2适用于以下场景：

1. **有标注数据**：已经有足够的标注数据用于训练
2. **批量处理**：需要一次性处理大量数据
3. **自动化流程**：需要集成到自动化数据处理流程中
4. **快速部署**：需要快速部署和使用错误检测系统

## 注意事项

1. **数据格式**：训练数据和待检测数据的列结构必须相同
2. **模型保存**：训练好的模型可以保存并在以后使用
3. **特征一致性**：训练和检测时使用的特征类型必须一致
4. **内存使用**：大型数据集可能需要较多的内存

## 示例输出

```
Preparing training data...
Train feature matrix shape: (32561, 500)
Test feature matrix shape: (32561, 500)
Train ground truth array length: 32561
Test ground truth array length: 32561
Training model...
Training completed in 45.32 seconds
Making predictions on test set...

Overall Results:
F1 Score: 0.8524
Precision: 0.8235
Recall: 0.8832
Training Time: 45.32 seconds
Total Train Samples: 32561
Total Test Samples: 32561

Per-Column Results:
age: F1=0.7823, Precision=0.7512, Recall=0.8165
workclass: F1=0.8234, Precision=0.8011, Recall=0.8472
education: F1=0.8756, Precision=0.8523, Recall=0.9001
...
```

## 联系方式

如有问题或建议，请联系开发团队。