"""
简化的ED2错误检测示例

这个示例展示了如何使用简化版的ED2进行错误检测：
1. 使用训练数据训练模型
2. 使用训练好的模型对待检测数据进行错误检测
3. 输出检测结果和各项指标
"""

import os
import sys
import numpy as np
import pandas as pd
from ml.classes.simplified_error_detection_lib import (
    train_and_evaluate_error_detection_model,
    detect_errors_in_new_data
)
from ml.datasets.adult.Adult import Adult
from ml.active_learning.classifier.SimplifiedXGBoostClassifier import SimplifiedXGBoostClassifier
from ml.configuration.Config import Config

def create_sample_dataset(data, train_ratio=0.8):
    """
    创建示例数据集，将原始数据集分割为训练集和测试集
    
    参数:
    - data: 原始数据集对象
    - train_ratio: 训练集比例
    
    返回:
    - train_data: 训练数据集
    - test_data: 测试数据集
    """
    from sklearn.model_selection import train_test_split
    
    # 获取数据
    X = data.dirty_pd.values
    y = data.matrix_is_error
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    
    # 创建新的数据集对象
    train_data = Adult()
    test_data = Adult()
    
    # 更新数据
    train_data.dirty_pd = pd.DataFrame(X_train, columns=data.clean_pd.columns)
    train_data.matrix_is_error = y_train
    
    test_data.dirty_pd = pd.DataFrame(X_test, columns=data.clean_pd.columns)
    test_data.matrix_is_error = y_test
    
    return train_data, test_data

def example_training_and_evaluation():
    """
    示例：训练和评估模型
    """
    print("=" * 60)
    print("示例1: 训练和评估模型")
    print("=" * 60)
    
    # 加载数据集
    print("加载数据集...")
    data = Adult()
    
    # 创建训练集和测试集
    print("创建训练集和测试集...")
    train_data, test_data = create_sample_dataset(data, train_ratio=0.8)
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 训练和评估模型
    print("训练和评估模型...")
    result = train_and_evaluate_error_detection_model(
        train_dataSet=train_data,
        test_dataSet=test_data,
        classifier_model=SimplifiedXGBoostClassifier,
        use_metadata=True,
        use_word2vec=True,
        w2v_size=50,  # 减小向量大小以加快训练速度
        use_cv=False,  # 关闭交叉验证以加快训练速度
        model_save_path="models/error_detection_model.pkl"
    )
    
    # 打印结果
    print("\n训练结果:")
    print(f"F1 Score: {result['overall']['f1']:.4f}")
    print(f"Precision: {result['overall']['precision']:.4f}")
    print(f"Recall: {result['overall']['recall']:.4f}")
    print(f"Training Time: {result['overall']['training_time']:.2f} seconds")
    
    print("\n按列结果:")
    for col_name, metrics in result['per_column'].items():
        print(f"{col_name}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return result

def example_error_detection_on_new_data():
    """
    示例：在新数据上进行错误检测
    """
    print("\n" + "=" * 60)
    print("示例2: 在新数据上进行错误检测")
    print("=" * 60)
    
    # 加载数据集
    print("加载数据集...")
    data = Adult()
    
    # 创建训练集和测试集
    print("创建训练集和测试集...")
    train_data, test_data = create_sample_dataset(data, train_ratio=0.8)
    
    # 训练模型
    print("训练模型...")
    train_result = train_and_evaluate_error_detection_model(
        train_dataSet=train_data,
        classifier_model=SimplifiedXGBoostClassifier,
        use_metadata=True,
        use_word2vec=True,
        w2v_size=50,
        use_cv=False,
        model_save_path="models/error_detection_model.pkl"
    )
    
    # 在新数据上进行错误检测
    print("在新数据上进行错误检测...")
    detection_result = detect_errors_in_new_data(
        model_path="models/error_detection_model.pkl",
        new_dataSet=test_data
    )
    
    # 打印结果
    print("\n错误检测结果:")
    print(f"错误率: {detection_result['error_rate']:.4f}")
    print(f"总错误数: {detection_result['total_errors']}")
    print(f"总样本数: {detection_result['total_samples']}")
    
    print("\n按列错误检测结果:")
    for col_name, predictions in detection_result['per_column_predictions'].items():
        error_count = np.sum(predictions)
        error_rate = np.mean(predictions)
        print(f"{col_name}: 错误数={error_count}, 错误率={error_rate:.4f}")
    
    return detection_result

def example_feature_importance():
    """
    示例：查看特征重要性
    """
    print("\n" + "=" * 60)
    print("示例3: 查看特征重要性")
    print("=" * 60)
    
    # 加载数据集
    print("加载数据集...")
    data = Adult()
    
    # 创建训练集和测试集
    print("创建训练集和测试集...")
    train_data, test_data = create_sample_dataset(data, train_ratio=0.8)
    
    # 训练模型
    print("训练模型...")
    result = train_and_evaluate_error_detection_model(
        train_dataSet=train_data,
        test_dataSet=test_data,
        classifier_model=SimplifiedXGBoostClassifier,
        use_metadata=True,
        use_word2vec=True,
        w2v_size=50,
        use_cv=False
    )
    
    # 获取特征重要性
    print("\n特征重要性 (Top 10):")
    feature_importance = result['feature_importance']
    for i, (feature_name, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature_name}: {importance:.4f}")
    
    return result

def main():
    """
    主函数
    """
    # 确保模型目录存在
    if not os.path.exists("models"):
        os.makedirs("models")
    
    print("简化版ED2错误检测示例")
    print("=" * 60)
    
    try:
        # 示例1: 训练和评估模型
        result1 = example_training_and_evaluation()
        
        # 示例2: 在新数据上进行错误检测
        result2 = example_error_detection_on_new_data()
        
        # 示例3: 查看特征重要性
        result3 = example_feature_importance()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()