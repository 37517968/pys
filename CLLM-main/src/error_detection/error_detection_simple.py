# -*- coding: utf-8 -*-
import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import Dataset
import traceback

def create_dataset(name, dirty_path, clean_path):
    """
    创建训练数据集，使用Dataset.py的写法
    从data/hospital/dirty.csv和clean.csv读取数据
    """
    # 获取当前文件所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建数据文件的相对路径
    dirty_path = os.path.join(current_dir, f"../../data/error_detection/{name}/{dirty_path}")
    clean_path = os.path.join(current_dir, f"../../data/error_detection/{name}/{clean_path}")

    # 创建数据集字典
    dataset_dict = {
        "name": name,
        "path": dirty_path,
        "clean_path": clean_path
    }
    
    # 使用Dataset类创建训练数据集
    dataset = Dataset(dataset_dict)
    
    return dataset

def main():
    """
    主函数，演示如何使用train_dataset
    """
    # 创建训练数据集
    dataset = create_dataset("hospital", "dirty copy.csv", "clean copy.csv")
    train_dataset = create_dataset("hospital", "dirty.csv", "clean.csv")
    ignore_columns = ["index", "Address2", "Address3"]
    
    model_save_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset.name, "ed2_model.pkl")
    try:
        from ed2_error_detection import ED2ErrorDetector
        
        # 创建错误检测器并训练
        detector = ED2ErrorDetector(ignore_columns=ignore_columns)
        # 使用交叉验证训练模型
        detector.train(train_dataset.dataframe, train_dataset.clean_dataframe, use_cv=True, cv_folds=5)
        print("ED2模型训练完成")
        
        # 保存训练好的模型
        detector.save_model(model_save_path)
        print(f"模型已保存到: {model_save_path}")
        detector = detector.load_model(model_save_path)
        print("模型加载成功")   
        # 使用训练好的模型进行错误检测
        print("\n使用训练好的模型进行错误检测...")
        detection_results = detector.detect_errors(dataset.dataframe)
        
        # 评估错误检测性能
        # 使用与Dataset类相同的处理方式，确保一致性
        import re
        try:
            # Python 3.x
            import html
            html_unescape = html.unescape
        except ImportError:
            # Python 2.x
            import HTMLParser
            html_unescape = HTMLParser.HTMLParser().unescape
        
        def value_normalizer(value):
            """与Dataset类相同的值规范化方法"""
            value = html_unescape(value)
            value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
            value = value.strip("\t\n ")
            return value
        
        def get_true_labels(dirty_df, clean_df, ignore_columns=None):
            dirty_df = dirty_df.drop(columns=ignore_columns, errors='ignore')
            clean_df = clean_df.drop(columns=ignore_columns, errors='ignore')
            labels = np.zeros(dirty_df.shape, dtype=bool)
            for i in range(dirty_df.shape[0]):
                for j in range(dirty_df.shape[1]):
                    dirty_value = value_normalizer(str(dirty_df.iloc[i, j]))
                    clean_value = value_normalizer(str(clean_df.iloc[i, j]))
                    labels[i, j] = dirty_value != clean_value
            return labels
        # 创建真实标签：比较脏数据和清洁数据
        true_labels_train = get_true_labels(train_dataset.dataframe, train_dataset.clean_dataframe, ignore_columns=ignore_columns)
        true_labels_test = get_true_labels(dataset.dataframe, dataset.clean_dataframe, ignore_columns=ignore_columns)
        
        
        true_labels = true_labels_test.flatten()
        predictions = detection_results['predictions']
        probabilities = detection_results['probabilities']
        
        # 打印基本统计信息
        print(f"\n真实标签中错误单元格数量: {np.sum(true_labels)}")
        print(f"预测为错误的单元格数量: {np.sum(predictions)}")
        print(f"总单元格数量: {len(true_labels)}")
        print(f"错误比例 (真实): {np.sum(true_labels) / len(true_labels) * 100:.2f}%")
        print(f"错误比例 (预测): {np.sum(predictions) / len(predictions) * 100:.2f}%")
        
        # 打印概率分布
        print(f"\n预测概率统计:")
        print(f"最小概率: {np.min(probabilities):.4f}")
        print(f"最大概率: {np.max(probabilities):.4f}")
        print(f"平均概率: {np.mean(probabilities):.4f}")
        print(f"中位数概率: {np.median(probabilities):.4f}")
        print(f"概率 > 0.5 的数量: {np.sum(probabilities > 0.5)}")
        print(f"概率 > 0.1 的数量: {np.sum(probabilities > 0.1)}")
        print(f"概率 > 0.01 的数量: {np.sum(probabilities > 0.01)}")
        
        # 尝试不同的阈值
        print("\n不同阈值下的性能:")
        for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_threshold = probabilities > threshold
            if np.sum(pred_threshold) > 0:  # 确保至少有一个正预测
                # 处理可能的除零情况
                try:
                    f1 = f1_score(true_labels, pred_threshold, zero_division=0)
                    precision = precision_score(true_labels, pred_threshold, zero_division=0)
                    recall = recall_score(true_labels, pred_threshold, zero_division=0)
                    print(f"阈值 {threshold}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, 预测错误数={np.sum(pred_threshold)}")
                except ValueError as e:
                    print(f"阈值 {threshold}: 无法计算指标 - {str(e)}, 预测错误数={np.sum(pred_threshold)}")
        
        # 使用默认阈值计算指标
        try:
            f1 = f1_score(true_labels, predictions, zero_division=0)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            
            print("\n默认阈值(0.5)下的错误检测性能评估:")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
        except ValueError as e:
            print(f"\n默认阈值(0.5)下的错误检测性能评估:")
            print(f"无法计算指标 - {str(e)}")
        
        # ==========================
        # 各列错误统计对比
        # ==========================
        print("\n各列错误统计对比:")
        print("{:<30s} {:>10s} {:>10s} {:>10s}".format("列名", "训练错误", "真实错误", "预测错误"))
        
        # 获取未被忽略的列名列表
        valid_columns = [col for col in dataset.dataframe.columns if col not in ignore_columns]
        
        for col_idx, col_name in enumerate(valid_columns):
            # 使用在有效列中的索引，而不是原始数据框中的索引
            train_error = np.sum(true_labels_train[..., col_idx])
            test_error = np.sum(true_labels_test[..., col_idx])
            pred_error = np.sum(detection_results['per_column_predictions'][col_name])
            print("{:<30s} {:>10d} {:>10d} {:>10d}".format(col_name, train_error, test_error, pred_error))
        
    
    except Exception as e:
        print(f"ED2错误检测过程中出现错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    train_dataset = main()