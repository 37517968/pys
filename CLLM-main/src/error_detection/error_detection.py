# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath('../cllm'))
sys.path.append(os.path.abspath('.'))

from dataset import Dataset
from cluster import *
from ed2_error_detection import ED2ErrorDetector, prepare_representative_samples_with_clean_data
from llm_aug import create_enhanced_representative_samples
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# api_details = {
#      "api_base": "https://sg.uiuiapi.com/v1",
#      "api_version": "uiui",
#      "api_key": "sk-pJXuMXNJJ0jin4umqzadRm1rADF1i7aRrxpd3GTvmsUDbUEw",
# }
api_details = {
     "api_base": "https://api.chatanywhere.tech",
     "api_key": "sk-L6xp4wATcmFSnJxCbsUcS0mIzN8AweFtxkKaKZ9VTwYLVe0q",
}


model_short_name = 'gpt-3.5-turbo' #gpt-3.5-turbo gpt-4.1-nano 'gpt-4' (do not use other short names)
model = "gpt-3.5-turbo" #gpt-3.5-turbo gpt-4.1-nano "gpt4_20230815" (use name of your model deployment)
llm_serving='together' # supported 'azure_openai', 'together', 'vllm'

seeds = [0,1,2,3,4,5,6,7,8,9]

n_samples = [10]

# 定义数据集配置
dataset_name = "hospital"
dataset_dictionary = {
    "name": dataset_name,
    "path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "dirty.csv")),
    "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "clean.csv"))
}

# 创建数据集实例
dataset = Dataset(dataset_dictionary)

# 可用的数据集列表
datasets = [dataset_name]

# 获取代表性样本（返回原始数据）
representative_samples = get_representative_samples(dataset.dataframe, n_clusters=10, similarity_metric='euclidean', return_original_data=True)
output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "representative_samples", "euclidean.csv")
Dataset.write_csv_dataset(output_path, representative_samples)
print("已保存使用 {} 方法得到的代表性样本到: {}".format('euclidean', output_path))
# 使用多种相似性度量方法进行实验（返回原始数据）
# comparison_results = compare_similarity_metrics(dataset.dataframe, n_clusters=10, return_original_data=True)

# 保存每种方法的结果
# for metric, samples_df in comparison_results.items():
#     output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "representative_samples", "{}.csv".format(metric))
#     Dataset.write_csv_dataset(output_path, samples_df)
#     print("已保存使用 {} 方法得到的代表性样本到: {}".format(metric, output_path))

# 使用ED2错误检测方法进行训练和检测

# 定义代表性样本路径
sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "representative_samples"))

# 准备代表性样本及其对应的干净数据
representative_samples_dataset = prepare_representative_samples_with_clean_data(dataset, sample_path, metric='euclidean')

# 使用LLM进行数据增强
print("\n开始使用LLM进行数据增强...")
enhanced_representative_samples_dataset = create_enhanced_representative_samples(
    representative_samples_dataset=representative_samples_dataset,
    n_samples=1000,  # 生成1000条增强样本
    api_details=api_details,
    llm_serving=llm_serving,
    model=model,
)
print("LLM数据增强完成")

# # 使用增强后的代表性样本进行训练
# try:
    
#     # 创建错误检测器并训练
#     detector = ED2ErrorDetector()
#     model = detector.train(enhanced_representative_samples_dataset.dataframe, enhanced_representative_samples_dataset.clean_dataframe)
#     print("ED2模型训练完成")
    
#     # 保存训练好的模型
#     model_save_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "error_detection", dataset_name, "ed2_model.pkl")
#     detector.save_model(model_save_path)
#     print(f"模型已保存到: {model_save_path}")
    
#     # 使用训练好的模型进行错误检测
#     print("\n使用训练好的模型进行错误检测...")
#     detection_results = detector.detect_errors(dataset.dataframe)
    
#     # 评估错误检测性能
#     true_labels = (dataset.dataframe.values != dataset.clean_dataframe.values).flatten()
#     predictions = detection_results['predictions']
#     probabilities = detection_results['probabilities']
    
#     # 打印基本统计信息
#     print(f"\n真实标签中错误单元格数量: {np.sum(true_labels)}")
#     print(f"预测为错误的单元格数量: {np.sum(predictions)}")
#     print(f"总单元格数量: {len(true_labels)}")
#     print(f"错误比例 (真实): {np.sum(true_labels) / len(true_labels) * 100:.2f}%")
#     print(f"错误比例 (预测): {np.sum(predictions) / len(predictions) * 100:.2f}%")
    
#     # 打印概率分布
#     print(f"\n预测概率统计:")
#     print(f"最小概率: {np.min(probabilities):.4f}")
#     print(f"最大概率: {np.max(probabilities):.4f}")
#     print(f"平均概率: {np.mean(probabilities):.4f}")
#     print(f"中位数概率: {np.median(probabilities):.4f}")
#     print(f"概率 > 0.5 的数量: {np.sum(probabilities > 0.5)}")
#     print(f"概率 > 0.1 的数量: {np.sum(probabilities > 0.1)}")
#     print(f"概率 > 0.01 的数量: {np.sum(probabilities > 0.01)}")
    
#     # 尝试不同的阈值
#     print("\n不同阈值下的性能:")
#     for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
#         pred_threshold = probabilities > threshold
#         if np.sum(pred_threshold) > 0:  # 确保至少有一个正预测
#             f1 = f1_score(true_labels, pred_threshold)
#             precision = precision_score(true_labels, pred_threshold)
#             recall = recall_score(true_labels, pred_threshold)
#             print(f"阈值 {threshold}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, 预测错误数={np.sum(pred_threshold)}")
    
#     # 使用默认阈值计算指标
#     f1 = f1_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions)
#     recall = recall_score(true_labels, predictions)
    
#     print("\n默认阈值(0.5)下的错误检测性能评估:")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
    
#     # 打印每列的错误检测结果
#     print("\n各列错误检测结果:")
#     for col_name, col_pred in detection_results['per_column_predictions'].items():
#         error_count = np.sum(col_pred)
#         print(f"{col_name}: {error_count} 个错误")
    
#     # 打印一些具体的错误示例
#     print("\n具体错误示例 (前10个):")
#     error_indices = np.where(predictions == True)[0][:10]
    
#     if len(error_indices) == 0:
#         print("没有预测到任何错误，无法显示错误示例")
#     else:
#         for i, idx in enumerate(error_indices):
#             row_idx = idx % dataset.dataframe.shape[0]
#             col_idx = idx // dataset.dataframe.shape[0]
#             col_name = dataset.dataframe.columns[col_idx]
#             true_value = dataset.dataframe.iloc[row_idx, col_idx]
#             clean_value = dataset.clean_dataframe.iloc[row_idx, col_idx]
#             prob = probabilities[idx]
#             print(f"{i+1}. 行 {row_idx}, 列 {col_name} ({col_idx}):")
#             print(f"   脏数据值: '{true_value}'")
#             print(f"   清洁数据值: '{clean_value}'")
#             print(f"   错误概率: {prob:.4f}")
#             print(f"   是否真实错误: {true_value != clean_value}")
    
# except Exception as e:
#     print(f"ED2错误检测过程中出现错误: {str(e)}")