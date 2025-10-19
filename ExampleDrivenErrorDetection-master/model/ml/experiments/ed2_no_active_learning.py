#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ED2错误检测脚本（无主动清洗）

这个脚本使用ED2（Example-Driven Error Detection）算法进行错误检测，
完全移除了主动清洗部分，直接使用提供的正负样本训练数据进行训练，
然后对待检测的测试数据进行错误检测，并返回F1分数、精确率和召回率。

使用方法:
    python ed2_no_active_learning.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time

# 添加ED2库路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.datasets.DataSetBasic import DataSetBasic
    from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
    from ml.active_learning.library import create_features, add_metadata_features, split_data_indices
    from ml.Word2VecFeatures.Word2VecFeatures import Word2VecFeatures as OriginalWord2VecFeatures
    from ml.features.ActiveCleanFeatures import ActiveCleanFeatures
    from ml.features.ValueCorrelationFeatures import ValueCorrelationFeatures
    from ml.features.BoostCleanMetaFeatures import BoostCleanMetaFeatures
    from scipy.sparse import vstack, hstack, csr_matrix
    from gensim.models import Word2Vec
except ImportError as e:
    print("错误：无法导入ED2库。")
    print("错误信息：", str(e))
    sys.exit(1)


class FixedWord2VecFeatures(OriginalWord2VecFeatures):
    """
    修复了Word2Vec训练参数问题的Word2VecFeatures类
    """
    
    def fit(self, data):
        # create dictionary: val -> word
        self.column_dictionaries = []
        words = np.zeros((data.shape[0], data.shape[1]), dtype=object)
        for column_i in range(data.shape[1]):
            col_val2word = {}
            for row_i in range(data.shape[0]):
                val = data[row_i, column_i]
                if not val in col_val2word:
                    col_val2word[val] = 'col' + str(column_i) + "_" + str(len(col_val2word))
                words[row_i, column_i] = col_val2word[val]
            self.column_dictionaries.append(col_val2word)

        # train word2vec - 修复epochs参数问题
        self.model = Word2Vec(words.tolist(), size=self.vector_size, window=words.shape[1] * 2, min_count=1, workers=4, negative=0, hs=1)
        # 使用更简单的训练方法，不使用epochs参数
        self.model.train(words.tolist())


class CustomDataSet(DataSetBasic):
    """
    自定义数据集类，用于加载用户提供的训练数据和测试数据
    """
    
    def __init__(self, dirty_data_path, clean_data_path, name="CustomData"):
        """
        初始化数据集
        
        参数:
        - dirty_data_path: 脏数据文件路径
        - clean_data_path: 干净数据文件路径
        - name: 数据集名称
        """
        # 加载数据
        self.dirty_pd = pd.read_csv(dirty_data_path, header=0, dtype=object, na_filter=False)
        self.dirty_pd = self.fillna_df(self.dirty_pd)
        
        # 加载干净数据
        self.clean_pd = pd.read_csv(clean_data_path, header=0, dtype=object, na_filter=False)
        self.clean_pd = self.fillna_df(self.clean_pd)
        
        # 比较脏数据和干净数据，生成错误标签
        self.matrix_is_error = self.dirty_pd.values != self.clean_pd.values
        
        # 调用父类构造函数
        super(CustomDataSet, self).__init__(name, self.dirty_pd, self.matrix_is_error)


def create_features_matrix(dataSet, train_indices, test_indices,
                          use_metadata=True, use_word2vec=True, w2v_size=100):
    """
    创建特征矩阵，移除了主动清洗部分
    """
    # 基础特征
    all_matrix_train, all_matrix_test, feature_name_list = create_features(
        dataSet, train_indices, test_indices, ngrams=1, runSVD=False, is_word=False, use_tf_idf=True)

    # 元数据特征
    if use_metadata:
        all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(
            dataSet, train_indices, test_indices, all_matrix_train,
            all_matrix_test, feature_name_list, use_meta_only=False)

    # Word2Vec特征
    if use_word2vec:
        w2v_features = FixedWord2VecFeatures(vector_size=w2v_size)
        all_matrix_train, all_matrix_test, feature_name_list = w2v_features.add_word2vec_features(
            dataSet, train_indices, test_indices, all_matrix_train,
            all_matrix_test, feature_name_list, word2vec_only=False)

    # 检查是否所有特征矩阵都为空
    if all_matrix_train is None or (hasattr(all_matrix_train, 'shape') and all_matrix_train.shape[0] == 0):
        print "Warning: No features generated, creating default feature matrix"
        # 创建一个简单的默认特征矩阵，避免连接空数组的问题
        num_train = len(train_indices) if train_indices is not None else dataSet.shape[0]
        num_test = len(test_indices) if test_indices is not None else dataSet.shape[0]
        
        all_matrix_train = csr_matrix(np.ones((num_train, 1)))
        all_matrix_test = csr_matrix(np.ones((num_test, 1)))
        feature_name_list = ['default_feature']

    return all_matrix_train, all_matrix_test, feature_name_list


def prepare_data_for_training(train_dataSet, test_dataSet=None):
    """
    为训练准备数据
    """
    # 分割训练数据索引
    train_indices = np.arange(train_dataSet.shape[0])
    
    # 如果没有提供测试集，使用训练集的一部分作为测试集
    if test_dataSet is None:
        train_indices, test_indices = split_data_indices(train_dataSet, 0.8, fold_number=0)
        test_dataSet = train_dataSet
    else:
        test_indices = np.arange(test_dataSet.shape[0])
    
    # 为训练集创建特征
    all_matrix_train, _, feature_name_list = create_features_matrix(
        train_dataSet, train_indices, test_indices)
    
    try:
        feature_matrix_train = all_matrix_train.tocsr()
    except:
        feature_matrix_train = all_matrix_train

    # 为测试集创建特征
    if test_dataSet is not train_dataSet:
        # 为测试集创建特征时，使用测试集自己的索引
        test_train_indices = np.arange(test_dataSet.shape[0])
        test_test_indices = np.arange(test_dataSet.shape[0])
        all_matrix_test, _, _ = create_features_matrix(
            test_dataSet, test_train_indices, test_test_indices)
    else:
        # 如果测试集是训练集的一部分，使用相同的特征矩阵
        all_matrix_test = all_matrix_train
    
    try:
        feature_matrix_test = all_matrix_test.tocsr()
    except:
        feature_matrix_test = all_matrix_test

    # 为训练集每列创建特征矩阵并添加列标识
    feature_matrix_train_per_column = []
    
    for column_i in xrange(train_dataSet.shape[1]):
        # 添加列标识的one-hot编码
        one_hot_part = np.zeros((len(train_indices), train_dataSet.shape[1]))
        one_hot_part[:, column_i] = 1
        
        # 将列标识特征与原有特征合并
        if feature_matrix_train is None or (hasattr(feature_matrix_train, 'shape') and feature_matrix_train.shape[0] == 0):
            feature_matrix_train_per_column.append(csr_matrix(one_hot_part))
        else:
            feature_matrix_train_per_column.append(hstack((feature_matrix_train, one_hot_part)).tocsr())
    
    # 垂直堆叠训练集所有列的特征矩阵
    all_columns_feature_matrix_train = vstack(feature_matrix_train_per_column)
    
    # 为测试集每列创建特征矩阵并添加列标识
    feature_matrix_test_per_column = []
    
    for column_i in xrange(test_dataSet.shape[1]):
        # 添加列标识的one-hot编码
        one_hot_part = np.zeros((len(test_indices), test_dataSet.shape[1]))
        one_hot_part[:, column_i] = 1
        
        # 将列标识特征与原有特征合并
        if feature_matrix_test is None or (hasattr(feature_matrix_test, 'shape') and feature_matrix_test.shape[0] == 0):
            feature_matrix_test_per_column.append(csr_matrix(one_hot_part))
        else:
            feature_matrix_test_per_column.append(hstack((feature_matrix_test, one_hot_part)).tocsr())
    
    # 垂直堆叠测试集所有列的特征矩阵
    all_columns_feature_matrix_test = vstack(feature_matrix_test_per_column)
    
    # 创建训练集标签数组（将所有列的标签展平）
    ground_truth_train_array = train_dataSet.matrix_is_error[train_indices, 0]
    for column_i in xrange(1, train_dataSet.shape[1]):
        ground_truth_train_array = np.concatenate((ground_truth_train_array, 
                                                   train_dataSet.matrix_is_error[train_indices, column_i]))

    # 创建测试集标签数组（将所有列的标签展平）
    if test_dataSet is train_dataSet:
        ground_truth_test_array = train_dataSet.matrix_is_error[test_indices, 0]
        for column_i in xrange(1, train_dataSet.shape[1]):
            ground_truth_test_array = np.concatenate((ground_truth_test_array, 
                                                    train_dataSet.matrix_is_error[test_indices, column_i]))
    else:
        ground_truth_test_array = test_dataSet.matrix_is_error[test_indices, 0]
        for column_i in xrange(1, test_dataSet.shape[1]):
            ground_truth_test_array = np.concatenate((ground_truth_test_array, 
                                                    test_dataSet.matrix_is_error[test_indices, column_i]))

    return (all_columns_feature_matrix_train, all_columns_feature_matrix_test, 
            ground_truth_train_array, ground_truth_test_array, 
            feature_name_list, train_indices, test_indices)


def train_and_evaluate_model(train_dataSet, test_dataSet=None, model_save_path=None):
    """
    训练和评估模型，完全移除主动清洗部分
    """
    # 准备训练数据
    print "Preparing training data..."
    (all_columns_feature_matrix_train, all_columns_feature_matrix_test,
     ground_truth_train_array, ground_truth_test_array,
     feature_name_list, train_indices, test_indices) = prepare_data_for_training(
        train_dataSet, test_dataSet)
    
    print "Train feature matrix shape:", all_columns_feature_matrix_train.shape
    print "Test feature matrix shape:", all_columns_feature_matrix_test.shape
    print "Train ground truth array length:", len(ground_truth_train_array)
    print "Test ground truth array length:", len(ground_truth_test_array)
    
    # 确保训练和测试数据的特征数量一致
    if all_columns_feature_matrix_train.shape[1] != all_columns_feature_matrix_test.shape[1]:
        print "Warning: Feature count mismatch between train and test data. Adjusting test data..."
        from scipy.sparse import hstack, csr_matrix
        
        if all_columns_feature_matrix_test.shape[1] > all_columns_feature_matrix_train.shape[1]:
            # 截断测试数据的特征
            all_columns_feature_matrix_test = all_columns_feature_matrix_test[:, :all_columns_feature_matrix_train.shape[1]]
        else:
            # 填充测试数据的特征
            padding = csr_matrix((all_columns_feature_matrix_test.shape[0],
                                all_columns_feature_matrix_train.shape[1] - all_columns_feature_matrix_test.shape[1]))
            all_columns_feature_matrix_test = hstack((all_columns_feature_matrix_test, padding))
        
        print "Adjusted test feature matrix shape:", all_columns_feature_matrix_test.shape
    
    # 初始化分类器
    classifier = XGBoostClassifier(all_columns_feature_matrix_train, all_columns_feature_matrix_test)
    
    # 设置分类器参数
    classifier.params[0] = {
        'learning_rate': 0.1,
        'colsample_bytree': 0.8,
        'silent': 1,
        'seed': 0,
        'objective': 'binary:logistic',
        'n_jobs': 4,
        'min_child_weight': 1,
        'subsample': 0.8,
        'max_depth': 3
    }
    
    # 训练模型
    print "Training model..."
    start_time = time.time()
    
    # 使用训练集训练模型 - 使用train_predict_all方法
    # 注意：这里我们使用相同的训练集作为预测集，以避免特征名不匹配的问题
    _, _, _, class_prediction = classifier.train_predict_all(
        all_columns_feature_matrix_train, ground_truth_train_array, 0,
        all_columns_feature_matrix_train, all_columns_feature_matrix_train)
    
    training_time = time.time() - start_time
    print "Training completed in {0:.2f} seconds".format(training_time)
    
    # 保存模型
    if model_save_path:
        print "Saving model to", model_save_path
        try:
            os.makedirs(os.path.dirname(model_save_path))
        except OSError:
            pass  # 目录已存在，忽略错误
        
        # 只保存XGBoost模型，而不是整个分类器对象
        if 0 in classifier.model:
            xgb_model = classifier.model[0]
            with open(model_save_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            print "XGBoost model saved successfully"
        else:
            print "Warning: No model to save"
    
    # 在测试集上进行预测
    print "Making predictions on test set..."
    # 使用train_predict_all方法进行预测，确保特征名称一致
    _, _, _, class_prediction = classifier.train_predict_all(
        all_columns_feature_matrix_train, ground_truth_train_array, 0,
        all_columns_feature_matrix_train, all_columns_feature_matrix_test)
    
    # 计算评估指标
    f1 = f1_score(ground_truth_test_array, class_prediction)
    precision = precision_score(ground_truth_test_array, class_prediction)
    recall = recall_score(ground_truth_test_array, class_prediction)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(ground_truth_test_array, class_prediction).ravel()
    
    # 按列计算评估指标
    per_column_metrics = {}
    test_dataset = test_dataSet if test_dataSet is not None else train_dataSet
    
    for col_i in xrange(test_dataset.shape[1]):
        start_idx = col_i * len(test_indices)
        end_idx = (col_i + 1) * len(test_indices)
        
        col_true = ground_truth_test_array[start_idx:end_idx]
        col_pred = class_prediction[start_idx:end_idx]
        
        per_column_metrics[test_dataset.clean_pd.columns[col_i]] = {
            'f1': f1_score(col_true, col_pred),
            'precision': precision_score(col_true, col_pred),
            'recall': recall_score(col_true, col_pred),
            'error_count': np.sum(col_pred),
            'error_rate': np.mean(col_pred)
        }
    
    # 返回结果
    result = {
        'overall': {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'total_train_samples': len(ground_truth_train_array),
            'total_test_samples': len(ground_truth_test_array),
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        },
        'per_column': per_column_metrics,
        'predictions': class_prediction,
        'ground_truth': ground_truth_test_array,
        'classifier': classifier
    }
    
    return result


def save_results(results, output_dir):
    """
    保存结果到文件
    """
    try:
        os.makedirs(output_dir)
    except OSError:
        pass  # 目录已存在，忽略错误
    
    # 保存整体评估指标
    with open(os.path.join(output_dir, 'overall_metrics.txt'), 'w') as f:
        f.write("整体评估指标:\n")
        f.write("F1 Score: {:.4f}\n".format(results['overall']['f1']))
        f.write("Precision: {:.4f}\n".format(results['overall']['precision']))
        f.write("Recall: {:.4f}\n".format(results['overall']['recall']))
        f.write("Training Time: {:.2f} seconds\n".format(results['overall']['training_time']))
        f.write("Total Train Samples: {}\n".format(results['overall']['total_train_samples']))
        f.write("Total Test Samples: {}\n".format(results['overall']['total_test_samples']))
        f.write("True Negatives: {}\n".format(results['overall']['true_negatives']))
        f.write("False Positives: {}\n".format(results['overall']['false_positives']))
        f.write("False Negatives: {}\n".format(results['overall']['false_negatives']))
        f.write("True Positives: {}\n".format(results['overall']['true_positives']))
    
    # 保存按列的评估指标
    with open(os.path.join(output_dir, 'per_column_metrics.csv'), 'w') as f:
        f.write("Column,F1,Precision,Recall,ErrorCount,ErrorRate\n")
        for col_name, metrics in results['per_column'].items():
            f.write("{},{:.4f},{:.4f},{:.4f},{},{:.4f}\n".format(
                col_name, 
                metrics['f1'], 
                metrics['precision'], 
                metrics['recall'],
                metrics['error_count'],
                metrics['error_rate']
            ))
    
    # 保存预测结果
    np.save(os.path.join(output_dir, 'predictions.npy'), results['predictions'])
    np.save(os.path.join(output_dir, 'ground_truth.npy'), results['ground_truth'])
    
    print "Results saved to", output_dir


def main():
    # 在这里设置文件路径
    train_dirty_data_path = "/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_dirty_data_hospital_1000_x.csv"  # 训练脏数据文件路径
    train_clean_data_path = "/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_clean_data_hospital_1000.csv"  # 训练干净数据文件路径
    test_dirty_data_path = "/home/stu/pys/CLLM-main/data/error_detection/hospital/dirty.csv"    # 测试脏数据文件路径
    test_clean_data_path = "/home/stu/pys/CLLM-main/data/error_detection/hospital/clean.csv"    # 测试干净数据文件路径
    output_dir = "/home/stu/pys/CLLM-main/data/error_detection/hospital"                          # 输出目录
    model_path = os.path.join(output_dir, "ed2_model.pkl")   # 模型保存路径
    
    # 检查文件是否存在
    for file_path in [train_dirty_data_path, train_clean_data_path, test_dirty_data_path, test_clean_data_path]:
        if not os.path.exists(file_path):
            print "错误：文件不存在:", file_path
            sys.exit(1)
    
    # 加载数据集
    print "加载训练数据..."
    train_dataSet = CustomDataSet(train_dirty_data_path, train_clean_data_path, "TrainData")
    
    print "加载测试数据..."
    test_dataSet = CustomDataSet(test_dirty_data_path, test_clean_data_path, "TestData")
    
    print "训练数据形状:", train_dataSet.shape
    print "测试数据形状:", test_dataSet.shape
    
    # 训练和评估模型
    print "训练和评估模型..."
    result = train_and_evaluate_model(train_dataSet, test_dataSet, model_path)
    
    # 打印结果
    print "\n整体结果:"
    print "F1 Score: {:.4f}".format(result['overall']['f1'])
    print "Precision: {:.4f}".format(result['overall']['precision'])
    print "Recall: {:.4f}".format(result['overall']['recall'])
    print "Training Time: {:.2f} seconds".format(result['overall']['training_time'])
    
    print "\n按列结果:"
    for col_name, metrics in result['per_column'].items():
        print "{}: F1={:.4f}, Precision={:.4f}, Recall={:.4f}, ErrorRate={:.4f}".format(
            col_name, metrics['f1'], metrics['precision'], metrics['recall'], metrics['error_rate'])
    
    # 保存结果
    save_results(result, output_dir)
    
    print "\n完成！"


if __name__ == "__main__":
    main()