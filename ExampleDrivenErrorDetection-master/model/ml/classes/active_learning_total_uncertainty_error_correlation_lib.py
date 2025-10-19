# -*- coding: utf-8 -*-
import pickle
import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import time
model_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, model_path)

from ml.active_learning.library import *
from ml.configuration.Config import Config
from ml.features.CompressedDeepFeatures import read_compressed_deep_features
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from ml.Word2VecFeatures.Word2VecFeatures import Word2VecFeatures
from ml.features.ActiveCleanFeatures import ActiveCleanFeatures
from ml.features.ValueCorrelationFeatures import ValueCorrelationFeatures
from ml.features.BoostCleanMetaFeatures import BoostCleanMetaFeatures
from ml.classes.train_data_cration import create_original_train_data
from ml.classes.train_data_cration import create_next_train_data
from ml.datasets.DataSet import DataSet
import operator

def go_to_next_column_prob(diff_certainty):
	certainty_columns ={}

	for key in diff_certainty.keys():
		certainty_columns[key] = (np.sum(diff_certainty[key]) / len(diff_certainty[key])) * 2

	return min(certainty_columns.iteritems(), key=operator.itemgetter(1))[0]


def go_to_next_column_round(column_id, dataSet):
	column_id = column_id + 1
	if column_id == dataSet.shape[1]:
		column_id = 0
	return column_id

def go_to_next_column_random(dataSet):
	my_list = dataSet.get_applicable_columns()
	id = np.random.randint(len(my_list))
	return my_list[id]

def go_to_next_column(dataSet, statistics,
					  use_max_pred_change_column_selection,
					  use_max_error_column_selection,
					  use_min_certainty_column_selection,
					  use_random_column_selection):
	if use_min_certainty_column_selection:
		return go_to_next_column_prob(statistics['certainty'])
	if use_max_pred_change_column_selection:
		return max(statistics['change'].iteritems(), key=operator.itemgetter(1))[0]
	if use_max_error_column_selection:
		return min(statistics['cross_val_f'].iteritems(), key=operator.itemgetter(1))[0]
	if use_random_column_selection:
		return go_to_next_column_random(dataSet)





def run_multi( params):
	try:
		return run(**params)
	except:
		return_dict = {}
		return_dict['labels'] = []
		return_dict['fscore'] = []
		return_dict['precision'] = []
		return_dict['recall'] = []
		return_dict['time'] = []
		return_dict['error'] = "Unexpected error:" + str(sys.exc_info()[0])

		return return_dict

def run(dataSet,
		 classifier_model,
		 ngrams=1,
		 iterations=1,
		 use_metadata=True, # 引入元数据（metadata）特征。
		 use_word2vec=False, # 能捕捉语义相似性的词嵌入特征。
		 use_boostclean_metadata=False, # 是否使用 BoostClean 系列算法专用的元数据增强
		 w2v_size=100 # Word2Vec向量大小，默认为100
		 ):
	
	
	dataSet.dirty_pd = dataSet.dirty_pd.astype(str)
	dataSet.clean_pd = dataSet.clean_pd.astype(str)
	# 得到下一轮的增强训练数据
	for iteration in range(iterations):
		print("=== 迭代 {}/{} ===".format(iteration + 1, iterations))
		
		# 记录训练数据创建开始时间
		data_creation_start_time = time.time()
		print("开始创建训练数据...")
		
		if iteration == 0:
			train_dirty_df, train_clean_df, ontology_df, error_analysis = create_original_train_data(dataSet.dirty_pd, dataSet.clean_pd,
			train_number=200, cluster_number=5)
			# 保证得到的也是String
			train_dirty_df = train_dirty_df.astype(str)
			train_clean_df = train_clean_df.astype(str)

			test_indices = np.arange(0, dataSet.dirty_pd.shape[0])
			train_indices = np.arange(dataSet.dirty_pd.shape[0], dataSet.dirty_pd.shape[0]+train_dirty_df.shape[0])

			dirty_pd = pd.concat([dataSet.dirty_pd, train_dirty_df], ignore_index=True)
			clean_pd = pd.concat([dataSet.clean_pd, train_clean_df], ignore_index=True)
		else:
			train_dirty_df, train_clean_df = create_next_train_data(dataSet.dirty_pd, dataSet.clean_pd, all_error_pred,
			ontology_df, error_analysis, train_number=200, example_number=5)
			# 保证得到的也是String
			train_dirty_df = train_dirty_df.astype(str)
			train_clean_df = train_clean_df.astype(str)
			start_idx = dataSet.train_indices[-1] + 1  # 从上次最后一个索引的下一个开始
			end_idx = start_idx + train_dirty_df.shape[0]
			new_indices = np.arange(start_idx, end_idx)
			train_indices = np.concatenate([dataSet.train_indices, new_indices])
			dirty_pd = pd.concat([dataSet.dirty_pd, train_dirty_df], ignore_index=True)
			clean_pd = pd.concat([dataSet.clean_pd, train_clean_df], ignore_index=True)
			
		# 记录训练数据创建结束时间
		data_creation_end_time = time.time()
		data_creation_time = data_creation_end_time - data_creation_start_time
		print("训练数据创建完成，耗时: {:.2f} 秒".format(data_creation_time))

		dataSet = DataSet(dataSet.name, dirty_pd, clean_pd, train_indices, dataSet.test_indices)

		# 记录特征创建开始时间
		feature_start_time = time.time()
		
		all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_indices, test_indices, ngrams,
																			is_word=False, use_tf_idf=True)

		if use_metadata:
			all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(dataSet, train_indices,
																						test_indices, all_matrix_train,
																						all_matrix_test, feature_name_list,
																						use_meta_only=False)



		if use_word2vec:
			w2v_features = Word2VecFeatures(vector_size=w2v_size)
			all_matrix_train, all_matrix_test, feature_name_list = w2v_features.add_word2vec_features(dataSet, train_indices,
																						test_indices,
																						all_matrix_train,
																						all_matrix_test,
																						feature_name_list,
																						use_word2vec_only=False)
		if use_boostclean_metadata:
			ac_features = BoostCleanMetaFeatures()  # boost clean metadatda
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																									train_indices,
																									test_indices,
																									all_matrix_train,
																									all_matrix_test,
																									feature_name_list,
																									use_boostclean_metadata_only=False)

		# 记录特征创建结束时间
		feature_end_time = time.time()
		feature_time = feature_end_time - feature_start_time
		print("特征创建完成，耗时: {:.2f} 秒".format(feature_time))

		# 转化为稀疏矩阵
		# try:
		# 	feature_matrix = all_matrix_train.tocsr()
		# except:
		# 	feature_matrix = all_matrix_train

		# 记录训练开始时间
		training_start_time = time.time()
		print("开始训练模型...")
		
		classifier = classifier_model(all_matrix_train, all_matrix_test)
		all_error_status = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=bool)
		all_error_pred = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=float)

		for column_id in range(dataSet.shape[1]):
			print("column: " + str(column_id))

			target_run, _ = getTarget(dataSet, column_id, train_indices, test_indices)

			num_errors = np.sum(target_run)
			
			train_X = all_matrix_train
			train_y = target_run

			# 确保交叉验证的折数至少为1，避免num_errors为0时出错
			folds = max(1, num_errors)
			classifier.run_cross_validation(train_X, train_y, folds, column_id)	#fold = num_errors
			# train_X, train_y训练 all_matrix_train, all_matrix_test预测
			y_pred_train, res_train, y_pred_test, res_test = classifier.train_predict_all(train_X, train_y, column_id, all_matrix_train, all_matrix_test)	
			all_error_status[:, column_id] = res_test
			all_error_pred[:, column_id] = y_pred_test
			# --- 输出该列训练集和测试集指标 ---
			f1_train = f1_score(train_y, res_train)
			precision_train = precision_score(train_y, res_train)
			recall_train = recall_score(train_y, res_train)
			print("Column {} Train Metrics: F1={:.4f}, Precision={:.4f}, Recall={:.4f}".format(column_id, f1_train, precision_train, recall_train))

			test_y = dataSet.matrix_is_error[test_indices, column_id]
			f1_test = f1_score(test_y, res_test)
			precision_test = precision_score(test_y, res_test)
			recall_test = recall_score(test_y, res_test)
			print("Column {} Test Metrics: F1={:.4f}, Precision={:.4f}, Recall={:.4f}".format(column_id, f1_test, precision_test, recall_test))
		# 记录训练结束时间
		training_end_time = time.time()
		training_time = training_end_time - training_start_time
		print("模型训练完成，耗时: {:.2f} 秒".format(training_time))
		
		# 计算最终指标
		metrics = {
			"f1": f1_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status.flatten()),
			"precision": precision_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status.flatten()),
			"recall": recall_score(dataSet.matrix_is_error[test_indices, :].flatten(), all_error_status.flatten())
		}

	return metrics
