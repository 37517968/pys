# -*- coding: utf-8 -*-
"""
简化版错误检测方法，基于 ed2 的实现，只保留核心的训练和检测功能
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, vstack, csr_matrix, lil_matrix
import xgboost as xgb
import pickle
import os
from dataset import Dataset


class ED2ErrorDetector:
    """
    简化版错误检测器，基于 ed2 的实现
    """
    
    def __init__(self, ignore_columns=None):
        """初始化错误检测器"""
        self.model = None
        self.feature_names = None
        self.column_names = None
        self.fitted_pipelines = {}  # 保存每列拟合好的Pipeline
        self.is_fitted = False  # 标记是否已经拟合
        self.ignore_columns = ignore_columns or []
        
        # 初始化新特征相关的变量
        self.ngram_pipelines = {}  # 保存每列拟合好的n-gram Pipeline
        self.unique_value_counts = {}  # 用于值出现次数特征
        self.column_dictionaries = []  # 用于Word2Vec特征
        self.word2vec_model = None  # Word2Vec模型
        self.w2v_vector_size = 100  # Word2Vec向量维度
        
    def create_features(self, dataframe, is_training=True):
        """
        创建特征矩阵，实现ED2的特征生成要求
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        - is_training: 是否为训练阶段
        
        返回:
        - 特征矩阵, 特征名称列表
        """
        dataframe = dataframe.drop(columns=self.ignore_columns, errors='ignore')
        feature_name_list = []
        feature_list = []
        
        # 字符级 n-gram 文本特征 (ρ_n-grams)
        ngram_features, ngram_feature_names = self._add_ngram_features(dataframe, is_training)
        feature_list.append(ngram_features)
        feature_name_list.extend(ngram_feature_names)
        
        # 元数据特征 (ρ_meta)
        meta_features, meta_feature_names = self._add_metadata_features(dataframe, is_training)
        feature_list.append(meta_features)
        feature_name_list.extend(meta_feature_names)
        
        # Word2Vec 单元格值相关特征 (ρ_w2v)
        w2v_features, w2v_feature_names = self._add_word2vec_features(dataframe, is_training)
        if w2v_features is not None:
            feature_list.append(w2v_features)
            feature_name_list.extend(w2v_feature_names)
        
        # 按列特征拼接 (ρ_concat)
        concat_features, concat_feature_names = self._add_concatenated_features(dataframe, is_training)
        if concat_features is not None:
            feature_list.append(concat_features)
            feature_name_list.extend(concat_feature_names)
        
        # 错误相关特征 (ρ_error) - 仅在非训练阶段（测试阶段）添加
        # if not is_training:
        #     error_features, error_feature_names = self._add_error_features(dataframe)
        #     if error_features is not None:
        #         feature_list.append(error_features)
        #         feature_name_list.extend(error_feature_names)
        
        # 水平堆叠所有特征
        feature_matrix = hstack(feature_list).tocsr()
        
        return feature_matrix, feature_name_list
    
    def _add_ngram_features(self, dataframe, is_training=True):
        """
        添加字符级 n-gram 文本特征 (ρ_n-grams)
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        - is_training: 是否为训练阶段
        
        返回:
        - n-gram 特征矩阵, n-gram 特征名称列表
        """
        feature_name_list = []
        feature_list = []
        
        # 使用与原始ED2相同的参数
        ngrams = 2  # 原始ED2使用2-gram
        use_tf_idf = True  # 原始ED2使用TF-IDF
        
        # 为每一列创建特征
        for column_id in range(dataframe.shape[1]):
            # 使用numpy数组而不是pandas Series，与原始ED2一致
            data_column = dataframe.iloc[:, column_id].astype(str).values
            column_name = dataframe.columns[column_id]
            
            if column_name in self.ignore_columns:
                print(f"Skipping column for n-gram features: {column_name}")
                continue

            try:
                if is_training:
                    # 训练阶段：创建并拟合新的Pipeline
                    if use_tf_idf:
                        pipeline = Pipeline([
                            ('vect', CountVectorizer(analyzer='char', lowercase=False, ngram_range=(1, ngrams))),
                            ('tfidf', TfidfTransformer())
                        ])
                    else:
                        pipeline = Pipeline([
                            ('vect', CountVectorizer(analyzer='char', lowercase=False, ngram_range=(1, ngrams)))
                        ])
                    
                    pipeline.fit(data_column)
                    feature_matrix = pipeline.transform(data_column).astype(float)
                    
                    # 保存拟合好的Pipeline
                    if not hasattr(self, 'ngram_pipelines'):
                        self.ngram_pipelines = {}
                    self.ngram_pipelines[column_name] = pipeline
                    
                    # 获取特征名称，参考原始ED2的命名方式
                    import operator
                    vocab_items = sorted(pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
                    feature_name_list.extend([f"{column_name}_ngram_{item[0]}" for item in vocab_items])
                else:
                    # 测试阶段：使用已保存的Pipeline
                    if not hasattr(self, 'ngram_pipelines') or column_name not in self.ngram_pipelines:
                        raise ValueError(f"No fitted n-gram pipeline found for column '{column_name}'. Make sure to train the model first.")
                    
                    pipeline = self.ngram_pipelines[column_name]
                    feature_matrix = pipeline.transform(data_column).astype(float)
                    
                    # 获取特征名称
                    import operator
                    vocab_items = sorted(pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
                    feature_name_list.extend([f"{column_name}_ngram_{item[0]}" for item in vocab_items])
                
                feature_list.append(feature_matrix)
                
            except ValueError:
                # 如果某一列无法处理，跳过
                pass
        
        # 水平堆叠所有n-gram特征
        if feature_list:
            ngram_matrix = hstack(feature_list).tocsr()
        else:
            # 如果没有特征，返回空矩阵
            ngram_matrix = csr_matrix((dataframe.shape[0], 0))
        
        return ngram_matrix, feature_name_list
    
    def _add_metadata_features(self, dataframe, is_training=True):
        """
        添加元数据特征 (ρ_meta)，包含值出现次数、字符串长度、数据类型、数值表示等
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        - is_training: 是否为训练阶段
        
        返回:
        - 元数据特征矩阵, 元数据特征名称列表
        """
        meta_feature_names = []
        meta_features = []
        
        # 初始化唯一值计数字典（仅在训练阶段）
        if is_training:
            self.unique_value_counts = {}
        
        # 为每列添加元数据特征
        for col_idx, col_name in enumerate(dataframe.columns):
            col_data = dataframe[col_name].astype(str).values
            
            # 1. 获取值出现次数特征（ρ_occurrence）[39]
            if is_training:
                self._get_number_of_occurrences_fit(col_data, col_idx)
            occurrence_feature, occurrence_name = self._get_number_of_occurrences_transform(col_data, col_idx)
            meta_features.append(occurrence_feature)
            meta_feature_names.append(f"{col_name}_{occurrence_name}")
            
            # 2. 字符串长度特征（ρ_string_length）[28]
            length_feature, length_name = self._string_length(col_data, col_idx)
            meta_features.append(length_feature)
            meta_feature_names.append(f"{col_name}_{length_name}")
            
            # 3. 数据类型特征（ρ_data_type）[39]
            data_type_features, data_type_names = self._get_data_type_features(col_data, col_idx)
            meta_features.extend(data_type_features)
            meta_feature_names.extend([f"{col_name}_{name}" for name in data_type_names])
            
            # 4. 数值表示特征（ρ_number）[21]
            numeric_feature, numeric_name = self._is_numerical(col_data, col_idx)
            meta_features.append(numeric_feature)
            meta_feature_names.append(f"{col_name}_{numeric_name}")
            
            # 5. 提取数值特征
            number_feature, number_name = self._extract_number(col_data, col_idx)
            meta_features.append(number_feature)
            meta_feature_names.append(f"{col_name}_{number_name}")
        
        # 水平堆叠所有元数据特征
        meta_matrix = hstack(meta_features).tocsr()
        
        return meta_matrix, meta_feature_names
    
    def _get_data_type_features(self, col_data, col_idx):
        """
        获取数据类型特征（ρ_data_type），识别字符串、数值、日期等类型
        
        参数:
        - col_data: 列数据
        - col_idx: 列索引
        
        返回:
        - 数据类型特征列表, 特征名称列表
        """
        features = []
        feature_names = []
        
        # 是否为数值
        is_numeric = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            is_numeric[i] = self._is_number(col_data[i])
        features.append(csr_matrix(is_numeric))
        feature_names.append("is_numeric")
        
        # 是否为整数
        is_integer = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            try:
                int(col_data[i])
                is_integer[i] = True
            except ValueError:
                pass
        features.append(csr_matrix(is_integer))
        feature_names.append("is_integer")
        
        # 是否为浮点数
        is_float = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            try:
                float_value = float(col_data[i])
                is_float[i] = ('.' in col_data[i] or 'e' in col_data[i].lower()) and self._is_number(col_data[i])
            except ValueError:
                pass
        features.append(csr_matrix(is_float))
        feature_names.append("is_float")
        
        # 是否为日期
        is_date = np.zeros((col_data.shape[0], 1), dtype=bool)
        from datetime import datetime
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y%m%d', '%d%m%Y', '%m%d%Y'
        ]
        for i in range(col_data.shape[0]):
            for fmt in date_formats:
                try:
                    datetime.strptime(col_data[i], fmt)
                    is_date[i] = True
                    break
                except ValueError:
                    pass
        features.append(csr_matrix(is_date))
        feature_names.append("is_date")
        
        # 是否为布尔值
        is_boolean = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            lower_val = col_data[i].lower()
            is_boolean[i] = lower_val in ['true', 'false', 'yes', 'no', '1', '0']
        features.append(csr_matrix(is_boolean))
        feature_names.append("is_boolean")
        
        # 是否为字母
        is_alphabetical = np.zeros((col_data.shape[0], 1), dtype=bool)
        import re
        for i in range(col_data.shape[0]):
            if re.match(r"^[A-Za-z_]+$", col_data[i]):
                is_alphabetical[i, 0] = True
        features.append(csr_matrix(is_alphabetical))
        feature_names.append("is_alphabetical")
        
        # 是否为字母数字
        is_alphanumeric = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            if re.match(r"^[A-Za-z0-9_]+$", col_data[i]):
                is_alphanumeric[i, 0] = True
        features.append(csr_matrix(is_alphanumeric))
        feature_names.append("is_alphanumeric")
        
        # 是否为纯文本（包含空格）
        is_text = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            if re.match(r"^[A-Za-z0-9_\s]+$", col_data[i]):
                is_text[i, 0] = True
        features.append(csr_matrix(is_text))
        feature_names.append("is_text")
        
        return features, feature_names
    
    def _get_number_of_occurrences_fit(self, col_data, col_idx):
        """在训练阶段拟合值出现次数特征"""
        if col_idx not in self.unique_value_counts:
            self.unique_value_counts[col_idx] = {}
        
        for i in range(col_data.shape[0]):
            value = col_data[i]
            if value in self.unique_value_counts[col_idx]:
                self.unique_value_counts[col_idx][value] += 1
            else:
                self.unique_value_counts[col_idx][value] = 1
    
    def _get_number_of_occurrences_transform(self, col_data, col_idx):
        """在转换阶段获取值出现次数特征"""
        feature = lil_matrix((col_data.shape[0], 1))
        for i in range(col_data.shape[0]):
            value = col_data[i]
            if col_idx in self.unique_value_counts and value in self.unique_value_counts[col_idx]:
                feature[i] = self.unique_value_counts[col_idx][value]
            else:
                # 如果值在训练阶段未见过，设为0
                feature[i] = 0
        
        return feature.tocsr(), 'occurrence_count'
    
    def _is_number(self, value):
        """检查值是否为数字"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_numerical(self, col_data, col_idx):
        """是否为数值特征"""
        feature = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            value = col_data[i]
            feature[i] = self._is_number(value)
        
        return csr_matrix(feature), 'is_numerical'
    
    def _string_length(self, col_data, col_idx):
        """字符串长度特征"""
        feature = np.zeros((col_data.shape[0], 1))
        for i in range(col_data.shape[0]):
            value = col_data[i]
            try:
                feature[i] = len(str(value.encode('utf-8')))
            except:
                feature[i] = len(str(value))
        return csr_matrix(feature), 'string_length'
    
    def _is_alphabetical(self, col_data, col_idx):
        """是否为字母特征"""
        import re
        feature = np.zeros((col_data.shape[0], 1), dtype=bool)
        for i in range(col_data.shape[0]):
            value = col_data[i]
            # 检查是否只包含字母和下划线
            if re.match(r"^[A-Za-z_]+$", value):
                feature[i, 0] = True
        
        return csr_matrix(feature), 'is_alphabetical'
    
    def _extract_number(self, col_data, col_idx):
        """提取数值特征"""
        feature = lil_matrix((col_data.shape[0], 1))
        for i in range(col_data.shape[0]):
            value = col_data[i]
            try:
                feature[i] = float(value)
            except ValueError:
                pass  # 如果无法转换为数字，保持为0
        return feature.tocsr(), 'extracted_number'
    
    def _add_word2vec_features(self, dataframe, is_training=True):
        """
        添加 Word2Vec 单元格值相关特征 (ρ_w2v)
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        - is_training: 是否为训练阶段
        
        返回:
        - Word2Vec 特征矩阵, Word2Vec 特征名称列表
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("Warning: gensim not installed. Skipping Word2Vec features.")
            return None, []
        
        feature_names = []
        
        if is_training:
            # 训练阶段：训练 Word2Vec 模型
            # 创建字典：值 -> 单词
            self.column_dictionaries = []
            words = np.zeros((dataframe.shape[0], dataframe.shape[1]), dtype=object)
            
            for column_i in range(dataframe.shape[1]):
                col_name = dataframe.columns[column_i]
                if col_name in self.ignore_columns:
                    continue
                    
                col_val2word = {}
                for row_i in range(dataframe.shape[0]):
                    val = str(dataframe.iloc[row_i, column_i])
                    if not val in col_val2word:
                        col_val2word[val] = 'col' + str(column_i) + "_" + str(len(col_val2word))
                    words[row_i, column_i] = col_val2word[val]
                self.column_dictionaries.append(col_val2word)
            
            # 训练 Word2Vec 模型
            vector_size = 100  # 嵌入向量维度
            self.word2vec_model = Word2Vec(
                words.tolist(),
                vector_size=vector_size,
                window=dataframe.shape[1] * 2,
                min_count=1,
                workers=4,
                negative=0,
                hs=1
            )
            self.word2vec_model.train(words.tolist(), total_examples=words.shape[0], epochs=10)
            
            # 保存向量维度
            self.w2v_vector_size = vector_size
        else:
            # 测试阶段：使用已训练的 Word2Vec 模型
            if not hasattr(self, 'word2vec_model'):
                print("Warning: No trained Word2Vec model found. Skipping Word2Vec features.")
                return None, []
        
        # 转换数据为 Word2Vec 特征
        words = np.zeros((dataframe.shape[0], dataframe.shape[1]), dtype=object)
        
        for column_i in range(dataframe.shape[1]):
            col_name = dataframe.columns[column_i]
            if col_name in self.ignore_columns:
                continue
                
            for row_i in range(dataframe.shape[0]):
                val = str(dataframe.iloc[row_i, column_i])
                if column_i < len(self.column_dictionaries) and val in self.column_dictionaries[column_i]:
                    words[row_i, column_i] = self.column_dictionaries[column_i][val]
        
        # 创建最终的特征矩阵
        vector_size = self.w2v_vector_size
        final_matrix = np.zeros((words.shape[0], vector_size * words.shape[1]))
        
        for column_i in range(words.shape[1]):
            col_name = dataframe.columns[column_i]
            if col_name in self.ignore_columns:
                continue
                
            for row_i in range(words.shape[0]):
                try:
                    if words[row_i, column_i] != 0:  # 确保不是空值
                        final_matrix[row_i, column_i * vector_size:(column_i + 1) * vector_size] = \
                            self.word2vec_model.wv[words[row_i, column_i]]
                except KeyError:
                    # 如果单词不在词汇表中，保持为零向量
                    pass
        
        # 生成特征名称
        for column_i in range(dataframe.shape[1]):
            col_name = dataframe.columns[column_i]
            if col_name in self.ignore_columns:
                continue
                
            for vec_i in range(vector_size):
                feature_names.append(f"{col_name}_word2vec_{vec_i}")
        
        return csr_matrix(final_matrix), feature_names
    
    def _add_concatenated_features(self, dataframe, is_training=True):
        """
        添加按列特征拼接 (ρ_concat)
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        - is_training: 是否为训练阶段
        
        返回:
        - 拼接特征矩阵, 拼接特征名称列表
        """
        # 获取所有列的 n-gram 特征
        ngram_features, ngram_feature_names = self._add_ngram_features(dataframe, is_training)
        
        # 获取所有列的元数据特征
        meta_features, meta_feature_names = self._add_metadata_features(dataframe, is_training)
        
        # 计算每列的特征数量
        num_cols = dataframe.shape[1]
        ngram_cols = ngram_features.shape[1] // num_cols if num_cols > 0 else 0
        meta_cols = meta_features.shape[1] // num_cols if num_cols > 0 else 0
        
        # 为每列创建拼接特征，然后水平堆叠
        concat_features = []
        concat_feature_names = []
        
        for col_idx in range(num_cols):
            col_name = dataframe.columns[col_idx]
            if col_name in self.ignore_columns:
                continue
                
            # 获取当前列的 n-gram 特征
            start_idx = col_idx * ngram_cols
            end_idx = (col_idx + 1) * ngram_cols
            col_ngram_features = ngram_features[:, start_idx:end_idx]
            
            # 获取当前列的元数据特征
            start_idx = col_idx * meta_cols
            end_idx = (col_idx + 1) * meta_cols
            col_meta_features = meta_features[:, start_idx:end_idx]
            
            # 获取其他列的特征
            other_cols_ngram = []
            other_cols_meta = []
            
            for other_col_idx in range(num_cols):
                if other_col_idx == col_idx:
                    continue
                    
                other_col_name = dataframe.columns[other_col_idx]
                if other_col_name in self.ignore_columns:
                    continue
                    
                # 获取其他列的 n-gram 特征
                start_idx = other_col_idx * ngram_cols
                end_idx = (other_col_idx + 1) * ngram_cols
                other_col_ngram = ngram_features[:, start_idx:end_idx]
                other_cols_ngram.append(other_col_ngram)
                
                # 获取其他列的元数据特征
                start_idx = other_col_idx * meta_cols
                end_idx = (other_col_idx + 1) * meta_cols
                other_col_meta = meta_features[:, start_idx:end_idx]
                other_cols_meta.append(other_col_meta)
            
            # 水平堆叠其他列的特征
            if other_cols_ngram and other_cols_meta:
                other_ngram_concat = hstack(other_cols_ngram)
                other_meta_concat = hstack(other_cols_meta)
                
                # 创建当前列与其他列特征的拼接
                col_concat = hstack([col_ngram_features, col_meta_features, other_ngram_concat, other_meta_concat])
                
                # 生成特征名称
                for i in range(col_concat.shape[1]):
                    concat_feature_names.append(f"{col_name}_concat_{i}")
                
                concat_features.append(col_concat)
        
        # 水平堆叠所有列的拼接特征，而不是垂直堆叠
        if concat_features:
            all_concat_features = hstack(concat_features).tocsr()
        else:
            # 如果没有特征，返回空矩阵
            all_concat_features = csr_matrix((dataframe.shape[0], 0))
        
        return all_concat_features, concat_feature_names
    
    def _add_error_features(self, dataframe):
        """
        添加错误相关特征 (ρ_error)
        
        参数:
        - dataframe: 输入的数据集 DataFrame
        
        返回:
        - 错误特征矩阵, 错误特征名称列表
        """
        if not hasattr(self, 'model') or self.model is None:
            print("Warning: No trained model found. Skipping error features.")
            return None, []
        
        # 获取所有列的预测结果
        error_features = []
        error_feature_names = []
        
        # 为每列创建特征
        for col_idx in range(dataframe.shape[1]):
            col_name = dataframe.columns[col_idx]
            if col_name in self.ignore_columns:
                continue
                
            # 获取其他列的错误概率
            other_col_probs = []
            other_col_names = []
            
            for other_col_idx in range(dataframe.shape[1]):
                if other_col_idx == col_idx:
                    continue
                    
                other_col_name = dataframe.columns[other_col_idx]
                if other_col_name in self.ignore_columns:
                    continue
                
                # 为其他列创建特征矩阵
                feature_matrix, _ = self.create_features(dataframe, is_training=False)
                
                # 添加列标识
                one_hot_part = np.zeros((dataframe.shape[0], dataframe.shape[1]))
                one_hot_part[:, other_col_idx] = 1
                col_feature_matrix = hstack((feature_matrix, one_hot_part)).tocsr()
                
                # 转换为DMatrix格式
                dtest = xgb.DMatrix(col_feature_matrix, feature_names=self.feature_names)
                
                # 预测错误概率
                probabilities = self.model.predict(dtest)
                other_col_probs.append(probabilities)
                other_col_names.append(other_col_name)
            
            # 水平堆叠其他列的错误概率
            if other_col_probs:
                col_error_features = np.column_stack(other_col_probs)
                error_features.append(col_error_features)
                
                # 生成特征名称
                for i, other_col_name in enumerate(other_col_names):
                    error_feature_names.append(f"{col_name}_error_prob_{other_col_name}")
        
        # 垂直堆叠所有列的错误特征
        if error_features:
            all_error_features = vstack([csr_matrix(feat) for feat in error_features])
        else:
            # 如果没有特征，返回空矩阵
            all_error_features = csr_matrix((dataframe.shape[0] * dataframe.shape[1], 0))
        
        return all_error_features, error_feature_names
    
    def prepare_training_data(self, train_dataframe, clean_dataframe):
        """
        准备训练数据，为每个单元格创建特征和标签
        
        参数:
        - train_dataframe: 训练数据 DataFrame
        - clean_dataframe: 清洁数据 DataFrame（用于生成标签）
        
        返回:
        - 特征矩阵, 标签数组
        """
        # 生成标签：比较训练数据和清洁数据，标记不同的单元格
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
        
        for col in getattr(self, "ignore_columns", []):
            if col in train_dataframe.columns:
                print(f"Skipping train for column: {col}")
                train_dataframe = train_dataframe.drop(columns=[col])
                clean_dataframe = clean_dataframe.drop(columns=[col])

        labels = np.zeros(train_dataframe.shape, dtype=bool)
        
        for i in range(train_dataframe.shape[0]):
            for j in range(train_dataframe.shape[1]):
                col_name = train_dataframe.columns[j]
                if col_name in self.ignore_columns:
                    labels[i, j] = False  # 忽略列永远标为干净
                    continue
                train_value = value_normalizer(str(train_dataframe.iloc[i, j]))
                clean_value = value_normalizer(str(clean_dataframe.iloc[i, j]))
                labels[i, j] = (train_value != clean_value)
        
        

        # 为训练集创建特征
        feature_matrix, feature_names = self.create_features(train_dataframe, is_training=True)
        
        # 为每列创建特征矩阵并添加列标识
        feature_matrix_per_column = []
        
        self.column_names = train_dataframe.columns.tolist()

        # 删除被忽略的列，保持一致性
        for column_i in range(train_dataframe.shape[1]):
            # 添加列标识的one-hot编码
            one_hot_part = np.zeros((train_dataframe.shape[0], train_dataframe.shape[1]))
            one_hot_part[:, column_i] = 1
            
            # 将列标识特征与原有特征合并
            feature_matrix_per_column.append(hstack((feature_matrix, one_hot_part)).tocsr())
        
        # 垂直堆叠所有列的特征矩阵
        all_columns_feature_matrix = vstack(feature_matrix_per_column)
        
        # 创建标签数组（将所有列的标签展平）
        labels_array = labels.flatten()
        
        self.feature_names = feature_names + [f"column_id_{i}" for i in range(train_dataframe.shape[1])]
        
        return all_columns_feature_matrix, labels_array
    
    def train(self, train_dataframe, clean_dataframe, use_cv=False, cv_folds=5):
        """
        训练错误检测模型
        
        参数:
        - train_dataframe: 训练数据 DataFrame
        - clean_dataframe: 清洁数据 DataFrame（用于生成标签）
        - use_cv: 是否使用交叉验证
        - cv_folds: 交叉验证折数
        
        返回:
        - 训练好的模型
        """
        print("=== 开始训练ED2错误检测模型 ===")
        print(f"训练数据形状: {train_dataframe.shape}")
        print(f"清洁数据形状: {clean_dataframe.shape}")
        print(f"忽略的列: {self.ignore_columns}")
        
        # 准备训练数据
        feature_matrix, labels = self.prepare_training_data(train_dataframe, clean_dataframe)
        
        print(f"特征矩阵形状: {feature_matrix.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"正样本数量: {np.sum(labels)}, 负样本数量: {np.sum(~labels)}")
        
        # 打印特征组成信息
        if hasattr(self, 'feature_names') and self.feature_names:
            ngram_count = sum(1 for name in self.feature_names if '_ngram_' in name)
            meta_count = sum(1 for name in self.feature_names if any(x in name for x in ['occurrence_count', 'string_length', 'is_numeric', 'is_integer', 'is_float', 'is_date', 'is_boolean', 'is_alphabetical', 'is_alphanumeric', 'is_text', 'extracted_number']))
            w2v_count = sum(1 for name in self.feature_names if '_word2vec_' in name)
            concat_count = sum(1 for name in self.feature_names if '_concat_' in name)
            error_count = sum(1 for name in self.feature_names if '_error_prob_' in name)
            column_id_count = sum(1 for name in self.feature_names if 'column_id_' in name)
            
            print("=== 特征组成 ===")
            print(f"字符级 n-gram 特征: {ngram_count}")
            print(f"元数据特征: {meta_count}")
            print(f"Word2Vec 特征: {w2v_count}")
            print(f"拼接特征: {concat_count}")
            print(f"错误相关特征: {error_count}")
            print(f"列标识特征: {column_id_count}")
            print(f"总特征数: {len(self.feature_names)}")
        
        # 设置XGBoost参数，参考原始ED2的默认参数
        params = {
            'learning_rate': 0.1,
            'colsample_bytree': 0.8,
            'verbosity': 0,  # 替代已弃用的silent参数
            'seed': 0,
            'objective': 'binary:logistic',
            'n_jobs': 4,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8
        }
        
        # 如果需要，运行交叉验证
        if use_cv:
            print(f"运行 {cv_folds} 折交叉验证...")
            params = self._run_cross_validation(feature_matrix, labels, cv_folds, params)
        
        # 处理类别不平衡，参考原始ED2的方法
        if np.sum(labels) > 0 and np.sum(~labels) > 0:
            ratio = float(np.sum(~labels)) / np.sum(labels)
            print(f"类别不平衡比例: {ratio}")
            params['scale_pos_weight'] = ratio
        
        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(feature_matrix, label=labels, feature_names=self.feature_names)
        
        # 训练模型，使用原始ED2的训练轮数
        print("训练XGBoost模型...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            verbose_eval=False
        )
        
        # 标记模型已训练
        self.is_fitted = True
        print("=== 模型训练完成 ===")
        
        return self.model
    
    def _run_cross_validation(self, X, y, folds, params):
        """
        运行交叉验证以找到最佳参数，参考原始ED2的实现
        
        参数:
        - X: 特征矩阵
        - y: 标签数组
        - folds: 交叉验证折数
        - params: 初始参数
        
        返回:
        - 更新后的参数
        """
        from sklearn.model_selection import GridSearchCV
        
        cv_params = {
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'max_depth': [3, 5, 7]
        }
        
        ind_params = params.copy()
        
        # 创建XGBoost分类器
        xgb_classifier = xgb.XGBClassifier(**ind_params)
        
        # 运行网格搜索
        optimized_GBM = GridSearchCV(
            xgb_classifier,
            cv_params,
            scoring='f1',
            cv=folds,
            n_jobs=1,
            verbose=0
        )
        
        print(f"Running cross-validation with {folds} folds...")
        print(X.shape)
        optimized_GBM.fit(X, y)
        
        # 更新参数为最佳参数
        best_params = optimized_GBM.best_params_
        print(f"Best CV parameters: {best_params}")
        
        # 更新参数
        params.update(best_params)
        
        return params
    
    def detect_errors(self, test_dataframe):
        """
        在新数据上检测错误
        
        参数:
        - test_dataframe: 测试数据 DataFrame
        
        返回:
        - 包含预测结果的字典
        """
        print("=== 开始错误检测 ===")
        print(f"测试数据形状: {test_dataframe.shape}")

        import copy
        test_dataframe = copy.deepcopy(test_dataframe)

        # 跳过 ignore_columns 中的列
        for col in getattr(self, "ignore_columns", []):
            if col in test_dataframe.columns:
                print(f"跳过检测列: {col}")
                test_dataframe = test_dataframe.drop(columns=[col])
                
        if self.model is None:
            raise ValueError("模型尚未训练。请先调用 train() 方法。")
        
        # 为测试数据创建特征
        feature_matrix, feature_names = self.create_features(test_dataframe, is_training=False)
        
        # 打印特征组成信息
        if feature_names:
            ngram_count = sum(1 for name in feature_names if '_ngram_' in name)
            meta_count = sum(1 for name in feature_names if any(x in name for x in ['occurrence_count', 'string_length', 'is_numeric', 'is_integer', 'is_float', 'is_date', 'is_boolean', 'is_alphabetical', 'is_alphanumeric', 'is_text', 'extracted_number']))
            w2v_count = sum(1 for name in feature_names if '_word2vec_' in name)
            concat_count = sum(1 for name in feature_names if '_concat_' in name)
            error_count = sum(1 for name in feature_names if '_error_prob_' in name)
            
            print("=== 测试特征组成 ===")
            print(f"字符级 n-gram 特征: {ngram_count}")
            print(f"元数据特征: {meta_count}")
            print(f"Word2Vec 特征: {w2v_count}")
            print(f"拼接特征: {concat_count}")
            print(f"错误相关特征: {error_count}")
            print(f"总特征数: {len(feature_names)}")
        
        # 为每列创建特征矩阵并添加列标识
        feature_matrix_per_column = []
        
        for column_i in range(test_dataframe.shape[1]):
            # 添加列标识的one-hot编码
            one_hot_part = np.zeros((test_dataframe.shape[0], test_dataframe.shape[1]))
            one_hot_part[:, column_i] = 1
            
            # 将列标识特征与原有特征合并
            feature_matrix_per_column.append(hstack((feature_matrix, one_hot_part)).tocsr())
        
        # 垂直堆叠所有列的特征矩阵
        all_columns_feature_matrix = vstack(feature_matrix_per_column)
        
        print(f"测试特征矩阵形状: {all_columns_feature_matrix.shape}")
        print(f"期望特征名称数量: {len(self.feature_names)}")
        print(f"实际特征矩阵列数: {all_columns_feature_matrix.shape[1]}")
        
        # 确保特征数量匹配
        if all_columns_feature_matrix.shape[1] != len(self.feature_names):
            print(f"警告: 特征数量不匹配。期望 {len(self.feature_names)}，实际 {all_columns_feature_matrix.shape[1]}")
            # 如果特征数量不匹配，调整特征矩阵或特征名称
            if all_columns_feature_matrix.shape[1] > len(self.feature_names):
                # 如果特征矩阵列数多于特征名称，截断特征矩阵
                all_columns_feature_matrix = all_columns_feature_matrix[:, :len(self.feature_names)]
            else:
                # 如果特征矩阵列数少于特征名称，填充零
                padding = csr_matrix((all_columns_feature_matrix.shape[0], len(self.feature_names) - all_columns_feature_matrix.shape[1]))
                all_columns_feature_matrix = hstack([all_columns_feature_matrix, padding])
        
        # 转换为DMatrix格式
        dtest = xgb.DMatrix(all_columns_feature_matrix, feature_names=self.feature_names)
        
        # 预测概率
        print("预测错误概率...")
        probabilities = self.model.predict(dtest)
        predictions = (probabilities > 0.5)
        
        # 按列组织预测结果
        per_column_predictions = {}
        per_column_probabilities = {}
        
        # 保证列名和特征对齐（过滤掉不存在的列）
        valid_columns = [c for c in self.column_names if c in test_dataframe.columns]

        for idx, col_name in enumerate(valid_columns):
            start_idx = idx * test_dataframe.shape[0]
            end_idx = (idx + 1) * test_dataframe.shape[0]

            col_pred = predictions[start_idx:end_idx]
            col_prob = probabilities[start_idx:end_idx]

            per_column_predictions[col_name] = col_pred
            per_column_probabilities[col_name] = col_prob
        
        # 统计检测结果
        total_errors = np.sum(predictions)
        error_rate = total_errors / len(predictions) if len(predictions) > 0 else 0
        print(f"检测到的错误总数: {total_errors}")
        print(f"错误率: {error_rate:.4f}")
        
        # 返回结果
        result = {
            'predictions': predictions,
            'probabilities': probabilities,
            'per_column_predictions': per_column_predictions,
            'per_column_probabilities': per_column_probabilities
        }
        
        print("=== 错误检测完成 ===")
        return result
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        参数:
        - filepath: 模型保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        从文件加载模型
        
        参数:
        - filepath: 模型文件路径
        
        返回:
        - 加载的模型实例
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def prepare_representative_samples_with_clean_data(dataset, sample_path, metric='euclidean'):
    """
    准备代表性样本及其对应的干净数据
    
    参数:
    - dataset: 原始数据集实例
    - sample_path: 代表性样本文件路径
    - metric: 使用的相似性度量方法
    
    返回:
    - train_data: 代表性样本数据 DataFrame
    - train_clean: 代表性样本对应的干净数据 DataFrame
    """
    # 加载代表性样本
    sample_file = os.path.join(sample_path, f"{metric}.csv")
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Representative samples file not found: {sample_file}")
    
    # 创建Dataset实例来加载代表性样本
    sample_dataset_dict = {
        "name": dataset.name,
        "path": sample_file
    }
    representative_samples_dataset = Dataset(sample_dataset_dict)
    representative_samples = representative_samples_dataset.dataframe
    
    print(f"Loaded {len(representative_samples)} representative samples using {metric} metric")
    
    # 从原始数据集中获取对应的清洁数据
    clean_data = dataset.clean_dataframe
    
    # 获取代表性样本的索引
    # 使用第一列ID来匹配对应的干净数据
    train_indices = []
    
    # 获取第一列的名称（ID列）
    id_column = dataset.dataframe.columns[0]
    
    # 创建ID到索引的映射，提高查找效率
    id_to_index = {str(row[id_column]): idx for idx, row in dataset.dataframe.iterrows()}
    
    # 遍历代表性样本，通过ID查找对应的索引
    for _, sample_row in representative_samples.iterrows():
        sample_id = str(sample_row[id_column])
        if sample_id in id_to_index:
            train_indices.append(id_to_index[sample_id])
    
    if not train_indices:
        raise ValueError("No matching rows found between representative samples and original dataset based on ID column")
    
    print(f"Found {len(train_indices)} matching rows in the original dataset using ID column '{id_column}'")
    
    # 使用代表性样本进行训练
    train_data = dataset.dataframe.iloc[train_indices]
    train_clean = clean_data.iloc[train_indices]
    representative_samples_dataset.clean_dataframe = train_clean  # 更新数据集的clean_dataframe属性
    
    return representative_samples_dataset

def create_joint_features(self, train_df, test_df):
    """
    在 train + test 合并数据上拟合特征（模拟原始ED2行为），
    再切分生成对应的 train/test 特征。
    """
    print("使用 train + test 数据同时生成特征 (ED2 joint mode)")
    
    # 合并数据
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 在合并数据上fit特征
    combined_features, feature_names = self.create_features(combined_df, is_training=True)
    
    # 切分回train/test
    n_train = len(train_df)
    X_train = combined_features[:n_train, :]
    X_test = combined_features[n_train:, :]
    
    return X_train, X_test, feature_names
