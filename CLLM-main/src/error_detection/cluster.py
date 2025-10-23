# -*- coding: utf-8 -*-

from error_detection.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def get_representative_samples_error(dirty_dataframe, clean_dataframe, n_clusters=10, similarity_metric='euclidean'):
    """
    使用 dirty_dataframe 进行聚类，并返回脏数据和干净数据对应的聚类结果
    
    参数:
    dirty_dataframe: 用于聚类的脏数据 DataFrame
    clean_dataframe: 对应的干净数据 DataFrame
    n_clusters: 聚类数量，默认为 10
    similarity_metric: 相似性度量方法，支持 'euclidean', 'manhattan', 'cosine'
    
    返回:
    两个 DataFrame，分别对应脏数据和干净数据的聚类结果
    """
    # 检查两个数据框的行数是否相同
    if len(dirty_dataframe) != len(clean_dataframe):
        raise ValueError("脏数据和干净数据的行数必须相同")
    
    # 复制原始数据，避免修改原始 DataFrame
    dirty_df = dirty_dataframe.copy()
    
    # 保存原始索引
    original_indices = dirty_df.index.tolist()
    
    # 处理分类变量：将分类变量转换为数值型
    for col in dirty_df.columns:
        if dirty_df[col].dtype == 'object':
            # 使用标签编码将分类变量转换为数值
            dirty_df[col] = pd.Categorical(dirty_df[col]).codes
    
    # 填充可能的缺失值
    dirty_df = dirty_df.fillna(0)
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dirty_df)
    
    # 应用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_
    
    # 计算每个样本到其聚类中心的距离
    distances = []
    for i in range(len(dirty_df)):
        cluster_id = clusters[i]
        center = cluster_centers[cluster_id]
        
        # 根据选择的相似性度量计算距离
        if similarity_metric == 'euclidean':
            distance = np.linalg.norm(scaled_data[i] - center)
        elif similarity_metric == 'manhattan':
            distance = np.sum(np.abs(scaled_data[i] - center))
        elif similarity_metric == 'cosine':
            distance = 1 - np.dot(scaled_data[i], center) / (np.linalg.norm(scaled_data[i]) * np.linalg.norm(center))
        else:
            raise ValueError(f"不支持的相似性度量方法: {similarity_metric}")
        
        distances.append(distance)
    
    # 将聚类标签和距离添加到原始数据
    dirty_df['cluster'] = clusters
    dirty_df['distance_to_center'] = distances
    dirty_df['original_index'] = original_indices  # 保存原始索引
    
    # 从每个聚类中选择样本
    representative_indices = []
    
    for cluster_id in range(n_clusters):
        cluster_data = dirty_df[dirty_df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # 按距离中心排序
            sorted_cluster = cluster_data.sort_values('distance_to_center')
            
            # 尝试找到脏数据和干净数据不同的样本
            found = False
            for _, row in sorted_cluster.iterrows():
                idx = row['original_index']
                dirty_sample = dirty_dataframe.loc[[idx]]
                clean_sample = clean_dataframe.loc[[idx]]
                
                # 检查脏数据和干净数据是否不同
                if not dirty_sample.equals(clean_sample):
                    representative_indices.append(idx)
                    found = True
                    break
            
            # 如果没有找到不同的样本，则使用距离中心最近的样本
            if not found:
                closest_sample = sorted_cluster.iloc[0]
                representative_indices.append(closest_sample['original_index'])
    
    # 使用选定的索引获取代表性样本
    dirty_representative_df = dirty_dataframe.loc[representative_indices].copy()
    clean_representative_df = clean_dataframe.loc[representative_indices].copy()
    
    return dirty_representative_df, clean_representative_df

def get_representative_samples_with_clean(dirty_dataframe, clean_dataframe, n_clusters=5, similarity_metric='euclidean'):
    """
    使用 dirty_dataframe 进行聚类，并返回脏数据和干净数据对应的聚类结果。
    每个聚类只选择距离中心最近的样本。
    
    参数:
    dirty_dataframe: 用于聚类的脏数据 DataFrame
    clean_dataframe: 对应的干净数据 DataFrame
    n_clusters: 聚类数量，默认为 5
    similarity_metric: 相似性度量方法，支持 'euclidean', 'manhattan', 'cosine'
    
    返回:
    两个 DataFrame，分别对应脏数据和干净数据的代表性样本
    """
    if len(dirty_dataframe) != len(clean_dataframe):
        raise ValueError("脏数据和干净数据的行数必须相同")
    
    dirty_df = dirty_dataframe.copy()
    original_indices = dirty_df.index.tolist()
    
    # 将分类变量转为数值型
    for col in dirty_df.columns:
        if dirty_df[col].dtype == 'object':
            dirty_df[col] = pd.Categorical(dirty_df[col]).codes
    dirty_df = dirty_df.fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dirty_df)
    
    # K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    cluster_centers = kmeans.cluster_centers_
    
    # 计算每个样本到聚类中心的距离
    distances = []
    for i in range(len(dirty_df)):
        center = cluster_centers[clusters[i]]
        if similarity_metric == 'euclidean':
            distance = np.linalg.norm(scaled_data[i] - center)
        elif similarity_metric == 'manhattan':
            distance = np.sum(np.abs(scaled_data[i] - center))
        elif similarity_metric == 'cosine':
            distance = 1 - np.dot(scaled_data[i], center) / (np.linalg.norm(scaled_data[i]) * np.linalg.norm(center))
        else:
            raise ValueError(f"不支持的相似性度量方法: {similarity_metric}")
        distances.append(distance)
    
    dirty_df['cluster'] = clusters
    dirty_df['distance_to_center'] = distances
    dirty_df['original_index'] = original_indices
    
    # 每个聚类选择距离中心最近的样本
    representative_indices = []
    for cluster_id in range(n_clusters):
        cluster_data = dirty_df[dirty_df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            closest_sample = cluster_data.sort_values('distance_to_center').iloc[0]
            representative_indices.append(closest_sample['original_index'])
    
    dirty_representative_df = dirty_dataframe.loc[representative_indices].copy()
    clean_representative_df = clean_dataframe.loc[representative_indices].copy()
    
    return dirty_representative_df, clean_representative_df


def get_representative_samples(dataframe, n_clusters=10, similarity_metric='euclidean', return_original_data=True):
    """
    使用 K-means 算法将数据集中的数据分为 n_clusters 个聚类，
    并取每个聚类最靠近中心的数据作为最有代表性的数据
    
    参数:
    dataframe: 输入的数据集 DataFrame
    n_clusters: 聚类数量，默认为 10
    similarity_metric: 相似性度量方法，支持 'euclidean', 'manhattan', 'cosine'
    return_original_data: 是否返回原始数据，默认为 True
    
    返回:
    如果 return_original_data=True，返回包含每个聚类最靠近中心的原始数据的 DataFrame
    如果 return_original_data=False，返回包含每个聚类最靠近中心的数据的 DataFrame（数值型）
    """
    # 复制原始数据，避免修改原始 DataFrame
    df = dataframe.copy()
    
    # 保存原始索引
    original_indices = df.index.tolist()
    
    # 处理分类变量：将分类变量转换为数值型
    for col in df.columns:
        if df[col].dtype == 'object':
            # 使用标签编码将分类变量转换为数值
            df[col] = pd.Categorical(df[col]).codes
    
    # 填充可能的缺失值
    df = df.fillna(0)
    
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # 应用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_
    
    # 计算每个样本到其聚类中心的距离
    distances = []
    for i in range(len(df)):
        cluster_id = clusters[i]
        center = cluster_centers[cluster_id]
        
        # 根据选择的相似性度量计算距离
        if similarity_metric == 'euclidean':
            distance = np.linalg.norm(scaled_data[i] - center)
        elif similarity_metric == 'manhattan':
            distance = np.sum(np.abs(scaled_data[i] - center))
        elif similarity_metric == 'cosine':
            distance = 1 - np.dot(scaled_data[i], center) / (np.linalg.norm(scaled_data[i]) * np.linalg.norm(center))
        else:
            raise ValueError(f"不支持的相似性度量方法: {similarity_metric}")
        
        distances.append(distance)
    
    # 将聚类标签和距离添加到原始数据
    df['cluster'] = clusters
    df['distance_to_center'] = distances
    df['original_index'] = original_indices  # 保存原始索引
    
    # 从每个聚类中选择距离中心最近的样本
    representative_samples = []
    representative_indices = []
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # 找到距离中心最近的样本
            closest_sample = cluster_data.loc[cluster_data['distance_to_center'].idxmin()]
            representative_samples.append(closest_sample)
            representative_indices.append(closest_sample['original_index'])
    
    # 创建包含代表性样本的 DataFrame
    representative_df = pd.DataFrame(representative_samples)
    
    # 移除添加的列
    representative_df = representative_df.drop(columns=['cluster', 'distance_to_center', 'original_index'])
    
    # 如果需要返回原始数据
    if return_original_data:
        # 使用原始索引从原始数据中获取原始数据
        original_representative_df = dataframe.loc[representative_indices].copy()
        return original_representative_df
    
    return representative_df

def compare_similarity_metrics(dataframe, n_clusters=10, return_original_data=True):
    """
    使用多种相似性度量方法进行实验，并比较结果
    
    参数:
    dataframe: 输入的数据集 DataFrame
    n_clusters: 聚类数量，默认为 10
    return_original_data: 是否返回原始数据，默认为 True
    
    返回:
    一个字典，包含每种相似性度量方法得到的代表性样本 DataFrame
    """
    similarity_metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    for metric in similarity_metrics:
        print(f"使用 {metric} 相似性度量进行聚类...")
        representative_samples = get_representative_samples(dataframe, n_clusters=n_clusters, similarity_metric=metric, return_original_data=return_original_data)
        results[metric] = representative_samples
        print(f"使用 {metric} 得到的代表性样本数量: {len(representative_samples)}")
    
    return results

def restore_original_data(numeric_samples, original_dataframe):
    """
    从数值型代表性样本恢复原始数据
    
    参数:
    numeric_samples: 数值型代表性样本 DataFrame
    original_dataframe: 原始数据 DataFrame
    
    返回:
    包含原始数据的代表性样本 DataFrame
    """
    # 检查是否有原始索引列
    if 'original_index' in numeric_samples.columns:
        # 使用原始索引从原始数据中获取原始数据
        original_indices = numeric_samples['original_index'].tolist()
        original_samples = original_dataframe.loc[original_indices].copy()
        
        # 移除原始索引列
        numeric_samples = numeric_samples.drop(columns=['original_index'])
        
        return original_samples
    else:
        # 如果没有原始索引列，尝试使用 DataFrame 的索引
        try:
            original_samples = original_dataframe.loc[numeric_samples.index].copy()
            return original_samples
        except Exception:
            raise ValueError("无法从数值型样本恢复原始数据：缺少原始索引信息")