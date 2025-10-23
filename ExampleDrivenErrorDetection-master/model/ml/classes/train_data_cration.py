# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import tempfile
import pickle
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
src_path = os.path.join(root_path, 'CLLM-main', 'src')
sys.path.insert(0, root_path)
sys.path.insert(0, src_path)
from error_detection.cluster import get_representative_samples_with_clean
from error_detection.llm_aug import *
def create_original_train_data(dirty_df, clean_df, train_number=200, cluster_number=5, isSave=True):
    """
    使用聚类方法从数据集中创建训练数据
    
    参数:
    dirty_df: 输入的脏数据 DataFrame
    clean_df: 对应的干净数据 DataFrame
    train_number: 需要的训练数据数量，默认为 100
    cluster_number: 聚类数量，默认为 5
    
    返回:
    包含代表性样本的训练数据 DataFrame
    """
    
    # 调用 get_representative_samples 方法，在 cllm_env 环境中运行
    # 使用两个DataFrame参数的版本
    dirty_representative, clean_representative = get_representative_samples_with_clean(
        dirty_df,
        clean_df,
        n_clusters=cluster_number,
        similarity_metric='euclidean'
    )
    error_analysis = analyse_data_error(
        dirty_representative,
        clean_representative,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/error.csv"
    )
    ontology_df = analyse_table_ontology(
        clean_representative,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/ontology.csv"
    )
    enhanced_clean_df = enhance_clean_data(
        clean_dataframe=clean_representative,
        ontology_dataframe=ontology_df,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_clean.csv",
        number=200,
    )
    enhanced_dirty_df = inject_errors(
        clean_dataframe=enhanced_clean_df,
        error_analysis=error_analysis,
        batch_size=20,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_dirty.csv",
    )
    # ontology_df = pd.read_csv("/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/ontology.csv")
    # error_analysis = pd.read_csv("/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/error.csv")
    # enhanced_dirty_df = pd.read_csv("/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_dirty.csv")
    # enhanced_clean_df = pd.read_csv("/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_clean.csv")

    original_train_data = pd.concat([dirty_representative, clean_representative, enhanced_dirty_df, enhanced_clean_df], ignore_index=True)
    
    original_train_data_clean = pd.concat([clean_representative, clean_representative, enhanced_clean_df, enhanced_clean_df], ignore_index=True)

    return original_train_data, original_train_data_clean, ontology_df, error_analysis

def create_next_train_data(dirty_df, clean_df, all_error_pred, ontology_df, error_analysis, train_number=200, example_number=5):
    """
    使用聚类方法从数据集中创建下一轮训练数据

    参数:
    dirty_df: 输入的脏数据 DataFrame
    clean_df: 对应的干净数据 DataFrame
    all_error_pred: 所有错误预测
    train_number: 需要的训练数据数量，默认为 200
    example_number: 每个示例的数量，默认为 5

    返回:
    包含代表性样本的训练数据 DataFrame
    """
    row_confidence = np.mean(all_error_pred, axis=1)
    uncertain_indices = np.argsort(row_confidence)[:]
    example_dirty_df = dirty_df.iloc[uncertain_indices[:example_number]].reset_index(drop=True)
    example_clean_df = clean_df.iloc[uncertain_indices[:example_number]].reset_index(drop=True)
    error_analysis = analyse_data_error_again(
        example_dirty_df,
        example_clean_df,
        error_analysis,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/error.csv"
    )
    enhanced_clean_df = enhance_clean_data(
        clean_dataframe=example_clean_df,
        ontology_dataframe=ontology_df,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_clean.csv",
        number=train_number,
    )
    print("DEBUG: Starting inject_errors function...")
    enhanced_dirty_df = inject_errors(
        clean_dataframe=enhanced_clean_df,
        error_analysis=error_analysis,
        batch_size=20,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_dirty.csv",
    )

    new_train_dirty_df = pd.concat([example_dirty_df, example_clean_df, enhanced_dirty_df, enhanced_clean_df], ignore_index=True)

    new_train_clean_df = pd.concat([example_clean_df, example_clean_df, enhanced_clean_df, enhanced_clean_df], ignore_index=True)

    return new_train_dirty_df, new_train_clean_df