# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import subprocess
import tempfile
import pickle

def run_in_cllm_env(module_path, func_name, *args, **kwargs):
    """
    在 cllm_env conda 环境中运行指定的函数
    
    参数:
    module_path: 要导入的模块路径，例如 'error_detection.cluster'
    func_name: 要运行的函数名
    *args: 函数的位置参数
    **kwargs: 函数的关键字参数
    
    返回:
    函数执行结果，自动根据返回值数量处理单个或多个 DataFrame
    """
    # 处理输入参数，将 DataFrame 保存为临时 CSV 文件
    temp_files = []
    processed_args = []
    
    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            # 为 DataFrame 创建临时 CSV 文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_arg{0}.csv'.format(i)) as csv_file:
                arg.to_csv(csv_file.name, index=False)
                temp_files.append(csv_file.name)
                processed_args.append("pd.read_csv('{0}')".format(csv_file.name))
        else:
            processed_args.append(repr(arg))
    
    # 处理关键字参数中的 DataFrame
    processed_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, pd.DataFrame):
            # 为 DataFrame 创建临时 CSV 文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_kwarg_{0}.csv'.format(k)) as csv_file:
                v.to_csv(csv_file.name, index=False)
                temp_files.append(csv_file.name)
                processed_kwargs[k] = "pd.read_csv('{0}')".format(csv_file.name)
        else:
            processed_kwargs[k] = repr(v)
    
    # 创建一个 Python 脚本来在 cllm_env 环境中运行
    script_lines = [
        "import sys, os, pandas as pd",
        "sys.path.append('/home/stu/pys/CllmDetection/CLLM-main/src')",  # 更新为实际路径
        "module = __import__('{module_path}', fromlist=[''])".format(module_path=module_path),
        "func = getattr(module, '{func_name}')".format(func_name=func_name),
    ]
    
    # 添加位置参数和关键字参数加载代码
    for i, arg_code in enumerate(processed_args):
        script_lines.append("arg{0} = {1}".format(i, arg_code))
    for k, v_code in processed_kwargs.items():
        script_lines.append("{0} = {1}".format(k, v_code))
    
    # 准备函数调用
    script_lines.append("args = [" + ", ".join(["arg{0}".format(i) for i in range(len(processed_args))]) + "]")
    script_lines.append("kwargs = {" + ", ".join(["'{0}': {1}".format(k, v) for k, v in processed_kwargs.items()]) + "}")
    
    # 运行函数
    script_lines.append("result = func(*args, **kwargs)")
    
    # 根据返回结果是否是 tuple 来决定如何处理
    script_lines.append("if isinstance(result, tuple):")
    script_lines.append("    result_files = []")
    script_lines.append("    for i, r in enumerate(result):")
    script_lines.append("        result_file = '/tmp/result_{0}.csv'.format(i)")
    script_lines.append("        r.to_csv(result_file, index=False)")
    script_lines.append("        result_files.append(result_file)")
    script_lines.append("else:")
    script_lines.append("    result_file = '/tmp/result.csv'")
    script_lines.append("    result.to_csv(result_file, index=False)")
    
    script_content = '\n'.join(script_lines)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as script_file:
        script_file.write(script_content)
        script_file_path = script_file.name
        temp_files.append(script_file_path)
    
    # 初始化result_files变量，确保在finally块中可用
    result_files = []
    
    try:
        # 在 cllm_env 环境中运行脚本
        subprocess.call([  # 使用 call 替代 check_call
            'conda', 'run', '-n', 'cllm_env',
            'python', script_file_path
        ])
        
        # 读取结果文件
        if os.path.exists('/tmp/result.csv'):
            result_files.append('/tmp/result.csv')
        for i in range(10):  # 如果有多个文件，最多读取 10 个
            result_file = '/tmp/result_{0}.csv'.format(i)
            if os.path.exists(result_file):
                result_files.append(result_file)
            else:
                break
        
        # 根据结果文件个数返回单个或多个 DataFrame
        if len(result_files) == 1:
            return pd.read_csv(result_files[0])
        else:
            results = [pd.read_csv(file) for file in result_files]
            return tuple(results)
    
    finally:
        # 删除临时文件
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        # 删除结果文件
        for file_path in result_files:
            if os.path.exists(file_path):
                os.unlink(file_path)




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
    dirty_representative, clean_representative = run_in_cllm_env(
        'error_detection.cluster',
        'get_representative_samples_with_clean',
        dirty_df,
        clean_df,
        n_clusters=cluster_number,
        similarity_metric='euclidean'
    )
    error_analysis = run_in_cllm_env(
        'error_detection.llm_aug',
        'analyse_data_error',
        dirty_representative,
        clean_representative,
        save_path = "/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/error.csv"
    )
    ontology_df = run_in_cllm_env(
        'error_detection.llm_aug',
        'analyse_table_ontology',
        clean_representative,
        save_path = "/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/ontology.csv"
    )
    enhanced_clean_df = run_in_cllm_env(
        'error_detection.llm_aug',
        'enhance_clean_data',
        clean_dataframe=clean_representative,
        ontology_dataframe=ontology_df,
        save_path = "/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_clean.csv",
        number=200,
    )
    print("DEBUG: Starting inject_errors function...")
    enhanced_dirty_df = run_in_cllm_env(
        'error_detection.llm_aug',
        'inject_errors',
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
    error_analysis = run_in_cllm_env(
        'error_detection.llm_aug',
        'analyse_data_error_again',
        example_dirty_df,
        example_clean_df,
        error_analysis,
        save_path = "/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/error.csv"
    )
    enhanced_clean_df = run_in_cllm_env(
        'error_detection.llm_aug',
        'enhance_clean_data',
        clean_dataframe=example_clean_df,
        ontology_dataframe=ontology_df,
        save_path = "/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_clean.csv",
        number=train_number,
    )
    print("DEBUG: Starting inject_errors function...")
    enhanced_dirty_df = run_in_cllm_env(
        'error_detection.llm_aug',
        'inject_errors',
        clean_dataframe=enhanced_clean_df,
        error_analysis=error_analysis,
        batch_size=20,
        save_path="/home/stu/pys/CllmDetection/ExampleDrivenErrorDetection-master/datasets/HOSP_HoloClean/enhanced_data/enhanced_dirty.csv",
    )

    new_train_dirty_df = pd.concat([example_dirty_df, example_clean_df, enhanced_dirty_df, enhanced_clean_df], ignore_index=True)

    new_train_clean_df = pd.concat([example_clean_df, example_clean_df, enhanced_clean_df, enhanced_clean_df], ignore_index=True)

    return new_train_dirty_df, new_train_clean_df