# -*- coding: utf-8 -*-
"""
使用LLM进行数据增强和错误注入的模块
"""

import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import sys
import os
sys.path.append(os.path.abspath('../cllm'))
sys.path.append(os.path.abspath('.'))
from llm_gen2 import llm_gen
from dataset import Dataset
from llm_error_gen_batch import inject_errors_with_llm_batch




def create_data_enhancement_templates(dataset_name, example_df=None):
    """
    创建数据增强的模板
    
    参数:
    - dataset_name: 数据集名称
    - example_df: 示例数据DataFrame，用于获取列名
    
    返回:
    - prompt: ChatPromptTemplate对象
    - generator_template: 生成器模板字符串
    - format_instructions: 格式说明字符串
    """
    # 为数据集的每一列创建响应模式
    response_schemas = []
    
    # 如果有示例DataFrame，使用它的列名
    if example_df is not None:
        columns = list(example_df.columns)
    elif dataset_name == "hospital":
        columns = ["index", "provider_number", "name", "address_1", "address_2", "address_3",
                  "city", "state", "zip", "county", "phone", "type", "owner",
                  "emergency_service", "condition", "measure_code", "measure_name",
                  "score", "sample", "state_average"]
    else:
        # 默认情况下，使用通用列名
        columns = [f"col_{i}" for i in range(20)]
    
    for col in columns:
        resp = ResponseSchema(
            name=col,
            description=f"data column for {col}",
        )
        response_schemas.append(resp)
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # 创建数据增强模板
    generator_template = """\
    You are a synthetic data generator.
    Your goal is to generate enhanced clean data that maintains the same structure and distribution as the original data but provides diverse variations.
    
    I will give you clean data examples first
    
    Generate realistic but diverse variations of the data while ensuring:
    - All data remains clean and error-free
    - The overall structure and format are preserved
    - The distribution of values is similar to the original data
    - The generated data is diverse and not just copies of the original
    
    example data: {data}
    
    {format_instructions}
    
    DO NOT COPY THE EXAMPLES but generate new diverse samples that maintain the same quality and structure as the original clean data.
    """
    
    prompt = ChatPromptTemplate.from_template(template=generator_template)
    
    return prompt, generator_template, format_instructions


def enhance_clean_data_with_llm(clean_dataframe, dataset_name, api_details,
                               llm_serving='together', model='gpt-3.5-turbo',
                               n_samples=None, temperature=0.75, max_tokens=4096,
                               n_processes=10, ic_samples=20):
    """
    使用LLM增强干净数据
    
    参数:
    - clean_dataframe: 干净数据的DataFrame
    - dataset_name: 数据集名称
    - api_details: API详细信息
    - llm_serving: LLM服务类型
    - model: 使用的模型
    - n_samples: 生成样本数量，默认为原始数据集的两倍
    - temperature: 温度参数
    - max_tokens: 最大令牌数
    - n_processes: 进程数
    - ic_samples: 示例样本数
    
    返回:
    - 增强后的干净数据DataFrame
    """
    if n_samples is None:
        n_samples = len(clean_dataframe) * 2
    
    # 获取数据增强模板，传递示例DataFrame以获取正确的列名
    prompt, generator_template, format_instructions = create_data_enhancement_templates(dataset_name, example_df=clean_dataframe)
    
    # 使用LLM生成增强数据
    enhanced_data = llm_gen(
        prompt=prompt,
        generator_template=generator_template,
        format_instructions=format_instructions,
        example_df=clean_dataframe,
        llm_serving=llm_serving,
        api_details=api_details,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        n_processes=n_processes
    )
    
    return enhanced_data


def create_enhanced_representative_samples(representative_samples_dataset, n_samples=1000,
                                         api_details=None, llm_serving='together',
                                         model='gpt-3.5-turbo', ic_samples=20):
    """
    创建增强的代表性样本数据集
    
    参数:
    - representative_samples_dataset: 已经包含脏数据和干净数据的代表性样本Dataset实例
    - n_samples: 要生成的样本数量
    - api_details: API详细信息
    - llm_serving: LLM服务类型
    - model: 使用的模型
    - ic_samples: 示例样本数
    
    返回:
    - 增强后的代表性样本Dataset实例
    """
    if api_details is None:
        api_details = {
            "api_base": "https://sg.uiuiapi.com/v1",
            "api_version": "uiui",
            "api_key": "sk-pJXuMXNJJ0jin4umqzadRm1rADF1i7aRrxpd3GTvmsUDbUEw",
        }
    
    # 获取原始的干净数据和脏数据
    original_clean_data = representative_samples_dataset.clean_dataframe
    original_dirty_data = representative_samples_dataset.dataframe
    dataset_name = representative_samples_dataset.name
    
    print(f"Original samples count: {len(original_clean_data)}")
    
    # 计算需要生成的样本数量（总样本数 - 原始样本数）
    n_enhanced_samples = n_samples - len(original_clean_data)
    
    if n_enhanced_samples <= 0:
        print("No need to generate enhanced samples, using original samples only")
        return representative_samples_dataset
    
    print(f"Generating {n_enhanced_samples} enhanced samples to reach total of {n_samples} samples...")
    
    # 使用LLM生成指定数量的干净数据
    print("Generating clean data with LLM...")
    enhanced_clean_data = enhance_clean_data_with_llm(
        clean_dataframe=original_clean_data,
        dataset_name=dataset_name,
        api_details=api_details,
        llm_serving=llm_serving,
        model=model,
        n_samples=n_enhanced_samples,
        ic_samples=ic_samples
    )
    
    # 为生成的干净数据注入错误，确保一一对应
    print("Injecting errors with LLM using batch processing...")
    enhanced_dirty_data = inject_errors_with_llm_batch(
        clean_dataframe=enhanced_clean_data,
        dataset_name=dataset_name,
        api_details=api_details,
        llm_serving=llm_serving,
        model=model,
        batch_size=20,  # 每批处理20个样本
        original_sample_count=len(original_dirty_data),  # 只对增强部分注入错误
        original_dirty_data=original_dirty_data  # 传递原始脏数据用于拼接
    )
    
    # 确保增强的干净数据和脏数据行数相同
    if len(enhanced_clean_data) != len(enhanced_dirty_data):
        print(f"Warning: Enhanced clean data ({len(enhanced_clean_data)}) and dirty data ({len(enhanced_dirty_data)}) have different row counts")
        # 取较小的行数以确保一一对应
        min_rows = min(len(enhanced_clean_data), len(enhanced_dirty_data))
        enhanced_clean_data = enhanced_clean_data.iloc[:min_rows]
        enhanced_dirty_data = enhanced_dirty_data.iloc[:min_rows]
        print(f"Adjusted to {min_rows} rows to ensure correspondence")
    
    # 将原始数据和增强数据拼接在一起
    final_clean_data = pd.concat([original_clean_data, enhanced_clean_data], ignore_index=True)
    final_dirty_data = pd.concat([original_dirty_data, enhanced_dirty_data], ignore_index=True)
    
    # 确保最终数据集的行数等于n_samples
    if len(final_clean_data) > n_samples:
        final_clean_data = final_clean_data.iloc[:n_samples]
        final_dirty_data = final_dirty_data.iloc[:n_samples]
    
    print(f"Final dataset size: {len(final_clean_data)} samples")
    
    # 创建新的Dataset实例
    enhanced_dataset_dict = {
        "name": f"enhanced_{representative_samples_dataset.name}",
        "path": representative_samples_dataset.path  # 使用原始路径，但实际数据将被替换
    }
    enhanced_dataset = Dataset(enhanced_dataset_dict)
    enhanced_dataset.dataframe = final_dirty_data
    enhanced_dataset.clean_dataframe = final_clean_data
    
    # 保存增强的干净数据和脏数据到CSV文件
    # 获取数据集名称和路径
    dataset_name = representative_samples_dataset.name
    base_path = os.path.dirname(representative_samples_dataset.path)
    
    # 创建保存路径
    clean_data_path = os.path.join(base_path, f"enhanced_clean_data_{dataset_name}.csv")
    dirty_data_path = os.path.join(base_path, f"enhanced_dirty_data_{dataset_name}.csv")
    
    # 保存数据到CSV文件
    print(f"Saving enhanced clean data to: {clean_data_path}")
    final_clean_data.to_csv(clean_data_path, index=False)
    
    print(f"Saving enhanced dirty data to: {dirty_data_path}")
    final_dirty_data.to_csv(dirty_data_path, index=False)
    
    print(f"Enhanced data saved successfully")
    
    return enhanced_dataset