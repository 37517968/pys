# -*- coding: utf-8 -*-
"""
使用LLM进行批量错误注入的模块，确保一一对应关系
"""

import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
# sys.path.append(os.path.abspath('../cllm'))
# sys.path.append(os.path.abspath('.'))
from src.cllm.llm_gen2 import llm_gen
from src.error_detection.dataset import Dataset


def create_error_injection_templates_batch(dataset_name, example_df=None):
    """
    创建批量错误注入的模板
    
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
        columns = ['index', 'provider number', 'hospital name', 'address1', 
        'address2', 'address3', 'city', 'state', 'zip code', 'county name', 
        'phone number', 'hospital type', 'hospital owner', 'emergency service', 
        'condition', 'measure code', 'measure name', 'score', 'sample', 'stateavg']
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
    
    # 创建批量错误注入模板
    generator_template = """\
    You are a data error generator.
    Your goal is to introduce realistic errors into the given clean data while maintaining the overall structure and distribution of the data.
    
    I will give you clean data examples. You must generate EXACTLY the same number of erroneous versions as the input examples, in the same order.

    For example, if I provide 20 data examples, you must generate exactly 3 erroneous versions, one for each input example.

    Generate realistic but erroneous versions of the data by introducing common types of errors such as:
    - Typos and spelling mistakes in text fields
    - Incorrect formatting (e.g., missing digits in zip codes, wrong phone number format)
    - Transposed digits or characters
    - Omissions or additions of characters
    - Incorrect values that are still plausible (e.g., wrong but valid city name in the same state)
    - Rounding or approximation errors in numeric fields
    
    example data: {data}
    
    {format_instructions}
    
    CRITICAL REQUIREMENTS:
    1. Generate EXACTLY ONE erroneous version for EACH input example
    2. Maintain the exact same order as the input examples
    3. Generate the exact same number of outputs as inputs
    4. Do not skip any input examples
    5. Do not generate extra examples beyond the input count
    6. IMPORTANT: Return ONLY the JSON response without any additional text or explanations
    7. Each output row must correspond to the input row at the same position
    
    DO NOT COPY THE EXAMPLES but generate new samples with realistic errors that could occur in real-world data entry processes.
    Ensure that the generated data has the same structure and similar distribution as the original data.
    """
    
    prompt = ChatPromptTemplate.from_template(template=generator_template)
    
    return prompt, generator_template, format_instructions


def inject_errors_with_llm_batch(clean_dataframe, dataset_name, api_details,
                                llm_serving='together', model='gpt-3.5-turbo',
                                temperature=0.75, max_tokens=4096,
                                batch_size=20, original_dirty_data=None):
    """
    使用LLM批量注入错误，确保一一对应关系
    
    参数:
    - clean_dataframe: 增强的干净数据的DataFrame
    - dataset_name: 数据集名称
    - api_details: API详细信息
    - llm_serving: LLM服务类型
    - model: 使用的模型
    - temperature: 温度参数
    - max_tokens: 最大令牌数
    - batch_size: 每批处理的样本数量
    - original_dirty_data: 原始的脏数据DataFrame，用于拼接
    
    返回:
    - 包含错误的DataFrame
    """
    # 获取错误注入模板，传递示例DataFrame以获取正确的列名
    prompt, generator_template, format_instructions = create_error_injection_templates_batch(dataset_name, example_df=original_dirty_data)
    
    print(f"Injecting errors for all {len(clean_dataframe)} samples in batches")
    

    # 分批处理数据
    all_error_data = []
    
    for i in range(0, len(clean_dataframe), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(clean_dataframe))
        batch_data = clean_dataframe.iloc[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start} to {batch_end-1}")
        
        # 使用LLM为这批数据生成包含错误的版本
        batch_error_data = llm_gen(
            prompt=prompt,
            generator_template=generator_template,
            format_instructions=format_instructions,
            example_df=batch_data,
            llm_serving=llm_serving,
            api_details=api_details,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            n_processes=1,  # 使用单进程以确保顺序
        )
        
        # 确保生成的错误数据与批次数据行数相同
        if len(batch_error_data) != len(batch_data):
            print(f"Warning: Generated {len(batch_error_data)} error samples for batch, but expected {len(batch_data)}")
            # 调整行数以匹配
            if len(batch_error_data) > len(batch_data):
                batch_error_data = batch_error_data.iloc[:len(batch_data)]
            else:
                # 如果生成的数据不足，使用原始数据填充缺失的行
                missing_rows = len(batch_data) - len(batch_error_data)
                print(f"Warning: Filling {missing_rows} missing rows with original data")
                original_rows = batch_data.iloc[len(batch_error_data):]
                batch_error_data = pd.concat([batch_error_data, original_rows], ignore_index=True)
        
        # 确保列名与原始数据一致
        if set(batch_error_data.columns) != set(batch_data.columns):
            print(f"Warning: Column mismatch. Expected: {list(batch_data.columns)}, Got: {list(batch_error_data.columns)}")
            # 只保留原始数据中存在的列
            batch_error_data = batch_error_data[batch_data.columns]
        
        # 添加行对应验证
        batch_error_data.index = batch_data.index  # 确保索引一致
        
        all_error_data.append(batch_error_data)
    
    # 合并所有批次的错误数据
    error_data = pd.concat(all_error_data, ignore_index=True)
    
    # 最终验证：确保错误数据与原始数据行数一致
    if len(error_data) != len(clean_dataframe):
        print(f"Critical Warning: Final error data count ({len(error_data)}) does not match clean data count ({len(clean_dataframe)})")
        # 如果行数不匹配，使用原始数据填充
        if len(error_data) < len(clean_dataframe):
            missing_rows = len(clean_dataframe) - len(error_data)
            print(f"Warning: Filling {missing_rows} missing rows with original clean data")
            missing_data = clean_dataframe.iloc[len(error_data):]
            error_data = pd.concat([error_data, missing_data], ignore_index=True)
        else:
            # 如果错误数据多于原始数据，截断
            error_data = error_data.iloc[:len(clean_dataframe)]
    
    print(f"Created error data with {len(error_data)} samples")
    
    # 验证行对应关系
    print("Validating row correspondence...")
    for i in range(min(5, len(clean_dataframe))):  # 检查前5行
        clean_row = clean_dataframe.iloc[i]
        error_row = error_data.iloc[i]
        # 检查是否有明显的差异（表明确实注入了错误）
        differences = sum(1 for j in range(len(clean_row)) if clean_row.iloc[j] != error_row.iloc[j])
        print(f"Row {i}: {differences} differences between clean and error data")
    
    return error_data

if __name__ == "__main__":
    aug_error_data = inject_errors_with_llm_batch(
        clean_dataframe=pd.read_csv("/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_clean_data_hospital_1000.csv", header=0, dtype=object),
        dataset_name="hospital",
        api_details = {
            "api_base": "https://api.chatanywhere.tech",
            "api_version": "free",
            "api_key": "sk-yys1JSCIegBHgSXi3u6HUnn9H5E7ecblGSiOAnLy1wpxYcMP",
        },
        batch_size=10,
        llm_serving='together',
        model='gpt-4',)
    aug_error_data.to_csv("/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_dirty_data_hospital_1000.csv", index=False)