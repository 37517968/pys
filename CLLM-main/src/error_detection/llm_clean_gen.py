# -*- coding: utf-8 -*-
"""
使用LLM进行干净数据增强的模块
"""

import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import sys
import os
# sys.path.append(os.path.abspath('../cllm'))
# sys.path.append(os.path.abspath('.'))

# from ..cllm.llm_gen2 import llm_gen
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.cllm.llm_gen2 import llm_gen

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
                               n_processes=10):
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
    
    返回:
    - 增强后的干净数据DataFrame
    """
    
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
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        n_processes=n_processes
    )
    
    return enhanced_data


def main():
    """
    主函数，用于生成增强的干净数据
    """
    # 设置输入和输出文件路径
    input_csv_path = "../../data/error_detection/hospital/representative_samples/euclidean.csv"
    output_csv_path = "../../data/error_detection/hospital/enhanced_data/enhanced_clean_data_hospital_1000.csv"
    
    # 设置要生成的数据数量
    n_samples = 1000
    
    # 设置其他参数
    dataset_name = "hospital"
    model = "gpt-3.5-turbo"
    temperature = 0.75
    max_tokens = 4096
    n_processes = 10
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_path):
        print(f"错误: 输入文件 '{input_csv_path}' 不存在")
        return
    
    try:
        # 读取输入CSV文件
        print(f"正在读取输入文件: {input_csv_path}")
        clean_dataframe = pd.read_csv(input_csv_path)
        print(f"成功读取 {len(clean_dataframe)} 行数据")
        
        # 设置API详细信息 (这里使用默认值，实际使用时可能需要根据实际情况修改)
        api_details = {
            "api_key": "sk-yys1JSCIegBHgSXi3u6HUnn9H5E7ecblGSiOAnLy1wpxYcMP",  # 请替换为实际的API密钥
            "api_base": "https://api.chatanywhere.tech"  # 请根据实际使用的LLM服务进行修改
        }
        
        print(f"开始生成 {n_samples} 条增强的干净数据...")
        print(f"使用模型: {model}")
        print(f"温度参数: {temperature}")
        
        # 生成增强数据
        enhanced_data = enhance_clean_data_with_llm(
            clean_dataframe=clean_dataframe,
            dataset_name=dataset_name,
            api_details=api_details,
            model=model,
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            n_processes=n_processes
        )
        
        # 保存增强数据到CSV文件
        print(f"正在保存增强数据到: {output_csv_path}")
        enhanced_data.to_csv(output_csv_path, index=False)
        print(f"成功生成并保存 {len(enhanced_data)} 条增强数据")
        
    except Exception as e:
        print(f"生成增强数据时发生错误: {str(e)}")


if __name__ == "__main__":
    main()