import pandas as pd
import numpy as np
import random
import os

def replace_char_with_x(text, probability=0.01):
    """
    以给定的概率将文本中的任意字符替换为'x'
    
    Args:
        text (str): 输入文本
        probability (float): 替换概率，默认为0.01（1%）
    
    Returns:
        str: 处理后的文本
    """
    if pd.isna(text) or text == '':
        return text
    
    text = str(text)
    result = []
    
    for char in text:
        # 跳过空格和标点符号，只替换字母和数字
        if char.isalnum() and random.random() < probability:
            result.append('x')
        else:
            result.append(char)
    
    return ''.join(result)

def generate_errors_in_csv(input_file, output_file, error_probability=0.01):
    """
    在CSV文件中生成错误，以给定的概率将任意字符替换为'x'
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        error_probability (float): 错误概率，默认为0.01（1%）
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"原始数据形状: {df.shape}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
    
    # 复制数据框，避免修改原始数据
    error_df = df.copy()
    
    # 统计总字符数和错误数
    total_chars = 0
    error_count = 0
    
    # 遍历每个单元格
    for row_idx in range(error_df.shape[0]):
        for col_idx in range(error_df.shape[1]):
            original_value = error_df.iloc[row_idx, col_idx]
            
            if pd.isna(original_value) or original_value == '':
                continue
            
            original_str = str(original_value)
            total_chars += len(original_str)
            
            # 应用错误生成函数
            error_value = replace_char_with_x(original_str, error_probability)
            
            # 统计错误数
            for i, (orig_char, error_char) in enumerate(zip(original_str, error_value)):
                if orig_char != error_char:
                    error_count += 1
            
            # 更新数据框
            error_df.iloc[row_idx, col_idx] = error_value
    
    # 保存处理后的CSV文件
    try:
        error_df.to_csv(output_file, index=False)
        print(f"成功保存带有错误的文件: {output_file}")
        print(f"总字符数: {total_chars}")
        print(f"错误字符数: {error_count}")
        print(f"实际错误率: {error_count / total_chars * 100:.2f}%")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

def main():
    # 文件路径
    base_dir = "../../data/error_detection/hospital"
    clean_file = os.path.join(base_dir, "enhanced_data/enhanced_clean_data_hospital_1000.csv")
    output_file = os.path.join(base_dir, "enhanced_data/enhanced_dirty_data_hospital_1000_x.csv")
    
    # 错误概率（1%）
    error_probability = 0.01
    
    print("开始生成错误数据...")
    print(f"输入文件: {clean_file}")
    print(f"输出文件: {output_file}")
    print(f"错误概率: {error_probability * 100}%")
    
    # 生成错误数据
    generate_errors_in_csv(clean_file, output_file, error_probability)
    
    print("\n错误数据生成完成！")

if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    main()