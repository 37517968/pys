import pandas as pd
import numpy as np
import os

def compare_csv_files(clean_file_path, dirty_file_path):
    """
    比较两个CSV文件的每个单元格，并输出不一样的单元格
    
    Args:
        clean_file_path (str): 干净数据文件的路径
        dirty_file_path (str): 有错误数据文件的路径
    
    Returns:
        dict: 包含不同单元格信息的字典
    """
    # 读取CSV文件
    try:
        clean_df = pd.read_csv(clean_file_path)
        dirty_df = pd.read_csv(dirty_file_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
    # 检查两个数据框的形状是否相同
    if clean_df.shape != dirty_df.shape:
        print(f"警告: 两个文件的形状不同。干净文件: {clean_df.shape}, 有错误文件: {dirty_df.shape}")
    
    # 获取共同的行数和列数
    min_rows = min(clean_df.shape[0], dirty_df.shape[0])
    min_cols = min(clean_df.shape[1], dirty_df.shape[1])
    
    # 获取列名
    clean_columns = clean_df.columns.tolist()
    dirty_columns = dirty_df.columns.tolist()
    
    # 比较每个单元格
    differences = []
    
    for row_idx in range(min_rows):
        for col_idx in range(min_cols):
            clean_value = clean_df.iloc[row_idx, col_idx]
            dirty_value = dirty_df.iloc[row_idx, col_idx]
            
            # 处理NaN值
            if pd.isna(clean_value) and pd.isna(dirty_value):
                continue
            
            # 将值转换为字符串进行比较，避免类型不匹配
            clean_str = str(clean_value).strip()
            dirty_str = str(dirty_value).strip()
            
            # 比较值
            if clean_str != dirty_str:
                difference_info = {
                    'row_index': row_idx,
                    'column_index': col_idx,
                    'column_name': clean_columns[col_idx] if col_idx < len(clean_columns) else dirty_columns[col_idx],
                    'clean_value': clean_str,
                    'dirty_value': dirty_str
                }
                differences.append(difference_info)
    
    return {
        'total_differences': len(differences),
        'differences': differences
    }

def print_differences(differences_info):
    """
    打印不同的单元格信息
    
    Args:
        differences_info (dict): 包含不同单元格信息的字典
    """
    if not differences_info:
        print("没有找到差异信息。")
        return
    
    print(f"总共找到 {differences_info['total_differences']} 个不同的单元格。")
    print("\n差异详情:")
    print("-" * 100)
    print(f"{'行号':<8} {'列号':<8} {'列名':<25} {'干净值':<30} {'错误值':<30}")
    print("-" * 100)
    
    for diff in differences_info['differences']:
        row_idx = diff['row_index']
        col_idx = diff['column_index']
        col_name = diff['column_name']
        clean_val = str(diff['clean_value'])[:28] + "..." if len(str(diff['clean_value'])) > 28 else str(diff['clean_value'])
        dirty_val = str(diff['dirty_value'])[:28] + "..." if len(str(diff['dirty_value'])) > 28 else str(diff['dirty_value'])
        
        print(f"{row_idx:<8} {col_idx:<8} {col_name:<25} {clean_val:<30} {dirty_val:<30}")

def save_differences_to_csv(differences_info, output_file):
    """
    将差异信息保存到CSV文件
    
    Args:
        differences_info (dict): 包含不同单元格信息的字典
        output_file (str): 输出文件路径
    """
    if not differences_info or not differences_info['differences']:
        print("没有差异信息可保存。")
        return
    
    # 创建数据框
    diff_df = pd.DataFrame(differences_info['differences'])
    
    # 保存到CSV
    try:
        diff_df.to_csv(output_file, index=False)
        print(f"差异信息已保存到: {output_file}")
    except Exception as e:
        print(f"保存差异信息时发生错误: {e}")

def main():
    # 文件路径
    base_dir = "../../data/error_detection/hospital/enhanced_data"
    # clean_file = os.path.join(base_dir, "enhanced_data/enhanced_clean_data_hospital_100.csv")
    # dirty_file = os.path.join(base_dir, "enhanced_data/enhanced_dirty_data_hospital_100_x.csv")
    clean_file = os.path.join(base_dir, "enhanced_clean_data_hospital_1000.csv")
    dirty_file = os.path.join(base_dir, "enhanced_dirty_data_hospital_1000.csv")
    output_file = os.path.join(base_dir, "clean_dirty_differences_enhanced.csv")
    
    print("开始比较CSV文件...")
    print(f"干净文件: {clean_file}")
    print(f"有错误文件: {dirty_file}")
    
    # 比较文件
    differences_info = compare_csv_files(clean_file, dirty_file)
    
    if differences_info:
        # 打印差异
        print_differences(differences_info)
        
        # 保存差异到CSV
        save_differences_to_csv(differences_info, output_file)
        
        # 统计每列的差异数量
        if differences_info['differences']:
            diff_df = pd.DataFrame(differences_info['differences'])
            column_diff_counts = diff_df['column_name'].value_counts()
            print("\n各列差异统计:")
            print("-" * 40)
            for col, count in column_diff_counts.items():
                print(f"{col:<30}: {count}")
    else:
        print("比较过程中出现错误，未能完成比较。")

if __name__ == "__main__":
    main()