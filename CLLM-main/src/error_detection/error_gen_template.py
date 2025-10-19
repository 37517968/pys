# error_gen_template.py
# -*- coding: utf-8 -*-
"""
随机注入多种错误到干净数据，用于生成脏数据样本
"""

import pandas as pd
import numpy as np
import random
import string

def random_string_typo(s, max_changes=1):
    """在字符串中随机插入、替换或删除字符"""
    if not isinstance(s, str) or s == "":
        return s
    s = list(s)
    for _ in range(random.randint(1, max_changes)):
        op = random.choice(["insert", "delete", "replace"])
        idx = random.randint(0, len(s) - 1)
        if op == "insert":
            s.insert(idx, random.choice(string.ascii_letters))
        elif op == "delete":
            s.pop(idx)
        elif op == "replace":
            s[idx] = random.choice(string.ascii_letters)
    return "".join(s)

def random_numeric_shift(n, max_shift=10):
    """对数字进行随机加减"""
    try:
        num = float(n)
        shift = random.uniform(-max_shift, max_shift)
        return num + shift
    except:
        return n

def inject_errors_to_dataframe(df, error_prob=0.2, numeric_shift=10):
    """
    对 DataFrame 随机注入错误
    - error_prob: 每个单元格注入错误的概率
    - numeric_shift: 数字列最大偏移值
    """
    df_err = df.copy()
    for col in df_err.columns:
        if col.lower() == "index":  # 跳过 index 列
            continue
        for i in range(len(df_err)):
            if random.random() < error_prob:
                val = df_err.at[i, col]
                # 根据类型随机选择错误方式
                if isinstance(val, str):
                    df_err.at[i, col] = random_string_typo(val)
                elif isinstance(val, (int, float)):
                    df_err.at[i, col] = random_numeric_shift(val, numeric_shift)
                else:
                    # 对其他类型随机置空
                    df_err.at[i, col] = None
    return df_err

def generate_error_csv(input_clean_csv, output_dirty_csv,
                       error_prob=0.2, numeric_shift=10):
    """
    从干净数据 CSV 生成脏数据 CSV
    """
    df_clean = pd.read_csv(input_clean_csv, dtype=object)
    df_dirty = inject_errors_to_dataframe(df_clean, error_prob, numeric_shift)
    df_dirty.to_csv(output_dirty_csv, index=False)
    print(f"✅ Generated dirty data saved to {output_dirty_csv}")
    return df_dirty

# === 测试运行 ===
if __name__ == "__main__":
    input_csv = "/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_clean_data_hospital_1000.csv"
    output_csv = "/home/stu/pys/CLLM-main/data/error_detection/hospital/enhanced_data/enhanced_dirty_data_hospital_1000.csv"
    generate_error_csv(input_csv, output_csv, error_prob=0.2, numeric_shift=5)
