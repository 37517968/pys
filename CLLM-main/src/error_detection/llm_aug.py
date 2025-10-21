# -*- coding: utf-8 -*-
"""
使用LLM进行数据增强的模块，包括创建增强的代表性样本数据集
"""
import argparse
from copy import error
import pandas as pd
import numpy as np
import sys
import os
import time
# sys.path.append(os.path.abspath('../cllm'))
# sys.path.append(os.path.abspath('.'))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.cllm.llm_gen2 import llm_gen_error_analysis
from src.cllm.llm_gen2 import llm_gen_table_ontology
from src.cllm.llm_gen2 import llm_gen_enhanced_clean_data
from src.cllm.llm_gen2 import llm_gen_inject_errors
from src.cllm.llm_gen2 import llm_gen_error_analysis_again
from .llm_clean_gen import enhance_clean_data_with_llm
from src.error_detection.llm_error_gen_one_to_one import inject_errors_with_llm_one_to_one
from src.error_detection.llm_error_gen_batch import inject_errors_with_llm_batch
from src.error_detection.dataset import Dataset
# from llm_gen2 import llm_gen_error_analysis
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

api_details = {
            "api_base": "https://jeniya.cn/v1",
            "api_version": "free",
            "api_key": "sk-9SSJQSfSLf7aUtfTCMhCgsxiw5deylXvOUuHtPCDDB3Sa1ds",
        }
model = "gpt-4o-mini"
llm_serving = "together"

def analyse_data_error(dataframe, clean_dataframe, save_path = None):
    """
    创建用于错误类型分析的固定模板
    返回:
    - prompt: ChatPromptTemplate对象
    - generator_template: 模板字符串
    - format_instructions: 格式说明字符串
    """
    # 定义输出结构（错误类型、受影响列、描述）
    response_schemas = [
        ResponseSchema(name="error_type", description="错误的类别，例如拼写错误、格式错误、缺失值、数值异常等"),
        ResponseSchema(name="affected_columns", description="涉及的列名列表"),
        ResponseSchema(name="error_description", description="该错误类型的详细说明和典型表现形式"),
    ]

    # 创建结构化输出解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # 固定模板
    generator_template = """\
You are a data error analyst.
Your task is to analyze the differences between clean and dirty data examples and identify the major error types that appear.

I will give you two sets of data:
- Clean samples: correct data without errors
- Dirty samples: corresponding data with errors

Please analyze what kinds of errors appear, which columns are affected, and describe their characteristics.

Repair data examples:
{data}

{format_instructions}

Your analysis should summarize recurring error patterns across the samples.
Do not simply list differences cell by cell, but generalize them into error categories.
"""

    # 创建可用于LLM的Prompt对象
    prompt = ChatPromptTemplate.from_template(generator_template)
    error_analysis = llm_gen_error_analysis(    
        prompt=prompt,
        generator_template=generator_template,
        clean_df=clean_dataframe,
        dirty_df=dataframe,
        api_details=api_details,
        format_instructions=format_instructions,
        llm_serving='together',
        model=model,
    )
    if not save_path is None:
        error_analysis.to_csv(save_path, index=False)
        print(f"Error analysis saved to: {save_path}")
    
    return error_analysis

def analyse_data_error_again(dataframe, clean_dataframe, error_analysis, save_path = None):
    """
    创建用于错误类型分析的固定模板
    返回:
    - prompt: ChatPromptTemplate对象
    - generator_template: 模板字符串
    - format_instructions: 格式说明字符串
    """
    # 定义输出结构（错误类型、受影响列、描述）
    response_schemas = [
        ResponseSchema(name="error_type", description="错误的类别，例如拼写错误、格式错误、缺失值、数值异常等"),
        ResponseSchema(name="affected_columns", description="涉及的列名列表"),
        ResponseSchema(name="error_description", description="该错误类型的详细说明和典型表现形式"),
    ]

    # 创建结构化输出解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # 固定模板
    generator_template = """\
You are a data error analyst.

You have an existing error analysis table that summarizes previously discovered error types in a dataset.
Now, you are provided with *new repair data examples* (dirty vs. clean pairs).
Your goal is to **update** the existing analysis:
- Add **new error types** that were not previously identified.
- Modify existing descriptions **only when clearly more accurate or consistent** with the new samples.
- Keep all other existing error types unchanged.

Inputs:
1. Previous error analysis:
{old_error_analysis}

2. Repair data examples (dirty vs. clean):
{data}

{format_instructions}

Output a concise, integrated updated error analysis table.
Avoid duplication of similar error types; merge when applicable.
"""

    # 创建可用于LLM的Prompt对象
    prompt = ChatPromptTemplate.from_template(generator_template)
    error_analysis = llm_gen_error_analysis_again(    
        prompt=prompt,
        generator_template=generator_template,
        clean_df=clean_dataframe,
        dirty_df=dataframe,
        old_error_analysis=error_analysis,
        api_details=api_details,
        format_instructions=format_instructions,
        llm_serving='together',
        model=model,
    )
    if not save_path is None:
        error_analysis.to_csv(save_path, index=False)
        print(f"Error analysis saved to: {save_path}")
    
    return error_analysis

def analyse_table_ontology(dataframe, save_path=None):
    """
    创建用于分析表格结构和列间关系（Ontology Tree）的固定模板。
    
    参数:
    - dataframe: 输入的DataFrame
    - save_path: 如果提供，将结果保存为文件
    
    返回:
    - ontology_result: DataFrame格式的本体树结果
    """
    # 定义输出结构
    response_schemas = [
        ResponseSchema(name="column_name", description="当前列的名称"),
        ResponseSchema(name="related_columns", description="与该列存在语义或统计关系的列列表"),
        ResponseSchema(name="relation_type", description="列之间的关系类型，例如：依赖、层级、聚合、外键、同义字段等"),
        ResponseSchema(name="relation_description", description="这种关系的简短说明，解释列间的语义或统计联系"),
    ]

    # 结构化解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # === 固定模板 ===
    generator_template = """\
You are a data ontology constructor.
Your goal is to analyze the given tabular data and infer relationships among its columns.

I will provide a small table sample with column names and several rows of data.

Please analyze the following:
1. What kinds of relationships exist among columns (hierarchical, dependency, correlation, aggregation, etc.)
2. For each column, identify its related columns and describe how they are related.
3. Summarize relationships as structured ontology entries.

Table sample:
{data}

{format_instructions}

Important:
- Output should describe relationships among columns, not row-level values.
- Be concise but informative in your relation_description.
- Do not invent columns not present in the input.
"""

    # === 创建 Prompt ===
    prompt = ChatPromptTemplate.from_template(generator_template)

    # === 调用LLM生成本体树 ===
    ontology_result = llm_gen_table_ontology(
        prompt=prompt,
        generator_template=generator_template,
        dataframe=dataframe,
        format_instructions=format_instructions,
        llm_serving="together",
        api_details=api_details,
        model=model
    )

    # === 保存结果 ===
    if save_path is not None:
        ontology_result.to_csv(save_path, index=False)
        print(f"✅ Ontology analysis saved to: {save_path}")

    return ontology_result

def enhance_clean_data(clean_dataframe, ontology_dataframe, number=200, save_path=None):
    """
    使用LLM根据表格本体树和示例干净数据生成增强的干净数据
    
    参数:
    - clean_dataframe: 原始干净数据 DataFrame
    - ontology_dataframe: 表格本体树 DataFrame（包含列关系与语义信息）
    - number: 要生成的增强样本数量
    - save_path: 保存生成数据的CSV路径（可选）
    
    返回:
    - enhanced_clean_df: 增强后的干净数据 DataFrame
    """
    # === Step 1. 定义输出格式 ===
    response_schemas = [
        ResponseSchema(name=col, description=f"Synthetic clean value for column '{col}'")
        for col in clean_dataframe.columns
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # === Step 2. 构造生成模板 ===
    generator_template = """\
You are a professional data synthesizer.
Your goal is to generate **new clean tabular data samples** that are realistic and consistent with the given data schema and ontology.

### Given:
- Example clean data samples: {data}
- Table ontology describing column meanings and relations: {ontology}

### Instructions:
- Maintain data realism and column relationships as defined in ontology.
- Keep data free from errors, missing values, or contradictions.
- Ensure distribution and formats are consistent with given examples.
- Do NOT repeat the same rows from examples — generate novel but valid clean data.
- Return {number} rows of new clean data.

{format_instructions}

Generate enhanced clean tabular data that respects the ontology and examples.
Output exactly the structured clean data records in JSON format.
"""

    prompt = ChatPromptTemplate.from_template(template=generator_template)

    all_results = []
    generated_count = 0
    iteration = 0

    # === Step 3. 调用LLM生成增强数据 ===
    while generated_count < number:
        iteration += 1
        remaining = number - generated_count
        current_batch = min(100, remaining)

        print(f"🚀 Iteration {iteration}: Generating {current_batch} new clean samples "
              f"({generated_count}/{number} done)...")

        df_batch = llm_gen_enhanced_clean_data(
            prompt=prompt,
            generator_template=generator_template,
            format_instructions=format_instructions,
            clean_df=clean_dataframe,
            ontology_df=ontology_dataframe,
            llm_serving='together',
            api_details=api_details,
            number=current_batch,
            model=model,
        )

        # 清理空结果
        if df_batch is not None and not df_batch.empty:
            all_results.append(df_batch)
            generated_count += len(df_batch)
        else:
            print("⚠️ LLM returned no valid data, retrying...")

        time.sleep(2)

    # === Step 5. 合并所有批次 ===
    enhanced_df = pd.concat(all_results, ignore_index=True)

    # === Step 4. 保存结果 ===
    if save_path is not None:
        enhanced_df.to_csv(save_path, index=False)
        print(f"✅ Enhanced clean data saved to: {save_path}")

    print(f"🎯 Successfully generated {len(enhanced_df)} enhanced clean samples.")
    return enhanced_df

def inject_errors(clean_dataframe, error_analysis, batch_size=10, save_path=None):
    """
    循环调用 LLM，根据 error_analysis 中的错误模式对 clean_dataframe 注入相似错误。
    生成对应数量的脏数据 DataFrame。

    参数:
    - clean_dataframe: 干净数据 DataFrame
    - error_analysis: DataFrame，包含 error_type / affected_columns / error_description
    - batch_size: 每次调用 LLM 生成的样本数量，默认为 10
    - save_path: 保存路径（可选）

    返回:
    - dirty_df: 注入错误后的脏数据 DataFrame
    """
    """
    - clean_dataframe: 干净数据 DataFrame
    - error_analysis: DataFrame，包含 error_type / affected_columns / error_description
    - save_path: 保存路径（可选）

    返回:
    - dirty_df: 注入错误后的脏数据 DataFrame
    """

    number = len(clean_dataframe)
    clean_dataframe = clean_dataframe.copy()
    clean_dataframe.insert(0, "row_index", range(len(clean_dataframe)))
    # === Step 1. 定义输出结构 ===
    response_schemas = [
        ResponseSchema(name=col, description=f"Value for column '{col}' after error injection")
        for col in clean_dataframe.columns
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # === Step 2. 定义提示词模板 ===
    generator_template = """\
You are a **data error injector**.

Your mission: inject realistic, diverse, and context-aware errors into clean tabular data so that
the resulting dirty dataset simulates both (A) the *known* error patterns and (B) *plausible* errors
that were NOT listed in the provided analysis.

--- INPUTS (available to you):
- Clean data (tabular JSON): {data}
- Reference error patterns (examples / analysis): {error_patterns}

--- GLOBAL RULES:
1. **One dirty record per clean record** (strict 1-to-1 mapping).
2. Preserve **row order** and keep the `row_index` field exactly unchanged for alignment.
3. Do **not** add/delete/reorder rows or add any extra identifier fields (except keep `row_index` unchanged).
4. Preserve column names and schema (same number of columns and types).
5. Output **only** pure JSON (no commentary), a list of objects, each matching one input row.

--- ERROR INJECTION ALLOCATION (strict):
- For the set of modified cells across the whole batch, you must split injected errors into two equal halves:
  * **50%** must be errors *derived from* the provided `error_patterns` (follow their style/characteristics).
  * **50%** must be **LLM-inferred/plausible** errors applied to columns **not** necessarily listed in `error_patterns`.
- This 50/50 split should hold approximately for each output batch you generate (if batch is small, hold to per-row as closely as possible).

--- PER-ROW INJECTION RULES:
- For each input row, inject **multiple errors** (not just one) — e.g., modify **2–4 cells per row** where reasonable. If a row is very short, inject at least 1 cell.
- At least **one** of the modified cells per row should come from the *pattern-based* half (i.e., follow `error_patterns` when applicable), and at least **one** should be LLM-inferred (i.e., a plausible error extrapolated/applied to other columns).
- If `error_patterns` do not mention any columns that exist in a row, still apply **plausible** errors to that row (LLM-inferred half must still be honored).

--- GUIDELINES FOR LLM-INFERRED ERRORS:
- Consider column semantics when inventing errors:
  * Names/strings → typos, capitalization changes, swapped tokens, missing prefixes/suffixes.
  * Addresses/cities → abbreviations, swapped components, punctuation errors.
  * Dates → alternate formats, off-by-one days, truncated year, swapped day/month.
  * Numeric → rounding, extra digits, decimal/locale mismatches, sign errors.
  * Categorical → label swap, missing/unknown tokens, truncated label.
- You may generalize patterns from one column to similar columns (e.g., a "name" typo pattern may also apply to "city", "address", "contact").
- Keep values **plausible** — don't produce nonsense tokens that would be obviously invalid for the column.

--- FORMAT & STABILITY:
- **Do not** add any new columns (e.g., do not add `row_index_copy`); keep schema identical.
- If you need to reference a row for alignment in the prompt, rely on the `row_index` already present — but **do not** output any new alignment fields.
- Output must be a JSON array of objects. Each object must have exactly the same keys (columns) as the input row and appear in the same order.

### Output:
Return the output in **JSON format**, as a list of objects.
Each object represents one dirty record and appears in the same order as the input.

{format_instructions}

"""




    prompt = ChatPromptTemplate.from_template(generator_template)

    # === Step 3. 初始化循环变量 ===
    all_results = []
    generated_count = 0
    iteration = 0

    while generated_count < number:
        iteration += 1
        remaining = number - generated_count
        current_batch = min(batch_size, remaining)

        print(f"🚀 Iteration {iteration}: injecting errors for {current_batch} samples "
              f"({generated_count}/{number} done)...")

        df_batch = llm_gen_inject_errors(
            prompt=prompt,
            generator_template=generator_template,
            format_instructions=format_instructions,
            clean_df=clean_dataframe[generated_count:generated_count + current_batch],
            error_analysis_df=error_analysis,
            llm_serving=llm_serving,
            api_details=api_details,
            n_samples=current_batch,
            model=model,
        )

        if df_batch is not None and not df_batch.empty:
            all_results.append(df_batch)
            generated_count += len(df_batch)
        else:
            print("⚠️ LLM returned empty batch, retrying...")

        time.sleep(2)

    dirty_df = pd.concat(all_results, ignore_index=True)

    if "row_index" in dirty_df.columns:
        dirty_df = dirty_df.sort_values(by="row_index").reset_index(drop=True)
        dirty_df = dirty_df.drop(columns=["row_index"], errors="ignore")

    if save_path is not None:
        dirty_df.to_csv(save_path, index=False)
        print(f"✅ Dirty data saved to: {save_path}")

    print(f"🎯 Successfully generated {len(dirty_df)} dirty samples.")
    return dirty_df

def create_enhanced_representative_samples(representative_samples_dataset = None, representative_clean_path = None, 
                                         representative_dirty_path = None, dataset_name=None, n_samples=1000,
                                         api_details=None, llm_serving='together',
                                         model='gpt-3.5-turbo'):
    """
    创建增强的代表性样本数据集
    
    参数:
    - representative_samples_dataset: 已经包含脏数据和干净数据的代表性样本Dataset实例
    - n_samples: 要生成的样本数量
    - api_details: API详细信息
    - llm_serving: LLM服务类型
    - model: 使用的模型
    
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
    if not representative_samples_dataset is None:
        original_clean_data = representative_samples_dataset.clean_dataframe
        original_dirty_data = representative_samples_dataset.dataframe
        dataset_name = representative_samples_dataset.name
    elif not (representative_clean_path is None and representative_dirty_path is None):
        original_clean_data = pd.read_csv(representative_clean_path, dtype=object)
        original_dirty_data = pd.read_csv(representative_dirty_path, dtype=object)
    
    
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
    )
    
    # 为生成的干净数据注入错误，确保一一对应
    print("Injecting errors with LLM using batch processing...")
    enhanced_dirty_data = inject_errors_with_llm_batch(
        clean_dataframe=enhanced_clean_data,
        dataset_name=dataset_name,
        api_details=api_details,
        llm_serving=llm_serving,
        model=model,
        batch_size=20,  # 使用批次大小20以确保严格的一一对应
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
    clean_data_path = os.path.join(base_path, "..", "enhanced_data", f"enhanced_clean_data_{dataset_name}_{n_samples}.csv")
    dirty_data_path = os.path.join(base_path, "..", "enhanced_data", f"enhanced_dirty_data_{dataset_name}_{n_samples}.csv")

    # 保存数据到CSV文件
    print(f"Saving enhanced clean data to: {clean_data_path}")
    final_clean_data.to_csv(clean_data_path, index=False)
    
    print(f"Saving enhanced dirty data to: {dirty_data_path}")
    final_dirty_data.to_csv(dirty_data_path, index=False)
    
    print(f"Enhanced data saved successfully")
    
    return enhanced_dataset

if __name__ == "__main__":
    dirty_df = pd.read_csv("/home/stu/pys/CLLM-main/data/error_detection/hospital/dirty.csv", dtype=object)
    clean_df = pd.read_csv("/home/stu/pys/CLLM-main/data/error_detection/hospital/clean.csv", dtype=object)
    save_path = "/home/stu/pys/CLLM-main/data/error_detection/error.csv"
    error = analyse_data_error(dirty_df[:5], clean_df[:5], save_path=save_path)
    print("Error Analysis Result:")
    print(error)
    