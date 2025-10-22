import re
import json
import pandas as pd
from copy import deepcopy
import openai
from langchain.prompts import ChatPromptTemplate
import time


def llm_gen(
    prompt,
    generator_template,
    format_instructions,
    example_df,
    llm_serving,
    api_details,
    n_samples=None,
    temperature=0.75,
    max_tokens=4096,
    model="gpt4_20230815",
    n_processes=1,
):
    
    """
    The function `llm_gen` generates synthetic data based on a given prompt using different language
    model APIs.
    
    Args:
      prompt: The prompt template for the language model.
      generator_template: The template for generating the prompt with data placeholders.
      format_instructions: Instructions for formatting the output data.
      example_df: A pandas DataFrame containing the complete example data used for generating synthetic data.
      llm_serving: The type of language model serving platform ("together", "vllm", or "azure_openai").
      api_details: Details needed for API authentication and endpoint configuration.
      n_samples: The number of synthetic data samples to generate. If None, generates data once without looping.
      temperature: Controls the randomness of the generated text. Lower values result in more deterministic outputs.
      max_tokens: The maximum number of tokens the language model can generate in response.
      model: The specific language model to use for generating synthetic data.
      n_processes: The number of parallel processes to use for generating responses from the language model.
    
    Returns:
      The function `llm_gen` returns a pandas DataFrame containing the generated synthetic data
    based on the provided inputs and parameters.
    """

    # 初始化变量
    init = True
    not_sufficient = True
    df_llm = pd.DataFrame()  # 初始化 df_llm 变量以避免 UnboundLocalError
    
    # 如果没有指定n_samples，只生成一次数据
    if n_samples is None:
        max_iterations = 1
    else:
        max_iterations = 500  # 最大迭代次数，与原始llm_gen.py保持一致
    
    # 循环生成数据
    for i in range(max_iterations):
        # 保持pandas dataframe行的原始顺序
        try:
            # 顺序采样一部分数据作为示例，保持原始顺序
            sample_size = min(20, len(example_df))  # 最多采样20条
            example_df_sample = example_df.head(sample_size).reset_index(drop=True)
            
            small_data = str(example_df_sample.to_dict(orient="records"))

            prompt = ChatPromptTemplate.from_template(template=generator_template)

            messages = prompt.format_messages(
                data=small_data, format_instructions=format_instructions
            )

            if llm_serving == "together":
                openai.api_base = api_details["api_base"]
                openai.api_key = api_details["api_key"]

            if llm_serving == "vllm":
                from openai import OpenAI

                # Set OpenAI's API key and API base to use vLLM's API server.
                openai_api_key = "EMPTY"
                openai_api_base = "http://localhost:8000/v1"

            if llm_serving == "azure_openai":
                openai.api_type = "azure"
                openai.api_base = api_details["api_base"]
                openai.api_version = api_details["api_version"]
                openai.api_key = api_details["api_key"]

            if llm_serving != "vllm":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a tabular synthetic data generation model.",
                    },
                    {"role": "user", "content": messages[0].content},
                ]

            else:
                prompt = messages[0].content
                prompt = "".join(messages[0].content.split("\n")[1:])
                messages = [
                    {
                        "role": "system",
                        "content": "You are a synthetic data generator.",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ]

            # 添加重试机制
            max_retries = 10
            retry_delay = 5  # 秒
            
            for retry in range(max_retries):
                try:
                    if llm_serving == "azure_openai":
                        response = openai.ChatCompletion.create(
                            engine=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=0.95,
                            n=n_processes,
                            frequency_penalty=0,
                            presence_penalty=0,
                            stop=None,
                        )
                        break  # 成功则跳出重试循环

                    if llm_serving == "together":
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=0.95,
                            n=n_processes,
                        )
                        break  # 成功则跳出重试循环
                        
                except openai.error.ServiceUnavailableError as e:
                    if retry < max_retries - 1:
                        print(f"Server overloaded, retrying in {retry_delay} seconds... (Attempt {retry + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        print(f"Max retries reached. Error: {e}")
                        raise e
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    raise e

            if llm_serving == "vllm":
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    n=n_processes,
                    frequency_penalty=0,
                    presence_penalty=0,
                    max_tokens=max_tokens,
                    stop=None,
                )

            df_list = []
            for idx in range(n_processes):

                try:

                    if llm_serving == "vllm":
                        data = response.choices[idx].message.content
                    else:
                        data = response["choices"][idx]["message"]["content"]

                    # Extract dict-like strings using improved regular expressions
                    # This pattern handles nested braces and multiple JSON objects
                    dict_strings = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', data)

                    # Convert dict-like strings to actual dictionaries
                    dicts = []
                    for ds in dict_strings:
                        try:
                            # Clean up the string before parsing
                            ds_clean = ds.strip()
                            # Remove any trailing commas that might cause JSON parsing errors
                            ds_clean = re.sub(r',\s*}', '}', ds_clean)
                            ds_clean = re.sub(r',\s*]', ']', ds_clean)
                            # Remove comments (// ...) that might cause JSON parsing errors
                            ds_clean = re.sub(r'//.*$', '', ds_clean, flags=re.MULTILINE)
                            # Remove any trailing commas after comment removal
                            ds_clean = re.sub(r',\s*}', '}', ds_clean)
                            ds_clean = re.sub(r',\s*]', ']', ds_clean)
                            dicts.append(json.loads(ds_clean))
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            print(f"Problematic string: {ds}")
                            continue

                except:
                    continue

                if llm_serving == "vllm":
                    df_tmp = deepcopy(pd.DataFrame(dicts))
                    df_tmp = df_tmp[
                        ~df_tmp.apply(
                            lambda row: any(
                                [
                                    isinstance(cell, str)
                                    and cell
                                    in ["integer", "float", "numeric", "categorical"]
                                    for cell in row
                                ]
                            ),
                            axis=1,
                        )
                    ]
                    df_list.append(df_tmp)

                else:
                    print(f"idx: {idx}")
                    if idx == 0:
                        df_tmp = deepcopy(pd.DataFrame(dicts))
                        print(f"df_tmp {df_tmp.head()}")
                        df_tmp = df_tmp[
                            ~df_tmp.apply(
                                lambda row: any(
                                    [
                                        isinstance(cell, str)
                                        and cell
                                        in [
                                            "integer",
                                            "float",
                                            "numeric",
                                            "categorical",
                                        ]
                                        for cell in row
                                    ]
                                ),
                                axis=1,
                            )
                        ]

                    else:
                        df_check = pd.DataFrame(dicts)
                        df_check = df_check[
                            ~df_check.apply(
                                lambda row: any(
                                    [
                                        isinstance(cell, str)
                                        and cell
                                        in [
                                            "integer",
                                            "float",
                                            "numeric",
                                            "categorical",
                                        ]
                                        for cell in row
                                    ]
                                ),
                                axis=1,
                            )
                        ]
                        try:
                            df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)
                        except UnboundLocalError as e:
                            print(f"Error occurred: idx - {idx}, llm_serving - {llm_serving}")
                            print(f"df_tmp not previously created should be: {pd.DataFrame(dicts)}")
                            df_tmp = deepcopy(pd.DataFrame(dicts))
                            df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)

            if llm_serving == "vllm":
                df_tmp = df_list[0]
                for df_check in df_list[1:]:
                    df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)

            if init == True:
                df_llm = deepcopy(df_tmp)
                init = False
            else:
                df_llm = pd.concat([df_llm, df_tmp], ignore_index=True)

            n_gen = df_llm.shape[0]
            print("Current = ", n_gen, df_llm.shape)

            # 如果指定了n_samples，检查是否达到目标数量
            if n_samples is not None and n_gen >= n_samples:
                print("Done...")
                print(n_gen, df_llm.shape)
                not_sufficient = False
                break
            # 如果没有指定n_samples，只生成一次就退出
            elif n_samples is None:
                print("Single generation completed...")
                not_sufficient = False
                break

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            continue

    return df_llm

def llm_gen_error_analysis(
    prompt,
    generator_template,
    format_instructions,
    clean_df,
    dirty_df,
    llm_serving,
    api_details,
    temperature=0.5,
    max_tokens=2048,
    model="gpt-turbo-3.5"
):
    """
    调用LLM进行错误类型分析。
    输入为成对的干净数据和脏数据，输出为结构化的错误类型描述。
    """

    # === 1. 构建输入示例 ===
    examples = []
    for i in range(min(len(clean_df), len(dirty_df))):
        row_clean = clean_df.iloc[i].to_dict()
        row_dirty = dirty_df.iloc[i].to_dict()
        examples.append({"clean": row_clean, "dirty": row_dirty})
    small_data = json.dumps(examples, ensure_ascii=False, indent=2)

    # === 2. 构建提示词 ===
    prompt = ChatPromptTemplate.from_template(template=generator_template)
    messages = prompt.format_messages(data=small_data, format_instructions=format_instructions)

    if llm_serving == "together":
        openai.api_base = api_details["api_base"]
        openai.api_key = api_details["api_key"]
        model_name = model
    elif llm_serving == "azure_openai":
        openai.api_type = "azure"
        openai.api_base = api_details["api_base"]
        openai.api_version = api_details["api_version"]
        openai.api_key = api_details["api_key"]
        model_name = model
    else:
        raise ValueError(f"Unsupported llm_serving: {llm_serving}")

    # === 3. 调用LLM ===
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in data quality analysis and error detection."},
                    {"role": "user", "content": messages[0].content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"⚠️ Error: {e}, retrying...")
            time.sleep(3)
    else:
        raise RuntimeError("LLM request failed after multiple retries.")

    # === 4. 解析输出 ===
    try:
        json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', output_text)
        parsed = [json.loads(block) for block in json_blocks]
        df_result = pd.DataFrame(parsed)
    except Exception:
        print("⚠️ Failed to parse JSON, returning raw text output.")
        df_result = pd.DataFrame([{"raw_output": output_text}])

    print("✅ Error analysis completed.")
    return df_result

def llm_gen_error_analysis_again(
    prompt,
    generator_template,
    format_instructions,
    clean_df,
    dirty_df,
    old_error_analysis,
    llm_serving,
    api_details,
    temperature=0.5,
    max_tokens=2048,
    model="gpt-3.5-turbo"
):
    """
    调用LLM对已有错误分析进行更新（again 版本）
    - 基于新的修复示例（clean vs dirty）
    - 自动补充、修改或保留已有的错误分析
    """

    # === 1. 构建输入示例 ===
    examples = []
    for i in range(min(len(clean_df), len(dirty_df))):
        row_clean = clean_df.iloc[i].to_dict()
        row_dirty = dirty_df.iloc[i].to_dict()
        examples.append({"clean": row_clean, "dirty": row_dirty})
    small_data = json.dumps(examples, ensure_ascii=False, indent=2)

    # === 2. 转换旧的 error_analysis 为字符串输入 ===
    old_error_analysis_json = old_error_analysis.to_dict(orient="records")
    old_error_analysis_str = json.dumps(old_error_analysis_json, ensure_ascii=False, indent=2)

    # === 3. 构建提示词 ===
    prompt = ChatPromptTemplate.from_template(template=generator_template)
    messages = prompt.format_messages(
        old_error_analysis=old_error_analysis_str,
        data=small_data,
        format_instructions=format_instructions
    )

    # === 4. 设置模型参数 ===
    if llm_serving == "together":
        openai.api_base = api_details["api_base"]
        openai.api_key = api_details["api_key"]
        model_name = model
    elif llm_serving == "azure_openai":
        openai.api_type = "azure"
        openai.api_base = api_details["api_base"]
        openai.api_version = api_details["api_version"]
        openai.api_key = api_details["api_key"]
        model_name = model
    else:
        raise ValueError(f"Unsupported llm_serving: {llm_serving}")

    # === 5. 调用LLM（带重试机制） ===
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional data quality analyst."},
                    {"role": "user", "content": messages[0].content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"⚠️ Error during LLM call: {e}, retrying ({retry+1}/3)...")
            time.sleep(3)
    else:
        raise RuntimeError("❌ LLM request failed after multiple retries.")

    # === 6. 解析输出 ===
    try:
        json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', output_text)
        parsed = [json.loads(block) for block in json_blocks]
        df_result = pd.DataFrame(parsed)
    except Exception:
        print("⚠️ Failed to parse JSON, returning raw text output.")
        df_result = pd.DataFrame([{"raw_output": output_text}])

    print("✅ Updated error analysis generated successfully.")
    return df_result

def llm_gen_table_ontology(
    prompt,
    generator_template,
    dataframe,
    format_instructions,
    llm_serving,
    api_details,
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=2048
):
    """
    调用LLM生成表格列之间的本体树关系。
    输入：DataFrame
    输出：DataFrame格式的列关系描述
    """

    # === 1. 构造输入示例 ===
    length = min(20, len(dataframe))
    small_df = dataframe.head(length).to_dict(orient="records")
    data_json = json.dumps(small_df, ensure_ascii=False, indent=2)

    # === 2. 构建Prompt ===
    messages = prompt.format_messages(data=data_json, format_instructions=format_instructions)

    # === 3. 配置LLM服务 ===
    if llm_serving == "together":
        openai.api_base = api_details["api_base"]
        openai.api_key = api_details["api_key"]
        model_name = model
    elif llm_serving == "azure_openai":
        openai.api_type = "azure"
        openai.api_base = api_details["api_base"]
        openai.api_version = api_details["api_version"]
        openai.api_key = api_details["api_key"]
        model_name = model
    else:
        raise ValueError(f"Unsupported llm_serving: {llm_serving}")

    # === 4. 调用LLM ===
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in data semantics and ontology analysis."},
                    {"role": "user", "content": messages[0].content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"⚠️ Error: {e}, retrying...")
            time.sleep(3)
    else:
        raise RuntimeError("LLM request failed after multiple retries.")

    # === 5. 解析输出 ===
    try:
        json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', output_text)
        parsed = [json.loads(block) for block in json_blocks]
        df_result = pd.DataFrame(parsed)
    except Exception:
        print("⚠️ Failed to parse JSON, returning raw text output.")
        df_result = pd.DataFrame([{"raw_output": output_text}])

    print("✅ Ontology analysis completed.")
    return df_result


def llm_gen_enhanced_clean_data(
    prompt,
    generator_template,
    format_instructions,
    clean_df,
    ontology_df,
    api_details,
    llm_serving='together',
    number=200,
    model="gpt-turbo-3.5",
    temperature=0.6,
    max_tokens=4096,
):
    """
    调用LLM生成增强的干净数据
    """

    # === Step 1. 构造输入 ===
    example_data = clean_df.head(min(10, len(clean_df))).to_dict(orient="records")
    ontology_data = ontology_df.to_dict(orient="records")

    small_data = json.dumps(example_data, ensure_ascii=False, indent=2)
    ontology_json = json.dumps(ontology_data, ensure_ascii=False, indent=2)

    messages = prompt.format_messages(
        data=small_data,
        ontology=ontology_json,
        number=number,
        format_instructions=format_instructions
    )

    # === Step 2. 设置API参数 ===
    if llm_serving == "together":
        openai.api_base = api_details["api_base"]
        openai.api_key = api_details["api_key"]
        model_name = model
    elif llm_serving == "azure_openai":
        openai.api_type = "azure"
        openai.api_base = api_details["api_base"]
        openai.api_version = api_details["api_version"]
        openai.api_key = api_details["api_key"]
        model_name = model
    else:
        raise ValueError(f"Unsupported LLM service type: {llm_serving}")

    # === Step 3. 调用LLM ===
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a data synthesizer specializing in structured data."},
                    {"role": "user", "content": messages[0].content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"⚠️ LLM request failed (attempt {retry+1}/3): {e}")
            time.sleep(5)
    else:
        raise RuntimeError("❌ LLM generation failed after multiple retries.")

    # === Step 4. 解析输出 ===
    try:
        json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', output_text)
        parsed_data = [json.loads(block) for block in json_blocks]
        df_result = pd.DataFrame(parsed_data)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}")
        df_result = pd.DataFrame([{"raw_output": output_text}])

    return df_result

def llm_gen_inject_errors(
    prompt,
    generator_template,
    format_instructions,
    clean_df,
    error_analysis_df,
    llm_serving,
    api_details,
    n_samples,
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=4096,
):
    """
    单次调用 LLM，根据 error_analysis_df 注入相似错误，生成一批脏数据。
    """

    # === 1. 采样部分干净样本作为输入示例 ===
    example_data = clean_df.head(min(50, len(clean_df))).to_dict(orient="records")

    # === 2. 错误类型分析摘要 ===
    error_patterns = error_analysis_df.to_dict(orient="records")

    data_json = json.dumps(example_data, ensure_ascii=False, indent=2)
    error_json = json.dumps(error_patterns, ensure_ascii=False, indent=2)

    messages = prompt.format_messages(
        data=data_json,
        error_patterns=error_json,
        format_instructions=format_instructions
    )

    # === 3. LLM连接设置 ===
    if llm_serving == "together":
        openai.api_base = api_details["api_base"]
        openai.api_key = api_details["api_key"]
        model_name = model
    elif llm_serving == "azure_openai":
        openai.api_type = "azure"
        openai.api_base = api_details["api_base"]
        openai.api_version = api_details["api_version"]
        openai.api_key = api_details["api_key"]
        model_name = model
    else:
        raise ValueError(f"Unsupported LLM service type: {llm_serving}")

    # === 4. 调用 LLM ===
    for retry in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a realistic data error injector for tabular datasets."},
                    {"role": "user", "content": messages[0].content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output_text = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"⚠️ Error: {e}, retrying ({retry+1}/3)...")
            time.sleep(5)
    else:
        print("❌ LLM request failed after 3 retries.")
        return pd.DataFrame()

    # === 5. 解析输出 ===
    try:
        json_blocks = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', output_text)
        parsed = [json.loads(block) for block in json_blocks]
        df_result = pd.DataFrame(parsed)
        print(f"✅ Injected {len(df_result)} dirty samples successfully.")
    except Exception as e:
        print(f"⚠️ Failed to parse LLM output: {e}")
        df_result = pd.DataFrame()

    return df_result
