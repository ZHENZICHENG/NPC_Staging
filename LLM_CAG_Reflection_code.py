import os
import json
import time
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError
from tqdm import tqdm


try:
    client = OpenAI(
        api_key="",
        base_url=""
    )
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    exit()

#
MODEL_NAME = ""


STAGING_CRITERIA_AND_ANATOMY_KNOWLEDGE= """
T分期：
T1: 肿瘤局限于鼻咽，或侵犯口咽、鼻腔、口咽黏膜、腭帆提肌。
T2: 肿瘤侵犯咽旁间隙、邻近软组织累及（腭帆张肌、椎前肌（头长肌）、翼内肌、翼外肌）。
T3: 肿瘤侵犯颅底骨质、颈椎、翼状结构、鼻旁窦，包括蝶骨基底、翼突、翼腭窝、斜坡、蝶骨大翼、棘孔、岩尖、破裂孔、枕骨大孔、圆孔、卵圆孔、舌下神经管、颈静脉孔、颈椎直接侵犯、蝶窦腔、上颌窦、筛窦、额窦。
T4: 肿瘤侵犯颅内、颅神经、下咽、眼眶、颞下窝、咀嚼肌间隙，或广泛的软组织区域浸润并超过翼外肌外侧缘，包括脑膜、喉咽旁间隙直接侵犯、喉咽黏膜、眼眶、颞下窝、海绵窦、腮腺。
N分期：
N0: 无区域淋巴结转移
N1: 单侧或双侧咽后淋巴结转移，单侧颈淋巴结转移（I、II、III、Va区）
N2: 双侧颈淋巴结转移（I、II、III、Va区）
N3: 淋巴结最大径>6cm，环状软骨尾侧缘以下淋巴结转移(IV区、Vb区/锁骨上窝、Vc区)，累及邻近肌肉、皮肤、神经血管束的晚期影像学结外侵犯
M分期：
M0：无远处转移
M1：有远处转移，包括颈椎、胸椎、胸骨等骨性转移，肝、肺等器官转移，腋窝淋巴结、隆突下淋巴结等远处非头颈部区域性淋巴结的转移
"""


DOMAIN_SPECIFIC_EXPERT_INFORMED_RULES = """
T分期优先依据MR报告判断，PET/CT、CT等其他模态作补充参考，若不同模态对T分期提示不一致，以MR报告为准。
“受侵“、”破坏“、”软组织增厚“、”占据“、”突出“、“累及”，“骨质信号减低”、“增强扫描可见强化”“可疑骨质破坏”都视为侵犯该结构。仅“咽旁间隙变窄”不代表咽旁间隙受侵。
MR报告中淋巴结的诊断标准：咽后淋巴结短径大于或等于5mm为阳性淋巴结转移，颈部淋巴结短径大于或等于10mm为阳性淋巴结转移，短径即报告描述中长*宽中较小的值。
MR报告中淋巴结短径未满足诊断标准，但MR检查结论中“考虑转移”“转移待排”“未排除转移可能”，也可判断为阳性。
MR报告中淋巴结短径未满足诊断标准，但描述有“边界不清”、”坏死““包膜欠完整、串珠状融合”，也可判断为阳性。
优先通过MR报告的淋巴结位置和短径判断N分期。判断淋巴结性质，可根据PET/CT的检查报告和结论来判断该淋巴结是“转移瘤“还是”炎症“。
做PET/CT的时候双手上举会导致颈部淋巴结的位置和磁共振稍有差异，所以淋巴结区域以MR报告为准。若MR报告无提示IV区淋巴结则不考虑N3期。
判断淋巴结下界是否到锁骨上窝或环状软骨尾侧缘以下（即IV区和V区）时，报告中II-V区的描述是包含了IV区的（分为N3期）。若不含IV区，影像报告一般会描述II、III、V区淋巴结。
不同区域的淋巴结按照报告分别判断，比如“Ib、II、III区大淋巴结，Ib-V区小淋巴结“，需分别考虑是否满足诊断标准，再进行分期。
“融合“不是高级别包膜外侵犯。
若报告描述为“小淋巴结、性质待定”，则考虑影像科医生认为转移可能性不大。
初诊鼻咽癌VI区转移几率几乎为０，若MR报告中出现VI区淋巴结的描述，可以认为是IV区误写成VI区，应当按照IV区来分期。
若MR检查所见和检查结论不一致，比如检查所见为“双侧”，检查结论为“单侧”，以检查结论的描述为准。
若MR报告中包括IV区在内的多个区多发淋巴结最大短径仅10mm，可以认为下颈IV区转移几率低。
检查所见和检查结论矛盾时，优先采纳检查结论。
M分期需同时考虑PET/CT、骨扫描、CT、MR的检查结论，若结论为”转移瘤“、”考虑转移“、”不除外转移“等，则认为有转移。
对于腮腺、颈椎等部位，肿瘤侵犯则涉及T分期，淋巴结转移/转移则涉及N或M分期。
"""


PROMPT_TEMPLATE = """
你是一名鼻咽癌临床肿瘤医生，需要根据患者病例信息判断鼻咽癌的T分期、N分期和M分期。若报告中明确存在某一级别所要求的结构侵犯或转移证据，则归入该级；若同时满足多个级别，则以最高级为准；若报告中未明确记载某一级别所要求的关键证据，则不能分入该级。输出的结果作为【初始分期评估】。请按以下步骤进行：
1. 分析患者的影像报告，识别原发肿瘤侵犯范围、是否存在淋巴结转移和远处转移。
2. 根据最新的分期标准，为患者预测肿瘤的T（原发肿瘤）、N（区域淋巴结）、M（远处转移）分期。
3. 输出分期结果并提供病历关键信息。

【分期标准及解剖知识】
{staging_criteria_and_anatomy_knowledge}


【输出格式】
输出结果包含:
- T分期结果：直接输出最终T分期，无需其他任何信息
- T摘取的病历关键信息：直接摘取病历原文中与判断T分期相关的内容，不需总结、分析
- N分期结果：直接输出最终N分期，无需其他任何信息
- N摘取的病历关键信息：直接摘取病历原文中与判断N分期相关的内容，不需总结、分析
- M分期结果：直接输出最终M分期，无需其他任何信息
- M摘取的病历关键信息：直接摘取病历原文中与判断M分期相关的内容，不需总结、分析

请以json的格式输出，输出内容格式如下(请确保输出json格式如下):
{{
    "T_stage_val":"T分期结果",
    "T_stage_source":"T摘取的病历关键信息",
    "N_stage_val":"N分期结果",
    "N_stage_source":"N摘取的病历关键信息",
    "M_stage_val":"M分期结果",
    "M_stage_source":"M摘取的病历关键信息"
}}
强调：结果仅输出上述json，不要输出其他内容。
举例：
{{
    "T_stage_val":"2",
    "T_stage_source":"肿瘤侵犯咽旁间隙",
    "N_stage_val":"2",
    "N_stage_source":"双颈淋巴结转移，最大短径12mm",
    "M_stage_val":"0",
    "M_stage_source":"无远处转移"
}}
【病历信息】：
{patient_case_record}
"""

REFLECTION_PROMPT_TEMPLATE = """
你是一名鼻咽癌临床肿瘤医生，需要根据患者病例信息、分期标准和解剖知识、领域特异性专家知情规则，复核初始评估的鼻咽癌TNM分期是否正确。

【判断步骤】
T、N、M 分期分别独立判断，并分别执行以下流程：
1.用【初始分期评估】结果找到【分期标准及解剖知识】中前一级和后一级分期
2.对比【分期标准及解剖知识】和【专家经验规则】确认该患者信息是否符合前一级和后一级分期标准，例如分期为T2，那么需要确认患者是否符合T1和T3标准。如果均不符合则输出最终分期结果。
3.如果符合其中任一相邻分期，则重新判断分期
4.若3次判断后仍存在相邻分期争议，输出中位分期作为最终分期结果。

【分期标准及解剖知识】
{staging_criteria_and_anatomy}

【专家经验规则】
{domain_specific_expert_informed_rules}

【病历信息】：
{patient_case_record}

【初始分期评估】：
{initial_prediction}

【输出格式】
输出结果包含:
- T分期结果：直接输出最终T分期，无需其他任何信息
- T摘取的病历关键信息：直接摘取病历原文中与判断T分期相关的内容，不需总结、分析
- N分期结果：直接输出最终N分期，无需其他任何信息
- N摘取的病历关键信息：直接摘取病历原文中与判断N分期相关的内容，不需总结、分析
- M分期结果：直接输出最终M分期，无需其他任何信息
- M摘取的病历关键信息：直接摘取病历原文中与判断M分期相关的内容，不需总结、分析

请以json的格式输出，输出内容格式如下(请确保输出json格式如下):
{{
    "T_stage_val":"T分期结果",
    "T_stage_source":"T摘取的病历关键信息",
    "N_stage_val":"N分期结果",
    "N_stage_source":"N摘取的病历关键信息",
    "M_stage_val":"M分期结果",
    "M_stage_source":"M摘取的病历关键信息"
}}
强调：结果仅输出上述json，不要输出其他内容。
举例：
{{
    "T_stage_val":"2",
    "T_stage_source":"肿瘤侵犯咽旁间隙",
    "N_stage_val":"2",
    "N_stage_source":"双颈淋巴结转移，最大短径12mm",
    "M_stage_val":"0",
    "M_stage_source":"无远处转移"
}}
"""


SYSTEM_PROMPT = "你是一名专业的鼻咽癌临床肿瘤医生，擅长TNM分期判断。请严格按照要求输出JSON格式结果。"


PATIENT_CASE_RECORD_TEMPLATE = """
MR检查所见:
{mr_report}
PET检查所见:
{pet_report}
CT检查结论:
{ct_report}
骨扫描结论:
{bone_scan}
"""


def load_patients_from_excel(file_path: str) -> list:
    try:
        df = pd.read_excel(file_path, dtype=str)
        df.fillna('', inplace=True)
        patients_list = df.to_dict('records')
        print(f"Loaded {len(patients_list)} patient records from {file_path}.")
        return patients_list
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}.")
        return []
    except Exception as e:
        print(f"Error while reading Excel file: {e}")
        return []


def extract_json_from_content(raw_content: str):
    try:
        start_index = raw_content.find('{')
        end_index = raw_content.rfind('}') + 1

        if start_index != -1 and end_index > start_index:
            json_str = raw_content[start_index:end_index]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON object was found in the API response.")
    except (json.JSONDecodeError, ValueError) as parse_error:
        raise ValueError(f"Failed to parse JSON: {parse_error}") from parse_error


def normalize_prediction_data(prediction_data, raw_content: str) -> dict:
    target_dict = None
    if isinstance(prediction_data, dict):
        target_dict = prediction_data
    elif isinstance(prediction_data, list) and prediction_data:
        if isinstance(prediction_data[0], dict):
            target_dict = prediction_data[0]

    if target_dict is None:
        raise TypeError(f"Failed to extract a valid dictionary object. API response: {raw_content}")

    return target_dict


def get_tnm_prediction(patient_record: dict) -> dict | None:
    global client, MODEL_NAME
    case_record_text = PATIENT_CASE_RECORD_TEMPLATE.format(
        mr_report=patient_record.get("mr_report", ""),
        pet_report=patient_record.get("pet_report", ""),
        ct_report=patient_record.get("ct_report", ""),
        bone_scan=patient_record.get("bone_scan", "")
    )
    user_prompt = PROMPT_TEMPLATE.format(
        staging_criteria_and_anatomy_knowledge=STAGING_CRITERIA_AND_ANATOMY_KNOWLEDGE,
        patient_case_record=case_record_text
    )

    raw_content = None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            stream=False,
            timeout=180
        )

        raw_content = response.choices[0].message.content
        initial_prediction_data = extract_json_from_content(raw_content)
        initial_target_dict = normalize_prediction_data(initial_prediction_data, raw_content)

        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            staging_criteria_and_anatomy=STAGING_CRITERIA_AND_ANATOMY_KNOWLEDGE,
            domain_specific_expert_informed_rules=DOMAIN_SPECIFIC_EXPERT_INFORMED_RULES,
            patient_case_record=case_record_text,
            initial_prediction=json.dumps(initial_target_dict, ensure_ascii=False, indent=2)
        )

        reflection_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": reflection_prompt}
            ],
            temperature=0.1,
            stream=False,
            timeout=180
        )

        raw_content = reflection_response.choices[0].message.content
        prediction_data = extract_json_from_content(raw_content)
        target_dict = normalize_prediction_data(prediction_data, raw_content)

        target_dict['custom_id'] = patient_record['custom_id']
        return target_dict

    except Exception as e:

        if raw_content:
            raise type(e)(f"{e} | Raw response: {raw_content}")
        else:
            raise e

def save_results(results: list, base_filename: str):
    if not results:
        print("Result list is empty; nothing to save.")
        return

    json_output_path = f"{base_filename}.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(results)} results to JSON file: {json_output_path}")

    csv_output_path = f"{base_filename}.csv"
    all_fieldnames = set()
    for res in results:
        all_fieldnames.update(res.keys())

    with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(list(all_fieldnames)))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to CSV file: {csv_output_path}")


if __name__ == "__main__":
    EXCEL_FILE_NAME = 'data.xlsx'
    patients_data = load_patients_from_excel(EXCEL_FILE_NAME)
    all_results = []

    if not patients_data:
        print("Data loading failed or the file is empty. Program stopped.")
    else:
        print(f"\nStarting parallel processing for {len(patients_data)} patient records (model: {MODEL_NAME})...")

        success_count = 0
        failure_count = 0

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_id = {
                    executor.submit(get_tnm_prediction, patient): patient['custom_id']
                    for patient in patients_data
                }

                pbar = tqdm(total=len(patients_data), desc="Processing progress", unit="record")

                for future in as_completed(future_to_id):
                    patient_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                            success_count += 1
                            pbar.set_postfix(last_id=patient_id, success=success_count, failure=failure_count)

                    except Exception as e:
                        failure_count += 1
                        pbar.set_postfix(last_id=patient_id, success=success_count, failure=failure_count)
                        pbar.write(f"[ERROR] Failed to process ID {patient_id}: {e}")

                    pbar.update(1)

                pbar.close()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")

        finally:
            print("\n--- Running final save step ---")
            save_results(all_results, "result.xlsx")
