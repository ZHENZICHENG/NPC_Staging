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

PROMPT_TEMPLATE = """
You are a clinical oncologist specializing in nasopharyngeal carcinoma. Based on the patient's clinical record, determine the T, N, and M classifications for nasopharyngeal carcinoma. If the report explicitly documents invasion of a structure or metastatic finding required for a given category, assign that category; if findings satisfy more than one category, assign the highest applicable category; if the key evidence required for a category is not explicitly documented, do not assign that category. Follow these steps:
1. Analyze the imaging reports and identify the extent of primary tumor invasion, the presence or absence of lymph-node metastasis, and the presence or absence of distant metastasis.
2. According to the latest staging criteria, predict the patient's T (primary tumor), N (regional lymph nodes), and M (distant metastasis) classifications.
3. Provide the staging results and the key supporting source text for each classification.


[Output Format]
The output must include:
- T classification result: directly output the final T classification without any additional information.
- Source information for T classification: directly extract the original text from the clinical record that is relevant to determining T classification; do not summarize or analyze.
- N classification result: directly output the final N classification without any additional information.
- Source information for N classification: directly extract the original text from the clinical record that is relevant to determining N classification; do not summarize or analyze.
- M classification result: directly output the final M classification without any additional information.
- Source information for M classification: directly extract the original text from the clinical record that is relevant to determining M classification; do not summarize or analyze.

Return the result in JSON format exactly as follows:
{{
    "T_stage_val":"T classification result",
    "T_stage_source":"Source information extracted for T classification",
    "N_stage_val":"N classification result",
    "N_stage_source":"Source information extracted for N classification",
    "M_stage_val":"M classification result",
    "M_stage_source":"Source information extracted for M classification"
}}
Important: output only the JSON above and no other content.
Example:
{{
    "T_stage_val":"2",
    "T_stage_source":"The tumor invades the parapharyngeal space",
    "N_stage_val":"2",
    "N_stage_source":"Bilateral cervical lymph-node metastasis, maximum short-axis diameter 12 mm",
    "M_stage_val":"0",
    "M_stage_source":"No distant metastasis"
}}
[Patient Clinical Record]
{patient_case_record}
"""


SYSTEM_PROMPT = "You are a professional clinical oncologist specializing in nasopharyngeal carcinoma and TNM staging. Strictly follow the required JSON output format."


PATIENT_CASE_RECORD_TEMPLATE = """
MR findings:
{mr_report}
PET findings:
{pet_report}
CT conclusion:
{ct_report}
Bone scan conclusion:
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
