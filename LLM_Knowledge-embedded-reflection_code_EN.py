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


STAGING_CRITERIA_AND_ANATOMY_KNOWLEDGE = """
T classification:
T1: The primary tumor is confined to the nasopharynx, or extends to the oropharynx, nasal cavity, oropharyngeal mucosa, or levator veli palatini muscle.
T2: The tumor invades the parapharyngeal space or involves adjacent soft tissues, including the tensor veli palatini muscle, prevertebral muscles (longus capitis), medial pterygoid muscle, or lateral pterygoid muscle.
T3: The tumor invades skull-base bony structures, the cervical vertebrae, pterygoid structures, or paranasal sinuses, including the sphenoid base, pterygoid process, pterygopalatine fossa, clivus, greater wing of the sphenoid bone, foramen spinosum, petrous apex, foramen lacerum, foramen magnum, foramen rotundum, foramen ovale, hypoglossal canal, jugular foramen, direct cervical-vertebral invasion, sphenoid sinus cavity, maxillary sinus, ethmoid sinus, or frontal sinus.
T4: The tumor invades intracranial structures, cranial nerves, the hypopharynx, orbit, infratemporal fossa, masticator space, or shows extensive soft-tissue infiltration beyond the lateral border of the lateral pterygoid muscle, including the meninges, direct invasion of the lateral hypopharyngeal space, hypopharyngeal mucosa, orbit, infratemporal fossa, cavernous sinus, or parotid gland.
N classification:
N0: No regional lymph-node metastasis.
N1: Unilateral or bilateral retropharyngeal lymph-node metastasis, or unilateral cervical lymph-node metastasis in levels I, II, III, or Va.
N2: Bilateral cervical lymph-node metastasis in levels I, II, III, or Va.
N3: Lymph-node metastasis with a greatest diameter greater than 6 cm, lymph-node metastasis below the caudal border of the cricoid cartilage (level IV, level Vb/supraclavicular fossa, or level Vc), or advanced radiologic extranodal extension involving adjacent muscles, skin, or the neurovascular bundle.
M classification:
M0: No distant metastasis.
M1: Distant metastasis is present, including osseous metastasis such as cervical vertebral, thoracic vertebral, or sternal metastasis; visceral metastasis such as liver or lung metastasis; or distant non-head-and-neck regional lymph-node metastasis such as axillary or subcarinal lymph-node metastasis.
"""


DOMAIN_SPECIFIC_EXPERT_INFORMED_RULES = """
T classification should be determined primarily from the MRI report. PET/CT, CT, and other modalities may be used only as supplementary information; if modalities conflict regarding T stage, the MRI report takes priority.
In imaging reports, terms such as "invasion", "destruction", "soft-tissue thickening", "occupation", "protrusion", "involvement", "decreased bone signal", "enhancement on contrast-enhanced scan", or "suspected bone destruction" should be interpreted as invasion of the described structure. However, "narrowing of the parapharyngeal space" alone does not indicate invasion of the parapharyngeal space.
Diagnostic criteria for metastatic lymph nodes on MR reports: a retropharyngeal lymph-node short-axis diameter of 5 mm or greater, or a cervical lymph-node short-axis diameter of 10 mm or greater, should be considered positive for lymph-node metastasis. The short axis is the smaller value in a reported length-by-width measurement.
If a lymph node does not meet the short-axis size threshold, but the MR conclusion states "metastasis considered", "metastasis to be excluded", or "metastasis cannot be excluded", it can still be judged positive.
If a lymph node does not meet the short-axis size threshold, but the report describes "ill-defined margins", "necrosis", "incomplete capsule", or "bead-like fusion", it can still be judged positive.
For N classification, prioritize the lymph-node location and short-axis diameter described in the MR report. For qualitative assessment of lymph-node nature, PET/CT findings and conclusions may be used to determine whether the lymph node is metastatic or inflammatory.
During PET/CT, arm elevation may cause slight differences in cervical lymph-node position compared with MR; therefore, lymph-node levels should be determined according to the MR report. If the MR report does not mention level IV lymph-node abnormality or metastasis, do not assign N3 on this basis.
When determining whether the lower border of lymph-node involvement reaches the supraclavicular fossa or below the caudal border of the cricoid cartilage (levels IV and V), a report describing levels II-V includes level IV and should be classified as N3. If level IV is not included, imaging reports usually specify levels II, III, and V.
Lymph nodes in different regions should be evaluated separately. For example, if a report describes "large lymph nodes in levels Ib, II, and III" and "small lymph nodes in levels Ib-V", each lymph-node group should be evaluated separately for whether it satisfies diagnostic criteria before assigning the final N classification.
"Fusion" of lymph nodes is not equivalent to high-grade extranodal extension.
If the report describes lymph nodes as "small lymph nodes" or "indeterminate nature", this usually indicates that the radiologist considers metastatic probability to be low.
Level VI lymph-node metastasis at initial diagnosis of nasopharyngeal carcinoma is extremely rare, nearly zero. If the MR report mentions level VI lymph nodes, this is likely a typographical error for level IV and should be classified according to level IV.
If MR findings and MR conclusions are inconsistent, for example findings state "bilateral" while the conclusion states "unilateral", prioritize the conclusion.
If an MR report describes multiple lymph nodes across several levels including level IV, but the maximum short-axis diameter is only 10 mm, the probability of lower-neck level IV metastasis can be considered relatively low.
When findings and conclusions are contradictory, prioritize the conclusion.
M classification should integrate the conclusions of all relevant examinations, including PET/CT, bone scan, CT, and MR. If a conclusion states "metastatic tumor", "metastasis considered", or "metastasis cannot be excluded", distant metastasis should be considered present.
For sites such as the parotid gland and cervical vertebrae, direct tumor invasion affects T classification, whereas lymph-node metastasis or metastatic disease in these sites affects N or M classification.
"""


PROMPT_TEMPLATE = """
You are a clinical oncologist specializing in nasopharyngeal carcinoma. Based on the patient's clinical record, determine the T, N, and M classifications for nasopharyngeal carcinoma. If the report explicitly documents invasion of a structure or metastatic finding required for a given category, assign that category; if findings satisfy more than one category, assign the highest applicable category; if the key evidence required for a category is not explicitly documented, do not assign that category. The output of this step will serve as the initial staging assessment. Follow these steps:
1. Analyze the imaging reports and identify the extent of primary tumor invasion, the presence or absence of lymph-node metastasis, and the presence or absence of distant metastasis.
2. According to the latest staging criteria, predict the patient's T (primary tumor), N (regional lymph nodes), and M (distant metastasis) classifications.
3. Provide the staging results and the key supporting source text for each classification.

[Staging Criteria and Anatomical Knowledge]
{staging_criteria_and_anatomy_knowledge}


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

REFLECTION_PROMPT_TEMPLATE = """
You are a clinical oncologist specializing in nasopharyngeal carcinoma. Based on the patient's clinical record, staging criteria and anatomical knowledge, and domain-specific expert-informed rules, review whether the initial TNM staging assessment for nasopharyngeal carcinoma is correct.

[Decision Procedure]
Evaluate T, N, and M classifications independently. For each classification, follow this procedure:
1. Use the initial staging assessment to identify the immediately preceding and immediately following classifications in the staging criteria and anatomical knowledge.
2. Compare the patient's information with the staging criteria and anatomical knowledge, as well as the expert-informed rules, to determine whether the patient satisfies either adjacent classification. For example, if the initial classification is T2, verify whether the patient satisfies T1 or T3 criteria. If neither adjacent classification is satisfied, output the final classification.
3. If either adjacent classification is satisfied, reassess the classification.
4. If uncertainty between adjacent classifications remains after three reassessments, output the middle classification as the final result.

[Staging Criteria and Anatomical Knowledge]
{staging_criteria_and_anatomy}

[Expert-Informed Rules]
{domain_specific_expert_informed_rules}

[Patient Clinical Record]
{patient_case_record}

[Initial Staging Assessment]
{initial_prediction}

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
