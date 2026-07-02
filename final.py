from mistralai.client import Mistral
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
import pandas as pd
import json
import time
import csv

from utils import (
    normalize_text,
    normalize_name,
    normalize_date,
    fuzzy_match_name,
    extract_postcode,
    remove_postcode,
    image_to_base64,
    parse_llm_json,
    ensure_csv,
    is_missing,
    calculate_accuracy,
    print_accuracy_report,
)

load_dotenv(dotenv_path="credentials")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATA_FOLDER = "data"
FIELDNAMES = ["filename", "name", "address", "date_of_birth", "occupation", "employer"]

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".png") and not f.startswith("._")]
results = []


def comparison():
    def compare_csvs(submit_path="submit.csv", truth_path="customer_table.csv"):
        submit_df = pd.read_csv(submit_path)
        truth_df = pd.read_csv(truth_path)

        text_columns = ["address", "occupation", "employer"]

        for col in text_columns:
            if col in submit_df.columns:
                submit_df[col] = submit_df[col].apply(normalize_text)
            if col in truth_df.columns:
                truth_df[col] = truth_df[col].apply(normalize_text)

        submit_df["name"] = submit_df["name"].apply(normalize_name)
        truth_df["name"] = truth_df["name"].apply(normalize_name)

        submit_df["date_of_birth"] = submit_df["date_of_birth"].apply(normalize_date)
        truth_df["date_of_birth"] = truth_df["date_of_birth"].apply(normalize_date)

        truth_names = truth_df["name"].dropna().unique().tolist()

        submit_df["matched_name"] = submit_df["name"].apply(
            lambda n: fuzzy_match_name(n, truth_names)
        )

        unmatched = submit_df[submit_df["matched_name"].isna()]
        matched = submit_df.dropna(subset=["matched_name"])

        merged = matched.merge(
            truth_df,
            left_on="matched_name",
            right_on="name",
            suffixes=("_submit", "_truth"),
            how="inner"
        )

        fields_to_check = {
            "name": 1,
            "date_of_birth": 1,
            "address": 2,
            "occupation": 2,
            "employer": 3,
        }

        results = []
        total_fields_checked = 0
        total_mismatches = 0

        for _, row in merged.iterrows():
            for field, category in fields_to_check.items():
                submit_val = row.get(f"{field}_submit")
                truth_val = row.get(f"{field}_truth")

                if field == "employer" and is_missing(submit_val):
                    continue

                if is_missing(submit_val) and is_missing(truth_val):
                    continue

                total_fields_checked += 1

                if submit_val == truth_val:
                    continue

                total_mismatches += 1
                results.append({
                    "filename": row.get("filename"),
                    "name": row.get("name"),
                    "date_of_birth": row.get("date_of_birth"),
                    "category": "N/A",
                    "field": "name",
                    "submit_value": row.get("name"),
                    "truth_value": "No matching name found",
                })

        # Count unmatched records (each counts as a mismatch on the name field)
        total_fields_checked += len(unmatched)
        total_mismatches += len(unmatched)

        for _, row in unmatched.iterrows():
            results.append({
                "filename": row.get("filename"),
                "name": row.get("name"),
                "date_of_birth": row.get("date_of_birth"),
                "category": "N/A",
                "field": "name",
                "submit_value": row.get("name"),
                "truth_value": "No matching name found",
            })

        accuracy = calculate_accuracy(total_fields_checked, total_mismatches)

        return pd.DataFrame(results), total_fields_checked, total_mismatches, accuracy

    mismatches_df, total_fields, total_mismatches, accuracy = compare_csvs()
    mismatches_df["sort_key"] = mismatches_df["category"].apply(lambda x: 999 if x == "N/A" else x)
    mismatches_df = mismatches_df.sort_values(by=["sort_key", "name"]).drop(columns="sort_key").reset_index(drop=True)
    print(mismatches_df)
    mismatches_df.to_csv("mismatches.csv", index=False)

    print_accuracy_report(total_fields, total_mismatches, accuracy)


from PIL import ImageEnhance

def correct_rotation(filepath):
    image = Image.open(filepath)

    boosted = image.resize(
        (image.width * 2, image.height * 2),
        Image.LANCZOS
    )

    try:
        osd_data = pytesseract.image_to_osd(boosted, output_type='dict')
        rotation_angle = osd_data['rotate']
        if rotation_angle != 0:
            print(f"Correcting rotation: {rotation_angle}°")
            image = image.rotate(rotation_angle, expand=True)
        else:
            print("No rotation needed")
    except pytesseract.TesseractError as e:
        print(f"OSD still failed: {e}")

    # greyscale conversion
    #image = image.convert('L')
    #image = image.convert('RGB')  # convert back so PNG encoding works correctly

    # contrast boost
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # sharpness boost
    sharpener = ImageEnhance.Sharpness(image)
    image = sharpener.enhance(2.0)

    return image

def extract_info(image):
    img_b64 = image_to_base64(image)

    response = mistral_client.chat.complete(
        model="pixtral-large-latest",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": (
                  "You are a document data extraction assistant. Extract the following fields from the document image and return ONLY a valid JSON object. No markdown, no explanation, no extra text — just the raw JSON.\n\n"

                    "Fields to extract:\n"
                    "- name: Full name of the individual. Read carefully and extract exactly as written.\n"
                    "- address: Full address including street, city, and postcode. Read carefully and extract exactly as written.\n"
                    "- date_of_birth: Found near 'DoB', 'Date of Birth', or similar labels. Format MUST be YYYY-MM-DD. This field is present in every document — search the entire document carefully before concluding it is missing.\n"
                    "- occupation: The person's job title. Often preceded by the word 'Job', 'Occupation', or similar. Do NOT include the word 'Job' in your output.\n"
                    "- employer: The name of the employing organisation. ONLY extract this from payslip documents. For bank statements and utility bills, set this field to null.\n\n"

                    "Document hints:\n"
                    "- Documents may be identified by prefixes such as BST (bank statement), PAY (payslip), or UTIL-**** (utility bill).\n"
                    "- If the document is a PAY document, carefully search for the employer name, even if it is not immediately adjacent to the employee details.\n"
                    "- If the document is a BST or UTIL document, employer should be null.\n"
                    "- Names may appear in headers, account holder sections, employee sections, customer sections, or recipient sections.\n"
                    "- Addresses may span multiple lines and should be combined into a single value, preserving all address information including the postcode.\n"
                    "- Date of birth may appear in personal information sections alongside the name or address, and may use abbreviations such as 'DoB'. Search the entire document before deciding it is missing.\n"
                    "- Occupation may appear within employment details, personal details, applicant information, or customer information.\n\n"

                    "Rules:\n"
                    "- If a field cannot be found after searching the entire document, set it to null.\n"
                    "- Do NOT infer, guess, or generate values that are not explicitly present in the document.\n"
                    "- Do NOT include the word 'Payslip' in the employer field.\n"
                    "- Do NOT include any text that does not directly correspond to one of the fields above.\n"
                    "- Names and addresses must be extracted exactly as they appear — do not correct spelling, punctuation, abbreviations, or formatting.\n"
                    "- Preserve all address components, including street, city, county (if present), and postcode.\n"
                    "- Return exactly one valid JSON object containing only the keys: name, address, date_of_birth, occupation, employer.\n"
                )},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]}
        ],
    )

    return parse_llm_json(response.choices[0].message.content)


def store():
    with open("submit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"\n--- Processing: {filename} ---")
        image = correct_rotation(filepath)
        data = extract_info(image)
        if data is not None:
            data["filename"] = filename
            results.append(data)
            print(json.dumps(data, indent=2))

            with open("submit.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writerow(data)

        time.sleep(10)

    df = pd.DataFrame(results)

    def add_postcode_column(df):
        df['postcode'] = df['address'].apply(extract_postcode)
        df['address'] = df['address'].apply(remove_postcode)
        df.to_csv("final_submit.csv", index=False)

    add_postcode_column(df)


store()
comparison()
print("hello")