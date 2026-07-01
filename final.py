from mistralai import Mistral
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

        for _, row in merged.iterrows():
            for field, category in fields_to_check.items():
                submit_val = row.get(f"{field}_submit")
                truth_val = row.get(f"{field}_truth")

                if field == "employer" and is_missing(submit_val):
                    continue

                if is_missing(submit_val) and is_missing(truth_val):
                    continue

                if submit_val == truth_val:
                    continue

                results.append({
                    "filename": row.get("filename"),
                    "name": row.get("name"),
                    "date_of_birth": row.get("date_of_birth"),
                    "category": "N/A",
                    "field": "name",
                    "submit_value": row.get("name"),
                    "truth_value": "No matching name found",
                })

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

        return pd.DataFrame(results)

    mismatches_df = compare_csvs()
    mismatches_df["sort_key"] = mismatches_df["category"].apply(lambda x: 999 if x == "N/A" else x)
    mismatches_df = mismatches_df.sort_values(by=["sort_key", "name"]).drop(columns="sort_key").reset_index(drop=True)
    print(mismatches_df)
    mismatches_df.to_csv("mismatches.csv", index=False)


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
            print(f"Correcting rotation: {rotation_angle}\u00b0")
            image = image.rotate(rotation_angle, expand=True)
        else:
            print("No rotation needed")
    except pytesseract.TesseractError as e:
        print(f"OSD still failed: {e}")

    return image


def extract_info(image):
    img_b64 = image_to_base64(image)

    response = mistral_client.chat.complete(
        model="pixtral-large-latest",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Extract the following fields from this document image and return ONLY valid JSON, "
                    "no markdown formatting, no explanation: "
                    "name, address, date_of_birth(typically in front of DoB in the format of YYYY-MM-DD, present in all documents), "
                    "occupation(often seen with job in front, DO NOT include the word 'Job'), "
                    "employer(for bank statements and utility bills do not include employer). "
                    "Read names and addresses carefully and exactly as written. "
                    "Text with no meaning associated with the fields specified should be ignored. "
                    "Within the employer field, payslip is present often. DO NOT keep this in the employer field. "
                    "Set any missing fields to null."
                )},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]}
        ],
    )

    return parse_llm_json(response.choices[0].message.content)


def store():
    ensure_csv("submit.csv", FIELDNAMES)

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"\n--- Processing: {filename} ---")
        image = correct_rotation(filepath)
        data = extract_info(image)
        if data is not None:
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
