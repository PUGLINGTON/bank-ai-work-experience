from mistralai import Mistral
from PIL import Image
import pytesseract
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import base64
import io
import time 
import csv
from rapidfuzz import fuzz, process

load_dotenv(dotenv_path="credentials")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATA_FOLDER = "data"
FIELDNAMES = ["filename", "name", "address", "date_of_birth", "occupation", "employer"]

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".png") and not f.startswith("._")]
results = []
def comparison():
    def normalize_text(value):
        if pd.isna(value):
            return value
        value = str(value).lower().strip()
        value = re.sub(r'\b(mr|mrs|ms|miss|dr|prof|account holder)\b\.?:?', '', value)
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    def normalize_name(value):
        if pd.isna(value):
            return value
        value = str(value).lower().strip()
        value = re.sub(r'\b(mr|mrs|ms|miss|dr|prof|account holder)\b\.?:?', '', value)
        value = re.sub(r'\s+', ' ', value).strip()

        parts = value.split()
        if len(parts) > 2:
            value = f"{parts[0]} {parts[-1]}"

        return value

    def normalize_date(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        # try common formats and convert to YYYY-MM-DD
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return pd.to_datetime(value, format=fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        return None


    def fuzzy_match_name(name, choices, threshold=85):
        if pd.isna(name) or not choices:
            return None
        match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            return match[0]
        return None


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

                submit_missing = pd.isna(submit_val) or submit_val == ""
                truth_missing = pd.isna(truth_val) or truth_val == ""

                if field == "employer" and submit_missing:
                    continue

                if submit_missing and truth_missing:
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
                "truth_value": "No matching name found"})

        for _, row in unmatched.iterrows():
            results.append({
                "filename": row.get("filename"),
                "name": row.get("name"),
                "date_of_birth": row.get("date_of_birth"),
                "category": "N/A",
                "field": "name",
                "submit_value": row.get("name"),
                "truth_value": "No matching name found"})

          

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
            print(f"Correcting rotation: {rotation_angle}°")
            image = image.rotate(rotation_angle, expand=True)
        else:
            print("No rotation needed")
    except pytesseract.TesseractError as e:
        print(f"OSD still failed: {e}")

    return image

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_info(image):
    img_b64 = image_to_base64(image)

    response = mistral_client.chat.complete(
        model="pixtral-large-latest",
        messages=[
            {"role": "user","content": [ 
                    {"type": "text","text": 
                            (
                         "Extract the following fields from this document image and return ONLY valid JSON, "
                        "no markdown formatting, no explanation: "
                        "name, address, date_of_birth(typically in front of DoB in the format of YYYY-MM-DD, present in all documents), "
                        "occupation(often seen with job in front, DO NOT include the word 'Job'), "
                        "employer(for bank statements and utility bills do not include employer). "
                        "Read names and addresses carefully and exactly as written. "
                        "Text with no meaning associated with the fields specified should be ignored. "
                        "Within the employer field, payslip is present often. DO NOT keep this in the employer field. "
                        "Set any missing fields to null."
                            ), 
                        },
                    {"type": "image_url","image_url": f"data:image/png;base64,{img_b64}",},
                ],
            }
        ],
    )

    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()

    if not clean:
        print("WARNING: empty response")
        return None

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print("WARNING: could not parse JSON")
        print(f"Raw response: {raw}")
        return None
    
with open("submit.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()

def store():
    if not os.path.exists("submit.csv"):
        with open("submit.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            
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
        postcode_pattern = r'([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})'

        def extract_postcode(address):
            if pd.isna(address):
                return None
            match = re.search(postcode_pattern, address)
            return match.group(1) if match else None

        def remove_postcode(address):
            if pd.isna(address):
                return address
            return re.sub(postcode_pattern, '', address).rstrip(', ').strip()

        df['postcode'] = df['address'].apply(extract_postcode)
        df['address'] = df['address'].apply(remove_postcode)

        df.to_csv("final_submit.csv", index=False)

    add_postcode_column(df)

store()
comparison()