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
    normalize_postcode,
    fuzzy_match_name,
    extract_postcode,
    remove_postcode,
    image_to_base64,
    parse_llm_json,
    ensure_csv,
    is_missing,
    clean_employer,
    extract_customer_id,
    extract_document_type,
    calculate_accuracy,
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

        # Normalize address postcodes and pipe separators
        if "address" in submit_df.columns:
            submit_df["address"] = submit_df["address"].apply(normalize_postcode)
        if "address" in truth_df.columns:
            truth_df["address"] = truth_df["address"].apply(normalize_postcode)

        # Clean employer field to strip payslip prefix
        if "employer" in submit_df.columns:
            submit_df["employer"] = submit_df["employer"].apply(clean_employer)

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

        known_mismatches = []
        total_fields_checked = 0
        total_mismatches = 0
        field_stats = {field: {"checked": 0, "mismatches": 0} for field in fields_to_check}

        for _, row in merged.iterrows():
            for field, category in fields_to_check.items():
                submit_val = row.get(f"{field}_submit")
                truth_val = row.get(f"{field}_truth")

                if field == "employer" and is_missing(submit_val):
                    continue

                if is_missing(submit_val) and is_missing(truth_val):
                    continue

                total_fields_checked += 1
                field_stats[field]["checked"] += 1

                if submit_val == truth_val:
                    continue

                total_mismatches += 1
                field_stats[field]["mismatches"] += 1
                known_mismatches.append({
                    "customer_id": extract_customer_id(row.get("filename")),
                    "document_type": extract_document_type(row.get("filename")),
                    "filename": row.get("filename"),
                    "name": row.get("name"),
                    "field": field,
                    "extracted_value": submit_val,
                    "expected_value": truth_val,
                })

        # Build unknown customers list
        unknown_customers = []
        for _, row in unmatched.iterrows():
            unknown_customers.append({
                "customer_id": extract_customer_id(row.get("filename")),
                "document_type": extract_document_type(row.get("filename")),
                "filename": row.get("filename"),
                "extracted_name": row.get("name"),
                "extracted_address": row.get("address"),
                "extracted_dob": row.get("date_of_birth"),
                "extracted_occupation": row.get("occupation"),
                "extracted_employer": row.get("employer"),
                "reason": "Not found in customer table",
            })

        return (
            pd.DataFrame(known_mismatches),
            pd.DataFrame(unknown_customers),
            total_fields_checked,
            total_mismatches,
            field_stats,
        )

    known_df, unknown_df, total_fields, total_mismatches, field_stats = compare_csvs()

    # Save known mismatches CSV
    if not known_df.empty:
        known_df = known_df.sort_values(by=["customer_id", "field"]).reset_index(drop=True)
    known_df.to_csv("known_mismatches.csv", index=False)
    print("\n=== Known Mismatches ===")
    print(known_df)

    # Save unknown customers CSV
    if not unknown_df.empty:
        unknown_df = unknown_df.sort_values(by=["customer_id"]).reset_index(drop=True)
    unknown_df.to_csv("unknown_customers.csv", index=False)
    print("\n=== Unknown Customers (not in customer table) ===")
    print(unknown_df)

    # Known mismatches accuracy
    overall_accuracy = calculate_accuracy(total_fields, total_mismatches)
    print(f"\n--- Known Customers Accuracy ---")
    print(f"Total fields compared: {total_fields}")
    print(f"Mismatches found:      {total_mismatches}")
    print(f"Matches:               {total_fields - total_mismatches}")
    print(f"Overall accuracy:      {overall_accuracy:.2f}%")

    print(f"\n--- Per-Field Accuracy ---")
    for field, stats in field_stats.items():
        acc = calculate_accuracy(stats["checked"], stats["mismatches"])
        print(f"  {field:<15} {acc:.2f}%  ({stats['mismatches']}/{stats['checked']} mismatches)")

    # Unknown customers stats
    print(f"\n--- Unknown Customers ---")
    print(f"Total records not in customer table: {len(unknown_df)}")
    if not unknown_df.empty:
        type_counts = unknown_df["document_type"].value_counts()
        for doc_type, count in type_counts.items():
            print(f"  {doc_type}: {count}")


def left_text_ratio(img):
    """Return the ratio of OCR text characters in the left half vs total.

    English documents have left-aligned text, so correctly oriented
    images will have more text on the left side.
    """
    width = img.width
    left_half = img.crop((0, 0, width // 2, img.height))
    right_half = img.crop((width // 2, 0, width, img.height))
    try:
        left_text = pytesseract.image_to_string(left_half).strip()
        right_text = pytesseract.image_to_string(right_half).strip()
    except pytesseract.TesseractError:
        return 0.5
    total = len(left_text) + len(right_text)
    if total == 0:
        return 0.5
    return len(left_text) / total


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
            candidates = [rotation_angle]
            if rotation_angle in (90, 270):
                candidates = [90, 270]
            elif rotation_angle == 180:
                candidates = [180]

            best_rotation = rotation_angle
            best_score = -1

            for angle in candidates:
                rotated = image.rotate(angle, expand=True)
                score = left_text_ratio(rotated)
                print(f"  Trying {angle}\u00b0 rotation: left-text ratio = {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_rotation = angle

            original_score = left_text_ratio(image)
            print(f"  Original (0\u00b0): left-text ratio = {original_score:.2f}")

            if original_score >= best_score and original_score >= 0.45:
                print("Original orientation is best, no rotation applied")
            else:
                print(f"Applying {best_rotation}\u00b0 rotation (left-text ratio: {best_score:.2f})")
                image = image.rotate(best_rotation, expand=True)
        else:
            print("No rotation needed")
    except pytesseract.TesseractError as e:
        print(f"OSD failed: {e}, trying text-distribution fallback")
        best_rotation = 0
        best_score = left_text_ratio(image)
        print(f"  Original (0\u00b0): left-text ratio = {best_score:.2f}")
        for angle in [90, 180, 270]:
            rotated = image.rotate(angle, expand=True)
            score = left_text_ratio(rotated)
            print(f"  Trying {angle}\u00b0 rotation: left-text ratio = {score:.2f}")
            if score > best_score:
                best_score = score
                best_rotation = angle
        if best_rotation != 0:
            print(f"Applying {best_rotation}\u00b0 rotation (left-text ratio: {best_score:.2f})")
            image = image.rotate(best_rotation, expand=True)
        else:
            print("Original orientation is best, no rotation applied")

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
            data["filename"] = filename
            results.append(data)
            print(json.dumps(data, indent=2))

            with open("submit.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writerow(data)

        time.sleep(10)

    df = pd.DataFrame(results)

    # Strip "PAYSLIP" from employer field
    if "employer" in df.columns:
        df["employer"] = df["employer"].apply(clean_employer)

    df['postcode'] = df['address'].apply(extract_postcode)
    df['address'] = df['address'].apply(remove_postcode)
    df.to_csv("final_submit.csv", index=False)


store()
comparison()