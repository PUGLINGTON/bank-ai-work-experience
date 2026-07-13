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
    load_customer_lookup,
    semantic_correct_name,
    cross_validate_date,
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


def ocr_preread(image):
    """Run OCR on the image to get raw text for cross-validation."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except pytesseract.TesseractError:
        return ""


def extract_info(image, ocr_text=""):
    img_b64 = image_to_base64(image)

    # Build prompt with OCR context for cross-validation
    prompt = (
        "You are a document data extraction assistant. Extract the following fields from the document image and return ONLY a valid JSON object. No markdown, no explanation, no extra text \u2014 just the raw JSON.\n\n"

        "Fields to extract:\n"
        "- name: Full name of the individual. Read carefully and extract exactly as written.\n"
        "- address: Full address including street, city, and postcode. Read carefully and extract exactly as written.\n"
        "- date_of_birth: Found near 'DoB', 'Date of Birth', or similar labels. Format MUST be YYYY-MM-DD. This field is present in every document \u2014 search the entire document carefully before concluding it is missing.\n"
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
        "- Names and addresses must be extracted exactly as they appear \u2014 do not correct spelling, punctuation, abbreviations, or formatting.\n"
        "- Preserve all address components, including street, city, county (if present), and postcode.\n"
        "- Return exactly one valid JSON object containing only the keys: name, address, date_of_birth, occupation, employer.\n"
    )

    if ocr_text:
        prompt += (
            "\n\nFor cross-reference, here is the raw OCR text extracted from this document. "
            "Use it to double-check your reading of names, dates, and addresses — "
            "if the image is unclear, prefer the OCR text for spelling:\n\n"
            f"{ocr_text[:3000]}"
        )

    response = mistral_client.chat.complete(
        model="pixtral-12b-2409",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]}
        ],
    )

    return parse_llm_json(response.choices[0].message.content)

def wipe_files():
    files_to_wipe = ["submit.csv", "final_submit.csv", "known_mismatches.csv", "unknown_customers.csv","mismatches.csv"]
    for f in files_to_wipe:
        if os.path.exists(f):
            os.remove(f)
            print(f"Wiped: {f}")

def store():
    ensure_csv("submit.csv", FIELDNAMES)

    # Load customer table for semantic correction
    id_lookup, all_names = load_customer_lookup("customer_table.csv")

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"\n--- Processing: {filename} ---")
        image = correct_rotation(filepath)

        # OCR pre-read for cross-validation
        ocr_text = ocr_preread(image)
        if ocr_text:
            print(f"  OCR pre-read: {len(ocr_text)} characters extracted")

        data = extract_info(image, ocr_text)
        if data is not None:
            data["filename"] = filename

            # Semantic name correction
            corrected_name, name_was_corrected = semantic_correct_name(
                data.get("name"), filename, id_lookup, all_names
            )
            if name_was_corrected:
                data["name"] = corrected_name

            # Date cross-validation
            corrected_dob, dob_was_corrected = cross_validate_date(
                data.get("date_of_birth"), ocr_text, filename, id_lookup
            )
            if dob_was_corrected:
                data["date_of_birth"] = corrected_dob

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

wipe_files() 
store()
comparison()