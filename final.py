from mistralai.client import Mistral
from PIL import Image, ImageOps, ImageFilter
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
    normalize_address,
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
from comparison import run_comparison

load_dotenv(dotenv_path="credentials")

_WIN_TESSERACT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(_WIN_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _WIN_TESSERACT

DATA_FOLDER = "data"
FIELDNAMES = ["filename", "name", "address", "date_of_birth", "occupation", "employer"]
API_DELAY_SECONDS = 10

files = (
    [f for f in os.listdir(DATA_FOLDER) if f.endswith(".png") and not f.startswith("._")]
    if os.path.isdir(DATA_FOLDER)
    else []
)
results = []

_mistral_client = None


def get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not found. Add it to the 'credentials' file.")
        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


def preprocess_for_ocr(image):
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    longest = max(gray.size)
    if longest < 2000:
        scale = 2000 / longest
        gray = gray.resize(
            (int(gray.width * scale), int(gray.height * scale)),
            Image.LANCZOS,
        )
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def left_text_ratio(img):
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

    # preprocess first so OSD has a better image to work with
    preprocessed = preprocess_for_ocr(image)
    preprocessed_rgb = preprocessed.convert("RGB")

    boosted = preprocessed_rgb.resize(
        (preprocessed_rgb.width * 2, preprocessed_rgb.height * 2),
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
                score = left_text_ratio(preprocess_for_ocr(rotated).convert("RGB"))
                print(f"  Trying {angle}° rotation: left-text ratio = {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_rotation = angle

            original_score = left_text_ratio(preprocessed_rgb)
            print(f"  Original (0°): left-text ratio = {original_score:.2f}")

            if original_score >= best_score and original_score >= 0.45:
                print("Original orientation is best, no rotation applied")
            else:
                print(f"Applying {best_rotation}° rotation (left-text ratio: {best_score:.2f})")
                image = image.rotate(best_rotation, expand=True)
        else:
            print("No rotation needed")
    except pytesseract.TesseractError as e:
        print(f"OSD failed: {e}, trying text-distribution fallback")
        best_rotation = 0
        best_score = left_text_ratio(preprocessed_rgb)
        print(f"  Original (0°): left-text ratio = {best_score:.2f}")
        for angle in [90, 180, 270]:
            rotated = image.rotate(angle, expand=True)
            score = left_text_ratio(preprocess_for_ocr(rotated).convert("RGB"))
            print(f"  Trying {angle}° rotation: left-text ratio = {score:.2f}")
            if score > best_score:
                best_score = score
                best_rotation = angle
        if best_rotation != 0:
            print(f"Applying {best_rotation}° rotation (left-text ratio: {best_score:.2f})")
            image = image.rotate(best_rotation, expand=True)
        else:
            print("Original orientation is best, no rotation applied")

    return image

def ocr_preread(image):
    try:
        processed = preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        return text.strip()
    except pytesseract.TesseractError:
        return ""


def extract_info(image, ocr_text=""):
    img_b64 = image_to_base64(image)

    prompt = (
        "You are a document data extraction assistant. Extract the following fields from the document image and return ONLY a valid JSON object. No markdown, no explanation, no extra text — just the raw JSON.\n\n"
        "Fields to extract:\n"
        "- name: Full name of the individual. Read carefully and extract exactly as written.\n"
        "- address: Full address including street, city, and postcode. Read carefully and extract exactly as written.\n"
        "- date_of_birth: Found near 'DoB', 'Date of Birth', or similar labels. Format MUST be YYYY-MM-DD. This field is present in every document — search the entire document carefully before concluding it is missing.\n"
        "- occupation: The person's job title. Often preceded by the word 'Job', 'Occupation', or similar. Do NOT include the word 'Job' in your output.\n"
        "- employer: The name of the employing organisation. ONLY extract this from payslip documents. For bank statements and utility bills, set this field to null.\n\n"
        "Document hints:\n"
        "- Documents may be identified by prefixes such as BST (bank statement), PAY (payslip), or UTIL (utility bill).\n"
        "- If the document is a PAY document, carefully search for the employer name.\n"
        "- If the document is a BST or UTIL document, employer should be null.\n"
        "- Names may appear in headers, account holder sections, employee sections, customer sections, or recipient sections.\n"
        "- Addresses may span multiple lines and should be combined into a single value, preserving all address information including the postcode.\n"
        "- Date of birth may appear in personal information sections alongside the name or address, and may use abbreviations such as 'DoB'.\n"
        "- Occupation may appear within employment details, personal details, applicant information, or customer information.\n\n"
        "Rules:\n"
        "- If a field cannot be found after searching the entire document, set it to null.\n"
        "- Do NOT infer, guess, or generate values that are not explicitly present in the document.\n"
        "- Do NOT include the word 'Payslip' in the employer field.\n"
        "- Do NOT include any text that does not directly correspond to one of the fields above.\n"
        "- Names and addresses must be extracted exactly as they appear — do not correct spelling, punctuation, abbreviations, or formatting.\n"
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

    response = get_mistral_client().chat.complete(
        model="pixtral-12b-2409",
        temperature=0,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]}
        ],
    )

    return parse_llm_json(response.choices[0].message.content)


def validate_extraction(data, ocr_text, filename):
    warnings = []

    name = data.get("name")
    if name and ocr_text:
        name_lower = name.lower()
        ocr_lower = ocr_text.lower()
        name_parts = name_lower.split()
        found_parts = sum(1 for part in name_parts if part in ocr_lower)
        if found_parts < len(name_parts) / 2:
            warnings.append(f"  WARNING: name '{name}' not well-supported by OCR text")

    dob = data.get("date_of_birth")
    if dob and ocr_text:
        dob_variants = [dob, dob.replace("-", "/"), dob.replace("-", "")]
        if not any(v in ocr_text for v in dob_variants):
            warnings.append(f"  WARNING: DOB '{dob}' not found in OCR text — may be misread")

    for w in warnings:
        print(w)

    return warnings


def process_file(filepath):
    filename = os.path.basename(filepath)
    image = correct_rotation(filepath)
    ocr_text = ocr_preread(image)
    data = extract_info(image, ocr_text)
    warnings = []
    if data is not None:
        data["filename"] = filename
        if data.get("employer"):
            data["employer"] = clean_employer(data["employer"])
        warnings = validate_extraction(data, ocr_text, filename)
    return data, ocr_text, warnings


def compare_record(data, truth_path="customer_table.csv"):
    truth_df = pd.read_csv(truth_path)

    for col in ["address", "occupation", "employer"]:
        if col in truth_df.columns:
            truth_df[col] = truth_df[col].apply(normalize_text)
    if "address" in truth_df.columns:
        truth_df["address"] = truth_df["address"].apply(normalize_address)
    truth_df["name"] = truth_df["name"].apply(normalize_name)
    truth_df["date_of_birth"] = truth_df["date_of_birth"].apply(normalize_date)

    rec = {
        "name": normalize_name(data.get("name")),
        "date_of_birth": normalize_date(data.get("date_of_birth")),
        "address": normalize_address(normalize_text(data.get("address"))),
        "occupation": normalize_text(data.get("occupation")),
        "employer": clean_employer(normalize_text(data.get("employer"))),
    }

    truth_names = truth_df["name"].dropna().unique().tolist()
    matched_name = fuzzy_match_name(rec["name"], truth_names)

    rows = []
    if matched_name is None:
        return rows, None

    truth_row = truth_df[truth_df["name"] == matched_name].iloc[0]
    for field in ["name", "date_of_birth", "address", "occupation", "employer"]:
        extracted = rec.get(field)
        expected = truth_row.get(field)
        if field == "employer" and is_missing(extracted):
            continue
        if is_missing(extracted) and is_missing(expected):
            continue
        rows.append({
            "field": field,
            "extracted": extracted,
            "expected": expected,
            "status": "match" if extracted == expected else "mismatch",
        })
    return rows, matched_name


def wipe_files():
    files_to_wipe = ["submit.csv", "final_submit.csv", "known_mismatches.csv", "unknown_customers.csv", "mismatches.csv"]
    for f in files_to_wipe:
        if os.path.exists(f):
            os.remove(f)
            print(f"Wiped: {f}")


def store():
    with open("submit.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"\n--- Processing: {filename} ---")
        image = correct_rotation(filepath)

        ocr_text = ocr_preread(image)
        if ocr_text:
            print(f"  OCR pre-read: {len(ocr_text)} characters extracted")

        data = extract_info(image, ocr_text)
        if data is not None:
            data["filename"] = filename
            validate_extraction(data, ocr_text, filename)
            results.append(data)
            print(json.dumps(data, indent=2))

            with open("submit.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writerow(data)

        time.sleep(API_DELAY_SECONDS)

    df = pd.DataFrame(results)

    if "employer" in df.columns:
        df["employer"] = df["employer"].apply(clean_employer)

    df['postcode'] = df['address'].apply(extract_postcode)
    df['address'] = df['address'].apply(remove_postcode)
    df.to_csv("final_submit.csv", index=False)


if __name__ == "__main__":
    wipe_files()
    store()
    run_comparison()