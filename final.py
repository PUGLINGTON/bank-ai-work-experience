from mistralai import Mistral
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import time
import csv
from rapidfuzz import fuzz

from utils import (
    normalize_text,
    normalize_name,
    normalize_date,
    split_address,
    strip_field_prefixes,
    fuzzy_match_name,
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

# Only set the Windows Tesseract path if it actually exists, so the same code
# runs unchanged on Linux/Mac where tesseract is already on PATH.
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
    """Lazily create the Mistral client so importing this module has no side effects."""
    global _mistral_client
    if _mistral_client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not found. Add it to the 'credentials' file."
            )
        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


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

        # Split address into street + postcode so each is compared independently
        for df in (submit_df, truth_df):
            if "address" in df.columns:
                split = df["address"].apply(split_address)
                df["address"] = split.apply(lambda t: t[0])
                df["postcode"] = split.apply(lambda t: t[1])

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
            "postcode": 2,
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

    return known_df, unknown_df, total_fields, total_mismatches, field_stats


def preprocess_for_ocr(image):
    """Grayscale, auto-contrast, upscale small images, and sharpen to boost OCR accuracy."""
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
    """Run OCR on a preprocessed image to get clean raw text for cross-validation."""
    try:
        processed = preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
        return text.strip()
    except pytesseract.TesseractError:
        return ""


def extract_info(image, ocr_text=""):
    img_b64 = image_to_base64(image)

    # Build prompt with OCR context for cross-validation
    prompt = (
        "Extract the following fields from this document image and return ONLY valid JSON, "
        "no markdown formatting, no explanation: "
        "name, address, date_of_birth(typically in front of DoB in the format of YYYY-MM-DD, present in all documents), "
        "occupation(often seen with job in front, DO NOT include the word 'Job'), "
        "employer(for bank statements and utility bills do not include employer). "
        "Read names and addresses carefully and exactly as written. "
        "Text with no meaning associated with the fields specified should be ignored. "
        "Within the employer field, payslip is present often. DO NOT keep this in the employer field. "
        "Set any missing fields to null."
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


def wipe_files():
    files_to_wipe = ["submit.csv", "final_submit.csv", "known_mismatches.csv", "unknown_customers.csv", "mismatches.csv"]
    for f in files_to_wipe:
        if os.path.exists(f):
            os.remove(f)
            print(f"Wiped: {f}")


def validate_extraction(data, ocr_text, filename):
    """Self-consistency checks using only the document itself (no customer table).

    Returns a list of {"field", "message"} dicts for any field that looks
    misread, so the caller can re-read those regions.
    """
    warnings = []

    # Check name appears in OCR text
    name = data.get("name")
    if name and ocr_text:
        name_lower = name.lower()
        ocr_lower = ocr_text.lower()
        name_parts = name_lower.split()
        found_parts = sum(1 for part in name_parts if part in ocr_lower)
        if found_parts < len(name_parts) / 2:
            warnings.append({"field": "name",
                             "message": f"name '{name}' not well-supported by OCR text"})

    # Check DOB appears in OCR text
    dob = data.get("date_of_birth")
    if dob and ocr_text:
        dob_variants = [dob, dob.replace("-", "/"), dob.replace("-", "")]
        if not any(v in ocr_text for v in dob_variants):
            warnings.append({"field": "date_of_birth",
                             "message": f"DOB '{dob}' not found in OCR text — may be misread"})

    for w in warnings:
        print(f"  WARNING: {w['message']}")

    return warnings


def _line_boxes(pre_image):
    """OCR the preprocessed image and group words into lines with bounding boxes."""
    data = pytesseract.image_to_data(
        pre_image, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )
    lines = {}
    for i, txt in enumerate(data["text"]):
        t = txt.strip()
        if not t:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        e = lines.setdefault(key, {"words": [], "l": [], "t": [], "r": [], "b": []})
        e["words"].append(t)
        e["l"].append(data["left"][i])
        e["t"].append(data["top"][i])
        e["r"].append(data["left"][i] + data["width"][i])
        e["b"].append(data["top"][i] + data["height"][i])
    return [
        {"text": " ".join(e["words"]),
         "box": (min(e["l"]), min(e["t"]), max(e["r"]), max(e["b"]))}
        for e in lines.values()
    ]


# Stable on-page labels used to anchor a re-read crop. We locate the *label*
# (which the model rarely misreads) rather than the extracted value, because the
# value is exactly what we suspect is wrong — matching on it would just re-crop
# the wrong region.
FIELD_LABELS = {
    "name": ["account holder", "account name", "employee name", "customer name",
             "name", "employee"],
    "date_of_birth": ["date of birth", "d.o.b", "dob", "born"],
    "address": ["address"],
    "occupation": ["occupation", "job title", "position", "role"],
    "employer": ["employer", "company name", "employer name"],
}

# Words that mean a "name" re-read actually grabbed an organisation / header.
NAME_STOPWORDS = {
    "bank", "council", "trust", "ltd", "limited", "plc", "statement", "payslip",
    "account", "holder", "branch", "sort", "code", "company", "employer",
    "group", "services", "retail", "logistics", "digital", "health", "albion",
    "university", "insurance", "current", "balance", "sample", "document",
}


def locate_line(pre_image, target_text, min_score=70):
    """Return the bounding box of the OCR line best matching target_text (or None)."""
    if not target_text:
        return None
    try:
        lines = _line_boxes(pre_image)
    except pytesseract.TesseractError:
        return None
    if not lines:
        return None
    target = str(target_text).lower()
    best, best_score = None, -1
    for ln in lines:
        score = fuzz.partial_ratio(target, ln["text"].lower())
        if score > best_score:
            best_score, best = score, ln
    if best is None or best_score < min_score:
        return None
    return best["box"]


def locate_label_region(pre_image, field_name):
    """Find the field's *label* and return a crop box over the value beside/below it.

    The value on a form usually sits to the right of its label on the same line,
    or on the line just underneath. Anchoring on the label (stable text) instead
    of the extracted value avoids re-cropping whatever wrong region the bad value
    happened to fuzzy-match. Returns a box, or None if the label isn't found.
    """
    labels = FIELD_LABELS.get(field_name, [])
    if not labels:
        return None
    try:
        lines = _line_boxes(pre_image)
    except pytesseract.TesseractError:
        return None
    if not lines:
        return None

    best, best_score = None, -1
    for ln in lines:
        text = ln["text"].lower()
        for kw in labels:
            score = fuzz.partial_ratio(kw, text)
            if score > best_score:
                best_score, best = score, ln
    if best is None or best_score < 85:
        return None

    left, top, right, bottom = best["box"]
    height = bottom - top
    # Same line to the right of the label, plus one line below it.
    return (left, top, pre_image.width, min(pre_image.height, bottom + int(height * 1.4)))


def refine_field(image, field_name, current_value):
    """Re-read one flagged field from a zoomed, heavily-preprocessed crop.

    Anchors the crop on the field's printed label (falling back to the extracted
    value only on a strong match), blows the region up, and asks the model to
    read just that value. Returns the refined string, or None if the region
    can't be found or the model returns nothing usable.
    """
    pre = preprocess_for_ocr(image)
    box = locate_label_region(pre, field_name)
    if box is None:
        box = locate_line(pre, current_value)
    if box is None:
        return None

    left, top, right, bottom = box
    pad_x = int((right - left) * 0.02) + 8
    pad_y = int((bottom - top) * 0.15) + 8
    crop = pre.crop((
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(pre.width, right + pad_x),
        min(pre.height, bottom + pad_y),
    ))
    # Blow up the small region so the model sees large, crisp glyphs.
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    crop = ImageOps.autocontrast(crop).filter(ImageFilter.SHARPEN)

    label = field_name.replace("_", " ")
    prompt = (
        f"This is a zoomed-in crop from an identity document. It should contain "
        f"the {label}, possibly next to a printed label such as '{label}:'. "
        f"Return ONLY the {label} value exactly as printed — no label, quotes, or "
        f"explanation. If the {label} is not clearly visible in this crop, reply "
        f"with exactly NONE."
    )
    try:
        response = get_mistral_client().chat.complete(
            model="pixtral-12b-2409",
            temperature=0,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": f"data:image/png;base64,{image_to_base64(crop.convert('RGB'))}"},
                ]}
            ],
        )
        refined = (response.choices[0].message.content or "").strip()
    except Exception:
        return None
    if not refined or refined.strip().lower() in {"none", "n/a", "na"}:
        return None
    return refined


def _looks_like_person_name(value):
    """True if value plausibly is a person's name (not an org / header)."""
    if not value:
        return False
    v = value.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,39}", v):
        return False
    tokens = [t for t in re.split(r"[ \-]", v.lower()) if t]
    if not (1 <= len(tokens) <= 4):
        return False
    return not any(t in NAME_STOPWORDS for t in tokens)


def _plausible_dob(value):
    """Return a normalized YYYY-MM-DD DOB if it's a plausible adult birth date.

    Rejects future/near-future dates (e.g. a statement date) and anything that
    would make the customer under 18 or over 120 — a re-read that grabs the
    wrong line usually lands on one of those.
    """
    norm = normalize_date(value)
    if not norm:
        return None
    dob = pd.Timestamp(norm)
    age_years = (pd.Timestamp.now() - dob).days / 365.25
    return norm if 18 <= age_years <= 120 else None


def accept_refinement(field_name, refined, old_value, ocr_text):
    """Decide whether a re-read may replace the original value.

    Guards against the re-read grabbing the wrong text: the new value must be
    field-plausible (a name must look like a name, a DOB must be a valid past
    date) and, for free-text fields, must actually be supported by the OCR text.
    Returns the value to use, or None to keep the original.
    """
    if not refined:
        return None
    if refined.strip().lower() == str(old_value or "").strip().lower():
        return None

    if field_name == "name":
        return refined if _looks_like_person_name(refined) else None
    if field_name == "date_of_birth":
        return _plausible_dob(refined)

    # Free-text fields: only trust a re-read the OCR text also supports.
    if ocr_text and fuzz.partial_ratio(refined.lower(), ocr_text.lower()) >= 85:
        return refined
    return None


def process_file(filepath, refine=True):
    """Rotation-correct, OCR, and extract a single document (document-only, bank-compliant).

    When a field is flagged as possibly misread, re-reads just that region with
    heavier preprocessing. Returns (data, ocr_text, warnings, refinements).
    """
    filename = os.path.basename(filepath)
    image = correct_rotation(filepath)
    ocr_text = ocr_preread(image)
    data = extract_info(image, ocr_text)
    warnings = []
    refinements = []
    if data is not None:
        data["filename"] = filename
        # Strip field labels and document-type header words leaked into values.
        for field in ("name", "date_of_birth", "address", "occupation", "employer"):
            if data.get(field):
                data[field] = strip_field_prefixes(data[field], field, filename)
        if data.get("employer"):
            data["employer"] = clean_employer(data["employer"])
        warnings = validate_extraction(data, ocr_text, filename)

        if refine:
            for w in warnings:
                field = w["field"]
                old_value = data.get(field)
                guess = refine_field(image, field, old_value)
                if guess:
                    guess = strip_field_prefixes(guess, field, filename)
                accepted = accept_refinement(field, guess, old_value, ocr_text)
                if accepted:
                    print(f"  REFINED {field}: '{old_value}' -> '{accepted}'")
                    refinements.append({"field": field, "old": old_value, "new": accepted})
                    data[field] = accepted
                elif guess:
                    print(f"  re-read rejected for {field}: '{guess}' (kept '{old_value}')")
    return data, ocr_text, warnings, refinements


def compare_record(data, truth_path="customer_table.csv"):
    """Audit-stage comparison of one extracted record against the customer table.

    Returns (rows, matched_name) where rows is a list of per-field dicts with
    keys field/extracted/expected/status. matched_name is None if the person is
    not found in the customer table.
    """
    truth_df = pd.read_csv(truth_path)

    for col in ["address", "occupation", "employer"]:
        if col in truth_df.columns:
            truth_df[col] = truth_df[col].apply(normalize_text)
    if "address" in truth_df.columns:
        truth_split = truth_df["address"].apply(split_address)
        truth_df["address"] = truth_split.apply(lambda t: t[0])
        truth_df["postcode"] = truth_split.apply(lambda t: t[1])
    truth_df["name"] = truth_df["name"].apply(normalize_name)
    truth_df["date_of_birth"] = truth_df["date_of_birth"].apply(normalize_date)

    rec_street, rec_postcode = split_address(normalize_text(data.get("address")))
    rec = {
        "name": normalize_name(data.get("name")),
        "date_of_birth": normalize_date(data.get("date_of_birth")),
        "address": rec_street,
        "postcode": rec_postcode,
        "occupation": normalize_text(data.get("occupation")),
        "employer": clean_employer(normalize_text(data.get("employer"))),
    }

    truth_names = truth_df["name"].dropna().unique().tolist()
    matched_name = fuzzy_match_name(rec["name"], truth_names)

    rows = []
    if matched_name is None:
        return rows, None

    truth_row = truth_df[truth_df["name"] == matched_name].iloc[0]
    for field in ["name", "date_of_birth", "address", "postcode", "occupation", "employer"]:
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


def store():
    ensure_csv("submit.csv", FIELDNAMES)

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"\n--- Processing: {filename} ---")

        # Document-only extraction with targeted re-read of flagged fields
        data, ocr_text, warnings, refinements = process_file(filepath)
        if data is not None:
            results.append(data)
            print(json.dumps(data, indent=2))

            with open("submit.csv", "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writerow(data)

        time.sleep(API_DELAY_SECONDS)

    df = pd.DataFrame(results)

    # Strip "PAYSLIP" from employer field
    if "employer" in df.columns:
        df["employer"] = df["employer"].apply(clean_employer)

    if 'address' in df.columns:
        split = df['address'].apply(split_address)
        df['address'] = split.apply(lambda t: t[0])
        df['postcode'] = split.apply(lambda t: t[1])
    df.to_csv("final_submit.csv", index=False)


if __name__ == "__main__":
    wipe_files()
    store()
    comparison()