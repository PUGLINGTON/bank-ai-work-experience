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
    extract_postcode,
    remove_postcode,
    image_to_base64,
    parse_llm_json,
)
from comparison import run_comparison, clean_employer

load_dotenv(dotenv_path="credentials")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATA_FOLDER = "data"
FIELDNAMES = ["filename", "name", "address", "date_of_birth", "occupation", "employer"]

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".png") and not f.startswith("._")]
results = []


from PIL import ImageEnhance


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
            # Try all candidate rotations and pick the best one
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
                print(f"  Trying {angle}° rotation: left-text ratio = {score:.2f}")
                if score > best_score:
                    best_score = score
                    best_rotation = angle

            # Also check if 0° (no rotation) is better
            original_score = left_text_ratio(image)
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
        # Fallback: try all four orientations and pick the best
        best_rotation = 0
        best_score = left_text_ratio(image)
        print(f"  Original (0°): left-text ratio = {best_score:.2f}")
        for angle in [90, 180, 270]:
            rotated = image.rotate(angle, expand=True)
            score = left_text_ratio(rotated)
            print(f"  Trying {angle}° rotation: left-text ratio = {score:.2f}")
            if score > best_score:
                best_score = score
                best_rotation = angle
        if best_rotation != 0:
            print(f"Applying {best_rotation}° rotation (left-text ratio: {best_score:.2f})")
            image = image.rotate(best_rotation, expand=True)
        else:
            print("Original orientation is best, no rotation applied")

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

    # Strip "PAYSLIP" from employer field
    if "employer" in df.columns:
        df["employer"] = df["employer"].apply(clean_employer)

    df['postcode'] = df['address'].apply(extract_postcode)
    df['address'] = df['address'].apply(remove_postcode)
    df.to_csv("final_submit.csv", index=False)


store()
run_comparison()