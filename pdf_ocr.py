import pymupdf
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
import re
from collections import Counter
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        # Use sort=True for better reading order in complex layouts
        full_text += page.get_text("text", sort=True)
    doc.close()
    return full_text

def needs_ocr(text):
    if len(text.strip()) < 200:
        return True
    readable = sum(c.isalnum() for c in text)
    return (readable / len(text)) < 0.4

def ocr_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    ocr_text = ""
    for image in images:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Noise removal
        gray = cv2.medianBlur(gray, 3)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(
            thresh,
            lang="eng",
            config=config
        )
        ocr_text += text + "\n"
    return ocr_text

def clean_extracted_text(raw_text: str) -> str:
    # Normalize line endings
    text = raw_text.replace('\r', '\n')

    # Remove excessive spaces and tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove page numbers (Page 1, 1/10, - 3 -)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    text = re.sub(r'[-–—]\s*\d+\s*[-–—]', '', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

    # Remove repeated headers and footers
    lines = text.split('\n')
    line_counts = Counter(line.strip() for line in lines if line.strip())
    cleaned_lines = [
        line for line in lines
        if line_counts[line.strip()] < 2 or len(line.strip()) > 50
    ]

    # Merge broken OCR lines
    merged_lines = []
    for line in cleaned_lines:
        line = line.strip()
        if not line:
            merged_lines.append("")
            continue
        if merged_lines and not merged_lines[-1].endswith(('.', ':', '?')):
            merged_lines[-1] += ' ' + line
        else:
            merged_lines.append(line)

    # Normalize blank lines
    text = '\n'.join(merged_lines)
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()

# Example usage:
file_path = "sample.pdf"
text_content = extract_text_from_pdf(file_path)

# Check if OCR is needed
ocr_text = ""
if needs_ocr(text_content):
    ocr_text = ocr_text_from_pdf(file_path)
else:
    ocr_text = text_content

print(ocr_text)

# Clean the extracted text
cleaned_text = clean_extracted_text(ocr_text)
print(cleaned_text)