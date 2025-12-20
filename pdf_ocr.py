import pymupdf
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
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

# Example usage:
file_path = "scanned.pdf"
text_content = extract_text_from_pdf(file_path)

# Check if OCR is needed
ocr_text = ""
if needs_ocr(text_content):
    ocr_text = ocr_text_from_pdf(file_path)
else:
    ocr_text = text_content

print(ocr_text)