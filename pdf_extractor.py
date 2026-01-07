import pymupdf
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
import os
import platform

# Set Tesseract path based on operating system
if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif platform.system() == "Linux":
    # On Linux, Tesseract is usually in PATH
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif platform.system() == "Darwin":  # macOS
    # On macOS, Tesseract is usually in PATH or Homebrew
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

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
