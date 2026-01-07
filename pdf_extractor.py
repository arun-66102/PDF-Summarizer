import pymupdf
from pdf2image import convert_from_path
import numpy as np

# Try to import OpenCV with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OpenCV not available: {e}")
    print("   OCR functionality will be limited")
    CV2_AVAILABLE = False

# Try to import Tesseract with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Tesseract not available: {e}")
    print("   OCR functionality will be disabled")
    TESSERACT_AVAILABLE = False

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
    if not CV2_AVAILABLE or not TESSERACT_AVAILABLE:
        print("❌ OCR not available - missing dependencies")
        return "OCR functionality not available. Please install OpenCV and Tesseract."
    
    try:
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
    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        return f"OCR processing failed: {str(e)}"
