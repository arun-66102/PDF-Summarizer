import os
import pymupdf
import numpy as np

# Try to import pdf2image with error handling
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Try to import OpenCV with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import Tesseract with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from PDF using PyMuPDF (fitz).
    Tries standard text sorting first, then falls back to blocks/layout modes.
    """
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            # Use sort=True for reading order
            page_text = page.get_text("text", sort=True)
            if not page_text.strip():
                # Fallback to block layout extraction
                blocks = page.get_text("blocks")
                page_text = "\n".join([b[4] for b in blocks if len(b) >= 5 and b[4].strip()])
            full_text += page_text + "\n"
        doc.close()
        return full_text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def needs_ocr(text: str) -> bool:
    """Check if the extracted text is too sparse or non-readable (scanned document)."""
    clean_strip = text.strip()
    if len(clean_strip) < 150:
        return True
    readable = sum(c.isalnum() for c in clean_strip)
    return (readable / len(clean_strip)) < 0.4


def ocr_text_from_pdf(pdf_path: str) -> str:
    """
    Attempts OCR on scanned PDFs.
    Uses Tesseract/OpenCV if available locally.
    Returns clear fallback message if native OCR dependencies are not installed.
    """
    if PDF2IMAGE_AVAILABLE and CV2_AVAILABLE and TESSERACT_AVAILABLE:
        try:
            print("Performing Tesseract OCR on scanned document...")
            images = convert_from_path(pdf_path, dpi=300)
            ocr_text = ""
            for image in images:
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 3)
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31, 2
                )
                config = "--oem 3 --psm 6"
                text = pytesseract.image_to_string(thresh, lang="eng", config=config)
                ocr_text += text + "\n"
            return ocr_text.strip()
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")

    return "SCANNED_DOCUMENT_NO_TEXT"
