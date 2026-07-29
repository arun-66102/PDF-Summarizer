import os
import base64
import requests
import pymupdf
import numpy as np

# Try to import pdf2image with error handling
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  pdf2image not available: {e}")
    PDF2IMAGE_AVAILABLE = False

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


def ocr_text_with_llama_vision(pdf_path: str) -> str:
    """
    Perform high-accuracy OCR using Llama 3.2 Vision via Groq API.
    Uses PyMuPDF for page rendering — requires zero C++ system binaries (Poppler/Tesseract).
    Works 100% on Vercel and serverless environments.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not found for Llama Vision OCR")
        return ""

    try:
        doc = pymupdf.open(pdf_path)
        extracted_text = ""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Process up to 5 pages for serverless performance
        for page_num in range(min(len(doc), 5)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            payload = {
                "model": "llama-3.2-11b-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this document image accurately. Return only the extracted text with proper layout preservation."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                page_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if page_text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n" + page_text
            else:
                print(f"⚠️ Groq Llama Vision OCR status on page {page_num + 1}: {response.status_code}")

        doc.close()
        return extracted_text.strip()
    except Exception as e:
        print(f"❌ Llama Vision OCR failed: {e}")
        return ""


def ocr_text_from_pdf(pdf_path):
    print("🔍 Performing OCR on document using Llama 3.2 Vision...")
    vision_text = ocr_text_with_llama_vision(pdf_path)
    if vision_text and len(vision_text.strip()) > 50:
        print("✅ Llama 3.2 Vision OCR completed successfully")
        return vision_text

    # Fallback to local Tesseract if available
    if PDF2IMAGE_AVAILABLE and CV2_AVAILABLE and TESSERACT_AVAILABLE:
        print("ℹ️ Falling back to Tesseract OCR...")
        try:
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
            return ocr_text
        except Exception as e:
            print(f"❌ Tesseract OCR failed: {e}")

    return vision_text or "No content could be extracted from scanned document."
