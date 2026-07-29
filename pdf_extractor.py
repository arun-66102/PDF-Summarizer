import os
import base64
import requests
import numpy as np
import pymupdf

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
    print("   Local Tesseract OCR will be disabled, fallback to Groq Vision OCR")
    CV2_AVAILABLE = False

# Try to import Tesseract with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Tesseract not available: {e}")
    print("   Fallback to Open-Source Llama 3.2 Vision OCR via Groq")
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
    return (readable / max(len(text), 1)) < 0.4


def groq_vision_ocr(pdf_path):
    """
    Open-Source OCR using Meta Llama 3.2 11B Vision via Groq API.
    Zero system binary dependencies — renders PDF pages using PyMuPDF (fitz)
    and sends PNG base64 frames to Groq Vision API.
    Works 100% in Vercel serverless environments.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set for Vision OCR")
        return "OCR failed: GROQ_API_KEY environment variable not set."

    # Remove quotes if present
    api_key = api_key.strip().strip('"').strip("'")

    try:
        doc = pymupdf.open(pdf_path)
        extracted_text = ""
        max_pages = min(len(doc), 5)  # Limit max pages for fast serverless execution

        print(f"👁️ Performing Llama 3.2 Vision OCR on {max_pages} page(s)...")

        for i in range(max_pages):
            page = doc[i]
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
                                "text": "Perform OCR on this document page. Extract all visible text accurately preserving reading order. Respond ONLY with the extracted text."
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

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                page_text = data["choices"][0]["message"]["content"].strip()
                extracted_text += f"\n--- Page {i+1} ---\n" + page_text
                print(f"✅ Vision OCR page {i+1}/{max_pages} completed ({len(page_text)} chars)")
            else:
                print(f"⚠️ Vision OCR page {i+1} failed with status {resp.status_code}: {resp.text[:100]}")

        doc.close()
        return extracted_text.strip() or "No text extracted via Vision OCR."

    except Exception as e:
        print(f"❌ Groq Vision OCR failed: {e}")
        return f"Vision OCR error: {str(e)}"


def ocr_text_from_pdf(pdf_path):
    """
    Extracts text from scanned PDF.
    Uses local Tesseract OCR if installed, otherwise seamlessly falls back to
    open-source Meta Llama 3.2 Vision via Groq API.
    """
    if PDF2IMAGE_AVAILABLE and CV2_AVAILABLE and TESSERACT_AVAILABLE:
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
                text = pytesseract.image_to_string(
                    thresh,
                    lang="eng",
                    config=config
                )
                ocr_text += text + "\n"
            if ocr_text.strip():
                return ocr_text
        except Exception as e:
            print(f"⚠️ Local Tesseract OCR failed, attempting Groq Vision OCR fallback: {e}")

    # Fallback for Serverless / Vercel: Meta Llama 3.2 Vision on Groq
    return groq_vision_ocr(pdf_path)
