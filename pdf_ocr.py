import pymupdf
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
import re
import tiktoken
import json
import time
import requests
from typing import List, Optional, Dict, Any
from collections import Counter
import logging
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

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Counts tokens using tiktoken (model-agnostic approximation).
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def split_into_sections(text: str) -> List[str]:
    """
    Splits text using headings and paragraph boundaries.
    """
    # Split on headings or double newlines
    sections = re.split(r'\n{2,}|(?=\n[A-Z][^\n]{3,}\n)', text)
    return [sec.strip() for sec in sections if sec.strip()]

def chunk_text(
    text: str,
    max_tokens: int = 3000,
    overlap_ratio: float = 0.15,
    model_name: str = "gpt-4"
) -> List[str]:
    """
    Chunks text semantically with overlap.
    """
    sections = split_into_sections(text)
    chunks = []

    current_chunk = ""
    current_tokens = 0
    overlap_tokens = int(max_tokens * overlap_ratio)

    for section in sections:
        section_tokens = count_tokens(section, model_name)

        if current_tokens + section_tokens <= max_tokens:
            current_chunk += "\n\n" + section
            current_tokens += section_tokens
        else:
            chunks.append(current_chunk.strip())

            # Create overlap from previous chunk
            encoded = tiktoken.encoding_for_model(model_name).encode(current_chunk)
            overlap_text = tiktoken.encoding_for_model(model_name).decode(
                encoded[-overlap_tokens:]
            )

            current_chunk = overlap_text + "\n\n" + section
            current_tokens = count_tokens(current_chunk, model_name)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def prepare_text_for_llm(
    clean_text: str,
    model_context_limit: int = 4000,
    model_name: str = "gpt-4"
) -> List[str]:
    """
    Decides whether chunking is required and returns text chunks.
    """
    total_tokens = count_tokens(clean_text, model_name)

    if total_tokens <= model_context_limit:
        return [clean_text]  # No chunking needed

    return chunk_text(
        clean_text,
        max_tokens=model_context_limit - 500,  # buffer for prompt
        overlap_ratio=0.15,
        model_name=model_name
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for different models"""
    GROQ_MODELS = {
        "llama3-8b": "llama-3.1-8b-instant",
        "llama3-70b": "llama-3.3-70b-versatile", 
        "llama-guard": "meta-llama/llama-guard-4-12b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "whisper": "whisper-large-v3",
        "whisper-turbo": "whisper-large-v3-turbo"
    }
    
    DEFAULT_MODEL = "llama3-8b"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    TIMEOUT = 60  # seconds (reduced from 120 for faster failure)
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    @classmethod
    def get_api_key(cls):
        """Get Groq API key from environment variable or file"""
        import os
        
        # First try environment variable
        api_key = os.getenv('GROQ_API_KEY')
        
        # If not found, try reading from file
        if not api_key:
            try:
                key_file = os.path.join(os.path.dirname(__file__), '.groq_api_key')
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                logger.info("API key loaded from .groq_api_key file")
            except FileNotFoundError:
                logger.error("API key not found in environment or .groq_api_key file")
                logger.info("Set it with: set GROQ_API_KEY=your_key_here")
                logger.info("Or create .groq_api_key file with your key")
            except Exception as e:
                logger.error(f"Error reading API key file: {e}")
        
        # Clean up the key - remove quotes if present
        if api_key:
            api_key = api_key.strip().strip('"').strip("'")
        
        if api_key and not api_key.startswith('gsk_'):
            logger.warning("API key doesn't start with 'gsk_' - might be invalid")
        
        return api_key

def groq_generate(
    prompt: str, 
    model: str = ModelConfig.DEFAULT_MODEL,
    max_retries: int = ModelConfig.MAX_RETRIES,
    timeout: int = ModelConfig.TIMEOUT,
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Sends prompt to Groq API with retry mechanism and better error handling.
    
    Args:
        prompt: The prompt to send to the model
        model: Model name (key from ModelConfig.GROQ_MODELS)
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        Generated text or None if all retries fail
    """
    api_key = ModelConfig.get_api_key()
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return None
    
    model_name = ModelConfig.GROQ_MODELS.get(model, ModelConfig.GROQ_MODELS[ModelConfig.DEFAULT_MODEL])
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model_name}")
            
            response = requests.post(
                ModelConfig.GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 429:
                logger.warning("Rate limit hit, waiting before retry...")
                time.sleep(ModelConfig.RETRY_DELAY * (attempt + 1))
                continue
            elif response.status_code == 401:
                logger.error("Invalid API key")
                return None
            elif response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(ModelConfig.RETRY_DELAY)
                    continue
                return None
            
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in response")
                if attempt < max_retries - 1:
                    time.sleep(ModelConfig.RETRY_DELAY)
                    continue
                return None
            
            output = response_data["choices"][0]["message"]["content"].strip()
            
            if not output:
                logger.warning("Empty response from Groq")
                if attempt < max_retries - 1:
                    time.sleep(ModelConfig.RETRY_DELAY)
                    continue
                return None
            
            logger.info(f"Successfully generated response ({len(output)} chars)")
            return output
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout after {timeout} seconds")
            if attempt < max_retries - 1:
                time.sleep(ModelConfig.RETRY_DELAY)
                continue
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(ModelConfig.RETRY_DELAY)
                continue
            return None
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(ModelConfig.RETRY_DELAY)
                continue
            return None
    
    logger.error(f"All {max_retries} attempts failed for model {model_name}")
    return None

def build_summary_prompt(text: str, chunk_context: bool = False) -> str:
    """
    Builds a prompt for summarization with optional chunk context.
    
    Args:
        text: Text to summarize
        chunk_context: Whether this is part of a larger document
    """
    context_note = "" if not chunk_context else "\nNOTE: This is part of a larger document. Focus on the key points in this section."
    
    return f"""
You are an enterprise document analyst.

Summarize the following document text.{context_note}

Requirements:
- One executive summary paragraph
- 5–7 bullet points
- Focus on actions, requests, deadlines, and responsibilities
- Neutral and factual tone
- Do NOT add new information
- If the text is empty or contains no meaningful content, respond with "No content to summarize"

Document Text:
{text}
"""
def summarize_chunks(
    chunks: List[str], 
    model: str = ModelConfig.DEFAULT_MODEL,
    progress_callback: Optional[callable] = None
) -> List[str]:
    """
    Summarizes each chunk with error handling and progress tracking.
    
    Args:
        chunks: List of text chunks to summarize
        model: Model name to use for summarization
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of summaries (one per chunk)
    """
    summaries = []
    failed_chunks = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))
            
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        if not chunk.strip():
            logger.warning(f"Chunk {i+1} is empty, skipping")
            summaries.append("No content to summarize")
            continue
            
        prompt = build_summary_prompt(chunk, chunk_context=len(chunks) > 1)
        summary = groq_generate(prompt, model)
        
        if summary:
            # Validate summary quality
            if len(summary.strip()) < 20:
                logger.warning(f"Chunk {i+1} summary seems too short: {summary[:50]}...")
            summaries.append(summary)
        else:
            logger.error(f"Failed to summarize chunk {i+1}")
            summaries.append(f"Failed to summarize chunk {i+1}")
            failed_chunks.append(i+1)
    
    if failed_chunks:
        logger.warning(f"Failed to summarize {len(failed_chunks)} chunks: {failed_chunks}")
    
    logger.info(f"Generated {len(summaries)} summaries")
    return summaries

def generate_final_summary(
    chunk_summaries: List[str], 
    model: str = ModelConfig.DEFAULT_MODEL
) -> Optional[str]:
    """
    Combines chunk summaries into a final coherent summary.
    
    Args:
        chunk_summaries: List of chunk summaries
        model: Model name to use for final summarization
        
    Returns:
        Final summary or None if generation fails
    """
    # Filter out failed summaries
    valid_summaries = [s for s in chunk_summaries if s and not s.startswith("Failed to summarize") and s != "No content to summarize"]
    
    if not valid_summaries:
        logger.error("No valid summaries to combine")
        return None
    
    combined = "\n\n".join(valid_summaries)
    
    prompt = f"""
You are a senior document analyst.

Combine the following summaries into ONE final coherent summary.

Requirements:
- One executive paragraph
- 5–7 bullet points
- Remove redundancy
- Preserve intent and key facts
- Maintain neutral and factual tone
- Do NOT add information not present in the source summaries

Partial Summaries:
{combined}
"""
    
    logger.info(f"Generating final summary from {len(valid_summaries)} valid summaries")
    final_summary = groq_generate(prompt, model)
    
    if final_summary:
        logger.info("Final summary generated successfully")
    else:
        logger.error("Failed to generate final summary")
    
    return final_summary

def process_pdf(
    file_path: str,
    model: str = ModelConfig.DEFAULT_MODEL,
    model_context_limit: int = 4000,
    progress_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Complete PDF processing pipeline with error handling.
    
    Args:
        file_path: Path to PDF file
        model: Model name to use
        model_context_limit: Context limit for the model
        progress_callback: Optional progress callback
        
    Returns:
        Final summary or None if processing fails
    """
    try:
        logger.info(f"Starting PDF processing for: {file_path}")
        
        # Extract text
        text_content = extract_text_from_pdf(file_path)
        
        # Check if OCR is needed
        ocr_text = ""
        if needs_ocr(text_content):
            logger.info("OCR required, performing OCR...")
            ocr_text = ocr_text_from_pdf(file_path)
        else:
            logger.info("Text extraction sufficient, skipping OCR")
            ocr_text = text_content
        
        # Clean the extracted text
        cleaned_text = clean_extracted_text(ocr_text)
        logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
        
        if not cleaned_text.strip():
            logger.warning("No meaningful text extracted from PDF")
            return "No meaningful content found in PDF"
        
        # Chunking the text for model
        text_chunks = prepare_text_for_llm(
            cleaned_text,
            model_context_limit=model_context_limit,
            model_name="gpt-4"  # works as approximation for token counting
        )
        logger.info(f"Text split into {len(text_chunks)} chunks")
        
        # Summary generation
        chunk_summaries = summarize_chunks(text_chunks, model, progress_callback)
        
        # Check if we have any valid summaries
        valid_summaries = [s for s in chunk_summaries if s and not s.startswith("Failed to summarize")]
        if not valid_summaries:
            logger.error("No valid chunk summaries generated")
            return "Failed to generate any summaries from the document"
        
        final_summary = generate_final_summary(chunk_summaries, model)
        
        if final_summary:
            logger.info("PDF processing completed successfully")
        else:
            logger.error("Failed to generate final summary")
        
        return final_summary
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return f"Error processing PDF: {str(e)}"

# Example usage:
if __name__ == "__main__":
    def progress_callback(current, total):
        print(f"Progress: {current}/{total} chunks processed")
    
    file_path = "sample.pdf"
    final_summary = process_pdf(
        file_path, 
        model="llama3-8b",  # Use Groq model
        progress_callback=progress_callback
    )
    
    if final_summary:
        print("\n" + "="*50)
        print("FINAL SUMMARY:")
        print("="*50)
        print(final_summary)
    else:
        print("Failed to generate summary")