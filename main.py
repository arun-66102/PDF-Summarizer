import logging
from typing import Optional
from pdf_extractor import extract_text_from_pdf, needs_ocr, ocr_text_from_pdf
from text_processor import clean_extracted_text, prepare_text_for_llm
from model_summarizer import summarize_chunks, generate_final_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(
    file_path: str,
    model: str = "llama3-8b",
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
