import logging
from typing import Optional
from pdf_extractor import extract_text_from_pdf, needs_ocr, ocr_text_from_pdf
from text_processor import clean_extracted_text, prepare_text_for_llm
from model_summarizer import summarize_chunks, generate_final_summary

# Document routing imports
try:
    from embedding_store import classify_text_to_department
    ROUTING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Document routing available (with embeddings)")
except ImportError as e:
    ROUTING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Document routing not available: {e}")
    logger.info("   Install with: pip install sentence-transformers chromadb")
    
    # Fallback keyword-based routing
    from department_corpus import DEPARTMENT_CORPUS
    
    def classify_text_to_department(text, top_k=1):
        """Fallback keyword-based department routing"""
        text_lower = text.lower()
        department_scores = {}
        
        for dept_code, dept_data in DEPARTMENT_CORPUS.items():
            score = 0
            matched_keywords = []
            
            for keyword in dept_data['keywords']:
                if keyword.lower() in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                department_scores[dept_code] = {
                    'score': score,
                    'department_name': dept_data['department_name'],
                    'matched_keywords': matched_keywords
                }
        
        # Sort by score (highest first)
        sorted_departments = sorted(
            department_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # Return department codes only
        return [dept_code for dept_code, data in sorted_departments[:top_k]]

def process_pdf(
    file_path: str,
    model: str = "llama3-8b",
    model_context_limit: int = 4000,
    progress_callback: Optional[callable] = None,
    enable_routing: bool = True
) -> dict:
    """
    Complete PDF processing pipeline with error handling.
    
    Args:
        file_path: Path to PDF file
        model: Model name to use
        model_context_limit: Context limit for the model
        progress_callback: Optional progress callback
        enable_routing: Whether to enable department routing
        
    Returns:
        Dictionary with summary and routing information
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
            return {
                "summary": "No meaningful content found in PDF",
                "routing": None,
                "error": "No content extracted"
            }
        
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
            return {
                "summary": "Failed to generate any summaries from the document",
                "routing": None,
                "error": "Summarization failed"
            }
        
        # Generate final summary
        final_summary = generate_final_summary(chunk_summaries, model)
        
        if not final_summary:
            logger.error("Failed to generate final summary")
            return {
                "summary": "Failed to generate final summary",
                "routing": None,
                "error": "Final summarization failed"
            }
        
        logger.info("PDF processing completed successfully")
        
        # Document routing (if enabled)
        routing_result = None
        if enable_routing and ROUTING_AVAILABLE:
            try:
                logger.info("Routing document to department...")
                routing_result = classify_text_to_department(final_summary, top_k=3)
                logger.info(f"Document routed to {len(routing_result)} departments")
            except Exception as e:
                logger.error(f"Routing failed: {e}")
                routing_result = []
        
        # Prepare result
        result = {
            "summary": final_summary,
            "file_path": file_path,
            "text_length": len(cleaned_text),
            "chunks_processed": len(text_chunks),
            "model_used": model,
            "routing": {
                "departments": routing_result if routing_result else [],
                "primary_department": routing_result[0] if routing_result else None,
                "method": "embedding" if ROUTING_AVAILABLE else "keyword_match",
                "available": ROUTING_AVAILABLE
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            "summary": f"Error processing PDF: {str(e)}",
            "routing": None,
            "error": str(e)
        }

# Example usage:
if __name__ == "__main__":
    def progress_callback(current, total):
        print(f"Progress: {current}/{total} chunks processed")
    
    file_path = "sample.pdf"
    
    # Process PDF with routing enabled
    result = process_pdf(
        file_path, 
        model="llama3-8b",  # Use Groq model
        progress_callback=progress_callback,
        enable_routing=True  # Enable department routing
    )
    
    # Display results
    print("\n" + "="*60)
    print("PDF PROCESSING RESULTS")
    print("="*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"📄 File: {result['file_path']}")
        print(f"📝 Text Length: {result['text_length']} characters")
        print(f"🔢 Chunks Processed: {result['chunks_processed']}")
        print(f"🤖 Model Used: {result['model_used']}")
        
        print("\n" + "="*40)
        print("SUMMARY")
        print("="*40)
        print(result['summary'])
        
        # Display routing information
        routing = result.get('routing', {})
        departments = routing.get('departments', [])
        
        if departments:
            print("\n" + "="*40)
            print("DEPARTMENT ROUTING")
            print("="*40)
            
            primary_dept = routing.get('primary_department')
            if primary_dept:
                print(f"🎯 Primary Department: {primary_dept}")
                print(f"🔍 Method: {routing.get('method', 'unknown')}")
            
            print(f"\n📋 Matched Department Codes ({len(departments)}):")
            for i, dept_code in enumerate(departments, 1):
                print(f"  {i}. {dept_code}")
        else:
            print("\n" + "="*40)
            print("DEPARTMENT ROUTING")
            print("="*40)
            if routing.get('available'):
                print("⚠️  No departments matched this document")
            else:
                print("❌ Routing disabled or unavailable")
    
    print("\n" + "="*60)