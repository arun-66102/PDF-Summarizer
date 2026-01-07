import logging
from typing import Optional

from pdf_extractor import extract_text_from_pdf, needs_ocr, ocr_text_from_pdf
from text_processor import clean_extracted_text, prepare_text_for_llm
from model_summarizer import summarize_chunks, generate_final_summary
from mailer import send_summary_to_department

# -------------------- LOGGER SETUP --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------- ROUTING IMPORTS --------------------
try:
    from embedding_store import classify_text_to_department_with_confidence
    ROUTING_AVAILABLE = True
    logger.info("✅ Document routing available (with embeddings)")
except ImportError as e:
    ROUTING_AVAILABLE = False
    logger.warning(f"⚠️ Document routing not available: {e}")
    logger.info("Install with: pip install sentence-transformers chromadb")

    from department_corpus import DEPARTMENT_CORPUS
<<<<<<< HEAD

    def classify_text_to_department(text, top_k=1):
        """Fallback keyword-based department routing"""
=======
    
    def classify_text_to_department_with_confidence(text, top_k=3, tie_threshold=0.05):
        """Fallback keyword-based department routing with confidence"""
>>>>>>> 18ddbc8a927c1e0c55156cb26eae55c77dc6ff48
        text_lower = text.lower()
        department_scores = {}

        for dept_code, dept_data in DEPARTMENT_CORPUS.items():
            score = 0
            for keyword in dept_data["keywords"]:
                if keyword.lower() in text_lower:
                    score += 1
            if score > 0:
                department_scores[dept_code] = score

        sorted_departments = sorted(
            department_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
<<<<<<< HEAD
=======
        
        # Handle ties for fallback
        if sorted_departments:
            primary_score = sorted_departments[0][1]['score']
            primary_departments = []
            
            for dept_code, data in sorted_departments:
                if (primary_score - data['score']) <= 0:  # Exact score match for ties
                    primary_departments.append(dept_code)
                else:
                    break
            
            return {
                'primary_departments': primary_departments,
                'all_matches': [{'department_code': code, **data} for code, data in sorted_departments],
                'confidence': primary_score / len(dept_data['keywords']),  # Normalized confidence
                'is_tie': len(primary_departments) > 1,
                'tie_threshold': tie_threshold
            }
        
        return {
            'primary_departments': [],
            'all_matches': [],
            'confidence': 0.0,
            'is_tie': False,
            'tie_threshold': tie_threshold
        }
>>>>>>> 18ddbc8a927c1e0c55156cb26eae55c77dc6ff48

        return [dept for dept, _ in sorted_departments[:top_k]]

# ======================================================
#                    MAIN PIPELINE
# ======================================================
def process_pdf(
    file_path: str,
    model: str = "llama3-8b",
    model_context_limit: int = 4000,
    progress_callback: Optional[callable] = None,
    enable_routing: bool = True
) -> dict:
    try:
        logger.info(f"Starting PDF processing for: {file_path}")

        # -------- TEXT EXTRACTION --------
        text_content = extract_text_from_pdf(file_path)

        if needs_ocr(text_content):
            logger.info("OCR required, performing OCR...")
            text_content = ocr_text_from_pdf(file_path)

        cleaned_text = clean_extracted_text(text_content)

        if not cleaned_text.strip():
            return {
                "summary": "No meaningful content found",
                "routing": None,
                "error": "Empty document"
            }

        # -------- CHUNKING --------
        text_chunks = prepare_text_for_llm(
            cleaned_text,
            model_context_limit=model_context_limit,
            model_name="gpt-4"
        )

        # -------- SUMMARIZATION --------
        chunk_summaries = summarize_chunks(text_chunks, model, progress_callback)
        final_summary = generate_final_summary(chunk_summaries, model)

        if not final_summary:
            return {
                "summary": "Final summarization failed",
                "routing": None,
                "error": "Summarization error"
            }

        logger.info("Final summary generated successfully")

        # -------- ROUTING --------
        routing_result = []
        primary_department = None

        if enable_routing:
            routing_result = classify_text_to_department(final_summary, top_k=3)
            primary_department = routing_result[0] if routing_result else None
            logger.info(f"Primary Department: {primary_department}")

        # -------- EMAIL --------
        if primary_department:
            try:
<<<<<<< HEAD
                send_summary_to_department(
                    summary=final_summary,
                    department=primary_department,
                    document_name="C:\Users\indhu\Downloads\STL_Delete_Functions_Explanation.pdf"
                )
            except Exception as e:
                logger.error(f"Email sending failed: {e}")

        logger.info("PDF processing completed successfully")

        return {
=======
                logger.info("Routing document to department...")
                routing_result = classify_text_to_department_with_confidence(final_summary, top_k=3)
                
                primary_depts = routing_result.get('primary_departments', [])
                logger.info(f"Document routed to {len(primary_depts)} primary department(s): {primary_depts}")
                
                if routing_result.get('is_tie'):
                    logger.info(f"Tie detected between departments: {primary_depts}")
                
            except Exception as e:
                logger.error(f"Routing failed: {e}")
                routing_result = {
                    'primary_departments': [],
                    'all_matches': [],
                    'confidence': 0.0,
                    'is_tie': False,
                    'tie_threshold': 0.05
                }
        
        # Prepare result
        result = {
>>>>>>> 18ddbc8a927c1e0c55156cb26eae55c77dc6ff48
            "summary": final_summary,
            "file_path": file_path,
            "text_length": len(cleaned_text),
            "chunks_processed": len(text_chunks),
            "model_used": model,
            "routing": {
<<<<<<< HEAD
                "departments": routing_result,
                "primary_department": primary_department,
                "method": "embedding" if ROUTING_AVAILABLE else "keyword",
=======
                "primary_departments": routing_result.get('primary_departments', []) if routing_result else [],
                "all_matches": routing_result.get('all_matches', []) if routing_result else [],
                "confidence": routing_result.get('confidence', 0.0) if routing_result else 0.0,
                "is_tie": routing_result.get('is_tie', False) if routing_result else False,
                "tie_threshold": routing_result.get('tie_threshold', 0.05) if routing_result else 0.05,
                "method": "embedding" if ROUTING_AVAILABLE else "keyword_match",
>>>>>>> 18ddbc8a927c1e0c55156cb26eae55c77dc6ff48
                "available": ROUTING_AVAILABLE
            }
        }

    except Exception as e:
        logger.exception("Unhandled error during PDF processing")
        return {
            "summary": str(e),
            "routing": None,
            "error": str(e)
        }


# ================== ROUTING SANITY TEST ==================
if __name__ == "__main__":
<<<<<<< HEAD
    test_text = "This document discusses machine learning, algorithms, and software development."
    print("Routing test output:")
    print(classify_text_to_department(test_text, top_k=1))
=======
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
        primary_depts = routing.get('primary_departments', [])
        
        if primary_depts:
            print("\n" + "="*40)
            print("DEPARTMENT ROUTING")
            print("="*40)
            
            if len(primary_depts) == 1:
                print(f"🎯 {primary_depts[0]}")
            else:
                print(f"🎯 {', '.join(primary_depts)}")
        else:
            print("\n" + "="*40)
            print("DEPARTMENT ROUTING")
            print("="*40)
            print("❌ No departments matched")
    
    print("\n" + "="*60)
>>>>>>> 18ddbc8a927c1e0c55156cb26eae55c77dc6ff48
