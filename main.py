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
    from embedding_store import classify_text_to_department
    ROUTING_AVAILABLE = True
    logger.info("✅ Document routing available (with embeddings)")
except ImportError as e:
    ROUTING_AVAILABLE = False
    logger.warning(f"⚠️ Document routing not available: {e}")
    logger.info("Install with: pip install sentence-transformers chromadb")

    from department_corpus import DEPARTMENT_CORPUS

    def classify_text_to_department(text, top_k=1):
        """Fallback keyword-based department routing"""
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
                send_summary_to_department(
                    summary=final_summary,
                    department=primary_department,
                    document_name="C:\Users\indhu\Downloads\STL_Delete_Functions_Explanation.pdf"
                )
            except Exception as e:
                logger.error(f"Email sending failed: {e}")

        logger.info("PDF processing completed successfully")

        return {
            "summary": final_summary,
            "file_path": file_path,
            "text_length": len(cleaned_text),
            "chunks_processed": len(text_chunks),
            "model_used": model,
            "routing": {
                "departments": routing_result,
                "primary_department": primary_department,
                "method": "embedding" if ROUTING_AVAILABLE else "keyword",
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
    test_text = "This document discusses machine learning, algorithms, and software development."
    print("Routing test output:")
    print(classify_text_to_department(test_text, top_k=1))
