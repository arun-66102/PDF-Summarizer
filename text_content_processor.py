import re
import logging

logger = logging.getLogger(__name__)


def validate_text_input(text: str, min_length: int = 50) -> bool:
    """
    Validates whether user-provided text input is meaningful enough to process.
    
    Args:
        text: Raw text input from user
        min_length: Minimum character length to consider valid
        
    Returns:
        True if text is valid for processing, False otherwise
    """
    if not text or not text.strip():
        logger.warning("Empty text input received")
        return False
    
    stripped = text.strip()
    
    # Check minimum length
    if len(stripped) < min_length:
        logger.warning(f"Text too short ({len(stripped)} chars, minimum {min_length})")
        return False
    
    # Check if text has enough readable characters (not just symbols/numbers)
    readable = sum(c.isalpha() for c in stripped)
    if readable / len(stripped) < 0.3:
        logger.warning("Text has too few alphabetic characters")
        return False
    
    return True


def extract_text_content(raw_text: str) -> str:
    """
    Extracts and normalizes text content from raw user input.
    Similar to extract_text_from_pdf but for direct text input.
    
    Handles:
    - Line ending normalization
    - Excess whitespace cleanup
    - Leading/trailing whitespace removal
    
    Args:
        raw_text: Raw text input from user
        
    Returns:
        Normalized text content ready for cleaning pipeline
    """
    if not raw_text:
        return ""
    
    text = raw_text
    
    # Normalize line endings (Windows \r\n, old Mac \r -> Unix \n)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    # Strip leading/trailing whitespace from entire content
    text = text.strip()
    
    logger.info(f"Extracted text content: {len(text)} characters, {len(text.split())} words")
    return text


def needs_processing(text: str) -> bool:
    """
    Checks if text content needs the full processing pipeline 
    (chunking + multi-step summarization) or can be summarized directly.
    
    Args:
        text: Text content to check
        
    Returns:
        True if text needs full chunked processing, False if it can be sent directly
    """
    # If text is very short, it doesn't need chunking
    word_count = len(text.split())
    if word_count < 100:
        logger.info(f"Text is short ({word_count} words), may not need chunking")
        return False
    return True


def get_text_stats(text: str) -> dict:
    """
    Returns basic statistics about the text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'line_count': 0,
            'paragraph_count': 0
        }
    
    lines = text.split('\n')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'line_count': len(lines),
        'paragraph_count': len(paragraphs)
    }
