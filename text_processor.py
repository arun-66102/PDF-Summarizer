import re
import tiktoken
from typing import List
from collections import Counter

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
