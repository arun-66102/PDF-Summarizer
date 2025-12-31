import json
import time
import requests
import logging
import os
from pathlib import Path
from typing import List, Optional

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
        # First try environment variable
        api_key = os.getenv('GROQ_API_KEY')
        
        # If not found, try reading from file
        if not api_key:
            try:
                key_file = Path(__file__).parent / '.groq_api_key'
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
