# AI-Powered Document Processing System

An intelligent document processing system that automates the extraction, summarization, and routing of PDF documents using cloud-based AI models via Groq API.

## Features

- **PDF Text Extraction**: Automatically extracts text from PDFs using OCR (Optical Character Recognition)
- **Cloud-Powered Summarization**: Generates concise summaries using Groq's fast Llama models
- **Multiple Model Options**: Choose from Llama 3, Llama 3.1, and Mixtral models
- **Intelligent Chunking**: Handles large documents with token-aware splitting
- **Retry Mechanism**: Built-in error handling with automatic retries
- **Rate Limit Handling**: Manages API limits with exponential backoff
- **Progress Tracking**: Monitor processing progress for long documents

## Prerequisites

- Python 3.8+
- Tesseract OCR (for OCR functionality)
- Groq API key (sign up at https://groq.com/)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PDF-Summarizer.git
   cd PDF-Summarizer
   ```

2. Install Tesseract OCR:
   - Windows: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up Groq API key:
   ```bash
   # Windows (Command Prompt)
   set GROQ_API_KEY=your_api_key_here
   
   # Windows (PowerShell)
   $env:GROQ_API_KEY="your_api_key_here"
   ```

## Usage

### Basic Usage
```python
from pdf_ocr import process_pdf

summary = process_pdf("document.pdf", model="llama3-8b")
print(summary)
```

### With Progress Tracking
```python
def progress_callback(current, total):
    print(f"Progress: {current}/{total} chunks processed")

summary = process_pdf(
    "document.pdf", 
    model="llama3-70b",
    progress_callback=progress_callback
)
```

### Available Models
- `llama3-8b` - Llama 3.1 8B Instant - **Default & Fastest**
- `llama3-70b` - Llama 3.3 70B Versatile - **Higher Quality**
- `llama-guard` - Llama Guard 4 12B - Content filtering
- `gpt-oss-120b` - OpenAI GPT-OSS 120B
- `gpt-oss-20b` - OpenAI GPT-OSS 20B
- `whisper` - Whisper Large V3 - Speech-to-text
- `whisper-turbo` - Whisper Large V3 Turbo - Faster speech-to-text

### Command Line Usage
```bash
python pdf_ocr.py
```

## Configuration

### Model Selection
Choose models based on your needs:
- **Speed**: `llama3-8b` (Llama 3.1 8B Instant)
- **Quality**: `llama3-70b` (Llama 3.3 70B Versatile)
- **OpenAI Compatible**: `gpt-oss-20b` or `gpt-oss-120b`
- **Content Safety**: `llama-guard` for filtering

### API Settings
The system automatically handles:
- Rate limiting with exponential backoff
- Retry logic for failed requests
- Timeout management
- Error logging

## Benefits over Local Models

- ✅ **Much faster inference** (Groq's optimized infrastructure)
- ✅ **No local GPU/CPU requirements**
- ✅ **Better reliability and uptime**
- ✅ **Access to latest models**
- ✅ **Production-ready infrastructure**
- ✅ **Built-in monitoring and logging**

## Project Structure

```
PDF-Summarizer/
├── config/               # Configuration files
├── input/                # Input PDF documents
├── output/               # Processed documents and summaries
├── src/                  # Source code
│   ├── __init__.py
│   ├── document_processor.py  # Main processing logic
│   ├── ocr.py            # OCR functionality
│   ├── summarizer.py     # Text summarization
│   └── classifier.py     # Department classification
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Arunkumar - [@LinkedIn](https://www.linkedin.com/in/arunkumar-rathinasamy-844085290/)

Project Link: [https://github.com/arun-66102/PDF-Summarizer](https://github.com/arun-66102/PDF-Summarizer)
