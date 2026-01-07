# AI-Powered Document Processing System

An intelligent document processing system that automates extraction, summarization, department routing, and email delivery of PDF documents using cloud-based AI models via Groq API.

## Features

- **PDF Text Extraction**: Automatically extracts text from PDFs using OCR (Optical Character Recognition)
- **Cloud-Powered Summarization**: Generates concise summaries using Groq's fast Llama models
- **Intelligent Department Routing**: Automatically routes documents to appropriate departments using embeddings
- **Multiple Primary Departments**: Handles ties and routes to multiple departments when needed
- **Automated Email Delivery**: Sends PDFs and summaries to department email addresses
- **Multiple Model Options**: Choose from Llama 3, Llama 3.1, and Mixtral models
- **Intelligent Chunking**: Handles large documents with token-aware splitting
- **Retry Mechanism**: Built-in error handling with automatic retries
- **Rate Limit Handling**: Manages API limits with exponential backoff
- **Progress Tracking**: Monitor processing progress for long documents

## Prerequisites

- Python 3.8+
- Tesseract OCR (for OCR functionality)
- Groq API key (sign up at https://groq.com/)
- Email configuration (SMTP settings)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/arun-66102/PDF-Summarizer.git
   cd PDF-Summarizer
   ```

2. Install Tesseract OCR:
   - **Windows**: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr` (Ubuntu/Debian) or `sudo yum install tesseract` (CentOS/RHEL)
   - **Docker**: `apt-get update && apt-get install -y tesseract-ocr`

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```bash
   # Groq API Key
   # Windows (Command Prompt)
   set GROQ_API_KEY=your_api_key_here
   
   # Windows (PowerShell)
   $env:GROQ_API_KEY="your_api_key_here"
   
   # Email Configuration (for automated delivery)
   set EMAIL_SENDER=your_email@gmail.com
   set EMAIL_PASSWORD=your_app_password
   set SMTP_SERVER=smtp.gmail.com
   set SMTP_PORT=587
   ```

## Usage

### Basic Usage
```python
from main import process_pdf

result = process_pdf("document.pdf", model="llama3-8b")
print(result['summary'])
print("Routed to:", result['routing']['primary_departments'])
```

### With Progress Tracking
```python
def progress_callback(current, total):
    print(f"Progress: {current}/{total} chunks processed")

result = process_pdf(
    "document.pdf", 
    model="llama3-70b",
    progress_callback=progress_callback,
    enable_routing=True  # Enable department routing and email
)
```

### Command Line Usage
```bash
python main.py
```

## Available Models

- `llama3-8b` - Llama 3.1 8B Instant - **Default & Fastest**
- `llama3-70b` - Llama 3.3 70B Versatile - **Higher Quality**
- `llama-guard` - Llama Guard 4 12B - Content filtering
- `gpt-oss-120b` - OpenAI GPT-OSS 120B
- `gpt-oss-20b` - OpenAI GPT-OSS 20B
- `whisper` - Whisper Large V3 - Speech-to-text
- `whisper-turbo` - Whisper Large V3 Turbo - Faster speech-to-text

## Department Routing

The system automatically routes documents to departments based on content:

### Supported Departments
- **CSE** - Computer Science and Engineering
- **EEE** - Electrical and Electronics Engineering  
- **MECH** - Mechanical Engineering
- **CIVIL** - Civil Engineering

### Routing Features
- **Embedding-based Classification**: Uses sentence transformers for accurate classification
- **Tie Handling**: Detects when documents match multiple departments equally
- **Confidence Scoring**: Provides confidence levels for routing decisions
- **Fallback Keywords**: Uses keyword matching when embeddings unavailable

### Email Integration
- **Automated Delivery**: Sends PDFs and summaries to routed departments
- **Multiple Recipients**: Handles ties by sending to multiple departments
- **Professional Templates**: Uses department-specific email formatting
- **PDF Attachments**: Includes original PDF files in emails

## Configuration

### Department Email Mapping
Edit `mailer.py` to configure department email addresses:
```python
DEPARTMENT_EMAILS = {
    "CSE": "cse.department@university.edu",
    "EEE": "eee.department@university.edu",
    "MECH": "mech.department@university.edu", 
    "CIVIL": "civil.department@university.edu"
}
```

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

## Project Structure

```
PDF-Summarizer/
├── main.py                    # Main processing pipeline with routing
├── pdf_extractor.py           # PDF text extraction and OCR
├── text_processor.py          # Text cleaning and chunking
├── model_summarizer.py       # Groq API integration and summarization
├── embedding_store.py        # Department classification using embeddings
├── department_corpus.py      # Department knowledge base
├── mailer.py                # Email delivery system
├── requirements.txt          # Python dependencies
├── sample.pdf               # Sample document for testing
├── EEE.pdf                 # Sample EEE document
├── workflow.png             # Workflow diagram
├── summary.png              # Summary diagram
└── README.md               # This file
```

## Workflow

1. **PDF Processing**: Extract text using OCR if needed
2. **Text Cleaning**: Clean and prepare text for processing
3. **Chunking**: Split large documents into manageable chunks
4. **Summarization**: Generate summary using Groq AI models
5. **Department Routing**: Classify document to appropriate departments
6. **Email Delivery**: Send PDF and summary to routed departments
7. **Status Reporting**: Provide clear success/failure feedback

## Output Format

```
============================================================
PDF PROCESSING RESULTS
============================================================
 File: document.pdf
 Text Length: 1719 characters
 Chunks Processed: 1
 Model Used: llama3-8b

SUMMARY
============================================================
[Generated summary content]

DEPARTMENT ROUTING
============================================================
 CSE, EEE

EMAIL SENDING
============================================================
 Sending PDF to 2 department(s): CSE, EEE
 Email sent to CSE
 Email sent to EEE
 Email Summary: 2/2 sent successfully
```

## Benefits over Local Models

- **Much faster inference** (Groq's optimized infrastructure)
- **No local GPU/CPU requirements**
- **Better reliability and uptime**
- **Access to latest models**
- **Production-ready infrastructure**
- **Built-in monitoring and logging**
- **Automated department routing**
- **Email delivery system**

## Contributing

1. Fork repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under MIT License. See `LICENSE` for more information.

## Contact

Arunkumar - [@LinkedIn](https://www.linkedin.com/in/arunkumar-rathinasamy-844085290/)
