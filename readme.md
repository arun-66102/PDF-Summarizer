# AI-Powered Document Processing System

An intelligent document processing system that automates the extraction, summarization, and routing of PDF documents to the appropriate departments using AI and OCR technologies.

## Features

- **PDF Text Extraction**: Automatically extracts text from PDFs using OCR (Optical Character Recognition)
- **AI-Powered Summarization**: Generates concise summaries of document content using advanced NLP models
- **Department Classification**: Automatically identifies the most relevant department based on document content
- **Intelligent Routing**: Forwards processed documents to the appropriate department for further action
- **Efficient Workflow**: Reduces manual effort and improves document handling efficiency

## Prerequisites

- Python 3.8+
- Tesseract OCR (for OCR functionality)
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

## Usage

1. Place your PDF files in the `input/` directory
2. Run the main processing script:
   ```bash
   python main.py
   ```
3. Processed documents will be saved in the `output/` directory
4. Check the logs in `logs/` for processing details

## Configuration

Edit the `config.yaml` file to customize:
- Department classification rules
- Summary length and style
- Output format and location
- Logging preferences

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

Project Link: [https://github.com/yourusername/PDF-Summarizer](https://github.com/arun-66102/PDF-Summarizer)
