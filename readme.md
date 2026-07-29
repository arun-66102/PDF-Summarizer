# RouteX — AI-Powered Document Processing & Routing

> Intelligent document processing that **summarizes**, **classifies**, and **routes** PDFs to the right department — automatically.

RouteX is a full-stack AI application powered by Groq's cloud inference. Upload a PDF or paste text through a sleek chat interface and RouteX will extract content, generate a concise summary, classify it to the appropriate engineering department, and optionally deliver it via email — all in seconds.

---

## ✨ Features

- **Chat-Style UI** — Streamlit-powered interface with a Chocolate Truffle theme; send PDFs or raw text like a message
- **FastAPI REST Backend** — A full REST API with JWT authentication and Server-Sent Events (SSE) progress streaming
- **React / Vite Frontend** — Modern single-page app connecting to the FastAPI backend
- **PDF Text Extraction** — Extracts text from PDFs via PyMuPDF with OCR fallback (Tesseract)
- **Cloud Summarization** — Groq-hosted Llama and GPT-OSS models for fast, high-quality summaries
- **Intelligent Department Routing** — Sentence-transformer embeddings classify documents to the correct department
- **Multi-Department Tie Handling** — Detects equal scores and routes to multiple departments
- **Automated Email Delivery** — Sends the PDF and summary directly to department inboxes
- **SSE Progress Streaming** — Real-time chunk-by-chunk progress updates over a live HTTP stream
- **Retry & Rate-Limit Handling** — Exponential back-off for Groq API limits
- **Download Summary** — One-click `.txt` download of any generated summary

---

## 🏗️ Architecture

```
RouteX/
├── app.py                     # Streamlit chat UI (standalone, all-in-one)
├── main.py                    # Core processing pipeline (process_pdf / process_text)
├── run_dev.py                 # Dev runner — starts FastAPI + Vite together
│
├── backend/                   # FastAPI REST API
│   ├── server.py              #   REST endpoints, SSE stream, CORS
│   ├── auth.py                #   JWT authentication (no external dependency)
│   ├── schemas.py             #   Pydantic request/response models
│   └── requirements.txt       #   Backend-specific packages
│
├── frontend/                  # React + Vite SPA
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api/               #   Axios API client
│   │   ├── components/        #   UI components
│   │   ├── hooks/             #   Custom React hooks
│   │   └── styles/
│   └── package.json
│
├── pdf_extractor.py           # PDF text extraction & OCR
├── text_processor.py          # Text cleaning and token-aware chunking
├── model_summarizer.py        # Groq API integration
├── embedding_store.py         # Sentence-transformer department classifier
├── department_corpus.py       # Department knowledge base
├── mailer.py                  # SMTP email delivery
├── text_content_processor.py  # Text-input processing helpers
│
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
├── LINUX_SETUP.md             # Linux / Docker setup guide
└── README.md                  # This file
```

---

## 🛠️ Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | |
| Node.js 18+ | For the React/Vite frontend |
| Tesseract OCR | Only needed for scanned/image PDFs |
| Groq API key | Free at [groq.com](https://groq.com/) |
| Gmail App Password | For automated email delivery (optional) |

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/arun-66102/PDF-Summarizer.git
cd PDF-Summarizer
```

### 2. Install system dependencies

**Tesseract OCR** (for scanned PDFs):
- **Windows** — Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS** — `brew install tesseract`
- **Linux / Docker** — See [LINUX_SETUP.md](LINUX_SETUP.md)

### 3. Create a Python virtual environment

```bash
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
.\venv\Scripts\activate         # Windows
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY     = "your_groq_api_key"
EMAIL_SENDER     = "you@gmail.com"
EMAIL_PASSWORD   = "your_gmail_app_password"
SMTP_SERVER      = "smtp.gmail.com"
SMTP_PORT        = 587
```

> **Optional auth overrides** (FastAPI backend):
> ```env
> ROUTEX_USERNAME   = admin          # default: admin
> ROUTEX_PASSWORD   = routex2026     # default: routex2026
> ROUTEX_JWT_SECRET = <random>       # auto-generated if omitted
> ```

---

## 🚀 Running RouteX

### Development (FastAPI + React)

Install Node dependencies once:

```bash
cd frontend && npm install && cd ..
```

Then start both servers with a single command:

```bash
python run_dev.py
```

| Service | URL |
|---|---|
| React frontend | http://localhost:5173 |
| FastAPI backend | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

### Backend only

```bash
uvicorn backend.server:app --reload --port 8000
```

---

## 🌐 Deploying to Vercel

RouteX is configured for 1-click zero-config deployment on Vercel using `vercel.json`.

1. Install Vercel CLI or connect your GitHub repository to Vercel.
2. Add the following **Environment Variables** in Vercel project settings:
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`)
   - `EMAIL_SENDER`: Sender email address (e.g. `your_email@gmail.com`)
   - `EMAIL_PASSWORD`: Gmail App Password
   - `SMTP_SERVER`: `smtp.gmail.com`
   - `SMTP_PORT`: `587`
3. Deploy:

```bash
vercel --prod
```

> **Serverless Note**: Large embedding libraries (`sentence-transformers`, `chromadb`) are automatically omitted in `api/requirements.txt` to keep the deployment within Vercel's serverless size limit (250MB). RouteX seamlessly falls back to keyword-based department routing.

---

## 🔌 REST API

All protected endpoints require a `Authorization: Bearer <token>` header.

### Auth

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Login — returns JWT token |

```json
// POST /api/auth/login
{ "username": "admin", "password": "routex2026" }
```

### Processing

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/process/text` | Summarize & route raw text |
| `POST` | `/api/process/pdf` | Upload & process a PDF file |
| `GET` | `/api/process/stream/{task_id}` | SSE stream of task progress |

### Email & Utilities

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/email` | Send results to routed departments |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/models` | List available AI models |

---

## 🤖 Available Models

| Key | Model | Best For |
|---|---|---|
| `llama3-8b` *(default)* | Llama 3.1 8B Instant | Speed |
| `llama3-70b` | Llama 3.3 70B Versatile | Quality |
| `llama-guard` | Llama Guard 4 12B | Content filtering |
| `gpt-oss-20b` | OpenAI GPT-OSS 20B | OpenAI-compatible |
| `gpt-oss-120b` | OpenAI GPT-OSS 120B | High-quality OSS |

---

## 🎯 Department Routing

Documents are automatically classified using **sentence-transformer embeddings** and matched against department knowledge corpora.

| Code | Department |
|---|---|
| `CSE` | Computer Science & Engineering |
| `EEE` | Electrical & Electronics Engineering |
| `MECH` | Mechanical Engineering |
| `CIVIL` | Civil Engineering |

**Routing features:**
- Confidence scoring for each department match
- Tie detection — routes to multiple departments when scores are equal
- Keyword fallback when embeddings are unavailable

### Configuring department email addresses

Edit `mailer.py`:

```python
DEPARTMENT_EMAILS = {
    "CSE":   "cse.department@university.edu",
    "EEE":   "eee.department@university.edu",
    "MECH":  "mech.department@university.edu",
    "CIVIL": "civil.department@university.edu",
}
```

---

## 🖥️ Sidebar Settings

The sidebar in the React frontend exposes the following controls at runtime:

| Setting | Description |
|---|---|
| **Model** | Select Groq model for summarization |
| **Context limit** | Token window (1,000 – 8,000) |
| **Department routing** | Toggle embedding-based classification |
| **Email delivery** | Toggle automated email sending |

Status badges indicate whether Groq and email are correctly configured.

---

## 📦 Python Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` | PDF parsing |
| `pytesseract` | OCR for scanned PDFs |
| `pdf2image` | PDF-to-image conversion |
| `opencv-python-headless` | Image processing |
| `tiktoken` | Token counting |
| `sentence-transformers` | Department classification embeddings |
| `chromadb` | Vector store |
| `python-dotenv` | `.env` file loading |
| `fastapi` + `uvicorn` | REST API server |
| `sse-starlette` | Server-Sent Events streaming |
| `python-multipart` | File upload support |
| `requests` | HTTP client for Groq API |

---

## 🔄 Processing Workflow

```
Upload PDF / Paste Text
        │
        ▼
  Extract Text (PyMuPDF + OCR)
        │
        ▼
  Clean & Chunk (token-aware)
        │
        ▼
  Summarize via Groq API ──► SSE Progress Stream
        │
        ▼
  Classify Department (embeddings)
        │
        ▼
  Email PDF + Summary to Department(s)
        │
        ▼
  Display Results in Chat / Return JSON
```

---

## 💬 Programmatic Usage

```python
from main import process_pdf, process_text

# Process a PDF
result = process_pdf("document.pdf", model="llama3-8b")
print(result["summary"])
print("Routed to:", result["routing"]["primary_departments"])

# Process raw text
result = process_text("Paste your text here...", model="llama3-70b")
print(result["summary"])
```

With progress tracking:

```python
def on_progress(current, total):
    print(f"  Chunk {current}/{total}")

result = process_pdf(
    "report.pdf",
    model="llama3-70b",
    progress_callback=on_progress,
    enable_routing=True,
)
```

---

## 🐧 Linux / Docker

See **[LINUX_SETUP.md](LINUX_SETUP.md)** for:
- System dependency installation (Ubuntu, CentOS, Alpine)
- Complete Docker setup with a production-ready `Dockerfile`
- Troubleshooting `libGL.so.1` and Tesseract errors

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**.

---

## 👤 Contact

**Arunkumar** — [@LinkedIn](https://www.linkedin.com/in/arunkumar-rathinasamy-844085290/)

*Powered by Groq AI · Built by RAG Retrievers*
