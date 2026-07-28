"""
RouteX — FastAPI backend server.

Wraps the existing processing pipeline (main.py) with REST endpoints,
JWT authentication, and SSE progress streaming.

Start with:
    uvicorn backend.server:app --reload --port 8000
"""

import os
import sys
import uuid
import asyncio
import tempfile
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import json

# Ensure project root is on sys.path so we can import main, pdf_extractor, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import process_pdf, process_text, send_pdf_to_departments, ROUTING_AVAILABLE
from backend.schemas import (
    LoginRequest, TokenResponse,
    ProcessTextRequest, ProcessResponse,
    EmailRequest, EmailResponse,
    HealthResponse, ModelsResponse, ModelInfo,
    RoutingResult,
)
from backend.auth import authenticate, create_token, require_auth, _TOKEN_EXPIRY

logger = logging.getLogger(__name__)

# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RouteX API",
    description="AI-powered document processing, summarization, and department routing",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory task progress store ───────────────────────────────────────────────
# task_id -> { "progress": 0-100, "status": str, "result": dict|None, "done": bool }
_tasks: dict[str, dict] = {}

MODEL_OPTIONS = {
    "llama3-8b": "Llama 3.1 8B · Fast",
    "llama3-70b": "Llama 3.3 70B · Quality",
    "llama-guard": "Llama Guard 4 · Filter",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
}


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    token = authenticate(body.username, body.password)
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=_TOKEN_EXPIRY,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & MODELS (public — no auth required)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse)
async def health():
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    email_ok = bool(os.getenv("EMAIL_SENDER") and os.getenv("EMAIL_PASSWORD"))
    status = "ok" if groq_ok else ("degraded" if email_ok else "error")
    return HealthResponse(
        groq_connected=groq_ok,
        email_configured=email_ok,
        routing_available=ROUTING_AVAILABLE,
        status=status,
    )


@app.get("/api/models", response_model=ModelsResponse)
async def models():
    return ModelsResponse(
        models=[ModelInfo(key=k, label=v) for k, v in MODEL_OPTIONS.items()],
        default="llama3-8b",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS TEXT (authenticated)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process/text", response_model=ProcessResponse)
async def api_process_text(
    body: ProcessTextRequest,
):
    """Process raw text input — summarize and route."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"progress": 0, "status": "Starting...", "result": None, "done": False}

    def progress_cb(current: int, total: int):
        pct = int(10 + (current / max(total, 1)) * 70)
        _tasks[task_id]["progress"] = pct
        _tasks[task_id]["status"] = f"Processing chunk {current}/{total}..."

    _tasks[task_id]["progress"] = 5
    _tasks[task_id]["status"] = "Preparing text..."

    result = await asyncio.to_thread(
        process_text,
        text_content=body.text,
        model=body.model,
        model_context_limit=body.context_limit,
        progress_callback=progress_cb,
        enable_routing=body.enable_routing,
    )

    _tasks[task_id]["progress"] = 100
    _tasks[task_id]["status"] = "Done"
    _tasks[task_id]["done"] = True
    _tasks[task_id]["result"] = result

    # Build response
    routing_data = result.get("routing")
    routing = RoutingResult(**routing_data) if routing_data else None

    return ProcessResponse(
        summary=result.get("summary", ""),
        text_length=result.get("text_length"),
        chunks_processed=result.get("chunks_processed"),
        model_used=result.get("model_used"),
        source=result.get("source", "text_input"),
        routing=routing,
        text_stats=result.get("text_stats"),
        error=result.get("error"),
        task_id=task_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS PDF (authenticated, multipart upload)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/process/pdf", response_model=ProcessResponse)
async def api_process_pdf(
    file: UploadFile = File(...),
    model: str = Form(default="llama3-8b"),
    context_limit: int = Form(default=4000),
    enable_routing: bool = Form(default=True),
):
    """Upload and process a PDF — extract, summarize, and route."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"progress": 0, "status": "Uploading...", "result": None, "done": False}

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    def progress_cb(current: int, total: int):
        pct = int(10 + (current / max(total, 1)) * 70)
        _tasks[task_id]["progress"] = pct
        _tasks[task_id]["status"] = f"Processing chunk {current}/{total}..."

    _tasks[task_id]["progress"] = 5
    _tasks[task_id]["status"] = "Extracting text from PDF..."

    try:
        result = await asyncio.to_thread(
            process_pdf,
            file_path=tmp_path,
            model=model,
            model_context_limit=context_limit,
            progress_callback=progress_cb,
            enable_routing=enable_routing,
        )
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    _tasks[task_id]["progress"] = 100
    _tasks[task_id]["status"] = "Done"
    _tasks[task_id]["done"] = True
    _tasks[task_id]["result"] = result
    # Keep tmp_path for potential email sending — store it
    _tasks[task_id]["tmp_path"] = tmp_path

    routing_data = result.get("routing")
    routing = RoutingResult(**routing_data) if routing_data else None

    return ProcessResponse(
        summary=result.get("summary", ""),
        text_length=result.get("text_length"),
        chunks_processed=result.get("chunks_processed"),
        model_used=result.get("model_used"),
        source="pdf",
        routing=routing,
        file_path=tmp_path,
        error=result.get("error"),
        task_id=task_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SSE PROGRESS STREAM
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/process/stream/{task_id}")
async def stream_progress(task_id: str):
    """SSE endpoint streaming progress updates for a processing task."""

    async def event_generator():
        while True:
            task = _tasks.get(task_id)
            if task is None:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Task not found"}),
                }
                break

            yield {
                "event": "progress",
                "data": json.dumps({
                    "progress": task["progress"],
                    "status": task["status"],
                    "done": task["done"],
                }),
            }

            if task["done"]:
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


# ═══════════════════════════════════════════════════════════════════════════════
# EMAIL (authenticated)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/email", response_model=EmailResponse)
async def send_email(
    body: EmailRequest,
):
    """Send processed results to routed departments via email."""
    primary_depts = body.routing.primary_departments
    if not primary_depts:
        return EmailResponse(
            success=False,
            message="No departments to send to",
        )

    pdf_path = body.pdf_path
    if not pdf_path or not os.path.exists(pdf_path):
        return EmailResponse(
            success=False,
            message="PDF file not found on server. Re-upload and process first.",
        )

    routing_dict = body.routing.model_dump()

    success = await asyncio.to_thread(
        send_pdf_to_departments,
        pdf_path,
        body.summary,
        routing_dict,
    )

    if success:
        return EmailResponse(
            success=True,
            sent_to=primary_depts,
            message=f"Emails sent to {len(primary_depts)} department(s)",
        )
    else:
        return EmailResponse(
            success=False,
            failed=primary_depts,
            message="Email sending failed",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    logger.info("RouteX API starting up...")
    logger.info(f"Groq API key: {'✅ set' if os.getenv('GROQ_API_KEY') else '❌ missing'}")
    logger.info(f"Routing available: {'✅' if ROUTING_AVAILABLE else '❌'}")


@app.on_event("shutdown")
async def shutdown():
    # Clean up any remaining temp files
    for task_id, task in _tasks.items():
        tmp_path = task.get("tmp_path")
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    logger.info("RouteX API shut down.")
