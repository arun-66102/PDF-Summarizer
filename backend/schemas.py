"""
Pydantic schemas for RouteX API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Auth ────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ── Process Text ────────────────────────────────────────────────────────────────

class ProcessTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text content to process")
    model: str = Field(default="llama3-8b", description="Model key to use")
    context_limit: int = Field(default=4000, ge=1000, le=8000, description="Context window limit")
    enable_routing: bool = Field(default=True, description="Enable department routing")


# ── Routing Result ──────────────────────────────────────────────────────────────

class RoutingMatch(BaseModel):
    department_code: str
    department_name: Optional[str] = None
    similarity_score: Optional[float] = None
    score: Optional[float] = None
    distance: Optional[float] = None
    matched_keywords: Optional[List[str]] = None
    method: Optional[str] = None


class RoutingResult(BaseModel):
    primary_departments: List[str] = []
    all_matches: List[Dict[str, Any]] = []
    confidence: float = 0.0
    is_tie: bool = False
    tie_threshold: float = 0.05
    method: str = "embedding"
    available: bool = True


# ── Process Response (shared for PDF and text) ──────────────────────────────────

class ProcessResponse(BaseModel):
    summary: str
    text_length: Optional[int] = None
    chunks_processed: Optional[int] = None
    model_used: Optional[str] = None
    source: Optional[str] = None
    routing: Optional[RoutingResult] = None
    text_stats: Optional[Dict[str, int]] = None
    file_path: Optional[str] = None
    error: Optional[str] = None
    task_id: Optional[str] = None  # for SSE progress tracking


# ── Email ───────────────────────────────────────────────────────────────────────

class EmailRequest(BaseModel):
    summary: str
    routing: RoutingResult
    pdf_path: Optional[str] = None  # server-side temp path from processing


class EmailResponse(BaseModel):
    success: bool
    sent_to: List[str] = []
    failed: List[str] = []
    message: str


# ── Health ──────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    groq_connected: bool
    email_configured: bool
    routing_available: bool
    status: str  # "ok" | "degraded" | "error"


# ── Models ──────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    key: str
    label: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default: str
