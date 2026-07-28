"""
JWT authentication for RouteX API.

Credentials are configured via environment variables:
  ROUTEX_USERNAME  (default: admin)
  ROUTEX_PASSWORD  (default: routex2026)
  ROUTEX_JWT_SECRET (default: auto-generated)
"""

import os
import time
import secrets
import hashlib
import hmac
import json
import base64
from typing import Optional
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── Config ──────────────────────────────────────────────────────────────────────

_JWT_SECRET = os.getenv("ROUTEX_JWT_SECRET", secrets.token_hex(32))
_USERNAME = os.getenv("ROUTEX_USERNAME", "admin")
_PASSWORD = os.getenv("ROUTEX_PASSWORD", "routex2026")
_TOKEN_EXPIRY = 60 * 60 * 24  # 24 hours

security = HTTPBearer()


# ── JWT helpers (minimal, no PyJWT dependency) ──────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _sign(header_payload: str) -> str:
    sig = hmac.new(
        _JWT_SECRET.encode(),
        header_payload.encode(),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(sig)


def create_token(username: str) -> tuple[str, int]:
    """Create a JWT token. Returns (token_string, expires_in_seconds)."""
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64url_encode(
        json.dumps({
            "sub": username,
            "iat": now,
            "exp": now + _TOKEN_EXPIRY,
        }).encode()
    )
    header_payload = f"{header}.{payload}"
    signature = _sign(header_payload)
    return f"{header_payload}.{signature}", _TOKEN_EXPIRY


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_payload = f"{parts[0]}.{parts[1]}"
        expected_sig = _sign(header_payload)

        if not hmac.compare_digest(expected_sig, parts[2]):
            return None

        payload = json.loads(_b64url_decode(parts[1]))

        if payload.get("exp", 0) < int(time.time()):
            return None

        return payload
    except Exception:
        return None


# ── Login ───────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> Optional[str]:
    """Validate credentials and return a JWT token, or None if invalid."""
    if username == _USERNAME and password == _PASSWORD:
        token, _ = create_token(username)
        return token
    return None


# ── FastAPI dependency ──────────────────────────────────────────────────────────

async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """FastAPI dependency that enforces a valid JWT Bearer token."""
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload
