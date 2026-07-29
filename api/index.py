"""
Vercel serverless entrypoint for RouteX FastAPI backend.

Vercel's Python runtime looks for a file named api/index.py and
imports the ASGI `app` object from it.

Path setup: Adds the project root to sys.path so that imports of
`main`, `backend`, `pdf_extractor`, etc. all resolve correctly
even though Vercel mounts the function at /var/task/api/.
"""

import sys
import os

# Add project root (/var/task) to path so all modules resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Re-export the FastAPI app — Vercel picks this up automatically.
from backend.server import app  # noqa: F401  (re-exported for Vercel)
