"""
RouteX Dev Runner — starts FastAPI backend then Vite dev server.

Waits for the backend to be healthy before starting Vite, so Vite's
proxy never hits ECONNREFUSED on the initial page load.

Usage:
    python run_dev.py
"""

import subprocess
import sys
import os
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

BACKEND_URL = "http://localhost:8000/api/health"
POLL_INTERVAL = 1      # seconds between health checks
MAX_WAIT      = 60     # seconds before giving up


def wait_for_backend():
    """Poll /api/health until the backend is up or we time out."""
    print("[~] Waiting for backend to be ready", end="", flush=True)
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_URL, timeout=2) as resp:
                if resp.status == 200:
                    print(" OK")
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)

    print(f"\n[!] Backend did not start within {MAX_WAIT}s. Check for errors above.")
    return False


def main():
    print("Starting RouteX development servers...\n")

    # ── Start FastAPI backend ─────────────────────────────────────────────────
    print("[*] Starting FastAPI backend on http://localhost:8000")
    backend = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.server:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        cwd=ROOT,
    )

    # ── Wait until backend health endpoint responds ───────────────────────────
    if not wait_for_backend():
        backend.terminate()
        sys.exit(1)

    # ── Start Vite frontend ───────────────────────────────────────────────────
    print("[+] Starting React frontend on http://localhost:5173")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
    )

    print("\n[*] Both servers running!")
    print("   Backend:  http://localhost:8000")
    print("   Frontend: http://localhost:5173")
    print("   API docs: http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop both servers.\n")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        try:
            backend.wait(timeout=5)
            frontend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
            frontend.kill()
        print("Servers stopped.")


if __name__ == "__main__":
    main()
