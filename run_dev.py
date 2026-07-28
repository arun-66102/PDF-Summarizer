"""
RouteX Dev Runner — starts both FastAPI and Vite dev servers.

Usage:
    python run_dev.py
"""

import subprocess
import sys
import os
import signal
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

def main():
    print("Starting RouteX development servers...\n")

    # Start FastAPI backend
    print("Starting FastAPI backend on http://localhost:8000")
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

    # Give backend a moment to start
    time.sleep(2)

    # Start Vite frontend
    print("Starting React frontend on http://localhost:5173")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
    )

    print("\nBoth servers running!")
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
