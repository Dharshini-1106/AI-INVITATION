"""Development server launcher.

Usage:
    D:\MINI\backend\.venv-ml311\Scripts\python.exe run.py

This script intentionally uses the project's dedicated ML venv interpreter
(.venv-ml311) so PaddleOCR and PaddlePaddle are available.  Do NOT rely on
the system PATH-selected `python`, global Python, or another venv.
"""
import os
import sys
import uvicorn

# Ensure the backend directory is on the path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    print(f"PYTHON EXECUTABLE: {sys.executable}")
    print(f"PYTHON VERSION: {sys.version}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )

