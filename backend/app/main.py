"""FastAPI application entry point."""
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.routes import analyze, calendar, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _log_startup_diagnostics():
    """Log environment and OCR engine availability at startup."""
    python_exec = sys.executable
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    paddleocr_available = False
    paddleocr_version = "N/A"
    paddle_version = "N/A"
    rapidocr_available = False

    try:
        import paddle
        paddle_version = getattr(paddle, "__version__", "N/A")
    except Exception:  # noqa: BLE001
        pass

    try:
        import paddleocr
        paddleocr_available = True
        paddleocr_version = getattr(paddleocr, "__version__", "N/A")
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR unavailable at startup (%s)", exc)

    try:
        from rapidocr_onnxruntime import RapidOCR
        rapidocr_available = True
    except Exception:  # noqa: BLE001
        pass

    primary_engine = "PaddleOCR" if paddleocr_available else "RapidOCR"
    rapidocr_role = "PRIMARY" if not paddleocr_available else "FALLBACK ONLY"

    logger.info("PYTHON EXECUTABLE: %s", python_exec)
    logger.info("PYTHON VERSION: %s", python_version)
    logger.info("PaddleOCR installed: %s", "YES" if paddleocr_available else "NO")
    logger.info("PaddleOCR version: %s", paddleocr_version)
    logger.info("PaddlePaddle version: %s", paddle_version)
    logger.info("RapidOCR installed: %s", "YES" if rapidocr_available else "NO")
    logger.info("PRIMARY OCR ENGINE: %s", primary_engine)
    logger.info("FALLBACK OCR ENGINE: RapidOCR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models in the background at startup."""
    logger.info("Starting %s v%s", settings.app_name, settings.version)
    _log_startup_diagnostics()
    try:
        from .core.ocr.ocreader import _get_paddleocr
        _get_paddleocr("en")
        _get_paddleocr(
            "ta",
            det_model_name="PP-OCRv5_mobile_det",
            rec_model_name="ta_PP-OCRv5_mobile_rec",
        )
    except Exception as exc:
        logger.warning("PaddleOCR pre-load skipped: %s", exc)
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-Driven Invitation Understanding & Adaptive Travel Planning System - Phase 1: "
                "Intelligent Invitation Understanding and Information Extraction.",
    lifespan=lifespan,
)

# CORS is permissive in development so any origin can call the API.
# This includes the Expo web dev server (port 8081), the Vite frontend
# (5173), the plain React dev server (3000), and any phone browser hitting
# the PC's LAN IP. The API uses no cookies, so allow_credentials is off.
_ALLOW_ALL_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOW_ALL_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analyze.router, prefix="/api/v1", tags=["invitation"])
app.include_router(calendar.router, prefix="/api/v1", tags=["calendar"])


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }

