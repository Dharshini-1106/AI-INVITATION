"""Health check endpoint."""
import time

from fastapi import APIRouter

from ...config import settings
from ...schemas.invitation import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.version,
    )

