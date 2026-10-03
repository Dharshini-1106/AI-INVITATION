"""Invitation upload & analysis endpoints."""
import logging
import threading
import time
from typing import List

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ...config import settings
from ...core.pipeline import run_pipeline
from ...schemas.invitation import InvitationResult
from ...auth import require_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["invitation"], dependencies=[Depends(require_user)])
_analysis_lock = threading.Lock()


def _run_pipeline_serialized(data: bytes, filename: str, source: str) -> dict:
    # The OCR engine caches mutable model instances; avoid overlapping calls
    # from separate upload requests while still freeing the ASGI event loop.
    with _analysis_lock:
        return run_pipeline(data, filename, source=source)

@router.get("/pipeline/stages", response_model=List[str])
def pipeline_stages():
    """Return the ordered list of pipeline stage names (for progress UI)."""
    from ...core.pipeline import PIPELINE_STAGES
    return PIPELINE_STAGES


@router.post("/analyze", response_model=InvitationResult)
async def analyze_invitation(
    file: UploadFile = File(...),
    x_invitation_source: str | None = Header(default=None),
):
    """Upload an invitation image (or PDF) and extract structured information."""
    filename = file.filename or "invitation"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in settings.allowed_extensions:
        if ext in {".pdf"}:
            # PDF support placeholder: try first-page render via PIL/OpenCV later
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. "
                       f"Allowed: {sorted(settings.allowed_extensions)}",
            )

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.max_upload_mb} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    source = "camera" if (x_invitation_source or "").lower() == "camera" else "gallery"
    label = "CAMERA" if source == "camera" else "GALLERY"
    logger.info("[%s IMAGE] filename=%s MIME=%s size=%d bytes",
                label, filename, file.content_type or "unknown", len(data))

    try:
        start = time.time()
        # OCR and image enhancement are synchronous CPU/model work. Keep them
        # off the ASGI event loop so health checks and other requests still
        # work while an invitation is being processed.
        result = await run_in_threadpool(
            _run_pipeline_serialized, data, filename, source
        )
        elapsed = time.time() - start
        logger.info("Analysis completed in %.2fs", elapsed)
        result["processing_notes"].append(f"Total processing time: {elapsed:.2f}s")
        logger.info("[%s FINAL API OUTPUT] bride_name=%r groom_name=%r",
                    label, result.get("bride_name", ""), result.get("groom_name", ""))
        return InvitationResult(**result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")

