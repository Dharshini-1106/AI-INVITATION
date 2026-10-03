"""Application configuration and settings."""
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".pdf"}


class Settings(BaseSettings):
    """Central settings for the backend."""

    app_name: str = "Invitation Understanding API"
    version: str = "1.0.0"
    upload_dir: Path = UPLOAD_DIR
    max_upload_mb: int = 20
    allowed_extensions: set[str] = ALLOWED_EXTENSIONS

    # Model toggles (disable heavy models for lightweight/offline runs).
    # PaddleOCR is the PRIMARY OCR engine; RapidOCR is FALLBACK ONLY.
    # When PaddleOCR succeeds, RapidOCR must NOT run.
    use_doclayout: bool = False
    use_layoutlm: bool = False
    use_ocr_ppocr: bool = True
    use_rapidocr: bool = True
    use_tamil_ocr: bool = True
    use_dual_ocr: bool = True
    use_sbert: bool = False
    use_ner: bool = False
    use_matching: bool = False

    # Confidence thresholds
    quality_threshold: float = 0.75
    min_layout_confidence: float = 0.35

    # Google Calendar API key for calendar integration
    google_calendar_api_key: str = ""
    google_maps_api_key: str = ""
    travel_timezone: str = "Asia/Kolkata"

    # Google OAuth2 client credentials for user calendar access
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = ""
    frontend_redirect_url: str = ""

    mongo_uri: str = ""
    jwt_secret: str = ""
    auth_cookie_secure: bool = False
    # Use "none" with AUTH_COOKIE_SECURE=true when frontend and API are
    # hosted on different sites. Keep "lax" for local/same-site deployments.
    auth_cookie_samesite: str = "lax"

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


settings = Settings()
