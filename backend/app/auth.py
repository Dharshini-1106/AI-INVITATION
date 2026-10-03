"""MongoDB backed account and revocable browser session helpers."""
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timezone

import bcrypt
from fastapi import Cookie, HTTPException, Request
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .config import settings

logger = logging.getLogger(__name__)
_client = None
_db = None
_rate_buckets: dict[str, list[float]] = {}
_indexes_ready = False


def _token_hash(token: str) -> str:
    if not settings.jwt_secret:
        raise HTTPException(503, "Session security is not configured on the server.")
    return hmac.new(settings.jwt_secret.encode(), token.encode(), hashlib.sha256).hexdigest()


def database():
    global _client, _db, _indexes_ready
    if not settings.mongo_uri:
        raise HTTPException(503, "Account storage is not configured. Set MONGO_URI on the server.")
    try:
        if _client is None:
            _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
            _db = _client["invi_db"]
        _client.admin.command("ping")
        if not _indexes_ready:
            _db.users.create_index("email", unique=True, name="users_email_unique")
            _db.sessions.create_index("expires_at", expireAfterSeconds=0, name="sessions_expiry_ttl")
            _db.sessions.create_index("token_hash", unique=True, name="sessions_token_unique")
            _db.events.create_index([("user_id", 1), ("updated_at", -1)], name="events_owner_updated")
            _indexes_ready = True
        return _db
    except PyMongoError as exc:
        logger.warning("MongoDB unavailable (%s)", type(exc).__name__)
        _client = None
        _db = None
        _indexes_ready = False
        raise HTTPException(503, "Account storage is temporarily unavailable.") from None


def initialize_auth_indexes():
    """Create indexes without dropping or altering existing data."""
    database()


def normalize_email(value: str) -> str:
    return value.strip().casefold()


def check_rate_limit(request: Request, action: str):
    key = f"{action}:{request.client.host if request.client else 'unknown'}"
    now = time.time()
    attempts = [stamp for stamp in _rate_buckets.get(key, []) if now - stamp < 900]
    if len(attempts) >= 8:
        raise HTTPException(429, "Too many attempts. Please try again later.")
    attempts.append(now)
    _rate_buckets[key] = attempts


def create_session(user_id) -> str:
    db = database()
    token = secrets.token_urlsafe(48)
    db.sessions.insert_one({"token_hash": _token_hash(token), "user_id": user_id,
                            "created_at": datetime.now(timezone.utc),
                            "expires_at": datetime.fromtimestamp(time.time() + 60 * 60 * 24 * 7, timezone.utc)})
    return token


def current_user(session: str | None):
    if not session:
        raise HTTPException(401, "Authentication required.")
    try:
        db = database()
        token_hash = _token_hash(session)
        record = db.sessions.find_one({"token_hash": token_hash, "expires_at": {"$gt": datetime.now(timezone.utc)}})
        if not record:
            raise HTTPException(401, "Authentication required.")
        user = db.users.find_one({"_id": record["user_id"], "status": "active"}, {"password_hash": 0})
        if not user:
            raise HTTPException(401, "Authentication required.")
        return {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}
    except PyMongoError:
        raise HTTPException(503, "Account storage is temporarily unavailable.") from None


def session_from_request(request: Request, invitation_session: str | None = None):
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return invitation_session or request.cookies.get("invitation_session")


def require_user(request: Request, invitation_session: str | None = Cookie(default=None, alias="invitation_session")):
    return current_user(session_from_request(request, invitation_session))

