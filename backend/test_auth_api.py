"""Regression tests for mobile bearer transport over the revocable session store."""
from datetime import datetime, timedelta, timezone
from io import BytesIO

import bcrypt
import pytest
from bson import ObjectId
from fastapi import HTTPException, Response
from starlette.datastructures import UploadFile
from starlette.requests import Request

from app import auth as auth_helpers
from app.api.routes import analyze, auth as auth_routes
from app.api.routes.auth import LoginInput


class MemoryCollection:
    def __init__(self):
        self.records = []

    def insert_one(self, record):
        saved = dict(record)
        saved.setdefault("_id", ObjectId())
        self.records.append(saved)
        return type("InsertResult", (), {"inserted_id": saved["_id"]})()

    def find_one(self, query, projection=None):
        for record in self.records:
            if all(
                (record.get(key) is not None and record[key] > value["$gt"])
                if isinstance(value, dict) and "$gt" in value
                else record.get(key) == value
                for key, value in query.items()
            ):
                result = dict(record)
                if projection:
                    for key, include in projection.items():
                        if not include:
                            result.pop(key, None)
                return result
        return None

    def delete_one(self, query):
        before = len(self.records)
        self.records[:] = [record for record in self.records if not all(
            record.get(key) == value for key, value in query.items()
        )]
        return type("DeleteResult", (), {"deleted_count": before - len(self.records)})()


class MemoryDatabase:
    def __init__(self):
        self.users = MemoryCollection()
        self.sessions = MemoryCollection()


def make_request(method="GET", path="/api/v1/test", headers=(), cookies=None):
    return Request({
        "type": "http", "method": method, "path": path,
        "headers": list(headers), "client": ("127.0.0.1", 1234),
        "server": ("test", 80), "scheme": "http", "query_string": b"",
        "root_path": "", "http_version": "1.1",
    })


@pytest.fixture
def session_db(monkeypatch):
    db = MemoryDatabase()
    user_id = ObjectId()
    db.users.records.append({
        "_id": user_id, "name": "Demo User", "email": "demo@example.com",
        "password_hash": bcrypt.hashpw(b"GoodPassword42", bcrypt.gensalt()).decode(),
        "status": "active",
    })
    monkeypatch.setattr(auth_helpers, "database", lambda: db)
    monkeypatch.setattr(auth_routes, "database", lambda: db)
    monkeypatch.setattr(auth_helpers.settings, "jwt_secret", "test-secret-with-enough-randomness")
    return db, user_id


def test_mobile_login_authenticates_stages_and_analysis_and_logout_revokes(session_db, monkeypatch):
    db, user_id = session_db
    request = make_request(
        "POST", "/api/auth/login", [(b"x-session-transport", b"bearer")]
    )
    login_result = auth_routes.login(
        LoginInput(email="demo@example.com", password="GoodPassword42"), request, Response()
    )
    token = login_result["session_token"]
    bearer_request = make_request(headers=[(b"authorization", f"Bearer {token}".encode())])
    assert auth_helpers.require_user(bearer_request, None)["id"] == str(user_id)

    # Authenticate before direct endpoint invocation, as FastAPI's router
    # dependency does. This exercises the same request extraction and handler.
    assert analyze.pipeline_stages()
    reached_pipeline = []
    monkeypatch.setattr(
        analyze, "run_pipeline",
        lambda *_args, **_kwargs: reached_pipeline.append(True) or {
            "processing_notes": [], "quality": {"score": 0},
        },
    )
    upload = UploadFile(filename="demo.jpg", file=BytesIO(b"image"), headers=None)
    import asyncio
    result = asyncio.run(analyze.analyze_invitation(upload, None))
    assert result.processing_notes
    assert reached_pipeline == [True]

    # Anonymous, malformed and expired bearer credentials remain rejected.
    for request_without_session in (
        make_request(),
        make_request(headers=[(b"authorization", b"Bearer invalid")]),
    ):
        with pytest.raises(HTTPException) as error:
                auth_helpers.require_user(request_without_session, None)
        assert error.value.status_code == 401

    expired = "expired-session-token"
    db.sessions.insert_one({
        "token_hash": auth_helpers._token_hash(expired), "user_id": user_id,
        "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
    })
    with pytest.raises(HTTPException) as error:
        auth_helpers.require_user(
            make_request(headers=[(b"authorization", f"Bearer {expired}".encode())]), None
        )
    assert error.value.status_code == 401

    logout_request = make_request(
        "POST", "/api/auth/logout", [(b"authorization", f"Bearer {token}".encode())]
    )
    auth_routes.logout(logout_request, Response())
    with pytest.raises(HTTPException) as error:
        auth_helpers.require_user(bearer_request, None)
    assert error.value.status_code == 401
