"""Focused tests for account normalization, password handling, and API protection."""
import bcrypt
import pytest
from bson import ObjectId
from fastapi import HTTPException, Response
from pymongo.errors import DuplicateKeyError
from starlette.requests import Request

from app import auth as auth_helpers
from app.api.routes import auth as auth_routes
from app.auth import normalize_email
from app.api.routes.auth import LoginInput, SignupInput
from app.api.routes import analyze, calendar, travel


def test_email_is_normalized_case_insensitively():
    assert normalize_email("  Alice.Example@Email.COM ") == "alice.example@email.com"


def test_password_hash_is_not_plaintext_and_verifies():
    password = "GoodPassword42"
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    assert password_hash != password
    assert bcrypt.checkpw(password.encode(), password_hash.encode())
    assert not bcrypt.checkpw(b"wrong-password", password_hash.encode())


def test_signup_rejects_mismatched_passwords():
    payload = SignupInput(name="A Person", email="person@example.com", password="GoodPassword42", confirm_password="WrongPassword42")
    with pytest.raises(HTTPException) as error:
        payload.matches()
    assert error.value.status_code == 422


def test_signup_requires_a_strong_password():
    payload = SignupInput(name="A Person", email="person@example.com", password="weakpassword", confirm_password="weakpassword")
    with pytest.raises(HTTPException):
        payload.matches()


def test_private_api_routes_have_auth_dependency():
    assert all(router.router.dependencies for router in (analyze, calendar, travel))


class FakeCollection:
    def __init__(self):
        self.records = []

    def insert_one(self, record):
        if "email" in record and any(item.get("email") == record["email"] for item in self.records):
            raise DuplicateKeyError("duplicate email")
        saved = dict(record)
        saved.setdefault("_id", ObjectId())
        self.records.append(saved)
        return type("InsertResult", (), {"inserted_id": saved["_id"]})()

    def find_one(self, query, projection=None):
        for record in self.records:
            matched = True
            for key, value in query.items():
                if isinstance(value, dict) and "$gt" in value:
                    if record.get(key) is None or record[key] <= value["$gt"]:
                        matched = False
                        break
                elif record.get(key) != value:
                    matched = False
                    break
            if matched:
                result = dict(record)
                if projection:
                    for key, include in projection.items():
                        if not include:
                            result.pop(key, None)
                return result
        return None

    def delete_one(self, query):
        before = len(self.records)
        self.records[:] = [r for r in self.records if not all(r.get(k) == v for k, v in query.items())]
        return type("DeleteResult", (), {"deleted_count": before - len(self.records)})()


class FakeDatabase:
    def __init__(self):
        self.users = FakeCollection()
        self.sessions = FakeCollection()


def fake_request():
    return Request({"type": "http", "method": "POST", "path": "/api/auth/test", "headers": [],
                    "client": ("127.0.0.1", 1234), "server": ("test", 80), "scheme": "http",
                    "query_string": b"", "root_path": "", "http_version": "1.1"})


def test_signup_saves_only_a_password_hash_and_rejects_duplicate(monkeypatch):
    db = FakeDatabase()
    monkeypatch.setattr(auth_routes, "database", lambda: db)
    payload = SignupInput(name="Alice Example", email="ALICE@example.com", password="GoodPassword42", confirm_password="GoodPassword42")
    result = auth_routes.signup(payload, fake_request())
    assert result["ok"] is True
    assert db.users.records[0]["email"] == "alice@example.com"
    assert db.users.records[0]["status"] == "active"
    assert "password" not in db.users.records[0]
    assert bcrypt.checkpw(b"GoodPassword42", db.users.records[0]["password_hash"].encode())
    with pytest.raises(HTTPException) as error:
        auth_routes.signup(payload, fake_request())
    assert error.value.status_code == 409


def test_login_creates_revocable_cookie_session_and_rejects_bad_credentials(monkeypatch):
    db = FakeDatabase()
    monkeypatch.setattr(auth_routes, "database", lambda: db)
    monkeypatch.setattr(auth_helpers, "database", lambda: db)
    monkeypatch.setattr(auth_helpers.settings, "jwt_secret", "test-secret-with-enough-randomness")
    user_id = ObjectId()
    db.users.records.append({"_id": user_id, "name": "Alice Example", "email": "alice@example.com",
                             "password_hash": bcrypt.hashpw(b"GoodPassword42", bcrypt.gensalt()).decode(), "status": "active"})

    response = Response()
    result = auth_routes.login(LoginInput(email="Alice@Example.com", password="GoodPassword42"), fake_request(), response)
    assert result["user"]["email"] == "alice@example.com"
    assert "invitation_session=" in response.headers["set-cookie"]
    assert len(db.sessions.records) == 1
    with pytest.raises(HTTPException) as error:
        auth_routes.login(LoginInput(email="alice@example.com", password="incorrect"), fake_request(), Response())
    assert error.value.status_code == 401
    with pytest.raises(HTTPException) as error:
        auth_routes.login(LoginInput(email="unknown@example.com", password="GoodPassword42"), fake_request(), Response())
    assert error.value.status_code == 401


def test_logout_invalidates_session_and_unauthenticated_user_is_rejected(monkeypatch):
    db = FakeDatabase()
    monkeypatch.setattr(auth_routes, "database", lambda: db)
    monkeypatch.setattr(auth_helpers, "database", lambda: db)
    monkeypatch.setattr(auth_helpers.settings, "jwt_secret", "test-secret-with-enough-randomness")
    with pytest.raises(HTTPException) as error:
        auth_helpers.current_user(None)
    assert error.value.status_code == 401

    user_id = ObjectId()
    db.users.records.append({"_id": user_id, "name": "Alice", "email": "alice@example.com", "status": "active"})
    token = auth_helpers.create_session(user_id)
    assert auth_helpers.current_user(token)["id"] == str(user_id)
    request = fake_request()
    request._cookies = {auth_routes.COOKIE_NAME: token}
    response = Response()
    auth_routes.logout(request, response)
    assert response.headers["set-cookie"].endswith("Max-Age=0; Path=/; SameSite=lax")
    with pytest.raises(HTTPException) as error:
        auth_helpers.current_user(token)
    assert error.value.status_code == 401

