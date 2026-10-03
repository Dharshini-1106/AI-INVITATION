"""Public signup/login endpoints and browser session lifecycle."""
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field, field_validator
from pymongo.errors import DuplicateKeyError, PyMongoError

from ...auth import check_rate_limit, create_session, current_user, database, normalize_email, session_from_request
from ...config import settings

router = APIRouter()
COOKIE_NAME = "invitation_session"


class SignupInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    confirm_password: str = Field(min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Enter your full name.")
        return value

    def matches(self):
        if self.password != self.confirm_password:
            raise HTTPException(422, "Passwords do not match.")
        if not any(c.islower() for c in self.password) or not any(c.isupper() for c in self.password) or not any(c.isdigit() for c in self.password):
            raise HTTPException(422, "Password must include uppercase, lowercase, and a number.")
        if len(self.password.encode("utf-8")) > 72:
            raise HTTPException(422, "Password must be at most 72 UTF-8 bytes.")


class LoginInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("password")
    @classmethod
    def bcrypt_limit(cls, value):
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password is invalid.")
        return value


def set_session_cookie(response: Response, token: str):
    response.set_cookie(COOKIE_NAME, token, httponly=True, secure=settings.auth_cookie_secure,
                        samesite=settings.auth_cookie_samesite, max_age=60 * 60 * 24 * 7, path="/")


@router.post("/signup", status_code=201)
def signup(payload: SignupInput, request: Request):
    check_rate_limit(request, "signup")
    payload.matches()
    db = database()
    email = normalize_email(str(payload.email))
    try:
        db.users.insert_one({"name": payload.name, "email": email,
                             "password_hash": bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt(rounds=12)).decode(),
                             "status": "active", "created_at": datetime.now(timezone.utc)})
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists.") from None
    except PyMongoError:
        raise HTTPException(503, "Account storage is temporarily unavailable.") from None
    return {"ok": True, "message": "Account created. Please log in."}


@router.post("/login")
def login(payload: LoginInput, request: Request, response: Response):
    check_rate_limit(request, "login")
    try:
        user = database().users.find_one({"email": normalize_email(str(payload.email)), "status": "active"})
    except PyMongoError:
        raise HTTPException(503, "Account storage is temporarily unavailable.") from None
    valid = False
    if user:
        try:
            valid = bcrypt.checkpw(payload.password.encode(), user["password_hash"].encode())
        except (ValueError, KeyError):
            valid = False
    if not valid:
        raise HTTPException(401, "Email or password is incorrect.")
    session_token = create_session(user["_id"])
    set_session_cookie(response, session_token)
    result = {"ok": True, "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}}
    # Expo/React Native Axios does not provide a dependable shared cookie jar.
    # Return the same opaque, revocable server session only to clients opting
    # into bearer transport; browser callers continue to use the HttpOnly cookie.
    if request.headers.get("x-session-transport", "").lower() == "bearer":
        result["session_token"] = session_token
    return result


@router.post("/logout")
def logout(request: Request, response: Response):
    token = session_from_request(request)
    storage_error = False
    if token:
        try:
            from ...auth import _token_hash
            database().sessions.delete_one({"token_hash": _token_hash(token)})
        except (PyMongoError, HTTPException):
            storage_error = True
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=settings.auth_cookie_secure,
                           samesite=settings.auth_cookie_samesite)
    if storage_error:
        raise HTTPException(503, "Account storage is temporarily unavailable.")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    return {"user": current_user(session_from_request(request))}
