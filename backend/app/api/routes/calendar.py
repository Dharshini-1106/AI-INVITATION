"""Google Calendar integration endpoints."""
import logging
import secrets
import time
from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from ...config import settings
from ...schemas.invitation import Event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calendar"])

GOOGLE_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"
GOOGLE_OAUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# In-memory token store: session_id -> {access_token, refresh_token, expires_at}
# NOTE: For production, replace with a database (e.g., PostgreSQL/Redis).
_tokens: dict[str, dict] = {}


def _usable(value) -> bool:
    return value is not None and str(value).strip() not in {"", "not available", "—", "-"}


def _get_user_token(session_id: str) -> Optional[dict]:
    token = _tokens.get(session_id)
    if not token:
        return None
    if token.get("expires_at", 0) < time.time():
        refreshed = _refresh_token(token.get("refresh_token"))
        if refreshed:
            _tokens[session_id] = refreshed
            return refreshed
        _tokens.pop(session_id, None)
        return None
    return token


def _refresh_token(refresh_token: str) -> Optional[dict]:
    if not refresh_token or not settings.google_oauth_client_secret:
        return None
    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "access_token": data.get("access_token", ""),
            "refresh_token": refresh_token,
            "expires_at": int(time.time()) + int(data.get("expires_in", 3600)) - 60,
        }
    except requests.RequestException:
        return None


def _calendar_event_payload(event: Event) -> dict:
    start_dt = event.startDateTime or event.dateTime or event.date
    end_dt = event.endDateTime or event.endTime

    payload = {
        "summary": event.summary or event.event_name or event.event_type or "Invitation Event",
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt or ""},
    }

    location_parts = [p for p in [event.location, event.address, event.venue] if _usable(p)]
    if location_parts:
        payload["location"] = ", ".join(location_parts)

    notes_parts = [p for p in [event.event_type, event.description, event.notes] if _usable(p)]
    if notes_parts:
        payload["description"] = "\n".join(notes_parts)

    return payload


@router.get("/calendar/auth-url")
def get_calendar_auth_url(session_id: str = Query(...)):
    """Return a Google OAuth authorization URL for the given session."""
    if not settings.google_oauth_client_id:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth client ID is not configured on the server.",
        )
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/calendar.events",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = GOOGLE_OAUTH_BASE + "?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return {"auth_url": url, "state": state}


@router.get("/calendar/oauth-callback")
def oauth_callback(code: str = Query(...), state: str = Query(""), session_id: str = Query(...)):
    """Handle Google OAuth callback, exchange code for tokens, store them."""
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured on the server.")
    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "code": code,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error("OAuth token exchange failed: %s", resp.text)
            return RedirectResponse(url=f"{settings.frontend_redirect_url}?error=token_exchange_failed")
        data = resp.json()
        _tokens[session_id] = {
            "access_token": data.get("access_token", ""),
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": int(time.time()) + int(data.get("expires_in", 3600)) - 60,
        }
        return RedirectResponse(url=f"{settings.frontend_redirect_url}?success=auth_complete")
    except requests.RequestException as exc:
        logger.error("OAuth callback failed: %s", exc)
        return RedirectResponse(url=f"{settings.frontend_redirect_url}?error=oauth_failed")


@router.post("/calendar/create")
def create_calendar_events(events: List[Event], session_id: str = Query(...)):
    """Create Google Calendar events using the authenticated user's access token."""
    token = _get_user_token(session_id)
    if not token:
        return {
            "ok": False,
            "results": [
                {
                    "index": idx,
                    "ok": False,
                    "error": "auth_required",
                }
                for idx in range(len(events))
            ],
        }

    if not events:
        raise HTTPException(status_code=400, detail="No events provided.")

    results = []
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Content-Type": "application/json",
    }

    for idx, event in enumerate(events):
        payload = _calendar_event_payload(event)
        url = f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events"

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as exc:
            logger.error("Calendar request failed for event %d: %s", idx, exc)
            results.append({"index": idx, "ok": False, "error": str(exc)})
            continue

        if resp.status_code == 200:
            created = resp.json()
            results.append({"index": idx, "ok": True, "eventId": created.get("id")})
        else:
            detail = resp.text
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            results.append({"index": idx, "ok": False, "error": f"HTTP {resp.status_code}: {detail}"})

    any_failed = any(not r["ok"] for r in results)
    if any_failed:
        raise HTTPException(status_code=500, detail={"results": results})

    return {"ok": True, "results": results}
