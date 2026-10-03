"""Authenticated persistence for user-owned invitation events and travel plans."""
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from ...auth import database, require_user

router = APIRouter(tags=["events"])


class EventPayload(BaseModel):
    event: dict[str, Any]


class TravelPlanPayload(BaseModel):
    travel_plan: dict[str, Any]
    request: dict[str, Any] = Field(default_factory=dict)


class SchedulePayload(BaseModel):
    schedule: dict[str, Any]


def _owner_filter(event_id: str, user: dict[str, Any]) -> dict[str, Any]:
    if not ObjectId.is_valid(event_id):
        raise HTTPException(404, "Event not found.")
    try:
        user_id = ObjectId(user["id"])
    except (KeyError, TypeError):
        raise HTTPException(401, "Authentication required.") from None
    return {"_id": ObjectId(event_id), "user_id": user_id}


def _serialize(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(document["_id"]),
        "event": document.get("event", {}),
        "travel_plan": document.get("travel_plan"),
        "schedule": document.get("schedule"),
        "created_at": document.get("created_at").isoformat() if document.get("created_at") else None,
        "updated_at": document.get("updated_at").isoformat() if document.get("updated_at") else None,
    }


def _storage_error() -> HTTPException:
    return HTTPException(503, "Event storage is temporarily unavailable.")


@router.post("/events", status_code=201)
def create_event(payload: EventPayload, user: dict[str, Any] = Depends(require_user)):
    now = datetime.now(timezone.utc)
    document = {
        "user_id": ObjectId(user["id"]),
        "event": payload.event,
        "travel_plan": None,
        "schedule": None,
        "created_at": now,
        "updated_at": now,
    }
    try:
        document["_id"] = database().events.insert_one(document).inserted_id
    except PyMongoError:
        raise _storage_error() from None
    return _serialize(document)


@router.get("/events")
def list_events(user: dict[str, Any] = Depends(require_user)):
    try:
        documents = database().events.find({"user_id": ObjectId(user["id"])}).sort("updated_at", -1)
        return [_serialize(document) for document in documents]
    except PyMongoError:
        raise _storage_error() from None


@router.get("/events/{event_id}")
def get_event(event_id: str, user: dict[str, Any] = Depends(require_user)):
    try:
        document = database().events.find_one(_owner_filter(event_id, user))
    except PyMongoError:
        raise _storage_error() from None
    if not document:
        raise HTTPException(404, "Event not found.")
    return _serialize(document)


@router.put("/events/{event_id}")
def update_event(event_id: str, payload: EventPayload, user: dict[str, Any] = Depends(require_user)):
    now = datetime.now(timezone.utc)
    try:
        result = database().events.update_one(
            _owner_filter(event_id, user),
            {"$set": {"event": payload.event, "updated_at": now}},
        )
        document = database().events.find_one(_owner_filter(event_id, user)) if result.matched_count else None
    except PyMongoError:
        raise _storage_error() from None
    if not document:
        raise HTTPException(404, "Event not found.")
    return _serialize(document)


@router.put("/events/{event_id}/travel-plan")
def save_travel_plan(event_id: str, payload: TravelPlanPayload, user: dict[str, Any] = Depends(require_user)):
    now = datetime.now(timezone.utc)
    travel_plan = {
        **payload.travel_plan,
        "request": payload.request,
        "calculated_at": now,
    }
    try:
        result = database().events.update_one(
            _owner_filter(event_id, user),
            {"$set": {"travel_plan": travel_plan, "updated_at": now}},
        )
        document = database().events.find_one(_owner_filter(event_id, user)) if result.matched_count else None
    except PyMongoError:
        raise _storage_error() from None
    if not document:
        raise HTTPException(404, "Event not found.")
    return _serialize(document)


@router.put("/events/{event_id}/schedule")
def save_schedule(event_id: str, payload: SchedulePayload, user: dict[str, Any] = Depends(require_user)):
    now = datetime.now(timezone.utc)
    schedule = {**payload.schedule, "updated_at": now}
    try:
        result = database().events.update_one(
            _owner_filter(event_id, user),
            {"$set": {"schedule": schedule, "updated_at": now}},
        )
        document = database().events.find_one(_owner_filter(event_id, user)) if result.matched_count else None
    except PyMongoError:
        raise _storage_error() from None
    if not document:
        raise HTTPException(404, "Event not found.")
    return _serialize(document)


@router.delete("/events/{event_id}", status_code=204)
def delete_event(event_id: str, user: dict[str, Any] = Depends(require_user)):
    try:
        result = database().events.delete_one(_owner_filter(event_id, user))
    except PyMongoError:
        raise _storage_error() from None
    if not result.deleted_count:
        raise HTTPException(404, "Event not found.")