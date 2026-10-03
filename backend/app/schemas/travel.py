"""Pydantic models for Google Routes travel planning."""
from typing import Literal

from pydantic import BaseModel, Field


class LocationInput(BaseModel):
    """A user-selected location, either a resolved coordinate pair or address."""

    type: Literal["current", "manual", "event"] | None = None
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None


class TravelPlanRequest(BaseModel):
    # Strings remain supported for existing clients; new clients send explicit
    # coordinate/address objects so display text can never replace GPS data.
    origin: str | LocationInput = ""
    destination: str | LocationInput = ""
    # Extracted venue name and full address, used to resolve the destination
    # to a specific Google Place before routing. The original text is always
    # preserved for display; only the routing waypoint is replaced.
    destination_venue: str = ""
    destination_address: str = ""
    event_date: str = ""
    event_start_time: str = ""
    event_timezone: str = ""
    travel_mode: Literal["DRIVE", "TWO_WHEELER", "TRANSIT"] = "DRIVE"
    preparation_minutes: int = Field(default=60, ge=0, le=1440)
    arrival_buffer_minutes: int = Field(default=15, ge=0, le=1440)


class TravelPlanResponse(BaseModel):
    distance_meters: int | None = None
    distance_text: str = ""
    travel_duration_seconds: int | None = None
    travel_duration_text: str = ""
    distance_km: float | None = None
    duration_minutes: int | None = None
    event_start_time: str = ""
    arrival_time: str = ""
    departure_time: str = ""
    ready_time: str = ""
    travel_mode: str
    destination: str
    destination_place_id: str = ""
    destination_place_name: str = ""
    origin: str
    timezone: str = ""
    google_maps_url: str
    schedule_available: bool = False
    message: str = ""
    schedule_message: str = ""
    traffic_aware: bool = False
