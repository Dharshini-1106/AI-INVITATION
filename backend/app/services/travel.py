"""Google Routes based travel planning service."""
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from ..config import settings
from ..schemas.travel import LocationInput, TravelPlanRequest

logger = logging.getLogger(__name__)

GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_MAPS_DIRECTIONS_URL = "https://www.google.com/maps/dir/?api=1"
_ROUTES_FIELD_MASK = "routes.duration,routes.distanceMeters"
_COORDINATE_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)
_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %B %Y",
    "%d %b %Y",
)
_TIME_FORMATS = (
    "%I:%M %p",
    "%I:%M%p",
    "%H:%M",
    "%I %p",
)


class TravelPlanningError(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 502,
        google_maps_url: str = "",
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.google_maps_url = google_maps_url


def _google_maps_url(origin: str, destination: str, travel_mode: str) -> str:
    mode_map = {
        "DRIVE": "driving",
        "TWO_WHEELER": "bicycling",
        "TRANSIT": "transit",
    }
    params = {
        "origin": origin,
        "destination": destination,
        "travelmode": mode_map.get(travel_mode, travel_mode.lower()),
    }
    return f"{GOOGLE_MAPS_DIRECTIONS_URL}&{urlencode(params)}"


def _waypoint(value: str) -> dict[str, Any]:
    match = _COORDINATE_RE.match(value)
    if match:
        return {
            "location": {
                "latLng": {
                    "latitude": float(match.group(1)),
                    "longitude": float(match.group(2)),
                }
            }
        }
    return {"address": value.strip()}


def _location_value(value: str | LocationInput, field: str) -> str:
    """Normalize API location objects while rejecting invalid GPS coordinates."""
    if isinstance(value, str):
        location = value.strip()
    else:
        latitude, longitude = value.latitude, value.longitude
        if latitude is not None or longitude is not None:
            if latitude is None or longitude is None or not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                raise TravelPlanningError(
                    f"{field.capitalize()} coordinates are invalid.",
                    f"{field}_invalid",
                    status_code=400,
                )
            return f"{latitude},{longitude}"
        location = (value.address or "").strip()
    return location


def _event_timezone(explicit_timezone: str | None) -> ZoneInfo:
    timezone_name = (explicit_timezone or settings.travel_timezone or "Asia/Kolkata").strip()
    is_configured_timezone = not bool((explicit_timezone or "").strip())
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        if is_configured_timezone:
            raise TravelPlanningError(
                "Travel timezone is not configured correctly.",
                "timezone_invalid",
                status_code=500,
            ) from exc
        raise TravelPlanningError(
            "Event timezone is invalid.",
            "timezone_invalid",
            status_code=400,
        ) from exc


def _parse_event_datetime(
    date_value: str,
    time_value: str,
    explicit_timezone: str | None = None,
) -> datetime | None:
    date_value = (date_value or "").strip()
    time_value = re.split(
        r"\s+(?:onwards?|onward|sharp|to|until)\b",
        (time_value or "").strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if explicit_timezone and explicit_timezone.strip():
        _event_timezone(explicit_timezone)
    if not date_value or not time_value:
        return None
    combined = f"{date_value} {time_value}".strip()

    try:
        parsed = datetime.fromisoformat(combined.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_event_timezone(explicit_timezone))
        return parsed
    except ValueError:
        pass

    parsed_date = None
    for fmt in _DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_value, fmt)
            break
        except ValueError:
            continue
    if parsed_date is None or not time_value:
        return None

    parsed_time = None
    for fmt in _TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(time_value, fmt)
            break
        except ValueError:
            continue
    if parsed_time is None:
        return None

    return datetime.combine(
        parsed_date.date(),
        parsed_time.time(),
        tzinfo=_event_timezone(explicit_timezone),
    )


def _google_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _duration_seconds(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", value or "")
    if not match:
        raise ValueError(f"Unsupported Google duration: {value}")
    return int(float(match.group(1)))


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} sec"
    hours, remainder = divmod(seconds, 3600)
    minutes = (remainder + 59) // 60
    if hours and minutes:
        return f"{hours} hr {minutes} min"
    if hours:
        return f"{hours} hr"
    return f"{minutes} min"


def _format_distance(meters: int) -> str:
    if meters >= 1000:
        value = f"{meters / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{value} km"
    return f"{meters} m"


def _schedule_values(
    event_datetime: datetime | None,
    travel_seconds: int | None,
    preparation_minutes: int,
    arrival_buffer_minutes: int,
) -> tuple[str, str, str, str, str, bool, str]:
    if event_datetime is None or travel_seconds is None:
        return "", "", "", "", event_datetime.tzinfo.key if event_datetime and event_datetime.tzinfo else "", False, "Event time unavailable — travel schedule cannot be calculated."

    timezone_name = event_datetime.tzinfo.key if event_datetime.tzinfo else ""
    arrival_target = event_datetime - timedelta(minutes=arrival_buffer_minutes)
    departure = arrival_target - timedelta(seconds=travel_seconds)
    ready = departure - timedelta(minutes=preparation_minutes)
    return (
        _google_timestamp(event_datetime),
        _google_timestamp(arrival_target),
        _google_timestamp(departure),
        _google_timestamp(ready),
        timezone_name,
        True,
        "",
    )


def _route_request(payload: dict[str, Any], maps_url: str) -> dict[str, Any]:
    try:
        response = requests.post(
            GOOGLE_ROUTES_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": settings.google_maps_api_key,
                "X-Goog-FieldMask": _ROUTES_FIELD_MASK,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        logger.warning("Google Routes request failed: %s", exc)
        raise TravelPlanningError(
            "Travel time is currently unavailable.",
            "route_unavailable",
            google_maps_url=maps_url,
        ) from exc

    if response.status_code != 200:
        try:
            response_data = response.json()
        except (ValueError, TypeError):
            response_data = {}
        error_data = response_data.get("error", {}) if isinstance(response_data, dict) else {}
        if not isinstance(error_data, dict):
            error_data = {}
        google_status = str(error_data.get("status", "")).upper()
        detail = str(error_data.get("message", "")) or response.text[:300]
        logger.warning("Google Routes returned %s: %s", response.status_code, detail)

        if response.status_code == 400 or google_status in {
            "INVALID_ARGUMENT",
            "BAD_REQUEST",
            "NOT_FOUND",
        }:
            raise TravelPlanningError(
                "Google Routes could not calculate this route. Check the origin and destination.",
                "route_invalid",
                status_code=400,
                google_maps_url=maps_url,
            )
        if response.status_code in {401, 403} or google_status in {
            "PERMISSION_DENIED",
            "REQUEST_DENIED",
        }:
            raise TravelPlanningError(
                "Google Maps credentials are invalid or unavailable.",
                "route_unauthorized",
                status_code=403,
                google_maps_url=maps_url,
            )
        if response.status_code == 429 or google_status in {
            "RESOURCE_EXHAUSTED",
            "RATE_LIMIT_EXCEEDED",
        }:
            raise TravelPlanningError(
                "Google Maps quota or rate limit was reached.",
                "route_quota_exceeded",
                status_code=429,
                google_maps_url=maps_url,
            )
        raise TravelPlanningError(
            "Travel time is currently unavailable.",
            "route_unavailable",
            status_code=502,
            google_maps_url=maps_url,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise TravelPlanningError(
            "Travel time is currently unavailable.",
            "route_unavailable",
            google_maps_url=maps_url,
        ) from exc


def _route_metrics(data: dict[str, Any], maps_url: str) -> tuple[int | None, int | None]:
    routes = data.get("routes") or []
    if not routes:
        raise TravelPlanningError(
            "Travel time is currently unavailable.",
            "route_unavailable",
            google_maps_url=maps_url,
        )

    route = routes[0]
    duration_value = route.get("duration")
    try:
        travel_seconds = _duration_seconds(duration_value) if duration_value else None
    except ValueError:
        travel_seconds = None

    distance_meters = route.get("distanceMeters")
    try:
        distance_meters = int(distance_meters) if distance_meters is not None else None
    except (TypeError, ValueError):
        distance_meters = None
    return travel_seconds, distance_meters


def plan_travel(request: TravelPlanRequest) -> dict[str, Any]:
    origin = _location_value(request.origin, "origin")
    destination = _location_value(request.destination, "destination")
    maps_url = _google_maps_url(origin, destination, request.travel_mode)
    if not origin:
        raise TravelPlanningError(
            "Starting location is required.",
            "origin_required",
            status_code=400,
            google_maps_url=maps_url,
        )
    if not destination:
        raise TravelPlanningError(
            "Destination is required.",
            "destination_required",
            status_code=400,
            google_maps_url=maps_url,
        )

    if not settings.google_maps_api_key:
        raise TravelPlanningError(
            "Google Maps travel calculation is not configured.",
            "google_maps_not_configured",
            status_code=503,
            google_maps_url=maps_url,
        )

    event_datetime = _parse_event_datetime(
        request.event_date,
        request.event_start_time,
        request.event_timezone,
    )
    target_arrival = (
        event_datetime - timedelta(minutes=request.arrival_buffer_minutes)
        if event_datetime
        else None
    )
    base_payload: dict[str, Any] = {
        "origin": _waypoint(origin),
        "destination": _waypoint(destination),
        "travelMode": request.travel_mode,
    }

    if request.travel_mode in {"DRIVE", "TWO_WHEELER"}:
        base_payload["routingPreference"] = "TRAFFIC_AWARE"
        route_data = _route_request(base_payload, maps_url)
        initial_duration, _ = _route_metrics(route_data, maps_url)
        if event_datetime and target_arrival and initial_duration is not None:
            recommended_departure = target_arrival - timedelta(seconds=initial_duration)
            refined_payload = dict(base_payload)
            refined_payload["departureTime"] = _google_timestamp(recommended_departure)
            route_data = _route_request(refined_payload, maps_url)
    elif event_datetime and target_arrival:
        base_payload["arrivalTime"] = _google_timestamp(target_arrival)
        route_data = _route_request(base_payload, maps_url)
    else:
        route_data = _route_request(base_payload, maps_url)

    travel_seconds, distance_meters = _route_metrics(route_data, maps_url)
    event_iso, arrival, departure, ready, timezone_name, schedule_available, message = _schedule_values(
        event_datetime,
        travel_seconds,
        request.preparation_minutes,
        request.arrival_buffer_minutes,
    )
    duration_text = _format_duration(travel_seconds) if travel_seconds is not None else ""
    return {
        "distance_meters": distance_meters,
        "distance_text": _format_distance(distance_meters) if distance_meters is not None else "",
        "travel_duration_seconds": travel_seconds,
        "travel_duration_text": duration_text,
        "distance_km": round(distance_meters / 1000, 3) if distance_meters is not None else None,
        "duration_minutes": (travel_seconds + 59) // 60 if travel_seconds is not None else None,
        "event_start_time": event_iso,
        "arrival_time": arrival,
        "departure_time": departure,
        "ready_time": ready,
        "travel_mode": request.travel_mode,
        "destination": destination,
        "origin": origin,
        "timezone": timezone_name,
        "google_maps_url": maps_url,
        "schedule_available": schedule_available,
        "message": message,
        "schedule_message": message,
        # Routes traffic preference is only meaningful for supported motor
        # vehicle modes. Transit is timetable-based, not a live traffic claim.
        "traffic_aware": request.travel_mode in {"DRIVE", "TWO_WHEELER"},
    }
