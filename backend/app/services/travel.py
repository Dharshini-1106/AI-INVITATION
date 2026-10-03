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
GOOGLE_PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_PLACE_FIELD_MASK = "places.id,places.displayName,places.formattedAddress,places.location,places.types"
_PLACE_STOP_WORDS = {
    "hall", "marriage", "ceremony", "wedding", "venue", "place", "center",
    "centre", "road", "street", "nagar", "colony", "area", "layout", "town",
    "city", "district", "state", "country", "india", "tamil", "nadu", "post",
    "office", "building", "tower", "complex", "mandapam", "mandap",
}
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
        destination_candidates: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.google_maps_url = google_maps_url
        self.destination_candidates = destination_candidates or []


def _google_maps_url(origin: str, destination: str, travel_mode: str, place_id: str = "") -> str:
    mode_map = {
        "DRIVE": "driving",
        "TWO_WHEELER": "two-wheeler",
        "TRANSIT": "transit",
    }
    params = {
        "origin": origin,
        "travelmode": mode_map.get(travel_mode, travel_mode.lower()),
    }
    if place_id:
        params["destination"] = destination
        params["destination_place_id"] = place_id
    else:
        params["destination"] = destination
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


def _destination_waypoint(value: str, place_id: str = "") -> dict[str, Any]:
    """Use Google's resolved place identity for both routing and Maps links."""
    if place_id:
        return {"placeId": place_id}
    return _waypoint(value)


def _place_score(candidate: dict[str, Any], venue: str, address: str) -> float:
    """Score a Google Places candidate against the extracted venue/address.

    The score rewards candidates whose name and address overlap the
    extracted context. It deliberately avoids awarding points for the
    locality alone, so an unrelated place with a similar town name does
    not win over the correct establishment.
    """
    score = 0.0
    display_name = candidate.get("displayName") or {}
    cand_name = (candidate.get("name") or display_name.get("text") or "").lower()
    cand_addr = (candidate.get("formatted_address") or candidate.get("formattedAddress") or "").lower()
    cand_types = set(t.lower() for t in (candidate.get("types") or []))

    venue_tokens = {
        tok for tok in re.split(r"\W+", venue.lower()) if tok and tok not in _PLACE_STOP_WORDS
    }
    address_tokens = {
        tok for tok in re.split(r"\W+", address.lower()) if tok and tok not in _PLACE_STOP_WORDS
    }
    address_tokens -= venue_tokens

    if venue_tokens and venue_tokens.issubset(set(cand_name.split())):
        score += 3.0
    elif venue_tokens:
        overlap = venue_tokens & set(cand_name.split())
        score += 1.5 * len(overlap)

    name_in_address = venue_tokens & set(cand_addr.split())
    score += 0.8 * len(name_in_address)

    addr_overlap = address_tokens & set(cand_addr.split())
    score += 0.25 * len(addr_overlap)

    if "establishment" in cand_types or "point_of_interest" in cand_types:
        score += 0.5
    if "restaurant" in cand_types or "lodging" in cand_types:
        score -= 1.0

    geom = candidate.get("geometry") or {}
    loc = candidate.get("location") or geom.get("location") or {}
    if (loc.get("latitude") is not None and loc.get("longitude") is not None) or (loc.get("lat") is not None and loc.get("lng") is not None):
        score += 0.1
    return score


def _resolve_place_id(venue: str, address: str, selected_place_id: str = "") -> dict[str, Any] | None:
    """Resolve OCR venue context with Places API (New), requiring confirmation for ambiguity."""
    if not settings.google_maps_api_key:
        return None

    query = f"{venue} {address}".strip()
    if not query:
        return None

    try:
        response = requests.post(
            GOOGLE_PLACES_SEARCH_URL,
            json={"textQuery": query, "languageCode": "en", "maxResultCount": 5},
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": settings.google_maps_api_key,
                "X-Goog-FieldMask": _PLACE_FIELD_MASK,
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        logger.warning("Google Places resolution failed: %s", exc)
        return None

    if response.status_code != 200:
        logger.warning(
            "Google Places resolution returned HTTP %s", response.status_code
        )
        return None

    try:
        data = response.json()
    except ValueError:
        logger.warning("Google Places returned unreadable response")
        return None

    candidates = data.get("places") or []
    if not candidates:
        logger.info("Google Places returned no candidates for: %s", query)
        return None

    normalized = []
    for candidate in candidates:
        location = candidate.get("location") or {}
        display_name = candidate.get("displayName") or {}
        normalized.append({
            "place_id": candidate.get("id") or "",
            "name": display_name.get("text") or candidate.get("name") or "",
            "formatted_address": candidate.get("formattedAddress") or candidate.get("formatted_address") or "",
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "types": candidate.get("types") or [],
        })
    normalized = [c for c in normalized if c["place_id"]]
    if not normalized:
        return None
    ranked = sorted(normalized, key=lambda c: _place_score(c, venue, address), reverse=True)
    if selected_place_id:
        best = next((c for c in normalized if c["place_id"] == selected_place_id), None)
        if best is None:
            return {"ambiguous": True, "candidates": normalized}
    else:
        best = ranked[0]
        best_score = _place_score(best, venue, address)
        next_score = _place_score(ranked[1], venue, address) if len(ranked) > 1 else -1
        if best_score < 3.5 or (len(ranked) > 1 and best_score - next_score < 1.0):
            return {"ambiguous": True, "candidates": ranked}

    logger.info(
        "Resolved destination place: %s (%s) score=%.2f",
        best.get("place_id"),
        best.get("name"),
        _place_score(best, venue, address),
    )
    return {
        **best,
    }


def _build_fallback_destination(venue: str, address: str) -> str:
    """Build a geocode-safe destination string when Google Places is unavailable.

    The full extracted address (venue + street + postal code) can geocode to a
    different location than the actual venue when the street/pin combination is
    ambiguous or OCR-mangled (e.g. ``Sivasami Maaligai Marriage Hall,
    Alangulam Road, Mukkudal, Tirunelveli - 627 758`` resolves ~63 km away from
    the real hall).

    Routing to ``venue, city, state`` (dropping the street, the postal code,
    and any landmark/locality fragments) resolves to the correct establishment
    because the venue name + city + state is unambiguous. The original
    extracted address is preserved for display; only the routing waypoint is
    rebuilt.
    """
    venue = (venue or "").strip().rstrip(".,;:")
    address = (address or "").strip().rstrip(".,;:")
    if not venue:
        return address

    # A Tamil-only venue title is often not geocodable even when the locality
    # has been normalized into an English map address. If Places could not
    # resolve that venue, route to the resolved locality instead of retrying
    # the same unsupported script as a combined free-text waypoint.
    if (re.search(r"[\u0B80-\u0BFF]", venue)
            and re.search(r"[A-Za-z]", address)):
        return address

    indian_states = {
        "tamil nadu", "kerala", "karnataka", "andhra pradesh", "telangana",
        "maharashtra", "gujarat", "rajasthan", "punjab", "haryana",
        "uttar pradesh", "bihar", "west bengal", "odisha", "assam",
        "madhya pradesh", "chhattisgarh", "jharkhand", "uttarakhand",
        "himachal pradesh", "tripura", "meghalaya", "manipur", "nagaland",
        "arunachal pradesh", "mizoram", "sikkim", "goa", "delhi",
    }
    country_words = {"india", "united states", "usa", "uk", "united kingdom"}

    street_keywords = re.compile(
        r"\b(?:road|rd\b|street|st\b|avenue|ave\b|lane|ln\b|drive|dr\b|"
        r"court|ct\b|place|pl\b|boulevard|blvd|nagar|colony|layout|"
        r"station|main\b|mg\b|estate|harbour|harbor|junction|crossing|"
        r"by-pass|bypass|highway|national|sector|phase|puram|post\b|office|"
        r"bus stop|busstand|ring road|gst road|bazaar|market|depot|terminal|"
        r"harbour estate|salt pans)\b",
        re.IGNORECASE,
    )

    segments = [s.strip().rstrip(".,;:") for s in address.split(",")]

    # Pass 1: find the state/pin anchor and the country (if any).
    state = ""
    country = ""
    pin_index = None
    state_index = None
    for i, seg in enumerate(segments):
        low = seg.lower()
        if not seg:
            continue
        if low in country_words:
            country = seg
            continue
        if low in indian_states:
            state = seg
            state_index = i
            continue
        if re.search(r"\d{3}\s*\d{3}|\d{5,6}", seg):
            pin_index = i
            if not state:
                m = re.match(
                    r"^(.+?)(?:\s*[-–—]?\s*\d{3}\s*\d{3}|\s*\d{5,6})",
                    seg,
                )
                if m:
                    state = m.group(1).strip()

    # Pass 2: the city is the nearest non-street segment before the pin/state
    # anchor. When the pin is embedded in the same segment as the city
    # (e.g. "Tirunelveli - 627 758"), extract the city portion from that
    # segment. If no anchor exists, use the first non-street segment.
    city = ""
    anchor_index = pin_index if pin_index is not None else state_index
    if anchor_index is not None:
        if pin_index is not None:
            pin_seg = segments[pin_index]
            m = re.match(
                r"^(.+?)(?:\s*[-–—]?\s*\d{3}\s*\d{3}|\s*\d{5,6})",
                pin_seg,
            )
            if m:
                prefix = m.group(1).strip()
                # Only treat the pin segment's prefix as the city when it is
                # NOT an Indian state name. When the pin segment is
                # "Tamil Nadu 628501", the city is the preceding segment.
                if prefix and prefix.lower() not in indian_states:
                    if not street_keywords.search(prefix):
                        city = prefix
        if not city:
            for j in range(anchor_index - 1, -1, -1):
                candidate = segments[j]
                if not candidate:
                    continue
                if street_keywords.search(candidate):
                    continue
                if candidate.lower() in country_words:
                    continue
                city = candidate
                break
    if not city:
        for seg in segments:
            if not seg:
                continue
            if street_keywords.search(seg):
                continue
            if seg.lower() in country_words:
                continue
            city = seg
            break

    parts = []
    for piece in (city, state, country):
        if piece and piece.lower() not in {p.lower() for p in parts}:
            parts.append(piece)

    if not parts:
        return venue
    context = ", ".join(parts)
    if context.lower() in venue.lower():
        return venue
    return f"{venue}, {context}"


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
    now = datetime.now(event_datetime.tzinfo)
    if event_datetime <= now:
        return (
            _google_timestamp(event_datetime),
            "",
            "",
            "",
            timezone_name,
            False,
            "This event has already passed. Google Maps can show current route time, but a future event-day schedule is unavailable.",
        )
    if event_datetime - timedelta(minutes=arrival_buffer_minutes) <= now:
        return (
            _google_timestamp(event_datetime),
            "",
            "",
            "",
            timezone_name,
            False,
            "The requested arrival time has passed. The route shows current travel time; an event-day schedule is unavailable.",
        )

    arrival_target = event_datetime - timedelta(minutes=arrival_buffer_minutes)
    departure = arrival_target - timedelta(seconds=travel_seconds)
    if departure <= now:
        return (
            _google_timestamp(event_datetime),
            _google_timestamp(arrival_target),
            "",
            "",
            timezone_name,
            False,
            "The calculated departure time has passed. Adjust the arrival buffer or event time to make a future schedule.",
        )
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
    logger.info(
        "Google Routes request: travel_mode=%s departure_time=%s arrival_time=%s",
        payload.get("travelMode"),
        payload.get("departureTime", ""),
        payload.get("arrivalTime", ""),
    )
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
            "The backend could not reach Google Routes. Check the PC's internet connection and try again.",
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
            safe_detail = detail.replace("\r", " ").replace("\n", " ").strip()[:240]
            invalid_future_timestamp = "timestamp" in safe_detail.lower() and "future" in safe_detail.lower()
            if invalid_future_timestamp:
                message = "Google Routes rejected its route timestamp. Check the PC clock and event timezone, then retry."
            else:
                message = "Google Routes could not calculate this route. Check the origin and destination."
            if safe_detail and not invalid_future_timestamp:
                message = f"{message} Google reported: {safe_detail}"
            raise TravelPlanningError(
                message,
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
        safe_detail = detail.replace("\r", " ").replace("\n", " ").strip()[:240]
        raise TravelPlanningError(
            f"Google Routes returned HTTP {response.status_code}: {safe_detail or 'no error details provided'}",
            "route_unavailable",
            status_code=502,
            google_maps_url=maps_url,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise TravelPlanningError(
            "Google Routes returned an unreadable response. Check the backend terminal for details.",
            "route_unavailable",
            google_maps_url=maps_url,
        ) from exc


def _route_metrics(
    data: dict[str, Any],
    maps_url: str,
    travel_mode: str = "",
) -> tuple[int | None, int | None]:
    routes = data.get("routes") or []
    if not routes:
        raise TravelPlanningError(
            "Google Maps found no route. Check the destination address or try another travel mode.",
            "route_not_found",
            status_code=404,
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
    logger.info(
        "Google Routes result: travel_mode=%s duration_seconds=%s distance_meters=%s",
        travel_mode,
        travel_seconds,
        distance_meters,
    )
    return travel_seconds, distance_meters


def plan_travel(request: TravelPlanRequest) -> dict[str, Any]:
    logger.info("Travel plan API request: travel_mode=%s", request.travel_mode)
    origin = _location_value(request.origin, "origin")
    raw_destination = _location_value(request.destination, "destination")

    if not origin:
        raise TravelPlanningError(
            "Enter your starting location or use your current location before calculating a route.",
            "origin_required",
            status_code=400,
            google_maps_url=_google_maps_url("", raw_destination, request.travel_mode),
        )
    if not raw_destination:
        raise TravelPlanningError(
            "Enter the event destination before calculating a route.",
            "destination_required",
            status_code=400,
            google_maps_url=_google_maps_url(origin, "", request.travel_mode),
        )

    # Resolve the destination to a specific Google Place when the caller
    # passed a free-text venue/address. This keeps the routing destination
    # aligned with the place Google Maps itself resolves, while the
    # original extracted text is preserved for display.
    venue = (request.destination_venue or "").strip() if hasattr(request, "destination_venue") else ""
    address = (request.destination_address or "").strip() if hasattr(request, "destination_address") else ""
    if isinstance(request.destination, str):
        address = address or raw_destination
    place = _resolve_place_id(
        venue,
        address,
        getattr(request, "destination_place_id", "") or "",
    ) if not _COORDINATE_RE.match(raw_destination) else None

    routing_destination = raw_destination
    place_id = ""
    display_destination = raw_destination
    maps_destination = raw_destination
    if place:
        if place.get("ambiguous"):
            maps_url = _google_maps_url(origin, raw_destination, request.travel_mode)
            raise TravelPlanningError(
                "Choose the correct Google Maps destination before calculating this route.",
                "destination_confirmation_required",
                status_code=409,
                google_maps_url=maps_url,
                destination_candidates=place.get("candidates", []),
            )
        place_id = place["place_id"] or ""
        maps_destination = place.get("name") or place.get("formatted_address") or raw_destination
    elif venue and not _COORDINATE_RE.match(raw_destination):
        if not getattr(request, "confirm_unverified_destination", False):
            maps_url = _google_maps_url(origin, raw_destination, request.travel_mode)
            raise TravelPlanningError(
                "Google Maps could not verify this venue. Confirm the typed destination to continue.",
                "destination_confirmation_required",
                status_code=409,
                google_maps_url=maps_url,
            )

    maps_url = _google_maps_url(origin, maps_destination, request.travel_mode, place_id)
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
    base_payload: dict[str, Any] = {
        "origin": _waypoint(origin),
        "destination": _destination_waypoint(routing_destination, place_id),
        "travelMode": request.travel_mode,
        "computeAlternativeRoutes": False,
    }

    # Use a single current route result for duration, distance, and the
    # event schedule. Omitting departureTime makes Routes use request time,
    # matching the Maps link's current "Leave now" estimate.
    if request.travel_mode in {"DRIVE", "TWO_WHEELER"}:
        base_payload["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
    route_data = _route_request(base_payload, maps_url)
    travel_seconds, distance_meters = _route_metrics(
        route_data, maps_url, request.travel_mode
    )
    event_iso, arrival, departure, ready, timezone_name, schedule_available, message = _schedule_values(
        event_datetime,
        travel_seconds,
        request.preparation_minutes,
        request.arrival_buffer_minutes,
    )
    duration_text = _format_duration(travel_seconds) if travel_seconds is not None else ""
    if request.travel_mode in {"DRIVE", "TWO_WHEELER"}:
        duration_label = "Google Maps traffic-aware estimate (at calculation time)"
    else:
        duration_label = "Google Maps route estimate (at calculation time)"
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
        "destination": display_destination,
        "destination_place_id": place_id,
        "destination_place_name": place.get("name") if place else "",
        "destination_resolved_address": place.get("formatted_address", "") if place else "",
        "destination_latitude": place.get("latitude") if place else None,
        "destination_longitude": place.get("longitude") if place else None,
        "origin": origin,
        "timezone": timezone_name,
        "google_maps_url": maps_url,
        "schedule_available": schedule_available,
        "message": message,
        "schedule_message": message,
        # Routes traffic preference is only meaningful for supported motor
        # vehicle modes. Transit is timetable-based, not a live traffic claim.
        "traffic_aware": request.travel_mode in {"DRIVE", "TWO_WHEELER"},
        "travel_duration_label": duration_label,
        "route_note": "The schedule applies this route duration, measured with traffic at calculation time, to the event time. Google Maps may show a different ETA if departure time or live traffic changes, or if it selects another route.",
    }
