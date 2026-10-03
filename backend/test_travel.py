"""Focused tests for Google Routes travel planning."""
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from app.config import settings
from app.schemas.travel import TravelPlanRequest, TravelPlanResponse
from app.services.travel import (
    TravelPlanningError,
    _google_maps_url,
    plan_travel,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self.payload


def _no_place(*_args, **_kwargs):
    """Disable Google Places resolution so tests stay hermetic."""
    return None


def _fake_place(*_args, **_kwargs):
    return {
        "place_id": "place-123",
        "name": "Reva Plaza",
        "formatted_address": "Reva Plaza, Kovilpatti Main Rd, Kovilpatti, Tamil Nadu 628501",
        "latitude": 9.1723,
        "longitude": 77.5612,
    }


class TravelPlanningTests(unittest.TestCase):
    def make_request(self, **overrides):
        values = {
            "origin": "Central Station",
            "destination": "Reva Plaza, Kovilpatti",
            "destination_venue": "Reva Plaza",
            "destination_address": "Kovilpatti Main Rd, Kovilpatti, Tamil Nadu 628501",
            "event_date": "October 2, 2026",
            "event_start_time": "6:00 PM",
            "event_timezone": "Asia/Kolkata",
            "travel_mode": "DRIVE",
            "preparation_minutes": 60,
            "arrival_buffer_minutes": 15,
            # Existing routing tests exercise route behavior after the user's
            # explicit acknowledgement of an unresolved free-text destination.
            "confirm_unverified_destination": True,
        }
        values.update(overrides)
        return TravelPlanRequest(**values)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_drive_uses_one_current_traffic_route_for_metrics_and_schedule(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{
                "duration": "2700s",
                "distanceMeters": 28000,
            }]
        })

        result = plan_travel(self.make_request())

        self.assertEqual(post.call_count, 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["routingPreference"], "TRAFFIC_AWARE_OPTIMAL")
        self.assertNotIn("departureTime", payload)
        self.assertNotIn("arrivalTime", payload)
        self.assertEqual(payload["computeAlternativeRoutes"], False)
        self.assertEqual(result["event_start_time"], "2026-10-02T18:00:00+05:30")
        self.assertEqual(result["arrival_time"], "2026-10-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-10-02T17:00:00+05:30")
        self.assertEqual(result["ready_time"], "2026-10-02T16:00:00+05:30")
        self.assertEqual(result["travel_duration_text"], "45 min")
        self.assertEqual(result["distance_text"], "28 km")
        self.assertEqual(result["travel_duration_seconds"], 2700)
        self.assertTrue(result["traffic_aware"])
        self.assertIn("traffic-aware", result["travel_duration_label"])
        self.assertEqual(result["schedule_message"], result["message"])

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_maps_duration_distance_and_schedule_use_same_route_response(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "6960s", "distanceMeters": 89900}]
        })

        result = plan_travel(self.make_request())
        api_response = TravelPlanResponse(**result)

        self.assertEqual(api_response.travel_duration_text, "1 hr 56 min")
        self.assertEqual(api_response.distance_text, "89.9 km")
        self.assertEqual(api_response.arrival_time, "2026-10-02T17:45:00+05:30")
        self.assertEqual(api_response.departure_time, "2026-10-02T15:49:00+05:30")
        self.assertEqual(api_response.ready_time, "2026-10-02T14:49:00+05:30")
        self.assertIn("at calculation time", api_response.travel_duration_label)
        self.assertIn("another route", api_response.route_note)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_transit_uses_current_route_without_traffic_preference(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "2400s", "distanceMeters": 18000}]
        })

        plan_travel(self.make_request(travel_mode="TRANSIT"))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["travelMode"], "TRANSIT")
        self.assertNotIn("routingPreference", payload)
        self.assertNotIn("arrivalTime", payload)
        self.assertNotIn("departureTime", payload)

    @patch.object(settings, "google_maps_api_key", "")
    def test_missing_google_key_still_returns_maps_url(self):
        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request())

        self.assertEqual(context.exception.code, "google_maps_not_configured")
        self.assertIn("origin=Central+Station", context.exception.google_maps_url)
        self.assertIn("travelmode=driving", context.exception.google_maps_url)

    def test_google_maps_url_uses_selected_mode(self):
        url = _google_maps_url("Start", "Destination", "TWO_WHEELER")
        self.assertIn("travelmode=two-wheeler", url)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_two_wheeler_uses_traffic_aware_route(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "1500s", "distanceMeters": 9000}]
        })

        result = plan_travel(self.make_request(travel_mode="TWO_WHEELER"))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["travelMode"], "TWO_WHEELER")
        self.assertEqual(payload["routingPreference"], "TRAFFIC_AWARE_OPTIMAL")
        self.assertEqual(result["travel_duration_text"], "25 min")
        self.assertEqual(result["distance_text"], "9 km")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_selected_drive_and_two_wheeler_modes_reach_google_unchanged(self, post, _place):
        post.side_effect = [
            FakeResponse({"routes": [{"duration": "11820s", "distanceMeters": 184000}]}),
            FakeResponse({"routes": [{"duration": "12060s", "distanceMeters": 184000}]}),
        ]
        common = {
            "origin": "Thiruchendur, Tamil Nadu",
            "destination": "Sri Nithya Mahal, Pandi Kovil, Madurai",
            "event_date": "October 23, 2026",
            "event_start_time": "10:00 AM",
            "event_timezone": "Asia/Kolkata",
            "preparation_minutes": 60,
            "arrival_buffer_minutes": 15,
        }

        drive = plan_travel(TravelPlanRequest(**common, travel_mode="DRIVE"))
        two_wheeler = plan_travel(
            TravelPlanRequest(**common, travel_mode="TWO_WHEELER")
        )

        self.assertEqual(post.call_args_list[0].kwargs["json"]["travelMode"], "DRIVE")
        self.assertEqual(post.call_args_list[1].kwargs["json"]["travelMode"], "TWO_WHEELER")
        self.assertEqual(drive["travel_duration_text"], "3 hr 17 min")
        self.assertEqual(two_wheeler["travel_duration_text"], "3 hr 21 min")
        self.assertEqual(drive["distance_meters"], 184000)
        self.assertEqual(drive["departure_time"], "2026-10-23T06:28:00+05:30")
        self.assertEqual(drive["ready_time"], "2026-10-23T05:28:00+05:30")
        self.assertEqual(two_wheeler["departure_time"], "2026-10-23T06:24:00+05:30")
        self.assertEqual(two_wheeler["ready_time"], "2026-10-23T05:24:00+05:30")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_coordinate_waypoints_are_sent_to_google(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "600s", "distanceMeters": 3000}]
        })

        plan_travel(self.make_request(
            origin="13.0827,80.2707",
            destination="12.9716,77.5946",
            event_start_time="",
        ))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["origin"]["location"]["latLng"]["latitude"], 13.0827)
        self.assertEqual(payload["origin"]["location"]["latLng"]["longitude"], 80.2707)
        self.assertEqual(payload["destination"]["location"]["latLng"]["latitude"], 12.9716)
        self.assertEqual(payload["destination"]["location"]["latLng"]["longitude"], 77.5946)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_structured_current_location_keeps_exact_coordinates(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "600s", "distanceMeters": 3000}]
        })

        plan_travel(self.make_request(
            origin={"type": "current", "latitude": 13.0827, "longitude": 80.2707},
            destination={"type": "event", "address": "Convention Center"},
            destination_venue="",
            destination_address="",
            event_start_time="",
        ))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["origin"]["location"]["latLng"]["latitude"], 13.0827)
        self.assertEqual(payload["origin"]["location"]["latLng"]["longitude"], 80.2707)
        self.assertEqual(payload["destination"], {"address": "Convention Center"})

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    def test_invalid_event_timezone_returns_client_error(self, _place):
        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request(event_timezone="Not/AZone"))

        self.assertEqual(context.exception.code, "timezone_invalid")
        self.assertEqual(context.exception.status_code, 400)

    def test_missing_origin_and_destination_include_maps_url(self):
        with patch("app.services.travel._resolve_place_id", side_effect=_no_place):
            for field in ("origin", "destination"):
                values = {
                    "origin": "Start",
                    "destination": "Venue",
                }
                values[field] = ""
                with self.subTest(field=field):
                    with self.assertRaises(TravelPlanningError) as context:
                        plan_travel(TravelPlanRequest(**values))
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertTrue(context.exception.google_maps_url)
                    self.assertIn("origin=", context.exception.google_maps_url)

    @patch("app.services.travel._resolve_place_id")
    def test_missing_origin_is_rejected_before_place_or_route_requests(self, place):
        with patch("app.services.travel.requests.post") as post:
            with self.assertRaises(TravelPlanningError) as context:
                plan_travel(self.make_request(origin=""))

        self.assertEqual(context.exception.code, "origin_required")
        self.assertIn("starting location", context.exception.message)
        place.assert_not_called()
        post.assert_not_called()

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_google_invalid_argument_returns_client_error(self, post, _place):
        post.return_value = FakeResponse({
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": "Invalid origin or destination",
            }
        }, status_code=400)

        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request())

        self.assertEqual(context.exception.code, "route_invalid")
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("travelmode=driving", context.exception.google_maps_url)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_google_past_timestamp_error_has_useful_message(self, post, _place):
        post.return_value = FakeResponse({
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": "Timestamp must be set to a future time.",
            }
        }, status_code=400)

        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request())

        self.assertIn("timestamp", context.exception.message.lower())
        self.assertIn("retry", context.exception.message.lower())
        self.assertNotIn("departureTime", post.call_args.kwargs["json"])

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_google_quota_error_is_preserved(self, post, _place):
        post.return_value = FakeResponse({
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded",
            }
        }, status_code=429)

        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request())

        self.assertEqual(context.exception.code, "route_quota_exceeded")
        self.assertEqual(context.exception.status_code, 429)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_missing_event_time_leaves_schedule_unavailable(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "1800s", "distanceMeters": 12000}]
        })

        result = plan_travel(self.make_request(event_start_time=""))

        self.assertFalse(result["schedule_available"])
        self.assertEqual(result["arrival_time"], "")
        self.assertEqual(result["departure_time"], "")
        self.assertEqual(result["ready_time"], "")
        self.assertEqual(result["travel_duration_text"], "30 min")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_time_qualifiers_use_event_start_time(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "1800s", "distanceMeters": 12000}]
        })

        result = plan_travel(self.make_request(event_start_time="6:00 PM onwards"))

        self.assertEqual(result["event_start_time"], "2026-10-02T18:00:00+05:30")
        self.assertEqual(result["arrival_time"], "2026-10-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-10-02T17:15:00+05:30")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_fake_place)
    @patch("app.services.travel.requests.post")
    def test_resolved_place_id_is_used_by_routes_and_maps(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "7260s", "distanceMeters": 113000}]
        })

        result = plan_travel(self.make_request())

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["destination"], {"placeId": "place-123"})
        self.assertEqual(result["destination_place_id"], "place-123")
        self.assertEqual(result["destination_place_name"], "Reva Plaza")
        self.assertEqual(result["distance_km"], 113.0)
        self.assertEqual(result["travel_duration_text"], "2 hr 1 min")
        self.assertIn("destination_place_id=place-123", result["google_maps_url"])
        self.assertIn("destination=Reva+Plaza", result["google_maps_url"])

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_november_birthday_schedule_uses_current_route_and_no_timestamp(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "6960s", "distanceMeters": 89900}]
        })

        result = plan_travel(self.make_request(
            event_date="November 2, 2026",
            event_start_time="6:00 PM",
            event_timezone="Asia/Kolkata",
            arrival_buffer_minutes=15,
        ))

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("departureTime", payload)
        self.assertEqual(result["event_start_time"], "2026-11-02T18:00:00+05:30")
        self.assertEqual(result["arrival_time"], "2026-11-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-11-02T15:49:00+05:30")
        self.assertEqual(result["travel_duration_text"], "1 hr 56 min")
        self.assertEqual(result["distance_text"], "89.9 km")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", return_value={
        "place_id": "sivasami-place-id",
        "name": "Sivasami Maaligai",
        "formatted_address": "Sivasami Maaligai - சிவசாமி மாளிகை, 52/46, Mukkudal, Tamil Nadu 627601",
    })
    @patch("app.services.travel.requests.post")
    def test_tiruchendur_to_sivasami_uses_same_place_and_current_result_in_maps(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "7080s", "distanceMeters": 89900}]
        })
        result = plan_travel(self.make_request(
            origin="Tiruchendur, Tamil Nadu",
            destination="Sivasami Maaligai, 436 Main Road, Kovilpatti, Tamil Nadu 628502",
            destination_venue="Sivasami Maaligai",
            destination_address="436, Main Road, Kovilpatti, Tamil Nadu - 628502",
            event_date="November 2, 2026",
            event_start_time="6:00 PM",
            preparation_minutes=60,
            arrival_buffer_minutes=15,
        ))

        self.assertEqual(post.call_count, 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["origin"], {"address": "Tiruchendur, Tamil Nadu"})
        self.assertEqual(payload["destination"], {"placeId": "sivasami-place-id"})
        self.assertEqual(payload["travelMode"], "DRIVE")
        self.assertNotIn("departureTime", payload)
        maps_params = parse_qs(urlparse(result["google_maps_url"]).query)
        self.assertEqual(maps_params["origin"], ["Tiruchendur, Tamil Nadu"])
        self.assertEqual(maps_params["destination"], ["Sivasami Maaligai"])
        self.assertEqual(maps_params["destination_place_id"], ["sivasami-place-id"])
        self.assertEqual(maps_params["travelmode"], ["driving"])
        self.assertEqual(result["travel_duration_text"], "1 hr 58 min")
        self.assertEqual(result["distance_text"], "89.9 km")
        self.assertEqual(result["arrival_time"], "2026-11-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-11-02T15:47:00+05:30")
        self.assertEqual(result["ready_time"], "2026-11-02T14:47:00+05:30")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", side_effect=_no_place)
    @patch("app.services.travel.requests.post")
    def test_coordinate_destination_skips_place_resolution(self, post, _place):
        post.return_value = FakeResponse({
            "routes": [{"duration": "600s", "distanceMeters": 3000}]
        })

        plan_travel(self.make_request(
            origin="13.0827,80.2707",
            destination="12.9716,77.5946",
            event_start_time="",
        ))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["destination"]["location"]["latLng"]["latitude"], 12.9716)
        self.assertEqual(payload["destination"]["location"]["latLng"]["longitude"], 77.5946)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_places_new_matching_address_forms_resolve_to_same_place_and_route(self, post):
        post.side_effect = [
            FakeResponse({"places": [{
                "id": "ChIJOwJ-L4Q9BDsRMibs9mtetCk",
                "displayName": {"text": "Sivasami Maaligai - Sivasami Hall"},
                "formattedAddress": "52/46, Mukkudal, Tamil Nadu 627601, India",
                "location": {"latitude": 8.7422923, "longitude": 77.5240218},
                "types": ["wedding_venue", "establishment"],
            }]}),
            FakeResponse({"places": [{
                "id": "ChIJOwJ-L4Q9BDsRMibs9mtetCk",
                "displayName": {"text": "Sivasami Maaligai - Sivasami Hall"},
                "formattedAddress": "52/46, Mukkudal, Tamil Nadu 627601, India",
                "location": {"latitude": 8.7422923, "longitude": 77.5240218},
                "types": ["wedding_venue", "establishment"],
            }]}),
            FakeResponse({"places": [{
                "id": "ChIJOwJ-L4Q9BDsRMibs9mtetCk",
                "displayName": {"text": "Sivasami Maaligai - Sivasami Hall"},
                "formattedAddress": "52/46, Mukkudal, Tamil Nadu 627601, India",
                "location": {"latitude": 8.7422923, "longitude": 77.5240218},
                "types": ["wedding_venue", "establishment"],
            }]}),
            FakeResponse({"routes": [{"duration": "7277s", "distanceMeters": 89917}]}),
        ]
        from app.services.travel import _resolve_place_id

        invitation_match = _resolve_place_id(
            "Sivasami Maaligai Marriage Hall",
            "Alangulam Road, Mukkudal, Tirunelveli - 627 758, Tamil Nadu, India",
        )
        maps_match = _resolve_place_id(
            "Sivasami Maaligai Marriage Hall",
            "Sivasami Maaligai - Sivasami Hall, 52/46, Mukkudal, Tamil Nadu 627601",
        )
        result = plan_travel(self.make_request(
            origin="Tiruchendur, Tamil Nadu",
            destination="Sivasami Maaligai Marriage Hall, Alangulam Road, Mukkudal, Tirunelveli - 627 758, Tamil Nadu, India",
            destination_venue="Sivasami Maaligai Marriage Hall",
            destination_address="Alangulam Road, Mukkudal, Tirunelveli - 627 758, Tamil Nadu, India",
        ))
        api_call = post.call_args
        self.assertEqual(invitation_match["place_id"], maps_match["place_id"])
        self.assertEqual(api_call.kwargs["json"]["origin"], {"address": "Tiruchendur, Tamil Nadu"})
        self.assertEqual(api_call.kwargs["json"]["destination"], {"placeId": invitation_match["place_id"]})
        self.assertEqual(result["destination"], "Sivasami Maaligai Marriage Hall, Alangulam Road, Mukkudal, Tirunelveli - 627 758, Tamil Nadu, India")
        self.assertEqual(result["destination_resolved_address"], "52/46, Mukkudal, Tamil Nadu 627601, India")
        self.assertEqual(result["travel_duration_seconds"], 7277)
        self.assertEqual(result["distance_meters"], 89917)
        self.assertEqual(result["distance_text"], "89.9 km")
        maps_params = parse_qs(urlparse(result["google_maps_url"]).query)
        self.assertEqual(maps_params["destination_place_id"], [invitation_match["place_id"]])
        self.assertEqual(maps_params["origin"], ["Tiruchendur, Tamil Nadu"])
        self.assertIn("places:searchText", post.call_args_list[0].args[0])

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post", return_value=FakeResponse({"places": [
        {"id": "place-a", "displayName": {"text": "Sivasami Maaligai Hall"}, "formattedAddress": "Mukkudal"},
        {"id": "place-b", "displayName": {"text": "Sivasami Maaligai Hall"}, "formattedAddress": "Mukkudal"},
    ]}))
    def test_ambiguous_places_are_returned_for_user_confirmation(self, _post):
        from app.services.travel import _resolve_place_id
        result = _resolve_place_id("Sivasami Maaligai Marriage Hall", "Alangulam Road, Mukkudal")
        self.assertTrue(result["ambiguous"])
        self.assertEqual({c["place_id"] for c in result["candidates"]}, {"place-a", "place-b"})

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel._resolve_place_id", return_value=None)
    @patch("app.services.travel.requests.post")
    def test_unverified_venue_requires_typed_destination_confirmation(self, route_post, _place):
        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request(confirm_unverified_destination=False))
        self.assertEqual(context.exception.code, "destination_confirmation_required")
        route_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
