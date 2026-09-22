"""Focused tests for Google Routes travel planning."""
import unittest
from unittest.mock import patch

from app.config import settings
from app.schemas.travel import TravelPlanRequest
from app.services.travel import TravelPlanningError, _google_maps_url, plan_travel


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self.payload


class TravelPlanningTests(unittest.TestCase):
    def make_request(self, **overrides):
        values = {
            "origin": "Central Station",
            "destination": "Reva Plaza, Kovilpatti",
            "event_date": "October 2, 2026",
            "event_start_time": "6:00 PM",
            "event_timezone": "Asia/Kolkata",
            "travel_mode": "DRIVE",
            "preparation_minutes": 60,
            "arrival_buffer_minutes": 15,
        }
        values.update(overrides)
        return TravelPlanRequest(**values)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_drive_uses_iterative_departure_and_local_timezone(self, post):
        post.side_effect = [
            FakeResponse({
                "routes": [{
                    "duration": "1800s",
                    "distanceMeters": 20000,
                }]
            }),
            FakeResponse({
                "routes": [{
                    "duration": "2700s",
                    "distanceMeters": 28000,
                }]
            }),
        ]

        result = plan_travel(self.make_request())

        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertNotIn("departureTime", first_payload)
        self.assertEqual(
            second_payload["departureTime"],
            "2026-10-02T17:15:00+05:30",
        )
        self.assertEqual(result["event_start_time"], "2026-10-02T18:00:00+05:30")
        self.assertEqual(result["arrival_time"], "2026-10-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-10-02T17:00:00+05:30")
        self.assertEqual(result["ready_time"], "2026-10-02T16:00:00+05:30")
        self.assertEqual(result["travel_duration_text"], "45 min")
        self.assertEqual(result["distance_text"], "28 km")
        self.assertEqual(result["schedule_message"], result["message"])

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_transit_uses_target_arrival_time(self, post):
        post.return_value = FakeResponse({
            "routes": [{"duration": "2400s", "distanceMeters": 18000}]
        })

        plan_travel(self.make_request(travel_mode="TRANSIT"))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["travelMode"], "TRANSIT")
        self.assertNotIn("routingPreference", payload)
        self.assertEqual(payload["arrivalTime"], "2026-10-02T17:45:00+05:30")

    @patch.object(settings, "google_maps_api_key", "")
    def test_missing_google_key_still_returns_maps_url(self):
        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request())

        self.assertEqual(context.exception.code, "google_maps_not_configured")
        self.assertIn("origin=Central+Station", context.exception.google_maps_url)
        self.assertIn("travelmode=driving", context.exception.google_maps_url)

    def test_google_maps_url_uses_selected_mode(self):
        url = _google_maps_url("Start", "Destination", "TWO_WHEELER")
        self.assertIn("travelmode=bicycling", url)

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_two_wheeler_uses_traffic_aware_route(self, post):
        post.return_value = FakeResponse({
            "routes": [{"duration": "1500s", "distanceMeters": 9000}]
        })

        result = plan_travel(self.make_request(travel_mode="TWO_WHEELER"))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["travelMode"], "TWO_WHEELER")
        self.assertEqual(payload["routingPreference"], "TRAFFIC_AWARE")
        self.assertEqual(result["travel_duration_text"], "25 min")
        self.assertEqual(result["distance_text"], "9 km")

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_coordinate_waypoints_are_sent_to_google(self, post):
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
    @patch("app.services.travel.requests.post")
    def test_structured_current_location_keeps_exact_coordinates(self, post):
        post.return_value = FakeResponse({
            "routes": [{"duration": "600s", "distanceMeters": 3000}]
        })

        plan_travel(self.make_request(
            origin={"type": "current", "latitude": 13.0827, "longitude": 80.2707},
            destination={"type": "event", "address": "Convention Center"},
            event_start_time="",
        ))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["origin"]["location"]["latLng"]["latitude"], 13.0827)
        self.assertEqual(payload["origin"]["location"]["latLng"]["longitude"], 80.2707)
        self.assertEqual(payload["destination"], {"address": "Convention Center"})

    @patch.object(settings, "google_maps_api_key", "test-key")
    def test_invalid_event_timezone_returns_client_error(self):
        with self.assertRaises(TravelPlanningError) as context:
            plan_travel(self.make_request(event_timezone="Not/AZone"))

        self.assertEqual(context.exception.code, "timezone_invalid")
        self.assertEqual(context.exception.status_code, 400)

    def test_missing_origin_and_destination_include_maps_url(self):
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

    @patch.object(settings, "google_maps_api_key", "test-key")
    @patch("app.services.travel.requests.post")
    def test_google_invalid_argument_returns_client_error(self, post):
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
    @patch("app.services.travel.requests.post")
    def test_google_quota_error_is_preserved(self, post):
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
    @patch("app.services.travel.requests.post")
    def test_missing_event_time_leaves_schedule_unavailable(self, post):
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
    @patch("app.services.travel.requests.post")
    def test_time_qualifiers_use_event_start_time(self, post):
        post.return_value = FakeResponse({
            "routes": [{"duration": "1800s", "distanceMeters": 12000}]
        })

        result = plan_travel(self.make_request(event_start_time="6:00 PM onwards"))

        self.assertEqual(result["event_start_time"], "2026-10-02T18:00:00+05:30")
        self.assertEqual(result["arrival_time"], "2026-10-02T17:45:00+05:30")
        self.assertEqual(result["departure_time"], "2026-10-02T17:15:00+05:30")


if __name__ == "__main__":
    unittest.main()
