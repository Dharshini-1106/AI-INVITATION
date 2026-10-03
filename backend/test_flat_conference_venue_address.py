"""Regression coverage for flattened conference poster OCR."""
import unittest

from app.core.understanding.parser import parse_invitation


class FlatConferenceVenueAddressTests(unittest.TestCase):
    def test_host_institution_wins_over_associated_center_and_city_is_complete(self):
        text = (
            "SATHYABAMA INSTITUTE OF SCIENCE AND TECHNOLOGY DEPARTMENT OF "
            "INFORMATION TECHNOLOGY In association with CENTRE FOR MOLECULAR "
            "AND NANOMEDICAL SCIENCES CHENNAI, INDIA CONFERENCE ON ADVANCES "
            "IN BIOSCIENCES, DATA SCIENCE AND AI ICABDAI 2026 "
            "16 - 18 NOVEMBER 2026 Emerging Technologies"
        )
        result = parse_invitation(
            {
                "event_name": "ICABDAI 2026",
                "event_type": "Conference",
                "venue": "CENTRE FOR MOLECULAR AND NANOMEDICAL SCIENCES",
                "address": "Chennai",
            },
            text,
        )

        expected_venue = "SATHYABAMA INSTITUTE OF SCIENCE AND TECHNOLOGY"
        self.assertEqual(result["venue"], expected_venue)
        self.assertEqual(result["address"], "Chennai, India")
        self.assertEqual(result["events"][0]["venue"], expected_venue)
        self.assertEqual(result["events"][0]["address"], "Chennai, India")
        self.assertEqual(result["people"], [])


if __name__ == "__main__":
    unittest.main()
