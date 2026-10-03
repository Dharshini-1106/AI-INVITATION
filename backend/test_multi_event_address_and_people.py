"""Regression checks for shared locations and noisy people on event posters."""
import unittest

from app.core.understanding.parser import parse_invitation


class MultiEventAddressAndPeopleTests(unittest.TestCase):
    def test_full_shared_address_fills_incomplete_event_and_rejects_garble(self):
        text = """THE AMERICAN COLLEGE
ANNUAL DAY
10:00 AM - 1:00 PM
Friday, 16 October 2026
Venue
THE AMERICAN COLLEGE
Madurai 625002
SPORTS DAY
2:00 PM - 5:30 PM
Tallakulam, Madurai 625002, Tamil Nadu, India
Trcomnn BMM KUEITEITT mw TTT mML am"""

        result = parse_invitation({}, text)

        expected_address = "Tallakulam, Madurai 625002, Tamil Nadu, India"
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][0]["event_name"], "ANNUAL DAY")
        self.assertEqual(result["events"][1]["event_name"], "SPORTS DAY")
        self.assertEqual(result["events"][0]["address"], expected_address)
        self.assertEqual(result["events"][1]["address"], expected_address)
        self.assertEqual(result["people"], [])


if __name__ == "__main__":
    unittest.main()
