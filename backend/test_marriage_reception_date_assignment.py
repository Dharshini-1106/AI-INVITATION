"""Regression coverage for relationship roles and multi-day wedding OCR."""
import unittest

from app.core.understanding.parser import parse_invitation


class MarriageReceptionDateAssignmentTests(unittest.TestCase):
    def test_full_couple_names_and_distinct_wedding_function_marriage_dates(self):
        text = """Venkatha Reddy
Daughter of Ankita Reddy & Mr. Suman Reddy
WEDS
Ananya Gauru
Son of Mrs. Poonan Gauru and Dr. Ankit Gauru
Wedding Function
Mahandi & Sangeet
21th December 23, 10:00am
Venue: Hotel Marine Blue, HYderabad
Marriage
23th December 23
Reception
25th December 23
DINNEr: 7:OOPM ONWARDS
Your Presence is requested"""

        result = parse_invitation({}, text)

        self.assertEqual(result["groom_name"], "Ananya Gauru")
        self.assertEqual(result["bride_name"], "Venkatha Reddy")
        self.assertEqual(
            result["people"],
            [
                {"name": "Ananya Gauru", "role": "Groom"},
                {"name": "Venkatha Reddy", "role": "Bride"},
            ],
        )
        self.assertEqual(len(result["events"]), 3)
        self.assertEqual(result["events"][0]["date"], "December 21, 2023")
        self.assertEqual(result["events"][1]["event_name"], "Marriage")
        self.assertEqual(result["events"][1]["date"], "December 23, 2023")
        self.assertEqual(result["events"][2]["event_name"], "Reception")
        self.assertEqual(result["events"][2]["date"], "December 25, 2023")
        self.assertEqual(result["events"][2]["time"], "7:00 PM onwards")


if __name__ == "__main__":
    unittest.main()
