"""Regression coverage for flattened hackathon poster OCR."""
import unittest

from app.core.understanding.parser import parse_invitation


class HackathonPosterExtractionTests(unittest.TestCase):
    def test_flattened_poster_title_date_venue_and_coordinators(self):
        text = (
            "SNS COLLEGE OF st shs GenAI TECHNOLOGY STREAM IV CS CLUSTER "
            "Powered Design INSTITUTIONS AN AUTONOMOUS INSTITUTION ORGANIZES "
            "Thinking COIMBATORE - 641035 www.snsgroups.com FrameWork "
            "HACKATHON 24 HOURS 2.0 HACKNEXT'26 Innovate Collaborate reate "
            "SERIES The Future! 08 - 09 PRIZE POOL UP TO OCTOBER 2026 IDEAS "
            "TODAY IMPACT TOMORROW 720,000 VENUE PARTICIPATION SNS AI CAMPUS "
            "CERTIFICATE COIMBATORE AI FOR ALL TOP TEAM GETS AN TEAM SIZE "
            "#12 INTERNSHIP 1 - 4 OPPORTUNITY PER TEAM FOOD & BEVERAGES "
            "3 MEALS + 4 REFRESHMENTS IDEATE BUILD CODE TURN YOUR "
            "REGISTRATION FEE 7500 INTO PERSON REAL-WORLD SOLUTIONS "
            "REGISTER NOW SCAN TO REGISTER STUDENT COORDINATORS "
            "FACULTY COORDINATOR A BRIGHTER Gugan KM Akshaya R Mohan Raj. R "
            "Mrs. K. Kalaivani +91 98949 06986 +91 93612 78375 "
            "+91 9025886070 24 HOURS HACKATHON SERIES 2026"
        )

        result = parse_invitation({}, text)

        self.assertEqual(
            result["event_name"], "HACKATHON 24 HOURS 2.0 — HACKNEXT'26"
        )
        self.assertEqual(result["event_type"], "Hackathon")
        self.assertEqual(result["date"], "08–09 October 2026")
        self.assertEqual(result["venue"], "SNS AI CAMPUS")
        self.assertEqual(result["address"], "Coimbatore - 641035")
        self.assertCountEqual(
            [person["name"] for person in result["people"]],
            ["Gugan KM", "Akshaya R", "Mohan Raj. R", "Mrs. K. Kalaivani"],
        )


if __name__ == "__main__":
    unittest.main()
