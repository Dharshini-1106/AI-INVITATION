"""Regression coverage for Tamil wedding OCR participant role recovery."""
import unittest

from app.core.understanding.parser import parse_invitation


OCR_TEXT = """OODTTTD Two Families One Heart Lifelong Happiness ீவிநாயகாய நம:II
TRADITION 3 CULTURE LOVE திருமண SL600T அழைப்பிதழ் FAMILY FOREVER WEDDING INVITATION E
We cordially invite you to grace A NEW BEGINNING A BEAUTIFUL Vignesh JOURNEY Better
ogether Together 8 SON OF Always 2 Mr. R. Selvaraj & Mrs. L. Meena Keerthana
DAUGHTER OF Mr. S. Ramesh & Mrs. P. Kavitha Together in love, today and always V
Friday 10:00 AM - 11:30 AM 23 October 2026 Muhurtham from With Love Venue Sri
Nithya Mahal O Our Families Pandi Kovil Ring Rd, Madurai - 625020, Madurai Tamil
Nadu, India. Your gracious presence will make and Blessings GOOD VIBES GOOD PEOPLE
BRIGHTER TOMORROWS SID6OT WEDDING INVITATION invite We cordially you to grace the
auspicious occasion of the r marriage of A BI ther Mr. R. Selvaraj & Mrs. L. Meena
Vignesh L SON OF ways"""


class ParticipantExtractionTests(unittest.TestCase):
    def test_parent_role_lines_override_invocation_and_slogan_ocr(self):
        result = parse_invitation({}, OCR_TEXT)

        self.assertEqual(result["bride_name"], "Keerthana")
        self.assertEqual(result["groom_name"], "Vignesh")
        self.assertEqual(
            result["people"],
            [
                {"name": "Vignesh", "role": "Groom"},
                {"name": "Keerthana", "role": "Bride"},
            ],
        )
        self.assertEqual(result["events"][0]["bride_name"], "Keerthana")
        self.assertEqual(result["events"][0]["groom_name"], "Vignesh")

    def test_relationship_extraction_is_not_tied_to_specific_names(self):
        text = OCR_TEXT.replace("Vignesh", "Arun").replace("Keerthana", "Megha")

        result = parse_invitation({}, text)

        self.assertEqual(result["bride_name"], "Megha")
        self.assertEqual(result["groom_name"], "Arun")

    def test_marriage_of_names_separated_by_slogans_and_address(self):
        text = """ருமண I ேய யந் நம: மீ கணபதியே நம: I1 860 0 6
WEDDING INVITATION
We cordially invite you to grace the auspicious occasion of the marriage of
V BETTER TOGETHER arthik A BRIGHTER TOMORROW Karthik TRADITION LOVE RESPECT
N TOGETHERNESS Nivetha FOREVER Together in love, today and always
Friday, 10:00 AM - 11:30 AM
23 October 2026
Muhurtham
Venue
CSK Mahal
Thirupparankundram GST Road, Poonga Bus Stop, Madurai 625005, Tamil Nadu, India."""

        result = parse_invitation({}, text)

        self.assertEqual(result["groom_name"], "Karthik")
        self.assertEqual(result["bride_name"], "Nivetha")
        self.assertEqual(
            result["people"],
            [
                {"name": "Karthik", "role": "Groom"},
                {"name": "Nivetha", "role": "Bride"},
            ],
        )
        self.assertIn("Poonga Bus Stop", result["address"])

    def test_high_confidence_marriage_regex_rejects_slogan_words(self):
        text = """WEDDING INVITATION
We cordially invite you to grace the auspicious occasion of the marriage of
PEOPLE BEGINNING A BEAUTIFUL Karthik BLESSINGS JOURNEY BRIGHT TOGETHER FUTURE
Nivetha Together in love, today and always
Friday, 10:00 AM - 11:30 AM
23 October 2026 (Muhurtham)
Venue
Sri Ram Mahal
Anaiyur Kulamangalam Main Rd, TRADITION, Kosakulam, Madurai, CULTURE, LOVE,
Tamil Nadu 625017, India."""

        result = parse_invitation({}, text)

        self.assertEqual(result["groom_name"], "Karthik")
        self.assertEqual(result["bride_name"], "Nivetha")
        self.assertEqual(
            result["people"],
            [
                {"name": "Karthik", "role": "Groom"},
                {"name": "Nivetha", "role": "Bride"},
            ],
        )
        self.assertIn("Kosakulam", result["address"])
        self.assertNotRegex(result["address"], r"(?i)tradition|culture|love")

    def test_flat_ocr_recovers_couple_and_full_address_over_truncated_hint(self):
        text = (
            "WEDDING INVITATION Two Families One Heart We cordially invite you "
            "to grace the auspicious occasion of the marriage of V BETTER "
            "TOGETHER arthik A BRIGHTER TOMORROW Karthik TRADITION LOVE "
            "RESPECT N TOGETHERNESS Nivetha FOREVER Together in love, today "
            "and always Friday, 10:00 AM - 11:30 AM 23 October 2026 Muhurtham "
            "o Venue CSK Mahal Thirupparankundram GST Road, Poonga Bus Stop, "
            "Madurai 625005, Tamil Nadu, India. LET'S CELEBRATE"
        )
        result = parse_invitation(
            {"address": "Thirupparankundram GST Road, Bus Stop, Madurai 625005, Tamil Nadu, India"},
            text,
        )

        self.assertEqual(result["groom_name"], "Karthik")
        self.assertEqual(result["bride_name"], "Nivetha")
        self.assertIn("Poonga Bus Stop", result["address"])
        self.assertNotIn("WEDDING INVITATION", result["address"])

    def test_address_cleanup_keeps_locality_split_before_bus_stop(self):
        from app.core.understanding.parser import (
            _restore_ocr_bus_stop_qualifier,
            _strip_address_decoratives,
        )

        cleaned = _strip_address_decoratives(
            "Thirupparankundram GST Road, Poonga, Bus Stop, "
            "Madurai 625005, Tamil Nadu, India"
        )

        self.assertIn("Poonga Bus Stop", cleaned)

        source = (
            "Thirupparankundram GST Road, Poonga Bus Stop, Madurai 625005, "
            "Tamil Nadu, India"
        )
        shortened = (
            "Thirupparankundram GST Road Bus Stop, Madurai 625005, "
            "Tamil Nadu, India"
        )
        restored = _restore_ocr_bus_stop_qualifier(shortened, source)
        self.assertEqual(
            restored,
            "Thirupparankundram GST Road, Poonga Bus Stop, Madurai 625005, Tamil Nadu, India",
        )

    def test_explicit_son_with_daughter_relationship_beats_invitation_pronoun(self):
        text = (
            "Together with your blessings we joyfully invite you to the "
            "WEDDING Son of Karthik with Nivetha Daughter of We request "
            "the pleasure of your presence."
        )
        result = parse_invitation({}, text)

        self.assertEqual(result["groom_name"], "Karthik")
        self.assertEqual(result["bride_name"], "Nivetha")
        self.assertEqual(
            result["people"],
            [
                {"name": "Karthik", "role": "Groom"},
                {"name": "Nivetha", "role": "Bride"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
