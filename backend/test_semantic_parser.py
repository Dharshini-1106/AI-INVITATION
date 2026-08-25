"""Focused parser coverage for relationship-aware OCR and event association."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import extract_date, format_date, parse_invitation  # noqa: E402


def parse(text):
    return parse_invitation({}, text)


class SemanticParserTests(unittest.TestCase):
    def test_normal_wedding_invitation(self):
        result = parse("Wedding\nMira Patel & Rohan Shah\n12 March 2030\nVenue: Lotus Hall")
        self.assertEqual(result["bride_name"], "Mira Patel")
        self.assertEqual(result["groom_name"], "Rohan Shah")
        self.assertEqual(result["date"], "March 12, 2030")

    def test_wedding_couple_ampersand_and_street_locality_mapping(self):
        result = parse("Samira & Richard\n"
                       "Invite you to their wedding celebration\n"
                       "Wed, Feb 25th, 2024\nAt 9 am\n"
                       "123 Anywhere St.,\nAny City, ST 12345\n"
                       "Reception to follow")
        self.assertEqual(result["event_name"], "Wedding Celebration")
        self.assertEqual(result["event_type"], "Wedding")
        self.assertEqual((result["bride_name"], result["groom_name"]),
                         ("Samira", "Richard"))
        self.assertEqual((result["date"], result["time"]),
                         ("February 25, 2024", "9:00 AM"))
        self.assertEqual((result["venue"], result["address"]),
                         ("123 Anywhere St.", "Any City, ST 12345"))

    def test_merged_time_and_duplicate_name_do_not_displace_couple_or_location(self):
        result = parse("TOGETHERWITH\nTHEIR FAMILIES\nSamira\nRichard\n"
                       "Invite you\nto their wedding celebration\n"
                       "123 Anywhere St.,\nWed Feb 25th, 2024\nAt9am\n"
                       "Any City,ST12345\nReception to follow\nRichaid\n"
                       "Reception toe follow")
        self.assertEqual(result["event_name"], "Wedding Celebration")
        self.assertEqual(result["event_type"], "Wedding")
        self.assertEqual((result["bride_name"], result["groom_name"]),
                         ("Samira", "Richard"))
        self.assertEqual((result["date"], result["time"]),
                         ("February 25, 2024", "9:00 AM"))
        self.assertEqual((result["venue"], result["address"]),
                         ("123 Anywhere St.", "Any City, ST 12345"))

    def test_wedding_couple_on_separate_lines(self):
        result = parse("Wedding\nSamira\n&\nRichard")
        self.assertEqual((result["bride_name"], result["groom_name"]),
                         ("Samira", "Richard"))

    def test_explicit_bride_and_groom_labels_take_priority(self):
        result = parse("Wedding\nBride: Samira\nGroom: Richard")
        self.assertEqual((result["bride_name"], result["groom_name"]),
                         ("Samira", "Richard"))

    def test_city_state_zip_does_not_invent_a_venue(self):
        result = parse("Wedding\nAny City, ST 12345")
        self.assertEqual(result["venue"], "")
        self.assertEqual(result["address"], "Any City, ST 12345")

    def test_weds_and_parent_context_excludes_parents(self):
        result = parse("Dev Mehta\nSon of Mr Kiran Mehta & Mrs Leela Mehta\nWeds\n"
                       "Nila Iyer\nDaughter of Mr Raman Iyer & Mrs Tara Iyer")
        self.assertEqual((result["groom_name"], result["bride_name"]),
                         ("Dev Mehta", "Nila Iyer"))

    def test_multiple_events_keep_independent_details(self):
        result = parse("Arun Das\nWeds\nMaya Sen\nPROGRAMME\n"
                       "Sangeet Ceremony\nFriday,13thFebruary2027\n5:00 PM onwards\n"
                       "Venue: Cedar House\nHill Road, Pune\n"
                       "Reception\nSunday,15thFobruary2027\n7:30 PM onwards\n"
                       "Venue: River Banquet\nPark Road, Mumbai")
        self.assertEqual(result["number_of_events"], 2)
        first, second = result["events"]
        self.assertEqual((first["date"], first["venue"]), ("February 13, 2027", "Cedar House"))
        self.assertEqual((second["date"], second["venue"]), ("February 15, 2027", "River Banquet"))
        self.assertEqual((second["bride_name"], second["groom_name"]), ("Maya Sen", "Arun Das"))

    def test_repeated_ocr_name_correction_is_relationship_scoped(self):
        result = parse("Abmeo Khan\nSon of Mr Rashid Khan & Mrs Farzana Khan\nWeds\n"
                       "Ayesha Rahman\nDaughter of Mr Shafiqul Rahman & Mrs Najma Rahman\n"
                       "Ahmeo Khan\nNikah Ceremony\nFriday,13thFebruary2027")
        self.assertEqual(result["groom_name"], "Ahmed Khan")
        self.assertEqual(result["bride_name"], "Ayesha Rahman")

    def test_dates_with_missing_spaces_and_numeric_forms(self):
        for raw, expected in [("Friday.13thFebruary2027", "February 13, 2027"),
                              ("Sunday,15thFobruary2027", "February 15, 2027"),
                              ("13/02/2027", "February 13, 2027"),
                              ("13-02-2027", "February 13, 2027")]:
            candidate = extract_date(raw)
            self.assertIsNotNone(candidate)
            self.assertEqual(format_date(candidate.value), expected)

    def test_tamil_english_mixed_invitation(self):
        result = parse("திருமண அழைப்பிதழ்\nKavya & Vignesh\nSunday, 2nd June 2029\n"
                       "Venue: Ananda Mahal\nChennai")
        self.assertEqual(result["date"], "June 2, 2029")
        self.assertEqual(result["venue"], "Ananda Mahal")

    def test_decorative_font_style_and_missing_fields(self):
        result = parse("NIKAH CEREMONY\nZara Ali\nWeds\nImran Noor\nMonday,16thFebuary2027")
        self.assertEqual(result["date"], "February 16, 2027")
        self.assertEqual(result["time"], "")
        self.assertEqual(result["venue"], "")

    def test_completely_different_invitation(self):
        result = parse("Housewarming Invitation\nSaturday, 9th October 2032\n"
                       "Time: 11 AM\nVenue: Maple Residency\nAddress: 40 Lake Lane, Kochi")
        self.assertEqual(result["event_type"], "Housewarming")
        self.assertEqual(result["date"], "October 9, 2032")
        self.assertEqual(result["venue"], "Maple Residency")


if __name__ == "__main__":
    unittest.main()
