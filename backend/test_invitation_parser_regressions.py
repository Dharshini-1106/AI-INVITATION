"""Regression coverage for generic invitation OCR/parser behaviour."""
import unittest

from app.core.understanding.parser import parse_invitation
from app.core.language import detect_language


def parse(text):
    return parse_invitation({}, text)


class InvitationParserRegressionTests(unittest.TestCase):
    def test_multi_event_dates_are_not_overwritten(self):
        result = parse(
            "Venkatha Reddy + Ananya Gauru\nMahandi & Sangeet\n"
            "21Th December 23, 10:00AM\nHotel Marine Blue, Hyderabad\n"
            "Marriage\n23th December 23\nReception\n25th December 23\n"
            "Dinner 7:00 PM onwards"
        )
        # Couple identified as parents of the event
        self.assertEqual((result["bride_name"], result["groom_name"]),
                         ("Venkatha Reddy", "Ananya Gauru"))
        # 3 independent events
        self.assertEqual(result["number_of_events"], 3)
        self.assertEqual(result["invitation_mode"], "multi")
        # 3 distinct dates - first event date must NOT be overwritten
        self.assertEqual(result["events"][0]["date"], "December 21, 2023")
        self.assertEqual(result["events"][1]["date"], "December 23, 2023")
        self.assertEqual(result["events"][2]["date"], "December 25, 2023")
        # Event-specific venue and location
        self.assertEqual(result["events"][0]["venue"], "Hotel Marine Blue")
        self.assertEqual(result["events"][0]["address"], "Hyderabad")
        # Dinner information on the correct event (last one)
        self.assertEqual(result["events"][2]["additional_information"],
                         "Dinner 7:00 PM onwards")
        # First two events should not have dinner info
        self.assertEqual(result["events"][0]["additional_information"], "")
        self.assertEqual(result["events"][1]["additional_information"], "")
        # People identified
        people_names = {p["name"] for p in result["people"]}
        self.assertTrue(people_names.issuperset({"Venkatha Reddy", "Ananya Gauru"}))

    def test_unlabelled_parent_pair_and_secondary_dinner(self):
        result = parse(
            "Saylee & Vaibhav\nJanuary 01 2024\n12:00 AM\n"
            "Taj Hotel, Bangalore\nDinner 7 PM onwards\nRSVP 123-456-7890"
        )
        self.assertEqual(result["date"], "January 1, 2024")
        self.assertEqual(result["time"], "12:00 AM")
        self.assertEqual((result["venue"], result["address"]), ("Taj Hotel", "Bangalore"))
        self.assertEqual(result["contact_number"], "123-456-7890")
        self.assertEqual(result["additional_information"], "Dinner 7 PM onwards")
        self.assertEqual({person["name"] for person in result["people"]}, {"Saylee", "Vaibhav"})
        # No event heading is printed, so the parser must not invent one.
        self.assertEqual(result["event_type"], "")
        # Baby-shower parents-to-be identified as people
        roles = {p["name"]: p["role"] for p in result["people"]}
        self.assertIn("Saylee", roles)
        self.assertIn("Vaibhav", roles)

    def test_yearless_date_and_missing_venue_remain_uninvented(self):
        result = parse("Anna Smith\nApril 12\n4 PM\n567 Your City NY 55767\nRSVP information")
        self.assertEqual(result["date"], "April 12")
        self.assertEqual(result["time"], "4:00 PM")
        self.assertEqual(result["venue"], "")
        self.assertEqual(result["address"], "567 Your City, NY 55767")
        self.assertEqual(result["event_type"], "")
        # Person identified in people array
        self.assertTrue(any(p["name"] == "Anna Smith" for p in result["people"]))
        # Year must NOT be invented
        self.assertNotIn("20", result["date"])
        self.assertNotIn("2024", result["date"])

    def test_birthday_age_printed_weekday_and_full_street_address(self):
        result = parse(
            "Noah's 30th Birthday\nSunday, November 1st, 2030\nAt 8 PM\n"
            "108 N Platinum Ave, Deming, NY 88300"
        )
        self.assertEqual(result["event_type"], "Birthday")
        self.assertEqual(result["birthday_age"], "30")
        self.assertEqual(result["printed_weekday"], "Sunday")
        self.assertEqual(result["date"], "November 1, 2030")
        self.assertEqual(result["address"], "108 N Platinum Ave, Deming, NY 88300")
        # Printed weekday preserved (not calculated)
        self.assertEqual(result["printed_weekday"], "Sunday")
        # Person identified
        self.assertTrue(any(p["name"] == "Noah" for p in result["people"]))

    def test_pair_location_and_onwards_time_without_an_event_heading(self):
        result = parse(
            "Bhoomika + Prashant\nSunday, June 08 2025\n10:30 AM onwards\n"
            "GK Hill View Resort\nKarnataka"
        )
        self.assertEqual(result["date"], "June 8, 2025")
        self.assertEqual(result["time"], "10:30 AM onwards")
        self.assertEqual((result["venue"], result["address"]),
                         ("GK Hill View Resort", "Karnataka"))
        self.assertEqual(result["printed_weekday"], "Sunday")
        self.assertEqual({person["name"] for person in result["people"]},
                         {"Bhoomika", "Prashant"})
        self.assertEqual(result["event_type"], "")

    def test_ordinal_dates_and_two_digit_years(self):
        # Ordinal suffixes (st, nd, th) and 2-digit years
        result = parse(
            "Wedding\n15Th March 24\nGrand Hall\nChennai"
        )
        self.assertEqual(result["date"], "March 15, 2024")
        self.assertEqual(result["events"][0]["date"], "March 15, 2024")

    def test_save_the_date_decoration_is_not_groom_name(self):
        result = parse(
            "Cordially invite you to grace the wedding ceremony\n"
            "Ms. Nivetha Ramesh\nSave\nSave the Date\n"
            "Sivasami Maaligai Marriage Hall\n"
            "29 September 2026\n10.30 AM\n"
            "Alangulam Road, Mukkudal, Tirunelveli - 627 758"
        )
        self.assertEqual(result["bride_name"], "Nivetha Ramesh")
        self.assertEqual(result["groom_name"], "")
        self.assertEqual(result["people"], [
            {"name": "Nivetha Ramesh", "role": "Bride"},
        ])

    def test_truncated_family_fragment_is_not_bride_name(self):
        result = parse(
            "Cordially invite you to grace the wedding ceremony\n"
            "Dr. Karthik Srinivasan\nMs. Nivetha Ramesh\n"
            "Srinivasan Family\nFamil\n"
            "29 September 2026\n10:30 AM\nSivasami Maaligai Marriage Hall"
        )
        self.assertEqual(result["bride_name"], "Nivetha Ramesh")
        self.assertEqual(result["groom_name"], "Dr. Karthik Srinivasan")
        self.assertEqual(result["people"], [
            {"name": "Nivetha Ramesh", "role": "Bride"},
            {"name": "Dr. Karthik Srinivasan", "role": "Groom"},
        ])

    def test_son_with_daughter_wedding_layout(self):
        result = parse(
            "Two Hearts One Life\nWEDDING\n"
            "Son of Karthik with Nivetha Daughter of\nFriday\n"
            "Mahalakshmi Mahal AC\n10.00 AM - 11.00 AM\n"
            "02 October 2026\nPASUVANTHANAI,\n(Muhurtham)\n"
            "Kovilpatti Main Rd, Kovilpatti, Tamil Nadu 628501"
        )
        self.assertEqual(result["bride_name"], "Nivetha")
        self.assertEqual(result["groom_name"], "Karthik")
        self.assertEqual(
            result["address"],
            "PASUVANTHANAI, Kovilpatti Main Rd, Kovilpatti, Tamil Nadu 628501",
        )

    def test_ocr_time_error_oo_to_00(self):
        # Letter O/o in time should be corrected to digit 0
        result = parse(
            "Couple A & B\nJanuary 01 2024\n1O:OO AM onwards\n"
            "Hotel\nCity\nRSVP 123-456-7890"
        )
        self.assertEqual(result["time"], "10:00 AM onwards")

    def test_venue_prefix_ocr_corruption(self):
        # "Venu:" prefix (OCR corruption of "Venue:") should be stripped
        result = parse(
            "Couple A & B\nJanuary 01 2024\n10:00 AM\n"
            "Venu: Green Palace\nChennai"
        )
        self.assertEqual(result["venue"], "Green Palace")

    def test_separate_line_to_range_birthday_invitation(self):
        result = parse(
            "A little princess\nis turning a year older...\nJoin us to celebrate\n"
            "her special day!\nYOU ARE INVITED TO\nAbhi Sri's\nBirthday Celebration\n"
            "Y\nOur little star is turning a year older!\nLittle Girl\n"
            "Please join us for a day filled with love,\nBig Dreams\nfun and happiness.\n"
            "2\n0\nC\nReva Plaza\n6.00 PM\nFriday\n436, Main Road,\n"
            "to 9.00 PM\n02 October 2026\nKovilpatti,\nTamil Nadu - 628502\n"
            "one\nYour presence will make our celebration\nmore special!\n"
            "See You There!\nEAT\nPLAY\nCELEBRATE\nREPEAT\nTiny Feet\n"
            "Big Happiness\nA Brighter Tomorrow"
        )
        self.assertEqual(result["event_name"], "Birthday Celebration")
        self.assertEqual(result["event_type"], "Birthday")
        self.assertEqual(result["date"], "October 2, 2026")
        self.assertEqual(result["time"], "6:00 PM")
        self.assertEqual(result["end_time"], "9:00 PM")
        self.assertEqual(result["printed_weekday"], "Friday")
        self.assertEqual(result["venue"], "Reva Plaza")
        self.assertEqual(
            result["address"],
            "436, Main Road, Kovilpatti, Tamil Nadu - 628502",
        )
        self.assertEqual(result["people"], [{"name": "Abhi Sri", "role": "Celebrant"}])
        self.assertEqual(result["events"][0]["end_time"], "9:00 PM")

    def test_dinner_and_reception_as_additional_info(self):
        result = parse(
            "Couple A & B\nSaturday, 10 March 2025\n6:00 PM\n"
            "Grand Hall\nMumbai\n"
            "Reception\n11th March 2025\n7:30 PM onwards\n"
            "Dinner 8 PM onwards"
        )
        # Dinner info should be on the event that contains it
        dinner_found = any(
            "Dinner" in ev.get("additional_information", "")
            for ev in result["events"]
        )
        self.assertTrue(dinner_found)

    def test_conference_identity_and_host_institution_win_over_partner_text(self):
        result = parse(
            "SATHYABAMA INSTITUTE OF SCIENCE AND TECHNOLOGY\n"
            "DEPARTMENT OF INFORMATION TECHNOLOGY\n"
            "In association with CENTRE FOR MOLECULAR AND NANOMEDICAL SCIENCES\n"
            "CHENNAI, INDIA\n"
            "CONFERENCE ON ADVANCES IN BIOSCIENCES, DATA SCIENCE AND AI\n"
            "ICABDAI 2026\n16 - 18 NOVEMBER 2026\nEmerging Technologies"
        )
        self.assertEqual(result["people"], [])
        self.assertEqual(result["event_name"], "ICABDAI 2026")
        self.assertEqual(result["event_type"], "Conference")
        self.assertEqual(result["venue"],
                         "SATHYABAMA INSTITUTE OF SCIENCE AND TECHNOLOGY")
        self.assertEqual(result["address"], "Chennai, India")

    def test_marriage_prose_selects_names_and_excludes_date_from_name(self):
        result = parse(
            "திருமண அழைப்பிதழ்\n"
            "the auspicious occasion of the marriage of\n"
            "Karthik\nand\nNivetha\n"
            "October 23, 2026 Friday 10:00 AM - 11:30 AM\n"
            "Sri Ram Mahal\n"
            "Anaiyur Kulamangalam Main Rd, Kosakulam, Madurai, "
            "Tamil Nadu 625017, India"
        )
        self.assertEqual(result["people"], [
            {"name": "Nivetha", "role": "Bride"},
            {"name": "Karthik", "role": "Groom"},
        ])
        self.assertEqual(result["event_type"], "Wedding")
        self.assertEqual(result["date"], "October 23, 2026")
        self.assertEqual(result["venue"], "Sri Ram Mahal")
        self.assertEqual(result["address"],
                         "Anaiyur Kulamangalam Main Rd, Kosakulam, Madurai, "
                         "Tamil Nadu 625017, India")

    def test_wedding_name_line_split_and_heading_preserved(self):
        result = parse(
            "We are delighted to invite you to celebrate the\n"
            "Wedding of\nKarthik  Sowmiya\n\n"
            "Marriage\nReception\nFriday\n02 October 2026\n"
            "10.00 AM - 12.00 Noon\nThursday\n01 October 2026\n"
            "7.00 PM - 9.00 PM\nSnow Hall,\n"
            "Salt Pans, Tuticorin Beach Road,\nTamil Nadu."
        )
        self.assertEqual(result["people"], [
            {"name": "Karthik", "role": "Person"},
            {"name": "Sowmiya", "role": "Person"},
        ])
        self.assertEqual((result["bride_name"], result["groom_name"]), ("", ""))
        self.assertEqual(result["event_name"], "Marriage")
        self.assertEqual(result["events"][0]["date"], "October 2, 2026")
        self.assertEqual(result["events"][0]["printed_weekday"], "Friday")
        self.assertEqual(result["events"][1]["date"], "October 1, 2026")
        self.assertEqual(result["events"][1]["printed_weekday"], "Thursday")

    def test_shared_schedule_date_and_location_inherit_across_events(self):
        result = parse(
            "ANNUAL DAY\nFriday, 16 October 2026\n"
            "10:00 AM - 1:00 PM\nTHE AMERICAN COLLEGE\n"
            "Tallakulam, Madurai - 625002, Tamil Nadu, India\n"
            "SPORTS DAY\n2:00 PM - 5:30 PM"
        )
        self.assertEqual(result["number_of_events"], 2)
        annual, sports = result["events"]
        for event in (annual, sports):
            self.assertEqual(event["date"], "October 16, 2026")
            self.assertEqual(event["printed_weekday"], "Friday")
            self.assertEqual(event["venue"], "THE AMERICAN COLLEGE")
            self.assertEqual(event["address"],
                             "Tallakulam, Madurai - 625002, Tamil Nadu, India")
        self.assertEqual((annual["time"], annual["end_time"]),
                         ("10:00 AM", "1:00 PM"))
        self.assertEqual((sports["time"], sports["end_time"]),
                         ("2:00 PM", "5:30 PM"))


class TamilMixedOcrRegressionTests(unittest.TestCase):
    """Tamil-English mixed OCR must not promote religious fragments to people,
    and must preserve full addresses, names, and language detection."""

    TAMIL_WEDDING_OCR = """OODTTTD
Two
Families
One Heart
Lifelong
Happiness
ீவிநாயகாய நம:II
TRADITION
3
CULTURE
LOVE
திருமண
SL600T
அழைப்பிதழ்
FAMILY
FOREVER
WEDDING
INVITATION
E
We cordially invite you to grace
A NEW
BEGINNING
A BEAUTIFUL
Vignesh
JOURNEY
Better
ogether
Together
8
SON OF
Always
2
Mr. R. Selvaraj & Mrs. L. Meena
Keerthana
DAUGHTER OF
Mr. S. Ramesh & Mrs. P. Kavitha
Together in love, today and always
V
Friday
10:00 AM - 11:30 AM
23 October 2026
Muhurtham
from
With Love
Venue
Sri Nithya Mahal
O
Our Families
Pandi Kovil Ring Rd,
Madurai - 625020,
Madurai
Tamil Nadu, India.
Your gracious presence will make
and Blessings
GOOD VIBES
GOOD PEOPLE
BRIGHTER TOMORROWS
SID6OT
WEDDING INVITATION
invite We cordially you to grace
the auspicious occasion of the r marriage of A BI
ther Mr. R. Selvaraj & Mrs. L. Meena Vignesh L SON OF
ways"""

    def test_tamil_religious_phrase_not_a_person(self):
        from app.core.matching.field_matcher import _looks_like_person_name
        # Tamil wedding/blessing fragments must never be treated as names.
        for fragment in ("ருமண", "மீ கணபதியே நம: I1", "திருமண அழைப்பிதழ்"):
            self.assertFalse(_looks_like_person_name(fragment),
                             msg=f"{fragment!r} should not look like a person")

    def test_mixed_tamil_english_language_detection(self):
        text = "திருமண அழைப்பிதழ் Wedding Invitation Tamil Nadu கணபதியே"
        # Even though English dominates, Tamil script is present -> mixed.
        self.assertIn("Tamil", detect_language(text))
        self.assertIn("English", detect_language(text))

    def test_address_noise_phrases_stripped(self):
        from app.core.understanding.parser import _normalize_address_text
        address = ("Thirupparankundram GST Road, Poonga Bus Stop, Madurai 625005, "
                   "Tamil Nadu, India, LET'S CELEBRATE, With Blessings from, "
                   "Our Families")
        result = _normalize_address_text(address)
        self.assertNotIn("CELEBRATE", result)
        self.assertNotIn("Blessings", result)
        self.assertNotIn("Our Families", result)
        self.assertIn("Madurai 625005", result)
        self.assertIn("Tamil Nadu, India", result)

    def test_tamil_wedding_full_extraction_via_layout_lines(self):
        result = parse(self.TAMIL_WEDDING_OCR)
        # Tamil religious phrase must not leak into the people fields.
        for field in ("bride_name", "groom_name"):
            self.assertNotIn("நம", result.get(field, ""), msg=field)
        # Names recovered from Son/Daughter-of context.
        people_names = {p["name"] for p in result["people"]}
        self.assertIn("Keerthana", people_names)
        self.assertIn("Vignesh", people_names)
        # Full address preserved.
        self.assertIn("Pandi Kovil Ring Rd", result["address"])
        self.assertIn("625020", result["address"])
        self.assertIn("Tamil Nadu, India", result["address"])
        # No decorative text in the address.
        self.assertNotIn("Blessings", result["address"])


if __name__ == "__main__":
    unittest.main()
