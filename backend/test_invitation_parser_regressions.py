"""Regression coverage for generic invitation OCR/parser behaviour."""
import unittest

from app.core.understanding.parser import parse_invitation


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


if __name__ == "__main__":
    unittest.main()
