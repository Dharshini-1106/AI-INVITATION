"""Comprehensive parser tests across many invitation layouts/formats.

Each test is a raw OCR text blob (lines joined by newline) with expected values.
Run: python test_invitations.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import parse_invitation  # noqa: E402


def check(label, raw, expected, fields=("event_type", "bride_name", "groom_name",
                                        "date", "time", "venue", "address",
                                        "contact_number")):
    # New parser signature: parse_invitation(matched_fields, raw_text, layout).
    # Here we pass empty matched fields so the generic rule-based fallback does
    # the extraction (validating the fallback path end-to-end).
    r = parse_invitation({}, raw)
    fails = []
    for f in fields:
        got = r.get(f, "")
        exp = expected.get(f, "")
        if got != exp:
            fails.append(f"{f}={got!r} (exp {exp!r})")
    status = "OK" if not fails else "FAIL"
    print(f"[{status}] {label}")
    for fl in fails:
        print(f"        {fl}")
    return status == "OK"


def main():
    tests = [
        # 1. Labeled modern invitation (the _t.py sample)
        ("labeled modern",
         ("TOGETHER WITH THEIRFAMILIES\n"
          "Samira Richard Invite you to their wedding celebration\n"
          "Wed, Feb 25th, 2024\n"
          "123 Anywhere St., At 9am\n"
          "Any City, ST 12345\n"
          "Reception to follow\n"
          "Contact: 9876543210"),
         {"event_type": "Wedding", "bride_name": "Samira", "groom_name": "Richard",
          "date": "February 25, 2024", "time": "9:00 AM",
          "venue": "", "address": "Any City, ST 12345",
          "contact_number": "9876543210"}),

        # 2. Traditional Indian wedding with explicit labels
        ("traditional labeled",
         ("Mr. Arjun Kumar & Ms. Priya Sharma\n"
          "request the pleasure of your company\n"
          "on the auspicious occasion of their wedding\n"
          "Date: February 25th, 2024\n"
          "Time: 6:30 PM\n"
          "Venue: Green Palace Convention Hall\n"
          "Address: 123 Anywhere Street, Chennai - 600001\n"
          "Contact: 9876543210"),
         {"event_type": "Wedding", "bride_name": "Priya Sharma",
          "groom_name": "Arjun Kumar", "date": "February 25, 2024",
          "time": "6:30 PM", "venue": "Green Palace Convention Hall",
          "address": "123 Anywhere Street, Chennai - 600001",
          "contact_number": "9876543210"}),

        # 3. Reception invitation
        ("reception",
         ("Mr. and Mrs. Sharma\n"
          "request the honour of your presence\n"
          "at the marriage reception of\n"
          "Priya & Arjun\n"
          "on Saturday, 15th March 2025\n"
          "at 7:00 PM\n"
          "Grand Ballroom, Taj Hotel\n"
          "MG Road, Bengaluru\n"
          "RSVP: +91 98765 43210"),
         {"event_type": "Reception", "bride_name": "Priya",
          "groom_name": "Arjun", "date": "March 15, 2025",
          "time": "7:00 PM", "venue": "Grand Ballroom, Taj Hotel",
          "address": "MG Road, Bengaluru",
          "contact_number": "+91 98765 43210"}),

        # 4. Single-line dense wedding
        ("dense single line",
         ("TOGETHER WITH THEIR FAMILIES Samira Richard Invite you to their "
          "wedding celebration Wed, Feb 25th, 2024 at 9 am 123 Anywhere St. "
          "Any City, ST 12345 Reception to follow Contact 9876543210"),
         {"event_type": "Wedding", "bride_name": "Samira", "groom_name": "Richard",
          "date": "February 25, 2024", "time": "9:00 AM",
          "venue": "", "address": "Any City, ST 12345",
          "contact_number": "9876543210"}),

        # 5. Engagement invitation
        ("engagement",
         ("We joyfully announce the engagement of\n"
          "Meera & Rahul\n"
          "on Sunday, 10th November 2024\n"
          "Venue: The Lily Pool, Sea View Garden\n"
          "Contact: 9876501234"),
         {"event_type": "Engagement", "bride_name": "Meera",
          "groom_name": "Rahul", "date": "November 10, 2024",
          "venue": "The Lily Pool, Sea View Garden",
          "contact_number": "9876501234"}),

        # 6. Birthday invitation
        ("birthday",
         ("Join us to celebrate\n"
          "Aarav's 5th Birthday\n"
          "on 12/05/2024 at 4:30 pm\n"
          "Fun Zone, 456 Park Road\n"
          "Greenville\n"
          "RSVP: 9988776655"),
         {"event_type": "Birthday", "date": "May 12, 2024", "time": "4:30 PM",
          "venue": "",
          "address": "456 Park Road, Greenville",
          "contact_number": "9988776655"}),

        # 7. Multi-event: wedding + reception
        ("multi-event wedding+reception",
         ("Wedding Celebration\n"
          "Priya & Arjun\n"
          "on 15th March 2025 at 6:00 PM\n"
          "Green Palace Hall, 123 Main Road\n"
          "Chennai\n"
          "\n"
          "Reception\n"
          "on 16th March 2025 at 7:00 PM\n"
          "Grand Ballroom, Taj Hotel\n"
          "MG Road, Bengaluru"),
         {"event_type": "Wedding Celebration", "bride_name": "Priya",
          "groom_name": "Arjun", "date": "March 15, 2025",
          "time": "6:00 PM", "venue": "Green Palace Hall, 123 Main Road",
          "address": "123 Main Road, Chennai"}),

        # 8. D/o (daughter of) / S/o (son of) invitation with parentheticals
        #    and period-separated date + resort venue + all-caps decorative header.
        ("D/o S/o parenthetical",
         ("WE SOLICITYOURGRACIOUSPRESENCE&\n"
          "BLESSINGS FORTHE WEDDING OF\n"
          "Sanya\n"
          "(D/o Lt. Mrs.Rajnee Chawla & Mr. Satpal Chawla)\n"
          "Sadashiw\n"
          "(S/o Mrs. Neeru Chawla & Mr.Rakesh Chawla)\n"
          "JOIN US FOR\n"
          "WEDDING CEREMONY AT\n"
          "MAPLE. SUKHMANI RESORT\n"
          "APRIL 17.2024\n"
          "Wednesday\n"
          "R.S.V.P\n"
          "Chawla's&Dua's"),
         {"event_type": "Wedding", "bride_name": "Sanya",
          "groom_name": "Sadashiw", "date": "April 17, 2024",
          "venue": "MAPLE. SUKHMANI RESORT", "address": ""}),

        # 9. Unseen layout: "with the blessings of ... the marriage of X with Y"
        #    (verb pattern, phone after a plus, no explicit venue label).
        ("blessings marriage of",
         ("WITH THE BLESSINGS OF OUR PARENTS\n"
          "we announce the marriage of\n"
          "Kavya with Rohan\n"
          "on Sunday, the 5th of May 2024\n"
          "at 6:30 in the evening\n"
          "The Grand Pavilion\n"
          "12 Lake View Road\n"
          "Pune\n"
          "RSVP +91 99887 76655"),
         {"event_type": "Marriage", "bride_name": "Kavya",
          "groom_name": "Rohan", "date": "May 5, 2024",
          "time": "6:30", "venue": "The Grand Pavilion",
          "address": "12 Lake View Road, Pune", "contact_number": "+91 99887 76655"}),

        # 10. Unseen layout: housewarming with a single headline, no names.
        ("housewarming",
         ("You are cordially invited to the housewarming\n"
          "of Mr. & Mrs. Naidu\n"
          "on Saturday, 14 December 2024 at 11:00 AM\n"
          "at Sunrise Towers, 88 Green Park\n"
          "Hyderabad, TG 500081\n"
          "Contact: 9000012345"),
         {"event_type": "Housewarming", "date": "December 14, 2024",
          "time": "11:00 AM",
          "venue": "Sunrise Towers, 88 Green Park",
          "address": "Hyderabad, TG 500081",
          "contact_number": "9000012345"}),

        # 11. Unseen layout: engagement with "&" joined names and a distinct
        #     venue/address split (street in venue, city+zip in address).
        ("engagement street/city split",
         ("We seek your blessings\n"
          "on the occasion of the engagement of\n"
          "Ananya & Vikram\n"
          "on Friday, 22 March 2024\n"
          "at 5 PM\n"
          "The Rose Garden\n"
          "45 MG Road\n"
          "Bengaluru, KA 560001\n"
          "Ph: 9870065432"),
         {"event_type": "Engagement", "bride_name": "Ananya",
          "groom_name": "Vikram", "date": "March 22, 2024",
          "time": "5:00 PM", "venue": "The Rose Garden",
          "address": "45 MG Road, Bengaluru, KA 560001",
          "contact_number": "9870065432"}),

        # 12. Unseen layout: standalone "at <place>" line, numeric date, and a
        #     bare 10-digit contact with no label.
        ("at phrase numeric date",
         ("You are requested to attend\n"
          "the wedding reception of\n"
          "Ishita & Advait\n"
          "Date: 07/11/2024\n"
          "Time: 7:30 pm\n"
          "at The Imperial Ballroom\n"
          "Kolkata\n"
          "9123456789"),
         {"event_type": "Reception", "bride_name": "Ishita",
          "groom_name": "Advait", "date": "November 7, 2024",
          "time": "7:30 PM", "venue": "The Imperial Ballroom",
          "address": "Kolkata", "contact_number": "9123456789"}),

        # 13. Tamil-style OCR (English transliteration) — multiday wedding with
        #     a "muhurtham" and a venue that contains a street suffix.
        ("tamil-style transliterated",
         ("திருமண அழைப்பிதழ்\n"
          "Arjun & Meera\n"
          "Muhurtham at 9:00 AM\n"
          "on 02/06/2024\n"
          "Hindu Temple Hall\n"
          "3 North Car Street\n"
          "Madurai 625001\n"
          "Contact 9840012345"),
         {"event_type": "Wedding", "bride_name": "Meera",
          "groom_name": "Arjun", "time": "9:00 AM",
          "date": "June 2, 2024", "venue": "Hindu Temple Hall",
          "address": "3 North Car St, Madurai 625001", "contact_number": "9840012345"}),
    ]

    passed = 0
    for label, raw, exp in tests:
        if check(label, raw, exp):
            passed += 1
    print(f"\n{passed}/{len(tests)} invitation tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())