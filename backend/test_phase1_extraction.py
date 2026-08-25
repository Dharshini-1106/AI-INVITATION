"""Acceptance checks for generalized Phase 1 structured extraction."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import parse_invitation  # noqa: E402


def check(label, raw, expected):
    result = parse_invitation({}, raw)
    fields = (
        "event_name", "event_type", "bride_name", "groom_name", "date",
        "time", "venue", "address", "contact_number", "number_of_events",
    )
    failures = [
        f"{field}={result.get(field)!r} (expected {expected[field]!r})"
        for field in fields
        if result.get(field) != expected[field]
    ]
    if failures:
        print(f"[FAIL] {label}: " + "; ".join(failures))
        return False
    print(f"[OK] {label}")
    return True


def main():
    cases = [
        (
            "standalone conjunction wedding",
            "TOGETHER WITH THEIR FAMILIES\n"
            "Asha\n&\nDev\nInvite you to their wedding celebration\n"
            "Wed, Feb 25th, 2024\nAt 9 am\n"
            "123 Anywhere St.,\nWed, Feb 25th, 2024\nAt 9 am\nReception to follow",
            {
                "event_name": "Wedding Celebration", "event_type": "Wedding",
                "bride_name": "Asha", "groom_name": "Dev",
                "date": "February 25, 2024", "time": "9:00 AM",
                "venue": "", "address": "123 Anywhere St.",
                "contact_number": "", "number_of_events": 1,
            },
        ),
        (
            "engagement with dot time",
            "We announce the engagement of\nMira\n&\nKiran\n"
            "25 June 2025\n10.30 a.m. onwards\n"
            "GK Hill View Resort\nKarnataka",
            {
                "event_name": "Engagement", "event_type": "Engagement",
                "bride_name": "Mira", "groom_name": "Kiran",
                "date": "June 25, 2025", "time": "10:30 AM onwards",
                "venue": "GK Hill View Resort", "address": "Karnataka",
                "contact_number": "", "number_of_events": 1,
            },
        ),
        (
            "birthday does not invent couple roles",
            "Join us for a birthday\n8 June 2025\nEvening 6 PM\n"
            "Sunset Park\n45 Lake Road",
            {
                "event_name": "Birthday", "event_type": "Birthday",
                "bride_name": "", "groom_name": "",
                "date": "June 8, 2025", "time": "6:00 PM",
                "venue": "", "address": "45 Lake Road", "contact_number": "",
                "number_of_events": 1,
            },
        ),
        (
            "explicit field labels",
            "Event: Annual Conference\nDate: 25/02/2024\nTime: 06:30 PM\n"
            "Venue: Civic Convention Hall\nAddress: Main Road, Metro City\n"
            "Contact No: +91 98765 43210",
            {
                "event_name": "Annual Conference", "event_type": "Conference",
                "bride_name": "", "groom_name": "",
                "date": "February 25, 2024", "time": "6:30 PM",
                "venue": "Civic Convention Hall", "address": "Main Road, Metro City",
                "contact_number": "+91 98765 43210", "number_of_events": 1,
            },
        ),
        (
            "at venue with city suffix",
            "Reception\n08 June 2025\nAt Sri Lakshmi Mahal, Chennai\n"
            "RSVP: 9876543210",
            {
                "event_name": "Reception", "event_type": "Reception",
                "bride_name": "", "groom_name": "",
                "date": "June 8, 2025", "time": "",
                "venue": "Sri Lakshmi Mahal", "address": "Chennai",
                "contact_number": "9876543210", "number_of_events": 1,
            },
        ),
        (
            "weds excludes parent names",
            "Together with our families\nAhmed Khan\nweds\nAyesha Rahman\n"
            "Son of Mr. Rashid Khan & Mrs. Farzana Khan\n"
            "13 February 2027\n5:00 PM\nRoyal Orchid Banquet\nMG Road, Hyderabad",
            {
                "event_name": "", "event_type": "",
                "bride_name": "Ayesha Rahman", "groom_name": "Ahmed Khan",
                "date": "February 13, 2027", "time": "5:00 PM",
                "venue": "Royal Orchid Banquet", "address": "MG Road, Hyderabad",
                "contact_number": "", "number_of_events": 1,
            },
        ),
    ]
    passed = sum(check(label, raw, expected) for label, raw, expected in cases)
    multi = parse_invitation({},
        "Ahmed Khan\nSon of Mr. Rashid Khan & Mrs. Farzana Khan\nWeds\n"
        "Ayesha Rahman\nDaughter of Mr. Shafiqul Rahman & Mrs. Najma Rahman\n"
        "Nikah Ceremony\n15 February 2027\n5:00 PM onwards\n"
        "Venue: Rahman Residence,\nBanjara Hills, Hyderabad\n"
        "Haldi Ceremony\n13 February 2027\n5:00 PM onwards\n"
        "Walima Reception\n16 February 2027\n7:30 PM onwards\n"
        "Venue: Royal Orchid Banquet,\nMG Road, Hyderabad")
    multi_ok = (
        multi["number_of_events"] == 3
        and multi["bride_name"] == "Ayesha Rahman"
        and multi["groom_name"] == "Ahmed Khan"
        and [event["event_name"] for event in multi["events"]]
            == ["Nikah Ceremony", "Haldi Ceremony", "Walima Reception"]
        and multi["events"][0]["address"] == "Banjara Hills, Hyderabad"
        and multi["events"][2]["address"] == "MG Road, Hyderabad"
    )
    print("[{}] multi-event role and association".format("OK" if multi_ok else "FAIL"))
    passed += int(multi_ok)
    total = len(cases) + 1
    print(f"{passed}/{total} Phase 1 extraction checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
