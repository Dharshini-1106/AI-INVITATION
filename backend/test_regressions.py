"""Regression tests for parser robustness fixes applied on real invitations.

Run: python test_regressions.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import (  # noqa: E402
    parse_invitation,
    extract_venue_variants,
    extract_address_variants,
    _venue_from_line,
)


def check(label, raw, expected, fields=("event_type", "bride_name",
                                        "groom_name", "date", "time",
                                        "venue", "address",
                                        "contact_number")):
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


def check_scalar(label, fn, expected):
    got = fn()
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] {label}: {got!r} (exp {expected!r})")
    return status == "OK"


def main():
    passed = 0
    total = 0

# FIX 1: bare "Venue" label with no value must not become a venue.
    r = _venue_from_line("Venue")
    passed += check_scalar("_venue_from_line bare 'Venue'", (lambda: r), "")
    total += 1
    r = _venue_from_line("Venue:")
    passed += check_scalar("_venue_from_line bare 'Venue:'", (lambda: r), "")
    total += 1

    # FIX 2: "at <time>" must not be captured as a venue label.
    r = _venue_from_line("at 5 PM")
    passed += check_scalar("_venue_from_line 'at 5 PM'", (lambda: r), "")
    total += 1

    # FIX 3: "at <event description>" must not be captured as a venue label.
    r = _venue_from_line("at the marriage reception of")
    passed += check_scalar("_venue_from_line 'at the marriage reception of'",
                           (lambda: r), "")
    total += 1

    # FIX 4: ordinary venue line still works.
    r = _venue_from_line("Venue: Green Palace Convention Hall")
    passed += check_scalar("_venue_from_line labeled venue",
                           (lambda: r), "Green Palace Convention Hall")
    total += 1

    # FIX 5: venue labels using a dash separator should also work.
    r = _venue_from_line("Venue - Green Palace Convention Hall")
    passed += check_scalar("_venue_from_line labeled venue with dash",
                           (lambda: r), "Green Palace Convention Hall")
    total += 1

    # FIX 6: real-world invitation where "at" is followed by a genuine venue.
    total += 1
    passed += check(
        "reception with at-venue + street",
        ("Mr. and Mrs. Sharma\n"
         "request the honour of your presence\n"
         "at the marriage reception of\n"
         "Priya & Arjun\n"
         "on Saturday, 15th March 2025\n"
         "at 7:00 PM\n"
         "at The Grand Ballroom\n"
         "12 Lake View Road\n"
         "Pune\n"
         "RSVP: +91 99887 76655"),
        {"event_type": "Reception", "bride_name": "Priya",
         "groom_name": "Arjun", "date": "March 15, 2025",
         "time": "7:00 PM", "venue": "The Grand Ballroom",
         "address": "12 Lake View Road, Pune", "contact_number": "+91 99887 76655"})

    # FIX 6: bare street + city (venue should NOT swallow the city address).
    total += 1
    passed += check(
        "venue label split from address",
        ("Wedding Invitation\n"
         "Kavya & Rohan\n"
         "request the pleasure of your company\n"
         "Sunday, 5th May 2024\n"
         "Venue: The Grand Pavilion\n"
         "12 Lake View Road\n"
         "Hyderabad, TG 500081\n"
         "Contact: 9000012345"),
        {"event_type": "Wedding", "bride_name": "Kavya",
         "groom_name": "Rohan", "date": "May 5, 2024",
         "venue": "The Grand Pavilion", "address": "12 Lake View Road, Hyderabad, TG 500081",
         "contact_number": "9000012345"})

    # FIX 7: housewarming where venue has street embedded and city separate.
    total += 1
    passed += check(
        "housewarming venue+street, address city",
        ("You are cordially invited to the housewarming\n"
         "on Saturday, 14 December 2024 at 11:00 AM\n"
         "at Sunrise Towers, 88 Green Park\n"
         "Hyderabad, TG 500081\n"
         "Contact: 9000012345"),
        {"event_type": "Housewarming",
         "date": "December 14, 2024", "time": "11:00 AM",
         "venue": "Sunrise Towers, 88 Green Park",
         "address": "Hyderabad, TG 500081",
         "contact_number": "9000012345"})

    total += 1
    passed += check(
        "engagement celebration split day/month lines",
        ("Together with our families\n"
         "We extend a warm invitation to join the\n"
         "Engagement Celebration of\n"
         "Bhoomika\n"
         "D/o Smt. Anjannamma & Sri. B Venkateshaiah\n"
         "With\n"
         "Prashant\n"
         "S/o Smt. B K Manjula & Sri. G M Giri Kumar\n"
         "08th\n"
         "Sunday June 2025\n"
         "10:30 AM Onwards\n"
         "Venue\n"
         "GK Hill View Resort\n"
         "Karnataka\n"
         "Contact: 8412835496"),
        {"event_type": "Engagement", "bride_name": "Bhoomika",
         "groom_name": "Prashant", "date": "June 8, 2025",
         "time": "10:30 AM onwards", "venue": "GK Hill View Resort",
         "address": "Karnataka", "contact_number": "8412835496"})

    print(f"\n{passed}/{total} regression checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

