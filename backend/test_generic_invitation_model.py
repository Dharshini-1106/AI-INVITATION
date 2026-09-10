"""Tests for the new generic people[] and invitation_mode fields.

Run: python test_generic_invitation_model.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import parse_invitation


def check(label, raw, checks):
    r = parse_invitation({}, raw)
    fails = []
    for path, expected in checks:
        parts = path.split(".")
        got = r
        for p in parts:
            if isinstance(got, dict):
                got = got.get(p, "")
            elif isinstance(got, list):
                try:
                    got = got[int(p)]
                except (IndexError, ValueError):
                    got = ""
            else:
                got = getattr(got, p, "")
        if got != expected:
            fails.append(f"{path}={got!r} (exp {expected!r})")
    status = "OK" if not fails else "FAIL"
    print(f"[{status}] {label}")
    for fl in fails:
        print(f"        {fl}")
    if status == "FAIL":
        print(f"    full result: {r}")
    return status == "OK"


def main():
    passed = 0
    total = 0

    # 1. Single wedding: people should be Bride + Groom
    total += 1
    passed += check(
        "single wedding -> people=Bride/Groom, mode=single",
        ("Wedding Celebration\n"
         "Priya & Arjun\n"
         "on 15th March 2025 at 6:00 PM\n"
         "Green Palace Hall\n"
         "Chennai\n"
         "Contact: 9000012345"),
        [
            ["invitation_mode", "single"],
            ["people.0.name", "Priya"],
            ["people.0.role", "Bride"],
            ["people.1.name", "Arjun"],
            ["people.1.role", "Groom"],
            ["bride_name", "Priya"],
            ["groom_name", "Arjun"],
            ["number_of_events", 1],
        ],
    )

    # 2. Multi-event wedding: people at top, multiple events
    total += 1
    passed += check(
        "multi-event wedding -> people shared, mode=multi",
        ("Venkatha Reddy & Ananya Gauru\n"
         "\n"
         "Mehndi Ceremony\n"
         "21 December 2023\n"
         "10:00 AM\n"
         "Hotel Marine Blue, Hyderabad\n"
         "\n"
         "Marriage\n"
         "23 December 2023\n"
         "\n"
         "Reception\n"
         "25 December 2023\n"
         "7:00 PM\n"
         "Hotel Marine Blue, Hyderabad"),
        [
            ["invitation_mode", "multi"],
            ["number_of_events", 3],
            ["people.0.name", "Venkatha Reddy"],
            ["people.0.role", "Bride"],
            ["people.1.name", "Ananya Gauru"],
            ["people.1.role", "Groom"],
            ["events.0.event_name", "Mehndi Ceremony"],
            ["events.0.date", "December 21, 2023"],
            ["events.0.time", "10:00 AM"],
            ["events.1.event_name", "Marriage"],
            ["events.1.date", "December 23, 2023"],
            ["events.2.event_name", "Reception"],
            ["events.2.date", "December 25, 2023"],
            ["events.2.time", "7:00 PM"],
        ],
    )

    # 3. Non-wedding event with two names -> generic Person roles
    total += 1
    passed += check(
        "anniversary -> generic people roles",
        ("Celebrating the 25th Anniversary of\n"
         "Rahul & Priya\n"
         "on Saturday, 10th May 2025\n"
         "at 7:00 PM\n"
         "The Grand Ballroom\n"
         "Mumbai\n"
         "Contact: 9876543210"),
        [
            ["invitation_mode", "single"],
            ["people.0.role", "Person"],
            ["people.1.role", "Person"],
            ["event_type", "Anniversary"],
        ],
    )

    # 4. Birthday with one name -> generic Person role
    total += 1
    passed += check(
        "birthday with name -> generic Person role",
        ("Join us to celebrate\n"
         "Rahul's 5th Birthday\n"
         "on 12/05/2024 at 4:30 pm\n"
         "Fun Zone, 456 Park Road\n"
         "Greenville\n"
         "RSVP: 9988776655"),
        [
            ["invitation_mode", "single"],
            ["event_type", "Birthday"],
            ["date", "May 12, 2024"],
            ["time", "4:30 PM"],
            ["venue", ""],
            ["address", "456 Park Road, Greenville"],
            ["contact_number", "9988776655"],
        ],
    )

    # 5. Engagement -> Bride/Groom roles
    total += 1
    passed += check(
        "engagement -> people Bride/Groom",
        ("We joyfully announce the engagement of\n"
         "Meera & Rahul\n"
         "on Sunday, 10th November 2024\n"
         "Venue: The Lily Pool\n"
         "Contact: 9876501234"),
        [
            ["invitation_mode", "single"],
            ["people.0.name", "Meera"],
            ["people.0.role", "Bride"],
            ["people.1.name", "Rahul"],
            ["people.1.role", "Groom"],
            ["event_type", "Engagement"],
        ],
    )

    # 6. Housewarming -> no people extracted (current behavior preserved)
    total += 1
    passed += check(
        "housewarming -> no people, backward compat preserved",
        ("You are cordially invited to the housewarming\n"
         "of Mr. & Mrs. Naidu\n"
         "on Saturday, 14 December 2024 at 11:00 AM\n"
         "at Sunrise Towers, 88 Green Park\n"
         "Hyderabad, TG 500081\n"
         "Contact: 9000012345"),
        [
            ["invitation_mode", "single"],
            ["people", []],
            ["event_type", "Housewarming"],
            ["date", "December 14, 2024"],
            ["time", "11:00 AM"],
            ["venue", "Sunrise Towers, 88 Green Park"],
            ["address", "Hyderabad, TG 500081"],
            ["contact_number", "9000012345"],
        ],
    )

    # 7. Multi-event with mixed event types
    total += 1
    passed += check(
        "multi-event with engagement + reception",
        ("Engagement of\n"
         "Kavya & Rohan\n"
         "on 10th March 2024 at 5:00 PM\n"
         "The Rose Garden\n"
         "\n"
         "Reception\n"
         "on 12th March 2024 at 7:00 PM\n"
         "Grand Ballroom, Taj Hotel\n"
         "MG Road, Bengaluru"),
        [
            ["invitation_mode", "multi"],
            ["number_of_events", 2],
            ["people.0.name", "Kavya"],
            ["people.0.role", "Bride"],
            ["people.1.name", "Rohan"],
            ["people.1.role", "Groom"],
            ["events.0.event_type", "Engagement"],
            ["events.1.event_type", "Reception"],
        ],
    )

    # 8. Single event with no names -> empty people
    total += 1
    passed += check(
        "seminar -> empty people (no person names in text)",
        ("Annual Tech Seminar\n"
         "on 20th December 2024 at 10:00 AM\n"
         "Convention Center\n"
         "Bangalore\n"
         "Contact: 9000012345"),
        [
            ["invitation_mode", "single"],
            ["people", []],
            ["event_type", "Seminar"],
            ["date", "December 20, 2024"],
            ["time", "10:00 AM"],
        ],
    )

    # 9. Multi-event wedding: shared venue is propagated to all events
    total += 1
    passed += check(
        "multi-event wedding -> shared venue propagated",
        ("Venkatha Reddy & Ananya Gauru\n"
         "\n"
         "Mehndi Ceremony\n"
         "21 December 2023\n"
         "10:00 AM\n"
         "Hotel Marine Blue, Hyderabad\n"
         "\n"
         "Marriage\n"
         "23 December 2023\n"
         "\n"
         "Reception\n"
         "25 December 2023\n"
         "7:00 PM\n"
         "Hotel Marine Blue, Hyderabad"),
        [
            ["invitation_mode", "multi"],
            ["number_of_events", 3],
            ["events.0.date", "December 21, 2023"],
            ["events.0.time", "10:00 AM"],
            ["events.0.venue", "Hotel Marine Blue"],
            ["events.1.date", "December 23, 2023"],
            ["events.1.time", ""],
            ["events.1.venue", "Hotel Marine Blue"],
            ["events.2.date", "December 25, 2023"],
            ["events.2.time", "7:00 PM"],
            ["events.2.venue", "Hotel Marine Blue"],
        ],
    )

    # 10. matched_fields must not overwrite per-event fields in multi-event
    total += 1
    r10 = parse_invitation(
        {"date": "23 December 2023", "time": "5:00 PM"},
        ("Mehndi Ceremony\n"
         "10:00 AM\n"
         "Hotel Marine Blue, Hyderabad\n"
         "\n"
         "Marriage\n"
         "23 December 2023\n"
         "\n"
         "Reception\n"
         "25 December 2023\n"
         "7:00 PM\n"
         "Hotel Marine Blue, Hyderabad"),
    )
    fails10 = []
    if r10.get("events", [])[0].get("date") != "":
        fails10.append(f"events.0.date={r10['events'][0].get('date')!r} (exp '')")
    if r10.get("events", [])[0].get("time") != "10:00 AM":
        fails10.append(f"events.0.time={r10['events'][0].get('time')!r} (exp '10:00 AM')")
    if r10.get("events", [])[1].get("date") != "December 23, 2023":
        fails10.append(f"events.1.date={r10['events'][1].get('date')!r} (exp 'December 23, 2023')")
    if r10.get("events", [])[2].get("date") != "December 25, 2023":
        fails10.append(f"events.2.date={r10['events'][2].get('date')!r} (exp 'December 25, 2023')")
    status10 = "OK" if not fails10 else "FAIL"
    print(f"[{status10}] matched_fields does not leak into per-event fields")
    for fl in fails10:
        print(f"        {fl}")
    if status10 == "FAIL":
        print(f"    full result: {r10}")
    passed += status10 == "OK"

    # 11. Tamil multi-event invitation (headings in English so splitter works)
    total += 1
    passed += check(
        "tamil multi-event -> shared venue propagated",
        ("ராஜேஷ் & பிரியங்கா திருமணம்\n"
         "\n"
         "Mehndi\n"
         "December 21, 2023\n"
         "10:00 AM\n"
         "Hotel Marine Blue, Hyderabad\n"
         "\n"
         "Marriage\n"
         "December 23, 2023\n"
         "\n"
         "Reception\n"
         "December 25, 2023\n"
         "7:00 PM\n"
         "Hotel Marine Blue, Hyderabad"),
        [
            ["invitation_mode", "multi"],
            ["number_of_events", 3],
            ["events.0.date", "December 21, 2023"],
            ["events.0.time", "10:00 AM"],
            ["events.0.venue", "Hotel Marine Blue"],
            ["events.1.date", "December 23, 2023"],
            ["events.1.time", ""],
            ["events.1.venue", "Hotel Marine Blue"],
            ["events.2.date", "December 25, 2023"],
            ["events.2.time", "7:00 PM"],
            ["events.2.venue", "Hotel Marine Blue"],
        ],
    )

    print(f"\n{passed}/{total} generic model tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())