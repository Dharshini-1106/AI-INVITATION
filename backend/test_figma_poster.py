"""Test for Figma poster extraction (non-couple event)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import parse_invitation
from app.core.entity.entity_ner import extract_entities
from app.core.matching.field_matcher import match_entities_to_fields


def check(label, text, exp_event_name, exp_event_type):
    text_lines = [l for l in text.split("\n") if l.strip()]
    corrected = text
    entities = extract_entities(corrected, text_lines)
    matched = match_entities_to_fields(entities)
    parsed = parse_invitation(matched, corrected)
    en = parsed.get("event_name", "")
    et = parsed.get("event_type", "")
    people = parsed.get("people", [])
    en_ok = exp_event_name.lower() in en.lower()
    et_ok = any(kw in et.lower() for kw in ["workshop", "training", "seminar"])
    no_false_people = all(
        p.get("name", "").lower() not in {"figma", "learn", "gain", "organizes"}
        for p in people
    )
    status = "OK" if (en_ok and et_ok and no_false_people) else "FAIL"
    print(f"[{status}] {label}")
    if not en_ok:
        print(f"  event_name: got {en!r}, expected containing {exp_event_name!r}")
    if not et_ok:
        print(f"  event_type: got {et!r}, expected containing {exp_event_type!r}")
    if not no_false_people:
        print(f"  people: {people}")
    return status == "OK"


def main():
    raw = """KEC KONGU ENGINEERING COLLEGE
ORGANIZES
FIGMA FOR UI/UX DESIGN:
FROM BASICS TO PROTOTYPING
KONGU ENGINEERING COLLEGE 9:00 AM to 4:00 PM
Friday
2026
23 OCTOBER
Perundurai Railway Station Road
Thoppupalayam,
Perundurai, Erode - 638 060,
Tamil Nadu, India"""

    passed = 0
    if check("Figma poster", raw, "Figma for UI/UX Design", "Workshop"):
        passed += 1

    print("\n[extra checks]")
    p = parse_invitation({}, raw)
    extra_ok = (
        p.get("date") == "October 23, 2026"
        and p.get("printed_weekday") == "Friday"
        and p.get("address") == (
            "Perundurai Railway Station Road, Thoppupalayam, "
            "Perundurai, Erode - 638 060, Tamil Nadu, India"
        )
        and p.get("venue") == "KEC KONGU ENGINEERING COLLEGE"
        and p.get("time") == "9:00 AM"
        and p.get("end_time") == "4:00 PM"
        and p.get("people", []) == []
    )
    print(f"  [{'OK' if extra_ok else 'FAIL'}] date/address/venue/time/weekday")
    if extra_ok:
        passed += 1

    print(f"\n{passed}/{2} tests passed")
    return 0 if passed == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
