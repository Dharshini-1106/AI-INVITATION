"""Unit tests for robust bride/groom name extraction across invitation formats.

These tests exercise the generalized, confidence-ranked name extractor
(`extract_names`), which is NOT tuned to any single invitation template. Each
case uses a distinct layout/phrasing to verify generalization.
Run: python test_names.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import extract_names  # noqa: E402


def check(label, text_lines, exp_bride, exp_groom):
    text = "\n".join(text_lines)
    cand = extract_names(text_lines, text, "Wedding")
    parsed = cand.value if cand else {"bride": "", "groom": ""}
    got_bride = parsed.get("bride", "")
    got_groom = parsed.get("groom", "")
    status = "OK" if (got_bride == exp_bride and got_groom == exp_groom) else "FAIL"
    print(f"[{status}] {label}: bride={got_bride!r} (exp {exp_bride!r}), "
          f"groom={got_groom!r} (exp {exp_groom!r})")
    return status == "OK"


def main():
    tests = [
        # Inline names before "Invite you" (the reported failing case)
        ("inline before invite",
         ["TOGETHER WITH THEIRFAMILIES",
          "Samira Richard Invite you to their wedding celebration"],
         "Samira", "Richard"),
        # Separate name lines above marker
        ("separate lines above marker",
         ["Samira", "Richard", "Invite you to their wedding"],
         "Samira", "Richard"),
        # Ampersand groom-first
        ("ampersand groom-first",
         ["Mr. Arjun & Ms. Priya request the honour"],
         "Priya", "Arjun"),
        # Ampersand bride-first (gender hints override)
        ("ampersand bride-first",
         ["Richard & Samira are getting married"],
         "Samira", "Richard"),
        # "with X & Y"
        ("with X & Y",
         ["the wedding of Aravind with Meena & family"],
         "Meena", "Aravind"),
        # "X weds Y"
        ("weds",
         ["Samira weds Richard"],
         "Samira", "Richard"),
        # "marriage of X with Y"
        ("marriage of X with Y",
         ["the marriage of Priya with Arjun"],
         "Priya", "Arjun"),
        # Explicit markers
        ("explicit markers",
         ["Bride: Samira", "Groom: Richard"],
         "Samira", "Richard"),
        # "daughter of ... with son of"
        ("daughter/son",
         ["Asha daughter of Mr. Kumar with Ravi son of Mr. Rao"],
         "Asha", "Ravi"),
        # D/o S/o parentheticals on separate lines (decorative all-caps header)
        ("D/o S/o parenthetical",
         ["WE SOLICITYOURGRACIOUSPRESENCE&",
          "BLESSINGS FORTHE WEDDING OF",
          "Sanya",
          "(D/o Lt. Mrs.Rajnee Chawla & Mr. Satpal Chawla)",
          "Sadashiw",
          "(S/o Mrs. Neeru Chawla & Mr.Rakesh Chawla)",
          "JOIN US FOR WEDDING CEREMONY"],
         "Sanya", "Sadashiw"),
    ]

    passed = 0
    for label, lines, eb, eg in tests:
        if check(label, lines, eb, eg):
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
