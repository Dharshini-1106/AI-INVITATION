"""Quick regression test for the specific invitation that had wrong Bride/Groom.

Bride: Shoomika  (appears above D/o line)
Groom: Prashanth (appears above S/o line, OCR'd as 'rashant')
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import extract_names, extract_family_roles


def main():
    # The exact OCR text lines from the reported failing invitation
    lines = [
        "Togetherwith ourfamihies",
        "Weextendawarminvitationto jonthe",
        "Ongagemont Colebration of",
        "Shoomika",
        "D/oSmtAnjinammmat",
        "Sri B Venkateshaiah",
        "rashant",
        "So SmBKManjula&",
        "Sri GMGin Kumar",
        "Sunday",
        "08th",
        "June",
        "2025",
        "10:30AMOnwards",
        "Venue",
        "GKHillViewResort",
        "DesignedByPrashan4urs9re8412835496",
        "Karnataka",
    ]
    text = "\n".join(lines)

    print("=== Family roles strategy ===")
    roles = extract_family_roles(lines)
    print(f"  bride={roles.get('bride', '')!r}, groom={roles.get('groom', '')!r}")

    print("\n=== extract_names (full pipeline) ===")
    cand = extract_names(lines, text, "Wedding")
    parsed = cand.value if cand else {"bride": "", "groom": ""}
    bride = parsed.get("bride", "")
    groom = parsed.get("groom", "")
    print(f"  bride={bride!r}, groom={groom!r}")
    print(f"  strategy={cand.strategy if cand else 'none'}")
    print(f"  confidence={cand.confidence if cand else 0}")

    bride_ok = "shoomika" in bride.lower()
    groom_ok = "prashanth" in groom.lower() or "rashant" in groom.lower()
    if bride_ok and groom_ok:
        print("\n✓ PASS: Bride and groom correctly extracted!")
    else:
        print("\n✗ FAIL:")
        if not bride_ok:
            print(f"  Expected bride containing 'shoomika', got {bride!r}")
        if not groom_ok:
            print(f"  Expected groom containing 'prashanth'/'rashant', got {groom!r}")

    return 0 if (bride_ok and groom_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

