import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.core.understanding.parser import (parse_invitation, _split_events,
                                           extract_venue_variants,
                                           extract_names_variants)

raw = """TOGETHER WITH OUR FAMILIES
ENGAGEMENT CELEBRATION OF
Bhoomika
(D/o Smt. Anjammamma & Sri. B Venkateshaiah)
WITH
Prashant
(S/o Smt. B K Manjula & Sri. G M Girikumar)
SUNDAY
08 JUNE 2025
10:30 AM onwards
GK Hill View Resort
Karnataka
Contact: 8412835496
"""
lines = [l for l in raw.split("\n") if l.strip()]
print("SPLIT EVENTS:", _split_events(lines))
print("\nVENUE VARIANTS:")
for c in extract_venue_variants(lines, raw):
    print("  ", c.strategy, repr(c.value), c.confidence)
print("\nNAME VARIANTS:")
for c in extract_names_variants(lines, raw):
    print("  ", c.strategy, c.confidence, c.value)
print("\nPARSED:")
r = parse_invitation({}, raw)
for k in ("event_name", "event_type", "bride_name", "groom_name", "date",
          "time", "venue", "address", "contact_number"):
    print(f"  {k}: {r.get(k)!r}")
print("  number_of_events:", r.get("number_of_events"))

