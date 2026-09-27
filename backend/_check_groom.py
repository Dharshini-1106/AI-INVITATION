"""Trace how 'Save' is extracted as groom name."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.entity import extract_entities
from app.core.matching import match_entities_to_fields
from app.core.postprocess.correction import correct_text

raw_text = """With the blessings of
"May two souls
walk together in dharma,
our beloved elders
|| Om Ganeshaya Namah ||
love and happiness always'
se
Cordially invite you to grace the wedding ceremony
of
Better
0
Together
M.B.B.S.,
Two
Families
Forever
with
8
E
Journey
Ms. Nivetha Ramesh
One Beautiful
B.E. (CSE),
We request the pleasure of your presence
and blessings on this auspicious occasion
9
Sivasami Maaligai
Tuesday
10.30 AM
Marriage Hall,
29 September 2026
to 12.00 PM
Alangulam Road, Mukkudal,
(Purattasi 13, Tuesday)
(Muhoortham)
Tirunelveli - 627 758
Tamil Nadu, India"""

text_lines = [l for l in raw_text.split('\n') if l.strip()]

# Step 1: Correct the text
corrected = correct_text(raw_text)
print("=== Corrected text ===")
for line in corrected.split('\n'):
    print(f"  {line!r}")

corrected_lines = [l for l in corrected.split('\n') if l.strip()]
print()

# Step 2: Extract entities
entities = extract_entities(corrected, corrected_lines, [])
print("=== All entities ===")
for e in entities:
    print(f"  type={e.etype} text={e.text!r} value={e.value!r} conf={e.confidence} strategy={e.strategy}")

print()

# Step 3: Match to fields
matched = match_entities_to_fields(entities)
print("=== Matched fields ===")
for k, v in matched.items():
    print(f"  {k}: {v!r}")
