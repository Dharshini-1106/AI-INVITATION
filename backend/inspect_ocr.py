"""Inspect actual OCR output to find evidence for bride, groom, venue, date."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Actual OCR text from the backend logs
raw = """ddudate rsi
aioudsrati psi
திருமணை (ுிக்காஹ்) அழைப்பிதழ்
எல்ணாம் வல்ல அல்ணால்றளின் பெரகுளாஜாம், நகிகள் நாயகம் (ுல் அவர்களின்
ni00n 60 im)
கmயிந்றக்கிழமை காலை 11.30 மணக்
@mpL  11.30 L00
காயபமழி முர்ளூம்) அல்ாம் S.s.மூகைதன் தம்பி – ஹாவிமா ஆபிதா
ஆக்யாின் பேத்தியுூம், எு்களின் புளி
A. முஹெம்மது ஆபிதா M.A.,
A. OMUI MA.
மணகளுக்கம்
106001L0@5t
காயாமாழி மு்ும்) றாவி S.அய்துல் காதர் – ஹறொகிமா கம்சா கவி
உடன்கும, பதன ்ாக் ம்முது காசிம் – muா ரூ்யு் நா
ஆகியரின் பேரஜும்
gd@umflr Buygon
ஜனமப் A.சலாகதச் – வனuா S. நகிளா மூஹாகிரா
ஆகியாரின் புல்வன்
BSummoOr1gs616r
S. முஹம்மது முஷ்தாகி் B.E.,
மணகணறு்கம்
106001bt
திருமணம் (டுக்காற்) செய்ய பரியார்களால் நிச்செயித்தவண்ணம், இன்ஷாோ அல்லா்,
பரமண்குறிச்சி அரஸ் திருமண மஹறாலில் நடைபெறும் நிக்காற்விற்கும் அதெனை
தொடர்ந்து நடைபெறும் வலிமா விருந்திலும் கலந்து கொண்டு சிறப்பிக்குமாறு அன்புடன்
அழைக்கின்றோம்.
SgBcor8pmb.
தங்கள் நல்வரவை இனிதே விரும்பும்
S BM B
A.ஆயிpா ஆகலா
A.ud
தங்கள் அன்புள்ள
Ssor orirou
ஜனாப் M.T.முகமது அன்வர் ஹீசனள் B.A. B.L.,
ஜgனாuா A.சிக்தகி வறாeaர்
36r A. DE
6/9A, பள்ளிவாசல் தெரு, கொயாமழி
பன்r : 8884950410, 7200853041
"""

lines = [l for l in raw.split("\n") if l.strip()]

print("=== OCR LINE INSPECTION ===\n")

print("--- Lines with Latin initials (potential names) ---")
for i, line in enumerate(lines):
    if any(line.strip().startswith(x + ".") for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        print(f"Line {i}: {line!r}")

print("\n--- Lines with degree suffixes (M.A., B.E., etc.) ---")
for i, line in enumerate(lines):
    if any(deg in line for deg in ["M.A.", "B.E.", "M.Sc.", "B.Tech.", "Ph.D.", "M.B.B.S.", "B.A.", "B.Com.", "M.Com.", "B.Sc.", "M.C.A.", "B.C.A."]):
        print(f"Line {i}: {line!r}")

print("\n--- Lines containing 'பரம' or 'மஹ' or 'அர' (venue candidates) ---")
for i, line in enumerate(lines):
    if any(kw in line for kw in ["பரம", "மஹ", "அர"]):
        print(f"Line {i}: {line!r}")

print("\n--- Lines containing date-like patterns ---")
for i, line in enumerate(lines):
    import re
    if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", line):
        print(f"Line {i}: {line!r}")
    if re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", line, re.IGNORECASE):
        print(f"Line {i}: {line!r}")

print("\n--- Lines containing 'August' or month names ---")
for i, line in enumerate(lines):
    if any(m in line.lower() for m in ["august", "july", "june", "may", "april", "march", "february", "january", "2026", "2025", "2024"]):
        print(f"Line {i}: {line!r}")

print("\n--- All Tamil lines with strong venue-like tokens ---")
for i, line in enumerate(lines):
    tokens = ["மஹால்", "மஹறால்", "மஹாலில்", "மண்டபம்", "வெளியீடு", "அரசு", "அரஸ்", "நடைபெறும்", "திருமண"]
    if any(t in line for t in tokens):
        print(f"Line {i}: {line!r}")
