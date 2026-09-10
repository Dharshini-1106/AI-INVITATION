import re

with open("app/core/understanding/parser.py", encoding="utf-8") as f:
    content = f.read()

old = '_TAMIL_NIKAH_FRAGMENTS = ("நிகா", "நிகாஹ்", "னிகா", "நிகாஹ", "நickersாஹ்", "நிகஹ்", "நிகஹ")'
new = '_TAMIL_NIKAH_FRAGMENTS = ("நிகா", "நிகாஹ்", "னிகா", "நிகாஹ", "நickersாஹ்", "நிகஹ்", "நிகஹ", "நickersாஹ்")'

if old in content:
    content = content.replace(old, new)
    with open("app/core/understanding/parser.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old string not found")
    # Find the line
    for i, line in enumerate(content.split("\n")):
        if "_TAMIL_NIKAH_FRAGMENTS" in line:
            print(f"Line {i+1}: {line!r}")
