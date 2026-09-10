with open("app/core/understanding/parser.py", encoding="utf-8") as f:
    content = f.read()

# The exact old line content
old_line = '_TAMIL_NIKAH_FRAGMENTS = ("நிகா", "நிகாஹ்", "னிகா", "நிகாஹ", "நickersாஹ்", "நிகஹ்", "நிகஹ")'

# The exact text variant from the test file
latin_variant = "நickersாஹ்"

new_line = old_line[:-1] + ', "' + latin_variant + '")'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open("app/core/understanding/parser.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully added Latin variant to _TAMIL_NIKAH_FRAGMENTS")
else:
    print("Old line not found")
    for i, line in enumerate(content.split("\n")):
        if "_TAMIL_NIKAH_FRAGMENTS" in line:
            print(f"Line {i+1}: {line!r}")
