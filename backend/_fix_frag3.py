with open("app/core/understanding/parser.py", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "_TAMIL_NIKAH_FRAGMENTS" in line and "=" in line:
        print(f"Found line {i+1}")
        print(f"Original: {line.rstrip()}")
        # Append the Latin variant "நickersாஹ்" to the tuple
        new_line = line.rstrip()[:-1] + ', "நickersாஹ்")\n'
        print(f"New: {new_line.rstrip()}")
        lines[i] = new_line
        break

with open("app/core/understanding/parser.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
print("Done")
