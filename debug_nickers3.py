import sys
sys.path.insert(0, r'D:\MINI\backend')

# Get the raw text for tamil_nikah test
text = ("கஸ்லாதலிர் ரஹ்மானிர் ரஹிம்\n"
        "திருமண (நickersாஹ்) அழைப்பிதழ்\n")

print("Text bytes:")
print(text.encode('utf-8'))
print()
print("'nickers' bytes:", "nickers".encode())
print()
print("Full text repr:", repr(text))
print()
# Search for n-i-c-k-e-r-s as individual chars
for i, ch in enumerate(text):
    if ch == 'n':
        print(f"Found 'n' at index {i}: {repr(text[i:i+10])}")