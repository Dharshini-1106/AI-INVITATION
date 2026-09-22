import sys
sys.path.insert(0, r'D:\MINI\backend')

# Read the actual test file
with open(r'D:\MINI\backend\test_tamil_invitations.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the nikah test text
idx = content.find('tamil_nikah')
if idx >= 0:
    # Find the text between the parentheses
    start = content.find('(', idx)
    end = content.find('),', start)
    if start >= 0 and end >= 0:
        test_text = content[start+1:end]
        print("First 500 chars of test text:")
        print(test_text[:500])
        print("\n---")
        print("Bytes of first 200 chars:")
        print(test_text[:200].encode())
        
        # Search for nickers-like patterns
        import re
        for m in re.finditer(r'nicker', test_text, re.IGNORECASE):
            print(f"\nFound 'nicker' at {m.start()}: {repr(test_text[m.start()-10:m.end()+10])}")
            print(f"  Bytes: {test_text[m.start()-10:m.end()+10].encode()}")