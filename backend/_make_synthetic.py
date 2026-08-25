"""Create proper synthetic Tamil image with Nirmala font."""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BACKEND = Path(r"D:/MINI/backend")
font_path = "C:/Windows/Fonts/Nirmala.ttf"

# Verify font works
try:
    font = ImageFont.truetype(font_path, 64)
    print("Nirmala font loaded successfully")
except Exception as e:
    print(f"Font load failed: {e}")
    sys.exit(1)

# Create image with Tamil text
img = Image.new("RGB", (600, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
text = "திருமணம்\nஅழைப்பிதழ்\nவரவேற்பு"
draw.text((40, 40), text, fill=(0, 0, 0), font=font)

synth_path = BACKEND / "_tamil_synthetic.png"
img.save(str(synth_path))
print(f"Saved: {synth_path}")
print(f"Size: {img.size}")

# Verify text is actually in the image by trying to read it back
# (PIL can't read text, but we can check the image has content)
import numpy as np
arr = np.array(img)
print(f"Image array shape: {arr.shape}")
print(f"Non-white pixels: {np.sum(arr < 250)}")
