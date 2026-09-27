"""Language detection for multilingual invitation text.

Supports English, Tamil, Hindi, Malayalam, Telugu and mixed-language detection.
Uses Unicode block ranges to score each script. Falls back to OCR or text heuristics.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Unicode ranges per script
SCRIPT_RANGES = {
    "Tamil": (0x0B80, 0x0BFF),
    "Hindi": (0x0900, 0x097F),  # Devanagari
    "Malayalam": (0x0D00, 0x0D7F),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "English": (0x0041, 0x007A),  # ASCII letters
}

# Sample words that help identify language for fallback
LANG_HINTS = {
    "Tamil": ["திருமண", "அழைப்பு", "வரவேற்பு", "மணமக்கள்", "தேதி", "நேரம்"],
    "Hindi": ["विवाह", "निमंत्रण", "स्वागत", "वर", "वधु", "तिथि"],
    "Malayalam": ["വിവാഹം", "ക്ഷണം", "സ്വാഗതം", "വരൻ", "വധു"],
    "Telugu": ["వివాహం", "ఆహ్వానం", "స్వాగతం", "పెళ్లి"],
    "Kannada": ["ವಿವಾಹ", "ಆಮಂತ್ರಣ", "ಸ್ವಾಗತ", "ಮದುವೆ"],
    "English": ["marriage", "wedding", "invitation", "reception", "bride", "groom"],
}


def detect_script_scores(text: str) -> Dict[str, float]:
    """Count characters belonging to each script's unicode range."""
    scores = {name: 0 for name in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                scores[name] += 1
                break
    return scores


def detect_language(text: str, ocr_lang: str = "") -> str:
    """Detect the primary language of an invitation text.

    Combination of script-range scoring and OCR-reported language.
    """
    if not text.strip():
        return ocr_lang or "English"

    scores = detect_script_scores(text)
    total_letters = sum(scores.values())
    if total_letters == 0:
        return ocr_lang or "English"

    # Normalize to percentages
    percentages = {k: v / total_letters for k, v in scores.items()}

    # Determine dominant script
    dominant = max(percentages, key=percentages.get)

    # Mixed-language detection: if more than one script carries a meaningful
    # share (>= 2%), report a mixed language rather than attributing the text
    # solely to the dominant (usually English) script.
    active_scripts = [name for name, pct in percentages.items() if pct >= 0.02]
    indic_active = any(
        name in ("Tamil", "Hindi", "Malayalam", "Telugu", "Kannada")
        for name in active_scripts
    )
    if len(active_scripts) >= 2 and indic_active:
        # Report the Indian language first for readability.
        indic_scripts = [
            n for n in ("Tamil", "Hindi", "Malayalam", "Telugu", "Kannada")
            if percentages.get(n, 0) >= 0.02
        ]
        return " + ".join(indic_scripts + ["English"])

    if percentages[dominant] < 0.35:
        # Mixed language
        return "Mixed Language"

    # For English fallback, also check hints for Indic scripts
    lowered = text.lower()
    for lang, hints in LANG_HINTS.items():
        if lang == "English":
            continue
        if any(h in text for h in hints):
            if percentages[dominant] < 0.5:
                return lang

    return dominant


def detect_language_from_image(image_path: str, ocr_text: str = "") -> str:
    """Detect language (placeholder for OCR-assisted detection)."""
    return detect_language(ocr_text)

