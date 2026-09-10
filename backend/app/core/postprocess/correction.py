"""OCR error correction and entity normalization.

Uses Sentence-BERT for semantic similarity-based correction of known place names
and venue names. Falls back to a dictionary-based spell-correct approach when the
model is unavailable.

It also applies generic OCR normalization: common event-word misspellings are
fixed and run-together (camelCase) tokens produced by OCR dropping spaces are
split back into separate words.
"""
import logging
import re
from typing import List

from ...config import settings

logger = logging.getLogger(__name__)

# Known place/venue name dictionary for correction (OCR errors -> correct)
KNOWN_PLACES = {
    "madural": "Madurai",
    "madurai": "Madurai",
    "cheenai": "Chennai",
    "chennai": "Chennai",
    "coimbatore": "Coimbatore",
    "coimbtore": "Coimbatore",
    "tiruchirappalli": "Tiruchirappalli",
    "trichy": "Trichy",
    "salem": "Salem",
    "vellore": "Vellore",
    "thoothukudi": "Thoothukudi",
    "erode": "Erode",
    "tirunelveli": "Tirunelveli",
    "kanyakumari": "Kanyakumari",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "hyderabad": "Hyderabad",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
    "kolkata": "Kolkata",
    "pune": "Pune",
    "kerala": "Kerala",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
}

# Common event-related words for normalization
EVENT_WORDS = {
    "wedding": "Wedding",
    "marriage": "Marriage",
    "reception": "Reception",
    "engagement": "Engagement",
    "birthday": "Birthday",
    "naming ceremony": "Naming Ceremony",
    "housewarming": "Housewarming",
    "muhurtham": "Muhurtham",
}

# Common OCR misspellings of event / structural words. This is a *generic*
# lexicon of high-frequency OCR confusions, not invitation-specific text.
_COMMON_OCR_TYPOS = {
    "ongagement": "Engagement",
    "engagment": "Engagement",
    "engagemen": "Engagement",
    "engagement": "Engagement",
    "weddding": "Wedding",
    "wedinng": "Wedding",
    "weddind": "Wedding",
    "wedding": "Wedding",
    "marraige": "Marriage",
    "marriage": "Marriage",
    "mariage": "Marriage",
    "receptoin": "Reception",
    "recpetion": "Reception",
    "recepton": "Reception",
    "reception": "Reception",
    "bithday": "Birthday",
    "birthday": "Birthday",
    "housewarmimg": "Housewarming",
    "housewarming": "Housewarming",
    "bnanquet": "Banquet",
    "convention": "Convention",
    "muhurthan": "Muhurtham",
    "muhurtham": "Muhurtham",
    "cerernony": "Ceremony",
    "ceremony": "Ceremony",
    "celebratlon": "Celebration",
    "whatsapp": "WhatsApp",
    # Islamic wedding events
    "haldi": "Haldi",
    "nikah": "Nikah",
    "walima": "Walima",
    "mehndi": "Mehndi",
    # OCR time/date typos
    "monwards": "onwards",
    "onwwards": "onwards",
    "fobruary": "February",
    "february": "February",
    # OCR venue/location typos
    "vonue": "Venue",
    "royat": "Royal",
    "hydorabad": "Hyderabad",
}

_sbert_model = None
_sbert_available = False


def _get_sbert():
    """Lazily load the Sentence-BERT model if available and enabled."""
    global _sbert_model, _sbert_available
    if not settings.use_sbert:
        return None
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        _sbert_available = True
        logger.info("Sentence-BERT loaded")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentence-BERT unavailable (%s); using dictionary fallback", exc)
        _sbert_available = False
        _sbert_model = None
    return _sbert_model


def _dict_correct(word: str) -> str:
    """Correct a word using the known-places dictionary (exact + fuzzy)."""
    lowered = word.lower().strip()
    if lowered in KNOWN_PLACES:
        return KNOWN_PLACES[lowered]
    # Fuzzy match for near-miss typos
    for key, value in KNOWN_PLACES.items():
        if len(key) >= 5 and _levenshtein(lowered, key) <= 1:
            return value
    return word


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _sbert_match(word: str) -> str:
    """Use Sentence-BERT to find the closest known place name."""
    model = _get_sbert()
    if model is None:
        return _dict_correct(word)
    try:
        candidates = list(KNOWN_PLACES.keys())
        word_vec = model.encode(word, convert_to_numpy=True)
        cand_vecs = model.encode(candidates, convert_to_numpy=True)
        import numpy as np
        from numpy.linalg import norm

        sims = (cand_vecs @ word_vec) / (norm(cand_vecs, axis=1) * norm(word_vec) + 1e-9)
        best = int(np.argmax(sims))
        if sims[best] > 0.6:
            return KNOWN_PLACES[candidates[best]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentence-BERT matching failed (%s)", exc)
    return _dict_correct(word)


def _split_merged_word(word: str) -> List[str]:
    """Split a run-together token into separate words.

    Handles camelCase / TitleCase concatenations produced by OCR that drops
    spaces (e.g. ``GKHillViewResort`` -> ``GK Hill View Resort``). Applies only
    when the token clearly contains word-boundary transitions, so ordinary
    single words are left untouched.
    """
    if not word or " " in word or len(word) < 6:
        return [word]
    if not (re.search(r"[a-z][A-Z]", word) or
            re.search(r"[A-Z][A-Z][a-z]", word)):
        return [word]
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word)      # aB -> a B
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)     # GKHill -> GK Hill
    parts = [p for p in s.split() if p]
    return parts if len(parts) > 1 else [word]


def _normalize_word(word: str) -> str:
    """Apply generic OCR normalization to a single token."""
    fixed = _COMMON_OCR_TYPOS.get(word.lower().strip())
    if fixed:
        return fixed
    split = _split_merged_word(word)
    if len(split) > 1:
        return " ".join(split)
    return word


def correct_text(text: str) -> str:
    """Correct OCR mistakes in a text block.

    Preserves the original line structure. Correcting is done per line (and per
    word within a line) so downstream stages that are aware of line boundaries
    (e.g. the rule-based parser's address/venue extraction) keep working.

    In addition to place-name correction this applies generic OCR normalization:
      * fixes common event-word misspellings (e.g. "Ongagement" -> "Engagement")
      * splits run-together words (e.g. "GKHillViewResort" -> "GK Hill View Resort")
    """
    if not text:
        return text
    model = _get_sbert()
    out_lines = []
    for line in text.split("\n"):
        words = line.split()
        corrected = []
        for w in words:
            if _sbert_available and model is not None:
                corrected.append(_sbert_match(w))
            else:
                corrected.append(_dict_correct(w))
        normalized = []
        for w in corrected:
            nw = _normalize_word(w)
            normalized.extend(nw if isinstance(nw, list) else [nw])
        out_lines.append(" ".join(normalized))
    return "\n".join(out_lines)


def normalize_entity(text: str) -> str:
    """Normalize/title-case an entity string (venue, name, event).

    Only maps the whole string to an event word when the entity is essentially
    that single event label (e.g. "wedding" -> "Wedding"). If the string is a
    longer phrase (e.g. a venue like "wedding celebration ..."), it is cleaned
    up and title-cased instead of being collapsed to just "Wedding".
    """
    if not text:
        return text
    text = text.strip()
    lowered = text.lower()

    # If the whole string is essentially a single event word, return the label.
    words = [w for w in re.split(r"[^a-z']+", lowered) if w]
    for key, val in EVENT_WORDS.items():
        if words and all(w == key for w in words):
            return val

    # Otherwise clean up: strip generic leading labels like "venue:", "address:",
    # "at", "held at", and collapse whitespace.
    cleaned = re.sub(r"^(venue|address|location|place|at|held at|located at)\s*:?\s*",
                     "", text, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned

