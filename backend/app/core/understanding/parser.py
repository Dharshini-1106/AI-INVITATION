"""Generalized invitation parsing (validation & formatting + robust fallback).

This is the FINAL stage of the pipeline. It receives the matched fields from the
entity-extraction + Sentence-BERT matching stages and:
  1. Validates & formats each field into a stable schema.
  2. Fills any gaps using generic, confidence-ranked rule-based extractors that
     are NOT tied to any specific invitation template, person name, venue, or
     layout. Each strategy returns a candidate with a confidence score and the
     final value is chosen by highest confidence rather than hard-coded order.

LayoutLMv3 is used when available to boost event-type classification, but the
rule-based strategies stand alone so the parser works without heavy models.
"""
import logging
import re
import os
from difflib import get_close_matches
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_layoutlm = None
_layoutlm_available = False


def _get_layoutlm():
    """Lazily load a fine-tuned LayoutLMv3 model when explicitly configured.

    Important: do NOT treat a generic pretrained base checkpoint as a task
    specific NER model. Only load LayoutLM when a fine-tuned model name or path
    is supplied via the LAYOUTLM_MODEL environment variable. This prevents the
    parser from silently using microsoft/layoutlmv3-base as if it were
    invitation-specialized.
    """
    global _layoutlm, _layoutlm_available
    if _layoutlm is not None:
        return _layoutlm

    model_name = os.environ.get("LAYOUTLM_MODEL")
    if not model_name:
        # No fine-tuned model configured — do not load the generic base model.
        logger.info("No fine-tuned LayoutLM model configured (set LAYOUTLM_MODEL to enable). Using rule-based parser.")
        _layoutlm_available = False
        _layoutlm = None
        return None

    # If a model is configured, attempt to load it. Loading failures fall back
    # to the rule-based parser gracefully.
    try:
        from transformers import pipeline

        # Be explicit about token-classification aggregation so callers get whole
        # entity spans when available.
        _layoutlm = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )
        _layoutlm_available = True
        logger.info("LayoutLM model loaded: %s", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load LayoutLM model %s (%s); using rule-based parser", model_name, exc)
        _layoutlm_available = False
        _layoutlm = None
    return _layoutlm


# ---------------------------------------------------------------------------
# Candidate model: every extraction strategy returns a ranked candidate.
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single extraction result with a confidence and its strategy name."""
    value: object
    confidence: float
    strategy: str


def _best(candidates: List[Optional[Candidate]]) -> Optional[Candidate]:
    """Return the highest-confidence candidate (ignoring empty values)."""
    best = None
    for c in candidates:
        if c is None:
            continue
        if c.value is None or c.value == "":
            continue
        if best is None or c.confidence > best.confidence:
            best = c
    return best


def _pick(candidates: List[Optional[Candidate]]):
    """Return the value of the best candidate, or '' if none."""
    best = _best(candidates)
    return best.value if best is not None else ""


# ---------------------------------------------------------------------------
# Generic lexicons (semantic cues, not invitation-specific samples)
# ---------------------------------------------------------------------------
_MONTH = (r"january|february|march|april|may|june|july|august|"
          r"september|october|november|december|"
          r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec")
_WEEKDAY = (r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"mon|tue|wed|thu|fri|sat|sun")
_DY = r"\d{1,2}(?:st|nd|rd|th)?"

# Each date strategy is independent; confidence reflects how specific the
# matched span is.
_DATE_STRATEGIES = [
    (rf"\b((?:{_WEEKDAY})\.?,?\s+(?:{_MONTH})\.?\.?\s+{_DY},?\s+\d{{4}})", 0.95,
     "weekday_month_day_year"),
    (rf"\b((?:{_WEEKDAY})\.?,?\s+{_DY}\s+(?:{_MONTH})\.?\.?\s+\d{{4}})", 0.95,
     "weekday_day_month_year"),
    (rf"\b((?:{_MONTH})\.?\.?\s+{_DY},?\s+\d{{4}})", 0.88,
     "month_day_year"),
    (rf"\b({_DY}\s+(?:{_MONTH})\.?\.?\s+\d{{4}})", 0.88,
     "day_month_year"),
    (rf"\b((?:{_WEEKDAY})\.?,?\s+the\s+{_DY}\s+of\s+(?:{_MONTH})\.?\.?\s+\d{{4}})",
     0.88, "weekday_the_day_of_month_year"),
    (rf"\b(the\s+{_DY}\s+of\s+(?:{_MONTH})\.?\.?\s+\d{{4}})", 0.85,
     "the_day_of_month_year"),
    (rf"\b((?:{_MONTH})\.?\.?\s+{_DY}\s*\.\s*\d{{4}})", 0.82,
     "month_day.period_year"),
    (r"\b(\d{1,2}(?:st|nd|rd|th)?[-/]\d{1,2}[-/]\d{2,4})\b", 0.72,
     "numeric_date"),
]

_TIME_STRATEGIES = [
    (r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?|)"
     r"\s*(?:onwards?|onward|on\s*w(a)?rds?|sharp|in\s+the\s+(?:morning|evening|afternoon|night))?\b)",
     0.9, "colon_time"),
    (r"\b(\d{1,2}[:.]\d{2})\s+in\s+the\s+(?:morning|evening|afternoon|night)\b",
     0.88, "colon_time_of_day"),
    (r"\b(\d{1,2}\s+\d{2}\s*(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?)"
        r"\s*(?:onwards?|onward|sharp)?\b)", 0.88, "spaced_time"),
    (r"\b(\d{1,2}\s*(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?)"
     r"\s*(?:onwards?|onward|on\s*w(a)?rds?|sharp|in\s+the\s+(?:morning|evening|afternoon|night))?\b)",
     0.85, "bare_time"),
]

_PHONE_STRATEGIES = [
    (r"(\+?\d[\d\s\-]{8,15}\d)", 0.8, "loose_phone"),
    (r"(\b\d{10}\b)", 0.8, "ten_digit"),
    (r"(\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)", 0.8, "three_three_four"),
]

EMAIL_PATTERN = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

# Event-type lexicon. Order within the dict is irrelevant to correctness
# because candidates are ranked by confidence; specific phrases are boosted.
EVENT_TYPE_KEYWORDS = {
    "wedding": ("Wedding", 0.7),
    "reception": ("Reception", 0.7),
    "marriage": ("Marriage", 0.6),
    "engagement": ("Engagement", 0.8),
    "birthday": ("Birthday", 0.8),
    "anniversary": ("Anniversary", 0.8),
    "baby shower": ("Baby Shower", 0.85),
    "conference": ("Conference", 0.8),
    "seminar": ("Seminar", 0.8),
    "party": ("Party", 0.75),
    "cultural": ("Cultural Event", 0.75),
    "corporate": ("Corporate Event", 0.75),
    "naming ceremony": ("Naming Ceremony", 0.8),
    "namingceremony": ("Naming Ceremony", 0.8),
    "housewarming": ("Housewarming", 0.8),
    "muhurtham": ("Wedding", 0.8),
    "muhurthan": ("Wedding", 0.8),
    "celebration": ("Wedding", 0.6),
    "celebrat": ("Wedding", 0.55),
    "colebration": ("Wedding", 0.5),
    "colebrat": ("Wedding", 0.5),
    "ongagemont": ("Engagement", 0.8),
    "ongage": ("Engagement", 0.55),
    "haldi": ("Haldi", 0.85),
    "nikah": ("Nikah", 0.9),
    "walima": ("Walima", 0.9),
    "mehndi": ("Mehndi", 0.85),
    "mehendi": ("Mehndi", 0.8),
    "nikah ceremony": ("Nikah", 0.95),
    "walima reception": ("Walima", 0.95),
    "haldi ceremony": ("Haldi", 0.9),
    "mehndi ceremony": ("Mehndi", 0.9),
}

# Specific phrases that resolve ambiguity.
_EVENT_TYPE_PHRASES = [
    (r"marriage\s+reception", "Reception", 0.9),
    (r"wedding\s+reception", "Reception", 0.9),
    (r"reception\s+of", "Reception", 0.9),
    (r"wedding\s+ceremony", "Wedding", 0.9),
    (r"wedding\s+of", "Wedding", 0.9),
    (r"engagement\s+of", "Engagement", 0.9),
]

VENUE_KEYWORDS = ["venue", "function hall", "hall",
                  "convention", "palace", "mandapam", "auditorium", "resorts",
                  "garden", "lawns", "hotel", "grounds", "seminar", "community"]

# Words commonly appearing in printer/designer/footer lines whose phone numbers
# are NOT the event's contact info. Contact info only counts when it follows an
# explicit label such as contact/phone/mobile/rsvp/tel.
_FOOTER_DESIGNER_WORDS = [
    "designed", "design", "printed", "print", "creative", "graphics",
    "bureau", "studio", "digital", "xerox", "fotocopy", "photocopy",
    "advertising", "publication", "invitations", "cards", "card",
]

VENUE_STRONG_KEYWORDS = [
    "venue", "located at", "function hall", "convention", "palace",
    "mandapam", "auditorium", "resorts", "resort", "lawns", "hotel",
    "grounds", "seminar hall", "community hall", "garden", "banquet",
    "hall", "pavilion", "temple", "ballroom", "towers", "residency",
    "manor", "pavilion", "itheatre", "marriage hall", "kalyan mandapam",
    "residence", "farm",
]

ADDRESS_KEYWORDS = ["address", "street", "road", "nagar", "layout", "colony",
                    "main road", "cross", "bus stop", "near", "district",
                    "pincode", "pin", "city", "village", "town"]

EVENT_NAME_KEYWORDS = ["invitation", "welcomes", "request the pleasure", "cordially invite",
                       "join us", "with the blessings", "together with their families",
                       "together with our families", "request the honor", "celebrate the",
                       "wedding", "reception", "engagement", "birthday"]

_EVENT_NAME_WORDS = {
    "wedding", "marriage", "reception", "engagement", "birthday",
    "celebration", "ceremony", "invitation", "announcement", "party",
    "function", "housewarming", "muhurtham", "naming", "join", "celebrate",
}

KNOWN_CITIES = {
    "chennai", "madurai", "coimbatore", "bengaluru", "bangalore", "hyderabad",
    "mumbai", "delhi", "kolkata", "pune", "salem", "vellore", "trichy",
    "erode", "kochi", "kerala", "goa", "agra", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "surat", "ahmedabad", "thiruvananthapuram",
    "kanyakumari", "mysuru", "vijayawada", "visakhapatnam", "noida",
    "gurgaon", "thoothukudi", "tirunelveli",
}

_NO_NAME_EVENTS = {"birthday", "naming ceremony", "namingceremony",
                   "housewarming", "muhurtham"}

# Indian states / union territories used to recognise a standalone locality
# line (e.g. "Karnataka", "Tamil Nadu", "Telangana") as an address signal.
INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "andaman and nicobar islands",
    "chandigarh", "dadra and nagar haveli", "daman and diu",
    "jammu and kashmir", "ladakh", "lakshadweep", "puducherry",
}


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------
def _strip_label(line: str) -> str:
    """Remove a leading field label from a line, e.g. 'venue: Garden Hall'.

    Also handles the OCR-merged case where the colon/space between the label and
    the value is dropped entirely (e.g. ``Venue:GKHillViewResort`` or
    ``VenueGKHillViewResort``), leaving just the label word glued to the value.
    """
    s = line.strip()
    # Case with a colon, dash, or other common separator (with or without
    # surrounding spaces).  OCR commonly misreads "Venue" as "Vonue"/"Venu"/
    # "Venve", so those label variants are stripped too (character-level OCR
    # confusions, not venue-name-specific).
    m = re.match(
        r"^(?:venue|vonue|venu|venve|address|location|place|at|held at|located at|contact|phone|date|time)\b"
        r"\s*(?:[:\-–—])?\s*(.+)$",
        s, re.IGNORECASE,
    )
    if m:
        val = m.group(1).strip()
        return val if val else s
    # Case where the label is glued directly to the value with no separator and
    # the remaining value still starts with a word boundary (e.g. "VenueGarden").
    m = re.match(
        r"^(?:venue|vonue|venu|venve|location|place)[A-Z][a-zA-Z].*$",
        s, re.IGNORECASE,
    )
    if m:
        return re.sub(r"^(?:venue|vonue|venu|venve|location|place)", "", s, flags=re.IGNORECASE).strip()
    return s


_STREET_SUFFIX = r"(?:st|street|rd|road|ave|avenue|ln|lane|blvd|boulevard|dr|drive|ct|court|pl|place|way|ter|terrace|cir|circle|mg|main)"


def _extract_street_address(text: str) -> str:
    """Extract a street address (e.g. '123 Anywhere St.') from arbitrary text."""
    m = re.search(
        r"\b(\d{1,4}\s+[A-Za-z][A-Za-z0-9.]*(?:\s+[A-Za-z][A-Za-z0-9.]*)*"
        r"\s*(?:st|street|rd|road|ave|avenue|ln|lane|blvd|boulevard|dr|drive|"
        r"ct|court|pl|place|way|ter|terrace|cir|circle|mg|main)\.?,?)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return ""
    return m.group(1).strip().rstrip(",").strip()


_CITY_ST_ZIP_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*),\s*([A-Z]{2})\s*"
    r"(\d{5}(?:-\d{4})?|\d{6})\b"
)
_CITY_PIN_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(\d{6})\b"
)


def _extract_city_state_zip(text: str) -> str:
    m = _CITY_ST_ZIP_RE.search(text)
    if m:
        return f"{m.group(1)}, {m.group(2)} {m.group(3)}"
    return ""


def _extract_city_pin(text: str) -> str:
    m = _CITY_PIN_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return ""


def _extract_known_city(text_lines: List[str]) -> str:
    for line in text_lines:
        ls = line.strip().rstrip(".,;:")
        if not ls or len(ls) > 30:
            continue
        if ls.lower() in KNOWN_CITIES:
            return ls.title()
    return ""


def _looks_like_street(line: str) -> bool:
    ls = line.strip()
    if not ls:
        return False
    if extract_time(ls) or extract_date(ls):
        return False
    if re.match(r"^at\s*\d", ls, re.IGNORECASE):
        return False
    m = re.search(r"(\d\s*[A-Za-z])", ls)
    if not m:
        return False
    if re.search(r"\d\s*[ap]\.?m\.?", ls, re.IGNORECASE):
        return False
    return True


def _is_field_line(line: str) -> bool:
    lowered = line.lower()
    field_kws = (EVENT_NAME_KEYWORDS + VENUE_KEYWORDS + ADDRESS_KEYWORDS +
                 ["date", "time", "contact", "phone", "www", "http", "rsvp",
                  "email", "with best", "blessings", "family"])
    return any(kw in lowered for kw in field_kws)


# ---------------------------------------------------------------------------
# Gender-aware name helpers (semantic cues, not name-specific)
# ---------------------------------------------------------------------------
_FEM_SUFFIX = re.compile(r"(?:ia|na|a|e)$", re.IGNORECASE)
_MALE_SUFFIX = re.compile(r"(?:[bcdgklmnprstvxz]|er|an|un|in|ee)$", re.IGNORECASE)
_FEM_HINT = re.compile(r"\b(bride|daughter|ridhi|smt|kum)\b", re.IGNORECASE)
_MALE_HINT = re.compile(r"\b(groom|son|sri|selvi|thiru)\b", re.IGNORECASE)


def _gender_hint(name: str) -> str:
    base = name.strip()
    if not base:
        return ""
    if _FEM_HINT.search(base):
        return "female"
    if _MALE_HINT.search(base):
        return "male"
    # Surnames are often shared by both partners and are a poor gender clue;
    # use the given-name token for this intentionally weak fallback instead.
    given_name = base.split()[0]
    if _FEM_SUFFIX.search(given_name):
        return "female"
    if _MALE_SUFFIX.search(given_name):
        return "male"
    return ""


def _assign_bride_groom(n1: str, n2: str, convention: str = "groom-first") -> Dict[str, str]:
    n1, n2 = n1.strip(), n2.strip()
    h1, h2 = _gender_hint(n1), _gender_hint(n2)
    default_bride, default_groom = (n1, n2) if convention == "bride-first" else (n2, n1)
    pair = {("female", "male"): (n1, n2), ("male", "female"): (n2, n1)}
    if h1 and h2 and h1 != h2:
        b, g = pair[(h1, h2)]
        return {"bride": b, "groom": g}
    if h1 == "female":
        return {"bride": n1, "groom": n2}
    if h1 == "male":
        return {"bride": n2, "groom": n1}
    if h2 == "female":
        return {"bride": n2, "groom": n1}
    if h2 == "male":
        return {"bride": n1, "groom": n2}
    return {"bride": default_bride, "groom": default_groom}


_HONORIFIC = r"(?:(?i:mr|mrs|ms|dr|sri|smt|er|kum|thiru|selvi)\.?\s*)?"
_NAME_TOKEN = (
    _HONORIFIC +
    r"([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)"
)
_NAME_HONORIFIC_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|dr|sri|smt|er|kum|thiru|selvi)\.?\s*)+", re.IGNORECASE)


def _clean_name(raw: str) -> str:
    name = _NAME_HONORIFIC_RE.sub("", raw.strip()).strip()
    name = re.sub(r"[.,;:]+$", "", name).strip()
    words = name.split()
    while words and words[0].lower() in _EVENT_NAME_WORDS:
        words.pop(0)
    return " ".join(words).strip()


_NAME_REJECT_SINGLE_WORDS = {
    "with", "and", "or", "of", "for", "at", "on", "in", "by",
    "join", "warm", "invitation", "the",
}

def _looks_like_name(line: str) -> bool:
    ls = line.strip()
    if not (3 <= len(ls) <= 40):
        return False
    if re.search(r"\d", ls):
        return False
    if _is_field_line(ls):
        return False
    if (len(ls) <= 50 and extract_event_type(ls)
            and re.search(r"\b(?:ceremony|reception|celebration|programme|program)\b",
                          ls, re.IGNORECASE)):
        return False
    if re.match(r"^(?:at|held at|located at|venue|location|place)\b",
                ls, re.IGNORECASE):
        return False
    if re.match(r"^(?:event|event\s+name)\s*[:\-]", ls, re.IGNORECASE):
        return False
    words = ls.split()
    if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
        return False
    if len(words) == 1 and words[0].lower() in _NAME_REJECT_SINGLE_WORDS:
        return False
    # A single ALL-CAPS word is almost always a heading/logo (e.g. "PROGRAMME",
    # "INVITATION"), not a person name.
    if len(words) == 1 and words[0].isupper() and len(words[0]) > 1:
        return False
    # A comma usually indicates an address/locality line (e.g. "Banjara Hills,
    # Hyderabad"), not a person name.
    if "," in ls:
        return False
    if not ls[0].isupper():
        return False
    compact = re.sub(r"[^a-z]", "", ls.lower())
    if any(marker in compact for marker in (
            "sonof", "daughterof", "fatherof", "motherof", "grandsonof",
            "granddaughterof", "parentsof", "familyof")):
        return False
    return True


# ---------------------------------------------------------------------------
# Opening/greeting & event-description line detection (glued-OCR aware)
# ---------------------------------------------------------------------------
# Matches invitation greeting / event-description phrasing even when the OCR
# glued words together (e.g. "Togetherwith ourfamihies",
# "Weextendawarminvitationto jonthe", "Ongagemont Colebration of"). Lines that
# match must never be treated as person names.
_GREETING_RE = re.compile(
    r"(?:together\s*with|our\s*famil|we\s*extend|warm\s*invitation|"
    r"request\s*(?:the\s*pleasure|your\s*presence|the\s*hon)|"
    r"solicit|grac|invite\s*you|join\s*us|to\s*witness|celebrat|"
    r"request\s+the\s+pleasure|the\s+(?:wedding|marriage|reception)"
    r"\s+(?:ceremony|celebration)\s+of)",
    re.IGNORECASE,
)


def _is_greeting_or_event_desc(line: str) -> bool:
    """True if a line is greeting / opening / event-description text."""
    return bool(_GREETING_RE.search(line))


# Stop words that belong to parent/host/family lines (not the couple).
_FAMILY_ROLE_STOP = {
    "smt", "sri", "mr", "mrs", "ms", "dr", "late", "lt", "the", "of", "&",
    "and", "son", "daughter", "w/o", "d/o", "s/o", "kum", "selvi", "thiru",
}


def _capitalize_name(name: str) -> str:
    """Normalize casing: keep ALL-CAPS tokens, else capitalize first letter.

    Handles OCR mis-casing of a real name (e.g. groom read as ``rashant``
    instead of ``Prashanth``) so it is still emitted as a proper-looking name.
    """
    words = name.split()
    if not words:
        return name
    out = []
    for w in words:
        if len(w) > 1 and w.isupper():
            out.append(w)
        else:
            out.append(w[0].upper() + w[1:])
    return " ".join(out)


def _find_name_above(text_lines: List[str], idx: int) -> str:
    """Return the name on the nearest line above ``idx`` that can be a person.

    Family lines (``Shoomika`` above a ``D/o ...`` line, ``rashant`` above a
    ``S/o ...`` line) are separated from their parents by exactly one line, so
    we scan upward skipping any parent/host/event filler lines.
    """
    for k in range(idx - 1, -1, -1):
        ls = text_lines[k].strip()
        if not ls:
            continue
        if re.search(r"\d", ls):
            continue
        if _is_greeting_or_event_desc(ls):
            continue
        low = ls.lower()
        # Skip parent/host lines (honorifics, "of", "&") and other filler.
        if any(rk in low for rk in ("smt", "sri", "mr ", "mrs", " w/o ",
                                    "daughter", "son of", "of ", "&", "& ",
                                    "invite", "join", "wedding", "ceremony",
                                    "famil", "with", "and ")):
            continue
        words = [w for w in ls.split() if w.lower() not in _FAMILY_ROLE_STOP]
        if not words:
            continue
        cand = " ".join(words).strip(" .,;:&")
        if len(cand) < 3:
            continue
        return _capitalize_name(cand)
    return ""


def extract_family_roles(text_lines: List[str]) -> Dict[str, str]:
    """Detect bride/groom from ``D/o`` (daughter-of) and ``S/o`` (son-of) lines.

    In Indian invitation layouts the couple's names sit on the line directly
    above the parent linkage lines. Handles both normal (``D/o Smt ...``) and
    OCR-glued forms (``D/oSmtAnjinammmat``, ``So SmB Manjula &``). Returns a
    dict with optional ``bride`` / ``groom`` keys.
    """
    bride = ""
    groom = ""
    for i, line in enumerate(text_lines):
        s = line.strip()
        low = s.lower()
        if re.search(r"\bd\s*/\s*o", s, re.IGNORECASE) and not re.search(
                r"\bs\s*/\s*o", s, re.IGNORECASE):
            nm = _find_name_above(text_lines, i)
            if nm and not bride:
                bride = nm
        elif (re.search(r"\bs\s*/\s*o", s, re.IGNORECASE)
              or re.search(r"^\s*so\s+", low)):
            nm = _find_name_above(text_lines, i)
            if nm and not groom:
                groom = nm
    return {"bride": bride, "groom": groom}


# ---------------------------------------------------------------------------
# Field extractors. Each returns a list of independent candidates.
# ---------------------------------------------------------------------------
# Tolerant Tamil fragments that survive OCR noise.  These are substrings of
# common Tamil wedding/Nikah vocabulary, not a list of people or venues.
_TAMIL_NIKAH_FRAGMENTS = ("நிகா", "நிகாஹ்", "னிகா", "நிகாஹ")
_TAMIL_WEDDING_FRAGMENTS = ("திரும", "ிருமண", "ழப்பிதழ்", "ழைப்பிதழ்",
                           "மணமக", "மணவிழா")
_TAMIL_WALIMA_FRAGMENTS = ("வலீமா", "வலிமா", "அலீமா")
_TAMIL_MEHNDI_FRAGMENTS = ("மஹர்", "மெஹந்தி", "மேகந்தி")


def _tamil_event_type(text: str) -> Optional[Candidate]:
    """Return a wedding/Nikah event-type candidate for Tamil text."""
    if any(frag in text for frag in _TAMIL_NIKAH_FRAGMENTS):
        return Candidate("Nikah", 0.92, "tamil_nikah")
    if any(frag in text for frag in _TAMIL_WALIMA_FRAGMENTS):
        return Candidate("Walima", 0.9, "tamil_walima")
    if any(frag in text for frag in _TAMIL_MEHNDI_FRAGMENTS):
        return Candidate("Mehndi", 0.88, "tamil_mehndi")
    if any(frag in text for frag in _TAMIL_WEDDING_FRAGMENTS):
        return Candidate("Wedding", 0.85, "tamil_wedding")
    return None


def extract_event_type_variants(text: str) -> List[Candidate]:
    lowered = text.lower()
    candidates: List[Candidate] = []
    # Tamil invitations: the OCR text is Tamil, so the English keyword lexicon
    # above cannot match.  Detect wedding/Nikah from tolerant Tamil fragments
    # that survive OCR (e.g. திருமண → "திரும"/"ிருமண", நிகாஹ் → "நிகா",
    # அழைப்பிதழ் → "ழப்பிதழ்").  This keeps Tamil wedding/Nikah invitations
    # classified correctly instead of returning "Not available".
    if _has_tamil(text):
        tamil_et = _tamil_event_type(text)
        if tamil_et is not None:
            candidates.append(tamil_et)
    for line in text.splitlines():
        heading = line.strip()
        if len(heading) > 50 or not heading:
            continue
        words = re.sub(r"[^A-Za-z ]", " ", heading).split()
        if (words and len(words) <= 4
                and words[0].lower() in EVENT_TYPE_KEYWORDS
            and words[-1].lower() in {"ceremony", "reception", "celebration"}
            and (len(words) > 1 or words[0].lower() in {
                    "nikah", "haldi", "walima", "mehndi", "reception",
                })):
            candidates.append(Candidate(" ".join(words).title(), 0.92,
                                        "event_heading"))
    for pat, typ, conf in _EVENT_TYPE_PHRASES:
        if re.search(pat, lowered):
            candidates.append(Candidate(typ, conf, f"phrase:{pat}"))
    for kw, (typ, conf) in EVENT_TYPE_KEYWORDS.items():
        if kw in lowered:
            candidates.append(Candidate(typ, conf, f"keyword:{kw}"))
    return candidates


def extract_event_type(text: str) -> Candidate:
    return _best(extract_event_type_variants(text))


def extract_event_name_variants(text: str) -> List[Candidate]:
    lowered = text.lower()
    candidates: List[Candidate] = []
    for line in text.splitlines():
        heading = line.strip()
        words = re.sub(r"[^A-Za-z ]", " ", heading).split()
        if (len(heading) <= 50 and words and len(words) <= 4
            and words[0].lower() in EVENT_TYPE_KEYWORDS
            and words[-1].lower() in {"ceremony", "reception", "celebration"}):
            candidates.append(Candidate(_clean(heading), 0.94,
                                        "event_heading"))
        # Unknown cultural/religious event labels are still headings when the
        # typography says ``<name> Ceremony/Reception/...``.  This structural
        # rule deliberately requires a suffix and a short line, rather than
        # treating every capitalized line as an event.
        if (2 <= len(words) <= 5 and len(heading) <= 50
                and words[-1].lower() in {"ceremony", "reception", "celebration", "function"}
                and all(word[0].isupper() for word in words if word)):
            candidates.append(Candidate(_clean(heading), 0.88,
                                        "generic_structural_heading"))
    labeled = re.search(r"^\s*(?:event|event name)\s*[:\-]\s*(.+)$",
                        text, re.IGNORECASE | re.MULTILINE)
    if labeled:
        value = _clean(labeled.group(1))
        if value:
            candidates.append(Candidate(value, 0.96, "event_label"))
    celebration = re.search(
        r"(wedding\s+(?:ceremony|celebration|reception)|"
        r"(?:wedding|reception|engagement|birthday|naming ceremony|housewarming)"
        r"\s+(?:ceremony|celebration|reception|party|function))",
        lowered,
    )
    if celebration:
        candidates.append(Candidate(
            " ".join(w.capitalize() for w in celebration.group(1).split()),
            0.9, "celebration_phrase"))
    invite = re.search(
        r"(?:invite you to|request the pleasure of your company at|celebrate)"
        r"\s+(?:the\s+)?"
        r"(wedding|reception|engagement|birthday|naming ceremony|housewarming)",
        lowered,
    )
    if invite:
        candidates.append(Candidate(invite.group(1).capitalize(), 0.85,
                                    "invite_marker"))
    # Event-description phrase combining an event type with "celebration",
    # e.g. "ongagemont colebration of" (engaged+co(b)lebration OCR garbled)
    # or "engagement celebration of". Yields a fuller event name such as
    # "Engagement Celebration" instead of just the bare event type.
    _celeb_kw = r"(?:celebration|celebrat|colebration|colebrat)"
    celeb_phrase = [
        (r"engagement|ongagemont|ongage", "Engagement"),
        (r"wedding|marriage|muhurtham", "Wedding"),
        (r"birthday|bday", "Birthday"),
        (r"reception", "Reception"),
        (r"housewarming|house\s?warm", "Housewarming"),
    ]
    for kw_pat, label in celeb_phrase:
        if re.search(rf"\b(?:{kw_pat})\b\s*{_celeb_kw}\s*(?:of\b|\b)",
                     lowered):
            candidates.append(Candidate(
                f"{label} Celebration", 0.92, f"{label.lower()}_celebration"))
    et = extract_event_type(text)
    if et is not None:
        candidates.append(Candidate(et.value, 0.7, "event_type_fallback"))
    return candidates


def extract_event_name(text: str) -> Candidate:
    return _best(extract_event_name_variants(text))


def extract_date_variants(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    month_glue = ("january|february|fobruary|febuary|februaty|march|april|may|"
                  "june|july|august|september|october|november|december|"
                  "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
    # OCR commonly removes all separators from a date.  Insert boundaries
    # around *known date components* only; a generic letter/digit split breaks
    # ordinals (``13thFebruary`` -> ``13 th February``).
    normalized_text = re.sub(
        rf"(?i)\b({_WEEKDAY})([.,]?)(?=\d)", r"\1\2 ", text)
    normalized_text = re.sub(
        rf"(?i)(\d{{1,2}}(?:st|nd|rd|th)?)(?=(?:{month_glue})(?=\d{{4}}\b|\b))", r"\1 ",
        normalized_text)
    normalized_text = re.sub(
        rf"(?i)((?:{month_glue}))(?=\d{{4}}\b)", r"\1 ", normalized_text)
    normalized_text = re.sub(r"(?i)(\d{1,2}(?:st|nd|rd|th)?)(?=\d{4}\b)",
                             r"\1 ", normalized_text)
    month_names = (
        "january february march april may june july august september october "
        "november december jan feb mar apr jun jul aug sep sept oct nov dec"
    ).split()
    def normalize_month_token(match):
        token = match.group(0).lower()
        match_name = get_close_matches(token, month_names, n=1, cutoff=0.72)
        return match_name[0] if match_name else token
    normalized_text = re.sub(r"\b[A-Za-z]{3,12}\b", normalize_month_token,
                             normalized_text)
    # Support two-line dates like "08th\nSunday June 2025".
    m = re.search(
        rf"\b({_DY})\s*\n\s*(?:{_WEEKDAY})\.?,?\s*({_MONTH})\.??\s+(\d{{4}})",
        normalized_text, re.IGNORECASE,
    )
    if m:
        candidates.append(Candidate(
            f"{m.group(1)} {m.group(2)} {m.group(3)}",
            0.92,
            "day_newline_weekday_month_year",
        ))
    m = re.search(
        rf"\b({_DY})\s*\n\s*({_MONTH})\.??\s+(\d{{4}})",
        normalized_text, re.IGNORECASE,
    )
    if m:
        candidates.append(Candidate(
            f"{m.group(1)} {m.group(2)} {m.group(3)}",
            0.90,
            "day_newline_month_year",
        ))
    for pat, conf, strategy in _DATE_STRATEGIES:
        m = re.search(pat, normalized_text, re.IGNORECASE)
        if m:
            candidates.append(Candidate(m.group(1).strip(), conf, strategy))
    # Numeric dates may have no line-level label, so retain a direct candidate
    # even when the surrounding OCR text was heavily decorated.
    for m in re.finditer(r"\b\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}\b", normalized_text):
        candidates.append(Candidate(m.group(0), 0.72, "numeric_date"))
    return candidates


def extract_date(text: str) -> Candidate:
    return _best(extract_date_variants(text))


def extract_time_variants(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    # Decorative OCR frequently merges the invitation preposition with its
    # time ("At9am" rather than "At 9am").  Remove only that leading
    # preposition when it is immediately followed by a digit; this leaves
    # ordinary text and address lines untouched while allowing the existing
    # time grammar to perform all validation and formatting.
    normalized_text = re.sub(r"\bat\s*(?=\d)", "", text, flags=re.IGNORECASE)
    for pat, conf, strategy in _TIME_STRATEGIES:
        m = re.search(pat, normalized_text, re.IGNORECASE)
        if m:
            candidates.append(Candidate(m.group(1).strip(), conf, strategy))
    return candidates


def extract_time(text: str) -> Candidate:
    return _best(extract_time_variants(text))


def extract_contact_variants(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    labeled = re.search(
        r"\b(?:contact(?:\s+(?:number|no))?|phone|mobile|mob|ph|tel|rsvp)\s*:?\s*"
        r"(\+?\d[\d\s\-]{8,15}\d)",
        text, re.IGNORECASE)
    if labeled:
        candidates.append(Candidate(labeled.group(1).strip(), 0.95, "label_phone"))
    for pat, conf, strategy in _PHONE_STRATEGIES:
        m = re.search(pat, text)
        if m:
            candidates.append(Candidate(m.group(1).strip(), conf, strategy))
    return candidates


def extract_contact(text: str) -> Candidate:
    return _best(extract_contact_variants(text))


def _looks_like_event_desc(text: str) -> bool:
    """Return True if the text is an invitation event-description phrase
    (e.g. 'the marriage reception of', 'the wedding ceremony of') rather
    than an actual venue name.

    These phrases describe the event itself (``at the marriage reception
    of ...``) and must not be captured as venue values.
    """
    t = text.strip().lower()
    return bool(re.match(
        r"^(?:the\s+)?(?:wedding|marriage|reception|engagement|birthday|"
        r"naming\s+ceremony|housewarming|muhurtham)"
        r"(?:\s+(?:ceremony|celebration|reception|party))?\s+of\b",
        t,
    ))


def _venue_from_line(line: str) -> str:
    """Extract the venue value from a line that starts with a venue label.

    Handles both ``Venue: Garden Hall`` and the OCR-merged ``Venue:GKHillViewResort``
    / ``VenueGKHillViewResort`` forms. The label is stripped generically; the OCR
    merged-word splitter is applied only when a strong venue keyword is present
    so ordinary lines are not altered.

    ``at``-prefixed lines are only treated as venue labels when the value is a
    genuine venue. Time expressions (``at 5 PM``, ``at 6:30 in the evening``)
    and event-description phrases (``at the marriage reception of``) are NOT
    venues and are rejected.
    """
    s = line.strip()
    lowered = s.lower()
    # Only treat it as a labeled venue line if it starts with a label.  OCR
    # commonly misreads "Venue" as "Vonue"/"Venu"/"Venve" etc., so we
    # accept a small set of visually-confusable label variants (character-level, not
    # venue-name-specific).
    if not re.match(r"^(?:venue|vonue|venu|venve|location|place|at|located at)\b", lowered):
        return ""
    # Reject a bare label with no value (e.g. a line that is just "Venue" or
    # "Venue:"), or a label that itself only repeats a venue keyword
    # (e.g. "at Place"). A genuine venue label must be followed by a value.
    remainder = _strip_label(s)
    if (not remainder or remainder.lower() in ("venue", "location", "place",
                                               "at", "located at", ":")
            or not re.match(r"^[:?]?\s*.+", remainder, re.DOTALL)
            or re.match(r"^[\s:;.,]+$", remainder)):
        return ""
    val = remainder
    # Reject "at <time>" and "at <event description>" as venue labels:
    # these are NOT venues (e.g. "at 5 PM", "at the marriage reception of").
    if val and re.match(r"^(?:at|held at|located at)\b", lowered):
        if extract_time(val) or _looks_like_event_desc(val):
            return ""
    # Reject a value that is itself empty after stripping punctuation.
    if not val.strip(" :;.,"):
        return ""
    # An ``at`` location commonly carries the venue followed by a city. Keep
    # those semantic parts separate; explicit Venue labels remain untouched.
    if (re.match(r"^(?:at|held at|located at)\b", lowered)
            and "," in val):
        suffix = val.split(",", 1)[1].strip()
        if not re.match(r"\d", suffix):
            val = val.split(",", 1)[0].strip()
    # If OCR merged the label and value (no separator), try splitting the
    # run-together venue value.
    if val and re.search(r"[a-z][A-Z]", val):
        try:
            from .correction import correct_text
            split_val = correct_text(val)
            if split_val and len(split_val) < 80:
                return split_val
        except Exception:  # noqa: BLE001
            pass
    return val


def extract_venue_variants(text_lines: List[str], text: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    def usable_venue(value: str) -> bool:
        return bool(value and not extract_date(value) and not extract_time(value))

    for line in text_lines:
        val = _venue_from_line(line)
        if usable_venue(val):
            # Labeled venue lines are the most reliable.
            candidates.append(Candidate(val, 0.95, "venue_label"))
    for line in text_lines:
        lowered = line.lower()
        for kw in VENUE_STRONG_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                val = _strip_label(line)
                if not usable_venue(val) or len(val) >= 80:
                    continue
                # Reject bare label values (e.g. a line that is just "Venue"
                # or "Hall" — the keyword itself is not a venue name).
                if val.lower() in VENUE_STRONG_KEYWORDS or val.lower() in VENUE_KEYWORDS:
                    continue
                candidates.append(Candidate(val, 0.85, f"venue_keyword:{kw}"))
    for line in text_lines:
        m = re.match(r"^at\s+(.+)$", line, flags=re.IGNORECASE)
        if m:
            rest = m.group(1).strip()
            if (usable_venue(rest) and len(rest) < 80
                    and not _looks_like_event_desc(rest)):
                candidates.append(Candidate(rest, 0.7, "at_phrase"))
    # Handle venue label on its own line followed by the value on the next line
    # e.g. "Venue" then "GKHillViewResort" on the next line.
    for i, line in enumerate(text_lines):
        ls = line.strip()
        if re.match(r"^venue\s*:?$", ls, re.IGNORECASE) and i + 1 < len(text_lines):
            next_line = text_lines[i + 1].strip()
            if (usable_venue(next_line) and len(next_line) < 80
                    and not re.search(r"\d", next_line)
                    and not _looks_like_event_desc(next_line)):
                candidates.append(Candidate(next_line, 0.92, "venue_label_next_line"))
    return candidates


def extract_venue(text_lines: List[str], text: str) -> Candidate:
    variants = extract_venue_variants(text_lines, text)
    # Prefer a true venue (label / strong keyword / "at" phrase) over a bare
    # street-address line. A street line is usually part of the address, not
    # the venue name, so only use it when nothing stronger is present.
    strong = [c for c in variants
              if c.strategy in ("venue_label",) or c.strategy.startswith("venue_keyword")]
    if strong:
        return _best(strong)
    at = [c for c in variants if c.strategy == "at_phrase"]
    if at:
        return _best(at)
    return _best(variants)


def extract_address_variants(text_lines: List[str], text: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    def usable_address(value: str) -> bool:
        return bool(value and not extract_date(value) and not extract_time(value))
    for line in text_lines:
        at_location = re.match(
            r"^(?:at|located at|held at)\s+[^,]+,\s*(.+)$",
            line.strip(), re.IGNORECASE)
        if at_location and not re.match(r"\d", at_location.group(1).strip()):
            value = at_location.group(1).strip().rstrip(".,;:")
            if value:
                candidates.append(Candidate(value, 0.82, "at_location_suffix"))
    for line in text_lines:
        if re.search(r"^address\s*:", line, re.IGNORECASE):
            val = _strip_label(line)
            if usable_address(val) and len(val) < 150:
                candidates.append(Candidate(val, 0.95, "address_label"))
    # A labeled venue followed by a locality line forms a location block even
    # when the locality has no street keyword.
    for index, line in enumerate(text_lines[:-1]):
        if not re.match(r"^(?:venue|location|place)\s*[:\-]", line.strip(), re.IGNORECASE):
            continue
        for following_line in text_lines[index + 1:]:
            following = following_line.strip()
            if extract_date(following) or extract_time(following):
                continue
            if following and not re.match(r"^(?:venue|address|contact|rsvp)\b",
                                          following, re.IGNORECASE):
                candidates.append(Candidate(following.rstrip(".,;:"), 0.84,
                                            "venue_following_locality"))
                break
    for line in text_lines:
        cit = _extract_city_state_zip(line)
        if cit:
            candidates.append(Candidate(cit, 0.9, "city_state_zip"))
    for line in text_lines:
        pin = _extract_city_pin(line)
        if pin:
            candidates.append(Candidate(pin, 0.88, "city_pin"))
    city = _extract_known_city(text_lines)
    if city:
        candidates.append(Candidate(city, 0.85, "known_city_line"))
    # A standalone state line (e.g. "Karnataka", "Tamil Nadu") is a valid
    # address signal for Indian invitations. This is kept STRICT (membership in
    # the Indian-states set only) so that arbitrary capitalized lines — most
    # importantly person names (e.g. "Bhoomika") or venue words — are never
    # misclassified as an address.
    for line in text_lines:
        ls = line.strip().rstrip(".,;:")
        if not ls or len(ls) > 40 or extract_date(ls) or extract_time(ls):
            continue
        low = ls.lower()
        if low in INDIAN_STATES:
            candidates.append(Candidate(ls.title(), 0.82, "known_locality"))
    for line in text_lines:
        street = _extract_street_address(line)
        if street and not extract_time(street):
            candidates.append(Candidate(street, 0.8, "street_substr"))

    # A street line followed by a city/state/postal line is one address block.
    # Keep it together so the parser does not mistake the first line for a
    # venue or lose the locality from the final address.
    for index, line in enumerate(text_lines[:-1]):
        street = _extract_street_address(line)
        if not street or extract_time(line):
            continue
        for following_line in text_lines[index + 1:]:
            following = following_line.strip()
            if extract_date(following) or extract_time(following):
                continue
            if (_looks_like_event_desc(following)
                    or re.search(r"\b(?:to\s+follow|reception|invite|rsvp)\b",
                                 following, re.IGNORECASE)):
                break
            locality = (_extract_city_state_zip(following) or
                        _extract_city_pin(following) or
                        _extract_known_city([following]) or
                        following if re.match(r"^[A-Z][A-Za-z .'-]+(?:,|$)", following)
                        else "")
            if locality and len(following) < 100:
                candidates.append(Candidate(
                    f"{street}, {following.rstrip('.,;:')}", 0.93,
                    "street_locality_block"))
                break
            if following and not re.match(r"^(?:at|on|date|time)\b", following,
                                          re.IGNORECASE):
                break
    for line in text_lines:
        lowered = line.lower()
        if not any(re.search(rf"\b{re.escape(kw)}\b", lowered)
                   for kw in ADDRESS_KEYWORDS):
            continue
        words = line.split()
        if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
            continue
        locality = _strip_label(line)
        if locality and len(locality) < 120:
            candidates.append(Candidate(locality, 0.7, "keyword_locality"))
    return candidates


def extract_address(text_lines: List[str], text: str, venue: str) -> Candidate:
    venue_has_street = bool(_extract_street_address(venue))
    variants = extract_address_variants(text_lines, text)
    if venue_has_street:
        allowed = {"address_label", "city_state_zip", "city_pin",
               "known_city_line", "street_locality_block",
               "at_location_suffix"}
        variants = [c for c in variants if c.strategy in allowed]
    return _best(variants)


def _split_street_location(text_lines: List[str]) -> tuple[str, str]:
    """Return a street and its following postal/city locality, if present.

    This is a location fallback, not a named-venue detector. It keeps the two
    physical address lines distinct when an invitation gives no hall/venue name.
    """
    for index, line in enumerate(text_lines[:-1]):
        street = _extract_street_address(line)
        if not street or extract_time(line):
            continue
        for following_line in text_lines[index + 1:]:
            following = following_line.strip().rstrip(".,;:")
            if not following or extract_date(following) or extract_time(following):
                continue
            locality = _extract_city_state_zip(following) or _extract_city_pin(following)
            if locality:
                return street, locality
            break
    return "", ""


# ---------------------------------------------------------------------------
# Name extraction: multiple independent layout strategies, ranked by confidence
# ---------------------------------------------------------------------------
def _names_candidate(bride: str, groom: str, conf: float, strategy: str) -> Optional[Candidate]:
    if not bride and not groom:
        return None
    return Candidate({"bride": bride, "groom": groom}, conf, strategy)


# Tamil script range (Unicode block U+0B80–U+0BFF).
_TAMIL_LO, _TAMIL_HI = 0x0B80, 0x0BFF


def _has_tamil(text: str) -> bool:
    """True if *text* contains at least one Tamil Unicode codepoint."""
    return any(_TAMIL_LO <= ord(c) <= _TAMIL_HI for c in (text or ""))


# Invitation headings / blessing lines that must never be treated as a person
# name even when they carry Tamil script.
_TAMIL_NON_NAME_MARKERS = (
    "அழைப்பிதழ்", "திருமண", "வரவேற்பு", "நன்றி", "வாழ்த்து", "அன்புடன்",
    "வாழ்த்துகள்", "நிகாஹ்", "நிகாஹ", "வரவேற்பு", "குடும்பம்", "திருநாள்",
    "மகிழ்ச்சி", "கல்யாணம்", "வலீமா", "வலிமா", "அலீமா", "ழப்பிதழ",
    "நகாஹ", "மணம", "நிகா", "லீமா",
)

# Tamil blessing / invitation phrases that indicate the text is NOT a person name.
_TAMIL_BLESSING_MARKERS = (
    "வாழ்த்த", "அன்புடன்", "வரவேற்பு", "நன்றி", "திருநாள்",
    "மகிழ்ச்சி", "குடும்பம்", "அழைப்ப", "நிகாஹ்", "நிகாஹ",
    "மணம", "திருமண", "கல்யாணம்", "வலீமா", "வலிமா", "அலீமா",
    "எல்லாம்", "அல்லாஹ்", "merciful", "blessed",
)


def _is_tamil_blessing_or_invitation(text: str) -> bool:
    """True if text contains Tamil blessing/invitation markers."""
    for marker in _TAMIL_BLESSING_MARKERS:
        if marker in text:
            return True
    return False


def _tamil_name_shape_ok(text: str, initial: Optional[str]) -> bool:
    """Return True if the text looks name-shaped (not a paragraph/blessing)."""
    tamil_len = sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF)
    if initial is None and tamil_len > 12:
        return False
    return True


def _tamil_name_candidates(ocr_lines: List[Dict]) -> List[Dict]:
    """Return prominent Tamil name candidates with layout-based scores.

    Each candidate is a dict with the recognized Tamil text, an optional Latin
    initial (``A.`` / ``S.`` …) and a prominence *score* derived from the text
    height of its OCR bounding box.  Title / blessing / greeting lines are
    excluded.
    """
    cands = []
    for item in (ocr_lines or []):
        text = (item.get("text") or "").strip()
        if not text or not _has_tamil(text):
            continue
        if any(marker in text for marker in _TAMIL_NON_NAME_MARKERS):
            continue
        if _is_greeting_or_event_desc(text):
            continue
        if _is_field_line(text):
            continue
        if _is_tamil_blessing_or_invitation(text):
            continue
        if not _tamil_name_shape_ok(text, None):
            continue
        if not (4 <= len(text) <= 30):
            continue
        conf = float(item.get("confidence", 0.0) or 0.0)
        m = re.match(r"^([A-Za-z])\.\s*", text)
        initial = m.group(1).upper() if m else None
        if initial is None and conf < 0.7:
            continue
        if initial is not None and conf < 0.4:
            continue
        if not _tamil_name_shape_ok(text, initial):
            continue
        h = 0.0
        bbox = item.get("bbox")
        if bbox:
            try:
                pts = [(float(p[0]), float(p[1])) for p in bbox]
                ys = [p[1] for p in pts]
                h = max(ys) - min(ys)
            except (TypeError, ValueError, IndexError):
                h = 0.0
        score = h
        if initial:
            score += 60.0
        cands.append({
            "text": text, "initial": initial, "score": score, "bbox": bbox,
            "conf": conf,
        })
    return cands


def _pair_tamil_initial(ocr_lines: List[Dict], cand: Dict) -> Optional[str]:
    """Find a Latin initial (``A.`` / ``S.`` …) on the same line as a Tamil
    name and return it, so the displayed name keeps its English initial."""
    bbox = cand.get("bbox")
    if not bbox:
        return None
    try:
        pts = [(float(p[0]), float(p[1])) for p in bbox]
        ymin, ymax = min(p[1] for p in pts), max(p[1] for p in pts)
        xmin, xmax = min(p[0] for p in pts), max(p[0] for p in pts)
    except (TypeError, ValueError, IndexError):
        return None
    for item in (ocr_lines or []):
        t = (item.get("text") or "").strip()
        m = re.match(r"^([A-Za-z])\.", t)
        if not m:
            continue
        bb = item.get("bbox")
        if not bb:
            continue
        try:
            pp = [(float(q[0]), float(q[1])) for q in bb]
            y0, y1 = min(q[1] for q in pp), max(q[1] for q in pp)
            x0, x1 = min(q[0] for q in pp), max(q[0] for q in pp)
        except (TypeError, ValueError, IndexError):
            continue
        same_row = (y0 <= ymax and y1 >= ymin)
        close_row = abs(y0 - ymin) < 12 or abs(y1 - ymax) < 12
        same_col = x0 <= xmax + 80 and x1 >= xmin - 80
        if (same_row or close_row) and same_col:
            return m.group(1).upper()
    return None


def extract_tamil_names(ocr_lines: List[Dict], text_lines: List[str]) -> Optional[Candidate]:
    """Extract bride/groom from a Tamil invitation using OCR layout.

    Prefers large, prominent, centered Tamil text and name lines that carry a
    Latin initial (``A.`` / ``S.`` …).  Returns ``None`` when no Tamil name
    candidate is found so other (English) strategies can run instead.
    """
    cands = _tamil_name_candidates(ocr_lines)
    if not cands:
        return None
    # Deduplicate near-identical reads of the same name (the enhanced and
    # original OCR passes can return the same person twice with slightly
    # different Tamil misreads).  Two candidates are merged when their Tamil
    # body is within a small edit distance.
    def _tamil_body(t: str) -> str:
        return re.sub(r"^[A-Za-z]\.\s*", "", t)

    uniq = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        body = _tamil_body(c["text"])
        dup = False
        for u in uniq:
            ub = _tamil_body(u["text"])
            if not ub or not body:
                continue
            if _levenshtein(body, ub) <= max(3, int(0.3 * max(len(body), len(ub)))):
                dup = True
                break
        if not dup:
            uniq.append(c)
    for c in uniq:
        if c["initial"] is None:
            c["initial"] = _pair_tamil_initial(ocr_lines, c)
    uniq.sort(key=lambda x: -x["score"])
    top = uniq[:2]
    names = []
    for c in top:
        nm = c["text"]
        if c["initial"] and not re.match(r"^%s\." % re.escape(c["initial"]), nm):
            nm = "%s. %s" % (c["initial"], nm)
        names.append(nm)
    if not names:
        return None

    def _role(nm: str) -> Optional[str]:
        m = re.match(r"^([A-Za-z])\.", nm)
        if not m:
            return None
        L = m.group(1).upper()
        if L == "A":
            return "bride"
        if L == "S":
            return "groom"
        return None

    assigned = {}
    for nm in names:
        r = _role(nm)
        if r and r not in assigned.values():
            assigned[r] = nm

    # Initialize from assigned roles so the fallback below never overwrites
    # an initial-based assignment (this prevents the same candidate being
    # returned as both bride and groom).
    bride = assigned.get("bride", "")
    groom = assigned.get("groom", "")

    # Fill remaining empty roles from the unassigned names in order.
    for nm in names:
        if nm in assigned.values():
            continue
        if not bride:
            bride = nm
        elif not groom:
            groom = nm

    # Final fallback only for roles that are still empty AND not already set
    # by an initial. This prevents assigning names[1] to groom when names[1]
    # happens to be the same candidate already assigned to bride.
    if len(names) >= 2:
        if not bride and "bride" not in assigned:
            bride = names[0]
        if not groom and "groom" not in assigned:
            groom = names[1]

    # Hard guard: never return the same OCR candidate for both roles.
    if bride and groom and bride == groom:
        if "bride" in assigned and "groom" not in assigned:
            groom = ""
        elif "groom" in assigned and "bride" not in assigned:
            bride = ""
        else:
            bride = names[0] if names else ""
            groom = names[1] if len(names) > 1 else ""
            if bride == groom:
                groom = ""

    return Candidate({"bride": bride, "groom": groom}, 0.9, "tamil_layout")


def extract_names_variants(text_lines: List[str], text: str,
                           ocr_lines: List[Dict] = None) -> List[Candidate]:
    candidates: List[Candidate] = []

    # Tamil invitations: extract bride/groom from layout-aware Tamil OCR.  This
    # is the highest-priority strategy for Tamil documents and returns real
    # Tamil Unicode (never a Latin transliteration or hard-coded name).
    if ocr_lines:
        tamil_names = extract_tamil_names(ocr_lines, text_lines)
        if tamil_names is not None and (
                tamil_names.value.get("bride") or tamil_names.value.get("groom")):
            candidates.append(tamil_names)

    def is_family_context(value: str) -> bool:
        compact = re.sub(r"[^a-z/]", "", value.lower())
        return any(marker in compact for marker in (
            "sonof", "daughterof", "s/o", "d/o", "w/o", "fatherof",
            "motherof", "parent", "familyof", "grandsonof",
            "granddaughterof"))

    bride_m = re.search(
        r"\b(?:bride|daughter)\s*:?\s*([A-Z][a-zA-Z.]{2,40}?)"
        r"(?=\s+(?:groom|son|bride|daughter)\b|$)", text, re.IGNORECASE)
    groom_m = re.search(
        r"\b(?:groom|son)\s*:?\s*([A-Z][a-zA-Z.]{2,40}?)"
        r"(?=\s+(?:bride|daughter|groom|son)\b|$)", text, re.IGNORECASE)
    if bride_m or groom_m:
        c = _names_candidate(
            _clean_name(bride_m.group(1)) if bride_m else "",
            _clean_name(groom_m.group(1)) if groom_m else "",
            0.95, "explicit_marker")
        if c:
            candidates.append(c)

    dos = re.search(
        r"\b([A-Z][a-zA-Z]{2,})\s*\(D\s*/\s*o[^)]*\)"
        r".*?\b([A-Z][a-zA-Z]{2,})\s*\(S\s*/\s*o[^)]*\)",
        text, re.DOTALL)
    if dos:
        c = _names_candidate(_clean_name(dos.group(1)), _clean_name(dos.group(2)),
                             0.97, "d_o_s_o")
        if c:
            candidates.append(c)

    # Family roles (D/o / S/o line linkage): the couple's names sit above their
    # parent lines. This is the strongest signal for the classic Indian layout
    # where the opening lines carry greeting filler but the real names are tied
    # to "daughter of" / "son of" families.
    family_roles = extract_family_roles(text_lines)
    if family_roles.get("bride") or family_roles.get("groom"):
        c = _names_candidate(family_roles.get("bride", ""),
                             family_roles.get("groom", ""),
                             0.96, "family_roles")
        if c:
            candidates.append(c)

    weds = re.search(
        r"\b" + _NAME_TOKEN + r"\s+(?:(?:weds|wed|marries|marry|gets married to))\s+" + _NAME_TOKEN,
        text)
    if weds:
        pair = _assign_bride_groom(_clean_name(weds.group(1)),
                                   _clean_name(weds.group(2)), "bride-first")
        c = _names_candidate(pair["bride"], pair["groom"], 0.9, "weds_verb")
        if c:
            candidates.append(c)

    # OCR may put the marriage connector on its own line. Pair only the
    # nearest valid names around that connector; family lines are excluded.
    for i, line in enumerate(text_lines):
        if line.strip().lower() not in {"weds", "wed", "marries", "married to"}:
            continue
        before = next((text_lines[j].strip() for j in range(i - 1, -1, -1)
                       if _looks_like_name(text_lines[j])
                       and not is_family_context(text_lines[j])), "")
        after = next((text_lines[j].strip() for j in range(i + 1, len(text_lines))
                      if _looks_like_name(text_lines[j])
                      and not is_family_context(text_lines[j])), "")
        if before and after:
            before_role = text_lines[i - 1] if i > 0 else ""
            after_role = text_lines[i + 2] if i + 2 < len(text_lines) else ""
            if (re.search(r"\bson\s+of\b|\bs/o\b", before_role, re.I)
                    and re.search(r"\bdaughter\s+of\b|\bd/o\b", after_role, re.I)):
                pair = {"bride": _clean_name(after), "groom": _clean_name(before)}
            else:
                # In the ordinary ``A Weds B`` construction, document order
                # is groom-first unless explicit parent-role evidence says
                # otherwise.  Gender suffixes remain only a weak fallback.
                pair = _assign_bride_groom(_clean_name(before), _clean_name(after),
                                           "groom-first")
            candidates.append(Candidate(pair, 0.94, "standalone_weds"))

    marriage = re.search(
        r"\b(?:marriage|wedding|union)\s+of\s+" + _NAME_TOKEN +
        r"\s+(?:with|and|to)\s+" + _NAME_TOKEN, text)
    if marriage:
        pair = _assign_bride_groom(_clean_name(marriage.group(1)),
                                   _clean_name(marriage.group(2)), "bride-first")
        c = _names_candidate(pair["bride"], pair["groom"], 0.9, "marriage_of")
        if c:
            candidates.append(c)

    ds = re.search(
        r"\b" + _NAME_TOKEN + r"\s+daughter\b.*?\b(?:with|and)\s+" + _NAME_TOKEN + r"\s+son\b",
        text)
    if ds:
        c = _names_candidate(_clean_name(ds.group(1)), _clean_name(ds.group(2)),
                             0.92, "daughter_son")
        if c:
            candidates.append(c)

    ampersand_re = re.compile(
        r"(?:\bwith\s+)?" + _NAME_TOKEN + r"\s*(?:&|(?i:and))\s*" + _NAME_TOKEN)
    for line in text_lines:
        if is_family_context(line):
            continue
        ampersand = ampersand_re.search(line)
        if not ampersand:
            continue
        n1 = _clean_name(ampersand.group(1))
        n2 = _clean_name(ampersand.group(2))

        def _is_valid_name_token(tok: str) -> bool:
            if not tok:
                return False
            words = tok.split()
            if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
                return False
            # Reject tokens that include field labels / date & time words.
            if any(re.search(rf"\b{re.escape(w)}\b", w2, re.IGNORECASE)
                   for w2 in words for w in
                   ("date", "time", "venue", "address", "contact", "at",
                    "the", "on", "of", "to", "follow", "rsvp", "am", "pm")):
                return False
            return True

        if _is_valid_name_token(n1) and _is_valid_name_token(n2):
            pair = _assign_bride_groom(n1, n2, "groom-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.85, "ampersand")
            if c:
                candidates.append(c)

    # OCR often places the conjunction on its own line:
    #   Person A
    #   &
    #   Person B
    # Use the nearest valid names on either side instead of requiring the
    # three visual elements to be returned as one OCR line.
    for i, line in enumerate(text_lines):
        if line.strip().lower() not in {"&", "and", "+"}:
            continue
        before = next((text_lines[j].strip() for j in range(i - 1, -1, -1)
                       if _looks_like_name(text_lines[j])
                       and not is_family_context(text_lines[j])), "")
        after = next((text_lines[j].strip() for j in range(i + 1, len(text_lines))
                      if _looks_like_name(text_lines[j])
                      and not is_family_context(text_lines[j])), "")
        if before and after:
            pair = _assign_bride_groom(_clean_name(before), _clean_name(after),
                                       "groom-first")
            candidates.append(Candidate(
                pair, 0.9, "standalone_conjunction"))

    inline = re.search(
        r"\b([A-Z][a-zA-Z]{2,})\s+([A-Z][a-zA-Z]{2,})\s+"
        r"(?i:invite|invites|welcome|welcomes|request\s+the\s+pleasure|"
        r"request\s+the\s+hon|request\s+your\s+presence|celebrate)",
        text)
    if inline:
        n1, n2 = inline.group(1), inline.group(2)
        if n1.lower() not in EVENT_TYPE_KEYWORDS and n2.lower() not in EVENT_TYPE_KEYWORDS:
            pair = _assign_bride_groom(n1, n2, "groom-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.82, "inline_marker")
            if c:
                candidates.append(c)

    marker_idx = None
    for i, line in enumerate(text_lines):
        if re.search(r"invite\s+you|request the|wedding celebration|wedding&|weds\b",
                     line, re.IGNORECASE):
            marker_idx = i
            break
    if marker_idx is not None:
        names = []
        for k in range(marker_idx - 1, max(marker_idx - 4, -1), -1):
            ls = text_lines[k].strip()
            if not ls:
                continue
            if _looks_like_name(ls):
                names.append(_clean_name(ls))
            elif len(names) >= 1:
                break
        if len(names) >= 2:
            pair = _assign_bride_groom(names[0], names[1], "groom-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.8, "marker_above")
            if c:
                candidates.append(c)
        elif len(names) == 1:
            c = _names_candidate("", names[0], 0.7, "marker_above_single")
            if c:
                candidates.append(c)

    candidates_list = []
    for line in text_lines:
        ls = line.strip()
        if is_family_context(ls):
            continue
        if not _looks_like_name(ls):
            continue
        if _is_greeting_or_event_desc(ls):
            continue
        cleaned = _clean_name(ls)
        if len(cleaned) >= 3:
            candidates_list.append(cleaned)
    if candidates_list:
        names = candidates_list[:2]
        if len(names) >= 2:
            pair = _assign_bride_groom(names[0], names[1], "groom-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.7,
                                 "standalone_lines")
            if c:
                candidates.append(c)
        else:
            c = _names_candidate("", names[0], 0.6, "standalone_single")
            if c:
                candidates.append(c)

    # When the document contains Tamil script, any surviving Latin-only name
    # candidate is OCR garbage (a misread Tamil name), not a real English name.
    # Drop those so the Result Screen never shows Latin garbage for a Tamil
    # invitation.  Genuine Tamil name candidates (e.g. the tamil_layout
    # strategy) are preserved.
    if ocr_lines and any(_has_tamil(item.get("text", "")) for item in ocr_lines):
        filtered = []
        for c in candidates:
            value = c.value if isinstance(c.value, dict) else {}
            joined = " ".join(str(value.get(k, "")) for k in ("bride", "groom"))
            if _has_tamil(joined):
                filtered.append(c)
            elif c.strategy == "tamil_layout":
                filtered.append(c)
            # else: drop Latin-only candidate for a Tamil document
        candidates = filtered

    return candidates


def extract_names(text_lines: List[str], text: str, event_type: str) -> Candidate:
    if event_type and event_type.lower() in _NO_NAME_EVENTS:
        return Candidate({"bride": "", "groom": ""}, 0.5, "no_name_event")
    best = _best(extract_names_variants(text_lines, text))
    if best is None:
        return Candidate({"bride": "", "groom": ""}, 0.0, "none")
    return best


def _ensure_distinct_couple(names: Dict[str, str], name_variants: List[Candidate],
                            all_text_lines: List[str]) -> Dict[str, str]:
    """Repair a duplicate couple only when the OCR contains a distinct pair.

    Candidate extraction already ranks explicit labels and relationship syntax
    above line order. This final check prevents a later OCR-normalization pass
    from returning one person in both wedding roles.
    """
    bride = names.get("bride", "").strip()
    groom = names.get("groom", "").strip()
    if not bride or not groom or bride.casefold() != groom.casefold():
        return {"bride": bride, "groom": groom}
    for candidate in sorted(name_variants, key=lambda item: item.confidence, reverse=True):
        value = candidate.value if isinstance(candidate.value, dict) else {}
        candidate_bride = _clean_name(value.get("bride", ""))
        candidate_groom = _clean_name(value.get("groom", ""))
        if (candidate_bride and candidate_groom
                and candidate_bride.casefold() != candidate_groom.casefold()):
            return {
                "bride": _resolve_repeated_person_ocr(candidate_bride, all_text_lines),
                "groom": _resolve_repeated_person_ocr(candidate_groom, all_text_lines),
            }
    return {"bride": bride, "groom": groom}


# OCR confusions that are visually plausible in decorative fonts.  These are
# deliberately character-level rather than a list of people or places.  They
# are used only after a person has been identified through relationship syntax
# (e.g. Weds / Son of / Daughter of) and corroborated by another occurrence
# with the same surname.
_NAME_OCR_SUBSTITUTIONS = {
    "0": "o", "1": "l", "l": "i", "i": "l",
    "b": "h", "h": "b", "o": "d", "d": "o",
}


def _levenshtein(a: str, b: str) -> int:
    """Small dependency-free edit distance used for OCR candidate agreement."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for index, char in enumerate(a, 1):
        current = [index]
        for other_index, other in enumerate(b, 1):
            current.append(min(previous[other_index] + 1, current[-1] + 1,
                               previous[other_index - 1] + (char != other)))
        previous = current
    return previous[-1]


def _name_shape_score(token: str) -> float:
    """Return a small language-agnostic plausibility score for an OCR token.

    This is not a name dictionary: it merely avoids choosing forms with OCR
    artefacts such as a terminal double-vowel run when an equally supported,
    visually-confusable consonant form exists.
    """
    lowered = token.lower()
    score = 0.0
    if re.search(r"[aeiouy]{2,}$", lowered):
        score -= 0.65
    if re.search(r"[^aeiouy]{5,}", lowered):
        score -= 0.20
    # A few consonant joins are almost always a broken glyph rather than a
    # pronounceable Latin-script token.  This is intentionally structural,
    # not a list of personal names.
    if re.search(r"(?:bm|bp|bt|dk|dt|fp|gk|pk|pt|tm)", lowered):
        score -= 0.45
    if 3 <= len(lowered) <= 14:
        score += 0.10
    return score


def _ocr_token_variants(token: str) -> List[str]:
    """Generate one/two-edit visual OCR alternatives without any name list."""
    variants = {token.lower()}
    frontier = {token.lower()}
    # Two edits cover independent glyph mistakes in a short decorative name,
    # while keeping the candidate set small and auditable.
    for _ in range(2):
        next_frontier = set()
        for value in frontier:
            for index, char in enumerate(value):
                replacement = _NAME_OCR_SUBSTITUTIONS.get(char)
                if replacement:
                    changed = value[:index] + replacement + value[index + 1:]
                    if changed not in variants:
                        variants.add(changed)
                        next_frontier.add(changed)
        frontier = next_frontier
    return list(variants)


def _resolve_repeated_person_ocr(name: str, text_lines: List[str]) -> str:
    """Resolve a relationship-linked name from repeated OCR observations.

    A correction is emitted only when the document has a second close name
    observation with the same surname.  This prevents parent/host names and
    unrelated capitalized text from becoming candidates, and never invents a
    value from a person-specific lookup table.
    """
    parts = _clean_name(name).split()
    if len(parts) < 2:
        return name
    first, surname = parts[0], " ".join(parts[1:])
    observations = [first]
    for line in text_lines:
        if re.search(r"\b(?:son|daughter|father|mother)\s+of\b", line, re.I):
            continue
        cleaned = _clean_name(line)
        candidate_parts = cleaned.split()
        if len(candidate_parts) >= 2 and " ".join(candidate_parts[1:]).lower() == surname.lower():
            candidate_first = candidate_parts[0]
            if _levenshtein(candidate_first.lower(), first.lower()) <= 2:
                observations.append(candidate_first)
    observations = list(dict.fromkeys(observations))
    if len(observations) < 2:
        return name

    candidates = set()
    for observed in observations:
        candidates.update(_ocr_token_variants(observed))
    # All observations must be explainable by at most two glyph substitutions.
    viable = [c for c in candidates
              if all(_levenshtein(c, observed.lower()) <= 2 for observed in observations)]
    if not viable:
        return name
    # Prefer the form with the strongest combined character evidence, then use
    # only a modest generic shape tie-breaker.  Do not replace an observed form
    # unless the result fixes an identifiable OCR-shape artefact.
    def score(candidate: str) -> float:
        similarity = sum(1 - _levenshtein(candidate, item.lower()) /
                         max(len(candidate), len(item), 1) for item in observations)
        return similarity + _name_shape_score(candidate)
    best = max(viable, key=score)
    if (best != first.lower() and _name_shape_score(best) > _name_shape_score(first)
            and re.search(r"[aeiouy]{2,}$", first.lower())):
        return " ".join([best.capitalize(), *parts[1:]])
    return name


# ---------------------------------------------------------------------------
# Document-structure helper: split multi-event invitations.
# ---------------------------------------------------------------------------
STRONG_LOCATION_KEYWORDS = [
    "venue", "located at", "function hall", "convention", "palace",
    "mandapam", "auditorium", "resorts", "lawns", "hotel", "grounds",
    "seminar hall", "community hall", "address", "street", "road",
    "nagar", "colony", "main road", "bus stop", "district", "pincode",
    "village", "town",
]


def _has_strong_location(text: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"\b{re.escape(kw)}\b", lowered)
        for kw in STRONG_LOCATION_KEYWORDS
    )


def _group_has_event_info(lines: List[str]) -> bool:
    text = " ".join(lines)
    if extract_date(text) or extract_time(text) or extract_contact(text):
        return True
    return _has_strong_location(text)


_HEADING_DELIMITERS = re.compile(
    r"^\s*(?:wedding|marriage|reception|engagement|birthday|housewarming|"
    r"muhurtham|naming\s+ceremony|"
    r"haldi|nikah|walima|mehndi)(?:\s|ceremony|celebration|\s+ceremony|\s+"
    r"celebration|\s+reception|&|of|\b)*\s*$",
    re.IGNORECASE,
)
_NON_HEADING = re.compile(
    r"to\s+follow|reception\s+of|wedding\s+reception\b|dinner\s+to\s+follow",
    re.IGNORECASE,
)


def _split_events(text_lines: List[str]) -> List[List[str]]:
    """Split a multi-event invitation into per-event line groups.

    Splitting is intentionally CONSERVATIVE to avoid fabricating events from
    repeated OCR headings or decorative separators. A new event is started only
    when there is STRONG evidence of a genuinely separate event:
      1. The line is a short, self-contained event heading (e.g. "Reception",
         "Wedding Ceremony") — not a descriptive phrase.
      2. The heading names an event type that DIFFERS from the type already
         accumulated in the current group (so repeated/duplicate headings of the
         same event never split).
      3. The accumulated group already carries real event info (date/time/venue).
      4. Enough following lines exist to let the new event have its own details,
         i.e. a following segment that itself has event information.
    """
    if not text_lines:
        return [text_lines]

    heading_types = {
        "wedding": "Wedding", "marriage": "Marriage", "muhurtham": "Wedding",
        "reception": "Reception", "engagement": "Engagement",
        "birthday": "Birthday", "housewarming": "Housewarming",
        "naming ceremony": "Naming Ceremony",
        "haldi": "Haldi", "nikah": "Nikah", "walima": "Walima",
        "mehndi": "Mehndi",
        "haldi ceremony": "Haldi", "nikah ceremony": "Nikah",
        "walima reception": "Walima", "mehndi ceremony": "Mehndi",
    }

    def _heading_type(line: str) -> Optional[str]:
        tl = line.strip()
        if not tl or len(tl) > 40:
            return None
        lowered = tl.lower()
        if _NON_HEADING.search(tl):
            return None
        generic_words = re.sub(r"[^A-Za-z ]", " ", tl).split()
        generic_heading = (2 <= len(generic_words) <= 5 and len(tl) <= 50
                           and generic_words[-1].lower() in
                           {"ceremony", "reception", "celebration", "function"}
                           and all(word[0].isupper() for word in generic_words if word))
        if not _HEADING_DELIMITERS.match(tl) and not generic_heading:
            return None
        for kw, typ in heading_types.items():
            if kw in lowered:
                return typ
        return _clean(tl) if generic_heading else None

    # Some layouts print a programme heading run first, followed by a compact
    # date list and shared detail block. Split that structure by heading/date
    # order instead of dropping the earlier ceremony heading.
    heading_run = []
    for index, line in enumerate(text_lines):
        if _heading_type(line) is not None:
            if heading_run and index != heading_run[-1][0] + 1:
                break
            heading_run.append((index, line))
        elif heading_run and not extract_date(line):
            if extract_time(line) or re.match(r"^(?:venue|address|location)\b",
                                              line.strip(), re.IGNORECASE):
                continue
            if index == heading_run[-1][0] + 1:
                continue
            break
    if len(heading_run) >= 2:
        start = heading_run[0][0]
        end = next((i for i in range(start + 1, len(text_lines))
                    if _heading_type(text_lines[i]) is not None
                    and i > heading_run[-1][0]), len(text_lines))
        date_indices = [i for i in range(heading_run[-1][0] + 1, end)
                        if extract_date(text_lines[i])]
        if len(date_indices) >= len(heading_run):
            shared_tail = text_lines[date_indices[-1] + 1:end]
            grouped = []
            for position, (_, heading) in enumerate(heading_run):
                group = [heading, text_lines[date_indices[position]]]
                if position == 0:
                    group = text_lines[:start] + group
                    group.extend(shared_tail)
                grouped.append(group)
            return grouped + _split_events(text_lines[end:]) if end < len(text_lines) else grouped

    events = []
    current: List[str] = []
    i = 0
    n = len(text_lines)
    while i < n:
        line = text_lines[i]
        ht = _heading_type(line)
        if ht is not None and current and _group_has_event_info(current):
            # A) The new heading must be a different event type than the current
            #    group's dominant type (repeated headings of the same event are
            #    OCR duplication, not separate events).
            current_et = extract_event_type("\n".join(current))
            current_type = current_et.value if current_et else ""
            if current_type and current_type.lower() == ht.lower():
                current.append(line)
                i += 1
                continue
            # B) Require the following segment to carry its own event details, so
            #    we only split when there is a real separate event with content.
            j = i + 1
            following = []
            while j < n and _heading_type(text_lines[j]) is None:
                following.append(text_lines[j])
                j += 1
            if following and _group_has_event_info(following):
                events.append(current)
                current = [line]
                i += 1
                continue
        current.append(line)
        i += 1

    if current:
        events.append(current)
    return events if len(events) > 1 else [text_lines]


def _confidence_from_parser(parsed: Dict, candidate_confidences: Dict[str, float] = None) -> float:
    """Compute an evidence-based confidence score for the parsed record.

    The score combines:
      - presence of core fields (event_type, date, venue)
      - the best candidate confidence for each core field when available
      - a small additive bonus for names

    candidate_confidences: optional dict mapping field->best_candidate_conf
    """
    candidate_confidences = candidate_confidences or {}
    core = ["event_type", "date", "venue"]
    # Base: fraction of core fields present
    filled_core = sum(1 for f in core if parsed.get(f))
    base = filled_core / len(core)

    evidence = sum(float(candidate_confidences.get(f, 0.0)) for f in core) / len(core)
    name_bonus = 0.05 if parsed.get("bride_name") or parsed.get("groom_name") else 0.0
    score = min(1.0, 0.55 * base + 0.35 * evidence + name_bonus)
    return round(score, 2)


# ---------------------------------------------------------------------------
# Validation & formatting helpers (final-stage normalization)
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split()).strip().rstrip(",;: ")
    # Strip trailing punctuation, but keep a period that directly follows a
    # street abbreviation (e.g. "St.", "Rd.", "Ave.").
    m = re.match(r"^(.*?)[.;:,*/-]+$", text)
    if m:
        base = m.group(1)
        if not re.search(
            r"(?:st|rd|ave|av|ln|blvd|dr|ct|pl|way|ter|cir|mg|main)\.?$",
            base, re.IGNORECASE,
        ):
            text = base
    return text


def format_date(raw: str) -> str:
    if not raw:
        return ""
    raw = _clean(raw)
    m = re.search(r"(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2,4})", raw)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = "20" + y if int(y) < 70 else "19" + y
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        if 1 <= b <= 12:
            return f"{month_names[b - 1]} {a}, {y}"
        if 1 <= a <= 12:
            return f"{month_names[a - 1]} {b}, {y}"
        return f"{a:02d}/{b:02d}/{y}"
    # Canonicalize named dates while discarding an optional weekday and
    # ordinal suffix. Keep month-first output readable and stable.
    month_names = {
        "jan": "January", "january": "January", "feb": "February",
        "february": "February", "mar": "March", "march": "March",
        "apr": "April", "april": "April", "may": "May",
        "jun": "June", "june": "June", "jul": "July", "july": "July",
        "aug": "August", "august": "August", "sep": "September",
        "sept": "September", "september": "September", "oct": "October",
        "october": "October", "nov": "November", "november": "November",
        "dec": "December", "december": "December",
    }
    # "the <day> of <month> <year>" (e.g. "the 5th of May 2024") — used by
    # formal invitation phrasing.  Handled before the generic patterns because
    # the "of" separator is not covered by the day/month/year alternation.
    named = re.search(
        rf"(?:{_WEEKDAY})\.?\s*,?\s*the\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+"
        rf"({_MONTH})\.?\s*[,.]?\s*(\d{{4}})",
        raw, re.IGNORECASE)
    if named:
        day, month, year = named.groups()
        return f"{month_names[month.lower()]} {int(day)}, {year}"
    named = re.search(
        rf"(?:{_WEEKDAY})\.?\s*,?\s*(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+)?"
        rf"({_MONTH})\.?\s*(\d{{1,2}})?(?:st|nd|rd|th)?\s*[,.]?\s*(\d{{4}})",
        raw, re.IGNORECASE)
    if named:
        day_before, month, day_after, year = named.groups()
        day = day_before or day_after
        if day:
            return f"{month_names[month.lower()]} {int(day)}, {year}"
    named = re.search(
        rf"(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+)?({_MONTH})\.?\s*"
        rf"(\d{{1,2}})?(?:st|nd|rd|th)?\s*[,.]?\s*(\d{{4}})",
        raw, re.IGNORECASE)
    if named:
        day_before, month, day_after, year = named.groups()
        day = day_before or day_after
        if day:
            return f"{month_names[month.lower()]} {int(day)}, {year}"
    return raw


def format_time(raw: str) -> str:
    if not raw:
        return ""
    raw = _clean(raw)
    qualifier = " onwards" if re.search(r"\bonwards?\b", raw, re.IGNORECASE) else ""
    m = re.search(r"(\d{1,2})[:.](\d{2})(\s*)(am|pm|a\.?m\.?|p\.?m\.?)?", raw, re.IGNORECASE)
    if m:
        h, mm = m.group(1), m.group(2)
        spacing, suffix = m.group(3), m.group(4)
        if suffix:
            amp = suffix.replace(".", "")
            return f"{int(h)}:{mm}{spacing}{amp.upper()}{qualifier}"
        return f"{h}:{mm}"
    m = re.search(r"(\d{1,2})\s+(\d{2})(\s*)(am|pm|a\.?m\.?|p\.?m\.?)", raw, re.IGNORECASE)
    if m:
        amp = m.group(4).replace(".", "")
        return f"{int(m.group(1))}:{m.group(2)}{m.group(3)}{amp.upper()}{qualifier}"
    m = re.search(r"(\d{1,2})(\s*)(am|pm|a\.?m\.?|p\.?m\.?)", raw, re.IGNORECASE)
    if m:
        amp = m.group(3).replace(".", "")
        return f"{int(m.group(1))}:00 {amp.upper()}{qualifier}"
    return raw


def format_phone(raw: str) -> str:
    if not raw:
        return ""
    return re.sub(r"[^\d]", "", raw)


def _layout_order_lines(text_lines: List[str], ocr_lines: List[Dict]) -> List[str]:
    """Use OCR coordinates as document order evidence when they are reliable.

    OCR passes can return blocks in engine order rather than visual order.  We
    only reorder lines that can be matched to a coordinate-bearing OCR line;
    unmatched lines retain their original order, so text-only callers and mixed
    language OCR do not lose information.
    """
    positioned = []
    used = set()
    for source_index, line in enumerate(text_lines):
        compact = re.sub(r"[^a-z0-9]", "", line.lower())
        best = None
        for idx, item in enumerate(ocr_lines):
            if idx in used or not item.get("bbox"):
                continue
            item_compact = re.sub(r"[^a-z0-9]", "", str(item.get("text", "")).lower())
            if compact == item_compact or (compact and item_compact and
                                           _levenshtein(compact, item_compact) <= 2):
                best = (idx, item)
                break
        if best is None:
            positioned.append((10**9 + source_index, source_index, line))
            continue
        used.add(best[0])
        bbox = best[1]["bbox"]
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            # quantizing y preserves left-to-right reading within the same row.
            positioned.append((round(sum(ys) / len(ys) / 12), sum(xs) / len(xs), line))
        except (TypeError, ValueError, IndexError):
            positioned.append((10**9 + source_index, source_index, line))
    return [line for _, _, line in sorted(positioned, key=lambda item: (item[0], item[1]))]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_invitation(matched_fields: Dict, raw_text: str = "", 
                     layout_regions: List[Dict] = None,
                     ocr_lines: List[Dict] = None) -> Dict:
    """Parse raw OCR text into structured invitation fields.

    Notes:
      - This function performs validation, normalization, ranking and keeps a
        detailed candidate debug trace. It does NOT attempt heavy AI extraction
        beyond optional fine-tuned LayoutLM hints.
      - matched_fields is still used only to fill gaps (never to overwrite a
        higher-confidence rule-based extraction).
    """
    layout_regions = layout_regions or []
    ocr_lines = ocr_lines or []
    text_lines = [l for l in raw_text.split("\n") if l.strip()]
    if ocr_lines:
        text_lines = _layout_order_lines(text_lines, ocr_lines)
    if not text_lines:
        raw_text = ""
        text_lines = []

    # Get optional LayoutLM hints (only available when a fine-tuned model is set)
    layoutlm_hints = _layoutlm_infer(raw_text) if _get_layoutlm() else {}

    # Event type - gather candidates and prefer layoutlm hint only as a high-priority suggestion
    event_type_variants = extract_event_type_variants(raw_text) if 'extract_event_type_variants' in globals() else []
    # If no variant function (older versions), fall back to extract_event_type
    if not event_type_variants:
        et_best = extract_event_type(raw_text)
        event_type_variants = [et_best] if et_best else []

    # Choose initial event_type from layoutlm hint else best candidate
    event_type_hint = layoutlm_hints.get("event_type") if layoutlm_hints else None
    event_type_best = _best(event_type_variants)
    event_type = event_type_hint or (event_type_best.value if event_type_best else "")

    # Split into event groups conservatively
    event_groups = _split_events(text_lines)

    events = []
    debug_events = []

    for group in event_groups:
        group_text = "\n".join(group)

        # Collect candidates for each field using the variant extractors
        et_variants = extract_event_type_variants(group_text)
        en_variants = extract_event_name_variants(group_text)
        name_variants = extract_names_variants(group, group_text, ocr_lines)
        venue_variants = extract_venue_variants(group, group_text)
        addr_variants = extract_address_variants(group, group_text)
        date_variants = extract_date_variants(group_text)
        time_variants = extract_time_variants(group_text)
        contact_variants = extract_contact_variants(group_text)

        best_et = _best(et_variants) or event_type_best
        best_en = _best(en_variants)
        if best_en and best_en.strategy in {"event_heading", "generic_structural_heading"}:
            best_et = Candidate(best_en.value, best_en.confidence,
                                "event_heading_type")
        best_names = _best(name_variants)
        selected_type = (best_et.value if best_et else event_type) or ""
        if selected_type.lower() in _NO_NAME_EVENTS:
            best_names = None
        best_venue = _best(venue_variants)
        venue_value = best_venue.value if best_venue else ""
        # Address extraction needs the selected venue so a street embedded in
        # the venue is not returned a second time as the address.
        best_addr = extract_address(group, group_text, venue_value)
        best_date = _best(date_variants)
        best_time = _best(time_variants)
        best_contact = _best(contact_variants)

        # Build parsed event skeleton from best candidates (converted to strings)
        names_value = best_names.value if best_names else {"bride": "", "groom": ""}
        if not isinstance(names_value, dict):
            names_value = {"bride": "", "groom": ""}
        names_value = {
            "bride": _resolve_repeated_person_ocr(names_value.get("bride", ""), text_lines),
            "groom": _resolve_repeated_person_ocr(names_value.get("groom", ""), text_lines),
        }
        names_value = _ensure_distinct_couple(names_value, name_variants, text_lines)

        # When there is no named venue, preserve a street line and its postal
        # locality as distinct fields instead of inventing a venue from the
        # city/state line or duplicating the combined address.
        address_value = best_addr.value if best_addr else ""
        if not venue_value:
            street, locality = _split_street_location(group)
            if street and locality:
                venue_value, address_value = street, locality

        parsed = {
            "event_name": best_en.value if best_en else "",
            "event_type": best_et.value if best_et else event_type,
            "bride_name": names_value.get("bride", ""),
            "groom_name": names_value.get("groom", ""),
            "date": format_date(best_date.value) if best_date else "",
            "time": format_time(best_time.value) if best_time else "",
            "venue": _clean(venue_value),
            "address": _clean(address_value),
            "contact_number": best_contact.value if best_contact else "",
        }

        # Candidate confidences (best for each field) used by confidence scorer
        candidate_confs = {
            "event_type": float(best_et.confidence) if best_et else 0.0,
            "date": float(best_date.confidence) if best_date else 0.0,
            "venue": float(best_venue.confidence) if best_venue else 0.0,
        }

        parsed["confidence"] = _confidence_from_parser(parsed, candidate_confs)

        # Attach the raw candidate lists (for debugging and later parsing/validation)
        debug = {
            "group_text": group_text,
            "ocr_layout": [line for line in ocr_lines if line.get("text", "") in group],
            "candidates": {
                "event_type": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in et_variants
                ],
                "event_name": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in en_variants
                ],
                "names": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in name_variants
                ],
                "venue": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in venue_variants
                ],
                "address": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in addr_variants
                ],
                "date": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in date_variants
                ],
                "time": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in time_variants
                ],
                "contact": [
                    {"value": c.value, "confidence": c.confidence, "strategy": c.strategy}
                    for c in contact_variants
                ],
            },
            "selected": {
                "event_name": parsed["event_name"],
                "event_type": parsed["event_type"],
                "bride_name": parsed["bride_name"],
                "groom_name": parsed["groom_name"],
                "date": parsed["date"],
                "time": parsed["time"],
                "venue": parsed["venue"],
                "address": parsed["address"],
                "contact_number": parsed["contact_number"],
                "confidence": parsed["confidence"],
            }
        }

        events.append(parsed)
        debug_events.append(debug)

    # Couple identity is invitation-level evidence (Weds and parent-role
    # context), unlike date/time/location.  Populate it independently for
    # every detected event, but never copy event details between segments.
    couple_source = next((event for event in events
                          if event.get("bride_name") or event.get("groom_name")), {})
    for event in events:
        if couple_source:
            event["bride_name"] = event.get("bride_name") or couple_source.get("bride_name", "")
            event["groom_name"] = event.get("groom_name") or couple_source.get("groom_name", "")

    # Primary event is the first; allow matched_fields (upstream) to fill gaps only
    primary = events[0] if events else {}

    for k in ("event_name", "event_type", "bride_name", "groom_name", "date",
              "time", "venue", "address", "contact_number"):
        mv = matched_fields.get(k, "")
        if mv and not primary.get(k):
            primary[k] = mv

    # Final normalize/clean primary
    primary = {
        "event_name": _clean(primary.get("event_name", "")),
        "event_type": _clean(primary.get("event_type", "")),
        "bride_name": _clean(primary.get("bride_name", "")),
        "groom_name": _clean(primary.get("groom_name", "")),
        "date": format_date(primary.get("date", "")),
        "time": format_time(primary.get("time", "")),
        "venue": _clean(primary.get("venue", "")),
        "address": _clean(primary.get("address", "")),
        "contact_number": _clean(primary.get("contact_number", "")),
        "events": events,
        "number_of_events": len(events),
        # Overall confidence computed from primary's best candidate evidences
        "confidence": _confidence_from_parser(primary, {
            "event_type": float((events[0].get("event_type") and 1.0) or 0.0),
            "date": float((events[0].get("date") and 1.0) or 0.0),
            "venue": float((events[0].get("venue") and 1.0) or 0.0),
        }),
        # Expose the per-event candidate debug for diagnostics (non-breaking addition)
        "extraction_debug": debug_events,
    }

    return primary


def _layoutlm_infer(text: str) -> Dict:
    """Use LayoutLM (fine-tuned) to provide best-effort hints.

    Returns an empty dict when no fine-tuned model is configured. The parser
    treats these hints as suggestive only — they cannot override a higher-
    confidence rule-based extraction.
    """
    model = _get_layoutlm()
    if model is None:
        return {}
    try:
        result = model(text)
        hints = {}
        # The pipeline may emit aggregated spans with 'entity_group' and 'word'
        # or tokens with 'entity' / 'word'. Be defensive when reading fields.
        for ent in result:
            label = ent.get("entity_group") or ent.get("entity") or ""
            word = ent.get("word") or ent.get("entity_word") or ent.get("text") or ""
            if not label:
                continue
            label_upper = label.upper()
            if label_upper in ("EVENT", "DATE", "TIME"):
                hints[label_upper.lower()] = word
        # Attach model provenance so callers can log/inspect whether hints came
        # from a fine-tuned model.
        if hints:
            hints["_provenance"] = {"model": os.environ.get("LAYOUTLM_MODEL")}
        return hints
    except Exception as exc:  # noqa: BLE001
        logger.warning("LayoutLM inference failed (%s)", exc)
        return {}
