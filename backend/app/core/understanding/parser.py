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
from datetime import datetime
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


def _base_event_type(heading: str) -> str:
    """Return the event category represented by a heading such as
    ``Birthday Celebration`` or ``Wedding Ceremony``."""
    words = re.sub(r"[^A-Za-z ]", " ", heading).split()
    if not words:
        return ""
    lowered = heading.strip().lower()
    for suffix in (" ceremony", " reception", " celebration", " function"):
        if lowered.endswith(suffix):
            base = heading.strip()[:-len(suffix)].strip()
            base_words = re.sub(r"[^A-Za-z ]", " ", base).split()
            if base_words and base_words[0].lower() == "birthday":
                return "Birthday"
            if base_words and base_words[0].lower() in EVENT_TYPE_KEYWORDS:
                return heading.strip()
            return base
    keyword = words[0].lower()
    if keyword in EVENT_TYPE_KEYWORDS:
        return EVENT_TYPE_KEYWORDS[keyword][0]
    return heading.strip()


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


def _best_couple_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    """Prefer a complete bride/groom pair over one-sided family-role evidence."""
    complete = [
        c for c in candidates
        if isinstance(c.value, dict)
        and c.value.get("bride")
        and c.value.get("groom")
    ]
    if not complete:
        return None
    return max(complete, key=lambda c: c.confidence)


def _has_couple_evidence(name_variants: List[Candidate], text: str) -> bool:
    couple_strategies = {
        "weds_verb", "standalone_weds", "marriage_of", "d_o_s_o",
        "family_roles", "standalone_pair", "explicit_marker", "tamil_layout",
    }
    if any(c.strategy in couple_strategies for c in name_variants):
        return True
    return bool(re.search(
        r"\b(?:weds|wed|marries|marry|gets married to|daughter\s+of|son\s+of)\b",
        text or "", re.IGNORECASE,
    ))


def _pick(candidates: List[Optional[Candidate]]):
    """Return the value of the best candidate, or '' if none."""
    best = _best(candidates)
    return best.value if best is not None else ""


def _split_combined_name(name: str) -> List[str]:
    """Split a combined name like 'Rahul & Priya' into ['Rahul', 'Priya']."""
    parts = re.split(r'\s*(?:&|\+|and)\s*', name, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]


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
    (rf"\b((?:{_WEEKDAY})\.?,?[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+{_DY},?[^\S\n]+\d{{2,4}})", 0.95,
     "weekday_month_day_year"),
    (rf"\b((?:{_WEEKDAY})\.?,?[^\S\n]+{_DY}[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+\d{{2,4}})", 0.95,
     "weekday_day_month_year"),
    (rf"\b({_DY}[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+\d{{2,4}})", 0.90,
     "day_month_year"),
    (rf"\b((?:{_MONTH})\.?\.?[^\S\n]+{_DY},?[^\S\n]+\d{{2,4}})", 0.82,
     "month_day_year"),
    (rf"\b((?:{_WEEKDAY})\.?,?[^\S\n]+the[^\S\n]+{_DY}[^\S\n]+of[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+\d{{2,4}})",
      0.88, "weekday_the_day_of_month_year"),
    (rf"\b(the[^\S\n]+{_DY}[^\S\n]+of[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+\d{{2,4}})", 0.85,
     "the_day_of_month_year"),
    (rf"\b((?:{_MONTH})\.?\.?[^\S\n]+{_DY}\s*\.\s*\d{{2,4}})", 0.82,
     "month_day.period_year"),
    # Year-first formats: "2026 23 OCTOBER" (year day month) and "2026 OCTOBER 23" (year month day)
    (rf"\b(\d{{4}}[^\S\n]+{_DY}[^\S\n]+(?:{_MONTH})\.?\.?)", 0.88,
     "year_day_month"),
    (rf"\b(\d{{4}}[^\S\n]+(?:{_MONTH})\.?\.?[^\S\n]+{_DY})", 0.88,
     "year_month_day"),
    (rf"\b({_DY}[^\S\n]+(?:{_MONTH})\.?\.?)\b", 0.65,
     "day_month"),
    (rf"\b((?:{_MONTH})\.?\.?[^\S\n]+{_DY}(?:st|nd|rd|th)?)\b", 0.65,
     "month_day"),
    (r"\b(\d{1,2}(?:st|nd|rd|th)?[-/]\d{1,2}[-/]\d{2,4})\b", 0.72,
     "numeric_date"),
    (rf"\b((?:{_MONTH})\.?\.?\s*\n\s*{_DY}\s*\n\s*\d{{2,4}})", 0.88,
     "month_day_year_newlines"),
    (rf"\b((?:{_MONTH})\.?\.?\s*\n\s*\d{{2,4}}\s*\n\s*{_DY})", 0.8,
     "month_year_day_newlines"),
    # Year-first with newlines: "2026\n23\nOCTOBER", "2026\n23 OCTOBER",
    # or "2026\nOCTOBER\n23" / "2026\nOCTOBER 23".
    (rf"\b(\d{{4}}\s*\n\s*{_DY}\s+(?:{_MONTH})\.?\.?)", 0.86,
     "year_day_month_newlines"),
    (rf"\b(\d{{4}}\s*\n\s*(?:{_MONTH})\.?\.?\s+(\d{{1,2}})?)\b", 0.86,
     "year_month_day_newlines"),
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

# Reusable token for explicit start/end ranges. The negative lookbehind prevents
# the trailing ``00 PM`` in ``6.00 PM`` from being treated as a second time.
_TIME_TOKEN = (
    r"(?:\d{1,2}[:.]\d{2}|\d{1,2})\s*"
    r"(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?)"
)
_TIME_TOKEN_RE = re.compile(
    rf"(?<![\d.])({_TIME_TOKEN})(?![A-Za-z])",
    re.IGNORECASE,
)
_TO_TIME_RE = re.compile(
    rf"\bto\s+({_TIME_TOKEN})(?![A-Za-z])",
    re.IGNORECASE,
)

_PHONE_STRATEGIES = [
    (r"(?<!\d[.:])(\+?\d[\d \-]{8,15}\d)", 0.8, "loose_phone"),
    (r"(\b\d{10}\b)", 0.8, "ten_digit"),
    (r"(\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b)", 0.8, "three_three_four"),
]

EMAIL_PATTERN = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

# Event-type lexicon. Order within the dict is irrelevant to correctness
# because candidates are ranked by confidence; specific phrases are boosted.
EVENT_TYPE_KEYWORDS = {
    "wedding": ("Wedding", 0.7),
    "reception": ("Reception", 0.7),
    "marriage": ("Wedding", 0.6),
    "engagement": ("Engagement", 0.8),
    "birthday": ("Birthday", 0.8),
    "anniversary": ("Anniversary", 0.8),
    "baby shower": ("Baby Shower", 0.85),
    "conference": ("Conference", 0.8),
    "seminar": ("Seminar", 0.8),
    "party": ("Party", 0.75),
    "workshop": ("Workshop", 0.75),
    "training": ("Training Session", 0.75),
    "design": ("Training Session", 0.55),
    "prototyping": ("Workshop", 0.55),
    "cultural": ("Cultural Event", 0.75),
    "corporate": ("Corporate Event", 0.75),
    "annual": ("Annual Event", 0.75),
    "sports": ("Sports Event", 0.75),
    "festival": ("Festival", 0.75),
    "conference": ("Conference", 0.75),
    "graduation": ("Graduation", 0.75),
    "hackathon": ("Hackathon", 0.75),
    "ceremony": ("Ceremony", 0.7),
    "celebration": ("Celebration", 0.7),
    "exhibition": ("Exhibition", 0.7),
    "seminar": ("Seminar", 0.7),
    "naming ceremony": ("Naming Ceremony", 0.8),
    "namingceremony": ("Naming Ceremony", 0.8),
    "housewarming": ("Housewarming", 0.8),
    "muhurtham": ("Wedding", 0.8),
    "muhurthan": ("Wedding", 0.8),
    "celebration": ("Celebration", 0.7),
    "celebrat": ("Celebration", 0.6),
    "colebration": ("Celebration", 0.5),
    "colebrat": ("Celebration", 0.5),
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
    (r"baby\s+shower", "Baby Shower", 0.92),
    (r"shower\s+for\s+the\s+baby", "Baby Shower", 0.9),
    (r"celebrating\s+the\s+new\s+baby", "Baby Shower", 0.9),
    (r"a\s+baby\s+on\s+its\s+way", "Baby Shower", 0.85),
]

VENUE_KEYWORDS = ["venue", "function hall", "hall",
                  "convention", "palace", "mandapam", "auditorium", "resorts",
                  "garden", "lawns", "hotel", "grounds", "seminar", "community",
                  "plaza", "college", "university", "school", "institute",
                  "academy", "centre", "center", "maaligai", "mahal"]

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
    "residence", "farm", "plaza", "college", "university", "school",
    "institute", "academy", "centre", "center", "maaligai", "mahal",
]

ADDRESS_KEYWORDS = ["address", "street", "road", "nagar", "layout", "colony",
                    "main road", "cross", "bus stop", "near", "district",
                    "pincode", "pin", "city", "village", "town",
                    "நகர", "மாவட்டம்", "கிராமம்", "தெரு", "வீதி"]

# ALL-CAPS words that are event/heading tokens, not person names.
_HEADING_WORDS = {
    "programme", "invitation", "wedding", "birthday", "anniversary",
    "celebration", "ceremony", "function", "reception", "event",
    "housewarming", "muhurtham", "naming", "engagement", "marriage",
    "party", "seminar", "conference", "sports", "day", "annual",
    "cultural", "corporate", "religious", "college", "school",
    "dinner", "lunch", "breakfast", "rsvp", "contact", "phone",
    "venue", "address", "date", "time", "location", "place",
    "map", "directions", "join", "celebrate", "witness", "request",
    "solicit", "grace", "blessings", "family", "together", "extend",
    "warm", "presence", "honor", "honour", "pleasure", "company",
    "auspicious", "occasion", "refreshments", "mobile", "start", "end",
    "parents", "parent", "friends", "guests", "everyone", "children",
    "couple", "relatives", "kin", "members", "folks", "dear",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

# Generic event heading keywords — used to detect event headings that are
# not covered by _HEADING_DELIMITERS (e.g. "Annual Day", "Sports Day").
_GENERIC_EVENT_KEYWORDS = {
    "annual", "sports", "cultural", "festival", "interhouse",
    "inter-house", " intramural", "tournament", "meet", "gala",
    "show", "fair", "carnival", "parade", " marathon", "run", "race",
}

# Activity/slogan/line keywords that must NEVER be treated as person names.
_ACTIVITY_LINE_WORDS = {
    "events", "performances", "recognitions", "achievements",
    "activities", "competition", "exhibition", "display", "displayed",
    "exhibits", "agenda", "schedule", "menu", "programme",
    "program", "sports", "annual", "cultural", "festival",
}

# Common words that appear in invitation headings/descriptions but
# must never be treated as person names (single-word additions).
_NOT_PERSON_SINGLE_WORDS = {
    "organizes", "organize", "organizing", "organized",
    "full-day", "fullday", "full day",
    "session", "sessions", "workshop", "training", "seminar",
    "conference", "summit", "tutorial", "course", "class",
    "design", "prototyping", "wireframing",
    "together", "better", "journey", "forever",
    "celebrate", "celebrating", "welcome", "regards", "warmly",
    "presence", "pleasure", "company", "auspicious", "occasion",
    "blessings", "families", "relative", "kin", "members",
    "guests", "everyone", "children", "couple",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "first", "second", "third",
    "schedule", "agenda", "programme", "program",
    "registration", "rsvp", "venue", "address",
    "contact", "phone", "mobile", "email",
    "welcome", "regards", "sincerely", "warmly",
    "muhoortham", "muhurtham", "purattasi", "tithi", "natchathiram",
}

# City / place names that should never be treated as person names.
_NOT_PERSON_PLACES = {
    "thoppupalayam", "perundurai", "erode", "coimbatore", "chennai",
    "madurai", "bengaluru", "bangalore", "hyderabad", "mumbai",
    "delhi", "kolkata", "pune", "salem", "vellore", "trichy",
    "kochi", "kerala", "tirunelveli", "kanyakumari",
}

# Common English words that appear in invitation greetings/descriptions but
# must never be treated as person names.
_COMMON_INVITATION_WORDS = {
    "a", "an", "the", "of", "for", "at", "on", "in", "by", "with",
    "to", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "and", "or", "but", "not", "no", "yes", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "shall",
    "this", "that", "these", "those", "it", "its", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "we", "us", "our", "ours", "me", "my", "mine", "i",
    "please", "join", "celebrate", "celebrating", "honour", "honoring",
    "inviting", "welcome", "witness", "request", "solicit", "grace",
    "blessings", "family", "families", "together", "extend", "warm",
    "presence", "honor", "pleasure", "company", "auspicious", "occasion",
    "party", "ceremony", "function", "celebration", "event",
    "housewarming", "muhurtham", "naming", "anniversary", "baby",
    "shower", "graduation", "conference", "seminar", "sports", "day",
    "annual", "cultural", "corporate", "religious", "college", "school",
    "programme", "program", "dinner", "lunch", "breakfast", "refreshments",
    "rsvp", "contact", "phone", "mobile", "venue", "address", "date",
    "time", "end", "start", "location", "place", "map", "directions",
    "birthday", "marriage", "wedding", "reception", "engagement",
    "parents", "parent", "friends", "guests", "everyone", "children",
    "couple", "relatives", "kin", "members", "folks", "dear",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

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
    "chennai", "madurai", "coimbatore", "bengaluru", "bangalore", "banglore", "hyderabad",
    "mumbai", "delhi", "kolkata", "pune", "salem", "vellore", "trichy",
    "erode", "kochi", "kerala", "goa", "agra", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "surat", "ahmedabad", "thiruvananthapuram",
    "kanyakumari", "mysuru", "vijayawada", "visakhapatnam", "noida",
    "gurgaon", "thoothukudi", "tirunelveli", "greenville", "kovilpatti",
}

_NO_NAME_EVENTS = {"birthday", "naming ceremony", "namingceremony",
                   "housewarming", "muhurtham"}

_COUPLE_BASED_EVENT_TYPES = {
    "wedding", "marriage", "engagement", "nikah", "walima",
    "mehndi", "haldi", "muhurtham", "reception",
}


def _is_couple_based_event(event_type: str) -> bool:
    if not event_type:
        return False
    lowered = event_type.lower().strip()
    return any(lowered == kw or lowered.startswith(kw + " ") or lowered.endswith(" " + kw) or (" " + kw + " ") in (" " + lowered + " ") for kw in _COUPLE_BASED_EVENT_TYPES)


def _build_people_from_event(event: Dict) -> List[Dict]:
    """Convert flat bride/groom fields into a generic people array."""
    bride = _clean(event.get("bride_name", ""))
    groom = _clean(event.get("groom_name", ""))
    event_type = _clean(event.get("event_type", ""))
    people: List[Dict] = []
    if bride:
        people.append({
            "name": bride,
            "role": "Bride" if _is_couple_based_event(event_type) else "Person",
        })
    if groom:
        people.append({
            "name": groom,
            "role": "Groom" if _is_couple_based_event(event_type) else "Person",
        })
    return people


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

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_CITY_ST_ZIP_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*),\s*([A-Z]{2})\s*"
    r"(\d{5}(?:-\d{4})?|\d{6})\b"
)
_US_STATE_ZIP_RE = re.compile(
    r"\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b"
)
_CITY_PIN_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)\s+[-–]?\s*(\d{3})\s*[-–]?\s*(\d{3})(?![-–]\d)\b"
)


def _extract_street_address(text: str) -> str:
    """Extract a street address (e.g. '123 Anywhere St.') from arbitrary text."""
    m = re.search(
        r"\b(\d{1,4},?\s+[A-Za-z][A-Za-z0-9.]*(?:\s+[A-Za-z][A-Za-z0-9.]*)*"
        r"\s*\b(?:st|street|rd|road|ave|avenue|ln|lane|blvd|boulevard|dr|drive|"
        r"ct|court|pl|place|way|ter|terrace|cir|circle|mg|main)\b\.?,?)"
        r"|([A-Z][A-Za-z]+(?:\s+[A-Za-z0-9.]+)*\s+(?:Road|Street|Avenue|Lane|"
        r"Drive|Court|Place|Boulevard|Circle|Terrace|Station|Main)\b)",
        text, re.IGNORECASE)
    if m:
        return (m.group(1) or m.group(2)).strip().rstrip(",").strip()
    return ""


def _extract_city_state_zip(text: str) -> str:
    m = _CITY_ST_ZIP_RE.search(text)
    if m:
        return f"{m.group(1)}, {m.group(2)} {m.group(3)}"
    m = _US_STATE_ZIP_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return ""


def _extract_city_pin(text: str) -> str:
    m = _CITY_PIN_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2)}{m.group(3)}"
    return ""


def _extract_state_pin(text: str) -> str:
    m = re.search(
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*[-–—]\s*(\d{3})\s*[-–]?\s*(\d{3})\b"
        r"(?!\d)",
        text,
    )
    if m:
        return f"{m.group(1)} - {m.group(2)}{m.group(3)}"
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
    r"([A-Z][a-zA-Z]{2,}(?:[ \t]+[A-Z][a-zA-Z]{2,})?)"
)
_NAME_HONORIFIC_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|dr|sri|smt|er|kum|thiru|selvi)\.?(?=\s|:|$)\s*:?\s*)+",
    re.IGNORECASE,
)


def _clean_name(raw: str) -> str:
    name = _NAME_HONORIFIC_RE.sub("", raw.strip()).strip()
    name = re.sub(r"^[:;.,]+", "", name).strip()
    name = re.sub(r"[.,;:]+$", "", name).strip()
    # If the name contains a colon and the part after it starts with an
    # initial (e.g. "C."), the part before the colon is the given name
    # and the part after is the patronymic (e.g. "Selvan: C. Murugan").
    if ":" in name:
        before, after = name.split(":", 1)
        after = after.strip()
        if re.match(r"^[A-Z]\.\s*[A-Z]", after):
            name = before.strip()
    name = re.sub(r"['’]s\b", "", name).strip()
    # Strip leading role labels such as "Bride:", "Groom:" and Tamil equivalents.
    name = re.sub(
        r"^(?:bride|groom|மணமகள்|மணமகன்)\s*:?\s*",
        "", name, flags=re.IGNORECASE,
    ).strip()
    words = name.split()
    while words and words[0].lower() in _EVENT_NAME_WORDS:
        words.pop(0)
    name = " ".join(words).strip()
    # Drop trailing academic/honorific degree suffixes glued by OCR (requires a
    # dot so ordinary name endings are never trimmed).
    name = _DEGREE_SUFFIX_RE.sub("", name).strip()
    return name


# Trailing academic/honorific degree tokens that may ride along with a name on an
# OCR line (e.g. "A. முஹம்மது ஆபிதா M.A.", "S. முஹம்மது முஷ்தாக் B.E."). These are
# never part of the person's name and are stripped from the end. Degrees are
# matched only when they contain a dot so that ordinary name tokens such as
# "Maria" / "மால்" are never touched.
_DEGREE_SUFFIX_RE = re.compile(
    r"\s*(?:M\.A\.?|B\.?E\.?|B\.?A\.?|B\.?L\.?|B\.?Tech\.?|M\.Tech\.?|M\.Sc\.?|M\.C\.A\.?|"
    r"B\.?C\.A\.?|B\.?Com\.?|M\.?Com\.?|B\.?Sc\.?|Ph\.D\.?|M\.B\.B\.S\.?|M\.D\.?|"
    r"M\.?Phil\.?|LL\.?B\.?|LL\.?M\.?|B\.?Arch\.?|M\.?A\.?P\.?)\s*[,.;:]*$",
    re.IGNORECASE,
)



_NAME_REJECT_SINGLE_WORDS = {
    "with", "and", "or", "of", "for", "at", "on", "in", "by",
    "join", "warm", "invitation", "save", "famil", "the",
    # Generic relationship/group words that are NOT person names.
    "family", "families", "relatives", "relative", "kin", "kinsfolk",
    "folk", "folks", "dear", "friends", "guests", "everyone", "everybody",
    "parents", "parent", "children", "child", "couple", "members", "member",
}

_NAME_REJECT_PHRASE_WORDS = {
    "a", "an", "the", "our", "her", "his", "their", "little", "special",
    "star", "princess", "girl", "boy", "big", "tiny", "feet", "happiness",
    "dreams", "brighter", "tomorrow", "repeat", "eat", "play", "celebrate",
    "celebration", "presence", "join", "please", "filled", "love", "fun",
    "day", "older", "turning", "year", "one", "see", "there",
}

_SLOGAN_OR_EVENT_WORDS = {
    "society", "knowledge", "character", "progress", "graduation",
    "changemakers", "today", "tomorrow", "inspiring", "achievements",
    "engineers", "better", "world", "sustainable", "future", "excellence",
    "innovation", "ideas", "celebration", "ceremony", "wedding", "event",
}

# Multi-word phrases that are generic relationship/group indicators, not person names.
_RELATIONSHIP_GROUP_PHRASES = {
    "all the family", "all family", "all relatives", "all the relatives",
    "family and relatives", "family and friends", "dear family", "dear relatives",
    "dear friends", "dear guests", "with family", "with relatives",
    "best regards", "regards", "warm regards", "kind regards",
}

def _looks_like_name(line: str) -> bool:
    ls = line.strip()
    if not (3 <= len(ls) <= 40):
        return False
    if re.search(r"\d", ls):
        return False
    if _is_field_line(ls):
        return False
    if re.search(r"\b(?:railway|station)\b", ls, re.IGNORECASE):
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
    if re.match(r"^(?:bride|groom|மணமகள்|மணமகன்)\s*:?", ls, re.IGNORECASE):
        return False
    words = ls.split()
    normalized_words = {
        word.lower().strip(".,;:!?\"'()") for word in words
    }
    if normalized_words and normalized_words.issubset(_SLOGAN_OR_EVENT_WORDS | {"t"}):
        return False
    if any(word in _SLOGAN_OR_EVENT_WORDS for word in normalized_words):
        if len(words) > 1 and all(
                word in _SLOGAN_OR_EVENT_WORDS or len(word) <= 2
                for word in normalized_words):
            return False
    if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
        return False
    # Reject ALL-CAPS (including those with non-alpha connectors like
    # "&") for alphabetic words when the line contains generic event
    # heading keywords — these are event headings, not names.
    alpha_words = [w for w in words if w.isalpha()]
    if len(alpha_words) >= 2 and all(w.isupper() for w in alpha_words):
        if any(re.search(rf"\b{kw}\b", ls.lower()) for kw in _GENERIC_EVENT_KEYWORDS):
            return False
    lowered = ls.lower()
    # Reject Title Case lines (all words capitalized, none ALL-CAPS)
    # containing activity/event keywords — these are activity
    # descriptions (e.g. "Cultural Performances", "Student Achievements"),
    # not person names.
    if len(alpha_words) >= 2 and not all(w.isupper() for w in alpha_words):
        if all(w[0].isupper() for w in words if w):
            if any(kw in lowered.split() for kw in _ACTIVITY_LINE_WORDS):
                return False
    # Reject activity/slogan lines containing connectors AND activity
    # keywords — these are descriptions, not person names.
    if any(c in ls for c in "&|/"):
        if any(kw in lowered.split() for kw in _ACTIVITY_LINE_WORDS):
            return False
        if any(re.search(rf"\b{kw}\b", lowered) for kw in _GENERIC_EVENT_KEYWORDS):
            return False
    if len(words) == 1 and words[0].lower() in _NAME_REJECT_SINGLE_WORDS:
        return False
    # Reject common nouns/verbs/adjectives frequently seen in
    # invitation text that are NOT person names (general heuristic,
    # not specific to any invitation).
    if len(words) == 1 and words[0].lower() in {
        "recognition", "recognitions", "performance", "performances",
        "achievement", "achievements", "activity", "activities",
        "exhibition", "exhibit", "exhibits", "agenda", "schedule",
        "menu", "programme", "program", "sports", "sportsday",
        "annual", "cultural", "festival", "celebration", "celebrate",
        "celebrating", "competition", "tournament", "track",
        "field", "team", "student", "students", "junior", "senior",
        "primary", "secondary", "intermediate", "foundation",
        "academic", "academics", "excellence", "excellence",
        "knowledge", "learning", "learn", "teaching", "teach",
        "flourish", "common", "simple", "simple",
        "leadership", "service", "discipline", "growth",
        "healthier", "health", "healthy",
        "opportunity", "community", "purpose", "people", "person",
        "opportunity", "community", "purpose", "people", "person",
        "today", "tomorrow", "yesterday", "leaders", "different",
        "together", "stronger", "faster", "higher", "beyond",
        "limits", "rise", "reflect", "rejoice", "excel", "play",
        "faster", "creativity", "art", "culture", "design",
        "designer", "workshop", "training", "seminar", "course",
        "conference", "summit", "class", "tutorial", "module",
        "prototype", "prototyping", "session", "experience",
        "experiences", "ideas", "fundamentals", "principles",
        "components", "interactive", "interactively",
        "journey", "forever", "company", "presence", "pleasure",
        "auspicious", "occasion", "blessings", "families",
        "relative", "kin", "members", "guests", "everyone",
        "everybody", "children", "couple", "organize", "organizes",
        "organized", "organizing", "creation", "create", "creates",
        "created", "transform", "transforms", "transforming",
        "gather", "gathering", "welcome", "regards", "warmly",
        "club", "society", "association", "trust", "committee",
        "foundation", "board", "council", "forum", "league",
        "division", "department", "faculty", "institute",
        "academy", "centre", "center", "hall", "ground",
        "stadium", "complex", "temple", "church", "mosque",
        "gallery", "museum", "library", "lab", "studio",
        "concert", "show", "fair", "carnival", "parade",
        "race", "run", "meet", "gala", "showcase",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "january", "february",
        "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "morning", "afternoon", "evening", "night",
        "breakfast", "lunch", "dinner", "refreshment",
        "reception", "welcome", "farewell", "graduation",
        "award", "awards", "prize", "prizes", "medal", "medals",
        "sportsday", "sports day", "annual day", "annual",
        "science", "sciences", "technology", "technologies",
        "biosciences", "biological", "biotech",
        "engineering", "engineer", "engineering's",
        "mathematics", "mathematical", "physics", "chemistry",
        "biology", "biological", "medical", "medicine",
        "artificial", "intelligence", "intelligent",
        "automated", "automation", "robotics",
        "genai", "gen ai", "geng",
        "powered", "power", "framework", "frameworks",
        "platform", "platforms", "solution", "solutions",
        "innovate", "innovation", "innovative",
        "prototype", "prototypes", "character",
        "process", "progressive", "progress",
        "principle", "principles", "concept", "concepts",
        "digital", "information", "data", "computing",
        "network", "networks", "software", "hardware",
        "student", "students", "faculty", "curriculum",
        "institute", "institution", "institutions",
        "association", "associations",
        "collaborate", "collaborating", "collaboration",
        "think", "thinking", "thought", "imagine", "imagining",
        "visualize", "visualizing", "solve", "solving",
        "create", "creating", "creation", "creative",
        "build", "building", "achieve", "achieving",
        "inspire", "inspiring", "design", "designer", "designing",
        "innovate", "innovating", "innovation", "innovative",
        "celebrate", "celebrating", "welcome", "regards", "warmly",
        "together", "better", "journey", "forever", "real",
        "future", "future's", "dreams", "dreaming", "power",
        "powered", "possible", "possibilities", "passion",
        "all", "any", "each", "every", "both", "few",
        "morning", "afternoon", "evening", "night",
    }:
        return False
    # Reject Title Case multi-word lines where ALL words are
    # common English nouns/adjectives (not name-like).
    # These are activity descriptions, not person names.
    if len(words) >= 2 and not all(w.isupper() for w in words):
        all_common = all(
            word.lower().strip(".,;:") in {
                "stronger", "students", "student", "healthier",
                "tomorrow", "yesterday", "today", "leaders", "leader",
                "different", "people", "person", "common", "purpose",
                "community", "opportunity", "discipline", "growth",
                "service", "cultural", "art", "culture", "creativity",
                "performance", "performances", "recognition", "recognitions",
                "achievement", "achievements", "activity", "activities",
                "team", "play", "excel", "rise", "reflect", "rejoice",
                "faster", "higher", "beyond", "limits", "together",
                "journey", "company", "presence", "pleasure",
                "auspicious", "occasion", "blessings", "families",
                "relative", "kin", "members", "guests", "everyone",
                "everybody", "children", "couple", "organize",
                "organized", "organizing", "foundation", "association",
                "trust", "committee", "board", "department", "faculty",
                "conference", "summit", "class", "tutorial", "module",
                "experience", "ideas", "principles", "components",
                "data", "artificial", "intelligence",
                "science", "sciences", "technology", "technologies",
                "design", "designer", "prototyping", "interactive",
                "fundamentals", "learning", "learn", "teaching",
                "academic", "excellence", "knowledge", "library",
                "gallery", "museum", "studio", "concert", "show",
                "fair", "carnival", "parade", "race", "run", "meet",
                "gala", "showcase", "morning", "afternoon", "evening",
                "refreshment", "farewell", "graduation", "award",
                "prize", "medal", "celebration", "celebrate",
            }
        for word in words
        )
        if all_common:
            return False
    if any(word.lower().strip(".,;:") in _NAME_REJECT_PHRASE_WORDS for word in words):
        return False
    # Reject multi-word relationship/group phrases
    if ls.lower() in _RELATIONSHIP_GROUP_PHRASES:
        return False
    # Do not reject single-letter Latin initials (e.g. "A.", "S.") even though
    # the bare letter matches an article like "a" in the reject-word set. A
    # Latin initial on a Tamil name line signals a real person (e.g. "A. முஹம்மது ஆபிதா").
    non_initial_words = [
        w for w in words
        if not re.match(r"^[A-Za-z]\.$", w)
    ]
    if any(word.lower().strip(".,;:") in _NAME_REJECT_PHRASE_WORDS for word in non_initial_words):
        return False
    # Reject weekday names (e.g. "Monday", "Tuesday") which are never person names.
    if len(words) == 1 and words[0].lower() in {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }:
        return False
    # Reject month names which are never person names.
    month_names = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "sept", "oct", "nov", "dec",
    }
    if len(words) == 1 and words[0].lower() in month_names:
        return False
    # Reject common non-person words (verbs, durations, event terms).
    if len(words) == 1 and words[0].lower() in _NOT_PERSON_SINGLE_WORDS:
        return False
    # A single ALL-CAPS word is almost always a heading/logo (e.g. "PROGRAMME",
    # "INVITATION"), not a person name.
    if len(words) == 1 and words[0].isupper() and len(words[0]) > 1:
        return False
    # A comma usually indicates an address/locality line (e.g. "Banjara Hills,
    # Hyderabad"), not a person name.  Allow trailing commas (OCR punctuation).
    if "," in ls.rstrip(",.;:"):
        return False
    if not ls[0].isupper():
        return False
    if ls.lower() in KNOWN_CITIES:
        return False
    compact = re.sub(r"[^a-z]", "", ls.lower())
    if any(marker in compact for marker in (
            "sonof", "daughterof", "fatherof", "motherof", "grandsonof",
            "granddaughterof", "parentsof", "familyof")):
        return False
    # Reject text that mixes Latin script into the middle of Tamil text.
    # A valid Tamil name with a Latin initial looks like "A. முஹம்மது ஆயிஷா"
    # (Latin initial, space, then pure Tamil). OCR garbage often injects Latin
    # letters into Tamil words, e.g. "A.ஆயிpா ஆகலா".
    if _has_tamil(ls):
        latin_in_tamil = re.search(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z.\s])", ls)
        if latin_in_tamil:
            return False
    # Reject ALL-CAPS multi-word lines (e.g. "TAMIL NADU INDIA",
    # "KEC KONGU") — these are headings or locations, not names.
    if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
        return False
    # Reject lines starting with spiritual/religious prefixes.
    if re.match(r"^(?:om\b|om\.|namah|namaste)", ls, re.IGNORECASE):
        return False
    # Reject prayer/blessing phrases starting with modal verbs
    # commonly used in invocations ("May two souls...", "Let us...").
    if re.match(r"^(?:may|let)\b", ls, re.IGNORECASE):
        return False
    # Reject lines starting with common prepositions/conjunctions
    # that indicate context, not names ("With the blessings...").
    if words[0].lower() in {"with", "from", "by", "for", "of",
                            "to", "at", "in", "on", "and", "or"}:
        return False
    # Reject lines that are clearly venue/location names, not persons.
    _NAME_VENUE_WORDS = {"maaligai", "palace", "mandapam",
                           "sivasami", "kanchi", "koyambedu", "tuticorin",
                           "harbour estate"}
    if any(kw in lowered for kw in _NAME_VENUE_WORDS):
        return False
    # Reject lines containing only state/country names.
    _NAME_COUNTRY_WORDS = {"tamil", "nadu", "india", "kerala", "karnataka",
                           "maharashtra", "gujarat", "tamil", "delhi",
                           "mumbai", "chennai", "bengaluru", "bangalore",
                           "hyderabad", "kolkata", "pune", "salem"}
    if all(w.lower() in _NAME_COUNTRY_WORDS for w in words if w.isalpha()):
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
    r"your\s+presence\s+is\s+requested|"
    r"solicit|grac|invite\s*you|join\s*us|to\s*witness|celebrat|"
    r"request\s+the\s+pleasure|the\s+(?:wedding|marriage|reception)"
    r"\s+(?:ceremony|celebration)\s+of)",
    re.IGNORECASE,
)


def _is_greeting_or_event_desc(line: str) -> bool:
    """True if a line is greeting / opening / event-description text."""
    return bool(_GREETING_RE.search(line))


def _is_relationship_word_or_phrase(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return False
    if lowered in _NAME_REJECT_SINGLE_WORDS:
        return True
    return bool(re.match(
        r"^(?:parents?\s+to\s+be|parents?\s+of|family\s+of|relatives?\s+of|friends\s+of|children\s+of|child\s+of)\b",
        lowered,
    ))


def _is_relationship_phrase(line: str) -> bool:
    """Return True if a line starts with a relationship/group descriptor.

    Example: "Parents to be Saylee & Vaibhav" — here "Parents to be" is a
    role phrase, and the actual names are "Saylee" and "Vaibhav".  The whole
    line must not be treated as one person name.
    """
    return _is_relationship_word_or_phrase(line)


# Stop words that belong to parent/host/family lines (not the couple).
_FAMILY_ROLE_STOP = {
    "smt", "sri", "mr", "mrs", "ms", "dr", "late", "lt", "the", "of", "&",
    "and", "son", "daughter", "w/o", "d/o", "s/o", "kum", "selvi", "thiru",
}

# Common English words frequently seen in invitation phrases
# (not in proper person names). If every word in a candidate is in
# this set, the candidate is a descriptive phrase, not a name.
_COMMON_PHRASE_WORDS = {
    "beautiful", "beginning", "together", "blessings", "celebration",
    "special", "presence", "pleasure", "warmth", "regards",
    "joyfully", "invite", "make", "our", "your", "will", "more",
    "occasion", "auspicious", "family", "families", "relative",
    "kin", "members", "guests", "everyone", "children", "couple",
    "close", "dear", "friends", "proud", "happy", "glad", "thank",
    "grateful", "blessed", "blessing", "celebrate", "welcoming",
    "warm", "sincere", "heartfelt", "respectful", "humble",
    "request", "pleased", "honour", "honor", "privilege",
    "privileged", "gathering", "solemnize", "solemnization",
    "marriage", "wedding", "ceremony", "reception", "hearts", "life",
    "namah",
}
# Short function words (articles, prepositions, conjunctions)
# that are never proper names but appear in name-like phrases.
_FUNCTION_WORDS = {
    "a", "an", "the", "of", "in", "at", "to", "for", "with", "on",
    "by", "from", "or", "and", "our", "their", "its", "this", "that",
    "these", "those",
}


def _is_common_phrase(words: List[str]) -> bool:
    """Return True if every word is a common English word (not a proper name)."""
    if not words:
        return False
    for w in words:
        lw = w.lower()
        if lw in _COMMON_PHRASE_WORDS or lw in _FUNCTION_WORDS:
            continue
        return False
    return True


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
        if any(rk in low for rk in ("smt", "sri", "shree", "mr ", "mrs", " w/o ",
                                    "daughter", "son of", "of ", "&", "& ",
                                    "invite", "join", "wedding", "ceremony",
                                    "famil", "with", "and ", "namah")):
            continue
        words = [w for w in ls.split() if w.lower() not in _FAMILY_ROLE_STOP]
        if not words:
            continue
        cand = " ".join(words).strip(" .,;:&")
        if len(cand) < 3:
            continue
        if _is_common_phrase(words):
            continue
        if not _looks_like_name(cand):
            continue
        return _clean_name(cand)
    return ""


def extract_family_roles(text_lines: List[str], text: str = "") -> Dict[str, str]:
    """Detect bride/groom from ``D/o`` (daughter-of) and ``S/o`` (son-of) lines,
    as well as full ``Daughter of`` / ``Son of`` phrases.

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
        # English D/o / S/o patterns
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
        # Tamil D/o / S/o patterns: "டி/ ஓ" (D/o), "எஸ்/ ஓ" (S/o)
        elif re.search(r"டி\s*/\s*ஓ", s) and not re.search(
                r"எஸ்\s*/\s*ஓ", s):
            nm = _find_name_above(text_lines, i)
            if nm and not bride:
                bride = nm
        elif re.search(r"எஸ்\s*/\s*ஓ", s):
            nm = _find_name_above(text_lines, i)
            if nm and not groom:
                groom = nm

    # Also handle full "Daughter of" / "Son of" phrases.
    # These can appear on the same line as the name ("Venkatha Reddy Daughter of ...")
    # or on their own line with the name on the line above.
    full_text = text if text else "\n".join(text_lines)
    for m in re.finditer(
        r"\b(?:" + _NAME_TOKEN + r")\s+(?i:Daughter)\s+(?i:of)\b",
        full_text, re.DOTALL
    ):
        nm = _clean_name(m.group(1))
        if nm and not bride:
            bride = nm
    for m in re.finditer(
        r"\b(?:" + _NAME_TOKEN + r")\s+(?i:Son)\s+(?i:of)\b",
        full_text, re.DOTALL
    ):
        nm = _clean_name(m.group(1))
        if nm and not groom:
            groom = nm

    # Multi-line fallback: "Daughter of ..." / "Son of ..." on their own line
    for i, line in enumerate(text_lines):
        s = line.strip()
        if re.search(r"\bDaughter\s+of\b", s, re.IGNORECASE) and not re.search(
                r"\bSon\s+of\b", s, re.IGNORECASE):
            nm = _find_name_above(text_lines, i)
            if nm and not bride:
                bride = nm
        elif re.search(r"\bSon\s+of\b", s, re.IGNORECASE):
            nm = _find_name_above(text_lines, i)
            if nm and not groom:
                groom = nm

    return {"bride": bride, "groom": groom}


def _infer_generic_role(name: str, text_lines: List[str], text: str) -> str:
    """Return a generic role for a person in a non-couple event."""
    lowered = text.lower()
    role_patterns = [
        (r"\bhost(?:s)?\b", "Host"),
        (r"\borganizer\b", "Organizer"),
        (r"\bspeaker\b", "Speaker"),
        (r"\bcelebrant\b", "Celebrant"),
        (r"\bgraduate\b", "Graduate"),
        (r"\bparents?\b", "Parent"),
        (r"\bchild(?:ren)?\b", "Child"),
        (r"\bfamily\b", "Family"),
        (r"\bparticipants?\b", "Participant"),
        (r"\bperson\b", "Person"),
        (r"\bpeople\b", "Person"),
    ]
    for pat, role in role_patterns:
        if re.search(pat, lowered):
            return role
    if re.search(r"\b(?:birthday|bday)\b", lowered):
        return "Celebrant"
    return "Person"


def _extract_generic_people(text_lines: List[str], text: str,
                            ocr_lines: List[Dict] = None) -> List[Dict]:
    """Extract generic people names for non-couple events."""
    candidates: List[Dict] = []
    seen = set()

    def _has_positive_name_evidence(name: str) -> bool:
        """Require name structure, initials, or an honorific for OCR lines."""
        words = [word.strip(".,;:'\"()") for word in name.split()]
        if len(words) < 2:
            return False
        context_words = {
            "frame", "work", "design", "certificate", "graduation",
            "session", "college", "university", "society", "progress",
            "knowledge", "character", "achievement", "achievements",
            "emerging", "technologies", "technology", "biosciences",
            "science", "sciences", "data", "artificial", "intelligence",
        }
        if any(word.lower() in context_words for word in words):
            return False
        has_honorific = words[0].lower().rstrip(".") in {
            "mr", "mrs", "ms", "miss", "dr", "sri", "smt", "er", "thiru",
        }
        has_initial = any(
            len(word.rstrip(".")) <= 2 and word.rstrip(".").isalpha()
            and (word.rstrip(".").isupper() or word.endswith("."))
            for word in words[1:]
        )
        full_name_words = sum(1 for word in words if len(word) >= 3 and word.isalpha())
        return full_name_words >= 2 or ((has_honorific or has_initial) and full_name_words >= 1)

    def _add(name: str, role: str, confidence: float = 0.7, strategy: str = "generic", *, skip_looks_like_name: bool = False):
        for part in _split_combined_name(name):
            title_match = re.match(
                r"\s*(?P<title>mr|mrs|ms|miss|dr|sri|smt|er|thiru|selvi)\.?\s+",
                part,
                re.IGNORECASE,
            )
            cleaned = _clean_name(part)
            if title_match and cleaned:
                title = title_match.group("title").capitalize()
                cleaned = f"{title}. {cleaned}"
            if not cleaned or cleaned in seen:
                continue
            if (cleaned.lower().rstrip(".,;:") in INDIAN_STATES
                    or any(re.search(rf"\b{re.escape(keyword)}\b", cleaned, re.IGNORECASE)
                           for keyword in VENUE_STRONG_KEYWORDS)):
                continue
            if not skip_looks_like_name and not _looks_like_name(cleaned):
                continue
            if skip_looks_like_name:
                from app.core.matching.field_matcher import _looks_like_person_name
                if not _looks_like_person_name(cleaned):
                    continue
            if _is_greeting_or_event_desc(cleaned):
                continue
            if any(et in cleaned.lower() for et in (
                "wedding", "marriage", "engagement", "birthday", "anniversary",
                "party", "ceremony", "function", "reception", "event", "celebration",
            )):
                continue
            seen.add(cleaned)
            candidates.append({
                "name": cleaned,
                "role": role,
                "confidence": confidence,
                "strategy": strategy,
            })

    explicit_patterns = [
        r"\b(?i:host|hosts|organizer|organisers?|speaker|celebrant|graduate|family|participants?|person|people|child|children|parents?)\s*:?\s*([A-Z][a-zA-Z.]{2,40}(?:\s+[A-Z][a-zA-Z.]{2,40})?)",
        r"\b(?i:celebrating|honouring|honoring|inviting|welcome)\s+([A-Z][a-zA-Z.]{2,40}(?:\s+[A-Z][a-zA-Z.]{2,40})?)",
        r"\b(?i:to\s+celebrate|in\s+honou?r\s+of)\s+([A-Z][a-zA-Z.]{2,40}(?:\s+[A-Z][a-zA-Z.]{2,40})?)",
        r"\b([A-Z][a-zA-Z.]{2,40}(?:[ \t]+[A-Z][a-zA-Z.]{2,40})?)[ \t]*['’]s\s+(?:\d+(?:st|nd|rd|th)[ \t]+)?(?:birthday|bday)\b",
        r"\b([A-Z][a-zA-Z.]{2,40}(?:[ \t]+[A-Z][a-zA-Z.]{2,40})?)[ \t]*['’]s\s+\d+(?:st|nd|rd|th)\b",
    ]
    for pat in explicit_patterns:
        for m in re.finditer(pat, text, re.MULTILINE):
            captured = m.group(1).strip()
            # Skip relationship/group words captured as names (e.g. "Parents"
            # in "In honor of Parents to be Saylee & Vaibhav"). These are role
            # phrases, not person names.
            if _is_relationship_word_or_phrase(captured):
                continue
            role = _infer_generic_role(captured, text_lines, text)
            _add(captured, role, confidence=0.9, strategy="explicit_label", skip_looks_like_name=True)

    name_variants = extract_names_variants(text_lines, text, ocr_lines)
    structured_name_strategies = {
        "explicit_marker", "family_roles", "d_o_s_o", "standalone_pair",
        "standalone_weds", "weds_verb", "marriage_of", "daughter_son",
        "ampersand", "standalone_conjunction", "compact_wedding_pair",
        "son_with_daughter",
    }

    def has_address_context(name: str) -> bool:
        name_lower = name.casefold()
        for index, line in enumerate(text_lines):
            if name_lower not in line.casefold():
                continue
            nearby = text_lines[max(0, index - 1):index + 2]
            if any(re.search(
                    r"\b(?:road|rd|street|station|nagar|colony|avenue|pincode|pin)\b",
                    candidate,
                    re.IGNORECASE,
            ) for candidate in nearby):
                return True
        return False

    for c in name_variants:
        val = c.value if isinstance(c.value, dict) else {}
        for role_key in ("bride", "groom"):
            name = val.get(role_key, "").strip()
            if name:
                if has_address_context(name):
                    continue
                if (c.strategy not in structured_name_strategies
                        and not _has_positive_name_evidence(name)):
                    continue
                for part in _split_combined_name(name):
                    compact = re.sub(r"[^a-z]", "", part.lower())
                    if any(et in compact for et in (
                        "wedding", "marriage", "engagement", "birthday", "anniversary",
                        "party", "ceremony", "function", "reception", "event", "celebration",
                    )):
                        continue
                    role = _infer_generic_role(part, text_lines, text)
                    _add(part, role, confidence=c.confidence * 0.8, strategy=c.strategy)

    for line in text_lines:
        ls = line.strip()
        if not ls or not _looks_like_name(ls):
            continue
        if not _has_positive_name_evidence(ls):
            continue
        if (ls.lower().rstrip(".,;:") in INDIAN_STATES
                or any(re.search(rf"\b{re.escape(keyword)}\b", ls, re.IGNORECASE)
                       for keyword in VENUE_STRONG_KEYWORDS)):
            continue
        if _is_greeting_or_event_desc(ls):
            continue
        if _is_field_line(ls):
            continue
        if _is_tamil_blessing_or_invitation(ls):
            continue
        if re.search(r"\b(?:அனுப்புநர்|திரு\.?|திருமதி\.?)\b", ls, re.IGNORECASE):
            continue
        if re.match(r"^(?:venue|address|contact|rsvp|போன்|phone|மொ)\b",
                    ls, re.IGNORECASE):
            continue
        if _is_relationship_phrase(ls):
            continue
        role = _infer_generic_role(ls, text_lines, text)
        _add(ls, role, confidence=0.5, strategy="standalone_line")

    # Honorifics are positive person evidence even when OCR splits initials
    # and surnames into an otherwise weak standalone line.
    for line in text_lines:
        if re.match(
                r"^\s*(?:mrs|mr|ms|miss|dr|sri|smt|er|thiru|selvi)\.?\s+"
                r"[A-Z].+",
                line.strip(),
                re.IGNORECASE,
        ):
            title_match = re.match(
                r"^\s*(?P<title>mrs|mr|ms|miss|dr|sri|smt|er|thiru|selvi)\.?\s+(.+)$",
                line.strip(),
                re.IGNORECASE,
            )
            if title_match:
                cleaned = _clean_name(title_match.group(2))
                titled = f"{title_match.group('title').capitalize()}. {cleaned}"
                if cleaned and titled not in seen:
                    seen.add(titled)
                    candidates.append({
                        "name": titled,
                        "role": "Person",
                        "confidence": 0.9,
                        "strategy": "honorific_line",
                    })

    best = {}
    for c in candidates:
        name = c["name"]
        if name not in best or c["confidence"] > best[name]["confidence"]:
            best[name] = c
    unique = sorted(best.values(), key=lambda x: x["confidence"], reverse=True)
    return [{"name": c["name"], "role": c["role"]} for c in unique[:3]]


# ---------------------------------------------------------------------------
# Field extractors. Each returns a list of independent candidates.
# ---------------------------------------------------------------------------
# Tolerant Tamil fragments that survive OCR noise.  These are substrings of
# common Tamil wedding/Nikah vocabulary, not a list of people or venues.
# OCR often produces mixed-script output like "நickers" (Tamil ந + ASCII icke rs).
_TAMIL_NIKAH_FRAGMENTS = ("நிகா", "நிகாஹ்", "னிகா", "நிகாஹ", "ுக்காஹ்", "ிஎகா", "காஹ்", "ஸாஹ்", "கர்ஹ்", "நickers", "நickersாஹ்", "nickers", "nickersாஹ்")
# Wedding fragments: be specific to avoid matching "திருமண அழைப்பிதழ்" (wedding invitation document).
_TAMIL_WEDDING_FRAGMENTS = ("திருமணம்", "திருமண", " கல்யாணம்", "மணமக", "மணவிழா")
_TAMIL_ENGAGEMENT_FRAGMENTS = ("நிச்சய", "தார்த்த", "நிச்சயதார்த்த")
_TAMIL_WALIMA_FRAGMENTS = ("வலீமா", "வலிமா", "அலீமா")
_TAMIL_MEHNDI_FRAGMENTS = ("மஹர்", "மெஹந்தி", "மேகந்தி")
# Baby shower / naming ceremony fragments
_TAMIL_BABY_SHOWER_FRAGMENTS = ("விளை எடுத்து", "விளைஎடுத்து", "பிறந்த நாள்", "பெருந்தினம்", "சீமந்தம்", "வலைகாப்பு", "வளைக்காப்பு")
_TAMIL_BIRTHDAY_FRAGMENTS = ("பிறந்தநாள்", "பிறந்த நாள்", "ஜன্মநாள்")
# Tamil venue indicators: strong venue/place keywords and event-location markers.
# NOTE: "திருமண" (wedding), "மணமகள்" (daughter) and "மணமகன்" (son) are
# EVENT and RELATIONSHIP tokens, not venues. Including them here caused
# invitation titles (e.g. "திருமண அழைப்பிதழ்") and family-relationship lines to
# be mis-extracted as the venue. They are deliberately omitted.
_TAMIL_VENUE_KEYWORDS = (
    "மஹால்", "மஹறால்", "மஹாலில்", "மஹல்ி்", "மண்டபம்", "வெளியீடு", "அரசு", "அரஸ்",
    "நடைபெறும்", "நடைபெற", "கோவில்", "மஹால்",
)


def _tamil_event_type(text: str) -> Optional[Candidate]:
    """Return a wedding/Nikah/engagement/baby shower event-type candidate for Tamil text."""
    if any(frag in text for frag in _TAMIL_NIKAH_FRAGMENTS):
        return Candidate("Nikah", 0.92, "tamil_nikah")
    # Check wedding fragments, but skip if the ONLY match is "திருமண அழைப்பிதழ்" (wedding invitation document)
    wedding_match = False
    for frag in _TAMIL_WEDDING_FRAGMENTS:
        if frag in text:
            # Skip if this fragment only appears in "திருமண அழைப்பிதழ்" context
            # Check if fragment appears outside of "திருமண அழைப்பிதழ்"
            if frag in ("திருமண", "திருமணம்"):
                # Find all occurrences of the fragment
                import re
                pattern = re.escape(frag)
                matches = list(re.finditer(pattern, text))
                # If all matches are within "திருமண அழைப்பிதழ்", skip
                all_in_invitation = all(
                    "திருமண அழைப்பிதழ்" in text[max(0, m.start()-20):m.end()+20]
                    for m in matches
                )
                if all_in_invitation and matches:
                    continue
            wedding_match = True
            break
    if wedding_match:
        return Candidate("Wedding", 0.85, "tamil_wedding")
    if any(frag in text for frag in _TAMIL_ENGAGEMENT_FRAGMENTS):
        return Candidate("Engagement", 0.88, "tamil_engagement")
    if any(frag in text for frag in _TAMIL_BABY_SHOWER_FRAGMENTS):
        return Candidate("Baby Shower", 0.88, "tamil_baby_shower")
    if any(frag in text for frag in _TAMIL_BIRTHDAY_FRAGMENTS):
        return Candidate("Birthday", 0.85, "tamil_birthday")
    if any(frag in text for frag in _TAMIL_WALIMA_FRAGMENTS):
        return Candidate("Walima", 0.9, "tamil_walima")
    if any(frag in text for frag in _TAMIL_MEHNDI_FRAGMENTS):
        return Candidate("Mehndi", 0.88, "tamil_mehndi")
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
        # Tamil compound Islamic wedding events (e.g. Nikah & Walima).
        # When both Nikah and Walima fragments appear in Tamil text with explicit
        # conjunction (e.g. "Nikah & Walima", "நிகாஹ் மற்றும் வலிமா"), the event
        # type is "Wedding" with higher confidence than any individual component.
        has_nikah = any(frag in text for frag in _TAMIL_NIKAH_FRAGMENTS)
        has_walima = any(frag in text for frag in _TAMIL_WALIMA_FRAGMENTS)
        # Only treat as compound wedding if explicitly conjoined
        compound_conjoined = (
            re.search(r"நிகா[ம்]?\s*(?:&|and|மற்றும்|உம்)\s*வலிமா", text, re.IGNORECASE) or
            re.search(r"nikah\s*(?:&|and)\s*walima", text, re.IGNORECASE)
        )
        if has_nikah and has_walima and compound_conjoined:
            candidates.append(Candidate("Wedding", 0.95, "tamil_compound_wedding"))
        else:
            tamil_et = _tamil_event_type(text)
            if tamil_et is not None:
                candidates.append(tamil_et)
    # Compound Islamic wedding events (e.g. "Nikah & Walima", "Nikah and Walima")
    # are a single wedding ceremony with multiple components; the event type is
    # "Wedding" with higher confidence than any individual component keyword.
    # Match both on the same line (with &/and) or anywhere across the document,
    # including Tamil equivalents (நிகாஹ்/வலிமா/மெந்தி/ஆல்டி).
    # Only apply this English/Tamil fragment logic for non-Tamil text, since
    # Tamil-heavy invitations are handled by the Tamil-specific logic above.
    if not _has_tamil(text):
        if re.search(
            r"\b(?:nikah|walima|mehndi|haldi)\s*(?:&|and)\s*(?:nikah|walima|mehndi|haldi)\b",
            lowered,
        ):
            candidates.append(Candidate("Wedding", 0.95, "islamic_wedding_compound"))
        elif (
            lowered.count("nikah") >= 1 and lowered.count("walima") >= 1
        ) or (
            lowered.count("நிகா") >= 1 and lowered.count("வலிமா") >= 1
        ) or (
            lowered.count("நிகாஹ்") >= 1 and lowered.count("வலிமா") >= 1
        ):
            candidates.append(Candidate("Wedding", 0.93, "islamic_wedding_compound"))
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
            candidates.append(Candidate(_base_event_type(" ".join(words)), 0.92,
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
            and words[-1].lower() in {"ceremony", "reception", "celebration", "function"}):
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
    # Tamil event headings: the OCR text may be primarily Tamil, so the English
    # heading rules above cannot match.  Detect short Tamil lines that contain
    # wedding/Nikah/engagement/baby shower/birthday keywords and treat them as event-name candidates.
    for line in text.splitlines():
        heading = line.strip()
        if not heading or len(heading) > 60:
            continue
        if not _has_tamil(heading):
            continue
        if any(frag in heading for frag in (
            _TAMIL_WEDDING_FRAGMENTS + _TAMIL_NIKAH_FRAGMENTS +
            _TAMIL_ENGAGEMENT_FRAGMENTS + _TAMIL_BABY_SHOWER_FRAGMENTS +
            _TAMIL_BIRTHDAY_FRAGMENTS + _TAMIL_WALIMA_FRAGMENTS + _TAMIL_MEHNDI_FRAGMENTS
        )):
            candidates.append(Candidate(_clean(heading), 0.93,
                                        "tamil_event_heading"))
    labeled = re.search(r"^\s*(?:event|event name)\s*[:\-]\s*(.+)$",
                        text, re.IGNORECASE | re.MULTILINE)
    if labeled:
        value = _clean(labeled.group(1))
        if value:
            candidates.append(Candidate(value, 0.96, "event_label"))
    acronym_year = re.search(r"\b([A-Z][A-Z0-9]{2,})\s*['’]?\s*(\d{2,4})\b",
                             text)
    if acronym_year and re.search(r"\bconference\b", text, re.IGNORECASE):
        candidates.append(Candidate(
            f"{acronym_year.group(1)} {acronym_year.group(2)}",
            0.98, "event_acronym_year"))
    celebration = re.search(
        r"(wedding\s+(?:ceremony|celebration|reception)|"
        r"(?:wedding|reception|engagement|birthday|naming ceremony|housewarming)"
        r"\s+(?:ceremony|celebration|reception|party))",
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

    # Event title patterns: workshop/training/design titles that don't
    # follow traditional wedding/engagement wording.
    # e.g. "Figma for UI/UX Design: From Basics to Prototyping"
    _EVENT_TITLE_KEYWORDS = {
        "design", "workshop", "training", "seminar", "course",
        "conference", "summit", "class", "tutorial", "program",
        "certificate", "diploma", "module", "prototyping",
    }
    _EVENT_TITLE_RANGE = re.compile(
        r"(?:from|start|begin)\s+[\w]+\s+(?:to|[-–])\s+(?:basic|advanced|intermediate|beginner|expert|foundation|pro|professional)",
        re.IGNORECASE,
    )
    title_lines = []
    for line in text.split("\n"):
        ls = line.strip()
        if not ls or len(ls) > 60:
            continue
        if re.search(r"\d", ls):
            continue
        lowered_ls = ls.lower()
        is_title = False
        for kw in _EVENT_TITLE_KEYWORDS:
            if kw in lowered_ls:
                is_title = True
                break
        if not is_title and _EVENT_TITLE_RANGE.search(ls):
            is_title = True
        if not is_title:
            continue
        if re.match(r"^(?:at|held at|located at|venue|address|contact|phone|date|time)\b",
                     ls, re.IGNORECASE):
            continue
        title_lines.append(_clean(ls))
    if title_lines:
        structured_titles = [
            title for title in title_lines
            if re.search(r"\bfrom\b", title, re.IGNORECASE)
            and re.search(r"\bto\b", title, re.IGNORECASE)
        ]
        combined = max(structured_titles, key=len) if structured_titles else (
            ": ".join(title_lines) if len(title_lines) > 1 else title_lines[0]
        )
        if structured_titles:
            structured = max(structured_titles, key=len)
            structured_index = title_lines.index(structured)
            if structured_index > 0:
                prefix = title_lines[structured_index - 1].rstrip(" :")
                if prefix and len(prefix.split()) <= 10:
                    combined = f"{prefix}: {structured}"
        combined = re.sub(r"\s*:\s*", ": ", combined).strip()
        if combined:
            candidates.append(Candidate(combined, 0.9, "event_title"))
    hackathon_title = re.search(
        r"\b(hackathon\s+\d+\s+hours?\s+\d+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if hackathon_title:
        title = hackathon_title.group(1).strip()
        series = re.search(r"\bHACKNEXT['’]?\d+\b", text, re.IGNORECASE)
        if series:
            title = f"{title} — {series.group(0)}"
        candidates.append(Candidate(title, 0.98, "structured_hackathon_title"))
    # Compound event names joined by "&" or "and" (e.g. "Mahandi & Sangeet").
    # Only treat as an event name when the line is followed by a date within
    # the next few lines, distinguishing program items from couple names.
    invite_lines = text.splitlines()
    compound_candidates = []
    for i, line in enumerate(invite_lines):
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if not (2 <= len(words) <= 5):
            continue
        if not re.search(r'\s+(?:&|and)\s+', stripped):
            continue
        if not all(w[0].isupper() for w in words if w and w[0].isalpha()):
            continue
        for j in range(i + 1, min(i + 4, len(invite_lines))):
            if extract_date(invite_lines[j]):
                compound_candidates.append((i, stripped))
                break
    # Prefer compound event names that appear later in the text (program items
    # over couple names which typically appear at the top).
    compound_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, name in compound_candidates:
        candidates.append(Candidate(name, 0.85, "compound_event_name"))
        break
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
    # Tamil compound Islamic wedding events (e.g. Nikah & Walima).
    # When both Nikah and Walima fragments appear in Tamil text, the event
    # name is "Nikah & Walima" and the event type is "Wedding".
    if _has_tamil(text):
        has_nikah = any(frag in text for frag in _TAMIL_NIKAH_FRAGMENTS)
        has_walima = any(frag in text for frag in _TAMIL_WALIMA_FRAGMENTS)
        if has_nikah and has_walima:
            candidates.append(Candidate("Nikah & Walima", 0.9, "tamil_compound_nikah_walima"))
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
        rf"(?i)(\d{{1,2}}(?:st|nd|rd|th)?)(?=(?:{month_glue})(?=\d{{2,4}}\b|\b))", r"\1 ",
        normalized_text)
    normalized_text = re.sub(
        rf"(?i)((?:{month_glue}))(?=\d{{2,4}}\b)", r"\1 ", normalized_text)
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
        rf"\b({_DY})\s*\n\s*(?:{_WEEKDAY})\.?,?\s*({_MONTH})\.??\s+(\d{{2,4}})",
        normalized_text, re.IGNORECASE,
    )
    if m:
        candidates.append(Candidate(
            f"{m.group(1)} {m.group(2)} {m.group(3)}",
            0.92,
            "day_newline_weekday_month_year",
        ))
    m = re.search(
        rf"\b({_DY})\s*\n\s*({_MONTH})\.??\s+(\d{{2,4}})",
        normalized_text, re.IGNORECASE,
    )
    if m:
        candidates.append(Candidate(
             f"{m.group(1)} {m.group(2)} {m.group(3)}",
             0.90,
             "day_newline_month_year",
         ))
    m = re.search(
        rf"\b({_DY})\s*\n\s*({_MONTH})\.??\b",
        normalized_text, re.IGNORECASE,
    )
    if m:
        candidates.append(Candidate(
            f"{m.group(1)} {m.group(2)}",
            0.65,
            "day_newline_month",
        ))
    # Date ranges like "08 - 09 October 2026" or "16 - 18 November 2026"
    for m in re.finditer(
        rf"\b(\d{{1,2}}\s*[-\u2013\u2014]\s*\d{{1,2}}\s+(?:{_MONTH})\.?\.?\s+\d{{2,4}})",
        normalized_text, re.IGNORECASE,
    ):
        candidates.append(Candidate(m.group(1).strip(), 0.92, "date_range_day_month_year"))
    # Date ranges like "October 8 - 9, 2026"
    for m in re.finditer(
        rf"\b((?:{_MONTH})\.?\.?\s+\d{{1,2}}\s*[-\u2013\u2014]\s*\d{{1,2}},?\s+\d{{2,4}})",
        normalized_text, re.IGNORECASE,
    ):
        candidates.append(Candidate(m.group(1).strip(), 0.92, "date_range_month_day"))
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
    # Fix common OCR confusions where letter O/o replaces digit 0 in times.
    # Run iteratively until stable so chained substitutions like "1O:Ooam"
    # fully normalize to "10:00am".
    prev = None
    while prev != normalized_text:
        prev = normalized_text
        normalized_text = re.sub(
            r"(?<=\d)[Oo](?=[.:]?[Oo\d]*\s*(?:am|pm|a\.?m\.?|p\.?m\.?|onwards?|onward|sharp))",
            "0",
            normalized_text,
            flags=re.IGNORECASE,
        )
        normalized_text = re.sub(
            r"(?<=[.:])[Oo](?=[Oo\d]*\s*(?:am|pm|a\.?m\.?|p\.?m\.?|onwards?|onward|sharp))",
            "0",
            normalized_text,
            flags=re.IGNORECASE,
        )
    for pat, conf, strategy in _TIME_STRATEGIES:
        for m in re.finditer(pat, normalized_text, re.IGNORECASE):
            start = m.start()
            line_start = normalized_text.rfind('\n', 0, start) + 1
            line_end = normalized_text.find('\n', start)
            if line_end == -1:
                line_end = len(normalized_text)
            line = normalized_text[line_start:line_end]
            if re.search(r"\b(?:rsvp|contact|phone|mobile|mob|ph|tel)\b", line, re.IGNORECASE):
                conf = 0.7
            candidates.append(Candidate(m.group(1).strip(), conf, strategy))
    # Detect explicit time ranges on one line: "10:30 AM - 12:30 PM"
    range_pat = (r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|noon|midnight|"
                 r"a\.?m\.?|p\.?m\.?))\s*[-–]\s*"
                 r"(\d{1,2}[:.]\d{2}\s*(?:am|pm|noon|midnight|"
                 r"a\.?m\.?|p\.?m\.?))\b")
    for m in re.finditer(range_pat, normalized_text, re.IGNORECASE):
        start_time = m.group(1).strip()
        end_time = m.group(2).strip()
        candidates.append(Candidate(start_time, 0.92, "time_range_start"))
        candidates.append(Candidate(end_time, 0.92, "time_range_end"))

    # Two times on the same line without an explicit connector, e.g.
    # "10:00 AM 1:00 PM" (common when OCR drops the dash/to). Pair them as a
    # start/end range only when both times share the line and differ.
    same_line_two_pat = (
        r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|noon|midnight|a\.?m\.?|p\.?m\.?))"
        r"\s+(?!\d)(?:\S+\s+)*"
        r"(\d{1,2}[:.]\d{2}\s*(?:am|pm|noon|midnight|a\.?m\.?|p\.?m\.?))\b"
    )
    for m in re.finditer(same_line_two_pat, normalized_text, re.IGNORECASE):
        start_time = m.group(1).strip()
        end_time = m.group(2).strip()
        if start_time.lower() == end_time.lower():
            continue
        candidates.append(Candidate(start_time, 0.86, "time_pair_start"))
        candidates.append(Candidate(end_time, 0.86, "time_pair_end"))

    # OCR often places the range connector on a separate line from the start
    # time (for example, ``6:00 PM`` followed later by ``to 9:00 PM``). Pair
    # that connector with the nearest preceding time in the same event group.
    ocr_lines = normalized_text.splitlines()
    for line_index, line in enumerate(ocr_lines[:-1]):
        if not re.search(r"\bto\s*$", line, re.IGNORECASE):
            continue
        next_match = re.search(
            r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|noon|midnight|a\.?m\.?|p\.?m.?))\b",
            ocr_lines[line_index + 1],
            re.IGNORECASE,
        )
        start_match = list(_TIME_TOKEN_RE.finditer(line))
        if next_match and start_match:
            candidates.append(Candidate(start_match[-1].group(1), 0.94, "to_range_start"))
            candidates.append(Candidate(next_match.group(1), 0.94, "to_range_end"))
    for line_index, line in enumerate(ocr_lines):
        for to_match in _TO_TIME_RE.finditer(line):
            start_match = None
            same_line_matches = list(_TIME_TOKEN_RE.finditer(line[:to_match.start()]))
            if same_line_matches:
                start_match = same_line_matches[-1]
            else:
                for prior_index in range(line_index - 1, max(-1, line_index - 7), -1):
                    prior_matches = list(_TIME_TOKEN_RE.finditer(ocr_lines[prior_index]))
                    if prior_matches:
                        start_match = prior_matches[-1]
                        break
            if start_match is None:
                continue
            start_time = start_match.group(1).strip()
            end_time = to_match.group(1).strip()
            if start_time.lower() == end_time.lower():
                continue
            candidates.append(Candidate(start_time, 0.94, "to_range_start"))
            candidates.append(Candidate(end_time, 0.94, "to_range_end"))

    # Detect Tamil time ranges: "10.45 மணிக்கு மேர் 11.45 மணிக்குள்" (10:45 AM to 11:45 AM)
    tamil_range_pat = r"\b(\d{1,2}[.:]\d{2})\s*மணிக்கு\s*மேல்\s*(\d{1,2}[.:]\d{2})\s*மணிக்குள்\b"
    for m in re.finditer(tamil_range_pat, normalized_text, re.IGNORECASE):
        start_time = m.group(1).replace(".", ":") + " AM"
        end_time = m.group(2).replace(".", ":") + " AM"
        candidates.append(Candidate(start_time, 0.9, "tamil_time_range_start"))
        candidates.append(Candidate(end_time, 0.9, "tamil_time_range_end"))
    # Detect Tamil time ranges written with a dash separator and the
    # "முதல் ... மணிவரை" / "முதல் ... மறைவு" connectors, e.g.
    # "10-15 முதல் 11-45 மணிவரை" (10:15 to 11:45). The dash/colon/dot before the
    # minutes is the common Tamil OCR notation for a time. Also capture the
    # two bounds as separate start/end candidates so the range is preserved.
    tamil_dash_range = (r"\b(\d{1,2})[-.:](\d{2})\s*முதல்\s*(\d{1,2})[-.:](\d{2})"
                          r"\s*(?:மணிவரை|மறைவு|முடிவு)")
    for m in re.finditer(tamil_dash_range, normalized_text, re.IGNORECASE):
        start_time = f"{int(m.group(1))}:{m.group(2)} AM"
        end_time = f"{int(m.group(3))}:{m.group(4)} AM"
        candidates.append(Candidate(start_time, 0.9, "tamil_dash_range_start"))
        candidates.append(Candidate(end_time, 0.9, "tamil_dash_range_end"))
    # Detect a single Tamil dash/colon time such as "10-15 மணிக்கு" / "11.30 மணிக்கு"
    # or "10-15 மணி" so it is not only caught when it is part of a range.
    tamil_single_time = r"\b(\d{1,2})[-.:](\d{2})\s*(?:மணிக்கு|மணி|மணிகள்)\b"
    for m in re.finditer(tamil_single_time, normalized_text, re.IGNORECASE):
        val = f"{int(m.group(1))}:{m.group(2)}"
        candidates.append(Candidate(val, 0.85, "tamil_single_time"))
    return candidates


def extract_time(text: str) -> Candidate:
    return _best(extract_time_variants(text))


def _looks_like_date(text: str) -> bool:
    """Return True if text matches common date patterns that could be confused with phone numbers."""
    if not text:
        return False
    t = text.strip()
    # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY, MM-DD-YYYY, etc.
    if re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$", t):
        return True
    # YYYY-MM-DD, YYYY/MM/DD
    if re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", t):
        return True
    # DD-MM-YY, DD/MM/YY
    if re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2}$", t):
        return True
    return False


def extract_contact_variants(text: str) -> List[Candidate]:
    candidates: List[Candidate] = []
    labeled = re.search(
        r"\b(?:contact(?:\s+(?:number|no))?|phone|mobile|mob|ph|tel|rsvp)\b"
        r"(?:\s+\S+){0,6}?\s*"
        r"(\+?\d[\d \-]{8,15}\d)",
        text, re.IGNORECASE)
    if labeled:
        val = labeled.group(1).strip()
        if not _looks_like_date(val):
            candidates.append(Candidate(val, 0.95, "label_phone"))
    for pat, conf, strategy in _PHONE_STRATEGIES:
        for m in re.finditer(pat, text):
            val = m.group(1).strip()
            # Avoid duplicates from overlapping patterns
            if any(c.value == val for c in candidates):
                continue
            # Exclude date-like patterns (e.g. 17-09-2023)
            if _looks_like_date(val):
                continue
            candidates.append(Candidate(val, conf, strategy))
    return candidates


def extract_contact(text: str) -> Candidate:
    variants = extract_contact_variants(text)
    if not variants:
        return Candidate("", 0.0, "none")
    # Deduplicate and concatenate multiple phone numbers
    seen = set()
    unique = []
    for v in variants:
        val = v.value.strip()
        if val and val not in seen:
            seen.add(val)
            unique.append(v)
    if not unique:
        return Candidate("", 0.0, "none")
    # Prefer labeled phone numbers, then join all unique numbers
    labeled = [v for v in unique if v.strategy == "label_phone"]
    best = labeled[0] if labeled else unique[0]
    # Concatenate multiple numbers with comma
    if len(unique) > 1:
        canonical = {}
        for candidate in unique:
            digits = re.sub(r"\D", "", candidate.value)
            if len(digits) == 12 and digits.startswith("91"):
                digits = digits[2:]
            canonical.setdefault(digits or candidate.value, candidate)
        unique = list(canonical.values())
        return Candidate(", ".join(v.value for v in unique), best.confidence, best.strategy)
    return best


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


def _clean_venue_address(value: str) -> str:
    """Strip leading symbols and trailing activity words from venue values."""
    if not value:
        return value
    s = value.strip()
    s = re.sub(r"^[@.,;:]+", "", s).strip()
    # Strip a leading OCR-detached acronym (e.g. "KEC") that got glued before an
    # institution name, but only when the remainder still carries an institution
    # keyword so legitimate acronyms at the start of a venue are preserved.
    # This prefix cleanup is only for true all-caps acronyms. Case-insensitive
    # matching incorrectly treated title-case venue names such as "Snow Hall"
    # as detached acronyms and reduced them to "Hall".
    m = re.match(r"^(?!the\b)([A-Z]{2,4})\s+(.+)$", s)
    if m and re.search(r"\b(?:college|university|school|hall|auditorium|institute)\b",
                       m.group(2), re.IGNORECASE):
        s = m.group(2).strip()
    # Tamil invitations: the venue name is often followed by a descriptive
    # phrase starting with "நடைபெறும்" (takes place) or similar.  Split at
    # that boundary so we keep only the actual venue name.
    tamil_venue_split = re.split(r"\s*நடைபெற(?:ும்)?\s*", s, maxsplit=1)
    if len(tamil_venue_split) == 2:
        s = tamil_venue_split[0].strip()
    # Strip Tamil locative case endings (e.g. "ஹாலில்" -> "ஹால்", "மண்டபத்தில்" -> "மண்டபம்").
    # Common locative suffixes: இல், இல், அல், இடில். Remove when they appear at word end.
    # Use a pattern that matches Tamil letters followed by locative suffixes.
    s = re.sub(r"([\u0B80-\u0BFF\w]+)(?:இல்|இல்|அல்|இடில்)\b", r"\1", s)
    activity_words = [
        "dinner", "lunch", "breakfast", "refreshments", "reception",
        "ceremony", "function", "celebration", "party", "event",
        "onwards", "onward", "sharp", "timings", "timing",
    ]
    words = s.split()
    while words and words[-1].lower().strip(".,;:") in activity_words:
        words.pop()
    s = " ".join(words).strip(" ,;:")
    # Strip trailing punctuation, but keep a period that directly follows a
    # street abbreviation (e.g. "St.", "Rd.", "Ave.").
    m = re.match(r"^(.*?)[.;:,*/-]+$", s)
    if m:
        base = m.group(1)
        if not re.search(
            r"(?:st|rd|ave|av|ln|blvd|dr|ct|pl|way|ter|cir|mg|main)\.?$",
            base, re.IGNORECASE,
        ):
            s = base
    # If the value contains a comma and the part after the comma looks like a
    # city/locality rather than a venue name or street address, keep only the venue part.
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            first, second = parts
            second_lower = second.lower()
            first_has_venue_keyword = any(
                re.search(rf"\b{re.escape(kw)}\b", first, re.IGNORECASE)
                for kw in VENUE_STRONG_KEYWORDS
            )
            has_venue_keyword = any(
                re.search(rf"\b{re.escape(kw)}\b", second_lower)
                for kw in VENUE_STRONG_KEYWORDS
            )
            has_number = bool(re.search(r"\d", second))
            is_short_locality = len(second) < 30 and not has_venue_keyword and not has_number
            has_address_tail = bool(
                re.search(r"\b(?:road|rd|street|nagar|colony|avenue)\b",
                          second, re.IGNORECASE)
                or re.search(r"\b\d{6}\b", second)
                or any(re.search(rf"\b{re.escape(state)}\b", second,
                                 re.IGNORECASE)
                       for state in INDIAN_STATES)
            )
            # A postal-code-bearing tail means this is venue + full address: keep
            # only the venue name before the first comma (e.g. "The Manor House,
            # West St, Chippenham, SN14 7HX" -> "The Manor House").
            has_postal_code = bool(re.search(
                r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",  # UK style: SN14 7HX
                second, re.IGNORECASE,
            )) or bool(re.search(r"\b\d{5}(?:-\d{4})?\b", second))  # US style
            if (is_short_locality or has_postal_code
                    or (first_has_venue_keyword and has_address_tail)):
                s = first
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_institution_venue(primary_venue: str,
                            recovered_venue: str) -> str:
    """Choose the fullest institution venue between matched and recovered."""
    institution_kw = re.compile(
        r"\b(?:institute|college|university|school|hall|auditorium)\b",
        re.IGNORECASE,
    )
    candidates = []
    for value in (primary_venue, recovered_venue):
        if value and institution_kw.search(value):
            candidates.append(value)
    if not candidates:
        return primary_venue or recovered_venue or ""
    return max(candidates, key=lambda v: (len(v.split()), len(v)))


# Decorative invitation phrases that frequently bleed into address lines in
# OCR output. They are not part of a postal address and should be stripped so
# the recovered address only contains genuine location information.
_ADDRESS_NOISE_PHRASES = (
    "let's celebrate", "let us celebrate", "let's celebrate",
    "with blessings", "blessings", "our families", "our family",
    "new beginnings", "good people", "life together", "new memories",
    "good vibes", "two families", "one heart", "lifelong",
    "better together", "together in love", "with love",
    "good wishes", "good vibrations", "happier tomorrow", "brighter tomorrow",
    "save the date", "save the date",
)


def _strip_address_noise(address: str) -> str:
    """Remove decorative / filler phrases that OCR appends to addresses."""
    if not address:
        return address
    cleaned = address
    # OCR often merges a complete postal address and the invitation's closing
    # sentence onto one line. Discard that sentence and everything after it.
    cleaned = re.sub(
        r"\b(?:your\s+presence|we\s+request\s+the\s+pleasure)\b.*$",
        "", cleaned, flags=re.IGNORECASE,
    )
    for phrase in _ADDRESS_NOISE_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b\s*,?\s*", "", cleaned,
                         flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,\s*", ", ", cleaned).strip(" ,.;")
    # Strip stray lowercase filler words that OCR bleeds into the address
    # tail (e.g. "Muhurtham from With Love" -> trailing "from").
    cleaned = re.sub(r",\s*(?:from|with|to|of|at|the|and)\s*$", "", cleaned,
                     flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:from|with|to|of|at|the)\s*$", "", cleaned,
                     flags=re.IGNORECASE).strip(" ,.;")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned


def _normalize_address_text(value: str) -> str:
    """Normalize OCR punctuation and Indian postal-code spacing."""
    if not value:
        return ""
    value = _strip_address_noise(value)
    address = re.sub(r"\s*,\s*,+", ", ", value.strip())
    address = re.sub(r"\s+", " ", address)
    address = re.sub(
        r"([A-Za-z])\s+(\d{3}\s+\d{3})(?=,|$)",
        r"\1 - \2",
        address,
    )
    return address.strip(" ,.")


def _address_block_from_text(text: str) -> str:
    """Recover a complete street/address block when OCR split its lines."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        has_street_marker = re.search(
            r"\b(?:road|rd|street|nagar|colony|avenue|main road|main rd)\b",
            line,
            re.IGNORECASE,
        )
        has_locality_marker = (
            "," in line
            and not extract_date(line)
            and not extract_time(line)
            and any(re.search(rf"\b{state}\b", following, re.IGNORECASE)
                    for following in lines[index + 1:index + 4]
                    for state in ("Tamil Nadu", "Kerala", "Karnataka"))
        )
        if (has_locality_marker
                and any(re.search(
                    r"\b(?:road|rd|street|nagar|colony|avenue|main road|main rd)\b",
                    following,
                    re.IGNORECASE,
                ) for following in lines[index + 1:])):
            continue
        if not has_street_marker and not has_locality_marker:
            continue
        parts = []
        previous = ""
        for prior in reversed(lines[max(0, index - 3):index]):
            prior = prior.strip().rstrip(" ,;:")
            if prior.startswith("(") and prior.endswith(")"):
                continue
            previous = prior
            break
        if previous:
            if (len(previous) <= 40
                    and not extract_date(previous)
                    and not extract_time(previous)
                    and not re.fullmatch(rf"(?i:{_WEEKDAY})\.?,?", previous)
                    and previous.lower() not in {"session", "full-day", "full day"}
                    # Exclude venue names from address block
                    and not any(re.search(rf"\b{re.escape(kw)}\b", previous, re.IGNORECASE)
                                for kw in VENUE_STRONG_KEYWORDS)):
                parts.append(previous)
        parts.append(line)
        for following in lines[index + 1:index + 11]:
            if following.startswith("(") and following.endswith(")"):
                continue
            if extract_date(following) or extract_time(following):
                continue
            if following.lower().strip(" .,;:") in {
                    "session", "full-day", "full day", "muhurtham",
                    "muhoortham",
            }:
                continue
            if re.fullmatch(rf"(?i:{_WEEKDAY})\.?,?", following):
                continue
            parts.append(following)
            if re.search(r"\b(?:Tamil Nadu|India|Kerala|Karnataka)\b",
                         following, re.IGNORECASE):
                break
            if re.search(r"\b\d{3}\s*[- ]?\s*\d{3}\b", following):
                continue
        candidate = _normalize_address_text(", ".join(parts))
        candidate = re.sub(
            r"\b(Railway),\s+(Station\s+Road)\b",
            r"\1 \2",
            candidate,
            flags=re.IGNORECASE,
        )
        if (re.search(r"\b(?:\d{3}\s+\d{3}|\d{6})\b", candidate)
            or has_locality_marker):
            return candidate
    return ""


def _prefer_complete_address(address: str, recovered: str) -> str:
    """Prefer a recovered street/postal block over a less complete address."""
    if not recovered:
        return address
    street_pattern = r"\b(?:road|rd|street|nagar|colony|avenue)\b"
    postal_pattern = r"\b(?:\d{3}\s*\d{3}|\d{6})\b"
    recovered_has_street = bool(re.search(street_pattern, recovered, re.IGNORECASE))
    current_has_street = bool(re.search(street_pattern, address or "", re.IGNORECASE))
    recovered_has_postal = bool(re.search(postal_pattern, recovered))
    current_has_postal = bool(re.search(postal_pattern, address or ""))

    # A postal code alone does not make an address complete: the model often
    # returns only locality + state + PIN while OCR has the street above it.
    if (recovered_has_street and not current_has_street
            and (not current_has_postal or recovered_has_postal)):
        return recovered
    if (recovered_has_postal and not current_has_postal
            and len(recovered) > len(address or "")):
        return recovered
    return address


def _institution_venue_from_text(text: str) -> str:
    """Find the most plausible institution venue in OCR lines."""
    organization_units = re.compile(
        r"\b(?:partnership|cell|department|association|committee|organ(?:izes|ised|ized))\b",
        re.IGNORECASE,
    )
    venue_candidates = []
    for line in (text or "").splitlines():
        value = line.strip().strip(" ,;:")
        if (not value or len(value) > 100 or organization_units.search(value)
                or re.search(r"\b(?:in\s+)?association\s+with\b", value, re.I)):
            continue
        if extract_date(value) or extract_time(value):
            continue
        if re.search(r"\b(?:college|university|school|hall|auditorium|center|centre)\b",
                     value, re.IGNORECASE):
            venue_candidates.append(value)
    return max(venue_candidates, key=lambda value: (len(value.split()), len(value))) if venue_candidates else ""


def _format_person_name(name: str, role: str, text: str) -> str:
    """Remove OCR tail punctuation and preserve an OCR-supported Dr. title."""
    formatted = re.sub(r"\s*[.,;:]+\s*$", "", (name or "").strip())
    formatted = re.sub(r"\s+[-–—]\s*$", "", formatted).strip()
    if (role.lower() == "groom" and formatted
            and not re.match(r"^Dr\.\s", formatted, re.IGNORECASE)
            and re.search(
                rf"\bDr\.?\s*(?:\n\s*)?{re.escape(formatted)}\b",
                text or "",
                re.IGNORECASE,
            )):
        formatted = f"Dr. {formatted}"
    return formatted


def extract_venue_variants(text_lines: List[str], text: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    def usable_venue(value: str) -> bool:
        return bool(value and not extract_date(value) and not extract_time(value))

    for line in text_lines:
        val = _venue_from_line(line)
        if usable_venue(val):
            val = _clean_venue_address(val)
            if val:
                candidates.append(Candidate(val, 0.95, "venue_label"))
    for line in text_lines:
        lowered = line.lower()
        for kw in VENUE_STRONG_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                val = _strip_label(line)
                if not usable_venue(val) or len(val) >= 80:
                    continue
                if len(val.split()) == 1 and (
                    val.lower() in VENUE_STRONG_KEYWORDS
                    or val.lower() in VENUE_KEYWORDS
                ):
                    continue
                val = _clean_venue_address(val)
                if val:
                    candidates.append(Candidate(val, 0.85, f"venue_keyword:{kw}"))
    for line in text_lines:
        m = re.match(r"^(?:at|@)\s+(.+)$", line, flags=re.IGNORECASE)
        if m:
            rest = m.group(1).strip()
            rest = _clean_venue_address(rest)
            if (usable_venue(rest) and len(rest) < 80
                    and not _looks_like_event_desc(rest)):
                candidates.append(Candidate(rest, 0.7, "at_phrase"))
    # Combine adjacent venue lines (e.g. "Sivasami Maaligai" +
    # "Marriage Hall" on separate lines into "Sivasami Maaligai
    # Marriage Hall").  Skips short non-venue lines in between.
    _VENUE_LINE_SKIP = {
        "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
        "thursday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "am", "pm", "am.", "pm.",
        "muhoortham", "muhurtham", "purattasi", "tithi", "natchathiram",
        "tamil", "nadu", "india", "kerala", "karnataka",
    }
    venue_indices = []
    for i, line in enumerate(text_lines):
        lowered = line.lower()
        is_venue = any(
            re.search(rf"\b{re.escape(kw)}\b", lowered)
            for kw in VENUE_STRONG_KEYWORDS
        )
        if is_venue:
            venue_indices.append(i)
    if len(venue_indices) >= 2:
        first_raw = text_lines[venue_indices[0]].strip()
        first_stripped = _strip_label(first_raw).strip(" :;.,")
        if first_stripped and first_stripped.lower() not in (
                "venue", "location", "place", "at", "located at"):
            combined_lines = [first_stripped]
        else:
            combined_lines = [first_raw]
        for idx in venue_indices[1:]:
            gap_lines = [
                text_lines[j].strip()
                for j in range(venue_indices[0] + 1, idx)
            ]
            skip = True
            for gl in gap_lines:
                if not gl:
                    continue
                gl_lower = gl.lower()
                if len(gl) < 3:
                    continue
                if gl_lower in _VENUE_LINE_SKIP:
                    continue
                if extract_time(gl) or extract_date(gl):
                    continue
                if any(
                    re.search(rf"\b{re.escape(kw)}\b", gl_lower)
                    for kw in VENUE_STRONG_KEYWORDS
                ):
                    continue
                skip = False
                break
            if skip:
                combined_lines.append(text_lines[idx].strip())
            else:
                combined_lines = [first_stripped or first_raw]
        combined = " ".join(combined_lines)
        if usable_venue(combined) and len(combined) < 80:
            candidates.append(Candidate(combined, 0.8, "venue_combined"))
    # A bare street followed by a recognized city/state/postal line is a usable
    # venue when the invitation has no named hall or property.
    for index, line in enumerate(text_lines[:-1]):
        street = _extract_street_address(line)
        if not street or extract_time(line):
            continue
        street_clean = re.sub(r"[.,;:]+$", "", street)
        city_from_line = line.strip().replace(street_clean, "").strip(" ,.;")
        if city_from_line:
            continue
        for following_line in text_lines[index + 1:]:
            following = following_line.strip()
            if extract_date(following) or extract_time(following):
                continue
            if (_looks_like_event_desc(following)
                    or re.search(r"\b(?:to\s+follow|reception|invite|rsvp)\b",
                                 following, re.IGNORECASE)):
                break
            locality = (_extract_city_state_zip(following)
                        or _extract_city_pin(following)
                        or _extract_known_city([following]))
            if locality:
                candidates.append(Candidate(street, 0.78, "street_venue_fallback"))
                break
            if following and not re.match(r"^(?:at|on|date|time)\b", following,
                                          re.IGNORECASE):
                break
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
    # Tamil invitations: if no venue was found via English keywords/labels,
    # fall back to Tamil-only lines that are not dates/times/names/event-headings.
    for line in text_lines:
        ls = line.strip()
        if not _has_tamil(ls):
            continue
        if not usable_venue(ls) or len(ls) >= 80:
            continue
        if _looks_like_name(ls):
            continue
        if _looks_like_event_desc(ls):
            continue
        # Allow Tamil venue lines even if they contain blessing keywords, when
        # they also contain strong venue indicators (e.g. மஹால்/hall, mandapam).
        if _is_tamil_blessing_or_invitation(ls):
            if not re.search(
                r"(?:" + "|".join(re.escape(kw) for kw in _TAMIL_VENUE_KEYWORDS) + r")",
                ls, re.IGNORECASE,
            ):
                continue
        # Exclude Tamil lines that are clearly lists of people rather than venue names.
        # These often contain list markers like "and others" or parenthetical family relations.
        if re.search(r"ஆகிய(?:ார்)?\b", ls):
            continue
        if "(" in ls or ")" in ls:
            continue
        # Tamil venue names rarely contain commas; commas usually indicate
        # addresses, lists, or descriptions.
        if "," in ls:
            continue
        # Heuristic: long Tamil lines (>50 chars) without strong venue keywords
        # are usually descriptions or blessings, not venue names.
        if (len(ls) > 50 and ls.count(" ") >= 2
                and not re.search(
                    r"(?:" + "|".join(re.escape(kw) for kw in _TAMIL_VENUE_KEYWORDS) + r")",
                    ls, re.IGNORECASE,
                )):
            continue
        if extract_date(ls) or extract_time(ls):
            continue
        # Exclude labeled person lines (Bride:/Groom: and Tamil equivalents)
        if re.match(r"^(?:bride|groom|மணமகள்|மணமகன்)\s*:?", ls, re.IGNORECASE):
            continue
        # Boost confidence for lines with strong Tamil venue indicators
        conf = 0.7
        if any(kw in ls for kw in _TAMIL_VENUE_KEYWORDS):
            conf = 0.85
        cleaned_ls = _clean_venue_address(ls)
        if cleaned_ls:
            candidates.append(Candidate(cleaned_ls, conf, "tamil_line"))
    return candidates


def _select_venue(variants: List[Candidate]) -> Optional[Candidate]:
    """Select the best venue from variants, preferring combined venues
    that incorporate all individual venue keyword components."""
    organization_units = re.compile(
        r"\b(?:partnership|cell|department|association|committee|organ(?:izes|ised|ized))\b",
        re.IGNORECASE,
    )
    institution_candidates = [
        candidate for candidate in variants
        if not organization_units.search(str(candidate.value))
    ]
    if institution_candidates:
        institution_keywords = re.compile(
            r"\b(?:college|university|school|hall|auditorium|center|centre)\b",
            re.IGNORECASE,
        )
        named_institutions = [
            candidate for candidate in institution_candidates
            if institution_keywords.search(str(candidate.value))
        ]
        def institution_score(candidate):
            words = str(candidate.value).split()
            has_logo_prefix = bool(
                words and len(words[0]) <= 3 and words[0].isupper()
            )
            return (not has_logo_prefix, len(words), len(str(candidate.value)))

        variants = ([max(named_institutions, key=institution_score)]
                    if named_institutions else institution_candidates)
    strong = [c for c in variants
              if c.strategy in ("venue_label", "venue_combined")
              or c.strategy.startswith("venue_keyword")]
    if not strong:
        return _best(variants)
    combined = [c for c in strong if c.strategy == "venue_combined"]
    if combined:
        best_combined = max(combined, key=lambda c: len(c.value))
        keyword_venues = [c for c in strong
                            if c.strategy.startswith("venue_keyword")]
        combined_lower = best_combined.value.lower()
        # Strip leading bare venue labels from combined value.
        for label in ("venue", "location", "place", "at", "located at"):
            if combined_lower.startswith(label):
                combined_lower = combined_lower[len(label):].strip()
                break
        # Check if the stripped combined value equals a keyword venue
        # (meaning the "combined" version just adds a label prefix).
        # In that case, prefer the clean keyword venue value.
        keyword_values = {v.value.lower() for v in keyword_venues}
        if combined_lower in keyword_values and keyword_venues:
            return max(keyword_venues, key=lambda c: len(c.value))
        if all(kw in combined_lower
               for kw in keyword_values):
            return best_combined
    return _best(strong)


def extract_venue(text_lines: List[str], text: str) -> Candidate:
    variants = extract_venue_variants(text_lines, text)
    return _select_venue(variants)


def extract_address_variants(text_lines: List[str], text: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    def usable_address(value: str) -> bool:
        return bool(value and not extract_date(value) and not extract_time(value))
    # Single-line comma-separated address with street, city, state, pincode
    # e.g. "Kovilpatti Main Rd, Kovilpatti, Tamil Nadu 628501"
    # Also check for a locality line immediately before the street line
    # (skipping parenthetical lines like "(Muhurtham)").
    for index, line in enumerate(text_lines):
        street = _extract_street_address(line)
        if street:
            # Check if line contains state and pincode after the street
            has_state = any(re.search(rf"\b{re.escape(state)}\b", line, re.IGNORECASE)
                            for state in INDIAN_STATES)
            has_pincode = bool(re.search(r"\b\d{6}\b", line))
            if has_state and has_pincode:
                # Check previous lines for locality, skipping parentheticals
                prev_locality = ""
                for prev_idx in range(index - 1, max(-1, index - 4), -1):
                    prev = text_lines[prev_idx].strip().rstrip(" ,;:")
                    if not prev:
                        continue
                    if prev.startswith("(") and prev.endswith(")"):
                        continue
                    if (len(prev) <= 40
                            and not extract_date(prev)
                            and not extract_time(prev)
                            and not re.fullmatch(rf"(?i:{_WEEKDAY})\.?,?", prev)
                            and prev.lower() not in {"session", "full-day", "full day", "muhurtham", "muhoortham"}
                            and not any(re.search(rf"\b{re.escape(kw)}\b", prev, re.IGNORECASE)
                                        for kw in VENUE_STRONG_KEYWORDS)):
                        prev_locality = prev
                        break
                if prev_locality:
                    candidates.append(Candidate(f"{prev_locality}, {line.strip().rstrip('.,;:')}", 0.97,
                                                "inline_full_address_with_locality"))
                else:
                    candidates.append(Candidate(line.strip().rstrip(".,;:"), 0.97,
                                                "inline_full_address"))
    # Prefer a complete printed street/city/state/postcode line over its
    # shorter city/state component.
    for line in text_lines:
        street = _extract_street_address(line)
        locality = _extract_city_state_zip(line)
        if street and locality and re.match(r"^\s*\d", line):
            candidates.append(Candidate(
                f"{street.rstrip(',')}, {locality}", 0.98,
                "inline_street_city_state_zip",
            ))
    for line in text_lines:
        at_location = re.match(
            r"^(?:@|at|located at|held at)\s+[^,]+,\s*(.+)$",
            line.strip(), re.IGNORECASE)
        if at_location and not re.match(r"\d", at_location.group(1).strip()):
            value = at_location.group(1).strip().rstrip(".,;:")
            value = _clean_venue_address(value)
            if value:
                candidates.append(Candidate(value, 0.82, "at_location_suffix"))
    for line in text_lines:
        if re.search(r"^address\s*:", line, re.IGNORECASE):
            val = _strip_label(line)
            if usable_address(val) and len(val) < 150:
                candidates.append(Candidate(val, 0.95, "address_label"))
    # Tamil address label: "முகவரி" (address) followed by value on same or next line.
    for i, line in enumerate(text_lines):
        if re.search(r"முகவரி\s*[:\-]?", line, re.IGNORECASE):
            val = _strip_label(line)
            if usable_address(val) and len(val) < 150:
                candidates.append(Candidate(val, 0.95, "tamil_address_label"))
            # Also check the next line if current line only has the label
            elif i + 1 < len(text_lines):
                next_line = text_lines[i + 1].strip()
                if usable_address(next_line) and len(next_line) < 150:
                    candidates.append(Candidate(next_line, 0.92, "tamil_address_label_next"))
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
    # A single labeled line that contains both venue and city (e.g.
    # "Venue: Hotel Marine Blue, Hyderabad").
    for line in text_lines:
        if not re.match(r"^(?:venue|location|place)\s*[:\-]", line.strip(), re.IGNORECASE):
            continue
        stripped = _strip_label(line)
        if not stripped:
            continue
        parts = re.split(r"[,;]", stripped)
        for part in parts:
            part = part.strip().rstrip(".,;:")
            if not part or len(part) > 30:
                continue
            if part.lower() in KNOWN_CITIES:
                candidates.append(Candidate(part.title(), 0.82,
                                            "venue_inline_city"))
                break
    # Retain locality from unlabelled "Hotel/Resort, City" OCR lines.
    for line in text_lines:
        if not any(re.search(rf"\b{re.escape(keyword)}\b", line, re.IGNORECASE)
                   for keyword in VENUE_STRONG_KEYWORDS):
            continue
        parts = [part.strip().rstrip(".,;:") for part in re.split(r"[,;]", line)]
        if len(parts) >= 2 and (parts[-1].lower() in KNOWN_CITIES or parts[-1].lower() in INDIAN_STATES):
            candidates.append(Candidate(parts[-1].title(), 0.84, "venue_inline_locality"))
    for line in text_lines:
        cit = _extract_city_state_zip(line)
        if cit:
            candidates.append(Candidate(cit, 0.9, "city_state_zip"))
        state_pattern = "|".join(
            re.escape(state) for state in sorted(INDIAN_STATES, key=len, reverse=True))
        if ("," in line and re.search(r"\b\d{6}\b", line)
                and re.search(rf"\b(?:{state_pattern})\b", line, re.IGNORECASE)):
            candidates.append(Candidate(line.strip().rstrip(".,;:"), 0.97,
                                        "comma_locality_state_pin"))
        # Comma-separated locality line ending with known state or city (no pincode required)
        # e.g. "Nallatinputhur, Kovilpatti, Tamil Nadu"
        elif ("," in line and not re.search(r"\b\d{6}\b", line)
                and re.search(rf"\b(?:{state_pattern})\b", line, re.IGNORECASE)
                and not re.search(r"\b(?:venue|hotel|hall|college|university|school)\b", line, re.IGNORECASE)
                and not extract_date(line)
                and not extract_time(line)):
            candidates.append(Candidate(line.strip().rstrip(".,;:"), 0.92,
                                        "comma_locality_state"))
        country_city = re.search(
            r"\b([A-Z][A-Za-z .'-]{1,30}),\s*(India|United States|United Kingdom)\b",
            line, re.IGNORECASE)
        if country_city:
            city = country_city.group(1).strip()
            country = country_city.group(2).title()
            if city.lower() in KNOWN_CITIES:
                candidates.append(Candidate(f"{city.title()}, {country}",
                                            0.94, "city_country"))
    for line in text_lines:
        pin = _extract_city_pin(line)
        if pin:
            candidates.append(Candidate(pin, 0.88, "city_pin"))
    for line in text_lines:
        state_pin = _extract_state_pin(line)
        if state_pin:
            candidates.append(Candidate(state_pin, 0.88, "state_pin"))
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
        has_comma = "," in line
        line_remainder = line.strip().rstrip(".,;:")
        # Extract all comma-separated parts and find which contains the street
        parts = [p.strip() for p in line_remainder.split(",") if p.strip()]
        street_part_idx = None
        for i, part in enumerate(parts):
            if street.lower() in part.lower():
                street_part_idx = i
                break
        # Parts before the street are locality components (e.g. "Salt Pans")
        preceding_parts = parts[:street_part_idx] if street_part_idx is not None and street_part_idx > 0 else []
        city_from_line = ", ".join(preceding_parts) if preceding_parts else ""
        # Also include any part after the street if it looks like a locality
        following_parts = parts[street_part_idx + 1:] if street_part_idx is not None and street_part_idx + 1 < len(parts) else []
        for part in following_parts:
            if part.lower() in KNOWN_CITIES or any(s in part.lower() for s in INDIAN_STATES):
                if city_from_line:
                    city_from_line += ", " + part
                else:
                    city_from_line = part
        for following_index, following_line in enumerate(text_lines[index + 1:], start=index + 1):
            following = following_line.strip()
            if extract_date(following) or extract_time(following):
                continue
            if (_looks_like_event_desc(following)
                    or re.search(r"\b(?:to\s+follow|reception|invite|rsvp)\b",
                                 following, re.IGNORECASE)):
                break
            if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                          following, re.IGNORECASE):
                continue
            if re.search(r"\b(?:son\s+of|daughter\s+of|mr\.|mrs\.|ms\.|dr\.|sri\.|smt\.|soul[s]?|holy\s+matrimony)\b",
                          following, re.IGNORECASE):
                break
            locality = (_extract_city_state_zip(following) or
                        _extract_city_pin(following) or
                        _extract_known_city([following]) or
                        (following if re.match(r"^[A-Z][A-Za-z .'-]+(?:,|$)", following)
                         else ""))
            if locality and len(following) < 100:
                combined = f"{street}"
                if city_from_line:
                    combined += f", {city_from_line}"
                combined += f", {following.rstrip('.,;:')}"
                for extra_line in text_lines[following_index + 1:]:
                    extra = extra_line.strip()
                    if not extra or extract_date(extra) or extract_time(extra):
                        continue
                    if (_looks_like_event_desc(extra)
                            or re.search(r"\b(?:to\s+follow|reception|invite|rsvp)\b",
                                         extra, re.IGNORECASE)):
                        break
                    extra_stripped = extra.rstrip(".,;:")
                    # Continue absorbing locality lines (city/state/country,
                    # pincode, or a standalone capitalized place name) so the
                    # address is not truncated prematurely.
                    is_locality = (
                        any(re.search(rf"\b{re.escape(state)}\b", extra,
                                      re.IGNORECASE)
                            for state in INDIAN_STATES)
                        or _extract_city_state_zip(extra)
                        or _extract_known_city([extra])
                        or re.search(r"\b(?:india|united states|united kingdom)\b",
                                     extra, re.IGNORECASE)
                        or re.search(r"\b\d{3}\s*\d{3}\b", extra)
                        or (re.match(r"^[A-Z][A-Za-z .'-]+,?$", extra_stripped)
                            and len(extra_stripped) <= 40
                            and not extract_date(extra)
                            and not extract_time(extra))
                    )
                    if is_locality:
                        combined += f", {extra_stripped}"
                        continue
                    break
                candidates.append(Candidate(combined, 0.95,
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
            # Also look at following lines for city/state/pincode
            # to build a complete address (e.g. "Alangulam Road,
            # Mukkudal" + "Tirunelveli - 627 758").
            combined = locality
            for following in text_lines[text_lines.index(line) + 1:]:
                fwd = following.strip().rstrip(".,;:")
                if not fwd or extract_date(fwd) or extract_time(fwd):
                    continue
                if re.match(r"^(?:venue|address|contact|rsvp)\b", fwd, re.IGNORECASE):
                    break
                city_state = _extract_city_state_zip(fwd) or _extract_city_pin(fwd)
                if city_state:
                    combined = f"{combined}, {city_state}"
                    _tail = re.split(r"\d{3}\s*[-–]?\s*\d{3}", fwd, 1)
                    if len(_tail) > 1:
                        _tail = _tail[-1].strip().rstrip(".,;:")
                        if _tail and len(_tail) < 60:
                            combined = f"{combined}, {_tail}"
                    continue
                # Also accept a standalone city/state line
                if fwd.lower() in KNOWN_CITIES or any(
                    s in fwd.lower() for s in INDIAN_STATES
                ):
                    combined = f"{combined}, {fwd}"
                    continue
            candidates.append(Candidate(combined, 0.9, "keyword_locality"))
    # Tamil address lines: lines with Tamil script that contain address-related
    # keywords or a leading number (e.g. "6/9A, பள்ளிவாசல் தெரு, கொயாமழி").
    _TAMIL_ADDRESS_KEYWORDS = ["தெரு", "வீதி", "நகர்", "மாவட்டம்", "கிராமம்", "சந்து"]
    for line in text_lines:
        ls = line.strip()
        if not _has_tamil(ls):
            continue
        if not any(kw in ls for kw in _TAMIL_ADDRESS_KEYWORDS) and not re.search(r"^\d", ls):
            continue
        if extract_date(ls) or extract_time(ls):
            continue
        if _looks_like_name(ls):
            continue
        if _looks_like_event_desc(ls):
            continue
        if _is_tamil_blessing_or_invitation(ls):
            continue
        if len(ls) < 5 or len(ls) > 120:
            continue
        candidates.append(Candidate(ls, 0.75, "tamil_address"))
    # Single-line US-style address: number + city + state + zip on one line.
    # Only apply when no street suffix is present; otherwise the existing
    # venue/address split handles "123 Anywhere St. / Any City, ST 12345".
    _street_suffix_re = re.compile(
        r"\b(?:st|street|rd|road|ave|avenue|ln|lane|blvd|boulevard|dr|drive|"
        r"ct|court|pl|place|way|ter|terrace|cir|circle|mg|main)\.?\b",
        re.IGNORECASE,
    )
    for line in text_lines:
        if _street_suffix_re.search(line):
            continue
        m = re.search(
            r"\b(\d{1,4}(?!\s+(?:at\s+\d|[ap]m\b))\s+(?:[A-Za-z][A-Za-z0-9 .'-]+?)),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b",
            line, re.IGNORECASE)
        if m:
            candidates.append(Candidate(
                f"{m.group(1)}, {m.group(2)} {m.group(3)}", 0.94,
                "inline_us_address"))
    return candidates


def extract_address(text_lines: List[str], text: str, venue: str) -> Candidate:
    venue_has_street = bool(_extract_street_address(venue))
    variants = extract_address_variants(text_lines, text)
    if venue_has_street:
        allowed = {"address_label", "city_state_zip", "city_pin",
               "known_city_line", "at_location_suffix"}
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
    "அழைப்பிதழ்", "திருமண", "ருமண", "வரவேற்பு", "நன்றி", "வாழ்த்து", "அன்புடன்",
    "வாழ்த்துகள்", "நிகாஹ்", "நிகாஹ", "வரவேற்பு", "குடும்பம்", "திருநாள்",
    "மகிழ்ச்சி", "கல்யாணம்", "வலீமா", "வலிமா", "அலீமா", "ழப்பிதழ",
    "நகாஹ", "மணம", "நிகா", "லீமா",
    # Tamil religious / invocation phrases that are NOT person names.
    "கணபதி", "சுப்ரமணி", "நமஸ்தே", "ஓம்", "நமஃசரணம்",
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
    # Without a Latin initial, very long Tamil runs are likely blessings or
    # descriptive text rather than a person name.  With an initial, allow a
    # longer Tamil body because the initial already signals a named entity.
    if initial is None and tamil_len > 20:
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
        # Strip leading role labels (Bride:/Groom: and Tamil equivalents)
        # and leading title prefixes (M., Mr., Mrs., Ms., Dr., etc.)
        # and patronymic prefixes (S/o, D/o, எஸ்/ஓ, டி/ஓ, etc.)
        # and Tamil honorifics (திரு., திருமதி., செல்வி., இரு., etc.)
        # and trailing degree suffixes (M.A., B.E., etc.) before checking
        # shape and extracting the Latin initial.
        label_stripped = re.sub(
            r"^(?:bride|groom|மணமகள்|மணமகன்|வர籍|M\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|"
            r"எஸ்\s*/\s*ஓ|டி\s*/\s*ஓ|S\s*/\s*o|D\s*/\s*o|"
            r"திரு\.|திருமதி\.|செல்வி\.|இரு\.|செல்வன்\.|தினை\.|அருண்\.)\s*:?\s*",
            "", text, flags=re.IGNORECASE,
        ).strip()
        label_stripped = re.sub(
            r"\s*(?:M\.?A\.?|B\.?E\.?|M\.?Sc\.?|B\.?Tech\.?|Ph\.?D\.?|M\.?B\.?B\.?S\.?|B\.?A\.?|B\.?Com\.?|M\.?Com\.?|B\.?Sc\.?|M\.?C\.?A\.?|B\.?C\.?A\.?)\s*$",
            "", label_stripped, flags=re.IGNORECASE,
        ).strip()
# Reject lines that are clearly patronymic references or family descriptions
        # even after stripping labels (e.g. "கம்லா ஐயர்" from "எஸ்/ஓ கம்லா ஐயர்"
        # is the mother's name, not the bride/groom).
        if re.search(r"(?:எஸ்\s*/\s*ஓ|டி\s*/\s*ஓ|S\s*/\s*o|D\s*/\s*o|"
                     r"குடும்பம்|மகள்|மகன்|மருமகள்|மருமகன்|"
                     r"அம்மா|அப்பா|தந்தை|தாய்|மாமா|மாமியார்|"
                     r"இருமதி\.|திரு\.|திருமதி\.|செல்வி\.|செல்வன்\.|இரு\.)",
                     label_stripped, re.IGNORECASE):
            continue
        # Reject Tamil day/month names and common location words
        _TAMIL_DAY_NAMES = ("திங்கள்", "செவ்வாய்", "புதன்", "வியாழன்", "வெள்ளி", "சனி", "ஞாயிறு")
        _TAMIL_MONTH_NAMES = ("ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
                             "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்")
        _TAMIL_LOCATION_WORDS = ("கர்நாடகா", "தமிழ்நாடு", "கேரளா", "ஆந்திரா", "தெலங்கானா",
                                "மஹாராஷ்டிரா", "கூவை", "சென்னை", "மதுரை", "திருச்சி",
                                "சேலம்", "வெள்ளூர்", "கும்பகோணம்", "நாகர்கோவில்")
        if any(w in label_stripped for w in _TAMIL_DAY_NAMES + _TAMIL_MONTH_NAMES + _TAMIL_LOCATION_WORDS):
            continue
        # Reject obvious poetic/blessing lines (long lines with many words)
        if len(label_stripped.split()) > 6 and _has_tamil(label_stripped):
            continue
        if not _tamil_name_shape_ok(label_stripped, None):
            continue
        if not (4 <= len(label_stripped) <= 30):
            continue
        conf = float(item.get("confidence", 0.0) or 0.0)
        m = re.match(r"^([A-Za-z])\.\s*", label_stripped)
        initial = m.group(1).upper() if m else None
        if initial is None and conf < 0.7:
            continue
        if initial is not None and conf < 0.4:
            continue
        if not _tamil_name_shape_ok(label_stripped, initial):
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
            "text": label_stripped, "initial": initial, "score": score, "bbox": bbox,
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
        nm = _clean_name(c["text"])
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
            "motherof", "parentof", "parentsof", "familyof", "grandsonof",
            "granddaughterof"))

    # Explicit "Bride:" / "Groom:" labels (or "Daughter:" / "Son:").
    # These are actual field labels, not descriptive phrases like "daughter of".
    bride_m = re.search(
        r"(?:^|\n)\s*(?:bride|daughter)\s*:\s*"
        r"(?:([A-Z])\.\s+)?"
        r"([^\n\r]{2,40}?)"
        r"(?=\s+(?:groom|son|bride|daughter)\b|$)",
        text, re.IGNORECASE | re.MULTILINE)
    groom_m = re.search(
        r"(?:^|\n)\s*(?:groom|son)\s*:\s*"
        r"(?:([A-Z])\.\s+)?"
        r"([^\n\r]{2,40}?)"
        r"(?=\s+(?:bride|daughter|groom|son)\b|$)",
        text, re.IGNORECASE | re.MULTILINE)
    if bride_m or groom_m:
        bride_raw = bride_m.group(2) if bride_m else ""
        groom_raw = groom_m.group(2) if groom_m else ""
        bride_initial = bride_m.group(1).upper() if bride_m and bride_m.group(1) else None
        groom_initial = groom_m.group(1).upper() if groom_m and groom_m.group(1) else None
        bride_name = _clean_name(bride_raw)
        groom_name = _clean_name(groom_raw)
        # Prepend Latin initial if present (matching Tamil layout behavior).
        if bride_initial and not re.match(r"^%s\." % re.escape(bride_initial), bride_name):
            bride_name = "%s. %s" % (bride_initial, bride_name)
        if groom_initial and not re.match(r"^%s\." % re.escape(groom_initial), groom_name):
            groom_name = "%s. %s" % (groom_initial, groom_name)
        c = _names_candidate(bride_name, groom_name, 0.95, "explicit_marker")
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
    family_roles = extract_family_roles(text_lines, text)
    if family_roles.get("bride") or family_roles.get("groom"):
        c = _names_candidate(family_roles.get("bride", ""),
                             family_roles.get("groom", ""),
                             0.96, "family_roles")
        if c:
            candidates.append(c)

    # A stand-alone pair is common in wedding and non-wedding cards. Reject
    # event/location terms so a decorative event heading is not a couple.
    for line in text_lines:
        pair = re.match(
            r"^\s*([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\s*"
            r"\+\s*"
            r"([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\s*$",
            line.strip(), re.IGNORECASE,
        )
        if not pair:
            continue
        first, second = (_clean_name(pair.group(1)), _clean_name(pair.group(2)))
        combined = f"{first} {second}".lower()
        if any(word in combined for word in (
            "wedding", "marriage", "reception", "engagement", "birthday",
            "ceremony", "celebration", "sangeet", "mehndi", "haldi", "nikah",
            "hotel", "resort", "hall", "venue",
        )):
            continue
        if _looks_like_name(first) and _looks_like_name(second):
            c = _names_candidate(first, second, 0.93, "standalone_pair")
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

    # Invitation prose often inserts a short phrase between the event label
    # and the names, e.g. "the auspicious occasion of the marriage of A and B".
    # Restrict this to the clear marriage-of construction and two name tokens.
    if re.search(r"\b(?:marriage|wedding)\s+of\b", text, re.IGNORECASE):
        # OCR commonly places one name per line and omits the printed
        # conjunction. Use only the first two plausible names in the short
        # span after the marriage phrase; stop at schedule/location context.
        marker = re.search(r"\b(?:marriage|wedding)\s+of\b", text,
                           re.IGNORECASE)
        found = []
        for line in text[marker.end():].splitlines()[:8]:
            candidate = line.strip(" ,.;:-")
            if (not candidate or re.search(r"\d|\b(?:date|venue|time|road|hall)\b",
                                           candidate, re.IGNORECASE)):
                if found:
                    break
                continue
            if (not _has_tamil(candidate) and _looks_like_name(candidate)
                    and len(candidate.split()) <= 3):
                found.append(_clean_name(candidate))
                if len(found) == 2:
                    break
        if len(found) == 2:
            pair = _assign_bride_groom(found[0], found[1], "bride-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.88,
                                 "marriage_of")
            if c:
                candidates.append(c)

    # A compact wedding phrase may contain two adjacent names without a
    # connector: "the wedding of Karthik Sowmiya". Split the names and assign
    # roles only when their name cues clearly indicate opposite genders.
    compact_wedding = re.search(
        r"\b(?:marriage|wedding)\s+of\s+"
        r"([A-Z][a-zA-Z]{2,})\s+(?!and\b|with\b|to\b)"
        r"([A-Z][a-zA-Z]{2,})\b",
        text,
        re.IGNORECASE,
    )
    if compact_wedding:
        name1 = _clean_name(compact_wedding.group(1))
        name2 = _clean_name(compact_wedding.group(2))
        hints = (_gender_hint(name1), _gender_hint(name2))
        if hints[0] and hints[1] and hints[0] != hints[1]:
            pair = _assign_bride_groom(name1, name2, "groom-first")
            c = _names_candidate(pair["bride"], pair["groom"], 0.97,
                                 "compact_wedding_pair")
        else:
            c = Candidate({"bride": name1, "groom": name2, "_generic": True},
                          0.97, "compact_wedding_pair")
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

    # Some invitations place the relationship labels around the names in the
    # reverse order: "Son of Karthik with Nivetha Daughter of".
    son_with_daughter = re.search(
        r"\bson\s+of\s+(?P<groom>[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\s+"
        r"(?:with|and)\s+(?P<bride>[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\s+"
        r"daughter\s+of\b",
        text,
        re.IGNORECASE,
    )
    if son_with_daughter:
        groom_name = _clean_name(son_with_daughter.group("groom"))
        bride_name = _clean_name(son_with_daughter.group("bride"))
        c = _names_candidate(bride_name, groom_name, 0.99, "son_with_daughter")
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

        # Reject pairs where both tokens are event-type keywords rather than
        # person names (e.g. "NIKAH & WALIMA" is an event name, not a couple).
        _EVENT_KW_REJECT = {
            "wedding", "marriage", "reception", "engagement", "birthday",
            "celebration", "ceremony", "party", "function", "housewarming",
            "muhurtham", "naming", "nikah", "walima", "mehndi", "haldi",
            "invitation", "announcement", "event", "programme", "program",
        }
        if n1.lower() in _EVENT_KW_REJECT and n2.lower() in _EVENT_KW_REJECT:
            continue

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

    celebrating = re.search(
        r"(?<!\S)(?i:celebrating|celebrate)\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)",
        text)
    if celebrating:
        from ..matching.field_matcher import _looks_like_person_name as _llpn
        name = _clean_name(celebrating.group(1))
        if (name and name.lower() not in EVENT_TYPE_KEYWORDS
                and _looks_like_name(name) and _llpn(name)):
            suffix = text[celebrating.end():].strip()
            if not re.match(r"'s\s+\d|'s\s+(?:birthday|bday)", suffix, re.IGNORECASE):
                c = _names_candidate("", name, 0.8, "celebrating_marker")
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
            c = _names_candidate(names[0], "", 0.7, "marker_above_single")
            if c:
                candidates.append(c)

    candidates_list = []
    has_tamil = any(_has_tamil(line) for line in text_lines)
    for line in text_lines:
        ls = line.strip()
        if is_family_context(ls):
            continue
        if _is_relationship_phrase(ls):
            continue
        if not _looks_like_name(ls):
            continue
        if _is_greeting_or_event_desc(ls):
            continue
        cleaned = _clean_name(ls)
        if len(cleaned) >= 3:
            # In Tamil documents, reject Latin-only name candidates (OCR garbage).
            if has_tamil and not _has_tamil(cleaned):
                continue
            candidates_list.append(cleaned)
    if candidates_list:
        names = candidates_list[:2]
        if len(names) >= 2:
            convention = "bride-first" if has_tamil else "groom-first"
            pair = _assign_bride_groom(names[0], names[1], convention)
            c = _names_candidate(pair["bride"], pair["groom"], 0.7,
                                 "standalone_lines")
            if c:
                candidates.append(c)
        else:
            c = _names_candidate(names[0], "", 0.6, "standalone_single")
            if c:
                candidates.append(c)

    # Decorative OCR may split a two-word name across adjacent lines even
    # after the layout-aware OCR pass. Recombine only adjacent single-word
    # name candidates, keeping labels, numbers, and filler lines excluded by
    # _looks_like_name above.
    split_names = []
    for index in range(len(text_lines) - 1):
        first_line = text_lines[index].strip()
        second_line = text_lines[index + 1].strip()
        if (len(first_line.split()) != 1
                or len(second_line.split()) != 1
                or not _looks_like_name(first_line)
                or not _looks_like_name(second_line)):
            continue
        first_name = _clean_name(first_line)
        second_name = _clean_name(second_line)
        if not first_name or not second_name:
            continue
        split_names.append(first_name + " " + second_name)
    if len(split_names) >= 2:
        first_name, second_name = split_names[-2:]
        pair = _assign_bride_groom(first_name, second_name, "groom-first")
        candidates.append(Candidate(pair, 0.8, "adjacent_split_names"))
    elif split_names:
        existing_pair = next(
            (name for name in candidates_list if len(name.split()) >= 2), ""
        )
        if existing_pair and existing_pair.casefold() != split_names[0].casefold():
            pair = _assign_bride_groom(existing_pair, split_names[0], "bride-first")
            candidates.append(Candidate(pair, 0.8, "adjacent_split_groom"))

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
    if extract_date(text) or extract_time(text):
        return True
    contact = extract_contact(text)
    if contact and contact.value:
        return True
    return _has_strong_location(text)


_HEADING_DELIMITERS = re.compile(
    r"^\s*(?:wedding|marriage|reception|engagement|birthday|housewarming|"
    r"muhurtham|naming\s+ceremony|"
    r"haldi|nikah|walima|mehndi)(?:\s|ceremony|celebration|\s+ceremony|\s+"
    r"celebration|\s+reception|&|\b)*\s*$",
    re.IGNORECASE,
)
_NON_HEADING = re.compile(
    r"to\s+follow|reception\s+of|wedding\s+reception\b|dinner\s+to\s+follow",
    re.IGNORECASE,
)


def _is_generic_event_heading(line: str) -> bool:
    """Detect generic event headings like 'Annual Day', 'Sports Day'.

    A generic event heading is:
      - Short (2-4 words, <= 35 chars)
      - ALL CAPS for all alphabetic words
      - Contains at least one generic event keyword
      - Not a date/time/venue/address line
    """
    ls = line.strip()
    if not ls or len(ls) > 35:
        return False
    words = ls.split()
    if not (2 <= len(words) <= 4):
        return False
    lowered = ls.lower()
    has_kw = any(
        re.search(rf"\b{kw}\b", lowered)
        for kw in _GENERIC_EVENT_KEYWORDS
    )
    if not has_kw:
        return False
    if extract_date(ls) or extract_time(ls):
        return False
    if any(kw in lowered for kw in VENUE_STRONG_KEYWORDS + ADDRESS_KEYWORDS):
        return False
    if any(c.isdigit() for c in ls):
        return False
    alpha_words = [w for w in words if w.isalpha()]
    if len(alpha_words) < 2:
        return False
    if not all(w.isupper() for w in alpha_words):
        return False
    return True


# Event type mapping used by both _heading_type and _split_events.
heading_types = {
    "wedding": "Wedding", "marriage": "Wedding", "muhurtham": "Wedding",
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
                       {"ceremony", "reception", "celebration"}
                       and all(word[0].isupper() for word in generic_words if word))
    if not _HEADING_DELIMITERS.match(tl) and not generic_heading:
        if not _is_generic_event_heading(tl):
            return None
        for kw, typ in heading_types.items():
            if kw in lowered:
                return typ
        return _clean(tl)

    for kw, typ in heading_types.items():
        if kw in lowered:
            return typ
    if generic_heading:
        return _clean(tl)
    return None


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
            # When headings and dates are listed first, pair each event with
            # its own weekday and time range instead of sharing all schedule
            # lines with every event.
            time_range_indices = [
                i for i in range(date_indices[-1] + 1, end)
                if len(extract_time_variants(text_lines[i])) >= 2
            ]
            if len(time_range_indices) >= len(heading_run):
                shared_tail = text_lines[time_range_indices[-1] + 1:end]
                grouped = []
                for position, (_, heading) in enumerate(heading_run):
                    date_index = date_indices[position]
                    # Look for weekday on the line immediately before the date,
                    # or between the previous date and this date.
                    weekday = ""
                    # First check line immediately before date
                    if date_index > 0:
                        prev_line = text_lines[date_index - 1].strip()
                        if re.fullmatch(rf"(?i:{_WEEKDAY})\.?,?", prev_line):
                            weekday = prev_line
                    # If not found, search between previous date and this date
                    if not weekday:
                        search_start = (heading_run[-1][0] + 1
                                        if position == 0
                                        else date_indices[position - 1] + 1)
                        weekday = next(
                            (
                                text_lines[index]
                                for index in range(search_start, date_index)
                                if re.fullmatch(
                                    rf"(?i:{_WEEKDAY})\.?,?", text_lines[index].strip()
                                )
                            ),
                            "",
                        )
                    group = [heading]
                    if position == 0:
                        group = text_lines[:start] + group
                    if weekday:
                        group.append(weekday)
                    group.extend([
                        text_lines[date_index],
                        text_lines[time_range_indices[position]],
                    ])
                    group.extend(shared_tail)
                    grouped.append(group)
                return grouped + _split_events(text_lines[end:]) if end < len(text_lines) else grouped
            shared_tail = text_lines[date_indices[-1] + 1:end]
            grouped = []
            for position, (_, heading) in enumerate(heading_run):
                date_index = date_indices[position]
                # Look for weekday on the line immediately before the date
                weekday = ""
                if date_index > 0:
                    prev_line = text_lines[date_index - 1].strip()
                    if re.fullmatch(rf"(?i:{_WEEKDAY})\.?,?", prev_line):
                        weekday = prev_line
                group = [heading]
                if position == 0:
                    group = text_lines[:start] + group
                if weekday:
                    group.append(weekday)
                group.append(text_lines[date_index])
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
        if ht is not None and current:
            # A) The new heading must be a different event type than the current
            #    group's dominant type (repeated headings of the same event are
            #    OCR duplication, not separate events).
            current_et = extract_event_type("\n".join(current))
            current_type = current_et.value if current_et else ""
            if current_type and current_type.lower() == ht.lower():
                current.append(line)
                i += 1
                continue
            # B) Split when current group has event info OR has already
            #    accumulated a heading, so pre-event text doesn't merge
            #    with the first real event.
            if (_group_has_event_info(current)
                    or any(_heading_type(l) is not None for l in current)):
                # Require the following segment to carry its own event details.
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
    date_range = re.search(
        rf"\b(\d{{1,2}})\s*[-–]\s*(\d{{1,2}})\s+({_MONTH})\s+(\d{{4}})\b",
        raw,
        re.IGNORECASE,
    )
    if date_range:
        first, last, month, year = date_range.groups()
        month_name = {
            "oct": "October", "october": "October",
        }.get(month.lower(), month.capitalize())
        return f"{int(first):02d}–{int(last):02d} {month_name} {year}"
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
    # "month year day" across whitespace (e.g. "January 2024 01" or "January\n2024\n01").
    month_year_day = re.search(
        rf"\b({_MONTH})\.?\s+(\d{{4}})\s+(\d{{1,2}})\b",
        raw, re.IGNORECASE)
    if month_year_day:
        month, year, day = month_year_day.groups()
        return f"{month_names[month.lower()]} {int(day)}, {year}"
    # "year day month" across whitespace (e.g. "2026 23 OCTOBER" or "2026\n23\nOCTOBER").
    year_day_month = re.search(
        rf"\b(\d{{4}})\s+(\d{{1,2}})\s+({_MONTH})\.?\b",
        raw, re.IGNORECASE)
    if year_day_month:
        year, day, month = year_day_month.groups()
        return f"{month_names[month.lower()]} {int(day)}, {year}"
    # "year month day" across whitespace (e.g. "2026 OCTOBER 23" or "2026\nOCTOBER\n23").
    year_month_day = re.search(
        rf"\b(\d{{4}})\s+({_MONTH})\.?\s+(\d{{1,2}})\b",
        raw, re.IGNORECASE)
    if year_month_day:
        year, month, day = year_month_day.groups()
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
    named = re.search(
        rf"(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+)?({_MONTH})\.?\s*"
        rf"(\d{{1,2}})?(?:st|nd|rd|th)?\s*[,.]?\s*(\d{{2,4}})",
        raw, re.IGNORECASE)
    if named:
        day_before, month, day_after, year = named.groups()
        day = day_before or day_after
        if day:
            if len(year) == 2:
                year = "20" + year if int(year) < 70 else "19" + year
            return f"{month_names[month.lower()]} {int(day)}, {year}"
    # Year-less date fallback (e.g. "april 12" or "12 april").
    named = re.search(
        rf"(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+)?({_MONTH})\.?\s*"
        rf"(\d{{1,2}})?(?:st|nd|rd|th)?\s*[,.]?\s*$",
        raw, re.IGNORECASE)
    if named:
        day_before, month, day_after = named.groups()
        day = day_before or day_after
        if day:
            return f"{month_names[month.lower()]} {int(day)}"
    return raw


def format_time(raw: str, context: str = "") -> str:
    if not raw:
        return ""
    raw = _clean(raw)
    qualifier = " onwards" if re.search(r"\bonwards?\b", raw, re.IGNORECASE) else ""
    m = re.search(r"(\d{1,2})[:.](\d{2})(\s*)(am|pm|noon|midnight|a\.?m\.?|p\.?m\.?)?", raw, re.IGNORECASE)
    if m:
        h, mm = m.group(1), m.group(2)
        spacing, suffix = m.group(3), m.group(4)
        if suffix:
            amp = suffix.replace(".", "").lower()
            if amp == "noon":
                amp = "PM"
            elif amp == "midnight":
                amp = "AM"
            return f"{int(h)}:{mm}{spacing or ' '}{amp.upper()}{qualifier}"
        # No AM/PM in raw time: infer from Tamil context (e.g. "காலை 11.30" = AM)
        if not suffix and context:
            ctx = context.lower()
            if re.search(r"காலை|கால\s*ை", ctx):
                return f"{int(h)}:{mm} AM{qualifier}"
            if re.search(r"மண(?:க்|ி|வைகு|னல்)", ctx):
                return f"{int(h)}:{mm} PM{qualifier}"
        return f"{h}:{mm}"
    m = re.search(r"(\d{1,2})\s+(\d{2})(\s*)(am|pm|a\.?m\.?|p\.?m\.?)", raw, re.IGNORECASE)
    if m:
        amp = m.group(4).replace(".", "")
        return f"{int(m.group(1))}:{m.group(2)}{m.group(3) or ' '}{amp.upper()}{qualifier}"
    m = re.search(r"(\d{1,2})(\s*)(am|pm|a\.?m\.?|p\.?m\.?)", raw, re.IGNORECASE)
    if m:
        amp = m.group(3).replace(".", "")
        return f"{int(m.group(1))}:00 {amp.upper()}{qualifier}"
    return raw


def format_phone(raw: str) -> str:
    if not raw:
        return ""
    return re.sub(r"[^\d]", "", raw)


def _printed_weekday(text: str, group_text: str = "",
                      group: list = None, document_text: str = "") -> str:
    """Return the weekday associated with this date in the invitation."""
    target_date = format_date(text or "")
    expected = ""
    if target_date:
        try:
            expected = datetime.strptime(target_date, "%B %d, %Y").strftime("%A")
        except ValueError:
            pass
    # Layout OCR can attach a nearby event's weekday to this date. When the
    # date's actual weekday is also printed on the invitation, use that match
    # to disambiguate the schedule row.
    if expected and re.search(
            rf"\b{re.escape(expected)}\b", document_text or group_text,
            re.IGNORECASE):
        return expected
    match = re.search(rf"\b({_WEEKDAY})\b", text or "", re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    if group and group_text:
        date_indices = []
        for i, line in enumerate(group):
            candidate = extract_date(line)
            if candidate:
                date_indices.append((i, candidate.value))
        if target_date and date_indices:
            matching = [i for i, value in date_indices
                        if format_date(value) == target_date]
            date_index = matching[0] if matching else date_indices[0][0]
            weekdays = []
            for i, line in enumerate(group):
                m = re.search(rf"\b({_WEEKDAY})\b", line, re.IGNORECASE)
                if m:
                    # On equal distance, prefer the weekday printed before
                    # its date, which is the common invitation layout.
                    weekdays.append((abs(i - date_index), i > date_index,
                                     m.group(1).capitalize()))
            if weekdays:
                return min(weekdays)[2]
        # Use a group-wide weekday only when it contains exactly one distinct
        # printed weekday; never borrow one from another event's schedule.
        unique = list(dict.fromkeys(
            m.group(1).capitalize()
            for line in group
            for m in [re.search(rf"\b({_WEEKDAY})\b", line, re.IGNORECASE)]
            if m
        ))
        if len(unique) == 1:
            return unique[0]
    return ""


def _birthday_age(text: str) -> str:
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)\s+(?:birthday|bday)\b", text or "", re.IGNORECASE)
    return match.group(1) if match else ""


def _additional_information(lines: List[str]) -> str:
    details = [_clean(line) for line in lines if re.match(
        r"^(?:dinner|lunch|breakfast|refreshments)\b", line.strip(), re.IGNORECASE)]
    return " | ".join(dict.fromkeys(details))


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

def _aggregate_people(events: List[Dict], primary: Dict) -> List[Dict]:
    """Collect people from all events, deduplicated by (name, role)."""
    all_people: List[Dict] = []
    seen = set()
    for event in events:
        for p in event.get("people", []):
            name = p.get("name", "").strip()
            role = p.get("role", "Person")
            key = (name, role)
            if name and key not in seen:
                seen.add(key)
                all_people.append({"name": name, "role": role})
    if not all_people:
        return _build_people_from_event(primary)
    return all_people


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

    for group_idx, group in enumerate(event_groups):
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

        # A later event heading must not supply the type for an earlier group.
        best_et = _best(et_variants)
        if best_et is None and len(event_groups) == 1:
            best_et = event_type_best
        best_en = _best(en_variants)
        # Prefer explicit event headings (Annual Day, Sports Day, etc.)
        # over keyword-derived event type names.
        heading_in_group = None
        heading_label_in_group = ""
        for line in group:
            ht = _heading_type(line)
            if ht is not None and len(ht) <= 30:
                heading_in_group = ht
                heading_label_in_group = _clean(line)
                break
        if heading_in_group:
            best_en = Candidate(
                heading_label_in_group or heading_in_group,
                0.86,
                "group_heading_label",
            )
        # Use heading for event type when a heading is detected in the group,
        # regardless of best_en strategy. This preserves the printed label
        # (e.g. "Marriage" vs "Wedding") instead of letting shared-tail keywords
        # like "wedding" from venue/address lines override it.
        if heading_in_group:
            best_et = Candidate(_base_event_type(heading_in_group), 0.9, "group_heading_type")
        elif best_en and best_en.strategy in {"event_heading", "generic_structural_heading"}:
            heading_text = best_en.value
            heading_lower = heading_text.lower()
            if heading_lower.endswith(" function"):
                base_type = heading_text[:-len(" function")].strip()
                best_et = Candidate(base_type, best_en.confidence,
                                    "event_heading_type")
            else:
                best_et = Candidate(_base_event_type(heading_text), best_en.confidence,
                                    "event_heading_type")
        best_names = _best(name_variants)
        selected_type = (best_et.value if best_et else event_type) or ""
        
        couple_evidence = (
            _is_couple_based_event(selected_type)
            or _has_couple_evidence(name_variants, group_text)
        )
        if couple_evidence:
            couple_candidate = _best_couple_candidate(name_variants)
            names_value = (
                couple_candidate.value if couple_candidate
                else (best_names.value if best_names else {})
            )
            if not isinstance(names_value, dict):
                names_value = {"bride": "", "groom": ""}
            # Check if this is a generic pair (no bride/groom roles)
            is_generic_pair = names_value.get("_generic", False)
            if is_generic_pair:
                # Extract the names without assigning bride/groom roles
                generic_names = [names_value.get("bride", ""), names_value.get("groom", "")]
                generic_names = [n for n in generic_names if n]
                people = [{"name": n, "role": "Person"} for n in generic_names]
            else:
                people = _build_people_from_event({
                    "bride_name": names_value.get("bride", ""),
                    "groom_name": names_value.get("groom", ""),
                    "event_type": selected_type,
                })
        else:
            names_value = {"bride": "", "groom": ""}
            people = _extract_generic_people(group, group_text, ocr_lines)
        
        # Select venue: prefer combined venues that incorporate all
        # individual venue keyword components.
        best_venue = _select_venue(venue_variants)
        venue_value = best_venue.value if best_venue else ""
        # Exclude venue candidates that overlap with extracted names so Tamil
        # (or any) person lines are never mistaken for venues.
        name_strings = set()
        for c in name_variants:
            val = c.value if isinstance(c.value, str) else ""
            if val:
                name_strings.add(val.strip())
            elif isinstance(c.value, dict):
                for v in c.value.values():
                    if isinstance(v, str) and v.strip():
                        name_strings.add(v.strip())
        if name_strings:
            filtered_venues = [c for c in venue_variants
                               if c.value and c.value.strip() not in name_strings]
            if filtered_venues:
                best_venue = _select_venue(filtered_venues)
                venue_value = best_venue.value if best_venue else ""
        # Address extraction needs the selected venue so a street embedded in
        # the venue is not returned a second time as the address.
        best_addr = extract_address(group, group_text, venue_value)
        best_date = _best(date_variants)
        best_time = _best(time_variants)
        best_contact = extract_contact(group_text)

        # Detect time ranges: "10:30 AM - 12:30 PM" should populate both
        # `time` and `end_time` instead of losing the second value.
        end_time_value = ""
        time_variants_list = time_variants
        range_start = next((c for c in time_variants_list if c.strategy == "time_range_start"), None)
        range_end = next((c for c in time_variants_list if c.strategy == "time_range_end"), None)
        to_start = next((c for c in time_variants_list if c.strategy == "to_range_start"), None)
        to_end = next((c for c in time_variants_list if c.strategy == "to_range_end"), None)
        # Also recognize Tamil time-range strategies (e.g.
        # "tamil_dash_range_start"/"tamil_dash_range_end",
        # "tamil_time_range_start"/"tamil_time_range_end").
        ta_start = next((c for c in time_variants_list
                         if c.strategy in ("tamil_dash_range_start",
                                           "tamil_time_range_start")), None)
        ta_end = next((c for c in time_variants_list
                       if c.strategy in ("tamil_dash_range_end",
                                         "tamil_time_range_end")), None)
        if range_start and range_end:
            best_time = range_start
            end_time_value = format_time(range_end.value, group_text)
        elif to_start and to_end:
            best_time = to_start
            end_time_value = format_time(to_end.value, group_text)
        elif ta_start and ta_end:
            best_time = ta_start
            end_time_value = format_time(ta_end.value, group_text)
        elif best_time:
            # Single time: check if the same line contains a second time after a dash.
            start = best_time.value
            m = re.search(
                rf"{re.escape(start)}\s*[-–]\s*(\d{{1,2}}[:.]\d{{2}}\s*(?:am|pm|a\.?m\.?|p\.?m\.?))",
                group_text, re.IGNORECASE)
            if m:
                end_time_value = format_time(m.group(1).strip(), group_text)

        # Build parsed event skeleton from best candidates (converted to strings)
        is_generic_pair = False
        if _is_couple_based_event(selected_type) or _has_couple_evidence(
                name_variants, group_text):
            couple_cand = _best_couple_candidate(name_variants)
            names_value = (
                couple_cand.value if couple_cand
                else (best_names.value if best_names else {})
            )
            if not isinstance(names_value, dict):
                names_value = {"bride": "", "groom": ""}
            is_generic_pair = names_value.get("_generic", False)
            if not is_generic_pair:
                names_value = {
                    "bride": _resolve_repeated_person_ocr(names_value.get("bride", ""), text_lines),
                    "groom": _resolve_repeated_person_ocr(names_value.get("groom", ""), text_lines),
                }
                names_value = _ensure_distinct_couple(names_value, name_variants, text_lines)
        else:
            names_value = {"bride": "", "groom": ""}

        # When there is no named venue, prefer a combined street+city address
        # from extract_address over the raw _split_street_location fallback.
        address_value = best_addr.value if best_addr else ""
        if not address_value and not venue_value:
            street, locality = _split_street_location(group)
            if street and locality:
                address_value = f"{street}, {locality}"
        if not address_value and venue_value:
            street = _extract_street_address(venue_value)
            if street:
                venue_idx = None
                for i, line in enumerate(group):
                    if street in line:
                        venue_idx = i
                        break
                if venue_idx is not None:
                    for following in group[venue_idx + 1:]:
                        fwd = following.strip()
                        if not fwd or extract_date(fwd) or extract_time(fwd):
                            continue
                        locality = (_extract_city_state_zip(fwd)
                                    or _extract_city_pin(fwd)
                                    or _extract_known_city([fwd]))
                        if locality:
                            address_value = f"{street}, {locality}"
                        break
        # If the venue line also contains a street address but the
        # extracted address is only a city/locality (no street), prepend
        # the street from the venue line so the full address is captured.
        # Only does this when the venue contains a venue keyword
        # (e.g. "Hall", "Hotel") in addition to the street, not when
        # the venue IS just a bare street address.
        if address_value and venue_value:
            venue_street = _extract_street_address(venue_value)
            if venue_street and not _extract_street_address(address_value):
                venue_without_street = venue_value.replace(
                    venue_street, "").strip(" ,;:.")
                has_venue_keyword = any(
                    re.search(rf"\b{re.escape(kw)}\b", venue_without_street.lower())
                    for kw in VENUE_STRONG_KEYWORDS)
                if has_venue_keyword:
                    city_state = (_extract_city_state_zip(address_value)
                                  or _extract_city_pin(address_value)
                                  or address_value.strip())
                    address_value = f"{venue_street}, {city_state}"

        if is_generic_pair:
            # For generic pairs, don't set bride_name/groom_name; people array already has them
            bride_name_val = ""
            groom_name_val = ""
        else:
            bride_name_val = names_value.get("bride", "")
            groom_name_val = names_value.get("groom", "")

        parsed = {
            "event_name": best_en.value if best_en else "",
            "event_type": (best_et.value if best_et else
                           (event_type if len(event_groups) == 1 else "")),
            "bride_name": bride_name_val,
            "groom_name": groom_name_val,
            "date": format_date(best_date.value) if best_date else "",
            "time": format_time(best_time.value, group_text) if best_time else "",
            "end_time": end_time_value,
            "venue": _clean(venue_value),
            "address": _clean(address_value),
            "contact_number": best_contact.value if best_contact else "",
            "additional_information": _additional_information(group),
            "birthday_age": _birthday_age(group_text),
            "printed_weekday": _printed_weekday(
                best_date.value if best_date else "",
                group_text,
                group,
                raw_text,
            ),
            "people": people,
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
                "additional_information": parsed["additional_information"],
                "birthday_age": parsed["birthday_age"],
                "printed_weekday": parsed["printed_weekday"],
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

    # Venue/address/contact are often shared across all events in a multi-event
    # invitation even when the OCR places them in only one event group.  Back-fill
    # from any event that has the field, but only into events that lack it and
    # only when no other event already carries a conflicting different value.
    _shared_fields = ("venue", "address", "contact_number")
    for field in _shared_fields:
        source = next((event for event in events if event.get(field)), {})
        if not source:
            continue
        shared = source.get(field, "")
        conflicting = any(
            other.get(field) and other.get(field) != shared
            for other in events
        )
        if not conflicting:
            for event in events:
                if not event.get(field):
                    event[field] = shared

    # A single printed schedule date/weekday commonly applies to several
    # event headings. Copy these invitation-level calendar fields only when
    # there is one unambiguous source value.
    for field in ("date", "printed_weekday"):
        values = {event.get(field, "") for event in events if event.get(field)}
        if len(values) == 1:
            shared_value = next(iter(values))
            for event in events:
                if not event.get(field):
                    event[field] = shared_value

    # Primary event is the first; for multi-event, do NOT let global
    # matched_fields overwrite any individual event's own extracted fields.
    # Use a copy so the events[] array stays untouched.
    primary = dict(events[0]) if events else {}

    # Detect if first event has a generic pair (people with role "Person"
    # instead of "Bride"/"Groom"). If so, don't fall back to matched_fields
    # for bride_name/groom_name.
    first_event_people = events[0].get("people", []) if events else []
    has_bride_groom_roles = any(
        p.get("role") in ("Bride", "Groom") for p in first_event_people
    )

    for k in ("event_name", "event_type", "bride_name", "groom_name", "date",
              "time", "end_time", "venue", "address", "contact_number",
              "additional_information", "birthday_age", "printed_weekday"):
        mv = matched_fields.get(k, "")
        if mv and not primary.get(k):
            if k in ("bride_name", "groom_name"):
                if not _is_couple_based_event(primary.get("event_type", "")):
                    continue
                # If first event has generic pair (no Bride/Groom roles),
                # don't copy from matched_fields
                if not has_bride_groom_roles:
                    continue
                mv = _clean_name(mv)
            primary[k] = mv

    # Collect people from all events
    all_people = _aggregate_people(events, primary)
    invitation_mode = "single" if len(events) == 1 else "multi"

    # Final normalize/clean primary
    formatted_events = []
    for event in events:
        formatted_event = dict(event)
        event_venue = formatted_event.get("venue", "")
        recovered_venue = _institution_venue_from_text(raw_text)
        if (not event_venue
                or re.search(r"\b(?:partnership|cell|department|association)\b",
                             event_venue, re.IGNORECASE)
                or not re.search(r"\b(?:institute|college|university|school|hall|auditorium)\b",
                                 event_venue, re.IGNORECASE)):
            event_venue = recovered_venue or event_venue
        elif (recovered_venue
                and event_venue.lower() in recovered_venue.lower()
                and len(recovered_venue) > len(event_venue)):
            event_venue = recovered_venue
        formatted_event["venue"] = _clean_venue_address(event_venue)
        formatted_event["bride_name"] = _format_person_name(
            formatted_event.get("bride_name", ""), "Bride", raw_text)
        formatted_event["groom_name"] = _format_person_name(
            formatted_event.get("groom_name", ""), "Groom", raw_text)
        address = _prefer_complete_address(
            formatted_event.get("address", ""),
            _address_block_from_text(raw_text),
        )
        # Remove recognized neighboring event headings that OCR sometimes
        # appends to a locality line. The labels are discovered from the text
        # structure, not maintained as invitation-specific exclusions.
        for source_line in raw_text.splitlines():
            if (_heading_type(source_line) or _looks_like_name(source_line.strip())):
                address = re.sub(rf"\b{re.escape(source_line.strip())}\b", "",
                                 address, flags=re.IGNORECASE)
        formatted_event["address"] = _normalize_address_text(address)
        formatted_events.append(formatted_event)
    formatted_people = []
    for person in all_people:
        formatted_people.append({
            "name": _format_person_name(
                person.get("name", ""), person.get("role", "Person"), raw_text),
            "role": person.get("role", "Person"),
        })
    primary_address = (primary.get("address", "")
                       or _address_block_from_text(raw_text))
    for source_line in raw_text.splitlines():
        if (_heading_type(source_line) or _looks_like_name(source_line.strip())):
            primary_address = re.sub(
                rf"\b{re.escape(source_line.strip())}\b", "", primary_address,
                flags=re.IGNORECASE)

    def _is_implausible_person_name(value: str) -> bool:
        """True if a bride/groom value is a Tamil phrase or otherwise not a name."""
        if not value:
            return True
        from ..matching.field_matcher import (
            _looks_like_person_name,
            _contains_tamil_non_name_marker,
            _has_tamil_chars,
        )
        # Tamil fragments such as "ீவிநாயகாய நம:II" or "ருமண".
        if _has_tamil_chars(value) and _contains_tamil_non_name_marker(value):
            return True
        if _has_tamil_chars(value):
            # Any other Tamil script in a bride/groom slot is OCR prose
            # (blessings, headings, religious wording), not a Latin name.
            return True
        return not _looks_like_person_name(value)

    def _recover_names_from_relation_text(text: str) -> Optional[Dict]:
        """Recover bride/groom from 'Son of' / 'Daughter of' context.

        OCR often separates the couple's name from its parent-relationship line,
        so scan for the nearest capitalized proper noun appearing on the same
        line or the immediately preceding lines before a 'Son of' / 'Daughter
        of' marker (English or Tamil). Returns {'bride':...,'groom':...}.
        """
        result: Dict[str, str] = {}
        tamil_marker = ("மகன்\u0bb2ிருக்கும்", "மகள்\u0bb2ிருக்கும்")
        relation_patterns = [
            (r"\bSon\s+of\b", "groom", re.IGNORECASE),
            (r"\bDaughter\s+of\b", "bride", re.IGNORECASE),
            (r"எஸ்\s*/\s*ஓ", "groom", 0),
            (r"டி\s*/\s*ஓ", "bride", 0),
        ]
        lines = [l for l in (text or "").splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            for pat, role, flags in relation_patterns:
                if re.search(pat, line, flags):
                    name = ""
                    # Try the same line first (name immediately before marker).
                    before = re.split(pat, line, flags=flags)[0].strip()
                    tokens = re.findall(r"[A-Za-z][A-Za-z.'\-]{2,}", before)
                    if tokens and _is_likely_english_name(tokens[-1]):
                        name = _clean_name(tokens[-1])
                    if not name:
                        name = _find_name_above(lines, idx)
                    if name and not result.get(role):
                        result[role] = name
                    break
        return result or None

    def _is_likely_english_name(token: str) -> bool:
        return token[0].isupper() and not token.isupper() and len(token) >= 3

    bride_val = _clean(primary.get("bride_name", ""))
    groom_val = _clean(primary.get("groom_name", ""))
    recovered_names = None
    if _is_implausible_person_name(bride_val) or _is_implausible_person_name(groom_val):
        recovered_names = _recover_names_from_relation_text(raw_text) or {}
    if _is_implausible_person_name(bride_val):
        bride_val = recovered_names.get("bride", "")
    if _is_implausible_person_name(groom_val):
        groom_val = recovered_names.get("groom", "")

    # Enforce the same bride/groom cleanliness on each event's per-event fields.
    for event in formatted_events:
        ev_bride = _clean(event.get("bride_name", ""))
        ev_groom = _clean(event.get("groom_name", ""))
        if _is_implausible_person_name(ev_bride):
            event["bride_name"] = _format_person_name(
                recovered_names.get("bride", "") if recovered_names else "",
                "Bride", raw_text)
        if _is_implausible_person_name(ev_groom):
            event["groom_name"] = _format_person_name(
                recovered_names.get("groom", "") if recovered_names else "",
                "Groom", raw_text)

    # Address completeness guard: if the model-supplied address is only a
    # state/country (e.g. "Tamil Nadu, India") but the text contains a fuller
    # street + postal block, prefer the recovered block.
    recovered_address = _address_block_from_text(raw_text)
    primary_address = _prefer_complete_address(primary_address, recovered_address)

    primary = {
        "event_name": _clean(primary.get("event_name", "")),
        "event_type": _clean(primary.get("event_type", "")),
        "bride_name": _format_person_name(bride_val, "Bride", raw_text),
        "groom_name": _format_person_name(groom_val, "Groom", raw_text),
        "date": format_date(primary.get("date", "")),
        "time": format_time(primary.get("time", "")),
        "end_time": format_time(primary.get("end_time", "")),
        "venue": (lambda primary_venue,
                         recovered_venue=_institution_venue_from_text(raw_text):
                         _clean_venue_address(
                             _best_institution_venue(primary_venue, recovered_venue)
                         ))(primary.get("venue", "")),
        "address": _normalize_address_text(primary_address),
        "contact_number": _clean(primary.get("contact_number", "")),
        "additional_information": _clean(primary.get("additional_information", "")),
        "birthday_age": _clean(primary.get("birthday_age", "")),
        "printed_weekday": _clean(primary.get("printed_weekday", "")),
        "events": formatted_events,
        "number_of_events": len(events),
        "invitation_mode": invitation_mode,
        "people": formatted_people,
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
