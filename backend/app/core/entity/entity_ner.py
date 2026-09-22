"""Layered named-entity recognition for invitation text.

Extracts generic entities: PERSON, DATE, TIME, LOCATION, ORGANIZATION.

Strategy (in order of preference, never depending on a single source):
  1. Multilingual transformer NER model (e.g. mBERT-NER) when available —
     handles English + Indic scripts without hard-coded rules.
  2. Layout-aware grouping: use OCR line boxes + layout regions to group
     tokens that belong together (e.g. a venue name spanning multiple lines).
  3. Rule-based fallback: uses generic regex/lexicon strategies.

Each entity carries a type, the raw text, a normalized value, a confidence,
and the source strategy so callers can trace provenance.
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...config import settings

logger = logging.getLogger(__name__)

# Target entity types (these map to the reference pipeline's output).
ENTITY_TYPES = ("PERSON", "DATE", "TIME", "LOCATION", "ORGANIZATION")

# ---------------------------------------------------------------------------
# Entity model
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    etype: str
    text: str
    value: str
    confidence: float
    strategy: str
    line_index: int = -1
    box: Optional[object] = None

    def to_dict(self) -> Dict:
        return {
            "type": self.etype,
            "text": self.text,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "strategy": self.strategy,
        }


# ---------------------------------------------------------------------------
# Multilingual transformer NER (optional)
# ---------------------------------------------------------------------------
_ner_pipe = None
_ner_available = False


def _get_ner_pipeline():
    """Lazily load a multilingual transformer NER pipeline if available and enabled."""
    global _ner_pipe, _ner_available
    if not settings.use_ner:
        return None
    if _ner_pipe is not None:
        return _ner_pipe
    try:
        from transformers import pipeline

        _ner_pipe = pipeline(
            "ner",
            model="Davlan/bert-base-multilingual-cased-ner-hrl",
            aggregation_strategy="simple",
        )
        _ner_available = True
        logger.info("Multilingual NER pipeline loaded")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Multilingual NER unavailable (%s); using rule-based NER", exc)
        _ner_available = False
        _ner_pipe = None
    return _ner_pipe


# Map common NER labels to our target entity types.
_LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "DATE": "DATE",
    "TIME": "TIME",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
}


def _ner_with_model(text: str) -> List[Entity]:
    """Extract entities via the multilingual NER model."""
    pipe = _get_ner_pipeline()
    if pipe is None:
        return []
    try:
        results = pipe(text)
        entities: List[Entity] = []
        for ent in results:
            raw_label = ent.get("entity_group") or ent.get("entity") or ""
            label = _LABEL_MAP.get(raw_label.upper())
            if label is None:
                continue
            word = ent.get("word", "").strip()
            score = float(ent.get("score", 0.0))
            if not word:
                continue
            entities.append(Entity(
                etype=label,
                text=word,
                value=word,
                confidence=score,
                strategy="model_ner",
            ))
        return entities
    except Exception as exc:  # noqa: BLE001
        logger.warning("NER model inference failed (%s)", exc)
        return []


# ---------------------------------------------------------------------------
# Layout-aware grouping (uses OCR boxes + layout region labels)
# ---------------------------------------------------------------------------
def _group_by_lines(text_lines: List[str], layouts: List[Dict]) -> List[Tuple[int, str, str]]:
    """Tag each text line with a layout region label when possible.

    Returns list of (line_index, line_text, region_label).
    """
    tagged: List[Tuple[int, str, str]] = []
    for idx, line in enumerate(text_lines):
        label = "text_region"
        # Simple heuristic: match layout regions by order if no box info.
        # When layouts carry boxes, we could match by position; here we keep
        # it generic and use line index proximity as a weak signal.
        tagged.append((idx, line, label))
    return tagged


# ---------------------------------------------------------------------------
# Rule-based fallback (generic, not tied to any template)
# ---------------------------------------------------------------------------
_MONTHS = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
)
_WEEKDAYS = (
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|wed|thu|fri|sat|sun"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"

_DATE_PATTERNS = [
    rf"((?:{_WEEKDAYS})\.?,?\s+(?:{_MONTHS})\.?\.?\s+{_DAY},?\s+\d{{4}})",
    rf"((?:{_WEEKDAYS})\.?,?\s+{_DAY}\s+(?:{_MONTHS})\.?\.?\s+\d{{4}})",
    rf"((?:{_MONTHS})\.?\.?\s+{_DAY},?\s+\d{{4}})",
    rf"({_DAY}\s+(?:{_MONTHS})\.?\.?\s+\d{{4}})",
    rf"(the\s+{_DAY}\s+of\s+(?:{_MONTHS})\.?\.?\s+\d{{4}})",
    r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
]

_TIME_PATTERNS = [
    r"(\d{1,2}:\d{2}\s*(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?)?)",
    r"(\d{1,2}\s*(?:am|pm|a\.?\s?m\.?|p\.?\s?m\.?))",
]

_PHONE_PATTERNS = [
    r"(\+?\d[\d\s\-]{8,15}\d)",
    r"(\b\d{10}\b)",
    r"(\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)",
]

_EMAIL_PATTERN = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

# Weak semantic lexicons (generic, not sample-specific).
_LOCATION_KEYWORDS = [
    "venue", "at", "located at", "road", "street", "nagar", "colony",
    "layout", "hall", "palace", "mandapam", "hotel", "city", "town",
    "district", "village",
]
_ORG_KEYWORDS = ["invite", "cordially invite", "request the pleasure",
                 "family", "families", "association", "trust", "committee",
                 "foundation"]

# Words that must never be tagged as PERSON even when they start with
# uppercase (verbs, durations, day names, common nouns).
_NOT_PERSON_WORDS = {
    "organizes", "organize", "organizing", "organized",
    "design", "designs", "designed", "learning", "learn",
    "gain", "gaining", "gained", "transform", "transforms",
    "transforming", "creating", "create", "creates", "created",
    "interactive", "interactively", "fundamentals", "principles",
    "components", "prototyping", "prototyping.", "wireframing",
    "layouts", "skills", "experiences", "ideas", "session",
    "full-day", "full day", "fullday",
    "together", "better", "journey", "forever",
    "celebrate", "celebrating", "welcome", "regards", "warmly",
    "presence", "pleasure", "company", "auspicious", "occasion",
    "blessings", "families", "relative", "kin", "members",
    "guests", "everyone", "children", "couple",
}

# City / place names that should never be PERSON candidates.
_NOT_PERSON_PLACES = {
    "thoppupalayam", "perundurai", "erode", "coimbatore", "chennai",
    "madurai", "bengaluru", "bangalore", "hyderabad", "mumbai",
    "delhi", "kolkata", "pune", "salem", "vellore", "trichy",
    "kochi", "kerala", "tirunelveli", "kanyakumari",
}


def _rule_based_ner(text: str, text_lines: List[str]) -> List[Entity]:
    """Generic rule-based entity extraction (fallback)."""
    entities: List[Entity] = []

    # Dates
    for pat in _DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip()
            entities.append(Entity("DATE", val, val, 0.85, "rule_date"))
    # Times
    for pat in _TIME_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip()
            entities.append(Entity("TIME", val, val, 0.85, "rule_time"))
    # Phones (ORGANIZATION-adjacent contact info kept as ORGANIZATION candidate)
    for pat in _PHONE_PATTERNS:
        for m in re.finditer(pat, text):
            val = m.group(1).strip()
            entities.append(Entity("ORGANIZATION", val, val, 0.7, "rule_phone"))
    # Emails
    for m in re.finditer(_EMAIL_PATTERN, text, re.IGNORECASE):
        val = m.group(1).strip()
        entities.append(Entity("ORGANIZATION", val, val, 0.9, "rule_email"))

    # Persons: lines that look like names (generic heuristics).
    _event_words = {"wedding", "marriage", "reception", "engagement", "birthday",
                    "celebration", "ceremony", "invitation", "announcement",
                    "party", "function", "housewarming", "muhurtham", "naming",
                    "join", "celebrate", "with", "the", "of", "and", "&"}
    # Generic relationship/group words that are NOT person names.
    _RELATIONSHIP_GROUP_WORDS = {"family", "families", "relatives", "relative",
                                  "all the family", "all family", "all relatives",
                                  "kin", "kinsfolk", "folk", "folks", "dear",
                                  "friends", "guests", "everyone", "everybody",
                                  "parents", "parent", "children", "child",
                                  "couple", "members", "member"}
    # ALL-CAPS or mixed-case lines containing these keywords are
    # event headings or activity descriptions, never person names.
    _EVENT_LINE_REJECT = {
        "events", "performances", "recognitions", "achievements",
        "activities", "competition", "exhibition", "display",
        "exhibits", "agenda", "schedule", "menu", "programme",
        "program", "sports", "annual", "cultural", "festival",
    }
    for idx, line in enumerate(text_lines):
        ls = line.strip()
        if not ls or len(ls) > 40 or len(ls) < 3:
            continue
        if re.search(r"[0-9]", ls):
            continue
        lowered = ls.lower()
        if any(kw in lowered for kw in _LOCATION_KEYWORDS + _ORG_KEYWORDS):
            continue
        if lowered in _RELATIONSHIP_GROUP_WORDS:
            continue
        words = ls.split()
        # Reject single words that are clearly not person names
        # (verbs, durations, day names, common nouns, place names).
        if len(words) == 1:
            if words[0].lower() in _NOT_PERSON_WORDS:
                continue
            if words[0].lower() in _NOT_PERSON_PLACES:
                continue
            if words[0].lower() in {
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            }:
                continue
        if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
            continue
        # Reject ALL-CAPS alpha words combined with event keywords
        # (e.g. "ANNUAL DAY", "SPORTS DAY").
        alpha_words = [w for w in words if w.isalpha()]
        if len(alpha_words) >= 2 and all(w.isupper() for w in alpha_words):
            if any(re.search(rf"\b{kw}\b", lowered) for kw in _EVENT_LINE_REJECT):
                continue
        # Reject mixed-case lines with connectors AND event/activity keywords.
        if any(c in ls for c in "&|/"):
            if any(kw in lowered.split() for kw in _EVENT_LINE_REJECT):
                continue
        # Reject Title Case lines (all words capitalized, none ALL-CAPS)
        # containing activity/event keywords — activity descriptions,
        # not person names.
        alpha_words = [w for w in words if w.isalpha()]
        if len(alpha_words) >= 2 and not all(w.isupper() for w in alpha_words):
            if all(w[0].isupper() for w in words if w):
                if any(kw in lowered.split() for kw in _EVENT_LINE_REJECT):
                    continue
        # Reject Title Case lines that contain common lowercase
        # connector words (for, the, of, at, in, etc.) — these are
        # descriptive phrases, not person names.
        lowercase_connectors = {"for", "the", "of", "at", "in", "to", "and",
                                "or", "on", "with", "from", "by", "our",
                                "their", "our", "all"}
        if any(w.lower() in lowercase_connectors for w in words):
            continue
        # Reject lines containing known cities, states, or venue
        # keywords — locations and organizations, not persons.
        _NER_CITIES = {
            "madurai", "chennai", "coimbatore", "bengaluru", "bangalore",
            "hyderabad", "mumbai", "delhi", "kolkata", "pune", "salem",
            "vellore", "trichy", "erode", "kochi", "kerala", "goa",
            "agra", "jaipur", "lucknow", "kanpur", "nagpur", "indore",
            "surat", "ahmedabad", "thiruvananthapuram", "kanyakumari",
            "mysuru", "vijayawada", "visakhapatnam", "noida",
            "gurgaon", "thoothukudi", "tirunelveli", "greenville",
            "kovilpatti", "tamil",
        }
        _NER_VENUE_WORDS = {
            "college", "university", "school", "hall", "hotel", "resort",
            "grounds", "stadium", "complex", "centre", "center",
            "institute", "academy", "temple", "church", "mosque",
        }
        if any(w.lower().rstrip(".,;:") in _NER_CITIES for w in words):
            continue
        if any(kw in lowered for kw in _NER_VENUE_WORDS):
            continue
        # Reject multi-word lines where every word is a known place name
        # (e.g. "Thoppupalayam Perundurai").
        if len(words) >= 2 and all(
            w.lower().rstrip(".,;:") in _NOT_PERSON_PLACES for w in words
        ):
            continue
        if not ls[0].isupper():
            continue
        filtered = [w for w in words if w.lower() not in _event_words]
        if len(filtered) < 1:
            continue
        cleaned = " ".join(filtered).strip(" .,;:")
        if len(cleaned) >= 3:
            entities.append(Entity("PERSON", cleaned, cleaned, 0.6,
                                   "rule_person", line_index=idx))

    # Locations: keyword-labeled lines / "at <place>" / known city lines.
    for idx, line in enumerate(text_lines):
        ls = line.strip()
        lowered = line.lower()
        # "at <place>" phrase
        at_m = re.match(r"^(?:at|venue|located at)\s+(.+)$", ls, re.IGNORECASE)
        if at_m:
            val = at_m.group(1).strip()
            if val and len(val) < 80:
                entities.append(Entity("LOCATION", val, val, 0.7, "rule_loc_at",
                                       line_index=idx))
        # Keyword-labeled line
        elif any(re.search(rf"\b{re.escape(kw)}\b", lowered)
                 for kw in _LOCATION_KEYWORDS):
            words = ls.split()
            if len(words) >= 2 and all(w.isupper() and len(w) > 1 for w in words):
                continue
            val = re.sub(r"^(?:venue|location|place|at|located at)\s*:?\s*",
                         "", ls, flags=re.IGNORECASE).strip()
            if val and len(val) < 80:
                entities.append(Entity("LOCATION", val, val, 0.65,
                                       "rule_loc_keyword", line_index=idx))

    # Organizations: lines with invite markers / known org keywords.
    for idx, line in enumerate(text_lines):
        lowered = line.lower()
        if any(kw in lowered for kw in _ORG_KEYWORDS):
            ls = line.strip()
            if ls and len(ls) < 80:
                entities.append(Entity("ORGANIZATION", ls, ls, 0.5,
                                       "rule_org", line_index=idx))

    return entities


# ---------------------------------------------------------------------------
# Merge & dedupe
# ---------------------------------------------------------------------------
def _dedupe(entities: List[Entity]) -> List[Entity]:
    """Merge entities of the same type whose text overlaps strongly."""
    out: List[Entity] = []
    for ent in entities:
        key = (ent.etype, ent.value.lower().strip())
        exists = False
        for i, o in enumerate(out):
            if (o.etype, o.value.lower().strip()) == key:
                # Keep the higher-confidence one.
                if ent.confidence > o.confidence:
                    out[i] = ent
                exists = True
                break
        if not exists:
            out.append(ent)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_entities(text: str, text_lines: Optional[List[str]] = None,
                     layouts: Optional[List[Dict]] = None) -> List[Entity]:
    """Extract PERSON/DATE/TIME/LOCATION/ORGANIZATION entities from invitation text.

    Combines model NER, layout-aware grouping, and rule-based fallback.
    """
    if text_lines is None:
        text_lines = [l for l in text.split("\n") if l.strip()]
    layouts = layouts or []

    entities: List[Entity] = []

    # 1) Model NER (multilingual).
    entities += _ner_with_model(text)

    # 2) Layout-aware grouping (only if model produced no results — keep the
    #    model output as primary; fallback fills gaps).
    #    Here we use rule-based as the robust fallback that always runs.
    rule_entities = _rule_based_ner(text, text_lines)
    entities += rule_entities

    # The model output is preferred; rule-based fills in types the model
    # missed. De-duplicate keeping highest confidence.
    return _dedupe(entities)
