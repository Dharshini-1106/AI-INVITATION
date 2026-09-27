"""Map extracted entities to final invitation fields using Sentence-BERT.

The matcher embeds each entity value and each target field description, then
assigns the best entity->field mapping by cosine similarity. This is generic:
it does not hard-code which specific text belongs to which field; it relies on
semantic similarity between the entity text and the field's natural-language
description.

Falls back to a type-based rule mapping when Sentence-BERT is unavailable.

Target fields (stable schema):
  event_name, event_type, bride_name, groom_name, date, time, venue, address,
  contact_number
"""
import logging
import re
from typing import Dict, List, Optional

from ...config import settings

logger = logging.getLogger(__name__)

# Field descriptions (semantic anchors used for embedding/matching).
FIELD_DESCRIPTIONS = {
    "event_name": "the name of the ceremony or event",
    "event_type": "the type of event such as wedding reception engagement birthday",
    "bride_name": "the name of the bride the woman getting married",
    "groom_name": "the name of the groom the man getting married",
    "date": "the date when the event takes place",
    "time": "the time when the event starts",
    "venue": "the place or building where the event is held",
    "address": "the street address city and postal code of the location",
    "contact_number": "the phone number or contact information to call",
}

# Entity type -> primary field mapping (used when SBERT is unavailable and as
# a strong prior for the type-based fallback).
_TYPE_FIELD = {
    "DATE": "date",
    "TIME": "time",
    "PERSON": "bride_name",  # refined later by gender/context
    "LOCATION": "venue",
    "ORGANIZATION": "event_name",
}

_sbert_model = None
_sbert_available = False


def _get_sbert():
    """Lazily load Sentence-BERT if available and enabled."""
    global _sbert_model, _sbert_available
    if not settings.use_sbert:
        return None
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        _sbert_available = True
        logger.info("Sentence-BERT loaded for field matching")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentence-BERT unavailable (%s); using rule matching", exc)
        _sbert_available = False
        _sbert_model = None
    return _sbert_model


def _entity_type(entity) -> str:
    """Return the entity type string whether passed as dict or object."""
    if isinstance(entity, dict):
        return entity.get("type", "")
    return getattr(entity, "etype", "") or getattr(entity, "type", "")


def _entity_value(entity) -> str:
    """Return the entity value string whether passed as dict or object."""
    if isinstance(entity, dict):
        return entity.get("value", "") or entity.get("text", "")
    return getattr(entity, "value", "") or getattr(entity, "text", "")


def _entity_strategy(entity) -> str:
    if isinstance(entity, dict):
        return entity.get("strategy", "")
    return getattr(entity, "strategy", "")


def _sbert_match_option_with_fields(entities: List[object]) -> Dict[str, str]:
    """Match each entity to the best field using Sentence-BERT similarity.

    Returns a dict field -> value (first/best match per field).
    """
    model = _get_sbert()
    if model is None:
        return {}
    try:
        import numpy as np
        from numpy.linalg import norm

        field_names = list(FIELD_DESCRIPTIONS.keys())
        field_vecs = model.encode(
            [FIELD_DESCRIPTIONS[f] for f in field_names], convert_to_numpy=True
        )
        result: Dict[str, str] = {}
        for ent in entities:
            val = _entity_value(ent)
            if not val:
                continue
            ent_vec = model.encode(val, convert_to_numpy=True)
            sims = (field_vecs @ ent_vec) / (
                norm(field_vecs, axis=1) * norm(ent_vec) + 1e-9
            )
            best_idx = int(np.argmax(sims))
            best_field = field_names[best_idx]
            best_score = float(sims[best_idx])
            if best_score < 0.35:
                continue
            # Keep the first/highest-confidence match per field.
            if best_field not in result or best_score > 0.5:
                result[best_field] = val
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentence-BERT field matching failed (%s)", exc)
        return {}


# Simple heuristics to filter out PERSON entities that are clearly
# not person names (event headings, common nouns, ALL-CAPS words).
_NOT_LIKE_NAME = {
    "excellence", "american", "indian", "tamil", "madurai",
    "chennai", "coimbatore", "bengaluru", "bangalore",
    "hyderabad", "mumbai", "delhi", "kolkata", "pune",
    "recognitions", "performances", "achievements",
    "knowledge", "flourish", "plaza", "com", "and",
    "stronger", "students", "tomorrow", "healthier",
    "different", "people", "today", "leaders",
    "discipline", "growth", "opportunity", "community",
    "common", "purpose", "people", "person",
    "art", "faster", "culture", "creativity", "higher",
    "beyond", "limits", "together", "play", "excel",
    "rise", "reflect", "rejoice", "persevere", "celebrate",
    "track", "recognition", "recognitions", "daily", "since",
    "her", "our", "your", "their",
}


# Tamil wedding / religious / invitation phrases (and key fragments) that OCR
# may surface as standalone tokens but are never person names. Kept in sync
# with the parser's `_TAMIL_NON_NAME_MARKERS` list.
_TAMIL_NON_NAME_MARKERS = (
    "அழைப்பிதழ்", "திருமண", "ருமண", "வரவேற்பு", "நன்றி", "வாழ்த்து", "அன்புடன்",
    "வாழ்த்துகள்", "நிகாஹ்", "நிகாஹ", "குடும்பம்", "திருநாள்",
    "மகிழ்ச்சி", "கல்யாணம்", "வலீமா", "வலிமா", "அலீமா",
    "நகாஹ", "மணம", "நிகா", "லீமா",
    "கணபதி", "சுப்ரமணி", "நமஸ்தே", "ஓம்", "நமஃசரணம்",
)


def _has_tamil_chars(value: str) -> bool:
    return any(0x0B80 <= ord(ch) <= 0x0BFF for ch in value)


def _contains_tamil_non_name_marker(value: str) -> bool:
    return any(marker in value for marker in _TAMIL_NON_NAME_MARKERS)


def _looks_like_person_name(value: str) -> bool:
    if not value or len(value.strip()) < 3:
        return False
    stripped = value.strip().strip(".,;:'\"")
    if stripped.lower() in _NOT_LIKE_NAME:
        return False
    # Reject Tamil lines that contain wedding / religious / invitation
    # phrases — these are OCR fragments (e.g. "ருமண", "கணபதியெ நம:"), not
    # person names. We must not translate Tamil OCR blindly.
    if _has_tamil_chars(stripped) and _contains_tamil_non_name_marker(stripped):
        return False
    words = stripped.split()
    if len(words) == 1 and stripped.isupper() and len(stripped) >= 3:
        return False
    if len(words) == 1 and stripped.lower() in {
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
        "venue", "event", "programme", "program", "festival",
        "sports", "annual", "cultural", "arts", "culture",
        "recognition", "performance", "competition", "exhibition",
    }:
        return False
    # Reject multi-word ALL-CAPS phrases (e.g. "ALL TOP", "REAL WORLD").
    if len(words) >= 2 and all(w.isupper() for w in words):
        return False
    # Reject names where any word is a common non-name word.
    if any(w.lower() in _NOT_LIKE_NAME for w in words):
        return False
    # Reject names where most words are very short (1-2 chars),
    # e.g. OCR artifacts like "C H A R A C T E R".
    if len(words) >= 2:
        short = sum(1 for w in words if len(w) <= 2)
        if short >= len(words) * 0.7:
            return False
    return True


def _gender_hint_from_name(name: str) -> str:
    """Return 'female', 'male', or '' based on honorifics in name."""
    name_lower = name.lower()
    if any(h in name_lower for h in ("ms.", "mrs.", "miss", "smt.", "selvi")):
        return "female"
    if any(h in name_lower for h in ("mr.", "dr.", "sri", "thiru", "er.", "kum.")):
        return "male"
    return ""


def _rule_match(entities: List[object]) -> Dict[str, str]:
    """Type-based fallback mapping (never depends on a model)."""
    result: Dict[str, str] = {}
    persons = []
    for ent in entities:
        etype = _entity_type(ent)
        val = _entity_value(ent)
        if not val:
            continue
        if etype == "DATE":
            result.setdefault("date", val)
        elif etype == "TIME":
            result.setdefault("time", val)
        elif etype == "PERSON":
            if _looks_like_person_name(val):
                persons.append((val, _gender_hint_from_name(val)))
        elif etype == "LOCATION":
            # Prefer venue over address unless a street keyword is present.
            if re.search(r"\b(street|road|rd|nagar|colony|pin|avenue|ave)\b",
                         val, re.IGNORECASE):
                result.setdefault("address", val)
            else:
                result.setdefault("venue", val)
        elif etype == "ORGANIZATION":
            # Contact-like (phone/email) -> contact_number
            if re.search(r"[\d@]", val):
                result.setdefault("contact_number", val)
            else:
                result.setdefault("event_name", val)

    # Assign persons to bride/groom using gender hints.
    # Only assign if we have clear gender signals or multiple distinct persons.
    female_names = [p[0] for p in persons if p[1] == "female"]
    male_names = [p[0] for p in persons if p[1] == "male"]
    neutral_names = [p[0] for p in persons if p[1] == ""]

    if female_names:
        result["bride_name"] = female_names[0]
    elif neutral_names:
        result["bride_name"] = neutral_names[0]

    # Only assign groom_name if we have a male name or multiple neutral names
    if male_names:
        result["groom_name"] = male_names[0]
    elif len(neutral_names) > 1:
        result["groom_name"] = neutral_names[1]
    elif len(persons) > 1 and not female_names and not male_names:
        # Fallback: if multiple persons but no gender hints, use second
        result["groom_name"] = persons[1][0]

    return result


def match_entities_to_fields(entities: List[object]) -> Dict[str, str]:
    """Map extracted entities to invitation fields.

    Uses Sentence-BERT semantic matching when available and enabled,
    otherwise falls back to a type-based rule mapping. Returns a dict of
    field -> value.
    """
    if not settings.use_matching:
        return _rule_match(entities)
    if _sbert_available:
        matched = _sbert_match_option_with_fields(entities)
        if matched:
            return matched
    return _rule_match(entities)
