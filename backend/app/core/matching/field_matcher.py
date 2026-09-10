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
            persons.append(val)
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

    # Assign two persons to bride/groom (order heuristic only when no gender
    # info; left as-is, parser will refine).
    if persons:
        if "bride_name" not in result:
            result["bride_name"] = persons[0]
        if len(persons) > 1 and "groom_name" not in result:
            result["groom_name"] = persons[1]
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
