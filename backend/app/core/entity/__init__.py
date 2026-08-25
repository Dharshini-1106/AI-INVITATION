"""Entity extraction package (PERSON/DATE/TIME/LOCATION/ORGANIZATION).

Layered NER: multilingual transformer model -> layout-aware grouping ->
rule-based fallback. Never depends on a single source.
"""
from .entity_ner import extract_entities

__all__ = ["extract_entities"]
