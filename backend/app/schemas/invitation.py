"""Pydantic request/response models for the invitation pipeline."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    """A person participating in or associated with the invitation."""

    name: str = ""
    role: str = "Person"


class ImageQuality(BaseModel):
    """Quality analysis result for an input image."""

    score: float = Field(description="BRISQUE quality score (lower = better quality)")
    is_blurred: bool = False
    is_noisy: bool = False
    is_dark: bool = False
    is_bright: bool = False
    is_rotated: bool = False
    low_resolution: bool = False
    needs_enhancement: bool = False
    applied_enhancements: List[str] = Field(default_factory=list)


class Event(BaseModel):
    """A single structured event extracted from an invitation."""

    event_name: str = ""
    event_type: str = ""
    bride_name: str = ""
    groom_name: str = ""
    date: str = ""
    time: str = ""
    end_time: str = ""
    venue: str = ""
    address: str = ""
    contact_number: str = ""
    additional_information: str = ""
    birthday_age: str = ""
    printed_weekday: str = ""
    occasion_detail: str = ""
    timezone: str = ""
    confidence: float = 0.0

    # Calendar-specific fields (optional, used by calendar endpoint)
    summary: str = ""
    dateTime: str = ""
    endDateTime: str = ""
    location: str = ""
    description: str = ""


class OCRLine(BaseModel):
    """OCR evidence retained for layout-aware semantic extraction/debugging."""

    text: str = ""
    bbox: List[List[float]] = Field(default_factory=list)
    confidence: float = 0.0


class InvitationResult(BaseModel):
    """Final structured result returned to the client."""

    invitation_mode: str = "single"
    people: List[Person] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)

    event_name: str = ""
    event_type: str = ""
    bride_name: str = ""
    groom_name: str = ""
    date: str = ""
    time: str = ""
    venue: str = ""
    address: str = ""
    contact_number: str = ""
    additional_information: str = ""
    birthday_age: str = ""
    printed_weekday: str = ""
    timezone: str = ""
    language: str = ""
    confidence_score: float = 0.0
    number_of_events: int = 1
    quality: ImageQuality = Field(default_factory=ImageQuality)
    raw_text: str = ""
    ocr_layout: List[OCRLine] = Field(default_factory=list)
    ocr_engine: str = ""
    ocr_confidence: Optional[float] = None
    tamil_character_count: int = 0
    english_character_count: int = 0
    fallback_used: bool = False
    processing_notes: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str = ""
    version: str = ""

