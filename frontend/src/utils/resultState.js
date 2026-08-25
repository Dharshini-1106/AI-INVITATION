import { InvitationResult } from '../models/InvitationResult';

const STORAGE_KEY = 'invitation-sense-last-result';

function toEventShape(value) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  return {
    event_name: value.event_name || '',
    event_type: value.event_type || '',
    bride_name: value.bride_name || '',
    groom_name: value.groom_name || '',
    date: value.date || '',
    time: value.time || '',
    venue: value.venue || '',
    address: value.address || '',
    contact_number: value.contact_number || '',
    confidence: value.confidence || 0,
  };
}

export function normalizeResultPayload(payload) {
  if (!payload) {
    return null;
  }

  if (payload instanceof InvitationResult) {
    return payload;
  }

  const source = payload.result || payload;
  if (!source || typeof source !== 'object') {
    return null;
  }

  const firstEvent = Array.isArray(source.events) && source.events.length > 0
    ? source.events[0]
    : null;

  return new InvitationResult({
    event_name: source.event_name || firstEvent?.event_name || '',
    event_type: source.event_type || firstEvent?.event_type || '',
    bride_name: source.bride_name || firstEvent?.bride_name || '',
    groom_name: source.groom_name || firstEvent?.groom_name || '',
    date: source.date || firstEvent?.date || '',
    time: source.time || firstEvent?.time || '',
    venue: source.venue || firstEvent?.venue || '',
    address: source.address || firstEvent?.address || '',
    contact_number: source.contact_number || firstEvent?.contact_number || '',
    language: source.language || '',
    confidence_score: source.confidence_score ?? source.confidence ?? 0,
    number_of_events: source.number_of_events ?? (Array.isArray(source.events) ? source.events.length : 1),
    events: Array.isArray(source.events) ? source.events.map(toEventShape).filter(Boolean) : [],
    quality: source.quality || {},
    raw_text: source.raw_text || '',
    ocr_layout: Array.isArray(source.ocr_layout) ? source.ocr_layout : [],
    processing_notes: source.processing_notes || [],
  });
}

export function saveAnalysisResult(result, previewUrl) {
  if (typeof window === 'undefined') {
    return null;
  }

  const normalized = normalizeResultPayload(result);
  if (!normalized) {
    return null;
  }

  const payload = { result: normalized, previewUrl: previewUrl || null };
  window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  return payload;
}

export function loadAnalysisResult() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw);
    return {
      result: normalizeResultPayload(parsed?.result || parsed),
      previewUrl: parsed?.previewUrl || null,
    };
  } catch {
    return null;
  }
}
