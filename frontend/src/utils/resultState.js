import { InvitationResult, Person } from '../models/InvitationResult';

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
    end_time: value.end_time || '',
    venue: value.venue || '',
    address: value.address || '',
    contact_address: value.contact_address || '',
    contact_number: value.contact_number || '',
    timezone: value.timezone || '',
    additional_information: value.additional_information || '',
    birthday_age: value.birthday_age || '',
    printed_weekday: value.printed_weekday || '',
    confidence: value.confidence || 0,
  };
}

function toPersonShape(value) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  return {
    name: value.name || '',
    role: value.role || '',
  };
}

function getUnlabeledWeddingPair(rawText) {
  if (!rawText || /\b(?:bride|groom)\s*:?\s*[A-Z][a-z]{2,}/i.test(rawText)) {
    return null;
  }
  const match = rawText.match(
    /\b(?:wedding|marriage)\s+of\s+([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b/
  );
  if (!match) {
    return null;
  }

  const [first, second] = [match[1], match[2]];
  const genderHint = (name) => {
    const givenName = name.trim().split(/\s+/)[0].toLowerCase();
    if (/(?:ia|na|a|e)$/.test(givenName)) return 'female';
    if (/(?:[bcdgklmnprstvxz]|er|an|un|in|ee)$/.test(givenName)) return 'male';
    return '';
  };
  const firstGender = genderHint(first);
  const secondGender = genderHint(second);
  if (firstGender && secondGender && firstGender !== secondGender) {
    const bride = firstGender === 'female' ? first : second;
    const groom = firstGender === 'male' ? first : second;
    return {
      people: [
        { name: bride, role: 'Bride' },
        { name: groom, role: 'Groom' },
      ],
      bride,
      groom,
    };
  }
  return {
    people: [
      { name: first, role: 'Person' },
      { name: second, role: 'Person' },
    ],
    bride: '',
    groom: '',
  };
}

function weekdayPrintedBesideDate(dateValue, rawText) {
  const match = String(dateValue || '').match(
    /^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})$/i
  );
  if (!match || !rawText) {
    return '';
  }
  const [, month, day, year] = match;
  const monthIndex = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
  ].indexOf(month.toLowerCase());
  if (monthIndex < 0) {
    return '';
  }
  const expected = new Date(Date.UTC(Number(year), monthIndex, Number(day)))
    .toLocaleDateString('en-US', { weekday: 'long', timeZone: 'UTC' });
  const normalizedDay = String(Number(day));
  const datePattern = `(?:${normalizedDay}|0${normalizedDay})\\s+${month}\\.?\\s+${year}`;
  const printed = rawText.match(
    new RegExp(`\\b(${expected})\\s*,?\\s*${datePattern}\\b`, 'i')
  );
  return printed ? expected : '';
}

export function normalizeResultPayload(payload) {
  if (!payload) {
    return null;
  }

  const source = payload.result || payload;
  if (!source || typeof source !== 'object') {
    return null;
  }

  const firstEvent = Array.isArray(source.events) && source.events.length > 0
    ? source.events[0]
    : null;

  const rawText = source.raw_text || '';
  const unlabeledPair = getUnlabeledWeddingPair(rawText);
  const derivedTamilCount = [...rawText].filter((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint >= 0x0B80 && codePoint <= 0x0BFF;
  }).length;
  const derivedEnglishCount = [...rawText].filter((character) => (
    (character >= 'A' && character <= 'Z')
    || (character >= 'a' && character <= 'z')
  )).length;

  const events = Array.isArray(source.events)
    ? source.events.map(toEventShape).filter(Boolean)
    : [];
  for (const event of events) {
    const printedWeekday = weekdayPrintedBesideDate(event.date, rawText);
    if (printedWeekday) {
      event.printed_weekday = printedWeekday;
    }
    if (unlabeledPair) {
      event.bride_name = unlabeledPair.bride;
      event.groom_name = unlabeledPair.groom;
    }
  }

  return new InvitationResult({
    invitation_mode: source.invitation_mode || 'single',
    people: unlabeledPair
      ? unlabeledPair.people.map((person) => new Person(person))
      : (Array.isArray(source.people) ? source.people.map(toPersonShape).filter(Boolean) : []),
    event_name: source.event_name || firstEvent?.event_name || '',
    event_type: source.event_type || firstEvent?.event_type || '',
    bride_name: unlabeledPair
      ? unlabeledPair.bride
      : (source.bride_name || firstEvent?.bride_name || ''),
    groom_name: unlabeledPair
      ? unlabeledPair.groom
      : (source.groom_name || firstEvent?.groom_name || ''),
    date: source.date || firstEvent?.date || '',
    time: source.time || firstEvent?.time || '',
    venue: source.venue || firstEvent?.venue || '',
    address: source.address || firstEvent?.address || '',
    contact_address: source.contact_address || firstEvent?.contact_address || '',
    contact_number: source.contact_number || firstEvent?.contact_number || '',
    timezone: source.timezone || firstEvent?.timezone || '',
    language: source.language || '',
    confidence_score: source.confidence_score ?? source.confidence ?? 0,
    number_of_events: source.number_of_events ?? (Array.isArray(source.events) ? source.events.length : 1),
    events,
    quality: source.quality || {},
    raw_text: rawText,
    ocr_layout: Array.isArray(source.ocr_layout) ? source.ocr_layout : [],
    ocr_engine: source.ocr_engine || '',
    ocr_confidence: source.ocr_confidence ?? null,
    tamil_character_count: source.tamil_character_count || derivedTamilCount,
    english_character_count: source.english_character_count || derivedEnglishCount,
    fallback_used: source.fallback_used || false,
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
