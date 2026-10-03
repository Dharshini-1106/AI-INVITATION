// Model representing a single extracted event from an invitation
export class InvitationEvent {
  constructor({
    event_name = '',
    event_type = '',
    bride_name = '',
    groom_name = '',
    date = '',
    time = '',
    end_time = '',
    venue = '',
    address = '',
    contact_address = '',
    contact_number = '',
    timezone = '',
    additional_information = '',
    birthday_age = '',
    printed_weekday = '',
    confidence = 0,
  } = {}) {
    this.event_name = event_name;
    this.event_type = event_type;
    this.bride_name = bride_name;
    this.groom_name = groom_name;
    this.date = date;
    this.time = time;
    this.end_time = end_time;
    this.venue = venue;
    this.address = address;
    this.contact_address = contact_address;
    this.contact_number = contact_number;
    this.timezone = timezone;
    this.additional_information = additional_information;
    this.birthday_age = birthday_age;
    this.printed_weekday = printed_weekday;
    this.confidence = confidence;
  }
}

// Model representing a person associated with the invitation
export class Person {
  constructor({ name = '', role = '' } = {}) {
    this.name = name;
    this.role = role;
  }
}

// Model representing the full invitation understanding result
export class InvitationResult {
  constructor({
    invitation_mode = 'single',
    people = [],
    event_name = '',
    event_type = '',
    bride_name = '',
    groom_name = '',
    date = '',
    time = '',
    venue = '',
    address = '',
    contact_address = '',
    contact_number = '',
    timezone = '',
    language = '',
    confidence_score = 0,
    number_of_events = 1,
    events = [],
    quality = {},
    raw_text = '',
    ocr_layout = [],
    ocr_engine = '',
    ocr_confidence = null,
    tamil_character_count = 0,
    english_character_count = 0,
    fallback_used = false,
    processing_notes = [],
  } = {}) {
    this.invitation_mode = invitation_mode;
    this.people = (people || []).map((p) => new Person(p));
    this.event_name = event_name;
    this.event_type = event_type;
    this.bride_name = bride_name;
    this.groom_name = groom_name;
    this.date = date;
    this.time = time;
    this.venue = venue;
    this.address = address;
    this.contact_address = contact_address;
    this.contact_number = contact_number;
    this.timezone = timezone;
    this.language = language;
    this.confidence_score = confidence_score;
    this.number_of_events = number_of_events;
    this.events = (events || []).map((e) => new InvitationEvent(e));
    this.quality = quality || {};
    this.raw_text = raw_text;
    this.ocr_layout = ocr_layout || [];
    this.ocr_engine = ocr_engine;
    this.ocr_confidence = ocr_confidence;
    this.tamil_character_count = tamil_character_count;
    this.english_character_count = english_character_count;
    this.fallback_used = fallback_used;
    this.processing_notes = processing_notes || [];
  }

  get primaryEvent() {
    if (this.events && this.events.length > 0) {
      return this.events[0];
    }
    return new InvitationEvent({
      event_name: this.event_name,
      event_type: this.event_type,
      bride_name: this.bride_name,
      groom_name: this.groom_name,
      date: this.date,
      time: this.time,
      venue: this.venue,
      address: this.address,
      contact_number: this.contact_number,
      timezone: this.timezone,
      confidence: this.confidence_score,
    });
  }

  get confidencePercent() {
    return Math.round((this.confidence_score || 0) * 100);
  }
}
