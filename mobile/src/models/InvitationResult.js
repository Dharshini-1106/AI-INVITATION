// Model representing a single extracted event
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
    contact_number = '',
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
    this.contact_number = contact_number;
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
    end_time = '',
    venue = '',
    address = '',
    contact_number = '',
    language = '',
    confidence_score = 0,
    number_of_events = 1,
    events = [],
    quality = {},
    raw_text = '',
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
    this.end_time = end_time;
    this.venue = venue;
    this.address = address;
    this.contact_number = contact_number;
    this.language = language;
    this.confidence_score = confidence_score;
    this.number_of_events = number_of_events;
    this.events = (events || []).map((e) => new InvitationEvent(e));
    this.quality = quality || {};
    this.raw_text = raw_text;
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
      end_time: this.end_time,
      venue: this.venue,
      address: this.address,
      contact_number: this.contact_number,
      confidence: this.confidence_score,
    });
  }

  get confidencePercent() {
    return Math.round((this.confidence_score || 0) * 100);
  }
}
