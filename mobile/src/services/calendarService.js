import * as Calendar from 'expo-calendar';

const NOT_AVAILABLE = new Set(['', 'not available', 'â€”', '-', 'null', 'none']);
const MONTHS = {
  jan: 0, january: 0, feb: 1, february: 1, mar: 2, march: 2,
  apr: 3, april: 3, may: 4, jun: 5, june: 5, jul: 6, july: 6,
  aug: 7, august: 7, sep: 8, sept: 8, september: 8, oct: 9,
  october: 9, nov: 10, november: 10, dec: 11, december: 11,
};
const REMINDER_OFFSETS = [
  { label: '2 days before', relativeOffset: -2880 },
  { label: '1 day before', relativeOffset: -1440 },
  { label: '1 hour before', relativeOffset: -60 },
];

function usable(value) {
  return !NOT_AVAILABLE.has(String(value || '').trim().toLowerCase());
}

function createDate(year, month, day) {
  const date = new Date(year, month, day);
  return date.getFullYear() === year && date.getMonth() === month && date.getDate() === day ? date : null;
}

// Parses invitation date parts explicitly; the original OCR value remains unchanged in the UI.
export function parseDate(value) {
  if (!usable(value)) return null;
  const input = String(value).trim();
  const dayFirst = input.match(/^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})$/i);
  if (dayFirst && MONTHS[dayFirst[2].toLowerCase()] !== undefined) return createDate(Number(dayFirst[3]), MONTHS[dayFirst[2].toLowerCase()], Number(dayFirst[1]));
  const monthFirst = input.match(/^([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+(\d{4})$/i);
  if (monthFirst && MONTHS[monthFirst[1].toLowerCase()] !== undefined) return createDate(Number(monthFirst[3]), MONTHS[monthFirst[1].toLowerCase()], Number(monthFirst[2]));
  const numeric = input.match(/^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$/);
  if (numeric) return createDate(Number(numeric[3]), Number(numeric[2]) - 1, Number(numeric[1]));
  const iso = input.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (iso) return createDate(Number(iso[1]), Number(iso[2]) - 1, Number(iso[3]));
  return null;
}

export function parseTime(value) {
  if (!usable(value)) return null;
  const input = String(value).trim().toUpperCase().replace(/\s+/g, ' ');
  const twelveHour = input.match(/^(\d{1,2})(?:[.:](\d{2}))?\s*(A\.?M\.?|P\.?M\.?)$/);
  if (twelveHour) {
    let hours = Number(twelveHour[1]);
    const minutes = Number(twelveHour[2] || 0);
    const isPm = twelveHour[3].startsWith('P');
    if (hours < 1 || hours > 12 || minutes > 59) return null;
    if (isPm && hours !== 12) hours += 12;
    if (!isPm && hours === 12) hours = 0;
    return { hours, minutes };
  }
  const twentyFour = input.match(/^(\d{1,2}):(\d{2})$/);
  if (!twentyFour) return null;
  const hours = Number(twentyFour[1]);
  const minutes = Number(twentyFour[2]);
  return hours <= 23 && minutes <= 59 ? { hours, minutes } : null;
}

export function eventDate(event) {
  const date = parseDate(event.date);
  if (!date) return null;
  const time = usable(event.time) && parseTime(event.time);
  if (time) date.setHours(time.hours, time.minutes, 0, 0);
  return date;
}

function eventEndDate(event, startDate) {
  const end = new Date(startDate.getTime() + 60 * 60 * 1000);
  const time = parseTime(event.end_time);
  if (!time) return end;
  end.setHours(time.hours, time.minutes, 0, 0);
  if (end <= startDate) end.setDate(end.getDate() + 1);
  return end;
}

function calendarDescription(event) {
  return [
    usable(event.event_type) && `Event type: ${event.event_type}`,
    usable(event.bride_name) && `Bride: ${event.bride_name}`,
    usable(event.groom_name) && `Groom: ${event.groom_name}`,
    usable(event.contact_number) && `Contact: ${event.contact_number}`,
  ].filter(Boolean).join('\n');
}

// expo-calendar's relativeOffset is minutes from the event start. Exclude
// reminders that have already passed, while leaving the event itself untouched.
export function reminderAlarms(startDate, now = new Date()) {
  return REMINDER_OFFSETS.filter(({ relativeOffset }) => (
    new Date(startDate.getTime() + relativeOffset * 60 * 1000) > now
  )).map(({ relativeOffset }) => ({ relativeOffset }));
}

export function validateCalendarEvents(events) {
  const invalid = events.findIndex((event) => !eventDate(event));
  if (invalid === -1) return { valid: true };
  const rawDate = String(events[invalid]?.date || '').trim();
  return {
    valid: false,
    eventIndex: invalid,
    reason: /\b\d{4}\b/.test(rawDate)
      ? `Date is invalid for Event ${invalid + 1}. Please enter a valid date before adding to calendar.`
      : `Year is required for Event ${invalid + 1}. Please enter the year before adding to calendar.`,
  };
}

async function createViaExpoCalendar(events) {
  let permission = await Calendar.getCalendarPermissionsAsync();
  if (permission.status !== 'granted') permission = await Calendar.requestCalendarPermissionsAsync();
  if (permission.status !== 'granted') return { ok: false, reason: 'Calendar permission is required to schedule this event.' };

  const calendars = await Calendar.getCalendarsAsync(Calendar.EntityTypes.EVENT);
  console.log('[calendarService] Calendars returned:', calendars.length);
  calendars.forEach((item) => {
    console.log('[calendarService] Calendar:', JSON.stringify({
      id: item.id,
      title: item.title,
      name: item.name,
      entityType: item.entityType,
      allowsModifications: item.allowsModifications,
      source: item.source,
      sourceName: item.source?.name,
      sourceType: item.source?.type,
      isPrimary: item.isPrimary,
      ownerAccount: item.ownerAccount,
      accessLevel: item.accessLevel,
    }));
  });
  const writableCalendars = calendars.filter((item) => (
    item?.id
    && (item.entityType === undefined || item.entityType === Calendar.EntityTypes.EVENT)
    && item.allowsModifications !== false
  ));
  const calendar = writableCalendars.find((item) => item.allowsModifications === true && item.isPrimary)
    || writableCalendars.find((item) => item.allowsModifications === true)
    || writableCalendars[0];
  if (!calendar?.id) return { ok: false, reason: 'No writable calendar is available on this phone.' };
  console.log('[calendarService] Selected calendar name:', calendar.title || calendar.name || 'Unnamed calendar');
  console.log('[calendarService] Selected calendar ID:', calendar.id);

  let created = 0;
  for (const event of events) {
    const startDate = eventDate(event);
    const alarms = reminderAlarms(startDate);
    REMINDER_OFFSETS.forEach(({ label, relativeOffset }, index) => {
      if (alarms.some((alarm) => alarm.relativeOffset === relativeOffset)) {
        console.log(`[calendarService] Reminder ${index + 1}: ${label}`);
      } else {
        console.log(`[calendarService] Reminder warning: ${label} is in the past and was not added.`);
      }
    });
    console.log('[calendarService] Reminder alarms:', JSON.stringify(alarms));

    const eventDetails = {
      title: usable(event.event_name) ? event.event_name : (event.event_type || 'Invitation event'),
      startDate,
      endDate: eventEndDate(event, startDate),
      location: [event.venue, event.address].filter(usable).join(', ') || undefined,
      notes: calendarDescription(event) || undefined,
      ...(alarms.length ? { alarms } : {}),
    };

    let eventId;
    try {
      eventId = await Calendar.createEventAsync(calendar.id, eventDetails);
    } catch (error) {
      // Calendar providers can reject alarm settings independently. Preserve
      // the primary event by retrying once with the unchanged event details.
      console.log('[calendarService] Reminder warning:', error?.message || String(error));
      const { alarms: ignoredAlarms, ...eventDetailsWithoutAlarms } = eventDetails;
      eventId = await Calendar.createEventAsync(calendar.id, eventDetailsWithoutAlarms);
    }
    if (!eventId) throw new Error('The phone calendar did not return an event ID.');
    console.log('[calendarService] Created event ID:', eventId);
    created += 1;
  }
  return { ok: true, created };
}

export async function createConfirmedCalendarEvents(events) {
  const validation = validateCalendarEvents(events);
  if (!validation.valid) return { ok: false, reason: validation.reason };
  try {
    return await createViaExpoCalendar(events);
  } catch (error) {
    return { ok: false, reason: error?.message || 'Unable to create the phone calendar event.' };
  }
}
