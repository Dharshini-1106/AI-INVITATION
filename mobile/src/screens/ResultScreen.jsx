import React, { useMemo, useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import colors from '../theme/colors';
import { createConfirmedCalendarEvents, validateCalendarEvents } from '../services/calendarService';
import TravelPlanModal from '../components/TravelPlanModal';
import { createEvent, saveEventSchedule, updateEvent } from '../services/api';

const EVENT_FIELDS = [
  ['event_name', 'Event Name'], ['event_type', 'Event Type'], ['date', 'Date'],
  ['day', 'Day'], ['time', 'Time'], ['end_time', 'End Time'], ['venue', 'Venue'],
  ['address', 'Address'], ['contact_number', 'Contact'],
];
const display = (value) => value === undefined || value === null || value === '' ? 'Not available' : String(value);

export default function ResultScreen({ navigation, route }) {
  const { result } = route.params || {};
  const rawText = result?.raw_text || '';
  const derivedTamilCount = [...rawText].filter((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint >= 0x0B80 && codePoint <= 0x0BFF;
  }).length;
  const derivedEnglishCount = [...rawText].filter((character) => (
    (character >= 'A' && character <= 'Z')
    || (character >= 'a' && character <= 'z')
  )).length;
  const engineFromNotes = (result?.processing_notes || [])
    .find((note) => /^OCR engine\(s\):/i.test(note))
    ?.replace(/^OCR engine\(s\):\s*/i, '')
    .split(',')
    .map((engine) => engine.trim())
    .filter((engine) => /paddleocr|rapidocr/i.test(engine))
    .join(', ');
  const engineFromResult = String(result?.ocr_engine || '')
    .split(',')
    .map((engine) => engine.trim())
    .filter((engine) => /paddleocr|rapidocr/i.test(engine))
    .join(', ');
  const ocrEngine = engineFromResult || engineFromNotes || result?.ocr_engine;
  const lineConfidences = (result?.ocr_layout || [])
    .map((line) => Number(line.confidence))
    .filter((confidence) => Number.isFinite(confidence) && confidence > 0);
  const derivedOcrConfidence = lineConfidences.length
    ? lineConfidences.reduce((total, confidence) => total + confidence, 0) / lineConfidences.length
    : null;
  const ocrConfidence = result?.ocr_confidence ?? derivedOcrConfidence;
  const resultPeople = (result?.people || []).filter(
    (person) => person && (person.name || person.role),
  );
  const fallbackPeople = resultPeople.length ? resultPeople : [
    result?.bride_name && { name: result.bride_name, role: 'Bride' },
    result?.groom_name && { name: result.groom_name, role: 'Groom' },
  ].filter(Boolean);
  const initialEvents = useMemo(
    () => (result?.events?.length ? result.events : [result || {}]).map((event) => ({
      ...event,
      people: event.people?.length ? event.people : fallbackPeople,
    })),
    [
      result?.events,
      result?.event_name,
      result?.event_type,
      result?.bride_name,
      result?.groom_name,
      result?.date,
      result?.time,
      result?.end_time,
      result?.venue,
      result?.address,
      result?.contact_number,
      result?.timezone,
      result?.people,
      fallbackPeople,
    ],
  );
  const [events, setEvents] = useState(() => initialEvents.map((event) => ({ ...event })));
  const [people, setPeople] = useState(() => {
    const extracted = resultPeople;
    if (extracted.length) return extracted.map((person) => ({ ...person }));
    return [
      result?.bride_name && { name: result.bride_name, role: 'Bride' },
      result?.groom_name && { name: result.groom_name, role: 'Groom' },
    ].filter(Boolean);
  });
  const [editing, setEditing] = useState(false);
  const [scheduling, setScheduling] = useState(false);
  const [message, setMessage] = useState('');
  const [scheduled, setScheduled] = useState(false);
  const [eventIds, setEventIds] = useState(() => initialEvents.map((event) => event.id || null));
  const [saving, setSaving] = useState(false);
  const update = (index, field, value) => setEvents((current) => current.map((event, i) => i === index ? { ...event, [field]: value } : event));
  const updatePerson = (index, value) => {
    const person = people[index];
    setPeople((current) => current.map((entry, i) => i === index ? { ...entry, name: value } : entry));
    const field = person?.role?.toLowerCase() === 'bride'
      ? 'bride_name'
      : person?.role?.toLowerCase() === 'groom' ? 'groom_name' : null;
    setEvents((current) => current.map((event) => ({
      ...event,
      ...(field ? { [field]: value } : {}),
      people: (event.people || []).map((entry, personIndex) => (
        personIndex === index ? { ...entry, name: value } : entry
      )),
    })));
  };

  const persistEvent = async (index) => {
    const saved = eventIds[index]
      ? await updateEvent(eventIds[index], events[index])
      : await createEvent(events[index]);
    setEventIds((current) => current.map((id, i) => i === index ? saved.id : id));
    return saved.id;
  };

  const saveEvents = async () => {
    setSaving(true);
    setMessage('');
    try {
      for (let index = 0; index < events.length; index += 1) await persistEvent(index);
      setEditing(false);
      setMessage('Event saved to My Events.');
      return true;
    } catch (error) {
      setMessage(error?.response?.data?.detail || error?.message || 'Unable to save this event. Check your connection and try again.');
      return false;
    } finally {
      setSaving(false);
    }
  };

  const schedule = async () => {
    const validation = validateCalendarEvents(events);
    if (!validation.valid) {
      setEditing(true);
      setMessage(validation.reason);
      return;
    }
    setScheduling(true);
    setMessage('');
    try {
      const savedIds = [];
      for (let index = 0; index < events.length; index += 1) savedIds.push(await persistEvent(index));
      const existingCalendarIds = await Promise.all(savedIds.map(async (id) => {
        const eventIndex = savedIds.indexOf(id);
        return events[eventIndex]?.schedule?.calendar_event_ids?.[0] || null;
      }));
      const outcome = await createConfirmedCalendarEvents(events, existingCalendarIds);
      if (!outcome.ok) {
        setMessage(outcome.reason);
        return;
      }
      await Promise.all(savedIds.map((id, index) => saveEventSchedule(id, {
        status: 'scheduled',
        scheduled_at: new Date().toISOString(),
        calendar_event_ids: outcome.eventIds?.[index] ? [outcome.eventIds[index]] : [],
        event_date: events[index].date || '',
        event_time: events[index].time || '',
      })));
      setScheduled(true);
      setMessage('Event added successfully.');
    } catch (error) {
      setMessage(error?.message || 'Unable to create the phone calendar event.');
    } finally {
      setScheduling(false);
    }
  };

  if (!result) return <SafeAreaView style={styles.container}><View style={styles.empty}><Text style={styles.muted}>No result found.</Text><TouchableOpacity onPress={() => navigation.replace('Home')}><Text style={styles.homeText}>Go Home</Text></TouchableOpacity></View></SafeAreaView>;

  return <SafeAreaView style={styles.container}><ScrollView contentContainerStyle={styles.content}>
    <Text style={styles.title}>{scheduled ? 'Event Added Successfully' : 'We extracted this information.'}</Text>
    <Text style={styles.subtitle}>{scheduled ? 'Your confirmed event details were added to your calendar.' : 'Is everything correct?'}</Text>

    <View style={styles.card}>
      <Text style={styles.eventTitle}>People / Participants</Text>
      {people.length === 0 ? <Text style={styles.muted}>No people identified</Text> : people.map((p, idx) => (
          <View key={idx} style={styles.row}>
            <Text style={styles.label}>{p.role || 'Person'}</Text>
            {editing && !scheduled ? (
              <TextInput
                style={styles.input}
                value={String(p.name ?? '')}
                onChangeText={(value) => updatePerson(idx, value)}
                placeholder="Enter name"
                placeholderTextColor={colors.textMuted}
                autoCapitalize="words"
              />
            ) : (
              <Text style={styles.value}>{display(p.name)}</Text>
            )}
          </View>
        ))}
      {!editing && !scheduled && people.length > 0 && (
        <TouchableOpacity onPress={() => setEditing(true)} accessibilityRole="button">
          <Text style={styles.editNames}>Edit names</Text>
        </TouchableOpacity>
      )}
    </View>

    {events.map((event, eventIndex) => {
      return (
        <View key={`${event.event_name || 'event'}-${eventIndex}`} style={styles.card}>
          <Text style={styles.eventTitle}>Event {eventIndex + 1}{event.event_name ? ` - ${event.event_name}` : ''}</Text>
          {EVENT_FIELDS.map(([field, label]) => (
            <View key={field} style={styles.row}>
              <Text style={styles.label}>{label}</Text>
              {editing && !scheduled ? (
                <TextInput
                  style={styles.input}
                  value={String(event[field] ?? '')}
                  onChangeText={(value) => update(eventIndex, field, value)}
                  placeholder="Not available"
                  placeholderTextColor={colors.textMuted}
                />
              ) : (
                <Text style={styles.value}>{display(field === 'day' ? (event.day || event.printed_weekday) : event[field])}</Text>
              )}
            </View>
          ))}
          <TravelPlanModal
            event={event}
            eventId={eventIds[eventIndex]}
            onEnsureSaved={() => persistEvent(eventIndex)}
          />
        </View>
      );
    })}

    <View style={styles.card}>
      <Text style={styles.eventTitle}>OCR details</Text>
      <Text style={styles.label}>Language</Text><Text style={styles.value}>{display(result.language)}</Text>
      <Text style={styles.label}>Engine / confidence</Text><Text style={styles.value}>{display(ocrEngine)} / {ocrConfidence == null ? 'Not available' : `${Math.round(ocrConfidence * 100)}%`}</Text>
      <Text style={styles.label}>Tamil / English characters</Text><Text style={styles.value}>{result.tamil_character_count || derivedTamilCount} / {result.english_character_count || derivedEnglishCount}</Text>
      {!!result.raw_text && <><Text style={styles.label}>Raw extracted text</Text><Text selectable style={styles.rawText}>{result.raw_text}</Text></>}
      {(result.processing_notes || []).map((note, index) => <Text key={`${note}-${index}`} style={styles.note}>• {note}</Text>)}
    </View>
    {!!message && <Text style={scheduled ? styles.success : styles.error}>{message}</Text>}
    {!scheduled && (editing ? <TouchableOpacity style={styles.primary} disabled={saving} onPress={saveEvents}><Text style={styles.primaryText}>{saving ? 'Saving...' : 'Save Edits'}</Text></TouchableOpacity> : <><TouchableOpacity style={styles.primary} disabled={scheduling || saving} onPress={saveEvents}><Text style={styles.primaryText}>{saving ? 'Saving...' : 'Save Event'}</Text></TouchableOpacity><TouchableOpacity style={styles.secondary} disabled={scheduling || saving} onPress={schedule}><Text style={styles.secondaryText}>{scheduling ? 'Scheduling...' : 'Save and Schedule'}</Text></TouchableOpacity><TouchableOpacity style={styles.secondary} onPress={() => setEditing(true)}><Text style={styles.secondaryText}>Edit Details</Text></TouchableOpacity></>)}
    <TouchableOpacity style={styles.home} onPress={() => navigation.replace('Home')}><Text style={styles.secondaryText}>Analyze Another Invitation</Text></TouchableOpacity>
  </ScrollView></SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background }, content: { padding: 20, paddingBottom: 40 }, empty: { flex: 1, justifyContent: 'center', alignItems: 'center' }, title: { color: colors.text, fontSize: 23, fontWeight: '700' }, subtitle: { color: colors.textMuted, marginTop: 8, lineHeight: 20 }, card: { backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 14, padding: 14, marginBottom: 14 }, eventTitle: { color: colors.secondary, fontSize: 16, fontWeight: '700', marginBottom: 8 }, row: { borderTopColor: colors.border, borderTopWidth: 1, paddingVertical: 9 }, label: { color: colors.textMuted, fontSize: 12, marginTop: 8 }, value: { color: colors.text, fontSize: 15, fontWeight: '600', marginTop: 3 }, editNames: { color: colors.primary, fontWeight: '700', textAlign: 'right', paddingVertical: 8 }, rawText: { color: colors.text, lineHeight: 20, marginTop: 4 }, note: { color: colors.textMuted, fontSize: 12, marginTop: 6 }, input: { color: colors.text, borderColor: colors.border, borderWidth: 1, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, marginTop: 5 }, primary: { backgroundColor: colors.primary, borderRadius: 12, padding: 15, alignItems: 'center', marginTop: 8 }, primaryText: { color: '#fff', fontWeight: '700' }, secondary: { borderColor: colors.border, borderWidth: 1, borderRadius: 12, padding: 15, alignItems: 'center', marginTop: 10 }, home: { alignItems: 'center', padding: 15, marginTop: 10 }, secondaryText: { color: colors.primary, fontWeight: '700' }, error: { color: '#dc2626', lineHeight: 20, marginVertical: 8 }, success: { color: colors.success, lineHeight: 20, marginVertical: 8, fontWeight: '600' }, muted: { color: colors.textMuted }, homeText: { color: colors.primary, marginTop: 12, fontWeight: '700' },
});
