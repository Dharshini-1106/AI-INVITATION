import React, { useMemo, useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import colors from '../theme/colors';
import { createConfirmedCalendarEvents, validateCalendarEvents } from '../services/calendarService';

const FIELDS = [
  ['event_name', 'Event Name'], ['event_type', 'Event Type'], ['bride_name', 'Bride'],
  ['groom_name', 'Groom'], ['date', 'Date'], ['time', 'Time'], ['end_time', 'End Time'],
  ['venue', 'Venue'], ['address', 'Address'], ['contact_number', 'Contact'],
];
const display = (value) => value === undefined || value === null || value === '' ? 'Not available' : String(value);

export default function ResultScreen({ navigation, route }) {
  const { result } = route.params || {};
  const initialEvents = useMemo(() => result?.events?.length ? result.events : [result?.primaryEvent || {}], [result]);
  const [events, setEvents] = useState(() => initialEvents.map((event) => ({ ...event })));
  const [editing, setEditing] = useState(false);
  const [scheduling, setScheduling] = useState(false);
  const [message, setMessage] = useState('');
  const [scheduled, setScheduled] = useState(false);
  const update = (index, field, value) => setEvents((current) => current.map((event, i) => i === index ? { ...event, [field]: value } : event));

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
      const outcome = await createConfirmedCalendarEvents(events);
      if (!outcome.ok) {
        setMessage(outcome.reason);
        return;
      }
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
    {events.map((event, eventIndex) => <View key={`${event.event_name || 'event'}-${eventIndex}`} style={styles.card}><Text style={styles.eventTitle}>Event {eventIndex + 1}{event.event_name ? ` - ${event.event_name}` : ''}</Text>{FIELDS.map(([field, label]) => <View key={field} style={styles.row}><Text style={styles.label}>{label}</Text>{editing && !scheduled ? <TextInput style={styles.input} value={String(event[field] ?? '')} onChangeText={(value) => update(eventIndex, field, value)} placeholder="Not available" placeholderTextColor={colors.textMuted} /> : <Text style={styles.value}>{display(event[field])}</Text>}</View>)}</View>)}
    <View style={styles.card}>
      <Text style={styles.eventTitle}>OCR details</Text>
      <Text style={styles.label}>Language</Text><Text style={styles.value}>{display(result.language)}</Text>
      <Text style={styles.label}>Engine / confidence</Text><Text style={styles.value}>{display(result.ocr_engine)} / {Math.round((result.ocr_confidence || 0) * 100)}%</Text>
      <Text style={styles.label}>Tamil / English characters</Text><Text style={styles.value}>{result.tamil_character_count || 0} / {result.english_character_count || 0}</Text>
      {!!result.raw_text && <><Text style={styles.label}>Raw extracted text</Text><Text selectable style={styles.rawText}>{result.raw_text}</Text></>}
      {(result.processing_notes || []).map((note, index) => <Text key={`${note}-${index}`} style={styles.note}>{note}</Text>)}
    </View>
    {!!message && <Text style={scheduled ? styles.success : styles.error}>{message}</Text>}
    {!scheduled && (editing ? <TouchableOpacity style={styles.primary} onPress={() => { setEditing(false); setMessage(''); }}><Text style={styles.primaryText}>Save Edits and Review</Text></TouchableOpacity> : <><TouchableOpacity style={styles.primary} disabled={scheduling} onPress={schedule}><Text style={styles.primaryText}>{scheduling ? 'Scheduling...' : 'Save and Schedule'}</Text></TouchableOpacity><TouchableOpacity style={styles.secondary} onPress={() => setEditing(true)}><Text style={styles.secondaryText}>Edit Details</Text></TouchableOpacity></>)}
    <TouchableOpacity style={styles.home} onPress={() => navigation.replace('Home')}><Text style={styles.secondaryText}>Analyze Another Invitation</Text></TouchableOpacity>
  </ScrollView></SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background }, content: { padding: 20, paddingBottom: 40 }, empty: { flex: 1, justifyContent: 'center', alignItems: 'center' }, title: { color: colors.text, fontSize: 23, fontWeight: '700' }, subtitle: { color: colors.textMuted, marginTop: 8, lineHeight: 20 }, card: { backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 14, padding: 14, marginBottom: 14 }, eventTitle: { color: colors.secondary, fontSize: 16, fontWeight: '700', marginBottom: 8 }, row: { borderTopColor: colors.border, borderTopWidth: 1, paddingVertical: 9 }, label: { color: colors.textMuted, fontSize: 12, marginTop: 8 }, value: { color: colors.text, fontSize: 15, fontWeight: '600', marginTop: 3 }, rawText: { color: colors.text, lineHeight: 20, marginTop: 4 }, note: { color: colors.textMuted, fontSize: 12, marginTop: 6 }, input: { color: colors.text, borderColor: colors.border, borderWidth: 1, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, marginTop: 5 }, primary: { backgroundColor: colors.primary, borderRadius: 12, padding: 15, alignItems: 'center', marginTop: 8 }, primaryText: { color: '#fff', fontWeight: '700' }, secondary: { borderColor: colors.border, borderWidth: 1, borderRadius: 12, padding: 15, alignItems: 'center', marginTop: 10 }, home: { alignItems: 'center', padding: 15, marginTop: 10 }, secondaryText: { color: colors.text, fontWeight: '700' }, error: { color: '#dc2626', lineHeight: 20, marginVertical: 8 }, success: { color: colors.success, lineHeight: 20, marginVertical: 8, fontWeight: '600' }, muted: { color: colors.textMuted }, homeText: { color: colors.primary, marginTop: 12, fontWeight: '700' },
});
