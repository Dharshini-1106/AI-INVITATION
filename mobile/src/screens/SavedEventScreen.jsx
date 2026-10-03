import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import colors from '../theme/colors';
import TravelPlanModal from '../components/TravelPlanModal';
import { createConfirmedCalendarEvents, validateCalendarEvents } from '../services/calendarService';
import { deleteEvent, getEvent, saveEventSchedule, updateEvent } from '../services/api';

const FIELDS = [
  ['event_name', 'Event Name'], ['event_type', 'Event Type'], ['date', 'Date'],
  ['time', 'Time'], ['end_time', 'End Time'], ['venue', 'Venue'], ['address', 'Address'],
  ['contact_number', 'Contact'], ['additional_information', 'Additional Information'],
];

function formatTravelDateTime(value, timezone) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  try {
    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'long',
      timeStyle: 'short',
      ...(timezone ? { timeZone: timezone } : {}),
    }).format(date);
  } catch {
    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'long',
      timeStyle: 'short',
    }).format(date);
  }
}

export default function SavedEventScreen({ navigation, route }) {
  const { eventId } = route.params || {};
  const [record, setRecord] = useState(null);
  const [draft, setDraft] = useState({});
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [scheduling, setScheduling] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const saved = await getEvent(eventId);
      setRecord(saved);
      setDraft(saved.event || {});
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || 'Unable to load this saved event.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => navigation.addListener('focus', load), [navigation, eventId]);

  const save = async () => {
    setSaving(true);
    setError('');
    setMessage('');
    try {
      const updated = await updateEvent(eventId, draft);
      setRecord(updated);
      setDraft(updated.event || {});
      setEditing(false);
      setMessage('Event changes saved.');
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || 'Unable to save event changes. Check your connection and try again.');
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    try {
      await deleteEvent(eventId);
      navigation.replace('SavedEvents');
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || 'Unable to delete this event.');
    }
  };

  const openMaps = async () => {
    const url = record?.travel_plan?.google_maps_url;
    if (!url) return;
    try {
      if (await Linking.canOpenURL(url)) await Linking.openURL(url);
      else setError('Google Maps could not be opened on this device.');
    } catch {
      setError('Google Maps could not be opened on this device.');
    }
  };

  const scheduleEvent = async () => {
    const validation = validateCalendarEvents([draft]);
    if (!validation.valid) {
      setError(validation.reason);
      return;
    }
    setScheduling(true);
    setError('');
    setMessage('');
    try {
      const existingIds = record.schedule?.calendar_event_ids || [];
      const result = await createConfirmedCalendarEvents([draft], existingIds);
      if (!result.ok) {
        setError(result.reason);
        return;
      }
      const updated = await saveEventSchedule(eventId, {
        status: 'scheduled',
        scheduled_at: record.schedule?.scheduled_at || new Date().toISOString(),
        calendar_event_ids: result.eventIds || existingIds,
        event_date: draft.date || '',
        event_time: draft.time || '',
      });
      setRecord(updated);
      setMessage(existingIds.length ? 'Calendar event updated.' : 'Calendar event added.');
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || requestError?.message || 'Unable to update the calendar event.');
    } finally {
      setScheduling(false);
    }
  };

  if (loading) return <SafeAreaView style={styles.container}><View style={styles.center}><ActivityIndicator color={colors.secondary} /><Text style={styles.muted}>Loading event...</Text></View></SafeAreaView>;
  if (!record) return <SafeAreaView style={styles.container}><View style={styles.center}><Text style={styles.error}>{error || 'Event not found.'}</Text><TouchableOpacity onPress={() => navigation.goBack()}><Text style={styles.link}>Back to My Events</Text></TouchableOpacity></View></SafeAreaView>;

  const savedOrigin = record.travel_plan?.request?.origin || record.travel_plan?.origin;
  const travelTimezone = record.travel_plan?.timezone || draft.timezone || 'Asia/Kolkata';
  const originLabel = typeof savedOrigin === 'string'
    ? savedOrigin
    : savedOrigin?.address || (savedOrigin?.latitude != null && savedOrigin?.longitude != null
      ? `${savedOrigin.latitude}, ${savedOrigin.longitude}`
      : 'Not available');

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <TouchableOpacity onPress={() => navigation.goBack()}><Text style={styles.link}>Back to My Events</Text></TouchableOpacity>
        <Text style={styles.title}>{draft.event_name || draft.event_type || 'Invitation event'}</Text>
        {FIELDS.map(([field, label]) => (
          <View key={field} style={styles.field}>
            <Text style={styles.label}>{label}</Text>
            {editing ? <TextInput style={styles.input} value={String(draft[field] || '')} onChangeText={(value) => setDraft((current) => ({ ...current, [field]: value }))} multiline={field === 'additional_information'} /> : <Text style={styles.value}>{draft[field] || 'Not available'}</Text>}
          </View>
        ))}
        {record.travel_plan ? <>
          <Text style={styles.section}>Saved Travel Plan</Text>
          <Text style={styles.value}>{record.travel_plan.travel_mode || 'Mode unavailable'} · {record.travel_plan.travel_duration_text || 'Duration unavailable'}</Text>
          <Text style={styles.detail}>From: {originLabel}</Text>
          <Text style={styles.detail}>Destination: {record.travel_plan.destination || draft.venue || draft.address || 'Not available'}</Text>
          {record.travel_plan.request?.preparation_minutes != null && <Text style={styles.detail}>Preparation time: {record.travel_plan.request.preparation_minutes} minutes</Text>}
          {!!record.travel_plan.arrival_time && <Text style={styles.detail}>Recommended arrival: {formatTravelDateTime(record.travel_plan.arrival_time, travelTimezone)}</Text>}
          {!!record.travel_plan.departure_time && <Text style={styles.detail}>Suggested departure: {formatTravelDateTime(record.travel_plan.departure_time, travelTimezone)}</Text>}
          {!!record.travel_plan.calculated_at && <Text style={styles.detail}>Plan calculated: {formatTravelDateTime(record.travel_plan.calculated_at)}</Text>}
          {!!record.travel_plan.google_maps_url && <TouchableOpacity style={styles.secondary} onPress={openMaps}><Text style={styles.secondaryText}>Open in Google Maps</Text></TouchableOpacity>}
        </> : null}
        <TravelPlanModal event={draft} eventId={record.id} savedPlan={record.travel_plan} />
        {record.schedule?.status === 'scheduled' ? <Text style={styles.detail}>Calendar: Scheduled for {[record.schedule.event_date, record.schedule.event_time].filter(Boolean).join(' · ') || 'saved event time'}</Text> : null}
        {!!message && <Text style={styles.success}>{message}</Text>}
        {!!error && <Text style={styles.error}>{error}</Text>}
        <TouchableOpacity style={styles.secondary} disabled={scheduling} onPress={scheduleEvent}><Text style={styles.secondaryText}>{scheduling ? 'Saving to calendar...' : record.schedule?.calendar_event_ids?.length ? 'Update Calendar Event' : 'Add to Calendar'}</Text></TouchableOpacity>
        {editing ? <TouchableOpacity style={styles.primary} disabled={saving} onPress={save}><Text style={styles.primaryText}>{saving ? 'Saving...' : 'Save Changes'}</Text></TouchableOpacity> : <TouchableOpacity style={styles.primary} onPress={() => { setError(''); setEditing(true); }}><Text style={styles.primaryText}>Edit Event</Text></TouchableOpacity>}
        <TouchableOpacity style={styles.delete} onPress={remove}><Text style={styles.deleteText}>Delete Event</Text></TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 20, paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  title: { color: colors.text, fontSize: 24, fontWeight: '700', marginTop: 18, marginBottom: 10 },
  section: { color: colors.secondary, fontSize: 18, fontWeight: '700', marginTop: 22, marginBottom: 8 },
  field: { borderBottomWidth: 1, borderColor: colors.border, paddingVertical: 10 },
  label: { color: colors.textMuted, fontSize: 12 },
  value: { color: colors.text, fontSize: 15, fontWeight: '600', marginTop: 4 },
  detail: { color: colors.textMuted, lineHeight: 21, marginTop: 7 },
  input: { color: colors.text, borderWidth: 1, borderColor: colors.border, borderRadius: 8, padding: 10, marginTop: 6 },
  link: { color: colors.primary, fontWeight: '700' },
  muted: { color: colors.textMuted },
  primary: { backgroundColor: colors.primary, borderRadius: 10, padding: 14, alignItems: 'center', marginTop: 16 },
  primaryText: { color: '#fff', fontWeight: '700' },
  secondary: { borderColor: colors.border, borderWidth: 1, borderRadius: 10, padding: 12, alignItems: 'center', marginTop: 12 },
  secondaryText: { color: colors.primary, fontWeight: '700' },
  delete: { padding: 14, alignItems: 'center', marginTop: 8 },
  deleteText: { color: '#dc2626', fontWeight: '700' },
  error: { color: '#dc2626', textAlign: 'center', marginTop: 12 },
  success: { color: colors.success, marginTop: 12, fontWeight: '600' },
});
