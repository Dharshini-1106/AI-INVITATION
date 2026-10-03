import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import colors from '../theme/colors';
import { listEvents } from '../services/api';

function eventTitle(event) {
  return event?.event_name || event?.event_type || 'Invitation event';
}

function participantSummary(event) {
  const namedPeople = [
    event?.bride_name && `Bride: ${event.bride_name}`,
    event?.groom_name && `Groom: ${event.groom_name}`,
  ].filter(Boolean);
  if (namedPeople.length) return namedPeople.join(' · ');

  return (event?.people || [])
    .filter((person) => person?.name)
    .map((person) => person.role && !/^person$/i.test(person.role)
      ? `${person.role}: ${person.name}` : `Name: ${person.name}`)
    .join(' · ');
}

export default function SavedEventsScreen({ navigation }) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    setError('');
    try {
      setEvents(await listEvents());
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || 'Unable to load saved events. Check your connection and try again.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => navigation.addListener('focus', () => load()), [navigation, load]);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => load(true)} tintColor={colors.secondary} />}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}><Text style={styles.back}>Back</Text></TouchableOpacity>
          <Text style={styles.title}>My Events</Text>
          <TouchableOpacity onPress={() => load(true)} accessibilityRole="button"><Text style={styles.refresh}>Refresh</Text></TouchableOpacity>
        </View>
        {loading ? <View style={styles.state}><ActivityIndicator color={colors.secondary} /><Text style={styles.muted}>Loading saved events...</Text></View> : null}
        {!loading && error ? <View style={styles.state}><Text style={styles.error}>{error}</Text><TouchableOpacity style={styles.button} onPress={() => load()}><Text style={styles.buttonText}>Try Again</Text></TouchableOpacity></View> : null}
        {!loading && !error && events.length === 0 ? <View style={styles.state}><Text style={styles.title}>No saved events</Text><Text style={styles.muted}>Events you save after reviewing an invitation will appear here.</Text></View> : null}
        {!loading && !error && events.map((record) => {
          const event = record.event || {};
          const participants = participantSummary(event);
          return (
            <TouchableOpacity
              key={record.id}
              style={styles.row}
              onPress={() => navigation.navigate('SavedEvent', { eventId: record.id })}
              accessibilityRole="button"
            >
              <Text style={styles.eventTitle}>{eventTitle(event)}</Text>
              {participants ? <Text style={styles.people}>{participants}</Text> : null}
              <Text style={styles.details}>{[event.date, event.time].filter(Boolean).join(' · ') || 'Date and time unavailable'}</Text>
              <Text style={styles.details}>{[event.venue, event.address].filter(Boolean).join(', ') || 'Venue unavailable'}</Text>
              <Text style={styles.plan}>{record.travel_plan ? 'Travel plan saved' : 'No travel plan saved'}</Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 20, paddingBottom: 36 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 },
  title: { color: colors.text, fontSize: 22, fontWeight: '700' },
  back: { color: colors.primary, fontWeight: '700' },
  refresh: { color: colors.secondary, fontWeight: '700' },
  row: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border, borderRadius: 12, padding: 16, marginBottom: 12 },
  eventTitle: { color: colors.text, fontSize: 17, fontWeight: '700' },
  people: { color: colors.text, fontSize: 14, fontWeight: '600', marginTop: 7 },
  details: { color: colors.textMuted, marginTop: 6, lineHeight: 20 },
  plan: { color: colors.secondary, fontWeight: '600', marginTop: 10 },
  state: { alignItems: 'center', padding: 30, gap: 12 },
  muted: { color: colors.textMuted, textAlign: 'center', lineHeight: 20 },
  error: { color: '#dc2626', textAlign: 'center', lineHeight: 20 },
  button: { backgroundColor: colors.primary, paddingHorizontal: 18, paddingVertical: 12, borderRadius: 9 },
  buttonText: { color: '#fff', fontWeight: '700' },
});
