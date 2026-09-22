import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import * as Location from 'expo-location';
import colors from '../theme/colors';
import { planTravel } from '../services/api';
import { scheduleTravelNotifications } from '../services/travelNotificationService';

const TRAVEL_MODES = [
  { value: 'DRIVE', label: 'Car' },
  { value: 'TWO_WHEELER', label: 'Bike / Scooter' },
  { value: 'TRANSIT', label: 'Bus / Public Transit' },
];

const PREPARATION_OPTIONS = ['30', '60', '90'];
const DEFAULT_TIMEZONE = 'Asia/Kolkata';
const INVALID_ORIGIN_PLACEHOLDERS = new Set([
  'current location',
  'current location selected',
  'select current location',
  'use current location',
  'undefined',
  'null',
  'none',
  'n/a',
]);

function isValidCoordinatePair(latitude, longitude) {
  if (latitude === null || longitude === null || latitude === '' || longitude === '') return false;
  const latitudeNumber = Number(latitude);
  const longitudeNumber = Number(longitude);
  return Number.isFinite(latitudeNumber)
    && latitudeNumber >= -90
    && latitudeNumber <= 90
    && Number.isFinite(longitudeNumber)
    && longitudeNumber >= -180
    && longitudeNumber <= 180;
}

function getManualOrigin(value) {
  const origin = String(value || '').trim();
  return origin && !INVALID_ORIGIN_PLACEHOLDERS.has(origin.toLowerCase()) ? origin : '';
}

function formatTime(value, timezone) {
  if (!value) return 'Not available';
  try {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: timezone || DEFAULT_TIMEZONE,
    }).format(new Date(value));
  } catch (error) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Not available' : date.toLocaleTimeString([], {
      hour: 'numeric',
      minute: '2-digit',
    });
  }
}

function getErrorMessage(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return detail.message;
  return error?.message || 'Travel time is currently unavailable.';
}

export default function TravelPlanModal({ event }) {
  const initialDestination = event?.address || event?.venue || '';
  const eventKey = JSON.stringify([
    event?.date,
    event?.time,
    event?.address,
    event?.venue,
    event?.timezone,
  ]);
  const previousEventKey = useRef(eventKey);
  const [open, setOpen] = useState(false);
  const [originMode, setOriginMode] = useState('current');
  const [origin, setOrigin] = useState(null);
  const [manualOrigin, setManualOrigin] = useState('');
  const [destination, setDestination] = useState(initialDestination);
  const [travelMode, setTravelMode] = useState('DRIVE');
  const [preparationMode, setPreparationMode] = useState('60');
  const [customPreparation, setCustomPreparation] = useState(60);
  const [arrivalBuffer, setArrivalBuffer] = useState(15);
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState('');
  const [mapsUrl, setMapsUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [locationLoading, setLocationLoading] = useState(false);

  useEffect(() => {
    if (previousEventKey.current !== eventKey) {
      previousEventKey.current = eventKey;
      setOrigin(null);
      setManualOrigin('');
    }
  }, [eventKey]);

  useEffect(() => {
    if (open) {
      setDestination(event?.address || event?.venue || '');
      setOriginMode('current');
      setTravelMode('DRIVE');
      setPreparationMode('60');
      setCustomPreparation(60);
      setArrivalBuffer(15);
      setOrigin(null);
      setManualOrigin('');
      setPlan(null);
      setError('');
      setMapsUrl('');
    }
  }, [open, eventKey]);

  const useCurrentLocation = async () => {
    setOriginMode('current');
    setLocationLoading(true);
    setError('');
    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      console.log('[travel] location permission:', permission?.status);
      if (permission?.status !== 'granted') {
        setOrigin(null);
        setError('Location permission is required to use your current location.');
        return;
      }

      const servicesEnabled = await Location.hasServicesEnabledAsync();
      console.log('[travel] location services:', servicesEnabled ? 'enabled' : 'disabled');
      if (!servicesEnabled) {
        try {
          await Location.enableNetworkProviderAsync();
          const servicesEnabledAfterPrompt = await Location.hasServicesEnabledAsync();
          console.log('[travel] location services:', servicesEnabledAfterPrompt ? 'enabled' : 'disabled');
          if (!servicesEnabledAfterPrompt) {
            throw new Error('Location services remain disabled.');
          }
        } catch (servicesError) {
          console.error('[travel] location services error:', servicesError?.message || servicesError);
          setOrigin(null);
          setError('Please enable location services and try again.');
          return;
        }
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });
      const latitude = location?.coords?.latitude;
      const longitude = location?.coords?.longitude;
      console.log('[travel] current coordinates:', latitude, longitude);
      if (!isValidCoordinatePair(latitude, longitude)) {
        throw new Error('Current location coordinates are unavailable or invalid.');
      }

      setOrigin({ type: 'current', latitude: Number(latitude), longitude: Number(longitude) });
      setOriginMode('current');
      setError('');
    } catch (locationError) {
      console.error('[travel] current location error:', locationError?.message || locationError);
      setOrigin(null);
      setError('Could not get your current location. Please try again.');
    } finally {
      setLocationLoading(false);
    }
  };

  const selectManualOrigin = () => {
    setOriginMode('manual');
    setOrigin(null);
    setManualOrigin('');
    setError('');
  };

  const submitPlan = async () => {
    const selectedDestination = destination.trim();
    let selectedOrigin = '';

    if (originMode === 'current') {
      const latitude = origin?.latitude;
      const longitude = origin?.longitude;
      if (!isValidCoordinatePair(latitude, longitude)) {
        setError('Could not get your current location. Please try again.');
        return;
      }
      selectedOrigin = { type: 'current', latitude: Number(latitude), longitude: Number(longitude) };
    } else {
      const address = getManualOrigin(manualOrigin);
      selectedOrigin = address ? { type: 'manual', address } : null;
      if (!selectedOrigin) {
        setError('Could not find this starting location. Please enter a valid address.');
        return;
      }
    }

    if (!selectedDestination) {
      setError('Enter the event destination.');
      return;
    }

    const preparationMinutes = preparationMode === 'custom'
      ? Number(customPreparation)
      : Number(preparationMode);
    const bufferMinutes = Number(arrivalBuffer);
    if (!Number.isFinite(preparationMinutes) || preparationMinutes < 0 || preparationMinutes > 1440) {
      setError('Preparation time must be between 0 and 1440 minutes.');
      return;
    }
    if (!Number.isFinite(bufferMinutes) || bufferMinutes < 0 || bufferMinutes > 1440) {
      setError('Arrival buffer must be between 0 and 1440 minutes.');
      return;
    }

    console.log('[travel] travel request origin:', selectedOrigin);
    setLoading(true);
    setError('');
    setMapsUrl('');
    try {
      const result = await planTravel({
        origin: selectedOrigin,
        destination: { type: 'event', address: selectedDestination },
        event_date: event?.date || '',
        event_start_time: event?.time || '',
        event_timezone: event?.timezone || DEFAULT_TIMEZONE,
        travel_mode: travelMode,
        preparation_minutes: preparationMinutes,
        arrival_buffer_minutes: bufferMinutes,
      });
      setPlan(result);
      setMapsUrl(result?.google_maps_url || '');
      scheduleTravelNotifications({ plan: result, event, destination: selectedDestination }).catch((notificationError) => {
        console.warn('[travel] reminder was not scheduled:', notificationError?.message || notificationError);
      });
    } catch (requestError) {
      const detail = requestError?.response?.data?.detail;
      setMapsUrl(typeof detail === 'object' ? detail?.google_maps_url || '' : '');
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  };

  const openMaps = async () => {
    if (!mapsUrl) return;
    try {
      const supported = await Linking.canOpenURL(mapsUrl);
      if (supported) await Linking.openURL(mapsUrl);
      else setError('Google Maps could not be opened on this device.');
    } catch (openError) {
      setError('Google Maps could not be opened on this device.');
    }
  };

  const selectedMode = TRAVEL_MODES.find((mode) => mode.value === travelMode) || TRAVEL_MODES[0];
  const scheduleAvailable = Boolean(plan?.schedule_available);
  const timezone = plan?.timezone || event?.timezone || DEFAULT_TIMEZONE;
  const scheduleMessage = scheduleAvailable
    ? `Your event starts at ${formatTime(plan.event_start_time, timezone)}. Leave by ${formatTime(plan.departure_time, timezone)} to reach the venue by ${formatTime(plan.arrival_time, timezone)}, before the event starts. Start getting ready by ${formatTime(plan.ready_time, timezone)}.`
    : plan?.schedule_message || plan?.message || '';

  return (
    <>
      <TouchableOpacity style={styles.travelButton} onPress={() => setOpen(true)}>
        <Text style={styles.travelButtonText}>Plan My Travel</Text>
      </TouchableOpacity>

      <Modal animationType="slide" transparent visible={open} onRequestClose={() => setOpen(false)}>
        <View style={styles.overlay}>
          <View style={styles.sheet}>
            <View style={styles.header}>
              <Text style={styles.headerTitle}>Travel Plan</Text>
              <TouchableOpacity onPress={() => setOpen(false)} hitSlop={12}>
                <Text style={styles.closeButton}>×</Text>
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.content} keyboardShouldPersistTaps="handled">
              <Text style={styles.eventName}>{event?.event_name || event?.event_type || 'Invitation event'}</Text>
              <Text style={styles.eventDate}>{[event?.date, event?.time].filter(Boolean).join(' · ') || 'Event time unavailable'}</Text>

              <Text style={styles.label}>Destination</Text>
              <TextInput
                style={styles.input}
                value={destination}
                onChangeText={setDestination}
                placeholder={initialDestination ? 'Extracted destination' : 'Enter destination'}
                placeholderTextColor={colors.textMuted}
              />
              {!initialDestination && <Text style={styles.hint}>Enter the venue or address manually.</Text>}

              <Text style={styles.label}>Where are you starting from?</Text>
              <View style={styles.originControls}>
                <TouchableOpacity
                  style={[styles.modeButton, originMode === 'current' && styles.modeButtonActive]}
                  onPress={useCurrentLocation}
                  disabled={locationLoading}
                >
                  <Text style={[styles.modeButtonText, originMode === 'current' && styles.modeButtonTextActive]}>
                    {locationLoading ? 'Finding location...' : 'Use Current Location'}
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.modeButton, originMode === 'manual' && styles.modeButtonActive]}
                  onPress={selectManualOrigin}
                >
                  <Text style={[styles.modeButtonText, originMode === 'manual' && styles.modeButtonTextActive]}>Enter Manually</Text>
                </TouchableOpacity>
              </View>
              {originMode === 'manual' && (
                <TextInput
                  style={styles.input}
                value={manualOrigin}
                onChangeText={setManualOrigin}
                  placeholder="Starting location"
                  placeholderTextColor={colors.textMuted}
                />
              )}
              {originMode === 'current' && origin && <Text style={styles.hint}>Current location selected. Your precise location is not saved.</Text>}

              <Text style={styles.label}>How are you travelling?</Text>
              <View style={styles.modeGrid}>
                {TRAVEL_MODES.map((mode) => (
                  <TouchableOpacity
                    key={mode.value}
                    style={[styles.modeButton, travelMode === mode.value && styles.modeButtonActive]}
                    onPress={() => setTravelMode(mode.value)}
                  >
                    <Text style={[styles.modeButtonText, travelMode === mode.value && styles.modeButtonTextActive]}>
                      {mode.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              <Text style={styles.label}>How much time do you need to get ready?</Text>
              <View style={styles.optionRow}>
                {PREPARATION_OPTIONS.map((minutes) => (
                  <TouchableOpacity
                    key={minutes}
                    style={[styles.optionButton, preparationMode === minutes && styles.optionButtonActive]}
                    onPress={() => setPreparationMode(minutes)}
                  >
                    <Text style={[styles.optionButtonText, preparationMode === minutes && styles.optionButtonTextActive]}>
                      {minutes} min
                    </Text>
                  </TouchableOpacity>
                ))}
                <TouchableOpacity
                  style={[styles.optionButton, preparationMode === 'custom' && styles.optionButtonActive]}
                  onPress={() => setPreparationMode('custom')}
                >
                  <Text style={[styles.optionButtonText, preparationMode === 'custom' && styles.optionButtonTextActive]}>Custom</Text>
                </TouchableOpacity>
              </View>
              {preparationMode === 'custom' && (
                <TextInput
                  style={styles.input}
                  value={String(customPreparation)}
                  onChangeText={(value) => setCustomPreparation(value.replace(/[^0-9]/g, ''))}
                  keyboardType="number-pad"
                  placeholder="Preparation minutes"
                  placeholderTextColor={colors.textMuted}
                />
              )}

              <Text style={styles.label}>How early do you want to reach?</Text>
              <View style={styles.bufferRow}>
                <TextInput
                  style={styles.bufferInput}
                  value={String(arrivalBuffer)}
                  onChangeText={(value) => setArrivalBuffer(value.replace(/[^0-9]/g, ''))}
                  keyboardType="number-pad"
                  placeholder="15"
                  placeholderTextColor={colors.textMuted}
                />
                <Text style={styles.bufferUnit}>minutes</Text>
              </View>

              <TouchableOpacity style={styles.calculateButton} onPress={submitPlan} disabled={loading || locationLoading}>
                {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.calculateButtonText}>Calculate Travel Plan</Text>}
              </TouchableOpacity>

              {!!error && <Text style={styles.error}>{error}</Text>}

              {plan && (
                <View style={styles.planCard}>
                  <Text style={styles.planTitle}>TRAVEL PLAN</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Event: </Text>{event?.event_name || event?.event_type || 'Invitation event'}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Date: </Text>{event?.date || 'Not available'}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Time: </Text>{event?.time || 'Not available'}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Destination: </Text>{plan.destination || selectedDestinationText(destination, event)}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Starting from: </Text>{originMode === 'current' ? 'Current location' : manualOrigin}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Travel mode: </Text>{selectedMode.label}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Google Maps travel time: </Text>{plan.travel_duration_text || 'Unavailable'}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Distance: </Text>{plan.distance_text || 'Unavailable'}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Get ready by: </Text>{formatTime(plan.ready_time, timezone)}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Leave by: </Text>{formatTime(plan.departure_time, timezone)}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Reach by: </Text>{formatTime(plan.arrival_time, timezone)}</Text>
                  <Text style={styles.planLine}><Text style={styles.planLabel}>Event starts: </Text>{formatTime(plan.event_start_time, timezone)}</Text>
                  {!!scheduleMessage && <Text style={styles.scheduleMessage}>{scheduleMessage}</Text>}
                  {!scheduleAvailable && <Text style={styles.unavailable}>Event time unavailable — travel schedule cannot be calculated.</Text>}
                  {plan.google_maps_url && (
                    <TouchableOpacity style={styles.mapsButton} onPress={openMaps}>
                      <Text style={styles.mapsButtonText}>Open in Google Maps</Text>
                    </TouchableOpacity>
                  )}
                </View>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </>
  );
}

function selectedDestinationText(destination, event) {
  return destination || event?.address || event?.venue || 'Not available';
}

const styles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: 'rgba(0, 0, 0, 0.62)', justifyContent: 'flex-end' },
  sheet: { flex: 1, maxHeight: '92%', backgroundColor: colors.background, borderTopLeftRadius: 22, borderTopRightRadius: 22, padding: 20 },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 },
  headerTitle: { color: colors.text, fontSize: 22, fontWeight: '700' },
  closeButton: { color: colors.textMuted, fontSize: 30, lineHeight: 32 },
  content: { flex: 1 },
  eventName: { color: colors.text, fontSize: 18, fontWeight: '700' },
  eventDate: { color: colors.textMuted, fontSize: 13, marginTop: 4, marginBottom: 16 },
  label: { color: colors.textMuted, fontSize: 12, marginTop: 14, marginBottom: 7 },
  hint: { color: colors.textMuted, fontSize: 12, marginTop: 5, lineHeight: 17 },
  input: { minHeight: 44, color: colors.text, backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, fontSize: 15 },
  originControls: { flexDirection: 'row', gap: 10 },
  modeGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  modeButton: { flex: 1, minWidth: '45%', minHeight: 42, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 10, paddingHorizontal: 10 },
  modeButtonActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  modeButtonText: { color: colors.textMuted, fontSize: 13, fontWeight: '600', textAlign: 'center' },
  modeButtonTextActive: { color: '#fff' },
  optionRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  optionButton: { minHeight: 40, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 10, paddingHorizontal: 13 },
  optionButtonActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  optionButtonText: { color: colors.textMuted, fontSize: 13, fontWeight: '600' },
  optionButtonTextActive: { color: '#fff' },
  bufferRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  bufferInput: { flex: 1, minHeight: 44, color: colors.text, backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, fontSize: 15 },
  bufferUnit: { color: colors.textMuted, fontSize: 14 },
  calculateButton: { minHeight: 48, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.primary, borderRadius: 12, marginTop: 18 },
  calculateButtonText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  error: { color: colors.error, fontSize: 13, lineHeight: 18, marginTop: 12 },
  planCard: { backgroundColor: colors.card, borderColor: colors.border, borderWidth: 1, borderRadius: 14, padding: 15, marginTop: 18 },
  planTitle: { color: colors.secondary, fontSize: 16, fontWeight: '700', letterSpacing: 1, marginBottom: 10 },
  planLine: { color: colors.text, fontSize: 14, lineHeight: 22 },
  planLabel: { color: colors.textMuted },
  scheduleMessage: { color: colors.success, fontSize: 14, lineHeight: 21, marginTop: 12 },
  unavailable: { color: colors.warning, fontSize: 13, lineHeight: 19, marginTop: 10 },
  mapsButton: { minHeight: 46, justifyContent: 'center', alignItems: 'center', borderColor: colors.secondary, borderWidth: 1, borderRadius: 11, marginTop: 14 },
  mapsButtonText: { color: colors.secondary, fontSize: 15, fontWeight: '700' },
  travelButton: { minHeight: 44, justifyContent: 'center', alignItems: 'center', borderColor: colors.secondary, borderWidth: 1, borderRadius: 11, marginTop: 12 },
  travelButtonText: { color: colors.secondary, fontSize: 14, fontWeight: '700' },
});
