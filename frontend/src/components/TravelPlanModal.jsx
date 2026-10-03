import React, { useEffect, useRef, useState } from 'react';
import colors from '../theme/colors';
import { planTravel } from '../services/api';

const TRAVEL_MODES = [
  { value: 'DRIVE', label: '🚗 Car' },
  { value: 'TWO_WHEELER', label: '🏍️ Bike' },
  { value: 'TRANSIT', label: '🚌 Bus / Public Transit' },
];

const PREPARATION_OPTIONS = [30, 60, 90];
const DEFAULT_TIMEZONE = 'Asia/Kolkata';

function formatTime(value, timezone) {
  if (!value) return 'Not available';
  try {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: timezone || DEFAULT_TIMEZONE,
    }).format(new Date(value));
  } catch {
    return new Date(value).toLocaleString();
  }
}

function formatDateTime(value, timezone) {
  if (!value) return 'Not available';
  try {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      timeZone: timezone || DEFAULT_TIMEZONE,
    }).format(new Date(value));
  } catch {
    return new Date(value).toLocaleString();
  }
}

function getErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  return detail?.message || error.message || 'Travel planning failed.';
}

function TravelPlanModal({ event }) {
  const initialDestination = event?.address || event?.venue || '';
  const eventKey = [
    event?.date,
    event?.time,
    event?.address,
    event?.venue,
    event?.timezone,
  ].join('|');
  const previousEventKey = useRef(eventKey);
  const [open, setOpen] = useState(false);
  const [originMode, setOriginMode] = useState('current');
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState(initialDestination);
  const [travelMode, setTravelMode] = useState('DRIVE');
  const [preparationMode, setPreparationMode] = useState('60');
  const [customPreparation, setCustomPreparation] = useState(60);
  const [arrivalBuffer, setArrivalBuffer] = useState(15);
  const [locationLoading, setLocationLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState('');
  const [mapsUrl, setMapsUrl] = useState('');

  useEffect(() => {
    if (previousEventKey.current !== eventKey) {
      previousEventKey.current = eventKey;
      setOrigin('');
    }
  }, [eventKey]);

  useEffect(() => {
    if (open) {
      setDestination(event?.address || event?.venue || '');
      setTravelMode('DRIVE');
      setPreparationMode('60');
      setCustomPreparation(60);
      setArrivalBuffer(15);
      setPlan(null);
      setError('');
      setMapsUrl('');
    }
  }, [open, eventKey]);

  const useCurrentLocation = () => {
    setError('');
    if (!navigator.geolocation) {
      setOriginMode('manual');
      setError('Current location is not available. Enter your starting location manually.');
      return;
    }

    setLocationLoading(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setOrigin(`${position.coords.latitude},${position.coords.longitude}`);
        setOriginMode('current');
        setLocationLoading(false);
      },
      () => {
        setOriginMode('manual');
        setLocationLoading(false);
        setError('Location permission denied. Enter your starting location manually.');
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 },
    );
  };

  const submitPlan = async () => {
    setError('');
    setPlan(null);
    if (!origin.trim()) {
      setError('Enter your starting location or use your current location.');
      return;
    }
    if (!destination.trim()) {
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

    setLoading(true);
    try {
      const response = await planTravel({
        origin: origin.trim(),
        destination: destination.trim(),
        destination_venue: event?.venue || '',
        destination_address: event?.address || '',
        event_date: event?.date || '',
        event_start_time: event?.time || '',
        event_timezone: event?.timezone || '',
        travel_mode: travelMode,
        preparation_minutes: preparationMinutes,
        arrival_buffer_minutes: bufferMinutes,
      });
      setPlan(response);
      setMapsUrl(response.google_maps_url || '');
    } catch (requestError) {
      setError(getErrorMessage(requestError));
      setMapsUrl(requestError.response?.data?.detail?.google_maps_url || '');
    } finally {
      setLoading(false);
    }
  };

  const selectedMode = TRAVEL_MODES.find((mode) => mode.value === travelMode);
  const timezone = plan?.timezone || event?.timezone || '';
  const eventStart = plan?.event_start_time || event?.time || '';
  const scheduleMessage = plan?.schedule_available
    ? `Your event starts at ${formatTime(eventStart, timezone)}. Leave by ${formatTime(plan.departure_time, timezone)} to reach the venue by ${formatTime(plan.arrival_time, timezone)}, before the event starts. Start getting ready by ${formatTime(plan.ready_time, timezone)}.`
    : 'Event time unavailable — travel schedule cannot be calculated.';

  return (
    <>
      <button style={styles.planButton} onClick={() => setOpen(true)}>
        Plan My Travel
      </button>

      {open && (
        <div style={styles.overlay} role="dialog" aria-modal="true" aria-label="Plan my travel">
          <div style={styles.modal}>
            <div style={styles.modalHeader}>
              <div>
                <h2 style={styles.modalTitle}>Plan My Travel</h2>
                <p style={styles.modalEvent}>
                  {event?.event_name || event?.event_type || 'Extracted event'}
                </p>
              </div>
              <button style={styles.closeButton} onClick={() => setOpen(false)} aria-label="Close">
                ×
              </button>
            </div>

            <div style={styles.section}>
              <p style={styles.label}>Where are you starting from?</p>
              <div style={styles.optionRow}>
                <button
                  style={originMode === 'current' ? styles.optionButtonActive : styles.optionButton}
                  onClick={useCurrentLocation}
                  disabled={locationLoading}
                >
                  {locationLoading ? 'Finding location...' : 'Use Current Location'}
                </button>
                <button
                  style={originMode === 'manual' ? styles.optionButtonActive : styles.optionButton}
                  onClick={() => setOriginMode('manual')}
                >
                  Enter Location Manually
                </button>
              </div>
              {originMode === 'manual' && (
                <input
                  style={styles.input}
                  value={origin}
                  onChange={(value) => setOrigin(value.target.value)}
                  placeholder="Starting location"
                />
              )}
              {originMode === 'current' && origin && (
                <p style={styles.valueText}>Current location selected</p>
              )}
            </div>

            <div style={styles.section}>
              <p style={styles.label}>Where are you going?</p>
              <input
                style={styles.input}
                value={destination}
                onChange={(value) => setDestination(value.target.value)}
                placeholder={initialDestination || 'Enter event destination'}
              />
              {!initialDestination && (
                <p style={styles.hint}>No extracted venue or address was found. Enter the destination manually.</p>
              )}
            </div>

            <div style={styles.section}>
              <p style={styles.label}>How are you travelling?</p>
              <div style={styles.modeGrid}>
                {TRAVEL_MODES.map((mode) => (
                  <button
                    key={mode.value}
                    style={travelMode === mode.value ? styles.modeButtonActive : styles.modeButton}
                    onClick={() => setTravelMode(mode.value)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>

            <div style={styles.section}>
              <p style={styles.label}>How much time do you need to get ready?</p>
              <div style={styles.optionRow}>
                {PREPARATION_OPTIONS.map((minutes) => (
                  <button
                    key={minutes}
                    style={preparationMode === String(minutes) ? styles.optionButtonActive : styles.optionButton}
                    onClick={() => setPreparationMode(String(minutes))}
                  >
                    {minutes} minutes
                  </button>
                ))}
                <button
                  style={preparationMode === 'custom' ? styles.optionButtonActive : styles.optionButton}
                  onClick={() => setPreparationMode('custom')}
                >
                  Custom
                </button>
              </div>
              {preparationMode === 'custom' && (
                <input
                  style={styles.input}
                  type="number"
                  min="0"
                  value={customPreparation}
                  onChange={(value) => setCustomPreparation(value.target.value)}
                  placeholder="Preparation minutes"
                />
              )}
            </div>

            <div style={styles.section}>
              <p style={styles.label}>How early do you want to reach?</p>
              <input
                style={styles.input}
                type="number"
                min="0"
                value={arrivalBuffer}
                onChange={(value) => setArrivalBuffer(value.target.value)}
              />
              <p style={styles.hint}>Minutes before the event start time.</p>
            </div>

            {error && <div style={styles.errorBox}>{error}</div>}
            {mapsUrl && (
              <a style={styles.mapsLink} href={mapsUrl} target="_blank" rel="noreferrer">
                Open in Google Maps
              </a>
            )}

            <button
              style={{ ...styles.submitButton, ...(loading ? styles.submitButtonDisabled : {}) }}
              onClick={submitPlan}
              disabled={loading}
            >
              {loading ? 'Calculating route...' : 'Calculate Travel Plan'}
            </button>

            {plan && (
              <div style={styles.planCard}>
                <p style={styles.planTitle}>TRAVEL PLAN</p>
                <div style={styles.planGrid}>
                  <span>Event</span><strong>{event?.event_name || event?.event_type || 'Extracted event'}</strong>
                  <span>Date</span><strong>{formatDateTime(plan.event_start_time, timezone)}</strong>
                  <span>Destination</span><strong>{plan.destination_place_name || plan.destination}</strong>
                  <span>Starting from</span><strong>{plan.origin}</strong>
                  <span>Travel mode</span><strong>{selectedMode?.label || plan.travel_mode}</strong>
                  <span>Google Maps travel time</span><strong>{plan.travel_duration_text || 'Unavailable'}</strong>
                  <span>Distance</span><strong>{plan.distance_text || 'Unavailable'}</strong>
                </div>

                {plan.schedule_available ? (
                  <>
                    <div style={styles.scheduleGrid}>
                      <div><span>Get ready by</span><strong>{formatTime(plan.ready_time, timezone)}</strong></div>
                      <div><span>Leave by</span><strong>{formatTime(plan.departure_time, timezone)}</strong></div>
                      <div><span>Reach by</span><strong>{formatTime(plan.arrival_time, timezone)}</strong></div>
                      <div><span>Event starts</span><strong>{formatTime(plan.event_start_time, timezone)}</strong></div>
                    </div>
                    <p style={styles.scheduleMessage}>{scheduleMessage}</p>
                  </>
                ) : (
                  <p style={styles.unavailableText}>{plan.message || scheduleMessage}</p>
                )}

                {plan.google_maps_url && (
                  <a style={styles.mapsLink} href={plan.google_maps_url} target="_blank" rel="noreferrer">
                    Open in Google Maps
                  </a>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

const styles = {
  planButton: {
    width: '100%', marginTop: '14px', padding: '12px', border: 'none', borderRadius: '11px',
    background: colors.gradient, color: '#fff', fontWeight: 700, fontSize: '14px', cursor: 'pointer',
  },
  overlay: {
    position: 'fixed', inset: 0, background: 'rgba(5, 5, 15, 0.78)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px',
  },
  modal: {
    width: '100%', maxWidth: '520px', maxHeight: '92vh', overflowY: 'auto', background: colors.card,
    border: `1px solid ${colors.border}`, borderRadius: '18px', padding: '20px',
  },
  modalHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', marginBottom: '18px' },
  modalTitle: { margin: 0, fontSize: '19px', color: colors.text },
  modalEvent: { margin: '4px 0 0', fontSize: '13px', color: colors.textMuted },
  closeButton: { background: 'transparent', border: 'none', color: colors.textMuted, fontSize: '26px', cursor: 'pointer', lineHeight: 1 },
  section: { marginBottom: '16px' },
  label: { margin: '0 0 8px', fontSize: '13px', fontWeight: 700, color: colors.text },
  optionRow: { display: 'flex', flexWrap: 'wrap', gap: '8px' },
  optionButton: {
    flex: '1 1 130px', padding: '10px', borderRadius: '10px', border: `1px solid ${colors.border}`,
    background: 'transparent', color: colors.textMuted, fontSize: '12px', fontWeight: 600, cursor: 'pointer',
  },
  optionButtonActive: {
    flex: '1 1 130px', padding: '10px', borderRadius: '10px', border: `1px solid ${colors.primary}`,
    background: colors.cardLight, color: colors.text, fontSize: '12px', fontWeight: 700, cursor: 'pointer',
  },
  input: {
    width: '100%', marginTop: '8px', boxSizing: 'border-box', padding: '11px 12px', borderRadius: '10px',
    border: `1px solid ${colors.border}`, background: colors.background, color: colors.text, fontSize: '14px',
  },
  hint: { margin: '7px 0 0', fontSize: '12px', color: colors.textMuted },
  valueText: { margin: '8px 0 0', fontSize: '12px', color: colors.success },
  modeGrid: { display: 'grid', gridTemplateColumns: '1fr', gap: '8px' },
  modeButton: {
    padding: '11px', borderRadius: '10px', border: `1px solid ${colors.border}`, background: 'transparent',
    color: colors.textMuted, fontSize: '13px', fontWeight: 600, cursor: 'pointer', textAlign: 'left',
  },
  modeButtonActive: {
    padding: '11px', borderRadius: '10px', border: `1px solid ${colors.primary}`, background: colors.cardLight,
    color: colors.text, fontSize: '13px', fontWeight: 700, cursor: 'pointer', textAlign: 'left',
  },
  errorBox: {
    padding: '11px 12px', borderRadius: '10px', background: 'rgba(255, 118, 117, 0.12)',
    border: `1px solid rgba(255, 118, 117, 0.4)`, color: '#ffb3b2', fontSize: '13px', marginBottom: '12px',
  },
  mapsLink: {
    display: 'inline-block', marginTop: '10px', color: colors.secondary, fontSize: '13px', fontWeight: 700,
  },
  submitButton: {
    width: '100%', padding: '12px', borderRadius: '11px', border: 'none', background: colors.gradient,
    color: '#fff', fontWeight: 700, fontSize: '14px', cursor: 'pointer',
  },
  submitButtonDisabled: { opacity: 0.65, cursor: 'not-allowed' },
  planCard: {
    marginTop: '18px', padding: '15px', borderRadius: '13px', border: `1px solid ${colors.border}`,
    background: colors.cardLight,
  },
  planTitle: { margin: '0 0 12px', fontSize: '14px', fontWeight: 800, letterSpacing: '0.06em', color: colors.secondary },
  planGrid: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.3fr)', gap: '8px 12px', fontSize: '12px' },
  'planGrid span': { color: colors.textMuted },
  'planGrid strong': { color: colors.text, overflowWrap: 'anywhere' },
  scheduleGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '14px', padding: '12px',
        borderRadius: '11px', background: colors.background, border: `1px solid ${colors.border}`,
  },
  'scheduleGrid div': { display: 'flex', flexDirection: 'column', gap: '3px' },
  'scheduleGrid span': { fontSize: '11px', color: colors.textMuted },
  'scheduleGrid strong': { fontSize: '14px', color: colors.text },
  scheduleMessage: {
 margin: '12px 0 0', fontSize: '13px', lineHeight: 1.55, color: colors.text },
  unavailableText: { margin: '12px 0 0', fontSize: '13px', color: colors.warning, lineHeight: 1.5 },
};

export default TravelPlanModal;
