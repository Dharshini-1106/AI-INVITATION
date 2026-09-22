import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import colors from '../theme/colors';
import { loadAnalysisResult, normalizeResultPayload } from '../utils/resultState';
import TravelPlanModal from '../components/TravelPlanModal';

const FIELD_ICONS = {
  event_name: '🎉',
  event_type: '🏷️',
  bride_name: '👰',
  groom_name: '🤵',
  date: '📅',
  time: '⏰',
  end_time: '🏁',
  venue: '📍',
  address: '🏠',
  contact_number: '📞',
  language: '🌍',
};

function PersonCard({ person }) {
  return (
    <div style={styles.personCard}>
      <span style={styles.personRole}>{person.role || 'Person'}</span>
      <span style={styles.personName}>{person.name || 'Not available'}</span>
    </div>
  );
}

function EventCard({ event, index }) {
  const fields = [
    { key: 'event_name', label: 'Event Name' },
    { key: 'event_type', label: 'Event Type' },
    { key: 'date', label: 'Date' },
    { key: 'printed_weekday', label: 'Day' },
    { key: 'time', label: 'Time' },
    { key: 'end_time', label: 'End Time' },
    { key: 'venue', label: 'Venue' },
    { key: 'address', label: 'Address' },
    { key: 'contact_number', label: 'Contact' },
  ];

  return (
    <div style={styles.eventCard}>
      <h3 style={styles.eventTitle}>
        {event.event_name ? `Event ${index + 1} - ${event.event_name}` : `Event ${index + 1}`}
      </h3>
      <div style={styles.fieldList}>
        {fields.map((f) => (
          <div key={f.key} style={styles.fieldRow}>
            <span style={styles.fieldLabel}>{f.label}</span>
            <span style={styles.fieldValue}>
              {event[f.key] || 'Not available'}
            </span>
          </div>
        ))}
      </div>
      <TravelPlanModal event={event} />
    </div>
  );
}

function ResultScreen() {
  const location = useLocation();
  const navigate = useNavigate();
  const [view, setView] = useState('summary');
  const [resultState, setResultState] = useState(() => {
    const fallback = loadAnalysisResult();
    if (fallback) {
      return fallback;
    }
    return location.state || {};
  });

  useEffect(() => {
    const fallback = loadAnalysisResult();
    if (fallback) {
      setResultState(fallback);
      return;
    }

    if (location.state?.result) {
      setResultState(location.state);
    }
  }, [location.state]);

  const { result, previewUrl } = resultState || {};
  const normalizedResult = normalizeResultPayload(result);

  if (!normalizedResult) {
    return (
      <div style={{ ...styles.container, textAlign: 'center', paddingTop: '80px' }}>
        <p style={{ color: colors.textMuted }}>No result found.</p>
        <button style={styles.homeBtn} onClick={() => navigate('/home')}>Go Home</button>
      </div>
    );
  }

  const confidence = normalizedResult.confidencePercent;
  const isMulti = normalizedResult.invitation_mode === 'multi';
  const events = (normalizedResult.events && normalizedResult.events.length > 0)
    ? normalizedResult.events
    : [normalizedResult.primaryEvent];

  // Build people list: prefer the new generic people array, fall back to
  // legacy bride/groom fields for backward compatibility.
  let people = (normalizedResult.people || []).filter(
    (p) => p && (p.name || p.role)
  );
  if (!people.length) {
    if (normalizedResult.bride_name) {
      people.push({ name: normalizedResult.bride_name, role: 'Bride' });
    }
    if (normalizedResult.groom_name) {
      people.push({ name: normalizedResult.groom_name, role: 'Groom' });
    }
  }

  const hasMultipleEvents = events.length > 1;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/home')}>←</button>
        <h1 style={styles.title}>Extracted Details</h1>
        <span style={styles.spacer} />
      </header>

      {previewUrl && (
        <div style={styles.previewWrap}>
          <img src={previewUrl} alt="Invitation" style={styles.preview} />
        </div>
      )}

      {/* Confidence + event count */}
      <div style={styles.statsRow}>
        <div style={styles.statCard}>
          <span style={styles.statLabel}>Confidence</span>
          <span style={styles.statValue}>{confidence}%</span>
        </div>
        <div style={styles.statCard}>
          <span style={styles.statLabel}>Events</span>
          <span style={styles.statValue}>{normalizedResult.number_of_events}</span>
        </div>
        <div style={styles.statCard}>
          <span style={styles.statLabel}>Language</span>
          <span style={styles.statValue}>{normalizedResult.language || '—'}</span>
        </div>
      </div>

      <div style={styles.ocrCard}>
        <h2 style={styles.sectionTitle}>OCR Details</h2>
        <div style={styles.ocrGrid}>
          <span style={styles.ocrLabel}>Engine</span>
          <span style={styles.ocrValue}>{normalizedResult.ocr_engine || 'Not available'}</span>
          <span style={styles.ocrLabel}>Confidence</span>
          <span style={styles.ocrValue}>
            {normalizedResult.ocr_confidence == null
              ? 'Not available'
              : `${Math.round(normalizedResult.ocr_confidence * 100)}%`}
          </span>
          <span style={styles.ocrLabel}>Tamil / English characters</span>
          <span style={styles.ocrValue}>
            {normalizedResult.tamil_character_count || 0} / {normalizedResult.english_character_count || 0}
          </span>
          <span style={styles.ocrLabel}>Fallback used</span>
          <span style={styles.ocrValue}>{normalizedResult.fallback_used ? 'Yes' : 'No'}</span>
        </div>
      </div>

      {/* People / Participants section */}
      {people.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>
            {isMulti ? 'People / Participants' : 'People / Participants'}
          </h2>
          <div style={styles.peopleList}>
            {people.map((p, idx) => (
              <PersonCard key={idx} person={p} />
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          style={{ ...styles.tab, ...(view === 'summary' ? styles.tabActive : {}) }}
          onClick={() => setView('summary')}
        >
          Summary
        </button>
        {hasMultipleEvents && (
          <button
            style={{ ...styles.tab, ...(view === 'events' ? styles.tabActive : {}) }}
            onClick={() => setView('events')}
          >
            Events ({events.length})
          </button>
        )}
        <button
          style={{ ...styles.tab, ...(view === 'raw' ? styles.tabActive : {}) }}
          onClick={() => setView('raw')}
        >
          Raw Text
        </button>
      </div>

      {/* Content */}
      {view === 'summary' && (
        <div style={styles.content}>
          {events.map((ev, idx) => (
            <EventCard key={idx} event={ev} index={idx} />
          ))}
        </div>
      )}

      {view === 'events' && hasMultipleEvents && (
        <div style={styles.content}>
          {events.map((ev, idx) => (
            <EventCard key={idx} event={ev} index={idx} />
          ))}
        </div>
      )}

      {view === 'raw' && (
        <div style={styles.rawCard}>
          <pre style={styles.rawText}>{normalizedResult.raw_text || 'No raw text extracted.'}</pre>
          {normalizedResult.processing_notes?.length > 0 && (
            <div style={styles.notes}>
              <p style={styles.notesTitle}>Processing Notes</p>
              {normalizedResult.processing_notes.map((n, i) => (
                <p key={i} style={styles.note}>• {n}</p>
              ))}
            </div>
          )}
        </div>
      )}

      <button style={styles.newBtn} onClick={() => navigate('/home')}>
        Analyze Another Invitation
      </button>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    maxWidth: '520px',
    margin: '0 auto',
    padding: '20px 24px 40px',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px',
  },
  backBtn: {
    background: colors.card, border: `1px solid ${colors.border}`, color: colors.text,
    width: '40px', height: '40px', borderRadius: '12px', fontSize: '20px', cursor: 'pointer',
  },
  title: { fontSize: '20px', fontWeight: 700, color: colors.text },
  spacer: { width: '40px' },
  previewWrap: { borderRadius: '16px', overflow: 'hidden', marginBottom: '16px', border: `1px solid ${colors.border}` },
  preview: { width: '100%', maxHeight: '160px', objectFit: 'cover', display: 'block' },
  statsRow: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginBottom: '20px' },
  statCard: {
    background: colors.card, border: `1px solid ${colors.border}`, borderRadius: '14px',
    padding: '14px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px',
  },
  statLabel: { fontSize: '12px', color: colors.textMuted },
  statValue: { fontSize: '18px', fontWeight: 700, color: colors.secondary },
  ocrCard: { background: colors.card, border: `1px solid ${colors.border}`, borderRadius: '14px', padding: '14px 16px', marginBottom: '20px' },
  ocrGrid: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '8px 16px', fontSize: '13px' },
  ocrLabel: { color: colors.textMuted },
  ocrValue: { color: colors.text, fontWeight: 600, textAlign: 'right', overflowWrap: 'anywhere' },
  section: { marginBottom: '20px' },
  sectionTitle: { fontSize: '16px', fontWeight: 700, color: colors.text, marginBottom: '10px' },
  peopleList: { display: 'flex', flexDirection: 'column', gap: '8px' },
  personCard: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 14px', background: colors.card, borderRadius: '12px',
    border: `1px solid ${colors.border}`,
  },
  personRole: { fontSize: '13px', color: colors.textMuted, fontWeight: 600 },
  personName: { fontSize: '15px', fontWeight: 700, color: colors.text },
  tabs: { display: 'flex', gap: '8px', marginBottom: '16px' },
  tab: {
    flex: 1, padding: '10px', borderRadius: '10px', background: 'transparent',
    border: `1px solid ${colors.border}`, color: colors.textMuted, fontSize: '14px', fontWeight: 600, cursor: 'pointer',
  },
  tabActive: { background: colors.card, color: colors.text, borderColor: colors.primary },
  content: { display: 'flex', flexDirection: 'column', gap: '14px' },
  eventCard: {
    background: colors.card, borderRadius: '14px', padding: '18px', border: `1px solid ${colors.border}`,
  },
  eventTitle: { fontSize: '17px', fontWeight: 700, color: colors.secondary, marginBottom: '12px' },
  fieldList: { display: 'flex', flexDirection: 'column', gap: '8px' },
  fieldRow: {
    display: 'flex', justifyContent: 'space-between', padding: '6px 0',
    borderBottom: `1px solid ${colors.border}`, fontSize: '13px',
  },
  fieldLabel: { color: colors.textMuted },
  fieldValue: { color: colors.text, fontWeight: 600, textAlign: 'right', maxWidth: '60%' },
  rawCard: { background: colors.card, borderRadius: '12px', padding: '16px', border: `1px solid ${colors.border}` },
  rawText: { whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '13px', color: colors.text, lineHeight: 1.6 },
  notes: { marginTop: '16px', borderTop: `1px solid ${colors.border}`, paddingTop: '12px' },
  notesTitle: { fontSize: '13px', fontWeight: 600, color: colors.textMuted, marginBottom: '8px' },
  note: { fontSize: '12px', color: colors.textMuted, marginBottom: '4px' },
  newBtn: {
    width: '100%', marginTop: '24px', background: colors.gradient, color: '#fff',
    border: 'none', borderRadius: '12px', padding: '14px', fontSize: '15px', fontWeight: 600, cursor: 'pointer',
  },
  homeBtn: {
    marginTop: '16px', background: colors.primary, color: '#fff', border: 'none',
    borderRadius: '12px', padding: '12px 24px', fontSize: '15px', cursor: 'pointer',
  },
};

export default ResultScreen;