import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import colors from '../theme/colors';

const FIELD_ICONS = {
  event_name: '🎉',
  event_type: '🏷️',
  bride_name: '👰',
  groom_name: '🤵',
  date: '📅',
  time: '⏰',
  venue: '📍',
  address: '🏠',
  contact_number: '📞',
  language: '🌍',
};

function ResultScreen() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};
  const [view, setView] = useState('summary'); // summary | events | raw

  if (!result) {
    return (
      <div style={{ ...styles.container, textAlign: 'center', paddingTop: '80px' }}>
        <p style={{ color: colors.textMuted }}>No result found.</p>
        <button style={styles.homeBtn} onClick={() => navigate('/home')}>Go Home</button>
      </div>
    );
  }

  const primary = result.primaryEvent;
  const confidence = result.confidencePercent;

  const fields = [
    { key: 'event_name', label: 'Event Name', value: primary.event_name },
    { key: 'event_type', label: 'Event Type', value: primary.event_type },
    { key: 'bride_name', label: 'Bride Name', value: primary.bride_name },
    { key: 'groom_name', label: 'Groom Name', value: primary.groom_name },
    { key: 'date', label: 'Date', value: primary.date },
    { key: 'day', label: 'Day', value: primary.day || primary.printed_weekday },
    { key: 'time', label: 'Time', value: primary.time },
    { key: 'end_time', label: 'End Time', value: primary.end_time },
    { key: 'venue', label: 'Venue', value: primary.venue },
    { key: 'address', label: 'Address', value: primary.address },
    { key: 'contact_number', label: 'Contact', value: primary.contact_number },
    { key: 'language', label: 'Language', value: result.language },
  ];

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
          <span style={styles.statValue}>{result.number_of_events}</span>
        </div>
        <div style={styles.statCard}>
          <span style={styles.statLabel}>Language</span>
          <span style={styles.statValue}>{result.language || '—'}</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          style={{ ...styles.tab, ...(view === 'summary' ? styles.tabActive : {}) }}
          onClick={() => setView('summary')}
        >
          Summary
        </button>
        {result.events.length > 1 && (
          <button
            style={{ ...styles.tab, ...(view === 'events' ? styles.tabActive : {}) }}
            onClick={() => setView('events')}
          >
            Events ({result.events.length})
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
        <div style={styles.fieldList}>
          {fields.map((f) => (
            <div key={f.key} style={styles.fieldCard}>
              <span style={styles.fieldIcon}>{FIELD_ICONS[f.key] || '•'}</span>
              <div style={styles.fieldBody}>
                <span style={styles.fieldLabel}>{f.label}</span>
                <span style={styles.fieldValue}>{f.value || '—'}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {view === 'events' && (
        <div style={styles.eventList}>
          {result.events.map((ev, idx) => (
            <div key={idx} style={styles.eventCard}>
              <h3 style={styles.eventTitle}>{ev.event_name || `Event ${idx + 1}`}</h3>
              <div style={styles.eventRow}><span>Type</span><span>{ev.event_type || '—'}</span></div>
              <div style={styles.eventRow}><span>Bride</span><span>{ev.bride_name || '—'}</span></div>
              <div style={styles.eventRow}><span>Groom</span><span>{ev.groom_name || '—'}</span></div>
              <div style={styles.eventRow}><span>Date</span><span>{ev.date || '—'}</span></div>
              <div style={styles.eventRow}><span>Day</span><span>{ev.day || ev.printed_weekday || '—'}</span></div>
              <div style={styles.eventRow}><span>Time</span><span>{ev.time || '—'}</span></div>
              <div style={styles.eventRow}><span>End Time</span><span>{ev.end_time || '—'}</span></div>
              <div style={styles.eventRow}><span>Venue</span><span>{ev.venue || '—'}</span></div>
              <div style={styles.eventRow}><span>Address</span><span>{ev.address || '—'}</span></div>
              <div style={styles.eventRow}><span>Contact</span><span>{ev.contact_number || '—'}</span></div>
            </div>
          ))}
        </div>
      )}

      {view === 'raw' && (
        <div style={styles.rawCard}>
          <pre style={styles.rawText}>{result.raw_text || 'No raw text extracted.'}</pre>
          {result.processing_notes?.length > 0 && (
            <div style={styles.notes}>
              <p style={styles.notesTitle}>Processing Notes</p>
              {result.processing_notes.map((n, i) => (
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
  tabs: { display: 'flex', gap: '8px', marginBottom: '16px' },
  tab: {
    flex: 1, padding: '10px', borderRadius: '10px', background: 'transparent',
    border: `1px solid ${colors.border}`, color: colors.textMuted, fontSize: '14px', fontWeight: 600, cursor: 'pointer',
  },
  tabActive: { background: colors.card, color: colors.text, borderColor: colors.primary },
  fieldList: { display: 'flex', flexDirection: 'column', gap: '10px' },
  fieldCard: {
    display: 'flex', alignItems: 'center', gap: '12px', padding: '14px',
    background: colors.card, borderRadius: '12px', border: `1px solid ${colors.border}`,
  },
  fieldIcon: { fontSize: '22px' },
  fieldBody: { display: 'flex', flexDirection: 'column', gap: '2px', flex: 1 },
  fieldLabel: { fontSize: '12px', color: colors.textMuted },
  fieldValue: { fontSize: '15px', fontWeight: 600, color: colors.text },
  eventList: { display: 'flex', flexDirection: 'column', gap: '14px' },
  eventCard: {
    background: colors.card, borderRadius: '14px', padding: '18px', border: `1px solid ${colors.border}`,
  },
  eventTitle: { fontSize: '17px', fontWeight: 700, color: colors.secondary, marginBottom: '12px' },
  eventRow: {
    display: 'flex', justifyContent: 'space-between', padding: '6px 0',
    borderBottom: `1px solid ${colors.border}`, fontSize: '13px',
  },
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
