import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import colors from '../theme/colors';
import { checkHealth } from '../services/api';
import { logout } from '../services/api';
import { useAuth } from '../auth/AuthContext';

function HomeScreen() {
  const navigate = useNavigate();
  const [backendStatus, setBackendStatus] = useState('checking'); // checking | online | offline
  const { user, setUser } = useAuth();
  const handleLogout = async () => { try { await logout(); } finally { setUser(null); navigate('/login', { replace: true }); } };

  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.brand}>
          <span style={styles.brandIcon}>📇</span>
          <span style={styles.brandName}>InvitationSense</span>
        </div>
        <span
          style={{
            ...styles.statusPill,
            background: backendStatus === 'online' ? 'rgba(0,184,148,0.15)' : 'rgba(255,118,117,0.15)',
            color: backendStatus === 'online' ? colors.success : colors.error,
          }}
        >
          {backendStatus === 'checking' ? '...' : backendStatus === 'online' ? '● Backend Online' : '● Backend Offline'}
        </span>
        <button style={styles.logout} onClick={handleLogout}>Log out</button>
      </header>

      <main style={styles.main}>
        <div style={styles.hero}>
          <h1 style={styles.heroTitle}>Understand Your Invitation</h1>
          <p style={styles.heroSub}>
            Upload or scan any invitation card — wedding, reception, birthday, or
            engagement — and let AI extract all the important details for you.
          </p>
        </div>

        <div style={styles.actions}>
          <button style={styles.actionCard} onClick={() => navigate('/upload')}>
            <span style={styles.actionIcon}>🖼️</span>
            <div>
              <h3 style={styles.actionTitle}>Upload from Gallery</h3>
              <p style={styles.actionDesc}>Choose an invitation image from your device</p>
            </div>
          </button>

          <button style={styles.actionCard} onClick={() => navigate('/scan')}>
            <span style={styles.actionIcon}>📷</span>
            <div>
              <h3 style={styles.actionTitle}>Scan with Camera</h3>
              <p style={styles.actionDesc}>Capture an invitation card in real time</p>
            </div>
          </button>
        </div>

        <div style={styles.features}>
          <div style={styles.feature}>
            <span style={styles.featureIcon}>🌍</span>
            <span style={styles.featureText}>Multilingual</span>
          </div>
          <div style={styles.feature}>
            <span style={styles.featureIcon}>🎨</span>
            <span style={styles.featureText}>Decorative Fonts</span>
          </div>
          <div style={styles.feature}>
            <span style={styles.featureIcon}>✨</span>
            <span style={styles.featureText}>Auto-Enhance</span>
          </div>
          <div style={styles.feature}>
            <span style={styles.featureIcon}>🧠</span>
            <span style={styles.featureText}>AI Powered</span>
          </div>
        </div>
      </main>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    maxWidth: '520px',
    margin: '0 auto',
    padding: '20px 24px 40px',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 0 20px',
  },
  brand: { display: 'flex', alignItems: 'center', gap: '10px' },
  brandIcon: { fontSize: '26px' },
  brandName: { fontWeight: 700, fontSize: '18px', color: colors.text },
  statusPill: {
    fontSize: '12px',
    fontWeight: 600,
    padding: '6px 12px',
    borderRadius: '20px',
  },
  logout: { background: 'transparent', color: colors.text, border: `1px solid ${colors.border}`, borderRadius: '10px', padding: '8px 12px', cursor: 'pointer' },
  main: { flex: 1 },
  hero: { marginTop: '24px', textAlign: 'center' },
  heroTitle: {
    fontSize: '30px',
    fontWeight: 700,
    color: colors.text,
    lineHeight: 1.2,
    marginBottom: '12px',
  },
  heroSub: {
    color: colors.textMuted,
    fontSize: '15px',
    lineHeight: 1.6,
    maxWidth: '420px',
    margin: '0 auto',
  },
  actions: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    marginTop: '36px',
  },
  actionCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '20px',
    background: colors.card,
    border: `1px solid ${colors.border}`,
    borderRadius: '16px',
    cursor: 'pointer',
    transition: 'transform 0.2s, border-color 0.2s',
    textAlign: 'left',
    color: colors.text,
  },
  actionIcon: { fontSize: '34px' },
  actionTitle: { fontSize: '17px', fontWeight: 600, marginBottom: '4px' },
  actionDesc: { fontSize: '13px', color: colors.textMuted },
  features: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
    marginTop: '36px',
  },
  feature: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '14px',
    background: colors.cardLight,
    borderRadius: '12px',
  },
  featureIcon: { fontSize: '20px' },
  featureText: { fontSize: '13px', fontWeight: 500, color: colors.text },
};

export default HomeScreen;
