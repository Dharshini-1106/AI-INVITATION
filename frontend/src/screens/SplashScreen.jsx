import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import colors from '../theme/colors';

function SplashScreen() {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/home');
    }, 2200);
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div style={styles.container}>
      <div style={styles.logoWrap}>
        <div style={styles.logo}>📇</div>
      </div>
      <h1 style={styles.title}>InvitationSense</h1>
      <p style={styles.subtitle}>AI-Driven Invitation Understanding</p>
      <div style={styles.loadDots}>
        <span style={styles.dot} />
        <span style={{ ...styles.dot, animationDelay: '0.2s' }} />
        <span style={{ ...styles.dot, animationDelay: '0.4s' }} />
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    background: 'radial-gradient(circle at 50% 30%, #1a1940 0%, #0F0E17 70%)',
  },
  logoWrap: {
    width: '96px',
    height: '96px',
    borderRadius: '28px',
    background: colors.gradient,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 12px 40px rgba(108, 92, 231, 0.4)',
    marginBottom: '8px',
  },
  logo: { fontSize: '48px' },
  title: {
    color: colors.text,
    fontSize: '34px',
    fontWeight: 700,
    letterSpacing: '-0.5px',
  },
  subtitle: {
    color: colors.textMuted,
    fontSize: '15px',
    fontWeight: 400,
  },
  loadDots: {
    display: 'flex',
    gap: '8px',
    marginTop: '18px',
  },
  dot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: colors.secondary,
    animation: 'pulse 1.2s ease-in-out infinite',
  },
};

export default SplashScreen;
