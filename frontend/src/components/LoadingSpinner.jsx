import React from 'react';
import colors from '../theme/colors';

function LoadingSpinner({ size = 60, label = 'Processing...' }) {
  return (
    <div style={styles.container}>
      <div style={{ width: size, height: size }}>
        <div className="spinner-ring" style={styles.spinner} />
      </div>
      {label && <p style={styles.label}>{label}</p>}
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '16px',
    padding: '24px',
  },
  spinner: {
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '4px solid rgba(255,255,255,0.15)',
    borderTopColor: colors.secondary,
    animation: 'spin 0.9s linear infinite',
  },
  label: {
    color: colors.textMuted,
    fontSize: '14px',
    fontWeight: 500,
    letterSpacing: '0.3px',
  },
};

export default LoadingSpinner;
