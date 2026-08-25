import React from 'react';
import colors from '../theme/colors';

function ErrorBanner({ message, onRetry }) {
  if (!message) return null;
  return (
    <div style={styles.container}>
      <div style={styles.icon}>⚠️</div>
      <div style={styles.content}>
        <p style={styles.title}>Something went wrong</p>
        <p style={styles.message}>{message}</p>
      </div>
      {onRetry && (
        <button style={styles.retry} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: 'rgba(255, 118, 117, 0.12)',
    border: `1px solid ${colors.error}`,
    borderRadius: '12px',
    padding: '14px 16px',
    margin: '12px 0',
  },
  icon: { fontSize: '22px' },
  content: { flex: 1 },
  title: {
    color: colors.error,
    fontWeight: 600,
    fontSize: '14px',
    marginBottom: '2px',
  },
  message: {
    color: colors.textMuted,
    fontSize: '13px',
    lineHeight: 1.4,
  },
  retry: {
    background: colors.error,
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    padding: '8px 16px',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '13px',
  },
};

export default ErrorBanner;
