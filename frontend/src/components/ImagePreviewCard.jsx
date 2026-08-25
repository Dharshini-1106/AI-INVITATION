import React from 'react';
import colors from '../theme/colors';

function ImagePreviewCard({ dataUrl, fileName, onRemove, onConfirm }) {
  return (
    <div style={styles.container}>
      <div style={styles.imageWrap}>
        <img src={dataUrl} alt="Invitation preview" style={styles.image} />
      </div>
      <div style={styles.info}>
        <p style={styles.fileName}>{fileName}</p>
        {onRemove && (
          <button style={styles.removeBtn} onClick={onRemove}>
            ✕ Remove
          </button>
        )}
      </div>
      {onConfirm && (
        <button style={styles.confirmBtn} onClick={onConfirm}>
          Analyze Invitation →
        </button>
      )}
    </div>
  );
}

const styles = {
  container: {
    background: colors.card,
    border: `1px solid ${colors.border}`,
    borderRadius: '16px',
    padding: '16px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
  },
  imageWrap: {
    width: '100%',
    maxHeight: '320px',
    overflow: 'hidden',
    borderRadius: '12px',
    backgroundColor: '#000',
  },
  image: {
    width: '100%',
    height: 'auto',
    maxHeight: '320px',
    objectFit: 'contain',
    display: 'block',
  },
  info: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '12px',
  },
  fileName: {
    color: colors.text,
    fontSize: '14px',
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  removeBtn: {
    background: 'transparent',
    color: colors.error,
    border: 'none',
    cursor: 'pointer',
    fontSize: '13px',
    fontWeight: 500,
  },
  confirmBtn: {
    width: '100%',
    marginTop: '12px',
    background: colors.gradient,
    color: '#fff',
    border: 'none',
    borderRadius: '12px',
    padding: '14px',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
};

export default ImagePreviewCard;
