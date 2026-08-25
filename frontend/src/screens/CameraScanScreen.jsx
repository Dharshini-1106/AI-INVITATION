import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import colors from '../theme/colors';
import ImagePreviewCard from '../components/ImagePreviewCard';
import ErrorBanner from '../components/ErrorBanner';
import { prepareFile } from '../services/imageService';

function CameraScanScreen() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');

  const handleCapture = async (file) => {
    if (!file) return;
    setError('');
    try {
      const prepared = await prepareFile(file);
      setPreview(prepared);
    } catch (e) {
      setError('Failed to read the captured image.');
    }
  };

  const goToProcessing = () => {
    if (!preview) return;
    navigate('/processing', { state: { file: preview.file, previewUrl: preview.dataUrl } });
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/home')}>←</button>
        <h1 style={styles.title}>Scan with Camera</h1>
        <span style={styles.spacer} />
      </header>

      <ErrorBanner message={error} />

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
        onChange={(e) => handleCapture(e.target.files?.[0])}
      />

      {!preview ? (
        <div style={styles.cameraArea}>
          <div style={styles.cameraFrame}>
            <div style={styles.cameraCornerTL} />
            <div style={styles.cameraCornerTR} />
            <div style={styles.cameraCornerBL} />
            <div style={styles.cameraCornerBR} />
            <div style={styles.cameraIcon}>📷</div>
            <p style={styles.cameraText}>Position the invitation card within the frame</p>
          </div>
          <button style={styles.captureBtn} onClick={() => fileInputRef.current?.click()}>
            <span style={styles.captureInner} />
          </button>
          <p style={styles.captureHint}>Tap to capture</p>
          <div style={styles.tips}>
            <span style={styles.tip}>Good lighting</span>
            <span style={styles.tip}>Flat card</span>
            <span style={styles.tip}>Readable text</span>
          </div>
        </div>
      ) : (
        <>
          <ImagePreviewCard
            dataUrl={preview.dataUrl}
            fileName={preview.name}
            onRemove={() => setPreview(null)}
            onConfirm={goToProcessing}
          />
          <button style={styles.recaptureBtn} onClick={() => setPreview(null)}>
            ↻ Recapture
          </button>
        </>
      )}
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
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '24px',
  },
  backBtn: {
    background: colors.card,
    border: `1px solid ${colors.border}`,
    color: colors.text,
    width: '40px',
    height: '40px',
    borderRadius: '12px',
    fontSize: '20px',
    cursor: 'pointer',
  },
  title: { fontSize: '20px', fontWeight: 700, color: colors.text },
  spacer: { width: '40px' },
  cameraArea: { display: 'flex', flexDirection: 'column', alignItems: 'center' },
  cameraFrame: {
    position: 'relative',
    width: '100%',
    aspectRatio: '1',
    borderRadius: '20px',
    background: 'linear-gradient(180deg, #1a1940 0%, #0F0E17 100%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  cameraCornerTL: {
    position: 'absolute', top: '20px', left: '20px',
    width: '40px', height: '40px', borderTop: `3px solid ${colors.secondary}`, borderLeft: `3px solid ${colors.secondary}`,
  },
  cameraCornerTR: {
    position: 'absolute', top: '20px', right: '20px',
    width: '40px', height: '40px', borderTop: `3px solid ${colors.secondary}`, borderRight: `3px solid ${colors.secondary}`,
  },
  cameraCornerBL: {
    position: 'absolute', bottom: '20px', left: '20px',
    width: '40px', height: '40px', borderBottom: `3px solid ${colors.secondary}`, borderLeft: `3px solid ${colors.secondary}`,
  },
  cameraCornerBR: {
    position: 'absolute', bottom: '20px', right: '20px',
    width: '40px', height: '40px', borderBottom: `3px solid ${colors.secondary}`, borderRight: `3px solid ${colors.secondary}`,
  },
  cameraIcon: { fontSize: '56px', marginBottom: '12px' },
  cameraText: { color: colors.textMuted, fontSize: '14px', maxWidth: '220px', textAlign: 'center' },
  captureBtn: {
    marginTop: '24px',
    width: '72px',
    height: '72px',
    borderRadius: '50%',
    background: 'transparent',
    border: `4px solid ${colors.primary}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
  },
  captureInner: {
    width: '52px',
    height: '52px',
    borderRadius: '50%',
    background: colors.primary,
  },
  captureHint: { color: colors.textMuted, fontSize: '13px', marginTop: '12px' },
  tips: { display: 'flex', gap: '8px', marginTop: '20px', flexWrap: 'wrap', justifyContent: 'center' },
  tip: {
    background: colors.cardLight,
    color: colors.text,
    padding: '8px 14px',
    borderRadius: '20px',
    fontSize: '12px',
  },
  recaptureBtn: {
    marginTop: '16px',
    width: '100%',
    background: 'transparent',
    color: colors.textMuted,
    border: `1px solid ${colors.border}`,
    borderRadius: '12px',
    padding: '14px',
    fontSize: '15px',
    cursor: 'pointer',
  },
};

export default CameraScanScreen;
