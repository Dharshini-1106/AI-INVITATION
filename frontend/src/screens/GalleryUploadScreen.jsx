import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import colors from '../theme/colors';
import ImagePreviewCard from '../components/ImagePreviewCard';
import ErrorBanner from '../components/ErrorBanner';
import { prepareFile, isValidImageFile } from '../services/imageService';

function GalleryUploadScreen() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');

  const handleFile = async (file) => {
    if (!file) return;
    if (!isValidImageFile(file)) {
      setError('Please select a valid image file (JPG, PNG, BMP, WEBP, TIFF, PDF).');
      return;
    }
    setError('');
    try {
      const prepared = await prepareFile(file);
      setPreview(prepared);
    } catch (e) {
      setError('Failed to read the selected file.');
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const goToProcessing = () => {
    if (!preview) return;
    navigate('/processing', { state: { file: preview.file, previewUrl: preview.dataUrl } });
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <button style={styles.backBtn} onClick={() => navigate('/home')}>←</button>
        <h1 style={styles.title}>Upload from Gallery</h1>
        <span style={styles.spacer} />
      </header>

      <ErrorBanner message={error} />

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.pdf"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {!preview ? (
        <div
          style={styles.dropzone}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          <div style={styles.dropIcon}>🖼️</div>
          <h3 style={styles.dropTitle}>Tap to browse</h3>
          <p style={styles.dropSub}>or drag & drop an invitation image here</p>
          <button style={styles.browseBtn} onClick={() => fileInputRef.current?.click()}>
            Choose Image
          </button>
        </div>
      ) : (
        <>
          <ImagePreviewCard
            dataUrl={preview.dataUrl}
            fileName={preview.name}
            onRemove={() => setPreview(null)}
            onConfirm={goToProcessing}
          />
        </>
      )}

      <div style={styles.hints}>
        <p style={styles.hintsTitle}>Supports</p>
        <div style={styles.hintTags}>
          {['Wedding', 'Reception', 'Birthday', 'Engagement'].map((tag) => (
            <span key={tag} style={styles.hintTag}>{tag}</span>
          ))}
        </div>
      </div>
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
  dropzone: {
    border: `2px dashed ${colors.border}`,
    borderRadius: '20px',
    padding: '48px 24px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'border-color 0.2s, background 0.2s',
    background: colors.card,
  },
  dropIcon: { fontSize: '52px', marginBottom: '12px' },
  dropTitle: { fontSize: '18px', fontWeight: 600, color: colors.text, marginBottom: '6px' },
  dropSub: { fontSize: '14px', color: colors.textMuted, marginBottom: '20px' },
  browseBtn: {
    background: colors.gradient,
    color: '#fff',
    border: 'none',
    borderRadius: '12px',
    padding: '12px 24px',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  hints: { marginTop: '28px' },
  hintsTitle: { fontSize: '13px', color: colors.textMuted, marginBottom: '10px' },
  hintTags: { display: 'flex', flexWrap: 'wrap', gap: '8px' },
  hintTag: {
    background: colors.cardLight,
    color: colors.text,
    padding: '8px 14px',
    borderRadius: '20px',
    fontSize: '13px',
  },
};

export default GalleryUploadScreen;
