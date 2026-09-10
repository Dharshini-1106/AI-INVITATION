import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import colors from '../theme/colors';
import { analyzeInvitation, getPipelineStages } from '../services/api';
import ErrorBanner from '../components/ErrorBanner';
import { saveAnalysisResult } from '../utils/resultState';

const DEFAULT_STAGES = [
  'Analyzing image quality (BRISQUE)',
  'Enhancing image',
  'Detecting invitation layout',
  'Performing multilingual OCR',
  'Correcting OCR mistakes',
  'Understanding invitation context',
  'Structuring extracted information',
];

function ProcessingScreen() {
  const location = useLocation();
  const navigate = useNavigate();
  const { file, previewUrl } = location.state || {};
  const [stages, setStages] = useState(DEFAULT_STAGES);
  const [activeStage, setActiveStage] = useState(0);
  const [error, setError] = useState('');
  const [percent, setPercent] = useState(0);

  useEffect(() => {
    if (!file) {
      navigate('/home');
      return;
    }

    // Fetch actual pipeline stages from backend (fallback to defaults)
    getPipelineStages().then(setStages).catch(() => {});

    let cancelled = false;
    const runAnalysis = async () => {
      // Simulate stage-by-stage progress while the API processes
      const stageTimer = setInterval(() => {
        if (cancelled) return;
        setActiveStage((prev) => {
          if (prev < stages.length - 1) {
            setPercent(Math.round(((prev + 1) / stages.length) * 100));
            return prev + 1;
          }
          return prev;
        });
      }, 900);

      try {
        const result = await analyzeInvitation(file);
        if (cancelled) return;
        clearInterval(stageTimer);
        setActiveStage(stages.length - 1);
        setPercent(100);
        saveAnalysisResult(result, previewUrl);
        setTimeout(() => navigate('/result', { state: { result, previewUrl } }), 600);
      } catch (e) {
        if (cancelled) return;
        clearInterval(stageTimer);
        if (e.code === 'ECONNABORTED') {
          setError('Analysis timed out. The invitation is taking longer than expected to process. Please try again with a clearer image or wait a bit longer.');
        } else if (!e.response) {
          setError('Cannot reach the backend. Please check that the backend is running on your PC and your phone/PC are on the same network.');
        } else if (e.response.status >= 500) {
          setError('Backend server error. Please try again later.');
        } else if (e.response.status >= 400) {
          setError(e.response?.data?.detail || `Request failed (${e.response.status}). Please try again.`);
        } else {
          setError('Analysis failed. Please check the backend is running and try again.');
        }
      }
    };

    runAnalysis();
    return () => {
      cancelled = true;
    };
  }, [file, navigate]);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Analyzing Invitation</h1>
      </header>

      {previewUrl && (
        <div style={styles.previewWrap}>
          <img src={previewUrl} alt="Processing" style={styles.preview} />
        </div>
      )}

      <ErrorBanner
        message={error}
        onRetry={() => window.location.reload()}
      />

      {!error && (
        <div style={styles.progressArea}>
          <div style={styles.percentWrap}>
            <svg width="120" height="120" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="52" fill="none" stroke={colors.cardLight} strokeWidth="10" />
              <circle
                cx="60" cy="60" r="52" fill="none"
                stroke={colors.secondary} strokeWidth="10" strokeLinecap="round"
                strokeDasharray={`${(percent / 100) * 326.7} 326.7`}
                transform="rotate(-90 60 60)"
                style={{ transition: 'stroke-dasharray 0.5s ease' }}
              />
            </svg>
            <div style={styles.percentText}>
              <span style={styles.percentNum}>{percent}%</span>
            </div>
          </div>

          <div style={styles.stageList}>
            {stages.map((stage, idx) => (
              <div key={idx} style={styles.stageRow}>
                <span
                  style={{
                    ...styles.stageDot,
                    background:
                      idx < activeStage ? colors.success :
                      idx === activeStage ? colors.secondary : colors.cardLight,
                  }}
                >
                  {idx < activeStage ? '✓' : idx === activeStage ? <span className="pulse">●</span> : ''}
                </span>
                <span
                  style={{
                    ...styles.stageText,
                    color: idx <= activeStage ? colors.text : colors.textMuted,
                  }}
                >
                  {stage}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    maxWidth: '520px',
    margin: '0 auto',
    padding: '24px',
  },
  header: { textAlign: 'center', marginBottom: '20px' },
  title: { fontSize: '22px', fontWeight: 700, color: colors.text },
  previewWrap: {
    borderRadius: '16px',
    overflow: 'hidden',
    marginBottom: '20px',
    border: `1px solid ${colors.border}`,
  },
  preview: { width: '100%', maxHeight: '180px', objectFit: 'cover', display: 'block' },
  progressArea: { display: 'flex', flexDirection: 'column', alignItems: 'center' },
  percentWrap: { position: 'relative', width: '120px', height: '120px' },
  percentText: {
    position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  percentNum: { fontSize: '22px', fontWeight: 700, color: colors.text },
  stageList: { marginTop: '24px', width: '100%', display: 'flex', flexDirection: 'column', gap: '14px' },
  stageRow: { display: 'flex', alignItems: 'center', gap: '12px' },
  stageDot: {
    width: '24px', height: '24px', borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: '12px', color: '#fff', flexShrink: 0,
  },
  stageText: { fontSize: '14px', fontWeight: 500 },
};

export default ProcessingScreen;
