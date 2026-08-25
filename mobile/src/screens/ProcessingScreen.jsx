import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import colors from '../theme/colors';
import { analyzeInvitation, getPipelineStages, rediscoverBackend, getResolvedBaseUrl } from '../services/api';
import API_CONFIG from '../config/apiConfig';
import ErrorBanner from '../components/ErrorBanner';

const DEFAULT_STAGES = [
  'Analyzing image quality (BRISQUE)',
  'Enhancing image',
  'Detecting invitation layout',
  'Performing multilingual OCR',
  'Correcting OCR mistakes',
  'Understanding invitation context',
  'Structuring extracted information',
];

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 24 },
  header: { alignItems: 'center', marginBottom: 20 },
  title: { fontSize: 22, fontWeight: '700', color: colors.text },
  previewWrap: {
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: colors.border,
  },
  preview: { width: '100%', height: 180 },
  progressArea: { alignItems: 'center' },
  percentText: { fontSize: 40, fontWeight: '700', color: colors.secondary, marginBottom: 16 },
  stageList: { marginTop: 8, width: '100%' },
  stageRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 14 },
  stageDot: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  stageDotInner: { color: '#fff', fontSize: 12 },
  stageText: { fontSize: 14, fontWeight: '500', flex: 1 },
});

export default function ProcessingScreen({ navigation, route }) {
  const { image } = route.params || {};
  const isCameraImage = image?.source === 'camera';
  const [stages, setStages] = useState(DEFAULT_STAGES);
  const [activeStage, setActiveStage] = useState(0);
  const [error, setError] = useState('');
  const [percent, setPercent] = useState(0);
  const stageRef = useRef(DEFAULT_STAGES);

  useEffect(() => {
    console.log('[MOBILE] App started');
    if (!image) {
      navigation.replace('Home');
      return;
    }

    let cancelled = false;

    // Fetch pipeline stages (fallback to defaults)
    getPipelineStages()
      .then((s) => {
        if (s && s.length && !cancelled) {
          stageRef.current = s;
          setStages(s);
        }
      })
      .catch(() => {});

    const stageTimer = setInterval(() => {
      if (cancelled) return;
      setActiveStage((prev) => {
        const total = stageRef.current.length;
        if (prev < total - 1) {
          setPercent(Math.round(((prev + 1) / total) * 100));
          return prev + 1;
        }
        return prev;
      });
    }, 900);

    const runAnalysis = async () => {
      try {
        if (isCameraImage) console.log('[CAMERA] Upload started');
        console.log('[MOBILE] Uploading image');
        const result = await analyzeInvitation(image);
        if (cancelled) return;
        clearInterval(stageTimer);
        setActiveStage(stageRef.current.length - 1);
        setPercent(100);
        if (isCameraImage) {
          console.log('[CAMERA] Upload completed');
          console.log('[CAMERA] Result received');
        }
        console.log('[MOBILE] Extraction completed');
        setTimeout(() => navigation.replace('Result', { result }), 600);
      } catch (e) {
        if (cancelled) return;
        clearInterval(stageTimer);

// Determine whether this is a network-level failure (backend unreachable)
        // vs. a server response. Axios sets e.response only when the server
        // actually replied; on a connection failure e.response is undefined and
        // e.message is the generic "Network Error".
        const isNetworkError =
          !e.response ||
          e.code === 'ERR_NETWORK' ||
          e.code === 'ECONNABORTED' ||
          e.code === 'ETIMEDOUT' ||
          e.code === 'ECONNREFUSED';

        if (isNetworkError) {
          const backendAddr = API_CONFIG.baseURL.replace(/\/api\/v1$/, '');
          setError(
            `Cannot reach the backend at ${backendAddr}. ` +
            'Make sure the backend is running on your PC ' +
            '(cd backend && python run.py) and the phone/PC are on the ' +
            'same network, or run: adb reverse tcp:8000 tcp:8000. ' +
            'See apiConfig.js for details.'
          );
        } else {
          const serverDetail = e.response?.data?.detail || e.message || '';
          setError(
            serverDetail
              ? `Server error: ${serverDetail} (${e.response.status})`
              : `Request failed (${e.response.status}). Check the backend and retry.`
          );
        }
      }
    };

    runAnalysis();

    return () => {
      cancelled = true;
      clearInterval(stageTimer);
    };
  }, [image, isCameraImage, navigation]);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.title}>Analyzing Invitation</Text>
        </View>

        {image && image.uri ? (
          <View style={styles.previewWrap}>
            <Image source={{ uri: image.uri }} style={styles.preview} resizeMode="cover" />
          </View>
        ) : null}

<ErrorBanner
          message={error}
          onRetry={() => {
            setError('');
            setActiveStage(0);
            setPercent(0);
            // Re-run backend discovery before retrying so the app can pick up
            // the backend address once the phone/PC connection is restored.
            rediscoverBackend()
              .then(() => navigation.replace('Processing', { image }))
              .catch(() => navigation.replace('Processing', { image }));
          }}
        />

        {!error && (
          <View style={styles.progressArea}>
            <Text style={styles.percentText}>{percent}%</Text>
            <View style={styles.stageList}>
              {stages.map((stage, idx) => {
                const bg =
                  idx < activeStage
                    ? colors.success
                    : idx === activeStage
                    ? colors.secondary
                    : colors.cardLight;
                const color = idx <= activeStage ? colors.text : colors.textMuted;
                return (
                  <View key={idx} style={styles.stageRow}>
                    <View style={[styles.stageDot, { backgroundColor: bg }]}>
                      {idx < activeStage ? (
                        <Text style={styles.stageDotInner}>✓</Text>
                      ) : idx === activeStage ? (
                        <Text style={styles.stageDotInner}>●</Text>
                      ) : null}
                    </View>
                    <Text style={[styles.stageText, { color }]}>{stage}</Text>
                  </View>
                );
              })}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
