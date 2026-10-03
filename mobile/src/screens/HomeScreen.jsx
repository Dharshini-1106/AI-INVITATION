import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import colors from '../theme/colors';
import { checkHealth, rediscoverBackend, getResolvedBaseUrl } from '../services/api';
import API_CONFIG from '../config/apiConfig';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 24, paddingBottom: 40 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  brand: { flexDirection: 'row', alignItems: 'center' },
  brandIcon: { fontSize: 26, marginRight: 8 },
  brandName: { fontWeight: '700', fontSize: 18, color: colors.text },
statusPill: { fontSize: 12, fontWeight: '600', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20 },
  headerActions: { alignItems: 'flex-end', gap: 8 },
  logoutButton: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 9, borderWidth: 1, borderColor: colors.border },
  logoutText: { color: colors.text, fontSize: 12, fontWeight: '600' },
  backendAddr: { fontSize: 11, color: colors.textMuted, textAlign: 'center', marginTop: 8 },
  hero: { marginTop: 24, alignItems: 'center' },
  heroTitle: { fontSize: 30, fontWeight: '700', color: colors.text, textAlign: 'center', lineHeight: 36 },
  heroSub: { color: colors.textMuted, fontSize: 15, lineHeight: 22, textAlign: 'center', marginTop: 12, maxWidth: 420 },
  actions: { marginTop: 32 },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    marginBottom: 16,
  },
  actionIcon: { fontSize: 34, marginRight: 16 },
  actionTitle: { fontSize: 17, fontWeight: '600', color: colors.text },
  actionDesc: { fontSize: 13, color: colors.textMuted, marginTop: 4 },
  features: { flexDirection: 'row', flexWrap: 'wrap', marginTop: 24 },
  feature: {
    width: '48%',
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    backgroundColor: colors.cardLight,
    borderRadius: 12,
    marginBottom: 12,
    marginRight: '4%',
  },
  featureIcon: { fontSize: 20, marginRight: 8 },
  featureText: { fontSize: 13, fontWeight: '500', color: colors.text },
});

export default function HomeScreen({ navigation, onLogout }) {
const [backendStatus, setBackendStatus] = useState('checking');
  const [backendAddr, setBackendAddr] = useState(API_CONFIG.baseURL);

  const checkBackend = () => {
    setBackendStatus('checking');
    return checkHealth()
      .then(() => {
        setBackendStatus('online');
        setBackendAddr(getResolvedBaseUrl() || API_CONFIG.baseURL);
      })
      .catch(() => {
        setBackendStatus('offline');
      });
  };

  const recheckBackend = () => {
    setBackendStatus('checking');
    rediscoverBackend()
      .then(() => checkBackend())
      .catch(() => checkBackend());
  };

  useEffect(() => {
    console.log('[MOBILE] App started');
    checkBackend();
  }, []);

  const statusColor = backendStatus === 'online' ? colors.success : colors.error;
  const statusText =
    backendStatus === 'checking'
      ? 'Checking backend...'
      : backendStatus === 'online'
      ? '● Backend Online'
      : '● Backend Offline';

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <View style={styles.brand}>
            <Text style={styles.brandIcon}>📇</Text>
            <Text style={styles.brandName}>InvitationSense</Text>
          </View>
          <View style={styles.headerActions}>
            <TouchableOpacity
              onPress={recheckBackend}
              style={[styles.statusPill, { backgroundColor: statusColor + '22' }]}
            >
              <Text style={{ color: statusColor }}>{statusText}</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={onLogout} style={styles.logoutButton} accessibilityRole="button">
              <Text style={styles.logoutText}>Log out</Text>
            </TouchableOpacity>
          </View>
        </View>

        {backendStatus === 'online' && (
          <Text style={styles.backendAddr}>API: {backendAddr}</Text>
        )}
        {backendStatus === 'offline' && (
          <Text style={styles.backendAddr}>
            Tap status to retry. Ensure backend is running (cd backend && python run.py)
            and phone/PC are on the same network, or run: adb reverse tcp:8000 tcp:8000
          </Text>
        )}

        <View style={styles.hero}>
          <Text style={styles.heroTitle}>Understand Your Invitation</Text>
          <Text style={styles.heroSub}>
            Upload or scan any invitation card — wedding, reception, birthday, or
            engagement — and let AI extract all the important details.
          </Text>
        </View>

        <View style={styles.actions}>
          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => navigation.navigate('SavedEvents')}
          >
            <Text style={styles.actionIcon}>📅</Text>
            <View>
              <Text style={styles.actionTitle}>My Events</Text>
              <Text style={styles.actionDesc}>Return to saved event details and travel plans</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => navigation.navigate('GalleryUpload')}
          >
            <Text style={styles.actionIcon}>🖼️</Text>
            <View>
              <Text style={styles.actionTitle}>Upload from Gallery</Text>
              <Text style={styles.actionDesc}>Choose an invitation image from your device</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionCard}
            onPress={() => navigation.navigate('CameraScan')}
          >
            <Text style={styles.actionIcon}>📷</Text>
            <View>
              <Text style={styles.actionTitle}>Scan with Camera</Text>
              <Text style={styles.actionDesc}>Capture an invitation card in real time</Text>
            </View>
          </TouchableOpacity>
        </View>

        <View style={styles.features}>
          <View style={styles.feature}><Text style={styles.featureIcon}>🌍</Text><Text style={styles.featureText}>Multilingual</Text></View>
          <View style={styles.feature}><Text style={styles.featureIcon}>🎨</Text><Text style={styles.featureText}>Decorative Fonts</Text></View>
          <View style={styles.feature}><Text style={styles.featureIcon}>✨</Text><Text style={styles.featureText}>Auto-Enhance</Text></View>
          <View style={styles.feature}><Text style={styles.featureIcon}>🧠</Text><Text style={styles.featureText}>AI Powered</Text></View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
