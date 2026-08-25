import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import colors from '../theme/colors';

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 118, 117, 0.12)',
    borderWidth: 1,
    borderColor: colors.error,
    borderRadius: 12,
    padding: 14,
    marginVertical: 12,
  },
  icon: { fontSize: 22, marginRight: 10 },
  content: { flex: 1 },
  title: { color: colors.error, fontWeight: '600', fontSize: 14, marginBottom: 2 },
  message: { color: colors.textMuted, fontSize: 13, lineHeight: 18 },
  retry: {
    backgroundColor: colors.error,
    color: '#fff',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
    fontWeight: '600',
    fontSize: 13,
    marginLeft: 8,
  },
});

export default function ErrorBanner({ message, onRetry }) {
  if (!message) return null;
  return (
    <View style={styles.container}>
      <Text style={styles.icon}>⚠️</Text>
      <View style={styles.content}>
        <Text style={styles.title}>Something went wrong</Text>
        <Text style={styles.message}>{message}</Text>
      </View>
      {onRetry ? (
        <TouchableOpacity onPress={onRetry}>
          <Text style={styles.retry}>Retry</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}
