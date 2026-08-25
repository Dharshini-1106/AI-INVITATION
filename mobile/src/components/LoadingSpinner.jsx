import React from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import colors from '../theme/colors';

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  label: {
    color: colors.textMuted,
    fontSize: 14,
    fontWeight: '500',
    marginTop: 16,
    textAlign: 'center',
  },
});

export default function LoadingSpinner({ size = 'large', label = 'Processing...' }) {
  return (
    <View style={styles.container}>
      <ActivityIndicator size={size} color={colors.secondary} />
      {label ? <Text style={styles.label}>{label}</Text> : null}
    </View>
  );
}
