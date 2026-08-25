import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import colors from '../theme/colors';

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  imageWrap: {
    width: '100%',
    maxHeight: 320,
    overflow: 'hidden',
    borderRadius: 12,
    backgroundColor: '#000',
  },
  image: {
    width: '100%',
    height: undefined,
    aspectRatio: 4 / 3,
    resizeMode: 'contain',
  },
  info: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  fileName: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
    marginRight: 8,
  },
  removeBtn: {
    color: colors.error,
    fontSize: 13,
    fontWeight: '500',
  },
  confirmBtn: {
    marginTop: 12,
    backgroundColor: colors.primary,
    color: '#fff',
    textAlign: 'center',
    paddingVertical: 14,
    borderRadius: 12,
    fontSize: 15,
    fontWeight: '600',
    overflow: 'hidden',
  },
});

export default function ImagePreviewCard({ dataUrl, fileName, onRemove, onConfirm }) {
  return (
    <View style={styles.container}>
      <View style={styles.imageWrap}>
        <Image source={{ uri: dataUrl }} style={styles.image} resizeMode="contain" />
      </View>
      <View style={styles.info}>
        <Text style={styles.fileName} numberOfLines={1}>{fileName}</Text>
        {onRemove ? (
          <TouchableOpacity onPress={onRemove}>
            <Text style={styles.removeBtn}>✕ Remove</Text>
          </TouchableOpacity>
        ) : null}
      </View>
      {onConfirm ? (
        <TouchableOpacity onPress={onConfirm}>
          <Text style={styles.confirmBtn}>Analyze Invitation →</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}
