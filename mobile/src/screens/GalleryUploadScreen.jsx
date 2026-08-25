import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
} from 'react-native';
import { pickImageFromLibrary } from '../services/imageService';
import colors from '../theme/colors';
import ImagePreviewCard from '../components/ImagePreviewCard';
import ErrorBanner from '../components/ErrorBanner';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: 24, paddingBottom: 40 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  backBtn: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    color: colors.text,
    width: 40,
    height: 40,
    borderRadius: 12,
    fontSize: 20,
    textAlign: 'center',
    lineHeight: 38,
  },
  title: { fontSize: 20, fontWeight: '700', color: colors.text, flex: 1, textAlign: 'center' },
  spacer: { width: 40 },
  dropzone: {
    borderWidth: 2,
    borderColor: colors.border,
    borderStyle: 'dashed',
    borderRadius: 20,
    padding: 48,
    alignItems: 'center',
    backgroundColor: colors.card,
  },
  dropIcon: { fontSize: 52, marginBottom: 12 },
  dropTitle: { fontSize: 18, fontWeight: '600', color: colors.text, marginBottom: 6 },
  dropSub: { fontSize: 14, color: colors.textMuted, marginBottom: 20, textAlign: 'center' },
  browseBtn: {
    backgroundColor: colors.primary,
    color: '#fff',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
    fontSize: 15,
    fontWeight: '600',
    overflow: 'hidden',
  },
  hints: { marginTop: 28 },
  hintsTitle: { fontSize: 13, color: colors.textMuted, marginBottom: 10 },
  hintTags: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  hintTag: {
    backgroundColor: colors.cardLight,
    color: colors.text,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    fontSize: 13,
    marginRight: 8,
    marginBottom: 8,
  },
});

export default function GalleryUploadScreen({ navigation }) {
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');

  const pickImage = async () => {
    try {
      const image = await pickImageFromLibrary();
      if (!image) return; // cancelled
      console.log('[MOBILE] Image selected', image.uri);
      setError('');
      setPreview(image);
    } catch (e) {
      setError(e.message || 'Error picking image.');
    }
  };

  const goToProcessing = () => {
    if (!preview) return;
    navigation.navigate('Processing', { image: preview });
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backBtn}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Upload from Gallery</Text>
          <View style={styles.spacer} />
        </View>

        <ErrorBanner message={error} />

        {!preview ? (
          <TouchableOpacity style={styles.dropzone} onPress={pickImage}>
            <Text style={styles.dropIcon}>🖼️</Text>
            <Text style={styles.dropTitle}>Tap to browse</Text>
            <Text style={styles.dropSub}>Select an invitation image from your device</Text>
            <TouchableOpacity onPress={pickImage} style={styles.browseBtn}>
              <Text style={styles.browseBtn}>Choose Image</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ) : (
          <ImagePreviewCard
            dataUrl={preview.uri}
            fileName={preview.fileName}
            onRemove={() => setPreview(null)}
            onConfirm={goToProcessing}
          />
        )}

        <View style={styles.hints}>
          <Text style={styles.hintsTitle}>Supports</Text>
          <View style={styles.hintTags}>
            {['Wedding', 'Reception', 'Birthday', 'Engagement'].map((tag) => (
              <Text key={tag} style={styles.hintTag}>{tag}</Text>
            ))}
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}
