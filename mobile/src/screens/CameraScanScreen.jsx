import React, { useRef, useState } from 'react';
import { ActivityIndicator, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import colors from '../theme/colors';
import ImagePreviewCard from '../components/ImagePreviewCard';
import ErrorBanner from '../components/ErrorBanner';

// Typical 12MP captures are already suitable for small invitation text. Only
// constrain unusually large sensor output to prevent excessive upload time.
const CAMERA_RESIZE_THRESHOLD = 5000;
const MAX_CAMERA_EDGE = 4096;

async function imageSize(uri) {
  try {
    const response = await fetch(uri);
    const blob = await response.blob();
    return blob.size;
  } catch {
    return undefined;
  }
}

function cameraLog(label, value) {
  // Camera diagnostics only. Do not include image bytes or invitation content.
  console.log(`[CAMERA] ${label}:`, value);
}

export default function CameraScanScreen({ navigation }) {
  const cameraRef = useRef(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [preview, setPreview] = useState(null);
  const [takingPhoto, setTakingPhoto] = useState(false);
  const [processingCapture, setProcessingCapture] = useState(false);
  const [error, setError] = useState('');
  const capture = async () => {
    if (!cameraRef.current || takingPhoto) return;
    try {
      setTakingPhoto(true);
      setError('');
      cameraLog('Capture started', '');
      // Do not use skipProcessing: Expo then applies device orientation before
      // returning the full captured image rather than a preview frame.
      const photo = await cameraRef.current.takePictureAsync({ quality: 1, exif: true, skipProcessing: false });
      cameraLog('Image captured', '');
      cameraLog('Image URI', photo.uri);
      cameraLog('Image width', photo.width);
      cameraLog('Image height', photo.height);
      cameraLog('Image type', 'image/jpeg');
      cameraLog('Image size', await imageSize(photo.uri));
      cameraLog('EXIF orientation', photo.exif?.Orientation ?? 'not present');

      cameraLog('Preprocessing started', '');
      const longestEdge = Math.max(photo.width, photo.height);
      const resize = longestEdge > CAMERA_RESIZE_THRESHOLD
        ? photo.width >= photo.height
          ? { width: MAX_CAMERA_EDGE }
          : { height: MAX_CAMERA_EDGE }
        : null;
      // Re-encoding the already orientation-processed capture as JPEG makes
      // the uploaded pixels upright without relying on EXIF metadata.
      const processed = await manipulateAsync(
        photo.uri,
        resize ? [{ resize }] : [],
        { compress: 0.95, format: SaveFormat.JPEG },
      );
      cameraLog('Preprocessing completed', '');
      cameraLog('Processed image width', processed.width);
      cameraLog('Processed image height', processed.height);
      cameraLog('Processed image size', await imageSize(processed.uri));
      setPreview({
        uri: processed.uri,
        fileName: 'invitation-capture.jpg',
        type: 'image/jpeg',
        source: 'camera',
      });
    }
    catch (e) { setError(e.message || 'Unable to capture the invitation.'); }
    finally { setTakingPhoto(false); }
  };
  const processCapture = () => {
    if (!preview || processingCapture) return;
    setProcessingCapture(true);
    navigation.navigate('Processing', { image: preview });
  };
  if (!permission) return <SafeAreaView style={styles.container} />;
  if (!permission.granted) return <SafeAreaView style={styles.container}><View style={styles.permission}><Text style={styles.title}>Camera access required</Text><Text style={styles.muted}>Allow camera access to scan an invitation card.</Text><TouchableOpacity style={styles.primary} onPress={requestPermission}><Text style={styles.primaryText}>Allow Camera</Text></TouchableOpacity><TouchableOpacity onPress={() => navigation.goBack()}><Text style={styles.back}>Go Back</Text></TouchableOpacity></View></SafeAreaView>;
  return <SafeAreaView style={styles.container}><View style={styles.content}>
    <View style={styles.header}><TouchableOpacity onPress={() => navigation.goBack()}><Text style={styles.back}>Back</Text></TouchableOpacity><Text style={styles.title}>Scan with Camera</Text><View style={styles.spacer} /></View>
    <ErrorBanner message={error} />
    {preview ? <><ImagePreviewCard dataUrl={preview.uri} fileName={preview.fileName} onRemove={() => { setPreview(null); setProcessingCapture(false); }} onConfirm={processCapture} /><TouchableOpacity style={styles.secondary} onPress={() => { setPreview(null); setProcessingCapture(false); }}><Text style={styles.secondaryText}>Retake Photo</Text></TouchableOpacity></> : <><View style={styles.cameraWrap}><CameraView ref={cameraRef} style={styles.camera} facing="back" /></View><Text style={styles.muted}>Position the invitation clearly in good lighting.</Text><TouchableOpacity style={styles.capture} onPress={capture} disabled={takingPhoto}>{takingPhoto ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryText}>Capture Invitation</Text>}</TouchableOpacity></>}
  </View></SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background }, content: { padding: 20, flex: 1 }, header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }, title: { color: colors.text, fontSize: 20, fontWeight: '700' }, spacer: { width: 45 }, back: { color: colors.primary, fontWeight: '700', paddingVertical: 8 }, cameraWrap: { overflow: 'hidden', borderRadius: 16, aspectRatio: .76, backgroundColor: '#000' }, camera: { flex: 1 }, muted: { color: colors.textMuted, textAlign: 'center', marginTop: 14, lineHeight: 20 }, capture: { backgroundColor: colors.primary, borderRadius: 12, padding: 16, alignItems: 'center', marginTop: 20 }, primaryText: { color: '#fff', fontWeight: '700' }, secondary: { borderColor: colors.border, borderWidth: 1, borderRadius: 12, padding: 15, alignItems: 'center', marginTop: 12 }, secondaryText: { color: colors.text, fontWeight: '700' }, permission: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 28 }, primary: { backgroundColor: colors.primary, borderRadius: 12, padding: 15, alignItems: 'center', alignSelf: 'stretch', marginTop: 24 },
});
