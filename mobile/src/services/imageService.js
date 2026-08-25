import { Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

/**
 * Cross-platform image picking service.
 * - Native (Android/iOS): uses expo-image-picker (camera + gallery).
 * - Web: uses a hidden <input type="file"> fallback for gallery upload.
 *   Camera not available on web browsers.
 */

// Permission helpers (native only)
export async function requestCameraPermission() {
  if (Platform.OS === 'web') return true;
  const { status } = await ImagePicker.requestCameraPermissionsAsync();
  return status === 'granted';
}

export async function requestMediaLibraryPermission() {
  if (Platform.OS === 'web') return true;
  const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
  return status === 'granted';
}

// Web fallback: programmatically open a file picker
function pickFromWeb() {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.style.display = 'none';
    document.body.appendChild(input);

    input.addEventListener('cancel', () => {
      document.body.removeChild(input);
      resolve(null);
    });

    input.addEventListener('change', () => {
      const file = input.files && input.files[0];
      document.body.removeChild(input);
      if (!file) {
        resolve(null);
        return;
      }
      const uri = URL.createObjectURL(file);
      resolve({
        uri,
        fileName: file.name || 'invitation.jpg',
        type: file.type || 'image/jpeg',
        // Keep the File object so we can send it as FormData on web
        file,
      });
    });

    input.click();
  });
}

/**
 * Pick an image from the gallery.
 * Returns { uri, fileName, type, file? } or null if cancelled.
 */
export async function pickImageFromLibrary() {
  if (Platform.OS === 'web') {
    return pickFromWeb();
  }

  const granted = await requestMediaLibraryPermission();
  if (!granted) {
    throw new Error('Photo library permission denied.');
  }

  const res = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    quality: 1,
    allowsMultipleSelection: false,
  });

  if (res.canceled) return null;
  const asset = res.assets && res.assets[0];
  if (!asset) return null;

  console.log('[MOBILE] Image selected', asset.uri);
  return {
    uri: asset.uri,
    fileName: asset.fileName || 'invitation.jpg',
    type: asset.mimeType || 'image/jpeg',
  };
}

/**
 * Capture an image with the camera (native only).
 * Returns { uri, fileName, type } or null if cancelled.
 */
export async function captureImage() {
  if (Platform.OS === 'web') {
    throw new Error('Camera capture is not supported on web. Use the gallery instead.');
  }

  const granted = await requestCameraPermission();
  if (!granted) {
    throw new Error('Camera permission denied.');
  }

  const res = await ImagePicker.launchCameraAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    quality: 1,
    cameraType: ImagePicker.CameraType.back,
  });

  if (res.canceled) return null;
  const asset = res.assets && res.assets[0];
  if (!asset) return null;

  console.log('[MOBILE] Image selected', asset.uri);
  return {
    uri: asset.uri,
    fileName: asset.fileName || 'capture.jpg',
    type: asset.mimeType || 'image/jpeg',
  };
}

export default {
  pickImageFromLibrary,
  captureImage,
  requestCameraPermission,
  requestMediaLibraryPermission,
};
