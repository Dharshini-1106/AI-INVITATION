import { Platform } from 'react-native';
import Constants from 'expo-constants';

function apiBaseUrl(url) {
  const trimmed = url.replace(/\/$/, '');
  return trimmed.endsWith('/api/v1') ? trimmed : `${trimmed}/api/v1`;
}

const configuredUrl = process.env.EXPO_PUBLIC_API_URL;
const emulatorUrl = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

// In Expo Go / development builds, hostUri is the computer running Metro.
// A physical Android phone can use that LAN address to reach the API too.
function expoHostUrl() {
  const hostUri = Constants.expoConfig?.hostUri
    || Constants.manifest2?.extra?.expoClient?.hostUri
    || Constants.manifest?.debuggerHost;
  if (!hostUri) return null;

  try {
    const hostname = new URL(`http://${hostUri}`).hostname;
    return hostname ? `http://${hostname}:8000/api/v1` : null;
  } catch {
    return null;
  }
}
const expoLanUrl = Platform.OS === 'android' ? expoHostUrl() : null;

const API_CONFIG = {
  baseURL: apiBaseUrl(configuredUrl || emulatorUrl),
  endpoints: {
    travelPlan: '/travel/plan',
  },
  timeout: 180000,
  analyzeTimeout: 600000,
};

function candidateBaseUrls() {
  return [...new Set([
    configuredUrl && apiBaseUrl(configuredUrl),
    expoLanUrl,
    API_CONFIG.baseURL,
    'http://127.0.0.1:8000/api/v1',
  ].filter(Boolean))];
}

export { candidateBaseUrls };
export default API_CONFIG;
