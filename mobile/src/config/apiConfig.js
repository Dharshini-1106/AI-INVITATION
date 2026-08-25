import { Platform } from 'react-native';

function apiBaseUrl(url) {
  const trimmed = url.replace(/\/$/, '');
  return trimmed.endsWith('/api/v1') ? trimmed : `${trimmed}/api/v1`;
}

const configuredUrl = process.env.EXPO_PUBLIC_API_URL;
const emulatorUrl = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

const API_CONFIG = {
  baseURL: apiBaseUrl(configuredUrl || emulatorUrl),
  timeout: 180000,
};

function candidateBaseUrls() {
  return [...new Set([
    configuredUrl && apiBaseUrl(configuredUrl),
    API_CONFIG.baseURL,
    'http://127.0.0.1:8000/api/v1',
  ].filter(Boolean))];
}

export { candidateBaseUrls };
export default API_CONFIG;
