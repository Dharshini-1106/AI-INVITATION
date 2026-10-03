import axios from 'axios';
import { Platform } from 'react-native';
import API_CONFIG, { candidateBaseUrls } from '../config/apiConfig';
import { InvitationResult } from '../models/InvitationResult';

let client = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Try each candidate backend base URL and pick the first one that responds
 * to /health. This lets the app connect whether the phone is on the same WiFi
 * as the PC, using USB adb reverse, or running on the web.
 */
async function discoverBackend() {
  const candidates = candidateBaseUrls();
  const failures = [];
  for (const base of candidates) {
    try {
      const probe = axios.create({
        baseURL: base,
        timeout: 4000,
        headers: { 'Content-Type': 'application/json' },
      });
      const res = await probe.get('/health');
      if (res.data && res.data.status === 'ok') {
        if (base !== client.defaults.baseURL) {
          client = axios.create({
            baseURL: base,
            timeout: API_CONFIG.timeout,
            withCredentials: true,
            headers: { 'Content-Type': 'application/json' },
          });
          // eslint-disable-next-line no-console
          console.log(`[api] Backend discovered at ${base}`);
        }
        return base;
      }
    } catch (e) {
      failures.push(`${base}: ${e?.message || 'unavailable'}`);
    }
  }
  if (failures.length === candidates.length && failures.length) {
    console.warn(`[api] Backend discovery failed: ${failures.join('; ')}`);
  }
  return client.defaults.baseURL || null;

}

// Run discovery once at module load. Subsequent calls use the resolved client.
let discoveryPromise = discoverBackend();

// Re-run discovery (used by the Home screen retry). Returns the resolved baseURL.
async function rediscoverBackend() {
  discoveryPromise = discoverBackend();
  return discoveryPromise;
}

// Resolve the backend client before making any request.
async function resolvedClient() {
  await discoveryPromise;
  if (!client.defaults.baseURL) {
    throw new Error('Backend URL is not configured. Build the APK with EXPO_PUBLIC_API_URL set to your PC LAN address.');
  }
  return client;
}

// Expose the currently resolved base URL (for UI display).
function getResolvedBaseUrl() {
  return client.defaults.baseURL;
}

// Health check
async function checkHealth() {
  const c = await resolvedClient();
  const res = await c.get('/health');
  return res.data;
}

async function authRequest(method, path, data) {
  const c = await resolvedClient();
  const baseURL = c.defaults.baseURL.replace(/\/api\/v1\/?$/, '');
  const response = await axios.request({
    method,
    url: `/api/auth/${path}`,
    baseURL,
    data,
    timeout: API_CONFIG.timeout,
    withCredentials: true,
    headers: { 'Content-Type': 'application/json' },
  });
  return response.data;
}

async function signup(payload) {
  return authRequest('post', 'signup', payload);
}

async function login(email, password) {
  return (await authRequest('post', 'login', { email, password })).user;
}

async function getCurrentUser() {
  return (await authRequest('get', 'me')).user;
}

async function logout() {
  return authRequest('post', 'logout', {});
}

// Get pipeline stages for progress display
async function getPipelineStages() {
  const c = await resolvedClient();
  const res = await c.get('/pipeline/stages');
  return res.data;
}

// Upload & analyze an invitation image
async function analyzeInvitation(imageAsset) {
  console.log('[MOBILE] Uploading image');
  const c = await resolvedClient();
  const formData = new FormData();

  if (Platform.OS === 'web' && imageAsset.file) {
    formData.append('file', imageAsset.file, imageAsset.fileName || 'invitation.jpg');
  } else {
    formData.append('file', {
      uri: imageAsset.uri,
      name: imageAsset.fileName || 'invitation.jpg',
      type: imageAsset.type || 'image/jpeg',
    });
  }

  const headers = { 'Content-Type': 'multipart/form-data' };
  // The Gallery request remains exactly as before. This camera-only marker is
  // diagnostic metadata for server logs; it does not change the file upload.
  if (imageAsset.source === 'camera') headers['X-Invitation-Source'] = 'camera';
  const res = await c.post('/analyze', formData, {
    headers,
    timeout: API_CONFIG.analyzeTimeout || 600000,
  });
  console.log('[MOBILE] Extraction completed');
  return new InvitationResult(res.data);
}

// Plan travel for an extracted event
async function planTravel(payload) {
  const c = await resolvedClient();
  const res = await c.post(API_CONFIG.endpoints.travelPlan, payload);
  return res.data;
}

// Create calendar events via backend Google Calendar integration
async function createCalendarEvents(events, sessionId) {
  const c = await resolvedClient();
  const res = await c.post('/calendar/create', events, { params: { session_id: sessionId } });
  return res.data;
}

// Get Google OAuth authorization URL from backend
async function getCalendarAuthUrl(sessionId) {
  const c = await resolvedClient();
  const res = await c.get('/calendar/auth-url', { params: { session_id: sessionId } });
  return res.data;
}

export {
  checkHealth,
  signup,
  login,
  getCurrentUser,
  logout,
  getPipelineStages,
  analyzeInvitation,
  rediscoverBackend,
  getResolvedBaseUrl,
  planTravel,
  createCalendarEvents,
  getCalendarAuthUrl,
};

export default {
  checkHealth,
  signup,
  login,
  getCurrentUser,
  logout,
  getPipelineStages,
  analyzeInvitation,
  rediscoverBackend,
  getResolvedBaseUrl,
  planTravel,
  createCalendarEvents,
  getCalendarAuthUrl,
};
