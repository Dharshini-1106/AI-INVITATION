import axios from 'axios';
import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import API_CONFIG, { candidateBaseUrls } from '../config/apiConfig';
import { InvitationResult } from '../models/InvitationResult';

const SESSION_TOKEN_KEY = 'invitation_session_token';
let sessionToken = null;
let client;
let sessionExpiredHandler = null;

async function readSessionToken() {
  if (Platform.OS !== 'web' && !sessionToken) {
    sessionToken = await SecureStore.getItemAsync(SESSION_TOKEN_KEY);
    if (sessionToken) client.defaults.headers.common.Authorization = `Bearer ${sessionToken}`;
  }
  return sessionToken;
}

async function saveSessionToken(token) {
  sessionToken = token || null;
  if (Platform.OS !== 'web') {
    if (sessionToken) await SecureStore.setItemAsync(SESSION_TOKEN_KEY, sessionToken);
    else await SecureStore.deleteItemAsync(SESSION_TOKEN_KEY);
  }
  if (sessionToken) client.defaults.headers.common.Authorization = `Bearer ${sessionToken}`;
  else delete client.defaults.headers.common.Authorization;
}

function createApiClient(baseURL) {
  const apiClient = axios.create({
    baseURL,
    timeout: API_CONFIG.timeout,
    withCredentials: true,
    headers: { 'Content-Type': 'application/json' },
  });
  // Attach the same revocable server session to every protected request.
  // Installing this on each discovered client also prevents backend
  // rediscovery from silently dropping the Authorization header.
  apiClient.interceptors.request.use(async (config) => {
    const token = await readSessionToken();
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });
  apiClient.interceptors.response.use(
    (response) => response,
    async (error) => {
      if (error.response?.status === 401) {
        await saveSessionToken(null);
        sessionExpiredHandler?.();
      }
      return Promise.reject(error);
    },
  );
  return apiClient;
}

client = createApiClient(API_CONFIG.baseURL);

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
          client = createApiClient(base);
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
  await readSessionToken();
  return client;
}

// Expose the currently resolved base URL (for UI display).
function getResolvedBaseUrl() {
  return client.defaults.baseURL;
}

function registerSessionExpiredHandler(handler) {
  sessionExpiredHandler = handler;
  return () => {
    if (sessionExpiredHandler === handler) sessionExpiredHandler = null;
  };
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
  const token = await readSessionToken();
  const headers = { 'Content-Type': 'application/json' };
  headers['X-Session-Transport'] = 'bearer';
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await axios.request({
    method,
    url: `/api/auth/${path}`,
    baseURL,
    data,
    timeout: API_CONFIG.timeout,
    withCredentials: true,
    headers,
  });
  return response.data;
}

async function signup(payload) {
  return authRequest('post', 'signup', payload);
}

async function login(email, password) {
  const result = await authRequest('post', 'login', { email, password });
  // Use the bearer token for API calls on every platform. Browser cookies can
  // be dropped when Expo Web and the API use different hostnames (for example
  // 127.0.0.1 vs localhost). Keep the web token in memory only; native clients
  // persist it in SecureStore.
  if (!result.session_token) {
    throw new Error('The account service did not return a session. Restart the backend and try logging in again.');
  }
  await saveSessionToken(result.session_token);
  return result.user;
}

async function getCurrentUser() {
  // Do not treat a legacy cookie session as a native session. Native logins
  // must have a SecureStore token so protected uploads can send it explicitly.
  if (Platform.OS !== 'web' && !(await readSessionToken())) return null;
  try {
    return (await authRequest('get', 'me')).user;
  } catch (error) {
    if (error.response?.status === 401) {
      await saveSessionToken(null);
      return null;
    }
    throw error;
  }
}

async function logout() {
  try { return await authRequest('post', 'logout', {}); }
  finally { await saveSessionToken(null); }
}

// Get pipeline stages for progress display
async function getPipelineStages() {
  const c = await resolvedClient();
  const token = await readSessionToken();
  const res = await c.get('/pipeline/stages', {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
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
  const token = await readSessionToken();
  if (token) headers.Authorization = `Bearer ${token}`;
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

async function createEvent(event) {
  const c = await resolvedClient();
  return (await c.post('/events', { event })).data;
}

async function listEvents() {
  const c = await resolvedClient();
  return (await c.get('/events')).data;
}

async function getEvent(eventId) {
  const c = await resolvedClient();
  return (await c.get(`/events/${encodeURIComponent(eventId)}`)).data;
}

async function updateEvent(eventId, event) {
  const c = await resolvedClient();
  return (await c.put(`/events/${encodeURIComponent(eventId)}`, { event })).data;
}

async function saveTravelPlan(eventId, travelPlan, request) {
  const c = await resolvedClient();
  return (await c.put(`/events/${encodeURIComponent(eventId)}/travel-plan`, {
    travel_plan: travelPlan,
    request,
  })).data;
}

async function saveEventSchedule(eventId, schedule) {
  const c = await resolvedClient();
  return (await c.put(`/events/${encodeURIComponent(eventId)}/schedule`, { schedule })).data;
}

async function deleteEvent(eventId) {
  const c = await resolvedClient();
  await c.delete(`/events/${encodeURIComponent(eventId)}`);
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
  registerSessionExpiredHandler,
  planTravel,
  createEvent,
  listEvents,
  getEvent,
  updateEvent,
  saveTravelPlan,
  saveEventSchedule,
  deleteEvent,
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
  registerSessionExpiredHandler,
  planTravel,
  createEvent,
  listEvents,
  getEvent,
  updateEvent,
  saveTravelPlan,
  saveEventSchedule,
  deleteEvent,
  createCalendarEvents,
  getCalendarAuthUrl,
};
