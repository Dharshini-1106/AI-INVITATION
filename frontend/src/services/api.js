import axios from 'axios';
import API_CONFIG from '../config/apiConfig';
import { InvitationResult } from '../models/InvitationResult';

const client = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

const authClient = axios.create({
  baseURL: API_CONFIG.baseURL.replace(/\/api\/v1\/?$/, ''), timeout: API_CONFIG.timeout,
  withCredentials: true, headers: { 'Content-Type': 'application/json' },
});
export async function signup(payload) { return (await authClient.post('/api/auth/signup', payload)).data; }
export async function login(email, password) { return (await authClient.post('/api/auth/login', { email, password })).data.user; }
export async function getCurrentUser() { return (await authClient.get('/api/auth/me')).data.user; }
export async function logout() { await authClient.post('/api/auth/logout'); }

// Health check
export async function checkHealth() {
  const res = await client.get(API_CONFIG.endpoints.health);
  return res.data;
}

// Get pipeline stages for progress display
export async function getPipelineStages() {
  const res = await client.get(API_CONFIG.endpoints.pipelineStages);
  return res.data;
}

export async function planTravel(payload) {
  const res = await client.post(API_CONFIG.endpoints.travelPlan, payload);
  return res.data;
}

// Upload & analyze an invitation image
export async function analyzeInvitation(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await client.post(API_CONFIG.endpoints.analyze, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: API_CONFIG.analyzeTimeout || 600000,
  });
  return new InvitationResult(res.data);
}

// Download a sample / test (optional)
export function getApiBaseUrl() {
  return API_CONFIG.baseURL;
}

export default {
  checkHealth,
  signup, login, getCurrentUser, logout,
  getPipelineStages,
  planTravel,
  analyzeInvitation,
  getApiBaseUrl,
};
