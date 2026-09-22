import axios from 'axios';
import API_CONFIG from '../config/apiConfig';
import { InvitationResult } from '../models/InvitationResult';

const client = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
  getPipelineStages,
  planTravel,
  analyzeInvitation,
  getApiBaseUrl,
};
