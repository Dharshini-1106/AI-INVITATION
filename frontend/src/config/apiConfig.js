// API configuration for the invitation understanding backend
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  endpoints: {
    health: '/health',
    analyze: '/analyze',
    pipelineStages: '/pipeline/stages',
  },
  timeout: 180000, // generous timeout for quick endpoints
  analyzeTimeout: 600000, // 10 minutes for AI OCR pipeline (slow invites)
};

export default API_CONFIG;
