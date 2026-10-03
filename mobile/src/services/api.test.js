jest.mock('react-native', () => ({ Platform: { OS: 'android' } }));
jest.mock('expo-secure-store', () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));
jest.mock('../config/apiConfig', () => ({
  __esModule: true,
  default: { baseURL: 'http://initial:8000/api/v1', timeout: 1000, analyzeTimeout: 1000, endpoints: {} },
  candidateBaseUrls: jest.fn(() => ['http://initial:8000/api/v1']),
}));

jest.mock('axios', () => {
  const instances = [];
  const create = jest.fn((config) => {
    const instance = {
      defaults: { ...config, headers: { common: {} } },
      interceptors: {
        request: { use: jest.fn((handler) => { instance.requestInterceptor = handler; }) },
        response: { use: jest.fn((success, failure) => { instance.responseErrorHandler = failure; }) },
      },
      get: jest.fn().mockResolvedValue({ data: { status: 'ok' } }),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
    };
    instances.push(instance);
    return instance;
  });
  return {
    __esModule: true,
    default: { create, request: jest.fn(), instances },
  };
});

import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import { login, rediscoverBackend, getCurrentUser, getPipelineStages, analyzeInvitation } from './api';
import { candidateBaseUrls } from '../config/apiConfig';

beforeEach(async () => {
  jest.clearAllMocks();
  candidateBaseUrls.mockReturnValue(['http://initial:8000/api/v1']);
  SecureStore.getItemAsync.mockResolvedValue(null);
  axios.request.mockResolvedValue({
    data: { user: { id: 'user-1' }, session_token: 'opaque-session-token' },
  });
  await rediscoverBackend();
});

test('login stores the server session and protected requests send its bearer token', async () => {
  await login('demo@example.com', 'GoodPassword42');
  expect(axios.request).toHaveBeenCalledWith(expect.objectContaining({
    headers: expect.objectContaining({ 'X-Session-Transport': 'bearer' }),
  }));
  expect(SecureStore.setItemAsync).toHaveBeenCalledWith('invitation_session_token', 'opaque-session-token');

  await getPipelineStages();
  const activeClient = axios.instances[0];
  const config = await activeClient.requestInterceptor({ headers: {} });
  expect(config.headers.Authorization).toBe('Bearer opaque-session-token');
  expect(activeClient.get).toHaveBeenCalledWith('/pipeline/stages', expect.objectContaining({
    headers: { Authorization: 'Bearer opaque-session-token' },
  }));
});

test('rediscovered clients keep the authenticated request interceptor', async () => {
  await login('demo@example.com', 'GoodPassword42');
  candidateBaseUrls.mockReturnValue(['http://rediscovered:8000/api/v1']);
  await rediscoverBackend();

  const rediscoveredClient = axios.instances.filter((instance) => instance.requestInterceptor).at(-1);
  const config = await rediscoveredClient.requestInterceptor({ headers: {} });
  expect(config.headers.Authorization).toBe('Bearer opaque-session-token');
});

test('multipart analysis explicitly includes the active bearer session', async () => {
  await login('demo@example.com', 'GoodPassword42');
  const activeClient = axios.instances.filter((instance) => instance.requestInterceptor).at(-1);
  activeClient.post.mockResolvedValue({ data: {} });
  await analyzeInvitation({ uri: 'file:///demo.jpg', fileName: 'demo.jpg', type: 'image/jpeg' });
  expect(activeClient.post).toHaveBeenCalledWith('/analyze', expect.any(FormData), expect.objectContaining({
    headers: expect.objectContaining({ Authorization: 'Bearer opaque-session-token' }),
  }));
});

test('an expired restored session is cleared and treated as signed out', async () => {
  await login('demo@example.com', 'GoodPassword42');
  axios.request.mockRejectedValueOnce({ response: { status: 401 } });
  await expect(getCurrentUser()).resolves.toBeNull();
  expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith('invitation_session_token');
});
