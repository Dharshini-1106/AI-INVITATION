import React from 'react';
import { act, create } from 'react-test-renderer';
import * as Location from 'expo-location';
import { planTravel } from '../services/api';
import TravelPlanModal from './TravelPlanModal';

jest.mock('react-native', () => {
  const React = require('react');
  const component = (name) => {
    const Component = ({ children, ...props }) => React.createElement(name, props, children);
    Component.displayName = name;
    return Component;
  };

  return {
    ActivityIndicator: component('ActivityIndicator'),
    Linking: { canOpenURL: jest.fn(), openURL: jest.fn() },
  Modal: component('Modal'),
  Platform: { OS: 'android' },
    ScrollView: component('ScrollView'),
    StyleSheet: { create: (styles) => styles },
    Text: component('Text'),
    TextInput: component('TextInput'),
    TouchableOpacity: component('TouchableOpacity'),
    View: component('View'),
  };
});

jest.mock('expo-location', () => ({
  Accuracy: { High: 4 },
  requestForegroundPermissionsAsync: jest.fn(),
  hasServicesEnabledAsync: jest.fn(),
  enableNetworkProviderAsync: jest.fn(),
  getCurrentPositionAsync: jest.fn(),
}));

jest.mock('expo-notifications', () => ({
  setNotificationHandler: jest.fn(),
  getPermissionsAsync: jest.fn().mockResolvedValue({ status: 'granted' }),
  requestPermissionsAsync: jest.fn(),
  setNotificationChannelAsync: jest.fn().mockResolvedValue(undefined),
  getAllScheduledNotificationsAsync: jest.fn().mockResolvedValue([]),
  cancelScheduledNotificationAsync: jest.fn(),
  scheduleNotificationAsync: jest.fn().mockResolvedValue('notification-id'),
  AndroidImportance: { HIGH: 4 },
}));

jest.mock('../services/api', () => ({
  getResolvedBaseUrl: jest.fn(() => 'http://localhost:8000/api/v1'),
  planTravel: jest.fn(),
}));

const event = {
  event_name: 'Wedding',
  date: 'October 2, 2026',
  time: '6:00 PM',
  address: '',
  venue: 'Convention Center',
  timezone: 'Asia/Kolkata',
};

function findByText(root, text) {
  return root.find((node) => node.props?.children === text);
}

async function tap(root, text) {
  const node = findByText(root, text);
  await act(async () => {
    await node.parent.props.onPress();
  });
}

function renderModal() {
  return create(<TravelPlanModal event={event} />);
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
  Location.requestForegroundPermissionsAsync.mockResolvedValue({ status: 'granted' });
  Location.hasServicesEnabledAsync.mockResolvedValue(true);
  Location.enableNetworkProviderAsync.mockResolvedValue(undefined);
  Location.getCurrentPositionAsync.mockResolvedValue({
    coords: { latitude: 12.9716, longitude: 77.5946 },
  });
  planTravel.mockResolvedValue({ google_maps_url: 'https://example.test/maps' });
});

afterEach(() => {
  console.log.mockRestore();
  console.error.mockRestore();
});

test('sends the current coordinates to the travel API', async () => {
  const component = renderModal();
  await tap(component.root, 'Plan My Travel');
  await tap(component.root, 'Use Current Location');

  expect(component.root.find((node) => node.props?.children === 'Starting location: Your current location is ready. It is used for this route calculation and is not saved in the app.')).toBeDefined();

  await tap(component.root, 'Calculate Travel Plan');

  expect(planTravel).toHaveBeenCalledWith(expect.objectContaining({
    origin: { type: 'current', latitude: 12.9716, longitude: 77.5946 },
    destination: { type: 'event', address: 'Convention Center' },
  }));
  expect(console.log).toHaveBeenCalledWith('[travel] location permission:', 'granted');
  expect(console.log).toHaveBeenCalledWith('[travel] location services:', 'enabled');
  expect(console.log).toHaveBeenCalledWith('[travel] current coordinates:', 12.9716, 77.5946);
  expect(console.log).toHaveBeenCalledWith('[travel] travel request origin:', { type: 'current', latitude: 12.9716, longitude: 77.5946 });
});

test('shows a clear error when location permission is denied', async () => {
  Location.requestForegroundPermissionsAsync.mockResolvedValue({ status: 'denied' });
  const component = renderModal();
  await tap(component.root, 'Plan My Travel');
  await tap(component.root, 'Use Current Location');

  expect(component.root.find((node) => node.props?.children === 'Location permission is required to use your current location.')).toBeDefined();
  expect(planTravel).not.toHaveBeenCalled();
});

test('shows a clear error when location services are disabled', async () => {
  Location.hasServicesEnabledAsync.mockResolvedValue(false);
  const component = renderModal();
  await tap(component.root, 'Plan My Travel');
  await tap(component.root, 'Use Current Location');

  expect(component.root.find((node) => node.props?.children === 'Please enable location services and try again.')).toBeDefined();
  expect(planTravel).not.toHaveBeenCalled();
});

test('prompts to enable location services and then uses the current coordinates', async () => {
  Location.hasServicesEnabledAsync
    .mockResolvedValueOnce(false)
    .mockResolvedValueOnce(true);
  const component = renderModal();
  await tap(component.root, 'Plan My Travel');
  await tap(component.root, 'Use Current Location');

  expect(Location.enableNetworkProviderAsync).toHaveBeenCalled();
  expect(planTravel).not.toHaveBeenCalled();

  await tap(component.root, 'Calculate Travel Plan');

  expect(planTravel).toHaveBeenCalledWith(expect.objectContaining({
    origin: { type: 'current', latitude: 12.9716, longitude: 77.5946 },
  }));
});

test('sends a manually entered starting location', async () => {
  const component = renderModal();
  await tap(component.root, 'Plan My Travel');
  await tap(component.root, 'Enter Manually');

  const originInput = component.root.findAllByType('TextInput').find((node) => node.props.placeholder === 'Starting location');
  await act(async () => {
    originInput.props.onChangeText('Central Station');
  });
  await tap(component.root, 'Calculate Travel Plan');

  expect(planTravel).toHaveBeenCalledWith(expect.objectContaining({
    origin: { type: 'manual', address: 'Central Station' },
    destination: { type: 'event', address: 'Convention Center' },
  }));
});

test('maps Car to DRIVE and Bike / Scooter to TWO_WHEELER in the API request', async () => {
  const car = renderModal();
  await tap(car.root, 'Plan My Travel');
  await tap(car.root, 'Enter Manually');
  const carOrigin = car.root.findAllByType('TextInput').find((node) => node.props.placeholder === 'Starting location');
  await act(async () => carOrigin.props.onChangeText('Thiruchendur, Tamil Nadu'));
  await tap(car.root, 'Calculate Travel Plan');
  expect(planTravel).toHaveBeenCalledWith(expect.objectContaining({
    travel_mode: 'DRIVE',
  }));

  planTravel.mockClear();
  const bike = renderModal();
  await tap(bike.root, 'Plan My Travel');
  await tap(bike.root, 'Enter Manually');
  const bikeOrigin = bike.root.findAllByType('TextInput').find((node) => node.props.placeholder === 'Starting location');
  await act(async () => bikeOrigin.props.onChangeText('Thiruchendur, Tamil Nadu'));
  await tap(bike.root, 'Bike / Scooter');
  await tap(bike.root, 'Calculate Travel Plan');
  expect(planTravel).toHaveBeenCalledWith(expect.objectContaining({
    travel_mode: 'TWO_WHEELER',
  }));
});
