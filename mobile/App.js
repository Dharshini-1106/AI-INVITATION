import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import HomeScreen from './src/screens/HomeScreen';
import GalleryUploadScreen from './src/screens/GalleryUploadScreen';
import CameraScanScreen from './src/screens/CameraScanScreen';
import ProcessingScreen from './src/screens/ProcessingScreen';
import ResultScreen from './src/screens/ResultScreen';
import AuthScreen from './src/screens/AuthScreen';
import SavedEventsScreen from './src/screens/SavedEventsScreen';
import SavedEventScreen from './src/screens/SavedEventScreen';
import { getCurrentUser, logout, registerSessionExpiredHandler } from './src/services/api';
import colors from './src/theme/colors';

const Stack = createNativeStackNavigator();

const globalScreenOptions = {
  headerShown: false,
  animation: 'fade',
};

export default function App() {
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(true);

  useEffect(() => {
    let mounted = true;
    getCurrentUser()
      .then((currentUser) => { if (mounted) setUser(currentUser); })
      .catch(() => { if (mounted) setUser(null); })
      .finally(() => { if (mounted) setCheckingSession(false); });
    return () => { mounted = false; };
  }, []);

  useEffect(() => registerSessionExpiredHandler(() => setUser(null)), []);

  const handleLogout = async () => {
    try { await logout(); } catch (error) { console.warn('[auth] Logout request failed:', error?.message); }
    setUser(null);
  };

  if (checkingSession) {
    return (
      <SafeAreaProvider>
        <View style={loadingStyles.container}>
          <ActivityIndicator size="large" color={colors.secondary} />
          <Text style={loadingStyles.label}>Checking your session…</Text>
        </View>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator key={user ? 'private' : 'public'} initialRouteName={user ? 'Home' : 'Login'} screenOptions={globalScreenOptions}>
          {user ? <>
            <Stack.Screen name="Home">
              {(props) => <HomeScreen {...props} onLogout={handleLogout} />}
            </Stack.Screen>
            <Stack.Screen name="GalleryUpload" component={GalleryUploadScreen} />
            <Stack.Screen name="CameraScan" component={CameraScanScreen} />
            <Stack.Screen name="Processing" component={ProcessingScreen} />
            <Stack.Screen name="Result" component={ResultScreen} />
            <Stack.Screen name="SavedEvents" component={SavedEventsScreen} />
            <Stack.Screen name="SavedEvent" component={SavedEventScreen} />
          </> : <>
            <Stack.Screen name="Login">
              {(props) => <AuthScreen {...props} setUser={setUser} />}
            </Stack.Screen>
            <Stack.Screen name="Signup">
              {(props) => <AuthScreen {...props} setUser={setUser} />}
            </Stack.Screen>
          </>}
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const loadingStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, alignItems: 'center', justifyContent: 'center' },
  label: { color: colors.textMuted, marginTop: 14, fontSize: 14 },
});
