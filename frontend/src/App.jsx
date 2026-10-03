import React from 'react';
import { Routes, Route } from 'react-router-dom';

import SplashScreen from './screens/SplashScreen';
import HomeScreen from './screens/HomeScreen';
import GalleryUploadScreen from './screens/GalleryUploadScreen';
import CameraScanScreen from './screens/CameraScanScreen';
import ProcessingScreen from './screens/ProcessingScreen';
import ResultScreen from './screens/ResultScreen';
import AuthScreen from './screens/AuthScreen';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './auth/AuthContext';

function App() {
  const guarded = (element) => <ProtectedRoute>{element}</ProtectedRoute>;
  return (
    <AuthProvider>
    <Routes>
      <Route path="/" element={<SplashScreen />} />
      <Route path="/login" element={<AuthScreen mode="login" />} />
      <Route path="/signup" element={<AuthScreen mode="signup" />} />
      <Route path="/home" element={guarded(<HomeScreen />)} />
      <Route path="/upload" element={guarded(<GalleryUploadScreen />)} />
      <Route path="/scan" element={guarded(<CameraScanScreen />)} />
      <Route path="/processing" element={guarded(<ProcessingScreen />)} />
      <Route path="/result" element={guarded(<ResultScreen />)} />
    </Routes>
    </AuthProvider>
  );
}

export default App;
