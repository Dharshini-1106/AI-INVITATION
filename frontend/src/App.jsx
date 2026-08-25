import React from 'react';
import { Routes, Route } from 'react-router-dom';

import SplashScreen from './screens/SplashScreen';
import HomeScreen from './screens/HomeScreen';
import GalleryUploadScreen from './screens/GalleryUploadScreen';
import CameraScanScreen from './screens/CameraScanScreen';
import ProcessingScreen from './screens/ProcessingScreen';
import ResultScreen from './screens/ResultScreen';

function App() {
  return (
    <Routes>
      <Route path="/" element={<SplashScreen />} />
      <Route path="/home" element={<HomeScreen />} />
      <Route path="/upload" element={<GalleryUploadScreen />} />
      <Route path="/scan" element={<CameraScanScreen />} />
      <Route path="/processing" element={<ProcessingScreen />} />
      <Route path="/result" element={<ResultScreen />} />
    </Routes>
  );
}

export default App;
