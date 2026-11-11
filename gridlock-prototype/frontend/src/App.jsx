import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SettingsProvider } from './contexts/SettingsContext';
import TopNav from './components/TopNav';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DetectionStudio from './pages/DetectionStudio';
import Settings from './pages/Settings';
import JudgeFeed from './pages/JudgeFeed';
import './App.css';
import './modern.css';

function App() {
  return (
    <SettingsProvider>
      <BrowserRouter>
        <div className="app-container">
          <TopNav />
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/detect" element={<DetectionStudio />} />
              <Route path="/judge" element={<JudgeFeed />} />
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Routes>
        </div>
      </BrowserRouter>
    </SettingsProvider>
  );
}

export default App;