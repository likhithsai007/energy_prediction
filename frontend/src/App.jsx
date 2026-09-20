import React, { useState, useEffect } from 'react';
import Forecast from './pages/Forecast';
import ModelAnalysis from './pages/ModelAnalysis';
import { fetchHealth } from './services/api';
import { Zap, BarChart3, LineChart, Cpu, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('forecast'); // 'forecast' | 'models'
  const [backendHealthy, setBackendHealthy] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(() => setBackendHealthy(true))
      .catch(() => setBackendHealthy(false));
  }, []);

  return (
    <div className="app-layout">
      {/* Top Navigation Bar */}
      <header className="navbar">
        <div className="nav-container">
          <div className="brand-group">
            <div className="brand-logo">
              <Zap size={22} className="brand-icon" />
            </div>
            <div>
              <h1 className="brand-name">SMART ENERGY</h1>
              <span className="brand-tagline">Consumption Forecasting & Pattern Analytics</span>
            </div>
          </div>

          <div className="nav-right">
            <nav className="nav-tabs">
              <button
                id="nav-forecast-tab"
                className={`nav-tab ${activeTab === 'forecast' ? 'active' : ''}`}
                onClick={() => setActiveTab('forecast')}
              >
                <LineChart size={17} />
                <span>Forecast / Analysis</span>
              </button>

              <button
                id="nav-models-tab"
                className={`nav-tab ${activeTab === 'models' ? 'active' : ''}`}
                onClick={() => setActiveTab('models')}
              >
                <Cpu size={17} />
                <span>Model Analysis</span>
              </button>
            </nav>

            <div className="status-indicator" title={backendHealthy ? 'FastAPI Backend Online' : 'Backend Connecting...'}>
              <span className={`status-dot ${backendHealthy ? 'online' : 'connecting'}`}></span>
              <span className="status-text">{backendHealthy ? 'ML Service Ready' : 'Connecting'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content View */}
      <main className="main-content">
        {activeTab === 'forecast' ? <Forecast /> : <ModelAnalysis />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <p>© 2026 Smart Energy Consumption Forecasting • Supervised ML & K-Means Clustering</p>
          <div className="footer-badges">
            <span className="footer-badge">FastAPI</span>
            <span className="footer-badge">scikit-learn</span>
            <span className="footer-badge">React + Vite</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
