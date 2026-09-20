import React from 'react';
import { Zap, Activity, Lightbulb } from 'lucide-react';

export default function ForecastCard({ totalKwh, totalWh, pattern, recommendation, modelName }) {
  const isHigh = pattern?.toLowerCase().includes('high');
  const isMedium = pattern?.toLowerCase().includes('medium');

  const patternBadgeClass = isHigh
    ? 'badge-high'
    : isMedium
    ? 'badge-medium'
    : 'badge-low';

  return (
    <div className="card forecast-summary-card">
      <div className="card-header-flex">
        <div className="card-title-group">
          <div className="icon-wrapper primary-glow">
            <Zap className="icon-primary" size={20} />
          </div>
          <div>
            <h3 className="card-title">Forecast Summary</h3>
            <span className="card-subtitle">Calculated using {modelName || 'Random Forest Regressor'}</span>
          </div>
        </div>
        <span className={`pattern-badge ${patternBadgeClass}`}>
          <Activity size={14} />
          {pattern || 'Optimized Pattern'}
        </span>
      </div>

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-label">Predicted Total Consumption</span>
          <div className="metric-value-row">
            <span className="metric-value">{totalKwh?.toLocaleString() || '0.000'}</span>
            <span className="metric-unit">kWh</span>
          </div>
          <span className="metric-hint">≈ {totalWh ? Math.round(totalWh).toLocaleString() : '0'} Wh total energy</span>
        </div>
      </div>

      {recommendation && (
        <div className="recommendation-box">
          <div className="rec-icon">
            <Lightbulb size={18} />
          </div>
          <div className="rec-content">
            <h4 className="rec-title">Smart Recommendation</h4>
            <p className="rec-text">{recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );
}
