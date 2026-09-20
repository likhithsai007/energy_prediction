import React from 'react';
import { Flame, Clock, TrendingUp } from 'lucide-react';

export default function PeakCard({ peakPeriod }) {
  const start = peakPeriod?.start || '--:--';
  const end = peakPeriod?.end || '--:--';
  const peakVal = peakPeriod?.value ? Number(peakPeriod.value).toFixed(1) : '0.0';
  const peakTime = peakPeriod?.peak_time || start;

  return (
    <div className="card peak-card">
      <div className="card-header-flex">
        <div className="card-title-group">
          <div className="icon-wrapper alert-glow">
            <Flame className="icon-alert" size={20} />
          </div>
          <div>
            <h3 className="card-title">Peak Demand Window</h3>
            <span className="card-subtitle">Highest instantaneous consumption interval</span>
          </div>
        </div>
      </div>

      <div className="peak-metrics-grid">
        <div className="metric-box">
          <div className="metric-header-row">
            <Clock size={16} className="text-muted" />
            <span className="metric-label">Peak Interval</span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value-large">{start} – {end}</span>
          </div>
          <span className="metric-hint">Maximum surge reached at {peakTime}</span>
        </div>

        <div className="metric-box">
          <div className="metric-header-row">
            <TrendingUp size={16} className="text-muted" />
            <span className="metric-label">Peak Power Output</span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value-large text-alert">{peakVal}</span>
            <span className="metric-unit">Wh</span>
          </div>
          <span className="metric-hint">≈ {(peakVal / 1000).toFixed(3)} kW peak load</span>
        </div>
      </div>
    </div>
  );
}
