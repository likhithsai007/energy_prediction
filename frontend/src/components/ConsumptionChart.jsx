import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot
} from 'recharts';
import { LineChart as ChartIcon } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="custom-chart-tooltip">
        <p className="tooltip-time">{data.full_time || `Time: ${label}`}</p>
        <div className="tooltip-row">
          <span className="tooltip-indicator"></span>
          <span className="tooltip-label">Consumption:</span>
          <span className="tooltip-val">{data.consumption} Wh</span>
        </div>
        <p className="tooltip-sub">({data.consumption_kwh} kWh)</p>
      </div>
    );
  }
  return null;
};

export default function ConsumptionChart({ predictions, peakPeriod }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="card chart-card empty-chart">
        <ChartIcon size={32} className="text-muted" />
        <p>No forecast data generated yet. Select date and time range to forecast.</p>
      </div>
    );
  }

  // Find peak point for annotation dot
  const peakTime = peakPeriod?.peak_time;
  const peakVal = peakPeriod?.value;

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <div>
          <h3 className="card-title">Consumption vs Time</h3>
          <span className="card-subtitle">
            10-minute continuous electricity consumption forecast (Wh)
          </span>
        </div>
        <div className="chart-legend">
          <span className="legend-dot"></span>
          <span className="legend-text">Forecasted Load (Wh)</span>
        </div>
      </div>

      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart
            data={predictions}
            margin={{ top: 20, right: 20, left: -10, bottom: 0 }}
          >
            <defs>
              <linearGradient id="energyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#0284c7" stopOpacity={0.45} />
                <stop offset="95%" stopColor="#0284c7" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
            <XAxis
              dataKey="time"
              stroke="#64748b"
              fontSize={12}
              tickLine={false}
              interval="preserveStartEnd"
              minTickGap={30}
            />
            <YAxis
              stroke="#64748b"
              fontSize={12}
              tickLine={false}
              unit=" Wh"
              domain={[0, 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="consumption"
              stroke="#0284c7"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#energyGradient)"
              animationDuration={800}
            />
            {peakTime && peakVal && (
              <ReferenceDot
                x={peakTime}
                y={peakVal}
                r={5}
                fill="#ef4444"
                stroke="#ffffff"
                strokeWidth={2}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
