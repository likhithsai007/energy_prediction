import React, { useState, useEffect } from 'react';
import { generateForecast } from '../services/api';
import ForecastCard from '../components/ForecastCard';
import PeakCard from '../components/PeakCard';
import ConsumptionChart from '../components/ConsumptionChart';
import { Calendar, Clock, Loader2, Sparkles, AlertCircle } from 'lucide-react';

export default function Forecast() {
  // Default values
  const [date, setDate] = useState('2026-09-22');
  const [startTime, setStartTime] = useState('00:00');
  const [endTime, setEndTime] = useState('23:59');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  // Auto-generate initial forecast on page load
  useEffect(() => {
    handleGenerate();
  }, []);

  const handleGenerate = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await generateForecast({
        date,
        start_time: startTime,
        end_time: endTime,
      });
      setForecastData(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to generate consumption forecast. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      {/* Control / Input Panel */}
      <section className="card control-card">
        <div className="control-header">
          <div className="section-title-group">
            <h2 className="section-title">Forecasting Configuration</h2>
            <p className="section-subtitle">
              Configure forecasting parameters. The backend automatically leverages the best-performing ML model.
            </p>
          </div>
        </div>

        <form onSubmit={handleGenerate} className="forecast-form">
          <div className="form-group">
            <label className="form-label" htmlFor="forecast-date">
              <Calendar size={15} /> Select Date
            </label>
            <input
              id="forecast-date"
              type="date"
              className="form-input"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="start-time">
              <Clock size={15} /> Start Time
            </label>
            <input
              id="start-time"
              type="time"
              className="form-input"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="end-time">
              <Clock size={15} /> End Time
            </label>
            <input
              id="end-time"
              type="time"
              className="form-input"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              required
            />
          </div>

          <div className="form-action">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              id="generate-forecast-btn"
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={16} /> Generating Forecast...
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Generate Forecast
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Forecast Results Grid */}
      {forecastData && (
        <>
          <div className="grid-two-col">
            <ForecastCard
              totalKwh={forecastData.total_consumption}
              totalWh={forecastData.total_consumption_wh}
              pattern={forecastData.pattern}
              recommendation={forecastData.recommendation}
              modelName={forecastData.model}
            />

            <PeakCard peakPeriod={forecastData.peak_period} />
          </div>

          <ConsumptionChart
            predictions={forecastData.predictions}
            peakPeriod={forecastData.peak_period}
          />
        </>
      )}
    </div>
  );
}
