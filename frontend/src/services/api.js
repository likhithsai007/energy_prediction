const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Check backend health
 */
export async function fetchHealth() {
  const response = await fetch(`${BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Request ML consumption forecast for a given date and time range
 */
export async function generateForecast(payload) {
  const response = await fetch(`${BASE_URL}/api/forecast`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Forecast request failed with status ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch comparative regression performance metrics (MAE, RMSE, R2, MAPE)
 */
export async function fetchModelPerformance() {
  const response = await fetch(`${BASE_URL}/api/models/performance`);
  if (!response.ok) {
    throw new Error('Failed to retrieve model performance benchmarks');
  }
  return response.json();
}

/**
 * Fetch K-Means clustering patterns and silhouette score
 */
export async function fetchPatterns() {
  const response = await fetch(`${BASE_URL}/api/patterns`);
  if (!response.ok) {
    throw new Error('Failed to retrieve K-Means clustering patterns');
  }
  return response.json();
}

/**
 * Fetch top feature importances of the chosen tree-based model
 */
export async function fetchFeatureImportance() {
  const response = await fetch(`${BASE_URL}/api/models/feature-importance`);
  if (!response.ok) {
    throw new Error('Failed to retrieve feature importance scores');
  }
  return response.json();
}
