import React from 'react';
import { Award, CheckCircle2, Layers, BarChart2 } from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';

export default function ModelTable({ performanceData, patternsData, featureImportanceData }) {
  const models = performanceData?.models || {};
  const selectedModel = performanceData?.selected_model || 'Random Forest';
  const selectionReason = performanceData?.selection_reason || 
    'Selected automatically based on the evaluation results of the regression models.';

  const featureImportance = featureImportanceData?.feature_importance || [];
  const kmeans = patternsData || {};

  return (
    <div className="model-analysis-container">
      {/* 1. Regression Comparison Table */}
      <div className="card table-card">
        <div className="card-header-flex">
          <div className="card-title-group">
            <div className="icon-wrapper primary-glow">
              <BarChart2 className="icon-primary" size={20} />
            </div>
            <div>
              <h3 className="card-title">Regression Model Benchmark Comparison</h3>
              <span className="card-subtitle">
                Supervised models evaluated on unseen test partition (20% holdout)
              </span>
            </div>
          </div>
        </div>

        <div className="table-responsive">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model</th>
                <th className="text-right">MAE (Wh)</th>
                <th className="text-right">RMSE (Wh)</th>
                <th className="text-right">R² Score</th>
                <th className="text-right">MAPE (%)</th>
                <th className="text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(models).map(([modelName, metrics]) => {
                const isSelected = modelName.toLowerCase() === selectedModel.toLowerCase();
                return (
                  <tr key={modelName} className={isSelected ? 'selected-row' : ''}>
                    <td className="font-semibold">
                      <div className="model-name-cell">
                        {isSelected && <Award size={16} className="text-primary-accent" />}
                        {modelName} {modelName.includes('Forest') || modelName.includes('Tree') ? 'Regressor' : ''}
                      </div>
                    </td>
                    <td className="text-right font-mono">{metrics.mae?.toFixed(2)}</td>
                    <td className="text-right font-mono">{metrics.rmse?.toFixed(2)}</td>
                    <td className="text-right font-mono highlight-r2">
                      <strong>{metrics.r2?.toFixed(4)}</strong>
                    </td>
                    <td className="text-right font-mono">{metrics.mape?.toFixed(2)}%</td>
                    <td className="text-center">
                      {isSelected ? (
                        <span className="badge-selected">
                          <CheckCircle2 size={12} /> Active Best Model
                        </span>
                      ) : (
                        <span className="badge-evaluated">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selected Model Justification */}
        <div className="selected-model-banner">
          <div className="selected-model-info">
            <span className="banner-subtitle">Production Selected Model</span>
            <h4 className="banner-title">{selectedModel} Regressor</h4>
            <p className="banner-reason">
              <strong>Reason: </strong> {selectionReason}
            </p>
          </div>
        </div>
      </div>

      <div className="grid-two-col">
        {/* 2. K-Means Pattern Analysis Section */}
        <div className="card kmeans-card">
          <div className="card-header-flex">
            <div className="card-title-group">
              <div className="icon-wrapper purple-glow">
                <Layers className="icon-purple" size={20} />
              </div>
              <div>
                <h3 className="card-title">K-Means Pattern Analysis</h3>
                <span className="card-subtitle">Unsupervised load profile clustering</span>
              </div>
            </div>
          </div>

          <div className="kmeans-metrics-row">
            <div className="mini-stat-card">
              <span className="mini-stat-label">Number of Clusters</span>
              <span className="mini-stat-value">{kmeans.n_clusters || 3}</span>
            </div>
            <div className="mini-stat-card">
              <span className="mini-stat-label">Silhouette Score</span>
              <span className="mini-stat-value font-mono">
                {kmeans.silhouette_score !== undefined ? kmeans.silhouette_score : '0.2888'}
              </span>
            </div>
          </div>

          <div className="cluster-list">
            {(kmeans.clusters || [
              { label: 'Low Consumption', avg_consumption_wh: 62.7, description: 'Baseline standby & minimal appliance activity' },
              { label: 'Medium Consumption', avg_consumption_wh: 103.4, description: 'Regular daytime household activity' },
              { label: 'High Consumption', avg_consumption_wh: 148.0, description: 'Peak multi-appliance concurrent operation' }
            ]).map((c, idx) => {
              const labelLower = c.label.toLowerCase();
              const badgeClass = labelLower.includes('high')
                ? 'badge-high'
                : labelLower.includes('medium')
                ? 'badge-medium'
                : 'badge-low';
              return (
                <div key={idx} className="cluster-item">
                  <div className="cluster-item-header">
                    <span className={`pattern-badge ${badgeClass}`}>{c.label}</span>
                    <span className="cluster-avg font-mono">Avg: {c.avg_consumption_wh} Wh</span>
                  </div>
                  <p className="cluster-desc">{c.description}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* 3. Feature Importance Bar Chart */}
        <div className="card feat-card">
          <div className="card-header-flex">
            <div>
              <h3 className="card-title">Feature Importance</h3>
              <span className="card-subtitle">
                Top predictors for {selectedModel} Regressor
              </span>
            </div>
          </div>

          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={featureImportance}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" stroke="#64748b" fontSize={11} domain={[0, 'dataMax + 0.05']} />
                <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={12} width={70} />
                <Tooltip
                  formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Importance']}
                  contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #cbd5e1' }}
                />
                <Bar dataKey="importance" fill="#0284c7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
