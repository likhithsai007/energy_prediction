import React, { useState, useEffect } from 'react';
import { fetchModelPerformance, fetchPatterns, fetchFeatureImportance } from '../services/api';
import ModelTable from '../components/ModelTable';
import { Loader2, AlertCircle } from 'lucide-react';

export default function ModelAnalysis() {
  const [performance, setPerformance] = useState(null);
  const [patterns, setPatterns] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const [perfRes, pattRes, featRes] = await Promise.all([
          fetchModelPerformance(),
          fetchPatterns(),
          fetchFeatureImportance(),
        ]);
        setPerformance(perfRes);
        setPatterns(pattRes);
        setFeatureImportance(featRes);
      } catch (err) {
        console.error(err);
        setError(err.message || 'Failed to fetch model analysis benchmarks.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="page-container loading-center">
        <Loader2 className="spinner" size={32} />
        <p>Loading model evaluation benchmarks and cluster patterns...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="error-banner">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header-intro">
        <h2 className="section-title">Model Evaluation & Selection Analysis</h2>
        <p className="section-subtitle">
          Transparent metrics breakdown explaining why the optimal ML regressor was selected over baseline alternatives.
        </p>
      </div>

      <ModelTable
        performanceData={performance}
        patternsData={patterns}
        featureImportanceData={featureImportance}
      />
    </div>
  );
}
