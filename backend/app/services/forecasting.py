import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .model_selection import ModelManager
from .recommendations import generate_recommendation
from ..models.train import DATA_PATH

# Load historical reference dataset for empirical sensor profiles
_df_reference = None

def get_reference_df():
    global _df_reference
    if _df_reference is None:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['NSM'] = df['date'].dt.hour * 3600 + df['date'].dt.minute * 60 + df['date'].dt.second
            df['month'] = df['date'].dt.month
            _df_reference = df
    return _df_reference

def generate_forecast(date_str: str, start_time_str: str, end_time_str: str):
    """
    Generates time-series consumption predictions using the best regression model,
    calculates peak period, determines the K-Means cluster pattern, and generates recommendations.
    """
    model_mgr = ModelManager.get_instance()
    reg_model, scaler, feature_cols, model_name = model_mgr.get_best_model_and_scaler()
    kmeans_model, scaler_kmeans, cluster_features, kmeans_meta = model_mgr.get_kmeans_and_scaler()
    
    df_ref = get_reference_df()
    
    # Parse date and time boundaries
    try:
        start_dt = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")
    except ValueError as e:
        # Fallback to standard 24-hour range if format fails
        start_dt = datetime.strptime(f"{date_str} 00:00", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date_str} 23:50", "%Y-%m-%d %H:%M")
        
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(hours=23, minutes=50)
        
    total_minutes = max((end_dt - start_dt).total_seconds() / 60.0, 1.0)
    step_minutes = 2 if total_minutes <= 30 else (5 if total_minutes <= 120 else 10)

    # Generate interval timestamps
    timestamps = []
    curr = start_dt
    while curr < end_dt:
        timestamps.append(curr)
        curr += timedelta(minutes=step_minutes)
    timestamps.append(end_dt)
        
    if not timestamps:
        timestamps = [start_dt]

    # Pre-calculate hour-based empirical environmental profiles from the real dataset
    # Group by (is_weekend, hour) to preserve physical household thermal and weather dynamics
    hourly_profiles = df_ref.groupby(['is_weekend', 'hour'])[feature_cols].mean().to_dict(orient='index')
    overall_mean = df_ref[feature_cols].mean().to_dict()

    feature_rows = []
    for dt_val in timestamps:
        h = dt_val.hour
        m = dt_val.minute
        s = dt_val.second
        dow = dt_val.weekday()
        is_wk = 1 if dow >= 5 else 0
        nsm = h * 3600 + m * 60 + s
        mon = dt_val.month
        
        # Base row from empirical dataset profile matching weekend status and hour
        key = (is_wk, h)
        if key in hourly_profiles:
            row = hourly_profiles[key].copy()
        else:
            row = overall_mean.copy()
            
        # Set exact temporal features for this specific timestamp
        row['hour'] = h
        row['day_of_week'] = dow
        row['is_weekend'] = is_wk
        row['NSM'] = nsm
        row['month'] = mon
        
        feature_rows.append(row)

    df_features = pd.DataFrame(feature_rows)[feature_cols]
    
    # Scale and predict
    X_scaled = scaler.transform(df_features)
    raw_predictions = reg_model.predict(X_scaled)
    # Energy consumption cannot be negative
    predictions_wh = np.maximum(raw_predictions, 0.0)
    
    # Format predictions for frontend consumption chart
    time_series_points = []
    total_energy_wh = 0.0
    
    peak_idx = int(np.argmax(predictions_wh))
    peak_dt = timestamps[peak_idx]
    peak_value = float(round(predictions_wh[peak_idx], 2))
    
    # Trapezoidal / interval integration for accurate kWh
    time_delta_hours = (step_minutes / 60.0)
    for dt_val, val in zip(timestamps, predictions_wh):
        val_rounded = float(round(val, 2))
        total_energy_wh += val_rounded * time_delta_hours
        time_series_points.append({
            "time": dt_val.strftime("%H:%M"),
            "full_time": dt_val.strftime("%Y-%m-%d %H:%M"),
            "consumption": val_rounded, # Wh
            "consumption_kwh": round(val_rounded / 1000.0, 3)
        })
        
    total_consumption_kwh = round(total_energy_wh / 1000.0, 3)
    avg_consumption_wh = float(round(np.mean(predictions_wh), 2))
    
    # Determine Peak Period window (strictly clamped within user's selected start_dt and end_dt)
    if total_minutes <= 60:
        peak_window_start_dt = start_dt
        peak_window_end_dt = end_dt
    else:
        peak_window_start_dt = max(start_dt, peak_dt - timedelta(minutes=30))
        peak_window_end_dt = min(end_dt, peak_window_start_dt + timedelta(hours=1))
        if (peak_window_end_dt - peak_window_start_dt).total_seconds() < 3600:
            peak_window_start_dt = max(start_dt, peak_window_end_dt - timedelta(hours=1))

    peak_period_data = {
        "start": peak_window_start_dt.strftime("%H:%M"),
        "end": peak_window_end_dt.strftime("%H:%M"),
        "value": peak_value,
        "peak_time": peak_dt.strftime("%H:%M")
    }
    
    # K-Means Consumption Pattern Classification
    # Construct cluster feature matrix for the forecasted period
    df_cluster_input = df_features.copy()
    df_cluster_input['Appliances'] = predictions_wh
    
    # Use average representative point or peak profile for clustering
    if scaler_kmeans is not None and kmeans_model is not None:
        cluster_df = df_cluster_input[cluster_features]
        scaled_kmeans_input = scaler_kmeans.transform(cluster_df)
        period_clusters = kmeans_model.predict(scaled_kmeans_input)
        
        # Most frequent cluster during the forecasted window
        dominant_cluster_id = int(pd.Series(period_clusters).mode()[0])
        
        # Map cluster ID to descriptive label
        pattern_name = "Medium Consumption Pattern"
        clusters_list = kmeans_meta.get("clusters", [])
        for c_info in clusters_list:
            if c_info.get("cluster_id") == dominant_cluster_id:
                pattern_name = f"{c_info.get('label')} Pattern"
                break
    else:
        # Fallback if kmeans metadata is unavailable
        if avg_consumption_wh > 150:
            pattern_name = "High Consumption Pattern"
        elif avg_consumption_wh > 80:
            pattern_name = "Medium Consumption Pattern"
        else:
            pattern_name = "Low Consumption Pattern"
            
    # Generate Recommendation
    recommendation = generate_recommendation(
        pattern_label=pattern_name,
        peak_period=peak_period_data,
        avg_value=avg_consumption_wh
    )
    
    return {
        "model": model_name,
        "total_consumption": total_consumption_kwh, # in kWh
        "total_consumption_wh": round(total_energy_wh, 2),
        "peak_period": peak_period_data,
        "pattern": pattern_name,
        "recommendation": recommendation,
        "predictions": time_series_points
    }
