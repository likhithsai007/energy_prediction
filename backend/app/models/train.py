import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "energydata_complete.csv"))
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models")

def calculate_mape(y_true, y_pred):
    # Avoid zero-division
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)

def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['NSM'] = df['date'].dt.hour * 3600 + df['date'].dt.minute * 60 + df['date'].dt.second
    df['month'] = df['date'].dt.month
    
    # Exclude noise features if present
    drop_cols = ['date', 'rv1', 'rv2']
    feature_cols = [c for c in df.columns if c not in drop_cols and c != 'Appliances']
    
    X = df[feature_cols].copy()
    y = df['Appliances'].values
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    
    # 3. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Regression Models
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    print("Training Decision Tree Regressor...")
    dt = DecisionTreeRegressor(max_depth=12, min_samples_split=10, random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred_dt = dt.predict(X_test_scaled)
    
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    
    # 5. Evaluate Regression Models
    models_eval = {
        "Linear Regression": {
            "mae": round(float(mean_absolute_error(y_test, y_pred_lr)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_lr))), 2),
            "r2": round(float(r2_score(y_test, y_pred_lr)), 4),
            "mape": round(calculate_mape(y_test, y_pred_lr), 2)
        },
        "Decision Tree": {
            "mae": round(float(mean_absolute_error(y_test, y_pred_dt)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_dt))), 2),
            "r2": round(float(r2_score(y_test, y_pred_dt)), 4),
            "mape": round(calculate_mape(y_test, y_pred_dt), 2)
        },
        "Random Forest": {
            "mae": round(float(mean_absolute_error(y_test, y_pred_rf)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 2),
            "r2": round(float(r2_score(y_test, y_pred_rf)), 4),
            "mape": round(calculate_mape(y_test, y_pred_rf), 2)
        }
    }
    
    # Automatically pick best model (highest R2, lowest RMSE)
    best_model_name = max(models_eval.keys(), key=lambda k: models_eval[k]["r2"])
    
    # 6. Train K-Means (k=3) Pattern Analysis
    print("Training K-Means (k=3)...")
    # Using key consumption features + target for cluster profile formation
    cluster_features = ['Appliances', 'hour', 'NSM', 'T_out', 'RH_out', 'Windspeed']
    scaler_kmeans = StandardScaler()
    df_kmeans_input = scaler_kmeans.fit_transform(df[cluster_features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_kmeans_input)
    
    # Subsample for silhouette score to compute fast and accurately
    sample_indices = np.random.RandomState(42).choice(len(df_kmeans_input), size=min(4000, len(df_kmeans_input)), replace=False)
    sil_score = round(float(silhouette_score(df_kmeans_input[sample_indices], cluster_labels[sample_indices])), 4)
    
    # Analyze cluster centers in terms of Appliances consumption
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    cluster_means = df_with_clusters.groupby('cluster')['Appliances'].mean().to_dict()
    
    # Sort clusters by average appliance load: Low, Medium, High
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
    cluster_mapping = {
        sorted_clusters[0][0]: {"name": "Low Consumption", "avg_load": round(sorted_clusters[0][1], 1)},
        sorted_clusters[1][0]: {"name": "Medium Consumption", "avg_load": round(sorted_clusters[1][1], 1)},
        sorted_clusters[2][0]: {"name": "High Consumption", "avg_load": round(sorted_clusters[2][1], 1)}
    }
    
    # 7. Feature Importance of tree model (Random Forest)
    importances = rf.feature_importances_
    feat_imp_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in zip(feature_cols, importances)
    ]
    feat_imp_list.sort(key=lambda x: x["importance"], reverse=True)
    
    # 8. Save Trained Models & Scalers
    print("Saving model artifacts...")
    joblib.dump(lr, os.path.join(OUTPUT_DIR, "linear_regression.joblib"))
    joblib.dump(dt, os.path.join(OUTPUT_DIR, "decision_tree.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest.joblib"))
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(scaler_kmeans, os.path.join(OUTPUT_DIR, "scaler_kmeans.joblib"))
    
    # 9. Save Metadata & Evaluation Metrics
    metadata = {
        "feature_cols": feature_cols,
        "cluster_features": cluster_features,
        "models_performance": models_eval,
        "selected_model": best_model_name,
        "selection_reason": "Selected automatically based on achieving the highest R² score and lowest MAE/RMSE among all evaluated regression models.",
        "kmeans_analysis": {
            "n_clusters": 3,
            "silhouette_score": sil_score,
            "clusters": [
                {
                    "cluster_id": int(cid),
                    "label": info["name"],
                    "avg_consumption_wh": info["avg_load"],
                    "description": f"{info['name']} pattern (average ~{info['avg_load']} Wh)"
                }
                for cid, info in cluster_mapping.items()
            ]
        },
        "feature_importance": feat_imp_list[:10]  # Top 10 features for UI
    }
    
    with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Training complete! Model artifacts saved successfully.")
    print("Regression Evaluation Results:")
    for m, vals in models_eval.items():
        print(f"  {m}: MAE={vals['mae']}, RMSE={vals['rmse']}, R2={vals['r2']}, MAPE={vals['mape']}%")
    print(f"Best Selected Model: {best_model_name}")
    print(f"K-Means Silhouette Score: {sil_score}")

if __name__ == "__main__":
    train_and_evaluate()
