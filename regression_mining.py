import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")

def perform_regression():
    print("="*60)
    print("STEP 2: REGRESSION ANALYSIS & OPTIMIZATION")
    print("="*60)
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    train = pd.read_csv("training_cleaned.csv")
    test = pd.read_csv("testing_cleaned.csv")
    
    drop_cols = ['date', 'High_Energy_Usage', 'Appliances']
    X_train = train.drop(columns=drop_cols)
    y_train = train['Appliances'].values
    X_test = test.drop(columns=drop_cols)
    y_test = test['Appliances'].values
    
    # Subsample for tuning
    np.random.seed(42)
    idx = np.random.choice(len(X_train), size=3000, replace=False)
    X_train_ss = X_train.iloc[idx]
    y_train_ss = y_train[idx]
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 1. Simple Linear Regression (T_out only)
    print("Training Simple Linear Regression (T_out)...")
    simple_lm = LinearRegression()
    simple_lm.fit(X_train[['T_out']], y_train)
    y_pred_simple = simple_lm.predict(X_test[['T_out']])
    simple_rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
    simple_mae = mean_absolute_error(y_test, y_pred_simple)
    simple_r2 = r2_score(y_test, y_pred_simple)
    print(f"  RMSE: {simple_rmse:.2f}, MAE: {simple_mae:.2f}, R2: {simple_r2:.4f}")
    
    # 2. Multiple Linear Regression (all features)
    print("Training Multiple Linear Regression (all features)...")
    multi_lm = LinearRegression()
    multi_lm.fit(X_train, y_train)
    y_pred_multi = multi_lm.predict(X_test)
    multi_rmse = np.sqrt(mean_squared_error(y_test, y_pred_multi))
    multi_mae = mean_absolute_error(y_test, y_pred_multi)
    multi_r2 = r2_score(y_test, y_pred_multi)
    print(f"  RMSE: {multi_rmse:.2f}, MAE: {multi_mae:.2f}, R2: {multi_r2:.4f}")
    
    # 3. KNN Regressor with CV Tuning
    print("Tuning KNN Regressor...")
    k_range = list(range(1, 21, 2))
    grid_knn = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': k_range},
                            cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
    grid_knn.fit(X_train_ss, y_train_ss)
    best_k = grid_knn.best_params_['n_neighbors']
    print(f"  Optimal K = {best_k}")
    
    best_knn = KNeighborsRegressor(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred_knn = best_knn.predict(X_test)
    knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    knn_mae = mean_absolute_error(y_test, y_pred_knn)
    knn_r2 = r2_score(y_test, y_pred_knn)
    print(f"  RMSE: {knn_rmse:.2f}, MAE: {knn_mae:.2f}, R2: {knn_r2:.4f}")
    
    # KNN tuning plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(k_range, -grid_knn.cv_results_['mean_test_score'], marker='o', color='#2b5c8f', linewidth=2)
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Cross-Validated RMSE")
    plt.title("KNN Regressor Hyperparameter Tuning", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/knn_regression_tuning.png', bbox_inches='tight')
    plt.close()
    print("  Saved 'plots/knn_regression_tuning.png'")
    
    # 4. Decision Tree Regressor with Tuning
    print("Tuning Decision Tree Regressor...")
    grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42),
                           {'max_depth': [3, 5, 8, 10, 15, 20], 'min_samples_split': [2, 10, 20]},
                           cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
    grid_dt.fit(X_train_ss, y_train_ss)
    print(f"  Optimal params: {grid_dt.best_params_}")
    
    best_dt = DecisionTreeRegressor(**grid_dt.best_params_, random_state=42)
    best_dt.fit(X_train, y_train)
    y_pred_dt = best_dt.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    dt_mae = mean_absolute_error(y_test, y_pred_dt)
    dt_r2 = r2_score(y_test, y_pred_dt)
    print(f"  RMSE: {dt_rmse:.2f}, MAE: {dt_mae:.2f}, R2: {dt_r2:.4f}")
    
    # Performance table
    perf_df = pd.DataFrame({
        'Model': ['Simple LM', 'Multiple LM', f'KNN (K={best_k})', 'Decision Tree'],
        'RMSE': [simple_rmse, multi_rmse, knn_rmse, dt_rmse],
        'MAE': [simple_mae, multi_mae, knn_mae, dt_mae],
        'R-squared': [simple_r2, multi_r2, knn_r2, dt_r2]
    })
    print("\n--- Regression Performance ---")
    print(perf_df.to_string(index=False))
    perf_df.to_csv("regression_performance.csv", index=False)
    
    # Actual vs Predicted scatter
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    preds = {'Simple LM': y_pred_simple, 'Multiple LM': y_pred_multi,
             f'KNN (K={best_k})': y_pred_knn, 'Decision Tree': y_pred_dt}
    for i, (name, pred) in enumerate(preds.items()):
        ax = axes.flatten()[i]
        ax.scatter(y_test, pred, alpha=0.3, color='#34495e', s=8)
        ax.plot([0, 800], [0, 800], color='red', linestyle='--', linewidth=2)
        ax.set_xlim(0, 800); ax.set_ylim(0, 800)
        ax.set_xlabel("Actual (Wh)"); ax.set_ylabel("Predicted (Wh)")
        ax.set_title(f"{name}", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/regression_predictions_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/regression_predictions_comparison.png'")
    print("="*60 + "\n")
    return perf_df

if __name__ == "__main__":
    perform_regression()
