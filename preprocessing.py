import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def perform_preprocessing():
    print("="*60)
    print("STEP 1: DATA PREPROCESSING & DESCRIPTIVE STATISTICS")
    print("="*60)
    
    csv_path = "energydata_complete.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file '{csv_path}' in workspace.")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 1. Descriptive Statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_summary = []
    for col in numerical_cols:
        stats_summary.append({
            'Feature': col,
            'Mean': round(df[col].mean(), 4),
            'Median': round(df[col].median(), 4),
            'Mode': round(df[col].mode()[0], 4),
            'Std Dev': round(df[col].std(), 4)
        })
    stats_df = pd.DataFrame(stats_summary)
    print("\n--- Descriptive Statistics (First 10 Features) ---")
    print(stats_df.head(10).to_string(index=False))
    stats_df.to_csv("descriptive_statistics.csv", index=False)
    print("Saved 'descriptive_statistics.csv'")
    
    # 2. Outlier Detection using IQR on Appliances
    q1 = df['Appliances'].quantile(0.25)
    q3 = df['Appliances'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df['Appliances'] < lower_bound) | (df['Appliances'] > upper_bound)]
    print(f"\n--- Outlier Detection (IQR) on 'Appliances' ---")
    print(f"Q1={q1}, Q3={q3}, IQR={iqr}")
    print(f"Bounds: [{lower_bound}, {upper_bound}]")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    # 3. Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['NSM'] = df['date'].dt.hour * 3600 + df['date'].dt.minute * 60 + df['date'].dt.second
    df['WeekStatus'] = df['date'].dt.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['Day_of_week'] = df['date'].dt.day_name()
    
    # Drop noise variables rv1 and rv2
    if 'rv1' in df.columns:
        df = df.drop(columns=['rv1', 'rv2'])
    
    # 4. Standard Scaling
    env_cols = [c for c in df.columns if c not in ['date', 'Appliances', 'WeekStatus', 'Day_of_week', 'NSM']]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[env_cols] = scaler.fit_transform(df[env_cols])
    
    # 5. One-Hot Encoding
    df_encoded = pd.get_dummies(df_scaled, columns=['WeekStatus', 'Day_of_week'],
                                prefix=['WeekStatus', 'Day_of_week'], prefix_sep='')
    dummy_cols = [c for c in df_encoded.columns if c.startswith('WeekStatus') or c.startswith('Day_of_week')]
    df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(float)
    
    # 6. Binary Classification Target
    df_encoded['High_Energy_Usage'] = (df_encoded['Appliances'] >= 100).astype(int)
    
    # 7. Train/Test Split
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        df_encoded, test_size=0.25, random_state=1,
        stratify=df_encoded['High_Energy_Usage']
    )
    
    train_data.to_csv("training_cleaned.csv", index=False)
    test_data.to_csv("testing_cleaned.csv", index=False)
    
    print(f"\nTrain shape: {train_data.shape}, Test shape: {test_data.shape}")
    print("Saved 'training_cleaned.csv' and 'testing_cleaned.csv'")
    print("="*60 + "\n")
    return train_data, test_data

if __name__ == "__main__":
    perform_preprocessing()
