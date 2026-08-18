# Appliances Energy Prediction: Data Mining Project Explanation

This document provides a comprehensive, step-by-step explanation of the codebase, a project overview, the target objective (what the project is used to obtain), and a detailed performance analysis of all trained models.

---

## 1. Project Overview

The **Appliances Energy Prediction** project utilizes a dataset containing 10-minute interval readings of temperature and humidity from wireless sensor networks placed in different rooms of a low-energy house, along with weather parameters from a nearby airport station (Chièvres, Belgium). The data spans a period of approximately 4 months.

### Objective and Goals
The primary objective of this project is to apply core data mining and machine learning paradigms to analyze, predict, and segment energy usage patterns. The pipeline achieves this through:
1. **Data Preprocessing & Descriptive Statistics**: Standardizing raw measurements, engineering temporal features, detecting outliers, and handling noise variables.
2. **Regression Analysis**: Predicting the continuous quantity of energy consumed by appliances (in Watt-hours, Wh).
3. **Classification Analysis**: Identifying peak energy events (binary variable `High_Energy_Usage` $\ge$ 100 Wh) based on environmental states.
4. **Clustering Analysis**: Discovering natural indoor environmental states by grouping indoor temperatures and humidities.

---

## 2. Step-by-Step Code Explanation

The project is structured into five Python modules:
1. [preprocessing.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/preprocessing.py) — Prepares and cleans the dataset.
2. [regression_mining.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/regression_mining.py) — Runs continuous value predictions.
3. [classification_mining.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/classification_mining.py) — Analyzes high energy usage events.
4. [clustering_mining.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/clustering_mining.py) — Clusters environmental measurements.
5. [run_data_mining.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/run_data_mining.py) — Coordinates the execution of the entire pipeline.

### Preprocessing (`preprocessing.py`)

This file is responsible for cleaning, transforming, and partitioning the raw dataset.

- **Importing Dependencies & Loading Data** (Lines 1-18): 
  Loads `energydata_complete.csv` using Pandas. It checks for file existence and reports the raw shape of the loaded DataFrame.
- **Descriptive Statistics** (Lines 20-35):
  Iterates over every numerical feature to calculate the **Mean**, **Median**, **Mode**, and **Standard Deviation**. These metrics are compiled into a DataFrame and exported to `descriptive_statistics.csv`.
  > [!NOTE]
  > Standard deviation indicates feature variance, helping check if features are on widely different scales. Mode and median are robust measures of central tendency, highlighting potential skewness in variables like `Appliances`.
- **Outlier Detection using Interquartile Range (IQR)** (Lines 37-48):
  Computes the first quartile ($Q_1$, 25th percentile) and third quartile ($Q_3$, 75th percentile) of the target variable `Appliances`. The IQR is computed as:
  \[IQR = Q_3 - Q_1\]
  The bounds are set to:
  \[\text{Lower Bound} = Q_1 - 1.5 \times IQR\]
  \[\text{Upper Bound} = Q_3 + 1.5 \times IQR\]
  Points outside this interval are flagged as outliers. Outliers represent abnormal surge events in energy consumption (e.g., running multiple heavy appliances simultaneously).
- **Feature Engineering** (Lines 49-58):
  - Converts the `date` string column to a Pandas `datetime` object.
  - Computes `NSM` (Number of Seconds from Midnight), ranging from $0$ to $86,400$. This captures daily cyclical behavior (e.g., higher energy usage in evenings vs nights).
  - Creates `WeekStatus` to classify days as `Weekday` or `Weekend` (where Monday is index 0 and Sunday is index 6).
  - Extracts the exact `Day_of_week` name.
  - Drops features `rv1` and `rv2`. These are random numeric noise variables introduced in the original research to test feature selection algorithms; they contain no explanatory power.
- **Standard Scaling** (Lines 59-64):
  Normalizes all environmental columns (temperatures `T1` to `T9`, humidities `RH_1` to `RH_9`, pressure, windspeed, visibility, etc.) using `StandardScaler`:
  \[z = \frac{x - \mu}{\sigma}\]
  This scales features to have $\mu=0$ and $\sigma=1$, which is critical for distance-sensitive algorithms like K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP).
- **One-Hot Encoding** (Lines 65-70):
  Converts categorical features (`WeekStatus` and `Day_of_week`) into binary dummy columns (e.g., `WeekStatusWeekday`, `WeekStatusWeekend`).
- **Binary Classification Target Creation** (Lines 71-72):
  Creates `High_Energy_Usage` by thresholding `Appliances` at 100 Wh:
  \[\text{High\_Energy\_Usage} = \begin{cases} 1 & \text{if Appliances} \ge 100 \\ 0 & \text{otherwise} \end{cases}\]
- **Train/Test Stratified Split** (Lines 74-87):
  Splits the dataset into 75% training and 25% testing sets. It uses stratified splitting based on `High_Energy_Usage` to preserve the proportion of high-energy usage events in both partitions. Setting `random_state=1` guarantees identical splits across runs. The sets are exported as `training_cleaned.csv` and `testing_cleaned.csv`.

---

### Regression Analysis (`regression_mining.py`)

This file predicts the continuous variable `Appliances` (Wh) representing energy consumption.

- **Data Loading & Feature Selection** (Lines 23-30):
  Loads the train/test sets, drops non-predictive columns (`date`), classification-specific targets (`High_Energy_Usage`), and separates the target variable `Appliances` ($y$) from predictors ($X$).
- **Subsampling** (Lines 32-36):
  Extracts a random subset of 3,000 samples for cross-validation hyperparameter tuning to ensure reasonable execution times.
- **Simple Linear Regression** (Lines 40-48):
  Fits a baseline 1D linear model using only the outside temperature (`T_out`) to predict appliance energy use. This model establishes a reference performance.
- **Multiple Linear Regression** (Lines 50-58):
  Fits a multivariate Ordinary Least Squares (OLS) model using all features.
- **K-Nearest Neighbors (KNN) Regressor & Hyperparameter Tuning** (Lines 60-87):
  Tunes the parameter $K$ (number of neighbors) across odd values from 1 to 19 using 3-fold cross-validation on the 3,000-sample subset. The score metric is Negative Root Mean Squared Error.
  - The tuning curve is plotted and saved to `knn_regression_tuning.png`.
  - The model is re-fitted on the **entire** training set using the optimal $K$ value ($K=19$), and predictions are generated on the test set.
- **Decision Tree Regressor Tuning** (Lines 89-103):
  Uses GridSearchCV to tune `max_depth` ($[3, 5, 8, 10, 15, 20]$) and `min_samples_split` ($[2, 10, 20]$) on the subsample. The best tree parameters are selected, and the final model is trained on the full training set.
- **Evaluation & Visualizations** (Lines 105-133):
  Computes RMSE, MAE, and $R^2$ scores for all four models, saves them to `regression_performance.csv`, and plots actual vs. predicted values for each model as panels in `regression_predictions_comparison.png` with a reference $y=x$ line.

---

### Classification Analysis (`classification_mining.py`)

This file trains classifiers to predict binary events where energy consumption spikes ($\ge 100$ Wh).

- **Data Setup & SVM Subsampling** (Lines 30-49):
  Loads the split sets and extracts a random subset of 1,500 training points. Subsampling is necessary because SVMs have a training complexity of $\mathcal{O}(N^3)$ and can be computationally expensive on larger datasets.
- **Logistic Regression** (Lines 50-55):
  Fits a regularized logistic model with a maximum of 5,000 iterations using the `lbfgs` solver.
- **KNN Classifier Tuning** (Lines 57-81):
  Tunes $K$ (from 1 to 15, step size 2) via Grid Search. The CV accuracy curve is plotted and saved to `knn_classification_tuning.png`. The optimal $K$ classifier is then fitted on the full training set.
- **Decision Tree Classifier Tuning** (Lines 83-94):
  Tunes `max_depth` and `min_samples_split` parameters using Grid Search, then fits the optimal tree model on the full training set.
- **Naive Bayes** (Lines 96-101):
  Trains a Gaussian Naive Bayes model. This algorithm assumes that features are conditionally independent given the class label.
- **Support Vector Machine (SVM) Classifier** (Lines 103-109):
  Fits a Support Vector Classifier with a Radial Basis Function (RBF) kernel on the 1,500 training samples. The model outputs decision function scores (margins) instead of calibrated probabilities to compute the ROC curve.
- **Perceptron** (Lines 110-115):
  Fits a baseline Perceptron (single-layer neural network) using stochastic gradient descent.
- **Multi-Layer Perceptron (MLP) Classifier** (Lines 117-122):
  Trains a feedforward artificial neural network with two hidden layers containing 32 and 16 neurons respectively, with a maximum of 200 epochs.
- **Evaluation Metrics and Heatmaps** (Lines 124-192):
  Calculates Accuracy, Precision, Recall, F1-score, and ROC-AUC. It generates a grid of confusion matrices saved as `confusion_matrices_panel.png` and plots overlaying ROC curves (True Positive Rate vs. False Positive Rate) saved as `classification_roc_curves.png`.
- **SVM 2D Boundary Projection** (Lines 194-222):
  Uses Principal Component Component (PCA) to reduce the 1,500-sample training dataset to two dimensions (`PC1` and `PC2`). A 2D SVM is fitted, and a grid boundary contour plot is saved as `svm_decision_boundary.png`, overlaying data points and marking support vectors with black circles.

---

### Clustering Analysis (`clustering_mining.py`)

This file uses unsupervised learning algorithms to group indoor temperature and humidity variables (`T1`-`T9` and `RH_1`-`RH_9`).

- **Feature Isolation & Subsampling** (Lines 19-36):
  Selects the 18 indoor sensors columns. It draws a random subset of 3,000 rows to calculate distance-based evaluation metrics like the Silhouette Score.
- **K-Means Cluster Tuning** (Lines 38-76):
  Tunes the number of clusters $K$ from 2 to 8. It evaluates:
  1. **Inertia**: The sum of squared distances of samples to their closest cluster center (lower is better, used for the Elbow method).
  2. **Silhouette Score**: Measures how cohesive clusters are and how well separated they are (higher is better).
  3. **Davies-Bouldin Index**: Measures the ratio of within-cluster distances to between-cluster distances (lower is better).
  The tuning profile curves are plotted in `kmeans_optimal_clusters.png`.
- **Fitting K-Means** (Lines 78-83):
  Fits the final K-Means model with $K=3$ clusters.
- **Hierarchical Agglomerative Clustering** (Lines 85-106):
  Generates a linkage matrix using Ward's minimum variance criterion on a small subset of 150 points and saves a dendrogram diagram (`hierarchical_dendrogram.png`). Fits Agglomerative Clustering with $K=3$ on the 3,000-sample subset.
- **Gaussian Mixture Model (GMM) Component Tuning** (Lines 108-139):
  Evaluates GMM performance for components 2 to 8 using information criteria:
  1. **BIC** (Bayesian Information Criterion): Penalizes model complexity more heavily to prevent overfitting.
  2. **AIC** (Akaike Information Criterion): Evaluates model goodness-of-fit.
  Plots these curves in `gmm_bic_aic_tuning.png`. Fits a final GMM with 3 components on the full environmental dataset.
- **Clustering Evaluation and Visual Projection** (Lines 141-182):
  Saves Silhouette and Davies-Bouldin scores to `clustering_performance.csv`. It projects the 18 environmental columns into a 2D space using PCA and plots the cluster assignments of K-Means vs. GMM as side-by-side scatter plots in `clustering_comparison_scatter.png`.

---

### Master Execution Pipeline (`run_data_mining.py`)

This script manages the execution order, runs each module sequentially, logs execution details, and tracks total run time.

- **Pipeline Execution** (Lines 20-37):
  1. Imports `preprocessing` and calls `perform_preprocessing()`, producing cleaned datasets.
  2. Imports `regression_mining` and calls `perform_regression()`, producing regression outputs.
  3. Imports `classification_mining` and calls `perform_classification()`, producing classification outputs.
  4. Imports `clustering_mining` and calls `perform_clustering()`, producing clustering outputs.
- **Execution Log Output** (Lines 39-60):
  Computes the elapsed time and lists all generated `.csv` data reports and `.png` visualization plots.

---

## 3. Real-World Applications

This pipeline is designed to extract actionable insights from smart home sensor networks:

1. **Intelligent Home Energy Management Systems (HEMS)**:
   By predicting appliance energy consumption ($y$) based on temperature and humidity, HEMS can schedule appliance operations during periods when heating/cooling demands are low, optimizing overall power usage.
2. **Demand Response and Peak Shaving**:
   Predicting binary peak energy events ($\ge 100$ Wh) allows energy utility companies and smart grids to issue load-shedding commands or prompt home-owners to shift high-power tasks (like laundry or dishwashing) to off-peak periods.
3. **HVAC Control Optimization**:
   Clustering environmental readings identifies typical weather patterns and indoor thermal zones. Home heating and cooling units can dynamically adjust target temperatures based on the identified cluster, preventing unnecessary energy consumption.
4. **Thermal Anomaly Detection**:
   If actual energy usage deviates significantly from the regression model's predictions (high residual errors), it may indicate appliance degradation or insulation issues.

---

## 4. Performance Analysis

### A. Regression Model Performance
The objective of the regression task is to predict the exact energy consumption (`Appliances` Wh). The evaluation metrics obtained from the test set are summarized below:

| Model | RMSE (Wh) | MAE (Wh) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Simple Linear Regression ($T_{out}$ only)** | 98.90 | 59.34 | 0.0076 |
| **Multiple Linear Regression (All features)** | 89.59 | 51.07 | 0.1856 |
| **KNN Regressor ($K=19$)** | 92.37 | 50.79 | 0.1343 |
| **Decision Tree Regressor** | 91.80 | 51.46 | 0.1450 |

#### Insights:
- **Outside Temperature ($T_{out}$) alone** has minimal predictive power ($R^2 = 0.0076$). This indicates that outside weather conditions by themselves cannot explain internal appliance usage patterns.
- **Multiple Linear Regression** yields the best performance with an $R^2$ of $0.1856$ (explaining $18.56\%$ of the variance in energy usage). This shows that combining indoor climate variables with engineered temporal features (like time of day via `NSM` and day of week) provides better predictive capability.
- **KNN and Decision Tree** models perform slightly worse than multiple linear regression in terms of $R^2$, but show comparable MAE values (~51 Wh).
- The overall low $R^2$ scores across all models (max 18.56%) indicate that appliance energy consumption is highly stochastic. It is heavily influenced by human behavior (e.g., turning on television sets, cooking, or using hair dryers) which is not captured by temperature and humidity sensors.

---

### B. Classification Model Performance
The classification task targets the binary label `High_Energy_Usage` (energy consumption $\ge 100$ Wh). The classification metrics are summarized below:

| Classifier Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7864 | 0.6562 | 0.4014 | 0.4981 | 0.8252 |
| **KNN Classifier** | 0.7912 | 0.6518 | 0.4497 | 0.5322 | 0.8313 |
| **Decision Tree Classifier** | **0.8622** | **0.7432** | **0.7306** | **0.7368** | **0.9068** |
| **Gaussian Naive Bayes** | 0.7270 | 0.4820 | 0.4513 | 0.4661 | 0.7674 |
| **SVM Classifier** | 0.7359 | 0.0000 | 0.0000 | 0.0000 | 0.8101 |
| **Perceptron** | 0.2720 | 0.2661 | 0.9992 | 0.4203 | 0.7293 |
| **MLP Classifier** | 0.7343 | 0.0000 | 0.0000 | 0.0000 | 0.2957 |

#### Insights:
- **Decision Tree Classifier** is the best model, achieving **86.22% Accuracy**, **73.68% F1-score**, and an **ROC-AUC of 0.9068**. Its non-linear structure is well-suited for capturing threshold-based logic (e.g., time-of-day combined with specific room temperature thresholds).
- **Logistic Regression and KNN** show moderate classification performance with ROC-AUCs of ~0.82-0.83.
- **SVM Classifier** and **MLP Classifier** output 0.00% Precision and Recall. This occurs because the models predict the majority class (label `0`, representing low energy usage) for all test samples. This behavior is common in imbalanced datasets (where class `0` represents ~73.6% of the data) when models are trained on smaller subsamples without adjusted class weights or specific hyperparameter tuning.
- **Perceptron** predicts the positive class (label `1`, representing high energy usage) for nearly all samples. This leads to a high recall (99.92%) but low overall accuracy (27.20%).

---

### C. Clustering Model Performance
Unsupervised clustering was applied to indoor climate features (`T1`-`T9`, `RH_1`-`RH_9`). The performance metrics are summarized below:

| Clustering Algorithm | Silhouette Score | Davies-Bouldin Index |
| :--- | :---: | :---: |
| **K-Means Clustering ($K=3$)** | **0.2911** | **1.1763** |
| **Hierarchical Agglomerative ($K=3$)** | 0.2556 | 1.3054 |
| **Gaussian Mixture Model ($K=3$)** | 0.1757 | 1.5555 |

#### Insights:
- **K-Means clustering** achieves the highest Silhouette Score (0.2911) and the lowest Davies-Bouldin Index (1.1763). This suggests that K-Means creates more cohesive and well-separated environmental clusters than GMM or Hierarchical clustering on this dataset.
- **Hierarchical (Ward) clustering** shows competitive metrics, confirming that a distance-based, variance-minimizing partition structure is appropriate for the climate data.
- **GMM (EM)** shows a lower Silhouette Score (0.1757) and a higher Davies-Bouldin Index (1.5555). Unlike K-Means, which assumes spherical clusters, GMM fits multi-dimensional ellipsoids with varying covariances. This flexibility can lead to overlapping boundaries in PCA-projected space, resulting in lower distance-based clustering metrics.
- The 3 identified clusters correspond to typical climate profiles:
  - **Cluster 0**: High-temperature, low-humidity periods (e.g., warm afternoons or heated rooms).
  - **Cluster 1**: Cold-temperature, high-humidity periods (e.g., winter days, early mornings, or unheated spaces).
  - **Cluster 2**: Transitional periods with moderate temperatures and humidities.
