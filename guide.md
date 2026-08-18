# User Guide and Execution Manual

This guide describes how to run, configure, and customize the Appliances Energy Prediction data mining scripts.

---

## 1. Running the Complete Pipeline

The easiest way to execute the entire data mining flow is through the master orchestrator script [run_data_mining.py](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/run_data_mining.py).

Activate your virtual environment and run:
```bash
python run_data_mining.py
```

### What this script does:
1. **Runs Preprocessing**: Parses raw measurements, handles outliers, computes standard statistics, drops noise, and partitions the dataset into `training_cleaned.csv` and `testing_cleaned.csv`.
2. **Runs Regression**: Trains models (Linear, KNN, Decision Tree) to predict the continuous appliance energy usage (in Wh) and evaluates metrics.
3. **Runs Classification**: Creates a binary target for high-energy consumption ($\ge 100$ Wh), trains classifiers (Logistic, KNN, Tree, NB, SVM, Perceptron, MLP), and generates prediction heatmaps.
4. **Runs Clustering**: Groups environmental data using K-Means, Agglomerative Hierarchical, and GMM algorithms, saving silhouettes and indices.

---

## 2. Running Individual Modules

You can execute any of the modules separately to inspect specific phases or customize the hyperparameter grids.

### A. Preprocessing & Outlier Auditing
Run the preprocessing script to inspect descriptive stats and outlier rates:
```bash
python preprocessing.py
```
* **Key Outputs**: `training_cleaned.csv`, `testing_cleaned.csv`, and `descriptive_statistics.csv`.

### B. Regression Modeling
Run the regression module to perform hyperparameter tuning of $K$ and tree depths:
```bash
python regression_mining.py
```
* **Key Outputs**:
  - `regression_performance.csv` (scores table)
  - `knn_regression_tuning.png` (tuning plot)
  - `regression_predictions_comparison.png` (actual vs. predicted scatter plots)

### C. Classification Modeling
Run the classification module to perform classifier audits and decision boundary mapping:
```bash
python classification_mining.py
```
* **Key Outputs**:
  - `classification_performance.csv` (scores table)
  - `knn_classification_tuning.png`
  - `confusion_matrices_panel.png` (7-panel confusion heatmaps)
  - `classification_roc_curves.png` (overlapping ROC lines)
  - `svm_decision_boundary.png` (2D PCA SVM boundary plot)

### D. Clustering Analysis
Run the clustering module to perform environmental segmentation:
```bash
python clustering_mining.py
```
* **Key Outputs**:
  - `clustering_performance.csv` (scores table)
  - `kmeans_optimal_clusters.png` (Elbow and Silhouette tuning curves)
  - `hierarchical_dendrogram.png` (Ward linkage dendrogram)
  - `gmm_bic_aic_tuning.png` (AIC/BIC curves)
  - `clustering_comparison_scatter.png` (2D PCA cluster comparison scatters)

---

## 3. Interactive Analysis (Jupyter Notebook)

For interactive data exploration and visual inspection, open the Jupyter notebook:

1. **Start the Jupyter Notebook server**:
   ```bash
   jupyter notebook
   ```
2. **Open the notebook file**:
   Select and open [data_mining_notebook.ipynb](file:///c:/Users/likhi/Desktop/Appliances-energy-prediction-data/data_mining_notebook.ipynb) in your browser.
3. **Execute cells**: Run each cell step by step to load libraries, execute scripts, and render the generated plots inline.

---

## 4. Customizing Parameters and Ensuring Reproducibility

- **Reproducibility**: All scripts enforce seed variables (`random_state=42` and `random_state=1` for partitions) to guarantee that cross-validation splits and stochastic training (e.g. MLP, SVM, KMeans) yield reproducible results.
- **Adjusting Subsamples**: In `classification_mining.py` and `clustering_mining.py`, tuning functions utilize random subsampling (e.g., $N=1500$ or $N=3000$) to expedite grid search and silhouette computations. You can modify these size thresholds inside the python files to scale up execution on high-performance machines.
