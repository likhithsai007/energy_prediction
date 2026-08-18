import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd

def main():
    print("="*80)
    print("            DATA MINING PROJECT: APPLIANCES ENERGY PREDICTION")
    print("="*80)
    print("This pipeline implements all required Data Mining phases:")
    print("1. Data Preprocessing (Descriptive statistics, outlier detection, scaling, encoding)")
    print("2. Regression Analysis (Linear, KNN, Decision Tree Regressors, tuning)")
    print("3. Classification Analysis (Logistic, KNN, Decision Tree, Naive Bayes, SVM, Perceptron, MLP)")
    print("4. Clustering Analysis (K-Means, Hierarchical Dendrogram, GMM EM, metrics)")
    print("="*80)
    
    start_total = time.time()
    
    # Step 1: Preprocessing
    print("\n--- Running Step 1: Preprocessing ---")
    import preprocessing
    train_df, test_df = preprocessing.perform_preprocessing()
    
    # Step 2: Regression
    print("\n--- Running Step 2: Regression ---")
    import regression_mining
    reg_perf = regression_mining.perform_regression()
    
    # Step 3: Classification
    print("\n--- Running Step 3: Classification ---")
    import classification_mining
    class_perf = classification_mining.perform_classification()
    
    # Step 4: Clustering
    print("\n--- Running Step 4: Clustering ---")
    import clustering_mining
    cluster_perf = clustering_mining.perform_clustering()
    
    total_time = time.time() - start_total
    
    print("\n" + "="*80)
    print(f"  PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("="*80)
    print("\nGenerated CSV Reports:")
    print("  - descriptive_statistics.csv")
    print("  - regression_performance.csv")
    print("  - classification_performance.csv")
    print("  - clustering_performance.csv")
    print("\nGenerated Visualizations:")
    print("  - knn_regression_tuning.png")
    print("  - regression_predictions_comparison.png")
    print("  - knn_classification_tuning.png")
    print("  - confusion_matrices_panel.png")
    print("  - classification_roc_curves.png")
    print("  - svm_decision_boundary.png")
    print("  - kmeans_optimal_clusters.png")
    print("  - hierarchical_dendrogram.png")
    print("  - gmm_bic_aic_tuning.png")
    print("  - clustering_comparison_scatter.png")
    print("="*80)

if __name__ == "__main__":
    main()
