import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Styling
sns.set_theme(style="white")

def perform_clustering():
    print("="*60)
    print("STEP 4: CLUSTERING ANALYSIS (UNSUPERVISED SEGMENTATION)")
    print("="*60)
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load cleaned data
    train = pd.read_csv("training_cleaned.csv")
    
    # Define indoor environmental features (T1-T9 and RH_1-RH_9) for clustering
    clustering_features = [
        'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
        'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9'
    ]
    
    # Check column existence
    clustering_features = [c for c in clustering_features if c in train.columns]
    X_clustering = train[clustering_features]
    print(f"Clustering dataset dimensions: {X_clustering.shape}")
    
    # Subsample for evaluation scores (calculating silhouette score on 14800 points is very slow. We'll use 3000 rows)
    np.random.seed(42)
    idx = np.random.choice(len(X_clustering), size=3000, replace=False)
    X_ss = X_clustering.iloc[idx]
    
    # 1. K-Means Optimal Clusters Tuning (Elbow Method & Silhouette Score)
    print("Running K-Means cluster tuning (Elbow & Silhouette)...")
    cluster_range = list(range(2, 9))
    inertias = []
    sil_scores = []
    db_scores = []
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_ss)
        inertias.append(kmeans.inertia_)
        
        sil = silhouette_score(X_ss, kmeans.labels_)
        db = davies_bouldin_score(X_ss, kmeans.labels_)
        sil_scores.append(sil)
        db_scores.append(db)
        print(f"K-Means clusters: {n_clusters} -> Silhouette: {sil:.3f}, Davies-Bouldin: {db:.3f}")
        
    # Plot Elbow & Silhouette curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Elbow
    ax1.plot(cluster_range, inertias, marker='o', color='#34495e', linewidth=2)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)")
    ax1.set_title("Elbow Method for Optimal K", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Silhouette
    ax2.plot(cluster_range, sil_scores, marker='o', color='#9b59b6', linewidth=2)
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score Profile", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plots/kmeans_optimal_clusters.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/kmeans_optimal_clusters.png'")
    
    # Select optimal clusters (let's use 3 clusters for final segmentation based on house levels / environmental ranges)
    optimal_k = 3
    print(f"Fitting final K-Means with optimal K = {optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_final.fit(X_clustering)
    kmeans_labels = kmeans_final.labels_
    
    # 2. Hierarchical Agglomerative Clustering
    print("Running Hierarchical Agglomerative Clustering...")
    # Generate Linkage Matrix and Dendrogram (using a smaller sample of 150 points for visual clarity)
    X_dendro = X_ss.iloc[:150]
    linkage_matrix = linkage(X_dendro, method='ward')
    
    plt.figure(figsize=(12, 7), dpi=300)
    dendrogram(linkage_matrix, labels=X_dendro.index.astype(str), leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage, 150 Samples)", fontsize=14, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Ward Distance", fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/hierarchical_dendrogram.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/hierarchical_dendrogram.png'")
    
    # Fit Agglomerative Clustering on the full dataset (using 3 clusters for direct comparison)
    print(f"Fitting Agglomerative Clustering with 3 clusters...")
    # Agglomerative clustering can be slow on 14800 points due to O(N^2) memory.
    # To run efficiently, we will fit Agglomerative on our subsample X_ss and report scores.
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agg_labels_ss = agg.fit_predict(X_ss)
    
    # 3. Expectation-Maximization (EM) Clustering via Gaussian Mixture Models
    print("Running Gaussian Mixture Model (EM) cluster tuning...")
    bic_scores = []
    aic_scores = []
    components_range = list(range(2, 9))
    
    for n_comp in components_range:
        gmm = GaussianMixture(n_components=n_comp, random_state=42)
        gmm.fit(X_ss)
        bic_scores.append(gmm.bic(X_ss))
        aic_scores.append(gmm.aic(X_ss))
        
    # Plot BIC/AIC curves
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(components_range, bic_scores, label='BIC (Bayesian Info Criterion)', marker='o', color='#e74c3c')
    plt.plot(components_range, aic_scores, label='AIC (Akaike Info Criterion)', marker='o', color='#3498db')
    plt.xlabel("Number of Components", fontsize=12)
    plt.ylabel("Information Score", fontsize=12)
    plt.title("GMM Components Tuning (EM Clustering)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/gmm_bic_aic_tuning.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/gmm_bic_aic_tuning.png'")
    
    # Fit GMM with 3 components (for direct comparison)
    print("Fitting GMM with 3 components...")
    gmm_final = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_final.fit(X_clustering)
    gmm_labels = gmm_final.predict(X_clustering)
    
    # Compile performance comparison on subsampled data
    kmeans_labels_ss = kmeans_labels[idx]
    gmm_labels_ss = gmm_labels[idx]
    
    clustering_perf = {
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
        'K-Means': [silhouette_score(X_ss, kmeans_labels_ss), davies_bouldin_score(X_ss, kmeans_labels_ss)],
        'Hierarchical (Ward)': [silhouette_score(X_ss, agg_labels_ss), davies_bouldin_score(X_ss, agg_labels_ss)],
        'GMM (EM)': [silhouette_score(X_ss, gmm_labels_ss), davies_bouldin_score(X_ss, gmm_labels_ss)]
    }
    
    perf_df = pd.DataFrame(clustering_perf)
    print("\n--- Clustering Performance Table ---")
    print(perf_df.to_string(index=False))
    perf_df.to_csv("clustering_performance.csv", index=False)
    
    # 4. Project clusters to 2D PCA Space for visual representation
    print("Visualizing clustering assignments in 2D PCA projected space...")
    pca = PCA(n_components=2)
    X_clustering_pca = pca.fit_transform(X_clustering)
    
    # Create subplots grid for cluster visual comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # KMeans visual
    sns.scatterplot(x=X_clustering_pca[:, 0], y=X_clustering_pca[:, 1], hue=kmeans_labels,
                    palette='viridis', ax=ax1, s=15, alpha=0.6, legend='full')
    ax1.set_title("K-Means Clustering (K=3)", fontweight='bold', fontsize=14)
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    
    # GMM visual
    sns.scatterplot(x=X_clustering_pca[:, 0], y=X_clustering_pca[:, 1], hue=gmm_labels,
                    palette='viridis', ax=ax2, s=15, alpha=0.6, legend='full')
    ax2.set_title("GMM EM Clustering (Components=3)", fontweight='bold', fontsize=14)
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    
    plt.tight_layout()
    plt.savefig('plots/clustering_comparison_scatter.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/clustering_comparison_scatter.png'")
    print("="*60 + "\n")
    return perf_df

if __name__ == "__main__":
    perform_clustering()
