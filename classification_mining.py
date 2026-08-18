import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

sns.set_theme(style="white")

def perform_classification():
    print("="*60)
    print("STEP 3: CLASSIFICATION ANALYSIS & MODEL COMPARISONS")
    print("="*60)
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    train = pd.read_csv("training_cleaned.csv")
    test = pd.read_csv("testing_cleaned.csv")
    
    drop_cols = ['date', 'High_Energy_Usage', 'Appliances']
    X_train = train.drop(columns=drop_cols)
    y_train = train['High_Energy_Usage'].values
    X_test = test.drop(columns=drop_cols)
    y_test = test['High_Energy_Usage'].values
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Class distribution — 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")
    
    # Subsample for SVM/tuning speed
    np.random.seed(42)
    idx = np.random.choice(len(X_train), size=1500, replace=False)
    X_train_ss = X_train.iloc[idx]
    y_train_ss = y_train[idx]
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # ---- 1. Logistic Regression ----
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_score_log = log_reg.predict_proba(X_test)[:, 1]
    
    # ---- 2. KNN Classifier ----
    print("Tuning KNN Classifier...")
    k_range = list(range(1, 16, 2))
    grid_knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': k_range},
                            cv=cv, scoring='accuracy', n_jobs=1)
    grid_knn.fit(X_train_ss, y_train_ss)
    best_k = grid_knn.best_params_['n_neighbors']
    print(f"  Optimal K = {best_k}")
    
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred_knn = best_knn.predict(X_test)
    y_score_knn = best_knn.predict_proba(X_test)[:, 1]
    
    # KNN tuning plot
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(k_range, grid_knn.cv_results_['mean_test_score'], marker='o', color='#2ecc71', linewidth=2)
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Cross-Validated Accuracy")
    plt.title("KNN Classifier Hyperparameter Tuning", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/knn_classification_tuning.png', bbox_inches='tight')
    plt.close()
    print("  Saved 'plots/knn_classification_tuning.png'")
    
    # ---- 3. Decision Tree Classifier ----
    print("Tuning Decision Tree Classifier...")
    grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           {'max_depth': [3, 5, 8, 12, 15], 'min_samples_split': [2, 10, 20]},
                           cv=cv, scoring='accuracy', n_jobs=1)
    grid_dt.fit(X_train_ss, y_train_ss)
    print(f"  Optimal params: {grid_dt.best_params_}")
    
    best_dt = DecisionTreeClassifier(**grid_dt.best_params_, random_state=42)
    best_dt.fit(X_train, y_train)
    y_pred_dt = best_dt.predict(X_test)
    y_score_dt = best_dt.predict_proba(X_test)[:, 1]
    
    # ---- 4. Naive Bayes ----
    print("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_score_nb = nb.predict_proba(X_test)[:, 1]
    
    # ---- 5. SVM Classifier (NO probability=True — use decision_function for ROC) ----
    print("Training SVM Classifier...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_ss, y_train_ss)
    y_pred_svm = svm.predict(X_test)
    y_score_svm = svm.decision_function(X_test)  # Use decision_function, NOT predict_proba
    
    # ---- 6. Simple Perceptron ----
    print("Training Perceptron...")
    perceptron = Perceptron(max_iter=1000, random_state=42)
    perceptron.fit(X_train, y_train)
    y_pred_per = perceptron.predict(X_test)
    y_score_per = perceptron.decision_function(X_test)
    
    # ---- 7. MLP Classifier ----
    print("Training MLP Classifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    y_score_mlp = mlp.predict_proba(X_test)[:, 1]
    
    # ---- Evaluation ----
    models = {
        'Logistic Regression': (y_pred_log, y_score_log),
        'KNN Classifier':      (y_pred_knn, y_score_knn),
        'Decision Tree':       (y_pred_dt, y_score_dt),
        'Naive Bayes':         (y_pred_nb, y_score_nb),
        'SVM Classifier':      (y_pred_svm, y_score_svm),
        'Perceptron':          (y_pred_per, y_score_per),
        'MLP Classifier':      (y_pred_mlp, y_score_mlp),
    }
    
    perf_results = []
    
    # Confusion matrix grid
    fig_cm, axes_cm = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes_flat = axes_cm.flatten()
    
    # ROC figure
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8), dpi=300)
    
    for i, (name, (pred, score)) in enumerate(models.items()):
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, score)
        
        fpr, tpr, _ = roc_curve(y_test, score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        
        perf_results.append({
            'Model': name, 'Accuracy': round(acc, 4),
            'Precision': round(prec, 4), 'Recall': round(rec, 4),
            'F1-score': round(f1, 4), 'ROC-AUC': round(auc, 4)
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes_flat[i], cbar=False)
        axes_flat[i].set_title(name, fontweight='bold', fontsize=11)
        axes_flat[i].set_xlabel('Predicted')
        axes_flat[i].set_ylabel('Actual')
    
    # Remove unused subplot cells
    for j in range(len(models), len(axes_flat)):
        fig_cm.delaxes(axes_flat[j])
    
    fig_cm.tight_layout()
    fig_cm.savefig('plots/confusion_matrices_panel.png', bbox_inches='tight')
    plt.close(fig_cm)
    print("Saved 'plots/confusion_matrices_panel.png'")
    
    # Finalize ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves — All Classifiers', fontweight='bold')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle='--', alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig('plots/classification_roc_curves.png', bbox_inches='tight')
    plt.close(fig_roc)
    print("Saved 'plots/classification_roc_curves.png'")
    
    perf_df = pd.DataFrame(perf_results)
    print("\n--- Classification Performance ---")
    print(perf_df.to_string(index=False))
    perf_df.to_csv("classification_performance.csv", index=False)
    
    # ---- SVM Decision Boundary in 2D PCA space ----
    print("Visualizing SVM decision boundary...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_ss)
    
    svm_2d = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_2d.fit(X_pca, y_train_ss)
    
    h = 0.1
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_ss,
               cmap=plt.cm.coolwarm, edgecolors='k', s=25, alpha=0.7)
    svs = svm_2d.support_vectors_
    ax.scatter(svs[:, 0], svs[:, 1], s=80, facecolors='none',
               edgecolors='black', linewidths=1.5, label='Support Vectors')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('SVM Decision Boundary & Support Vectors (2D PCA)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plots/svm_decision_boundary.png', bbox_inches='tight')
    plt.close()
    print("Saved 'plots/svm_decision_boundary.png'")
    print("="*60 + "\n")
    return perf_df

if __name__ == "__main__":
    perform_classification()
