"""
The Role of AI and Bioinformatics in Early Disease Diagnosis
Project: Breast Cancer Diagnosis Classification using ML
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) via scikit-learn
Author: Keerat Singh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("outputs", exist_ok=True)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

RANDOM_SEED = 42
OUT_DIR = "outputs"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Class counts (0=malignant, 1=benign):")
print(y.value_counts())

# ---------------------------------------------------------------
# 2. EDA PLOTS
# ---------------------------------------------------------------

# 2a. Class balance
plt.figure(figsize=(5, 4))
counts = y.value_counts().rename({0: "Malignant", 1: "Benign"})
sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
            palette=["#d62728", "#2ca02c"], legend=False)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_balance.png")
plt.close()

# 2b. Correlation heatmap (mean features only, for readability)
mean_features = [c for c in X.columns if "mean" in c]
plt.figure(figsize=(10, 8))
sns.heatmap(X[mean_features].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Mean Features)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png")
plt.close()

# 2c. Boxplots of key features by diagnosis
key_features = ["mean radius", "mean concave points", "mean texture", "worst perimeter"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
plot_df = X.copy()
plot_df["diagnosis"] = y.map({0: "Malignant", 1: "Benign"})
for ax, feat in zip(axes, key_features):
    sns.boxplot(data=plot_df, x="diagnosis", y=feat, hue="diagnosis",
                palette=["#d62728", "#2ca02c"], legend=False, ax=ax)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_feature_boxplots.png")
plt.close()

# ---------------------------------------------------------------
# 3. TRAIN / TEST SPLIT + SCALING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. DEFINE MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

results = []
roc_data = {}
confusion_matrices = {}

for name, model in models.items():
    # 5-fold CV accuracy on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")

    # Fit on full training set, evaluate on held-out test set
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "CV Accuracy (mean)": round(cv_scores.mean(), 4),
        "CV Accuracy (std)": round(cv_scores.std(), 4),
        "Test Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall (Sensitivity)": round(rec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    })

    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison_results.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by test accuracy: {best_model_name}")

# ---------------------------------------------------------------
# 5. CONFUSION MATRIX (best model)
# ---------------------------------------------------------------
cm = confusion_matrices[best_model_name]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix_best_model.png")
plt.close()

# ---------------------------------------------------------------
# 6. ROC CURVES (all models)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_roc_curves.png")
plt.close()

# ---------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(7, 5))
sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index,
            palette="viridis", legend=False)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_feature_importance.png")
plt.close()

top_features.to_csv(f"{OUT_DIR}/top_feature_importances.csv", header=["importance"])

print("\n=== Top 10 Most Important Features (Random Forest) ===")
print(top_features.to_string())

# ---------------------------------------------------------------
# 8. PCA VISUALIZATION (2D)
# ---------------------------------------------------------------
X_scaled_full = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled_full)

plt.figure(figsize=(6, 5))
plot_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
plot_pca_df["diagnosis"] = y.map({0: "Malignant", 1: "Benign"}).values
sns.scatterplot(data=plot_pca_df, x="PC1", y="PC2", hue="diagnosis",
                 palette=["#d62728", "#2ca02c"], alpha=0.7)
plt.title(f"PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_pca_projection.png")
plt.close()

print("\nPCA explained variance (PC1 + PC2):", round(pca.explained_variance_ratio_.sum(), 4))

print("\nAll plots and result files saved to:", OUT_DIR)
