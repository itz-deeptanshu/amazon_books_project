"""
03_modeling.py
--------------
Predictive modeling on the Amazon Bestsellers dataset.

Task: Binary classification -- predict whether a bestselling book is
"Fiction" or "Non Fiction" from its User Rating, Reviews, Price, and Year.

Mirrors the structure of the stroke-prediction project:
  1. Feature/target setup + train-test split
  2. Preprocessing (scaling)
  3. Train & compare multiple models
  4. Evaluate with accuracy, precision, recall, F1, ROC-AUC
  5. Feature importance from the best tree-based model
  6. Confusion matrix + ROC curve for the best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)

sns.set_theme(style="whitegrid")
FIG_DIR = "figures"

# ---------------------------------------------------------------
# 1. LOAD + FEATURE SETUP
# ---------------------------------------------------------------
df = pd.read_csv("data/amazon_bestsellers_cleaned.csv")

le = LabelEncoder()
y = le.fit_transform(df["Genre"])          # Fiction=0, Non Fiction=1 (alphabetical)
X = df[["User Rating", "Reviews", "Price", "Year"]].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
fitted = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    })
    fitted[name] = (model, probs, preds)

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("=" * 70)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("=" * 70)
print(results_df.round(3).to_string(index=False))

# ---------------------------------------------------------------
# 3. VISUALIZE MODEL COMPARISON
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
results_df.set_index("Model")[["Accuracy", "F1", "ROC-AUC"]].plot(kind="bar", ax=ax)
ax.set_title("Model Comparison")
ax.set_ylim(0, 1)
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_model_comparison.png")
plt.close()

# ---------------------------------------------------------------
# 4. BEST MODEL: CONFUSION MATRIX + ROC CURVE
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model, best_probs, best_preds = fitted[best_name]
print(f"\nBest model: {best_name}")
print("\nClassification report:\n", classification_report(y_test, best_preds, target_names=le.classes_))

cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix ({best_name})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, best_probs)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"{best_name} (AUC={results_df.iloc[0]['ROC-AUC']:.3f})", color="#2b6cb0")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Best Model)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_roc_curve.png")
plt.close()

# ---------------------------------------------------------------
# 5. FEATURE IMPORTANCE (tree-based models)
# ---------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", color="#805ad5", ax=ax)
    ax.set_title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/13_feature_importance.png")
    plt.close()
    print("\nFeature importances:\n", importances.sort_values(ascending=False))
else:
    print(f"\n{best_name} has no native feature_importances_ attribute.")

results_df.to_csv("data/model_comparison_results.csv", index=False)
print("\nModel comparison results saved to data/model_comparison_results.csv")
print("All figures saved to figures/")

print("\n" + "=" * 70)
print("CAVEAT")
print("=" * 70)
print("Genre in this dataset is nearly balanced (~53/47) and only weakly")
print("related to Rating/Reviews/Price/Year (see correlation heatmap from")
print("02_eda.py) -- so expect modest lift over the ~53% majority-class")
print("baseline. That's an honest finding, not a bug: it says Genre isn't")
print("well-predicted by these four numeric features alone, which is a")
print("useful, reportable insight in its own right.")
