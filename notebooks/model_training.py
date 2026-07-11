"""
Phase 3b — Model Training & Evaluation
MSME Shop Creditworthiness Scoring Engine
--------------------------------------------
Trains two models on the engineered feature table + proxy labels:
  1. Random Forest (baseline)
  2. XGBoost (improvement target)

Compares them on accuracy, precision, recall, F1, and confusion matrix.
Saves the better-performing model to /model/credit_model.pkl for use
in the FastAPI scoring service (Phase 5).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import os

# Set paths relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- 1. Load features and health profiles for proxy labels ----------
features_df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
health_path = os.path.join(DATA_DIR, "_shops_with_health_profile.csv")
health_df = pd.read_csv(health_path)

# Merge on shop_id
df = pd.merge(
    features_df,
    health_df[["shop_id", "_health_profile"]],
    on="shop_id"
)

# Apply proxy labeling rule:
# 1 (creditworthy) if strong health profile OR average health profile with
# gst_compliance_rate >= 0.85 and payment_collection_ratio >= 0.90.
# 0 (not creditworthy) otherwise (weak health profile, or average health
# profile that does not meet the criteria).
df["credit_label"] = df.apply(
    lambda row: 1 if row["_health_profile"] == "strong" or (
        row["_health_profile"] == "average"
        and row["gst_compliance_rate"] >= 0.85
        and row["payment_collection_ratio"] >= 0.90
    ) else 0,
    axis=1
)

# Save labeled features
labeled_features_path = os.path.join(DATA_DIR, "labeled_features.csv")
df.to_csv(labeled_features_path, index=False)
print(f"Proxy labels generated and saved to {labeled_features_path}\n")

# ---------- 2. Prepare training and test sets ----------
FEATURE_COLS = [
    "revenue_cv", "revenue_growth_slope", "payment_collection_ratio",
    "avg_dso", "repeat_customer_rate", "gst_compliance_rate"
]

X = df[FEATURE_COLS]
y = df["credit_label"]

# 80/20 split, stratified to preserve 50/50 class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test class balance:  {y_test.value_counts().to_dict()}\n")


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"--- {name} ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}\n")
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


# ---------- Model 1: Random Forest (baseline) ----------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=5, random_state=42
)
rf_model.fit(X_train, y_train)
rf_results = evaluate("Random Forest (baseline)", rf_model, X_test, y_test)

# ---------- Model 2: XGBoost ----------
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)
xgb_results = evaluate("XGBoost", xgb_model, X_test, y_test)

# ---------- Compare and pick the winner ----------
results_df = pd.DataFrame([rf_results, xgb_results])
print("=== Model Comparison ===")
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df["f1"].idxmax(), "model"]
best_model = xgb_model if "XGBoost" in best_model_name else rf_model

print(f"\nBest model (by F1): {best_model_name}")

# Save both models + comparison table
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(best_model, os.path.join(MODEL_DIR, "credit_model.pkl"))
results_df.to_csv(os.path.join(DATA_DIR, "model_comparison.csv"), index=False)

# Feature importance from the winning model (quick preview before Phase 4 SHAP)
importances = pd.Series(
    best_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)
print(f"\nFeature importances ({best_model_name}):")
print(importances.round(3))
