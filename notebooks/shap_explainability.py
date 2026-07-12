"""
Phase 4 — SHAP Explainability
MSME Shop Creditworthiness Scoring Engine
--------------------------------------------
Loads the trained Random Forest model and generates:
  1. Global feature importance (SHAP summary) — which features matter
     most across all shops
  2. Per-shop waterfall explanation — exactly why ONE shop got its score

Outputs saved to /docs/shap_summary.png and /docs/shap_waterfall_<shop_id>.png
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set paths relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

FEATURE_COLS = [
    "revenue_cv", "revenue_growth_slope", "payment_collection_ratio",
    "avg_dso", "repeat_customer_rate", "gst_compliance_rate"
]

model = joblib.load(os.path.join(MODEL_DIR, "credit_model.pkl"))
df = pd.read_csv(os.path.join(DATA_DIR, "labeled_features.csv"))
X = df[FEATURE_COLS]

# TreeExplainer is purpose-built for tree-based models (Random Forest, XGBoost)
# -> much faster and exact, vs the general-purpose KernelExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP's output shape varies by version:
#   - older shap: list of 2 arrays, one per class -> shap_values[1]
#   - newer shap: single array shaped (samples, features, classes)
#     -> shap_values[:, :, 1]
# We care about class 1 ("creditworthy") contributions either way.
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
    expected_value = explainer.expected_value[1]
elif shap_values.ndim == 3:
    shap_values_class1 = shap_values[:, :, 1]
    expected_value = explainer.expected_value[1]
else:
    shap_values_class1 = shap_values
    expected_value = explainer.expected_value

# ---------- Global summary plot ----------
plt.figure()
shap.summary_plot(
    shap_values_class1, X, feature_names=FEATURE_COLS, show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(DOCS_DIR, "shap_summary.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()
summary_path = os.path.join(DOCS_DIR, "shap_summary.png")
print(f"Saved global summary plot -> {summary_path}")

# Calculate proxy_credit_score based on prediction probability
# Mapping probability (0.0 to 1.0) to credit score range (300 to 900)
probabilities = model.predict_proba(X)[:, 1]
df["proxy_credit_score"] = 300 + probabilities * 600

# ---------- Per-shop waterfall: pick strong and weak shop ----------
example_shops = {
    "strongest": df.loc[df["proxy_credit_score"].idxmax(), "shop_id"],
    "weakest": df.loc[df["proxy_credit_score"].idxmin(), "shop_id"],
}

for label, shop_id in example_shops.items():
    idx = df.index[df["shop_id"] == shop_id][0]

    explanation = shap.Explanation(
        values=shap_values_class1[idx],
        base_values=expected_value,
        data=X.iloc[idx].values,
        feature_names=FEATURE_COLS,
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    filepath = os.path.join(DOCS_DIR, f"shap_waterfall_{shop_id}_{label}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall for {shop_id} ({label}) -> {filepath}")

print("\nDone. SHAP explainability layer complete.")
