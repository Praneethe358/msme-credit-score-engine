"""
Phase 5 — FastAPI Backend
MSME Shop Creditworthiness Scoring Engine
--------------------------------------------
Endpoints:
  POST /score   -> upload shop's invoices.csv (+ optional customers.csv,
                    gst_filings.csv) -> returns score, label, feature
                    values, and SHAP-based explanation
  GET  /health  -> status check

Run locally:
  uvicorn main:app --reload --port 8000
"""

import io
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from feature_utils import compute_features_for_shop, FEATURE_COLS

app = FastAPI(title="MSME Credit Scoring API", version="1.0")

# Allow the Next.js frontend (different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./model/credit_model.pkl"
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)


class ScoreResponse(BaseModel):
    shop_id: str
    proxy_credit_score: float
    credit_label: str
    loan_eligibility_tier: str
    features: dict
    shap_explanation: dict


def score_to_tier(score: float) -> str:
    if score >= 750:
        return "Recommended — eligible up to ₹5L"
    elif score >= 650:
        return "Conditional — eligible up to ₹2L with review"
    elif score >= 500:
        return "Caution — collateral/guarantor recommended"
    else:
        return "Not recommended — refer to microfinance"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score", response_model=ScoreResponse)
async def score_shop(
    invoices_file: UploadFile = File(...),
    customers_file: Optional[UploadFile] = File(None),
    gst_file: Optional[UploadFile] = File(None),
    shop_id: str = "UPLOADED_SHOP",
):
    # --- parse uploaded CSVs ---
    invoices_df = pd.read_csv(io.BytesIO(await invoices_file.read()))
    customers_df = (
        pd.read_csv(io.BytesIO(await customers_file.read())) if customers_file else None
    )
    gst_df = (
        pd.read_csv(io.BytesIO(await gst_file.read())) if gst_file else None
    )

    # --- feature engineering (same logic as Phase 2, shared module) ---
    features = compute_features_for_shop(invoices_df, customers_df, gst_df, shop_id)
    X = pd.DataFrame([{k: features[k] for k in FEATURE_COLS}])

    # --- prediction ---
    pred_proba = model.predict_proba(X)[0][1]  # probability of class 1 ("creditworthy")
    proxy_credit_score = round(300 + pred_proba * 600, 0)
    credit_label = "creditworthy" if pred_proba >= 0.5 else "risky"

    # --- SHAP explanation for this single shop ---
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    elif shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
    else:
        shap_row = shap_values[0]

    shap_explanation = {
        feature: round(float(value), 4)
        for feature, value in zip(FEATURE_COLS, shap_row)
    }

    return ScoreResponse(
        shop_id=shop_id,
        proxy_credit_score=proxy_credit_score,
        credit_label=credit_label,
        loan_eligibility_tier=score_to_tier(proxy_credit_score),
        features=features,
        shap_explanation=shap_explanation,
    )
