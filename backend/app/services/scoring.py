import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from app.config import MODEL_PATH
from app.schemas.scoring import ShopFeatures


class ScoringService:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found at {self.model_path}")

    def predict_score(self, features: ShopFeatures) -> Tuple[int, str, float]:
        """
        Predicts credit score, tier, and probability for a given set of features.
        Returns:
            (credit_score, tier, probability)
        """
        if self.model is None:
            # Fallback if model is not loaded
            return 300, "Reject", 0.0

        # Prepare features for prediction (in the exact order used in training)
        feature_cols = [
            "revenue_cv",
            "revenue_growth_slope",
            "payment_collection_ratio",
            "avg_dso",
            "repeat_customer_rate",
            "gst_compliance_rate",
        ]
        
        feature_dict = features.model_dump()
        df = pd.DataFrame([feature_dict])[feature_cols]

        # Get probability of class 1 ("creditworthy")
        try:
            prob = float(self.model.predict_proba(df)[0, 1])
        except Exception as e:
            print(f"Error running prediction: {e}")
            prob = 0.0

        # Scale probability (0.0 to 1.0) to credit score range (300 to 900)
        credit_score = int(300 + prob * 600)

        # Assign tier based on score
        if credit_score >= 750:
            tier = "Gold"
        elif credit_score >= 600:
            tier = "Silver"
        elif credit_score >= 500:
            tier = "Bronze"
        else:
            tier = "Reject"

        return credit_score, tier, prob


# Singleton instance
scoring_service = ScoringService()
