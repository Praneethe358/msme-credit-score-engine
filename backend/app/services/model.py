import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.schemas.predict import MSMEPredictionInput, MSMEPredictionOutput

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "msme_risk_model.joblib")

class MSMEModelService:
    def __init__(self):
        self.model = None
        self.model_version = "1.0.0"
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Ensures that a trained model is saved and ready to load."""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not os.path.exists(MODEL_PATH):
            print("Model file not found. Training a starter Random Forest model...")
            self._train_and_save_starter_model()
        else:
            self.model = joblib.load(MODEL_PATH)
            print("Successfully loaded existing model.")

    def _train_and_save_starter_model(self):
        """Generates synthetic MSME credit data and trains a RandomForestClassifier."""
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic features
        annual_revenue = np.random.uniform(20000, 1000000, n_samples)
        years_in_business = np.random.uniform(0.5, 20, n_samples)
        credit_score = np.random.randint(500, 850, n_samples)
        existing_debt = np.random.uniform(0, 300000, n_samples)
        requested_amount = np.random.uniform(5000, 250000, n_samples)

        # Create a simple rule-based probability of approval for synthetic labels
        # Higher revenue, years in business, credit score increase approval chance
        # Higher existing debt and requested amount decrease it
        score = (
            (annual_revenue / 100000) * 1.5 +
            years_in_business * 0.8 +
            ((credit_score - 500) / 50) * 2.0 -
            (existing_debt / 50000) * 1.2 -
            (requested_amount / 50000) * 1.0
        )
        # Add some noise
        score += np.random.normal(0, 1.5, n_samples)

        # Binary classification target (1: Approved/Low Risk, 0: Rejected/High Risk)
        # Set threshold to approve top 70% of businesses
        threshold = np.percentile(score, 30)
        approved = (score > threshold).astype(int)

        # Train a model
        df = pd.DataFrame({
            "annual_revenue": annual_revenue,
            "years_in_business": years_in_business,
            "credit_score": credit_score,
            "existing_debt": existing_debt,
            "requested_amount": requested_amount
        })

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(df, approved)

        # Save model
        joblib.dump(clf, MODEL_PATH)
        self.model = clf
        print(f"Starter model trained and saved to {MODEL_PATH}")

    def predict(self, data: MSMEPredictionInput) -> MSMEPredictionOutput:
        """Runs predictions using the loaded RandomForest model."""
        if self.model is None:
            self.model = joblib.load(MODEL_PATH)

        # Prepare input features
        features = pd.DataFrame([{
            "annual_revenue": data.annual_revenue,
            "years_in_business": data.years_in_business,
            "credit_score": data.credit_score,
            "existing_debt": data.existing_debt,
            "requested_amount": data.requested_amount
        }])

        # Predict probability of approval
        prob = self.model.predict_proba(features)[0][1]
        eligible = bool(prob >= 0.5)

        # Calculate recommended maximum loan limit
        # Simple formula: 20% of annual revenue minus 30% of existing debt
        recommended_limit = max(0.0, (data.annual_revenue * 0.25) - (data.existing_debt * 0.35))
        
        # Calculate risk score (0-100), higher probability -> lower risk
        risk_score = float(np.clip(100.0 - (prob * 100.0), 0.0, 100.0))

        return MSMEPredictionOutput(
            eligible=eligible,
            approval_probability=float(prob),
            recommended_limit=float(recommended_limit),
            risk_score=risk_score,
            model_version=self.model_version
        )

# Singleton instance
model_service = MSMEModelService()
