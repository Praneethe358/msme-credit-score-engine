import os
import shap
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for non-interactive plot saving
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from app.config import DOCS_DIR
from app.services.scoring import scoring_service

FEATURE_COLS = [
    "revenue_cv",
    "revenue_growth_slope",
    "payment_collection_ratio",
    "avg_dso",
    "repeat_customer_rate",
    "gst_compliance_rate",
]


class ExplainabilityService:
    def __init__(self):
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        if scoring_service.model is not None:
            try:
                self.explainer = shap.TreeExplainer(scoring_service.model)
            except Exception as e:
                print(f"Error initializing TreeExplainer: {e}")
        else:
            print("Scoring service model not initialized. Explainer cannot start.")

    def explain_features(self, features_df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """
        Computes SHAP values for class 1 (creditworthy) and expected base value.
        """
        if self.explainer is None:
            return {col: 0.0 for col in FEATURE_COLS}, 0.0

        try:
            shap_values = self.explainer.shap_values(features_df[FEATURE_COLS])
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return {col: 0.0 for col in FEATURE_COLS}, 0.0

        # Retrieve class 1 expected value and shap values
        # shap output shape variations: list (older versions) vs ndarray (newer versions)
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1][0]
            expected_value = float(self.explainer.expected_value[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_class1 = shap_values[0, :, 1]
            expected_value = float(self.explainer.expected_value[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_values_class1 = shap_values[0]
            expected_value = float(self.explainer.expected_value)
        else:
            shap_values_class1 = shap_values[0]
            expected_value = float(self.explainer.expected_value)

        # Map feature names to contributions
        explanations = {
            col: float(val) for col, val in zip(FEATURE_COLS, shap_values_class1)
        }
        return explanations, expected_value

    def generate_waterfall_plot(
        self, shop_id: str, features_df: pd.DataFrame
    ) -> str:
        """
        Generates and saves a SHAP waterfall plot for a shop.
        Returns the filename.
        """
        if self.explainer is None:
            return ""

        try:
            shap_values = self.explainer.shap_values(features_df[FEATURE_COLS])
            
            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1][0]
                expected_value = self.explainer.expected_value[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_class1 = shap_values[0, :, 1]
                expected_value = self.explainer.expected_value[1]
            else:
                shap_values_class1 = shap_values[0]
                expected_value = self.explainer.expected_value

            explanation = shap.Explanation(
                values=shap_values_class1,
                base_values=expected_value,
                data=features_df[FEATURE_COLS].iloc[0].values,
                feature_names=FEATURE_COLS,
            )

            # Draw and save the waterfall plot
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            
            filename = f"shap_waterfall_{shop_id}.png"
            filepath = os.path.join(DOCS_DIR, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            return filename
        except Exception as e:
            print(f"Error generating waterfall plot for {shop_id}: {e}")
            return ""


# Singleton instance
explainability_service = ExplainabilityService()
