import os
import pandas as pd
from typing import List, Dict, Optional
from app.config import (
    SHOPS_PATH,
    LABELED_FEATURES_PATH,
    INVOICES_PATH,
    CUSTOMERS_PATH,
    GST_FILINGS_PATH,
)
from app.schemas.scoring import ShopBase, ShopFeatures, ShopDetailResponse
from app.services.scoring import scoring_service


class DataLoaderService:
    def __init__(self):
        self.shops: Dict[str, Dict[str, Any]] = {}
        self.preload_data()

    def preload_data(self):
        """
        Loads the precalculated data and populates in-memory cache.
        """
        if not os.path.exists(SHOPS_PATH) or not os.path.exists(LABELED_FEATURES_PATH):
            print("Required data files not found for preloading.")
            return

        try:
            shops_df = pd.read_csv(SHOPS_PATH)
            features_df = pd.read_csv(LABELED_FEATURES_PATH)

            # Convert shops to list of dicts
            for _, shop_row in shops_df.iterrows():
                shop_id = str(shop_row["shop_id"])
                
                # Find matching features
                feat_row = features_df[features_df["shop_id"] == shop_id]
                if feat_row.empty:
                    continue

                feat_dict = feat_row.iloc[0].to_dict()
                
                # Build ShopFeatures schema
                features = ShopFeatures(
                    revenue_cv=float(feat_dict["revenue_cv"]),
                    revenue_growth_slope=float(feat_dict["revenue_growth_slope"]),
                    payment_collection_ratio=float(feat_dict["payment_collection_ratio"]),
                    avg_dso=float(feat_dict["avg_dso"]),
                    repeat_customer_rate=float(feat_dict["repeat_customer_rate"]),
                    gst_compliance_rate=float(feat_dict["gst_compliance_rate"]),
                )

                # Get score, tier, and probability using the model
                score, tier, prob = scoring_service.predict_score(features)

                # Determine waterfall filename
                # If pre-rendered files exist: docs/shap_waterfall_SHOP002_strongest.png etc.
                # We can map them, or just point to /docs/shap/shap_waterfall_SHOPXXX.png
                waterfall_url = f"/api/shops/{shop_id}/shap"

                shop_base = ShopBase(
                    shop_id=shop_id,
                    shop_name=str(shop_row["shop_name"]),
                    shop_type=str(shop_row["shop_type"]),
                    city=str(shop_row["city"]),
                    gst_registered=bool(shop_row["gst_registered"]),
                    onboarded_date=pd.to_datetime(shop_row["onboarded_date"]).date(),
                    months_active=int(shop_row["months_active"]),
                )

                self.shops[shop_id] = {
                    "shop": shop_base,
                    "features": features,
                    "score": score,
                    "tier": tier,
                    "probability": prob,
                    "shap_waterfall_url": waterfall_url,
                }

            print(f"Preloaded {len(self.shops)} shops successfully.")
        except Exception as e:
            print(f"Error preloading dataset: {e}")

    def get_all_shops(self) -> List[Dict[str, Any]]:
        """
        Returns all preloaded shops.
        """
        return list(self.shops.values())

    def get_shop_by_id(self, shop_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a preloaded shop detail by its ID.
        """
        return self.shops.get(shop_id)


# Singleton instance
data_loader_service = DataLoaderService()
