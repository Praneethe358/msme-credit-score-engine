from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import date


class ShopFeatures(BaseModel):
    revenue_cv: float
    revenue_growth_slope: float
    payment_collection_ratio: float
    avg_dso: float
    repeat_customer_rate: float
    gst_compliance_rate: float


class ShopBase(BaseModel):
    shop_id: str
    shop_name: str
    shop_type: str
    city: str
    gst_registered: bool
    onboarded_date: date
    months_active: int


class CreditScoreResponse(BaseModel):
    shop_id: Optional[str] = None
    score: int
    tier: str
    probability: float
    features: ShopFeatures
    shap_explanations: Optional[Dict[str, float]] = None


class ShopDetailResponse(BaseModel):
    shop: ShopBase
    features: ShopFeatures
    score: int
    tier: str
    probability: float
    shap_waterfall_url: Optional[str] = None
