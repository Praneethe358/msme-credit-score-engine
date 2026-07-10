from pydantic import BaseModel, Field

class MSMEPredictionInput(BaseModel):
    annual_revenue: float = Field(..., description="Annual revenue of the MSME in USD", example=150000.0)
    years_in_business: float = Field(..., description="Number of years the business has been operating", example=3.5)
    credit_score: int = Field(..., description="Business owner credit score (300-850)", example=680)
    existing_debt: float = Field(..., description="Current outstanding business debt in USD", example=20000.0)
    requested_amount: float = Field(..., description="Requested loan amount in USD", example=50000.0)

class MSMEPredictionOutput(BaseModel):
    eligible: bool = Field(..., description="Whether the MSME is eligible for the requested loan")
    approval_probability: float = Field(..., description="Probability of approval (0.0 to 1.0)")
    recommended_limit: float = Field(..., description="Maximum recommended loan limit in USD")
    risk_score: float = Field(..., description="Calculated risk score (0.0 to 100.0), lower is better")
    model_version: str = Field(..., description="Version of the model used for prediction")
