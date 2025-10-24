"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    year: int = Field(..., ge=2020, le=2030, description="Year for prediction (2020-2030)")
    threshold: float = Field(0.5, ge=0, le=1, description="Classification threshold (0-1)")

    class Config:
        schema_extra = {
            "example": {
                "lat": -3.8248,
                "lon": -50.2500,
                "year": 2024,
                "threshold": 0.5
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    lat: float
    lon: float
    year: int
    risk_probability: float = Field(..., ge=0, le=1)
    predicted_class: int = Field(..., ge=0, le=1)
    threshold: float
    confidence: float = Field(..., ge=0, le=1)
    confidence_label: Literal["low", "medium", "high"]
    risk_category: Literal["very_low", "low", "medium", "high", "very_high"]
    timestamp: str

    class Config:
        schema_extra = {
            "example": {
                "lat": -3.8248,
                "lon": -50.2500,
                "year": 2024,
                "risk_probability": 0.87,
                "predicted_class": 1,
                "threshold": 0.5,
                "confidence": 0.74,
                "confidence_label": "high",
                "risk_category": "very_high",
                "timestamp": "2025-10-23T15:30:00"
            }
        }


class ExplainRequest(BaseModel):
    """Request schema for SHAP explanation."""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    year: int = Field(..., ge=2020, le=2030)
    top_k: int = Field(5, ge=1, le=20, description="Number of top features to return")

    class Config:
        schema_extra = {
            "example": {
                "lat": -3.8248,
                "lon": -50.2500,
                "year": 2024,
                "top_k": 5
            }
        }


class FeatureContribution(BaseModel):
    """Feature contribution from SHAP."""
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases", "decreases"]
    contribution_pct: float


class Explanation(BaseModel):
    """SHAP explanation details."""
    top_features: List[FeatureContribution]
    base_value: float
    total_contribution: float


class ExplainResponse(PredictionResponse):
    """Response schema for explanation (includes prediction + SHAP)."""
    explanation: Explanation

    class Config:
        schema_extra = {
            "example": {
                "lat": -3.8248,
                "lon": -50.2500,
                "year": 2024,
                "risk_probability": 0.87,
                "predicted_class": 1,
                "threshold": 0.5,
                "confidence": 0.74,
                "confidence_label": "high",
                "risk_category": "very_high",
                "timestamp": "2025-10-23T15:30:00",
                "explanation": {
                    "top_features": [
                        {
                            "feature": "delta_1yr",
                            "value": -0.45,
                            "shap_value": 0.23,
                            "direction": "increases",
                            "contribution_pct": 34.2
                        }
                    ],
                    "base_value": 0.53,
                    "total_contribution": 0.34
                }
            }
        }


class LocationInput(BaseModel):
    """Single location for batch processing."""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    year: int = Field(..., ge=2020, le=2030)


class BatchRequest(BaseModel):
    """Request schema for batch prediction."""
    locations: List[LocationInput] = Field(..., min_items=1, max_items=100)
    threshold: float = Field(0.5, ge=0, le=1)

    @validator('locations')
    def validate_locations(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 locations per batch request")
        return v

    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {"lat": -3.8248, "lon": -50.2500, "year": 2024},
                    {"lat": -3.2356, "lon": -50.4530, "year": 2024}
                ],
                "threshold": 0.5
            }
        }


class BatchResponse(BaseModel):
    """Response schema for batch prediction."""
    total: int
    successful: int
    failed: int
    results: List[PredictionResponse]

    class Config:
        schema_extra = {
            "example": {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {
                        "lat": -3.8248,
                        "lon": -50.2500,
                        "year": 2024,
                        "risk_probability": 0.87,
                        "predicted_class": 1,
                        "threshold": 0.5,
                        "confidence": 0.74,
                        "confidence_label": "high",
                        "risk_category": "very_high",
                        "timestamp": "2025-10-23T15:30:00"
                    }
                ]
            }
        }


class ModelInfo(BaseModel):
    """Model metadata."""
    model_type: str
    n_features: int
    feature_names: List[str]
    training_samples: int
    training_years: str
    validation_auroc: float
    validation_samples: int
    model_date: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: str
