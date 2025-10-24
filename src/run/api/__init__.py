"""
FastAPI REST API for Deforestation Prediction
"""

from .main import app
from .schemas import (
    PredictionRequest, PredictionResponse,
    ExplainRequest, ExplainResponse,
    BatchRequest, BatchResponse,
    ModelInfo, HealthResponse
)

__all__ = [
    'app',
    'PredictionRequest', 'PredictionResponse',
    'ExplainRequest', 'ExplainResponse',
    'BatchRequest', 'BatchResponse',
    'ModelInfo', 'HealthResponse'
]
