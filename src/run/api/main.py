"""
FastAPI REST API for Deforestation Prediction

Endpoints:
- POST /predict: Single location prediction
- POST /explain: Prediction with SHAP explanation
- POST /batch: Batch predictions (max 100 locations)
- GET /model-info: Model metadata
- GET /health: Health check

Run with: uvicorn src.run.api.main:app --reload --port 8000
Access docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import sys
from typing import Union

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.run.api.schemas import (
    PredictionRequest, PredictionResponse,
    ExplainRequest, ExplainResponse,
    BatchRequest, BatchResponse,
    ModelInfo, HealthResponse, ErrorResponse
)
from src.run.model_service import DeforestationModelService

# Initialize FastAPI app
app = FastAPI(
    title="Deforestation Early Warning API",
    description="Predict tropical deforestation risk 90 days in advance using AlphaEarth embeddings",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@deforestation-ai.org",
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware (public demo - allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Public demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service (loaded once at startup)
model_service = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_service
    try:
        print("Loading model...")
        model_service = DeforestationModelService()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Deforestation Early Warning API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "explain": "POST /explain",
            "batch": "POST /batch",
            "model_info": "GET /model-info",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and model availability
    """
    model_loaded = model_service is not None

    if not model_loaded:
        status_val = "unhealthy"
    else:
        status_val = "healthy"

    return HealthResponse(
        status=status_val,
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict deforestation risk for a single location.

    Args:
        request: Location (lat, lon, year) and threshold

    Returns:
        Risk prediction with confidence levels

    Example:
        ```json
        {
          "lat": -3.8248,
          "lon": -50.2500,
          "year": 2024,
          "threshold": 0.5
        }
        ```
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        result = model_service.predict(
            lat=request.lat,
            lon=request.lon,
            year=request.year,
            threshold=request.threshold
        )
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=ExplainResponse, tags=["Predictions"])
async def explain(request: ExplainRequest):
    """
    Get prediction with SHAP explanation.

    Args:
        request: Location (lat, lon, year) and top_k features

    Returns:
        Prediction with top contributing features

    Example:
        ```json
        {
          "lat": -3.8248,
          "lon": -50.2500,
          "year": 2024,
          "top_k": 5
        }
        ```
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        result = model_service.explain_prediction(
            lat=request.lat,
            lon=request.lon,
            year=request.year,
            top_k=request.top_k
        )

        # Check if SHAP is available
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )

        return ExplainResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.post("/batch", response_model=BatchResponse, tags=["Predictions"])
async def batch_predict(request: BatchRequest):
    """
    Predict for multiple locations (max 100).

    Args:
        request: List of locations and threshold

    Returns:
        Batch prediction results

    Example:
        ```json
        {
          "locations": [
            {"lat": -3.8248, "lon": -50.2500, "year": 2024},
            {"lat": -3.2356, "lon": -50.4530, "year": 2024}
          ],
          "threshold": 0.5
        }
        ```
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    if len(request.locations) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 locations per batch request"
        )

    try:
        # Convert to list of tuples for model service
        locations = [
            (loc.lat, loc.lon, loc.year)
            for loc in request.locations
        ]

        # Get predictions
        results = model_service.predict_batch(
            locations=locations,
            threshold=request.threshold
        )

        # Separate successful and failed predictions
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]

        # Convert successful predictions to PredictionResponse
        prediction_responses = [
            PredictionResponse(**r) for r in successful
        ]

        return BatchResponse(
            total=len(results),
            successful=len(successful),
            failed=len(failed),
            results=prediction_responses
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get model metadata and performance metrics.

    Returns:
        Model type, features, training data, and validation performance
    """
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        info = model_service.get_model_info()
        return ModelInfo(**info)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("DEFORESTATION EARLY WARNING API")
    print("=" * 80)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
