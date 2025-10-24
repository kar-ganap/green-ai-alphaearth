# RUN Phase: System Architecture

**Date**: October 23, 2025
**Status**: Design Phase
**Purpose**: Production-ready deforestation early warning system

---

## Executive Summary

The RUN phase builds on the production-ready XGBoost model (0.913 AUROC) from WALK phase by creating a complete, deployable system with:
- REST API for programmatic access
- Interactive dashboard for visual exploration
- SHAP-based explanations
- Historical validation ("playback")
- Cost-benefit analysis
- Satellite imagery integration

**Target users**: Conservation organizations, park rangers, policymakers, researchers

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
├──────────────────────────┬──────────────────────────────────────┤
│   Web Dashboard          │   REST API                            │
│   (Streamlit)            │   (FastAPI)                           │
│   - Map interface        │   - /predict                          │
│   - Visual analytics     │   - /explain                          │
│   - Historical playback  │   - /batch                            │
│   - ROI calculator       │   - /model-info                       │
└──────────┬───────────────┴────────────┬──────────────────────────┘
           │                            │
           └────────────┬───────────────┘
                        │
           ┌────────────▼────────────┐
           │   Model Service         │
           │   (Core Business Logic) │
           │   - Feature extraction  │
           │   - Predictions         │
           │   - SHAP explanations   │
           │   - Batch processing    │
           └────────────┬────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │XGBoost  │    │ Earth   │    │  Data   │
   │ Model   │    │ Engine  │    │ Storage │
   │ (0.913  │    │ API     │    │ (Cache) │
   │ AUROC)  │    │         │    │         │
   └─────────┘    └─────────┘    └─────────┘
```

---

## Component Specifications

### 1. Model Service ✅ (COMPLETED)

**File**: `src/run/model_service.py`

**Capabilities**:
```python
class DeforestationModelService:
    # Single prediction with confidence
    predict(lat, lon, year, threshold=0.5) -> dict

    # Batch predictions
    predict_batch(locations, threshold=0.5) -> list[dict]

    # SHAP explanations
    explain_prediction(lat, lon, year, top_k=5) -> dict

    # Model metadata
    get_model_info() -> dict
```

**Features**:
- Loads production XGBoost model
- Extracts 70D features (3D annual + 66D multiscale + 1D year)
- Risk categorization (very_low → very_high)
- Confidence levels (low/medium/high)
- SHAP-based feature importance

**Status**: ✅ Complete

---

### 2. REST API (FastAPI)

**File**: `src/run/api/main.py`

**Endpoints**:

#### `POST /predict`
Predict deforestation risk for a single location.

**Request**:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "year": 2024,
  "threshold": 0.5
}
```

**Response**:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "year": 2024,
  "risk_probability": 0.87,
  "predicted_class": 1,
  "risk_category": "very_high",
  "confidence": 0.74,
  "confidence_label": "high",
  "threshold": 0.5,
  "timestamp": "2025-10-23T15:30:00"
}
```

#### `POST /explain`
Get SHAP explanation for a prediction.

**Request**:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "year": 2024,
  "top_k": 5
}
```

**Response**:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "risk_probability": 0.87,
  "explanation": {
    "top_features": [
      {
        "feature": "delta_1yr",
        "value": -0.45,
        "shap_value": 0.23,
        "direction": "increases",
        "contribution_pct": 34.2
      },
      ...
    ],
    "base_value": 0.53,
    "total_contribution": 0.34
  }
}
```

#### `POST /batch`
Predict for multiple locations.

**Request**:
```json
{
  "locations": [
    {"lat": -3.8248, "lon": -50.2500, "year": 2024},
    {"lat": -3.2356, "lon": -50.4530, "year": 2024}
  ],
  "threshold": 0.5
}
```

**Response**: Array of prediction objects

#### `GET /model-info`
Get model metadata.

**Response**:
```json
{
  "model_type": "XGBClassifier",
  "n_features": 70,
  "training_samples": 847,
  "training_years": "2020-2024",
  "validation_auroc": 0.913,
  "model_date": "2025-10-23"
}
```

#### `GET /health`
Health check endpoint.

**Technology**: FastAPI 0.100+, Pydantic for validation, CORS enabled

---

### 3. Web Dashboard (Streamlit)

**File**: `src/run/dashboard/app.py`

**Pages**:

#### **Page 1: Prediction Explorer**
- Interactive map (Folium/Plotly)
- Click to predict or enter coordinates
- Year selector (2020-2030)
- Risk visualization (color-coded)
- Confidence indicator
- SHAP waterfall chart for selected location

**Layout**:
```
┌─────────────────────────────────────────────────┐
│  Deforestation Early Warning System             │
├─────────────────────────────────────────────────┤
│  [Map View]                │  [Prediction]      │
│                            │  Location: -3.82,  │
│   [Interactive Map]        │            -50.25  │
│   - Click to predict       │  Year: 2024        │
│   - Risk overlay           │  Risk: 87% (High)  │
│   - Zoom/pan              │  Confidence: High   │
│                            │                     │
│                            │  [SHAP Explanation] │
│                            │  Top 5 features:    │
│                            │  • delta_1yr: +34%  │
│                            │  • coarse_emb_3: +18% │
│                            │  ...                │
└────────────────────────────┴─────────────────────┘
```

#### **Page 2: Historical Playback**
- Timeline slider (2020-2024)
- Show actual clearings from validation set
- Overlay model predictions made 90 days before
- "Hit rate" statistics
- Animated playback

**Features**:
```python
# Show how well we predicted 2023 clearings
- Load 2023 validation samples
- For each clearing, show:
  * Actual clearing date
  * Model prediction 90 days prior
  * Risk score at prediction time
  * Whether caught (risk > threshold)

# Summary stats:
- Total clearings: 125
- Caught by model: 98 (78%)
- Average lead time: 87 days
- False positive rate: 15%
```

#### **Page 3: Cost-Benefit Calculator**
- Interactive ROI calculator
- Sliders for:
  * Cost per alert investigation ($)
  * Ecosystem value per hectare ($/ha)
  * Enforcement success rate (%)
- Real-time ROI calculation
- Charts showing break-even analysis

**Formula**:
```python
# Benefits
hectares_saved = true_positives * avg_hectares_per_clearing * success_rate
value_saved = hectares_saved * value_per_hectare

# Costs
total_cost = total_alerts * cost_per_investigation

# ROI
roi = (value_saved - total_cost) / total_cost * 100
```

#### **Page 4: Batch Analysis**
- Upload CSV with locations
- Batch predictions
- Download results
- Risk heatmap
- Priority ranking

#### **Page 5: Model Performance**
- Validation metrics dashboard
- Performance by use-case
- Performance by year
- Confusion matrix
- ROC curve

**Technology**: Streamlit 1.28+, Plotly for charts, Folium for maps

---

### 4. Satellite Imagery Integration

**File**: `src/run/utils/imagery.py`

**Functionality**:
```python
def get_satellite_imagery(lat, lon, date, days_before=30):
    """
    Fetch Sentinel-2 RGB imagery for location.

    Returns:
        dict with:
        - image_url: URL to RGB composite
        - cloud_cover: % cloud coverage
        - acquisition_date: actual image date
        - resolution: spatial resolution
    """
```

**Dashboard Integration**:
- Before/after comparison for predictions
- Side-by-side RGB composites
- 3-month intervals (before → clearing → after)
- Visual verification of predictions

**Data Source**: Sentinel-2 via Earth Engine

---

### 5. Production Features

#### **Logging**
```python
# src/run/utils/logging_config.py
- Request logging (all API calls)
- Prediction logging (for monitoring)
- Error logging (with stack traces)
- Performance metrics (latency, throughput)
```

#### **Caching**
```python
# src/run/utils/cache.py
- Feature extraction cache (lat/lon/year → features)
- Prediction cache (features → prediction)
- TTL: 24 hours for features, 1 hour for predictions
- Storage: Redis or local file-based
```

#### **Monitoring**
```python
# src/run/utils/monitoring.py
- Prediction distribution tracking
- Drift detection (feature drift)
- Performance metrics
- Alert on anomalies
```

#### **Rate Limiting**
```python
# API rate limiting
- 100 requests/minute for /predict
- 10 requests/minute for /batch
- 1000 requests/hour per IP
```

---

## Data Flow

### Prediction Request Flow

```
1. User Request (Dashboard or API)
   ↓
2. Model Service.predict(lat, lon, year)
   ↓
3. Feature Extraction
   ├─→ Earth Engine: Get AlphaEarth embedding
   ├─→ Calculate annual features (delta_1yr, delta_2yr, acceleration)
   ├─→ Calculate multiscale features (64 embeddings + 2 stats)
   └─→ Normalize year feature
   ↓
4. XGBoost Model Prediction
   ├─→ Input: 70D feature vector
   └─→ Output: Risk probability [0, 1]
   ↓
5. Post-processing
   ├─→ Apply threshold
   ├─→ Calculate confidence
   ├─→ Categorize risk
   └─→ (Optional) Compute SHAP values
   ↓
6. Return Results
   └─→ JSON response
```

---

## Deployment Architecture

### Development
```
localhost:8000 (API)
localhost:8501 (Dashboard)
```

### Production Options

#### **Option A: Single Server**
```
nginx → FastAPI (8000)
     → Streamlit (8501)
     → Model Service (shared)
```

#### **Option B: Containerized (Docker)**
```dockerfile
# docker-compose.yml
services:
  api:
    image: deforestation-api
    ports: ["8000:8000"]
  dashboard:
    image: deforestation-dashboard
    ports: ["8501:8501"]
  redis:
    image: redis:latest
```

#### **Option C: Cloud (Recommended)**
```
AWS/GCP/Azure:
- API: Cloud Run / Lambda / App Service
- Dashboard: Streamlit Cloud / Cloud Run
- Model: Cloud Storage
- Cache: ElastiCache / Memorystore
- Monitoring: CloudWatch / Cloud Monitoring
```

---

## Technology Stack

### Core
- **Python**: 3.9+
- **Model**: XGBoost 2.0+
- **Data**: NumPy, Pandas
- **Earth Engine**: earthengine-api

### API
- **Framework**: FastAPI 0.100+
- **Validation**: Pydantic
- **ASGI Server**: Uvicorn
- **Documentation**: Auto-generated OpenAPI/Swagger

### Dashboard
- **Framework**: Streamlit 1.28+
- **Maps**: Folium / Plotly
- **Charts**: Plotly Express
- **Tables**: Pandas DataFrames

### ML Explainability
- **SHAP**: 0.42+ (TreeExplainer for XGBoost)

### Optional Production
- **Caching**: Redis
- **Monitoring**: Prometheus + Grafana
- **Logging**: Loguru / structlog

---

## Directory Structure

```
src/run/
├── __init__.py
├── model_service.py          # ✅ Core prediction service
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── routes.py             # API endpoints
│   └── models.py             # Pydantic schemas
├── dashboard/
│   ├── __init__.py
│   ├── app.py                # Main Streamlit app
│   ├── pages/
│   │   ├── 1_prediction.py   # Prediction explorer
│   │   ├── 2_playback.py     # Historical playback
│   │   ├── 3_roi.py          # ROI calculator
│   │   ├── 4_batch.py        # Batch analysis
│   │   └── 5_performance.py  # Model metrics
│   └── components/
│       ├── map.py            # Map visualization
│       ├── charts.py         # Chart components
│       └── shap_viz.py       # SHAP visualizations
└── utils/
    ├── __init__.py
    ├── imagery.py            # Satellite imagery
    ├── logging_config.py     # Logging setup
    ├── cache.py              # Caching utilities
    └── monitoring.py         # Monitoring utils
```

---

## Implementation Plan

### Phase 1: Foundation (2-3 hours)
1. ✅ Model Service (DONE)
2. FastAPI skeleton + basic endpoints
3. Streamlit skeleton + basic UI

### Phase 2: Core Features (3-4 hours)
4. API: All endpoints with validation
5. Dashboard: Prediction page with map
6. SHAP integration

### Phase 3: Enhanced Features (3-4 hours)
7. Historical playback page
8. Satellite imagery integration
9. ROI calculator

### Phase 4: Production Features (2-3 hours)
10. Logging and monitoring
11. Caching
12. Batch processing
13. Error handling

### Phase 5: Polish & Testing (1-2 hours)
14. End-to-end testing
15. Documentation
16. Deployment guide

**Total Estimated Time**: 11-16 hours

---

## Success Criteria

### Functional
- ✅ Make predictions for any Amazon location
- ✅ Explain predictions with SHAP
- ✅ Show historical validation
- ✅ Calculate ROI
- ✅ Visualize satellite imagery

### Non-Functional
- ⚡ Response time < 5s for single prediction
- 📊 Support 100+ concurrent users
- 🛡️ Error rate < 1%
- 📝 All endpoints documented (OpenAPI)
- 🎨 Dashboard is intuitive (< 5 min learning curve)

### Documentation
- API reference (auto-generated)
- User guide for dashboard
- Deployment guide
- Architecture diagram

---

## Open Questions for Review

1. **Imagery Source**: Use Sentinel-2 (free, 10m resolution) or Planet (paid, 3m resolution)?

2. **Caching Strategy**: Redis (requires server) or file-based (simpler, slower)?

3. **Dashboard Hosting**: Streamlit Cloud (easiest) or self-host (more control)?

4. **API Authentication**: Public (demo) or require API keys (production)?

5. **Batch Limits**: Max locations per batch request? (suggest: 100)

6. **Historical Data**: How far back for playback? (suggest: 2023-2024)

7. **Additional Features**:
   - Email/Slack alerts?
   - Webhook notifications?
   - Export to KML/GeoJSON?

---

## Next Steps

**After approval**:
1. Build FastAPI (api/main.py, api/routes.py)
2. Build Streamlit dashboard (dashboard/app.py + pages)
3. Add satellite imagery integration
4. Add production features (logging, caching)
5. Test end-to-end
6. Document everything
7. Create deployment guide

**For transfer learning** (after RUN phase):
- Collect Congo Basin validation data
- Test zero-shot performance
- Fine-tune with 100-200 labels
- Document transfer methodology

---

## Estimated Deliverables

At completion of RUN phase, you will have:

1. **Working REST API** (localhost:8000)
   - Documented with Swagger UI
   - All endpoints functional
   - Example requests/responses

2. **Interactive Dashboard** (localhost:8501)
   - 5 pages fully functional
   - Professional UI/UX
   - Demo-ready

3. **Documentation**
   - API reference
   - User guide
   - Deployment guide
   - Architecture document (this)

4. **Production-Ready Code**
   - Error handling
   - Logging
   - Monitoring hooks
   - Docker setup (optional)

---

**Status**: Awaiting approval to proceed with implementation

**Questions/Feedback**: Please review and provide input on:
- Overall architecture
- Feature priorities
- Technology choices
- Open questions above
