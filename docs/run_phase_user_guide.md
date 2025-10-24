# RUN Phase User Guide

**Deforestation Early Warning System - Production Deployment**

Date: October 23, 2025
Status: Complete and Ready for Deployment
Version: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [REST API](#rest-api)
5. [Dashboard](#dashboard)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## Overview

The RUN phase provides a complete production-ready system for deforestation prediction:

- **REST API**: Programmatic access via FastAPI (5 endpoints)
- **Dashboard**: Interactive web interface via Streamlit (5 pages)
- **Model**: XGBoost classifier (0.913 AUROC, 340 validation samples)
- **Features**: 70D (Annual + Multiscale + Temporal)
- **Geography**: Brazilian Amazon (transferable to other regions)

**Capabilities**:
- Single location prediction with SHAP explanations
- Batch predictions (up to 100 locations)
- Historical validation playback (2021-2024)
- ROI calculator for cost-benefit analysis
- Model performance metrics

---

## Installation

### Prerequisites

- Python 3.9+
- Google Earth Engine account (authenticated)
- Trained XGBoost model at `data/processed/final_xgb_model_2020_2024.pkl`

### Step 1: Install Dependencies

```bash
# Install RUN phase requirements
pip install -r requirements_run.txt

# Or if using uv (recommended):
uv pip install -r requirements_run.txt
```

### Step 2: Verify Earth Engine Authentication

```bash
# Authenticate with Google Earth Engine (if not already done)
earthengine authenticate

# Test authentication
python -c "import ee; ee.Initialize(); print('✓ Earth Engine authenticated')"
```

### Step 3: Verify Model File

Ensure the trained model exists:

```bash
ls -lh data/processed/final_xgb_model_2020_2024.pkl
```

If missing, train the model first:

```bash
# Train final production model (from WALK phase)
python src/walk/13_train_xgboost_69d.py
```

---

## Quick Start

### Option 1: Run Dashboard (Recommended for Exploration)

```bash
# Start Streamlit dashboard
streamlit run src/run/dashboard/app.py

# Dashboard will open at: http://localhost:8501
```

**Features**:
- Home page with overview
- Page 1: Prediction Explorer (map + SHAP)
- Page 2: Historical Playback (validation results)
- Page 3: ROI Calculator (cost-benefit analysis)
- Page 4: Batch Analysis (CSV upload)
- Page 5: Model Performance (metrics)

### Option 2: Run API (Recommended for Integration)

```bash
# Start FastAPI server
uvicorn src.run.api.main:app --reload --port 8000

# API will be available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

**Endpoints**:
- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Single prediction
- `POST /explain`: Prediction with SHAP
- `POST /batch`: Batch predictions
- `GET /model-info`: Model metadata

### Option 3: Run Both (Production Setup)

```bash
# Terminal 1: Start API
uvicorn src.run.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Dashboard
streamlit run src/run/dashboard/app.py --server.port 8501
```

---

## REST API

### Base URL

```
http://localhost:8000
```

### Endpoint 1: Single Prediction

**POST /predict**

Request:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "year": 2024,
  "threshold": 0.5
}
```

Response:
```json
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
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": -3.8248,
    "lon": -50.2500,
    "year": 2024,
    "threshold": 0.5
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "lat": -3.8248,
        "lon": -50.2500,
        "year": 2024,
        "threshold": 0.5
    }
)

result = response.json()
print(f"Risk: {result['risk_probability']:.1%}")
print(f"Category: {result['risk_category']}")
```

### Endpoint 2: SHAP Explanation

**POST /explain**

Request:
```json
{
  "lat": -3.8248,
  "lon": -50.2500,
  "year": 2024,
  "top_k": 5
}
```

Response includes all prediction fields plus:
```json
{
  ...
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

### Endpoint 3: Batch Predictions

**POST /batch**

Request:
```json
{
  "locations": [
    {"lat": -3.8248, "lon": -50.2500, "year": 2024},
    {"lat": -3.2356, "lon": -50.4530, "year": 2024}
  ],
  "threshold": 0.5
}
```

Response:
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    { /* prediction result 1 */ },
    { /* prediction result 2 */ }
  ]
}
```

**Limits**:
- Maximum 100 locations per request
- Use pagination for larger datasets

### Endpoint 4: Model Info

**GET /model-info**

Response:
```json
{
  "model_type": "XGBClassifier",
  "n_features": 70,
  "feature_names": ["delta_1yr", "delta_2yr", ...],
  "training_samples": 847,
  "training_years": "2020-2024",
  "validation_auroc": 0.913,
  "validation_samples": 340,
  "model_date": "2025-10-23"
}
```

### Endpoint 5: Health Check

**GET /health**

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-23T15:30:00",
  "version": "1.0.0"
}
```

### Interactive Documentation

Visit `http://localhost:8000/docs` for:
- Interactive API testing (Swagger UI)
- Schema documentation
- Example requests/responses
- Authentication (if enabled)

---

## Dashboard

### Running the Dashboard

```bash
streamlit run src/run/dashboard/app.py
```

Access at: `http://localhost:8501`

### Page 1: Prediction Explorer

**Features**:
- Interactive map (click to predict)
- Manual coordinate entry
- Year selector (2020-2030)
- Risk gauge visualization
- SHAP waterfall chart (top 10 features)

**Usage**:
1. Click on the map OR enter coordinates manually
2. Select prediction year
3. Adjust threshold (default: 0.5)
4. Click "Predict Risk"
5. View results and SHAP explanation

**SHAP Interpretation**:
- **Red bars**: Features increasing deforestation risk
- **Blue bars**: Features decreasing deforestation risk
- **Magnitude**: Larger = stronger influence

### Page 2: Historical Playback

**Features**:
- Validate model on past clearings (2021-2024)
- Performance metrics by dataset
- Confusion matrix
- Temporal analysis
- Download results

**Usage**:
1. Select validation dataset (sidebar)
2. Filter by year
3. Adjust threshold
4. View performance metrics

**Metrics**:
- **Recall (Catch Rate)**: % of actual clearings caught
- **Precision**: % of alerts that are true clearings
- **Accuracy**: Overall correct predictions
- **AUROC**: Discrimination ability (0.913)

### Page 3: ROI Calculator

**Features**:
- Interactive cost/benefit analysis
- Real-time ROI calculation
- Break-even analysis
- Sensitivity analysis

**Usage**:
1. Enter cost parameters:
   - Cost per investigation
   - Fixed annual costs
2. Enter benefit parameters:
   - Ecosystem value ($/ha)
   - Carbon value ($/ha)
   - Enforcement success rate
3. Adjust model performance sliders
4. View ROI and break-even point

**Interpretation**:
- **Positive ROI**: System generates more value than cost
- **Negative ROI**: Consider optimizing parameters
- **Break-even**: Minimum hectares needed to protect

### Page 4: Batch Analysis

**Features**:
- Upload CSV with locations
- Batch predictions (max 100)
- Risk heatmap
- Priority ranking
- Download results

**CSV Format**:
```csv
location_id,name,lat,lon,year
1,Site A,-3.8248,-50.2500,2024
2,Site B,-3.2356,-50.4530,2024
```

**Required columns**: `lat`, `lon`, `year`
**Optional columns**: `location_id`, `name`

**Usage**:
1. Download sample CSV (expand instructions)
2. Prepare your CSV file
3. Upload file
4. Click "Run Batch Predictions"
5. View results and download

### Page 5: Model Performance

**Features**:
- Comprehensive metrics dashboard
- Performance by use-case
- Performance by year
- Confusion matrix
- ROC curve
- Threshold analysis

**Insights**:
- AUROC: 0.913 (excellent)
- Recall: 78% at threshold=0.5
- Precision: 74% at threshold=0.5
- Validated on 340 challenging samples

---

## Troubleshooting

### Issue 1: Model Not Found

**Error**: `Failed to load model: [Errno 2] No such file or directory`

**Solution**:
```bash
# Train the production model first
python src/walk/13_train_xgboost_69d.py

# Verify model exists
ls -lh data/processed/final_xgb_model_2020_2024.pkl
```

### Issue 2: Earth Engine Not Authenticated

**Error**: `ee.EEException: Please authenticate to Earth Engine`

**Solution**:
```bash
# Authenticate
earthengine authenticate

# Initialize in Python
python -c "import ee; ee.Initialize()"
```

### Issue 3: Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Option 1: Use different port
uvicorn src.run.api.main:app --port 8001

# Option 2: Kill existing process
lsof -ti:8000 | xargs kill -9
```

### Issue 4: Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'streamlit_folium'`

**Solution**:
```bash
# Reinstall all dependencies
pip install -r requirements_run.txt

# Or install specific package
pip install streamlit-folium
```

### Issue 5: SHAP Installation Failed

**Error**: SHAP installation requires C++ compiler

**Solution**:
```bash
# Install build tools first
# macOS:
xcode-select --install

# Linux:
sudo apt-get install build-essential

# Then install SHAP
pip install shap
```

### Issue 6: Dashboard Map Not Loading

**Error**: Map appears blank in Prediction Explorer

**Solution**:
- Check internet connection (map tiles require network)
- Refresh browser
- Clear browser cache
- Try different browser

---

## Advanced Configuration

### Custom Model Path

```python
# In src/run/model_service.py
model_service = DeforestationModelService(
    model_path="path/to/custom/model.pkl"
)
```

### API Configuration

```python
# In src/run/api/main.py

# Change host/port
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Allow external access
        port=8000,
        reload=True
    )

# Disable CORS (production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Dashboard Configuration

```bash
# Custom port
streamlit run src/run/dashboard/app.py --server.port 8502

# Disable caching (development)
streamlit run src/run/dashboard/app.py --server.runOnSave true
```

### Performance Optimization

**1. Enable Caching** (future enhancement):
```python
# Add Redis caching for feature extraction
# See src/run/utils/cache.py (planned)
```

**2. Batch Processing**:
```python
# Process large datasets in chunks
chunk_size = 100
for chunk in chunks(locations, chunk_size):
    results = model_service.predict_batch(chunk)
```

**3. Async Predictions** (future enhancement):
```python
# Use async/await for concurrent predictions
# See FastAPI async endpoints
```

---

## Production Deployment

### Option 1: Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements_run.txt

# Expose ports
EXPOSE 8000 8501

# Start both services
CMD uvicorn src.run.api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/run/dashboard/app.py --server.port 8501
```

Build and run:
```bash
docker build -t deforestation-api .
docker run -p 8000:8000 -p 8501:8501 deforestation-api
```

### Option 2: Cloud Deployment

**Streamlit Cloud** (Dashboard):
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

**AWS/GCP/Azure** (API):
- AWS: Elastic Beanstalk or Lambda
- GCP: Cloud Run or App Engine
- Azure: App Service

### Option 3: Local Production

```bash
# Use production ASGI server
gunicorn src.run.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Next Steps

**After completing RUN phase**:

1. **Transfer Learning**:
   - Test model on Congo Basin
   - Collect 100-200 validation samples
   - Fine-tune model for new region
   - Document transfer methodology

2. **Operational Deployment**:
   - Field testing with conservation teams
   - Feedback collection
   - Model monitoring and retraining
   - Alert system integration

3. **System Enhancements**:
   - Email/Slack alerts
   - Webhook notifications
   - Export to KML/GeoJSON
   - Real-time monitoring dashboard

---

## Support

**Documentation**:
- Architecture: `docs/run_phase_architecture.md`
- This guide: `docs/run_phase_user_guide.md`
- API reference: `http://localhost:8000/docs` (when running)

**Issues**:
- Check troubleshooting section
- Verify all dependencies installed
- Ensure Earth Engine authenticated
- Check model file exists

**Contact**:
- GitHub Issues: [Project Repository]
- Email: support@deforestation-ai.org

---

**Status**: System is production-ready! 🎉

**Version**: 1.0.0
**Last Updated**: October 23, 2025
