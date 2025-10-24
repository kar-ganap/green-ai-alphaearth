# System Architecture

**AlphaEarth Deforestation Early Warning System**

**Date**: October 2025
**Version**: 1.0
**Status**: ✅ Production-ready

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Feature Extraction Pipeline](#feature-extraction-pipeline)
4. [Model Architecture](#model-architecture)
5. [Production System Architecture](#production-system-architecture)
6. [Phase Progression (CRAWL → WALK → RUN)](#phase-progression)
7. [Component Interactions](#component-interactions)
8. [Deployment Architecture](#deployment-architecture)

---

## High-Level Architecture

### System Overview

```mermaid
graph TB
    subgraph "User Interfaces"
        UI1[Web Dashboard<br/>Streamlit]
        UI2[REST API<br/>FastAPI]
        UI3[Python SDK<br/>Model Service]
    end

    subgraph "Core Services"
        MS[Model Service<br/>Prediction Engine]
        FE[Feature Extractor<br/>70D Feature Generation]
        ME[Model Explainer<br/>SHAP Analysis]
    end

    subgraph "External Services"
        EE[Google Earth Engine<br/>AlphaEarth Embeddings]
        HC[Hansen Dataset<br/>Forest Change Labels]
    end

    subgraph "Storage"
        MD[Production Model<br/>final_xgb_model_2020_2024.pkl]
        VD[Validation Data<br/>Hard Validation Sets]
        CC[Cache<br/>Embedding Cache]
    end

    UI1 --> MS
    UI2 --> MS
    UI3 --> MS
    MS --> FE
    MS --> ME
    MS --> MD
    FE --> EE
    FE --> CC
    ME --> MD
    VD -.validation.-> MS

    style UI1 fill:#e1f5ff
    style UI2 fill:#e1f5ff
    style MS fill:#fff4e6
    style FE fill:#fff4e6
    style EE fill:#f3e5f5
    style MD fill:#e8f5e9
```

### Key Components

| Component | Technology | Purpose | Status |
|-----------|-----------|---------|--------|
| **Model Service** | Python, XGBoost | Core prediction engine | ✅ Production |
| **Feature Extractor** | Python, Google Earth Engine | 70D feature generation | ✅ Production |
| **REST API** | FastAPI, Pydantic | Programmatic access | ✅ Production |
| **Dashboard** | Streamlit, Folium | Interactive UI | ✅ Production |
| **Explainer** | SHAP | Model interpretability | ✅ Production |

---

## Data Flow Architecture

### End-to-End Prediction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as REST API / Dashboard
    participant MS as Model Service
    participant FE as Feature Extractor
    participant EE as Earth Engine
    participant XGB as XGBoost Model
    participant SHAP as SHAP Explainer

    U->>API: Request prediction<br/>(lat, lon, year)
    API->>MS: predict(lat, lon, year)

    MS->>FE: extract_features(lat, lon, year)

    Note over FE: Annual Features (3D)
    FE->>EE: get_embedding(year-2)
    EE-->>FE: emb_t-2 (64D)
    FE->>EE: get_embedding(year-1)
    EE-->>FE: emb_t-1 (64D)
    FE->>EE: get_embedding(year)
    EE-->>FE: emb_t (64D)

    Note over FE: Compute deltas
    FE->>FE: delta_1yr = ||emb_t - emb_t-1||<br/>delta_2yr = ||emb_t - emb_t-2||<br/>acceleration = delta_1yr - delta_2yr

    Note over FE: Multiscale Features (66D)
    FE->>EE: get_coarse_embedding(1km radius)
    EE-->>FE: coarse_emb (64D) + stats (2D)

    Note over FE: Year Feature (1D)
    FE->>FE: year_norm = (year - 2020) / 4.0

    FE-->>MS: features (70D)

    MS->>XGB: predict_proba(features)
    XGB-->>MS: risk_probability

    alt SHAP Explanation Requested
        MS->>SHAP: explain(features)
        SHAP-->>MS: top_k feature contributions
    end

    MS-->>API: prediction result + explanation
    API-->>U: {risk_probability, confidence, explanation}
```

### Caching Strategy

```mermaid
graph LR
    subgraph "Feature Extraction with Caching"
        REQ[Feature Request<br/>lat, lon, year]
        CACHE{Cache<br/>Hit?}
        EE[Earth Engine<br/>API Call]
        COMPUTE[Compute<br/>Features]
        STORE[Store in<br/>Cache]
        RETURN[Return<br/>Features]
    end

    REQ --> CACHE
    CACHE -->|Yes| RETURN
    CACHE -->|No| EE
    EE --> COMPUTE
    COMPUTE --> STORE
    STORE --> RETURN

    style CACHE fill:#ffe082
    style EE fill:#f3e5f5
    style RETURN fill:#c8e6c9
```

**Cache Keys**: MD5 hash of `(lat, lon, date, collection_name)`
**Cache Storage**: Local file system (data/cache/)
**Hit Rate**: ~85% for validation sets, ~40% for new locations

---

## Feature Extraction Pipeline

### 70D Feature Architecture

```mermaid
graph TB
    subgraph "Input"
        LOC[Location<br/>lat, lon, year]
    end

    subgraph "Annual Features (3D)"
        E1[AlphaEarth<br/>year-2]
        E2[AlphaEarth<br/>year-1]
        E3[AlphaEarth<br/>year]
        D1[delta_1yr<br/>norm of E3 - E2]
        D2[delta_2yr<br/>norm of E3 - E1]
        ACC[acceleration<br/>D1 - D2]
    end

    subgraph "Multiscale Features (66D)"
        COARSE[AlphaEarth Coarse<br/>1km radius]
        EMB64[Embeddings<br/>64D]
        HET[Heterogeneity<br/>std(embeddings)]
        RNG[Range<br/>max - min]
    end

    subgraph "Temporal Feature (1D)"
        YEAR[Normalized Year<br/>(year - 2020) / 4.0]
    end

    subgraph "Output"
        FEAT[70D Feature Vector]
    end

    LOC --> E1
    LOC --> E2
    LOC --> E3
    LOC --> COARSE

    E1 --> D2
    E2 --> D1
    E3 --> D1
    E3 --> D2

    D1 --> ACC
    D2 --> ACC

    COARSE --> EMB64
    COARSE --> HET
    COARSE --> RNG

    LOC --> YEAR

    D1 --> FEAT
    D2 --> FEAT
    ACC --> FEAT
    EMB64 --> FEAT
    HET --> FEAT
    RNG --> FEAT
    YEAR --> FEAT

    style FEAT fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    style LOC fill:#e1f5ff
```

### Feature Importance Distribution

```mermaid
pie title "Feature Importance (%)"
    "delta_1yr" : 32
    "coarse_emb_15" : 8
    "delta_2yr" : 7
    "coarse_heterogeneity" : 6
    "acceleration" : 5
    "coarse_emb_7" : 4
    "coarse_emb_23" : 3
    "Other coarse embeddings" : 28
    "normalized_year" : 3
    "Other stats" : 4
```

**Key Insights**:
- Annual deltas dominate (44% combined)
- Multiscale context critical (landscape fragmentation = 6%)
- Individual coarse embeddings less important than their aggregate

---

## Model Architecture

### XGBoost Pipeline

```mermaid
graph LR
    subgraph "Input Layer"
        F70[70D Features]
    end

    subgraph "XGBoost Ensemble"
        T1[Tree 1<br/>depth=6]
        T2[Tree 2<br/>depth=6]
        TN[Tree N<br/>depth=6]
        AGG[Gradient Boosting<br/>Aggregation]
    end

    subgraph "Output Layer"
        LOGIT[Logits]
        SIGMOID[Sigmoid]
        PROB[Risk Probability<br/>0, 1]
    end

    F70 --> T1
    F70 --> T2
    F70 --> TN

    T1 --> AGG
    T2 --> AGG
    TN --> AGG

    AGG --> LOGIT
    LOGIT --> SIGMOID
    SIGMOID --> PROB

    style F70 fill:#e1f5ff
    style PROB fill:#c8e6c9
```

### Training Configuration

```mermaid
graph TB
    subgraph "Hyperparameters"
        HP[n_estimators: 200<br/>max_depth: 6<br/>learning_rate: 0.1<br/>subsample: 0.8<br/>colsample_bytree: 0.8<br/>scale_pos_weight: 1.12]
    end

    subgraph "Training Data"
        TD[847 samples<br/>400 cleared<br/>447 stable<br/>2020-2024]
    end

    subgraph "Validation Strategy"
        VAL[Spatial CV<br/>5 folds<br/>3km separation]
    end

    subgraph "Trained Model"
        MODEL[XGBoost Model<br/>0.913 AUROC]
    end

    HP --> MODEL
    TD --> MODEL
    VAL --> MODEL

    style MODEL fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

### SHAP Explanation Architecture

```mermaid
graph LR
    subgraph "Input"
        FEAT[Features<br/>70D]
        MODEL[XGBoost<br/>Model]
    end

    subgraph "SHAP Processing"
        EXPL[TreeExplainer]
        SHAP[SHAP Values<br/>70D]
        RANK[Rank by<br/>Absolute Value]
    end

    subgraph "Output"
        TOP[Top-K Features<br/>with Contributions]
        VIZ[Waterfall Chart]
    end

    FEAT --> EXPL
    MODEL --> EXPL
    EXPL --> SHAP
    SHAP --> RANK
    RANK --> TOP
    TOP --> VIZ

    style VIZ fill:#c8e6c9
```

---

## Production System Architecture

### Service Layer Design

```mermaid
graph TB
    subgraph "User Layer"
        WEB[Web Browser]
        CLI[CLI Tools]
        SCRIPT[Python Scripts]
    end

    subgraph "Interface Layer"
        DASH[Streamlit Dashboard<br/>:8501]
        API[FastAPI REST API<br/>:8000]
        SDK[Python SDK<br/>model_service.py]
    end

    subgraph "Service Layer"
        MS[Model Service]
        FE[Feature Extractor]
        ME[Model Explainer]
    end

    subgraph "Data Layer"
        MODEL[XGBoost Model<br/>PKL]
        CACHE[Embedding Cache<br/>PKL]
        VAL[Validation Sets<br/>PKL]
    end

    subgraph "External APIs"
        EE[Google Earth Engine<br/>AlphaEarth]
    end

    WEB --> DASH
    CLI --> API
    SCRIPT --> SDK

    DASH --> MS
    API --> MS
    SDK --> MS

    MS --> FE
    MS --> ME
    MS --> MODEL

    FE --> EE
    FE --> CACHE

    ME --> MODEL

    style MS fill:#fff4e6
    style MODEL fill:#e8f5e9
    style EE fill:#f3e5f5
```

### API Endpoint Architecture

```mermaid
graph LR
    subgraph "FastAPI Application"
        ROOT[GET /<br/>API Info]
        HEALTH[GET /health<br/>Health Check]
        PREDICT[POST /predict<br/>Single Prediction]
        EXPLAIN[POST /explain<br/>SHAP Explanation]
        BATCH[POST /batch<br/>Batch Predictions]
        INFO[GET /model-info<br/>Model Metadata]
    end

    subgraph "Middleware"
        CORS[CORS<br/>Middleware]
        EXCEPT[Exception<br/>Handler]
        VALID[Pydantic<br/>Validation]
    end

    subgraph "Model Service"
        MS[DeforestationModelService]
    end

    ROOT --> CORS
    HEALTH --> CORS
    PREDICT --> VALID
    EXPLAIN --> VALID
    BATCH --> VALID
    INFO --> CORS

    CORS --> EXCEPT
    VALID --> EXCEPT

    EXCEPT --> MS

    style PREDICT fill:#ffe082
    style EXPLAIN fill:#ffe082
    style BATCH fill:#ffe082
```

### Dashboard Page Architecture

```mermaid
graph TB
    subgraph "Streamlit App"
        MAIN[Main Page<br/>app.py]
    end

    subgraph "Dashboard Pages"
        P1[1. Prediction Explorer<br/>Interactive Map]
        P2[2. Historical Playback<br/>Validation Results]
        P3[3. ROI Calculator<br/>Cost-Benefit Analysis]
        P4[4. Batch Analysis<br/>CSV Upload]
        P5[5. Model Performance<br/>Metrics Dashboard]
    end

    subgraph "Shared Services"
        MS[Model Service]
        CACHE[Session Cache]
    end

    MAIN --> P1
    MAIN --> P2
    MAIN --> P3
    MAIN --> P4
    MAIN --> P5

    P1 --> MS
    P2 --> MS
    P2 --> CACHE
    P4 --> MS
    P5 --> MS

    style P1 fill:#e1f5ff
    style P2 fill:#e1f5ff
    style P3 fill:#e1f5ff
    style P4 fill:#e1f5ff
    style P5 fill:#e1f5ff
```

---

## Phase Progression

### CRAWL → WALK → RUN Evolution

```mermaid
graph TB
    subgraph "CRAWL Phase: Validation"
        C1[Single Location Test<br/>Proof of Concept]
        C2[Multi-Location Test<br/>Statistical Significance]
        C3[Temporal Sensitivity<br/>Early Warning Feasibility]
        C4[Geographic Test<br/>Generalization Check]
        CD[Decision:<br/>Proceed to WALK?]
    end

    subgraph "WALK Phase: Development"
        W1[Annual Features<br/>3D Deltas]
        W2[Hard Validation Sets<br/>Honest Evaluation]
        W3[Multiscale Embeddings<br/>+7 AUROC points]
        W4[Spatial CV<br/>Prevent Leakage]
        W5[Production Model<br/>0.913 AUROC]
        WD[Decision:<br/>Ready for Production?]
    end

    subgraph "RUN Phase: Deployment"
        R1[Model Service<br/>Inference Engine]
        R2[REST API<br/>FastAPI]
        R3[Dashboard<br/>Streamlit]
        R4[SHAP Explanations<br/>Interpretability]
        RD[Production:<br/>Ready for Users]
    end

    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> CD
    CD -->|✅ Yes| W1

    W1 --> W2
    W2 --> W3
    W3 --> W4
    W4 --> W5
    W5 --> WD
    WD -->|✅ Yes| R1

    R1 --> R2
    R2 --> R3
    R3 --> R4
    R4 --> RD

    style CD fill:#fff4e6,stroke:#ff6f00,stroke-width:2px
    style WD fill:#fff4e6,stroke:#ff6f00,stroke-width:2px
    style RD fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

---

## Component Interactions

### Core Component Dependency Graph

```mermaid
graph TB
    subgraph "Utilities (src/utils/)"
        CFG[config.py<br/>Configuration]
        EEC[earth_engine.py<br/>EE Client]
        GEO[geo.py<br/>Geographic Utils]
        VIZ[visualization.py<br/>Plotting]
    end

    subgraph "CRAWL (src/crawl/)"
        C1[01_single_location_test.py]
        C2[02_multi_location_test.py]
        C3[03_temporal_sensitivity.py]
        C4[04_geographic_test.py]
    end

    subgraph "WALK (src/walk/)"
        W1[01_data_preparation.py]
        W2[02_baseline_suite.py]
        WH[diagnostic_helpers.py]
        WA[annual_features.py]
        W8[08_multiscale_embeddings.py]
        W35[35_train_production_model.py]
    end

    subgraph "RUN (src/run/)"
        MS[model_service.py]
        API[api/main.py]
        DASH[dashboard/app.py]
    end

    CFG --> EEC
    CFG --> C1
    CFG --> C2
    CFG --> C3
    CFG --> C4

    EEC --> C1
    EEC --> C2
    EEC --> W1
    EEC --> WH
    EEC --> WA
    EEC --> W8
    EEC --> MS

    GEO --> C3
    GEO --> C4
    GEO --> W1

    WH --> W2
    WA --> W2
    WH --> W35
    WA --> W35
    W8 --> W35

    W35 --> MS
    WH --> MS
    WA --> MS
    W8 --> MS

    MS --> API
    MS --> DASH

    style MS fill:#fff4e6,stroke:#ff6f00,stroke-width:2px
    style API fill:#e1f5ff
    style DASH fill:#e1f5ff
```

### Data Dependency Graph

```mermaid
graph LR
    subgraph "Raw Data"
        HANSEN[Hansen Dataset<br/>Google Earth Engine]
        ALPHA[AlphaEarth<br/>Google Earth Engine]
    end

    subgraph "Processed Training Data"
        RAW[walk_dataset.pkl<br/>847 samples]
    end

    subgraph "Validation Data"
        RISK[hard_val_risk_ranking<br/>69 samples]
        COMP[hard_val_comprehensive<br/>81 samples]
        RAPID[hard_val_rapid_response<br/>68 samples]
    end

    subgraph "Feature Sets"
        FEAT_RISK[*_risk_ranking_*_features.pkl<br/>with 70D features]
        FEAT_COMP[*_comprehensive_*_features.pkl<br/>with 70D features]
        FEAT_RAPID[*_rapid_response_*_features.pkl<br/>with 70D features]
    end

    subgraph "Models"
        XGB[final_xgb_model_2020_2024.pkl<br/>Production Model]
        RF[final_rf_model_2020_2024.pkl<br/>Comparison Model]
    end

    HANSEN --> RAW
    ALPHA --> RAW

    RAW --> XGB
    RAW --> RF

    RISK --> FEAT_RISK
    COMP --> FEAT_COMP
    RAPID --> FEAT_RAPID

    ALPHA --> FEAT_RISK
    ALPHA --> FEAT_COMP
    ALPHA --> FEAT_RAPID

    XGB -.evaluation.-> FEAT_RISK
    XGB -.evaluation.-> FEAT_COMP
    XGB -.evaluation.-> FEAT_RAPID

    style XGB fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

---

## Deployment Architecture

### Local Development Setup

```mermaid
graph TB
    subgraph "Developer Machine"
        CODE[Source Code<br/>src/]
        ENV[Python venv<br/>requirements_run.txt]
        CONFIG[config.yaml<br/>Earth Engine Auth]
    end

    subgraph "Local Services"
        API_LOCAL[FastAPI<br/>localhost:8000]
        DASH_LOCAL[Streamlit<br/>localhost:8501]
    end

    subgraph "External Services"
        EE_API[Google Earth Engine<br/>API]
    end

    subgraph "Local Storage"
        CACHE_LOCAL[data/cache/<br/>Embedding Cache]
        MODEL_LOCAL[data/processed/<br/>Model Files]
    end

    CODE --> ENV
    ENV --> API_LOCAL
    ENV --> DASH_LOCAL

    CONFIG --> API_LOCAL
    CONFIG --> DASH_LOCAL

    API_LOCAL --> EE_API
    DASH_LOCAL --> EE_API

    API_LOCAL --> CACHE_LOCAL
    API_LOCAL --> MODEL_LOCAL
    DASH_LOCAL --> CACHE_LOCAL
    DASH_LOCAL --> MODEL_LOCAL

    style API_LOCAL fill:#e1f5ff
    style DASH_LOCAL fill:#e1f5ff
```

### Production Deployment Options

#### Option 1: Docker Compose

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        NGINX[NGINX<br/>Reverse Proxy<br/>:80]
        API_DOCKER[FastAPI Container<br/>:8000]
        DASH_DOCKER[Streamlit Container<br/>:8501]
        REDIS[Redis Cache<br/>:6379]
    end

    subgraph "Volumes"
        VOL_MODEL[models/<br/>Model Files]
        VOL_CACHE[cache/<br/>Embeddings]
    end

    subgraph "External"
        EE_PROD[Google Earth Engine]
    end

    USER[Users] --> NGINX
    NGINX -->|/api| API_DOCKER
    NGINX -->|/| DASH_DOCKER

    API_DOCKER --> VOL_MODEL
    API_DOCKER --> VOL_CACHE
    API_DOCKER --> REDIS
    API_DOCKER --> EE_PROD

    DASH_DOCKER --> VOL_MODEL
    DASH_DOCKER --> VOL_CACHE
    DASH_DOCKER --> REDIS
    DASH_DOCKER --> EE_PROD

    style NGINX fill:#ffecb3
    style API_DOCKER fill:#e1f5ff
    style DASH_DOCKER fill:#e1f5ff
    style REDIS fill:#ffcdd2
```

#### Option 2: Cloud Deployment (AWS/GCP)

```mermaid
graph TB
    subgraph "Cloud Infrastructure"
        LB[Load Balancer]

        subgraph "API Tier"
            API1[API Instance 1]
            API2[API Instance 2]
            APIN[API Instance N]
        end

        subgraph "Dashboard Tier"
            DASH1[Dashboard Instance 1]
        end

        subgraph "Storage"
            S3[Object Storage<br/>S3/GCS<br/>Models & Cache]
            REDIS_CLOUD[ElastiCache/MemoryStore<br/>Redis]
        end
    end

    subgraph "External"
        EE_CLOUD[Google Earth Engine]
    end

    USERS[Users] --> LB
    LB --> API1
    LB --> API2
    LB --> APIN
    LB --> DASH1

    API1 --> S3
    API2 --> S3
    APIN --> S3
    DASH1 --> S3

    API1 --> REDIS_CLOUD
    API2 --> REDIS_CLOUD
    APIN --> REDIS_CLOUD
    DASH1 --> REDIS_CLOUD

    API1 --> EE_CLOUD
    API2 --> EE_CLOUD
    APIN --> EE_CLOUD
    DASH1 --> EE_CLOUD

    style LB fill:#ffecb3
    style S3 fill:#c8e6c9
    style REDIS_CLOUD fill:#ffcdd2
```

---

## Performance Characteristics

### Latency Breakdown

```mermaid
graph LR
    REQ[Request] -->|0ms| VAL[Validation<br/>50ms]
    VAL -->|50ms| FE[Feature Extract<br/>3000ms]
    FE -->|3050ms| PRED[Prediction<br/>50ms]
    PRED -->|3100ms| SHAP[SHAP Explain<br/>2000ms]
    SHAP -->|5100ms| RESP[Response<br/>50ms]
    RESP -->|5150ms| END[Total: ~5.2s]

    style FE fill:#ffcdd2
    style SHAP fill:#ffe082
    style END fill:#c8e6c9
```

**Bottlenecks**:
1. **Feature Extraction (3s)**: Earth Engine API calls (3 annual + 1 coarse)
2. **SHAP Computation (2s)**: Tree traversal for all features
3. **Model Inference (<100ms)**: Fast (XGBoost optimized)

**Optimization Strategies**:
- ✅ **Caching**: Pre-extract embeddings (85% hit rate on validation)
- ✅ **Batching**: Group Earth Engine requests
- 🔮 **Future**: Async processing, Redis caching

---

## Summary

This architecture document provides multiple views of the system:

1. **High-Level**: User interfaces → Services → External APIs
2. **Data Flow**: Request → Feature Extraction → Prediction → Explanation
3. **Feature Pipeline**: 70D feature composition and extraction
4. **Model Architecture**: XGBoost ensemble with SHAP
5. **Production System**: FastAPI + Streamlit deployment
6. **Phase Progression**: CRAWL → WALK → RUN evolution
7. **Component Interactions**: Dependency graph across all modules
8. **Deployment**: Local development and production options

**Key Architectural Decisions**:
- **Modular Design**: Clear separation between feature extraction, prediction, and explanation
- **Caching Strategy**: File-based caching for Earth Engine API calls
- **Multiple Interfaces**: REST API, Dashboard, Python SDK for different users
- **Interpretability First**: SHAP explanations integrated at architecture level
- **Spatial Validation**: Built into training pipeline to prevent leakage

**Production Status**: ✅ **READY**
- All components implemented and tested
- 0.913 AUROC on hard validation sets
- <10s latency for single predictions
- Comprehensive documentation

---

**Related Documents**:
- Learning Journey: `learning_journey_crawl_to_run.md`
- User Guide: `run_phase_user_guide.md`
- Cleanup Summary: `repository_cleanup_summary.md`

**Last Updated**: 2025-10-24
**Version**: 1.0
