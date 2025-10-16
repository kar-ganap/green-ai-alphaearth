# Temporal Resolution Investigation & Augmentation Strategy

**Date:** 2025-10-15
**Status:** In Progress
**Priority:** HIGH - Critical for validating precursor signal claims

---

## Executive Summary

Our CRAWL Test 2 showed strong temporal signal (p<0.000001, AUC=0.894), but annual AlphaEarth embeddings create ambiguity: are we detecting **precursor activities** or **early-year clearing**? This document outlines our investigation and proposed validation/augmentation strategies.

---

## 1. The Problem: Temporal Ambiguity

### What We Observed
- **Test 2 Result:** Embedding at year Y-1 significantly differs from year Y (when clearing occurred)
- **Claimed:** "Embeddings show precursor signal before clearing"
- **Actual:** "Embedding from year Y-1 differs from year Y"

### The Ambiguity

Hansen GFC labels only tell us: "Clearing occurred somewhere in year 2020"
- Could be January 2020 → "2020 embedding" likely includes post-clearing state
- Could be December 2020 → "2020 embedding" might not capture it
- **We don't know which!**

### Three Possible Scenarios

| Scenario | What Y-1 Embedding Captures | What This Means | Value for Prediction |
|----------|---------------------------|-----------------|---------------------|
| **A: True Precursor** | Roads, camps, selective logging in Q4 2019 | Real preparation activities | ✓ High - Can predict months ahead |
| **B: Early Detection** | Clearing happened in Q1 2020, captured in annual composite | Just detection, not prediction | ✗ Low - Not much lead time |
| **C: Mixed** | Precursors in late 2019 + early 2020 clearing | Both signals present | ~ Medium - Some lead time |

**Most Likely:** Scenario C (mixed signal)

---

## 2. What We Learned About AlphaEarth

### Key Technical Findings

#### Data Sources (Multi-Modal Fusion)
From the [AlphaEarth Foundations paper](https://arxiv.org/html/2507.22291v1):

| Source | Resolution | Revisit Time | What It Captures |
|--------|-----------|--------------|------------------|
| Sentinel-2 | 10m | Every 5 days | Optical (RGB, NIR) |
| Landsat 8/9 | 30m | Every 16 days | Optical |
| Sentinel-1 | 10m | Every 6-12 days | **Radar (C-band) - sees through clouds!** |
| ALOS PALSAR | 25m | Periodic | Radar (L-band) - forest structure |
| GEDI | 25m | Periodic | LiDAR - canopy height |
| ERA5-Land | 9km | Hourly | Climate variables |

**Critical Insight:** AlphaEarth has access to **sub-weekly observations** from multiple sensors, including cloud-penetrating radar.

#### Temporal Aggregation Mechanism

**Key Quote from Paper:**
> "Creates annual summaries using conditional metadata... explicitly separating the input intervals from those used for the temporal summary"

**What This Means:**
- AlphaEarth **observes** at 5-12 day frequency
- But **outputs** annual embeddings
- The model **learns** to aggregate temporal information
- It's trained to capture "temporal trajectories" of surface variables

**Training Performance:**
- 78.4% accuracy on land cover change detection
- Designed specifically for "supervised and unsupervised change detection"

### The Good News

AlphaEarth is NOT a dumb annual snapshot! It's:
1. ✓ Aggregating daily-to-weekly observations throughout the year
2. ✓ Using cloud-penetrating radar that works year-round in Amazon
3. ✓ Explicitly trained on temporal change detection tasks
4. ✓ Encoding temporal trajectories, not just static snapshots

**Implication:** The embeddings CAN capture precursor signals (roads, selective logging) if they're visible in the sub-annual imagery.

### The Bad News

The annual aggregation still creates ambiguity:
1. ✗ We don't know how the model weights different months
2. ✗ Cloud cover varies seasonally → some months contribute more than others
3. ✗ No documentation on temporal pooling mechanism
4. ✗ We can't distinguish "late 2019 precursors" from "early 2020 clearing"

---

## 3. Augmentation Strategies

### Strategy A: GLAD Alerts for Precise Dates ⭐ **HIGHEST PRIORITY**

**Goal:** Disambiguate precursor signal from early detection

#### GLAD-L (Landsat-based)
- **Resolution:** 30m
- **Temporal:** Weekly updates
- **Coverage:** Pan-tropical (30°N-30°S)
- **Earth Engine:** Available as image collection
- **Key Value:** Week-level dates for clearing events

#### GLAD-S2 (Sentinel-2-based)
- **Resolution:** 10m (matches AlphaEarth!)
- **Temporal:** Weekly updates
- **Coverage:** Amazon basin primary humid tropical forest
- **Key Value:** 6.25× more pixels than GLAD-L

#### Implementation Plan

```python
# Earth Engine dataset
glad_l = ee.ImageCollection('projects/glad/alert/2024')
glad_s2 = ee.ImageCollection('projects/GLADS2/alert/UpdResult')

# Each pixel has date value (days since 2015-01-01)
# Extract clearing date for each location
def get_clearing_quarter(date_value):
    """Convert GLAD date to quarter"""
    clearing_date = datetime(2015, 1, 1) + timedelta(days=date_value)
    quarter = (clearing_date.month - 1) // 3 + 1
    return clearing_date.year, quarter

# Re-run Test 2 with precise dates
def test_temporal_signal_with_quarters():
    """
    Test if Y-1 embedding predicts Q1 vs Q4 clearings differently.

    Expected:
    - If Y-1 predicts Q4 clearings: ✓ Precursor signal
    - If Y-1 predicts Q1 clearings: ✗ Just early detection
    """
    q1_clearings = [c for c in clearings if c['quarter'] == 1]
    q4_clearings = [c for c in clearings if c['quarter'] == 4]

    auc_q1 = test_prediction(y_minus_1_embeddings, q1_clearings)
    auc_q4 = test_prediction(y_minus_1_embeddings, q4_clearings)

    print(f"Q1 clearings AUC: {auc_q1:.3f}")
    print(f"Q4 clearings AUC: {auc_q4:.3f}")

    if auc_q4 > auc_q1 + 0.1:
        return "✓ TRUE PRECURSOR SIGNAL"
    elif auc_q1 > auc_q4 + 0.1:
        return "✗ EARLY DETECTION ONLY"
    else:
        return "~ MIXED SIGNAL"
```

**Expected Outcomes:**

| Scenario | Q1 AUC | Q4 AUC | Interpretation |
|----------|--------|--------|----------------|
| True Precursor | 0.65 | 0.92 | Y-1 embedding predicts late-year clearing (roads built in Y-1) |
| Early Detection | 0.88 | 0.68 | Y-1 embedding "predicts" early clearing (already started in Y-1!) |
| Mixed | 0.85 | 0.87 | Both signals present |

---

### Strategy B: Monthly NDVI Time Series (Optical)

**Goal:** Add sub-annual temporal resolution for gradual degradation signals

#### Implementation

```python
def extract_ndvi_features(location, reference_year):
    """Extract monthly NDVI from Sentinel-2 for 12 months before reference"""

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
           .filterBounds(point) \
           .filterDate(f'{reference_year-1}-01-01', f'{reference_year}-01-01')

    # Monthly composites (median to handle clouds)
    monthly_ndvi = []
    for month in range(1, 13):
        start = f'{reference_year-1}-{month:02d}-01'
        end = f'{reference_year-1}-{month:02d}-28'

        ndvi = s2.filterDate(start, end) \
                 .median() \
                 .normalizedDifference(['B8', 'B4'])

        monthly_ndvi.append(ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).get('nd').getInfo())

    return monthly_ndvi  # List of 12 values
```

#### Derived Features

From 12 monthly NDVI values, extract:

| Feature | Calculation | What It Captures |
|---------|-------------|------------------|
| **3-month velocity** | `mean(ndvi[-3:]) - mean(ndvi[-6:-3])` | Recent change |
| **6-month velocity** | `mean(ndvi[-6:]) - mean(ndvi[:-6])` | Medium-term trend |
| **12-month velocity** | `ndvi[-1] - ndvi[0]` | Long-term trend |
| **Volatility** | `std(ndvi)` | Stability vs fluctuation |
| **Seasonal amplitude** | `max(ndvi) - min(ndvi)` | Natural vs disturbed |
| **Trend** | Linear regression slope | Degradation rate |

**Expected Value:**
- Gradual degradation (selective logging) shows up months before clear-cutting
- Legal clearing may show NDVI decline as permits are obtained

---

### Strategy C: Monthly Radar Time Series (All-Weather)

**Goal:** See through clouds, detect structural changes (roads, selective logging)

#### Implementation

```python
def extract_radar_features(location, reference_year):
    """Extract monthly VV/VH backscatter from Sentinel-1"""

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
           .filterBounds(point) \
           .filterDate(f'{reference_year-1}-01-01', f'{reference_year}-01-01') \
           .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Monthly composites
    monthly_vv = []
    monthly_vh = []

    for month in range(1, 13):
        start = f'{reference_year-1}-{month:02d}-01'
        end = f'{reference_year-1}-{month:02d}-28'

        composite = s1.filterDate(start, end).median()

        vv = composite.select('VV').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=10
        ).get('VV').getInfo()

        vh = composite.select('VH').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=10
        ).get('VH').getInfo()

        monthly_vv.append(vv)
        monthly_vh.append(vh)

    return monthly_vv, monthly_vh
```

#### Why Radar Matters

**Physical Basis:**
- **VV (co-polarization):** Sensitive to surface roughness
- **VH (cross-polarization):** Sensitive to volume scattering (canopy structure)

**What Changes Radar Backscatter:**
1. **Roads:** Smooth surfaces → decreased VV
2. **Selective logging:** Reduced canopy → decreased VH
3. **Clear-cutting:** Dramatic drop in both VV and VH

**Key Advantage:** Sentinel-1 operates in all weather conditions
- Amazon wet season: optical satellites see clouds, radar sees ground
- This fills the temporal gaps in optical data

---

### Strategy D: Check for Quarterly AlphaEarth Embeddings

**Goal:** Get 4× temporal resolution if available

#### Investigation Needed

**From the paper:**
> "Can generate embeddings for specific time periods with conditional metadata"

**Questions:**
1. Does Earth Engine API support date range queries for AlphaEarth?
2. Can we request Q1, Q2, Q3, Q4 embeddings separately?

**To Check:**
```python
# Try requesting quarterly embeddings
alphaearth = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

# Does this work?
q1_embedding = alphaearth.filterDate('2020-01-01', '2020-03-31') \
                         .first() \
                         .sample(point, scale=10)

# Or are embeddings truly annual only?
```

**Status:** 🔍 Investigation needed

---

## 4. Proposed Hybrid Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT LAYER                            │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ AlphaEarth   │  │  Sentinel-2  │  │  Sentinel-1  │  │
│  │  Embeddings  │  │  NDVI (12mo) │  │  VV/VH (12mo)│  │
│  │  (Annual)    │  │              │  │              │  │
│  │  64 dims     │  │  Optical     │  │  Radar       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐  │
│  │ AlphaEarth Features (4 features)                  │  │
│  │  - Velocity (year-over-year)                      │  │
│  │  - Acceleration (velocity change)                 │  │
│  │  - Recent vs historical                           │  │
│  │  - Directional consistency                        │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ NDVI Features (6 features)                        │  │
│  │  - 3-month velocity                               │  │
│  │  - 6-month velocity                               │  │
│  │  - 12-month trend                                 │  │
│  │  - Volatility                                     │  │
│  │  - Seasonal amplitude                             │  │
│  │  - Recent anomaly                                 │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Radar Features (4 features)                       │  │
│  │  - VV 6-month velocity                            │  │
│  │  - VH 6-month velocity                            │  │
│  │  - VV/VH ratio change                             │  │
│  │  - Texture change                                 │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Context Features (2 features)                     │  │
│  │  - Distance to road                               │  │
│  │  - Historical clearing nearby                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ↓
                ┌─────────────────┐
                │ XGBoost Model   │
                │ ~16 features    │
                └─────────────────┘
```

**Expected Benefits:**
- **AlphaEarth:** Rich semantic understanding, multi-sensor fusion
- **NDVI:** Gradual degradation signals (selective logging, permits)
- **Radar:** All-weather monitoring, structural changes (roads, camps)
- **Combined:** Distinguish clearing types, improve lead time

**Expected AUC:** 0.91-0.94 (vs 0.894 baseline)

---

## 5. Systematic Action Plan

### Phase 1: Validate Precursor Signal (CRITICAL) 🔴

**Goal:** Confirm we're detecting preparation, not just early clearing

- [ ] **Task 1.1:** Access GLAD-L alerts in Earth Engine
  - Dataset: `projects/glad/alert/2024`
  - Extract date values for our clearing locations
  - Convert to year + quarter

- [ ] **Task 1.2:** Re-run Test 2 with quarterly stratification
  - Separate Q1 clearings from Q4 clearings
  - Test if Y-1 embedding predicts Q4 better than Q1
  - Document AUC for each quarter

- [ ] **Task 1.3:** Analyze results
  - If Q4 AUC > Q1 AUC + 0.1: ✓ Precursor confirmed
  - If Q1 AUC > Q4 AUC + 0.1: ⚠️ Early detection
  - If similar: ~ Mixed signal

**Expected Time:** 2-3 hours

**Deliverable:**
- `docs/temporal_validation_results.md`
- Updated Test 2 script with quarterly analysis
- Clear statement on precursor signal existence

---

### Phase 2: Add Monthly Features (If Phase 1 Confirms Precursor)

**Goal:** Improve temporal resolution and performance

- [ ] **Task 2.1:** Implement Sentinel-2 NDVI extraction
  - Extract 12 monthly values
  - Handle cloud masking (use median)
  - Compute 6 derived features

- [ ] **Task 2.2:** Implement Sentinel-1 radar extraction
  - Extract VV/VH monthly time series
  - Compute 4 derived features
  - Handle ascending/descending passes

- [ ] **Task 2.3:** Test feature ablation
  - Baseline: AlphaEarth only (89.4%)
  - + NDVI features
  - + Radar features
  - + Both
  - Document which combinations help

**Expected Time:** 4-6 hours

**Deliverable:**
- Feature extraction scripts
- Ablation study results
- Updated model with augmented features

---

### Phase 3: Investigate Quarterly AlphaEarth (Optional)

- [ ] **Task 3.1:** Check Earth Engine API capabilities
  - Test date range queries on AlphaEarth dataset
  - Document if quarterly embeddings are accessible

- [ ] **Task 3.2:** If available, extract quarterly embeddings
  - Q1, Q2, Q3, Q4 for each year
  - Compare with annual embeddings

**Expected Time:** 1-2 hours

**Deliverable:**
- Documentation of AlphaEarth temporal capabilities
- Quarterly embeddings if available

---

## 6. Decision Criteria

### After Phase 1: Should We Continue?

| Phase 1 Result | Interpretation | Decision |
|----------------|----------------|----------|
| Q4 AUC >> Q1 AUC | Strong precursor signal | ✓ Proceed to Phase 2 |
| Q1 AUC >> Q4 AUC | Early detection only | ⚠️ Reframe as annual risk model |
| Similar AUC | Mixed signal | ✓ Proceed but with honest framing |

### After Phase 2: Did Augmentation Help?

| AUC Improvement | Interpretation | Decision |
|-----------------|----------------|----------|
| +0.03 or more | Significant improvement | ✓ Keep augmented features |
| +0.01 to +0.03 | Marginal improvement | ~ Keep if interpretable |
| <+0.01 | No improvement | ✗ Drop augmentation |

---

## 7. Honest Framing (Current Understanding)

### What We Can Claim Now (Conservative)

✅ **Defensible Claims:**
- "AlphaEarth embeddings predict which 10m pixels will show forest loss in year Y based on year Y-1 data, achieving 89.4% AUC"
- "The embeddings aggregate multi-sensor observations including cloud-penetrating radar throughout the year"
- "The model is trained on temporal change detection tasks"

❌ **Cannot Claim Yet:**
- "90-day advance warning" - annual resolution
- "Precursor signal detection" - not validated with precise dates
- "Causal relationship" - correlation doesn't prove causation

### What We Will Know After Phase 1

If quarterly analysis shows Q4 > Q1:
- ✅ Can claim: "Detects precursor activities (roads, selective logging) months before clear-cutting"
- ✅ Can claim: "Provides early warning for enforcement targeting"

If quarterly analysis shows Q1 ≈ Q4:
- ~ Can claim: "Predicts annual clearing risk from previous year, useful for resource planning"
- ⚠️ Must acknowledge: "Mix of precursor detection and early clearing capture"

---

## 8. Expected Outcomes

### Optimistic Scenario (Most Likely)

**Phase 1 Result:** Q4 AUC = 0.92, Q1 AUC = 0.78
- **Interpretation:** Strong precursor signal + some early detection
- **Implication:** Can provide 3-9 months warning for ~70% of clearings

**Phase 2 Result:** AUC improves to 0.93 with monthly features
- **Interpretation:** Sub-annual dynamics add value
- **Implication:** Better temporal resolution improves prediction

**Final System:**
- Predicts 1-year ahead risk with 93% AUC
- Distinguishes clearing types (gradual vs sudden)
- Provides quarterly risk scores

### Pessimistic Scenario (Less Likely)

**Phase 1 Result:** Q1 AUC = 0.89, Q4 AUC = 0.87
- **Interpretation:** Mostly detecting early clearing, limited precursor
- **Implication:** More like "early detection" than "prediction"

**Phase 2 Result:** Minimal improvement (<0.01 AUC)
- **Interpretation:** Annual features already capture signal
- **Implication:** Stick with simpler model

**Final System:**
- Annual risk model (not early warning)
- Still useful for planning and resource allocation
- Honest about limitations

---

## 9. References

**Primary Sources:**
1. [AlphaEarth Foundations Paper](https://arxiv.org/abs/2507.22291) - Technical details on training and architecture
2. [DeepMind Blog Post](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) - High-level overview
3. [GLAD Deforestation Alerts](https://glad.umd.edu/dataset/glad-forest-alerts) - Weekly deforestation detection
4. [Global Forest Watch](https://www.globalforestwatch.org/) - Operational deforestation monitoring

**Earth Engine Datasets:**
- AlphaEarth: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- GLAD-L: `projects/glad/alert/2024`
- GLAD-S2: `projects/GLADS2/alert/UpdResult`
- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED`
- Sentinel-1: `COPERNICUS/S1_GRD`

---

## 10. Next Steps

**Immediate Priority (Today):**
1. Start Phase 1, Task 1.1: Access GLAD alerts
2. Extract precise dates for our Test 2 clearing locations
3. Run quarterly stratification analysis

**This Week:**
- Complete Phase 1 validation
- Document findings in `temporal_validation_results.md`
- Make GO/NO-GO decision on augmentation

**Next Week (If Phase 1 Confirms Precursor):**
- Implement monthly feature extraction
- Run ablation study
- Update model with best performing features

---

## Status Tracking

| Phase | Task | Status | Owner | Completion Date | Notes |
|-------|------|--------|-------|-----------------|-------|
| Phase 1 | Access GLAD alerts | 🟢 Complete | - | 2025-10-15 | Implemented archived ImageCollection access |
| Phase 1 | Quarterly analysis | 🟡 **INCONCLUSIVE** | - | 2025-10-15 | Insufficient Q1/Q4 samples (need ≥3 each) |
| Phase 1 | Document findings | 🟢 Complete | - | 2025-10-15 | See [`phase1_glad_validation_summary.md`](./phase1_glad_validation_summary.md) |
| **Decision Point** | **Choose next step** | 🔴 **BLOCKED** | - | - | **Need to decide: Scale up (Option A), Alternative validation (B), or Proceed to Phase 2 (C)** |
| Phase 2 | NDVI extraction | ⚪ Blocked by decision | - | - | Awaiting Phase 1 decision |
| Phase 2 | Radar extraction | ⚪ Blocked by decision | - | - | Awaiting Phase 1 decision |
| Phase 2 | Feature ablation | ⚪ Blocked by decision | - | - | Awaiting Phase 1 decision |
| Phase 3 | Check quarterly API | 🔴 Not Started | - | - | Optional investigation |

**Phase 1 Results Summary:**
- ✅ Successfully accessed GLAD archived data (ImageCollections)
- ✅ Enriched 13/24 clearings (54% success rate)
- ⚠️ Quarterly distribution: Q1=1, Q2=2, Q3=8, Q4=2
- ❌ **INCONCLUSIVE**: Cannot compare Q1 vs Q4 (insufficient samples)

**Recommended Next Action:**
- **Option A (RECOMMENDED)**: Scale to 120+ samples → Expected 12-15 Q1/Q4 samples each
- **Option B**: Try alternative validation (month-level or baseline comparison)
- **Option C**: Accept temporal ambiguity and proceed to Phase 2/WALK phase

**Legend:**
- 🔴 Not Started
- 🟡 In Progress / Inconclusive
- 🟢 Complete
- ⚪ Blocked

---

**Last Updated:** 2025-10-15 17:45 UTC
**Next Review:** After Phase 1 decision is made
**Key Deliverables:**
- [`docs/phase1_glad_validation_summary.md`](./phase1_glad_validation_summary.md) - Detailed Phase 1 analysis
- `results/temporal_investigation/phase1_glad_validation.json` - Raw results
