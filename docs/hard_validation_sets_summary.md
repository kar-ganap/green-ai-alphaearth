# Hard Validation Sets - Complete Summary

**Date:** October 16, 2025
**Phase:** WALK - Model Evaluation on Challenging Cases

## Table of Contents
1. [Context and Motivation](#context-and-motivation)
2. [Hard Validation Set Design](#hard-validation-set-design)
3. [Implementation](#implementation)
4. [Evaluation Results](#evaluation-results)
5. [Key Findings](#key-findings)
6. [Next Steps](#next-steps)

---

## Context and Motivation

### The Problem
The baseline model showed excellent performance on the original test set:
- **ROC-AUC:** 98.3%
- **Accuracy:** 95.5%
- **Recall:** 91.7%

However, this test set was drawn from the same distribution as the training data. We needed to understand model performance on **genuinely challenging cases** that reflect real-world operational scenarios.

### Research Foundation
Based on scholarly literature on challenging deforestation detection cases, we identified key failure modes:
1. **Selective logging** (most challenging)
2. **Small-scale clearing** (< 0.5-5 ha)
3. **Forest degradation** vs deforestation
4. **Secondary forest** re-clearing
5. **Fire vs clearing** confusion
6. **Fragmented/edge** clearing

### Use Case Alignment
We designed hard validation sets to align with four distinct operational use cases, prioritized as:
1. **Rapid Response / Law Enforcement** (highest priority)
2. **Comprehensive Monitoring / Carbon Accounting**
3. **Risk Ranking / Prioritization**
4. **Edge Cases / Failure Analysis**

---

## Hard Validation Set Design

### Set 1: Rapid Response (Priority 1)
**Purpose:** Early warning system for law enforcement patrol routing

**Composition:** 28 samples (19 clearing, 9 intact)
- Small-scale clearings (< 5 ha)
- Edge expansion patterns (gradual encroachment)
- Hard negatives (edge forests that remain intact)

**Key Metrics:**
- Precision @ confidence thresholds (alert quality)
- Top-K precision (patrol prioritization)

**Sampling Strategy:**
```python
# Small-scale clearings
loss_area < 5 ha
year_range: 2015-2023

# Edge expansion (hard negatives)
distance_to_forest_edge < 90m
stable forests
```

### Set 2: Risk Ranking (Priority 3)
**Purpose:** Risk stratification for resource allocation

**Composition:** 43 samples (8 clearing, 35 intact)
- Risk-stratified: High (hotspots) / Medium (edges) / Low (bastions)
- Imbalanced to reflect real-world risk distribution

**Key Metrics:**
- NDCG (Normalized Discounted Cumulative Gain)
- Calibration curves
- Risk-stratified performance

**Sampling Strategy:**
```python
# High risk: Active deforestation zones
hansen_loss_2015_2023 > 10% within 10km radius

# Low risk: Protected areas
protected_areas OR low_deforestation_regions
```

### Set 3: Comprehensive (Priority 2)
**Purpose:** Complete monitoring for carbon accounting and MRV

**Composition:** 70 samples (20 clearing, 50 intact)
- Size-stratified: Small / Medium / Large clearings
- Geographic diversity across all regions
- Stable forests for false positive assessment

**Key Metrics:**
- Size-stratified performance
- Balanced accuracy
- Area-weighted metrics

**Sampling Strategy:**
```python
# Size stratification
small:  0.5-5 ha
medium: 5-20 ha
large:  > 20 ha

# Geographic balance
Amazon (Para, Madre de Dios)
Congo Basin (DRC, Republic of Congo)
Southeast Asia (Sumatra)
```

### Set 4: Edge Cases (Priority 4)
**Purpose:** Identify systematic failure modes

**Composition:** 22 samples (10 clearing, 12 intact)
- Very small patches (< 1 ha)
- Fire-prone regions (fire vs clearing confusion)
- Edge intact forests (hardest negatives)

**Key Metrics:**
- Per-challenge-type performance
- Hardest cases identification

**Sampling Strategy:**
```python
# Challenge types
small_scale:  < 1 ha patches
fire_prone:   regions with active fire detections
edge_intact:  < 90m from forest edge, stable
```

### Sampling Challenges and Solutions

**Challenge 1: Limited samples in recent years (2020-2023)**
- **Solution:** Expanded year range to 2015-2023
- **Result:** Increased samples from 102 to 173 total

**Challenge 2: Restrictive filters yielding empty results**
- **Problem:** `connectedPixelCount()` filter failed
- **Solution:** Removed filter, used direct area calculations
- **Result:** Successful sampling across all regions

**Challenge 3: Fire + loss intersection too strict**
- **Problem:** No samples in fire-prone regions
- **Solution:** Relaxed to sample all loss pixels in fire-prone areas
- **Result:** 5 fire-prone samples captured

---

## Implementation

### File Structure
```
src/walk/
├── 01b_hard_validation_sets.py          # Sampling script
├── 01c_extract_features_for_hard_sets.py # Feature extraction
└── 03_evaluate_all_sets.py              # Evaluation framework

data/processed/
├── hard_val_rapid_response.pkl          # 28 samples (raw)
├── hard_val_risk_ranking.pkl            # 43 samples (raw)
├── hard_val_comprehensive.pkl           # 70 samples (raw)
├── hard_val_edge_cases.pkl              # 22 samples (raw)
├── hard_val_rapid_response_features.pkl # 28 samples (enriched)
├── hard_val_risk_ranking_features.pkl   # 43 samples (enriched)
├── hard_val_comprehensive_features.pkl  # 70 samples (enriched)
└── hard_val_edge_cases_features.pkl     # 22 samples (enriched)

results/walk/
└── evaluation_all_sets.json             # Complete evaluation results
```

### Feature Extraction Pipeline
For each sample, we extract:

**1. AlphaEarth Embeddings (64-dimensional)**
- Q1: year-1, June (9-12 months before clearing)
- Q2: year, March (6-9 months before)
- Q3: year, June (3-6 months before)
- Q4: year, September (0-3 months before)

**2. Temporal Features**
```python
distances:     L2 distance from Q1 baseline for each timepoint
velocities:    Change between consecutive timepoints (Q1→Q2, Q2→Q3, Q3→Q4)
accelerations: Change in velocities
trend_consistency: Fraction of increasing distance steps
```

**Extraction Results:**
- Rapid Response: 28/28 samples ✓ (31 seconds)
- Risk Ranking: 43/43 samples ✓ (48 seconds)
- Comprehensive: 70/70 samples ✓ (77 seconds)
- Edge Cases: 22/22 samples ✓ (6 seconds)

**Total:** 163/163 samples successfully extracted (100% success rate)

---

## Evaluation Results

### Performance Comparison

| Validation Set | ROC-AUC | Accuracy | Precision | Recall | Samples |
|----------------|---------|----------|-----------|--------|---------|
| **Original Test** | 98.3% | 95.5% | 100.0% | 91.7% | 22 |
| **Rapid Response** | 65.5% | 50.0% | 77.8% | 36.8% | 28 |
| **Risk Ranking** | 73.2% | 79.1% | 42.9% | 37.5% | 43 |
| **Comprehensive** | 73.4% | 71.4% | 50.0% | **10.0%** | 70 |
| **Edge Cases** | **51.7%** | 45.5% | 0.0% | **0.0%** | 22 |

### Set 1: Rapid Response - Moderate Failure

**Confusion Matrix:**
```
              Predicted
              Clear  Intact
Actual Clear    7      12      ← Missed 63% of clearings!
       Intact   2       7
```

**Confidence Threshold Analysis:**
```
70% threshold: 66.7% precision, 10.5% recall, 3 alerts
80% threshold: NO ALERTS (0 predictions)
90% threshold: NO ALERTS (0 predictions)
```

**Top-K Ranking (for patrol routing):**
```
Top-5:  80% precision (4/5 are clearings)
Top-10: 80% precision (8/10 are clearings) ← Most useful!
Top-20: 75% precision (15/20 are clearings)
```

**Key Insights:**
- ✅ Model can prioritize well (80% precision in top-10)
- ❌ Lacks confidence for automated alerts
- ❌ Misses 63% of clearings overall

### Set 2: Risk Ranking - Poor Ranking Quality

**Confusion Matrix:**
```
              Predicted
              Clear  Intact
Actual Clear    3       5      ← Only caught 3/8 clearings
       Intact   4      31
```

**NDCG Scores (ideal = 1.0):**
```
NDCG@5:  0.277  (poor)
NDCG@10: 0.297  (poor)
NDCG@20: 0.480  (below average)
```

**Risk-Stratified Performance:**
```
High risk areas:   37.5% recall (missed 5/8 clearings)
Medium risk areas:  0.0% recall (no ground truth clearings)
Low risk areas:     0.0% recall (correct - no clearings)
```

**Calibration Issue:**
```
Bin      Predicted  Actual   Samples
0.8-1.0    98.6%      0%        1    ← Severely overconfident!
0.6-0.8    71.6%     60%        5
0.4-0.6    47.5%      0%        2
0.2-0.4    31.2%     23%       13
0.0-0.2    11.2%      9%       22
```

**Key Insights:**
- ❌ NDCG < 0.5 means poor ranking quality
- ❌ Overconfident predictions (98.6% → 0% actual)
- ❌ Cannot effectively stratify risk levels

### Set 3: Comprehensive - Catastrophic Recall Failure

**Confusion Matrix:**
```
              Predicted
              Clear  Intact
Actual Clear    2      18      ← Missed 90% of clearings!
       Intact   2      48
```

**Size-Stratified Performance:**
```
Small patches (< 5 ha):    16.7% recall (1/6 detected)
Medium patches (5-20 ha):  11.1% recall (1/9 detected)
Large patches (> 20 ha):   Not enough samples
```

**Balanced Accuracy:** 53% (barely better than random)

**Key Insights:**
- ❌ 10% recall is catastrophic for carbon accounting
- ❌ Size bias: smaller clearings completely missed
- ✅ 50% precision means few false positives when it does alert
- ❌ Cannot be used for comprehensive monitoring in current state

### Set 4: Edge Cases - Complete Failure

**Confusion Matrix:**
```
              Predicted
              Clear  Intact
Actual Clear    0      10      ← Missed EVERY clearing!
       Intact   2      10
```

**Challenge Type Performance:**
```
Challenge Type      Error Rate  Detected
Small-scale (<1ha)    100%       0/5
Fire-prone regions    100%       0/5
Edge intact (FP)       17%       2/12 false positives
```

**Hardest Types (sorted by error rate):**
1. Small-scale clearing: 100% miss rate
2. Fire-prone regions: 100% miss rate
3. Edge intact forests: 17% false positive rate

**Key Insights:**
- ❌ ROC-AUC of 51.7% means near-random performance
- ❌ Model is completely blind to edge cases
- ❌ Small-scale and fire-prone clearing undetectable
- ❌ Systematic failure mode identified

---

## Key Findings

### What the Baseline Model Learned
✅ Successfully detects **large, obvious clearing events** in typical conditions
✅ Works well when deforestation patterns match training distribution
✅ Can prioritize locations reasonably well for patrol routing (top-10: 80% precision)

### Critical Failure Modes

**1. Small-Scale Clearing (Catastrophic)**
- **Evidence:** 100% miss rate on patches < 1 ha, 83-89% miss rate on patches < 5 ha
- **Impact:** Makes model unsuitable for early warning systems
- **Root Cause:** Temporal features from 64-dim embeddings lack spatial resolution

**2. Fire-Prone Regions (Catastrophic)**
- **Evidence:** 100% miss rate in fire-prone regions
- **Impact:** Cannot distinguish fire from clearing
- **Root Cause:** Similar spectral signatures, no fire-specific features

**3. Size Bias (Severe)**
- **Evidence:** Recall drops from 91.7% → 10-17% for small/medium patches
- **Impact:** Systematically underestimates deforestation area
- **Root Cause:** Model trained on size-diverse but temporally-focused features

**4. Calibration Issues (Severe)**
- **Evidence:** 98.6% predicted probability → 0% actual in highest confidence bin
- **Impact:** Cannot trust model confidence scores for risk ranking
- **Root Cause:** Simple logistic regression poorly calibrated on hard cases

**5. Low Confidence (Moderate)**
- **Evidence:** Zero predictions above 80% threshold
- **Impact:** Cannot support automated alert systems
- **Root Cause:** Model uncertainty on out-of-distribution samples

### Use-Case Impact Assessment

**Option 1: Rapid Response / Law Enforcement** (Priority 1)
- **Current State:** ❌ FAILED - 36.8% recall, no high-confidence alerts
- **Usable For:** Patrol routing via top-10 prioritization (80% precision)
- **Not Usable For:** Automated alerts, early warning
- **Gap to Operational:** Need 80%+ recall for effective deployment

**Option 2: Comprehensive Monitoring / Carbon** (Priority 2)
- **Current State:** ❌ FAILED - 10% recall is catastrophic
- **Usable For:** Nothing at current performance
- **Not Usable For:** Carbon accounting, MRV, complete monitoring
- **Gap to Operational:** Need 90%+ recall, size-invariant detection

**Option 3: Risk Ranking / Prioritization** (Priority 3)
- **Current State:** ❌ FAILED - NDCG@10 of 0.297, poor calibration
- **Usable For:** Very rough prioritization only
- **Not Usable For:** Risk scores, resource allocation, threat assessment
- **Gap to Operational:** Need NDCG > 0.7, calibrated probabilities

**Option 4: Edge Cases / Failure Analysis** (Priority 4)
- **Current State:** ❌ FAILED - 0% recall, random performance
- **Usable For:** Demonstrates systematic blind spots
- **Not Usable For:** Any operational edge case detection
- **Gap to Operational:** Need specialized models for challenging scenarios

---

## Next Steps

### Immediate Priorities (in order)

#### 1. Add Spatial Features (CRITICAL for small-scale detection)
**Why:** Current model only uses temporal features from 64-dim embeddings; adding spatial context could help detect small clearing patterns

**What to extract:**
- **Neighborhood features:** Statistics from surrounding pixels (mean, std, gradients)
- **Texture features:** GLCM (Gray Level Co-occurrence Matrix) for pattern detection
- **Fragmentation metrics:** Edge density, patch connectivity
- **Multi-scale embeddings:** Extract at multiple spatial resolutions

**Expected Impact:**
- Improve small-scale detection (currently 0-17% recall → target 60%+)
- Reduce size bias in comprehensive monitoring
- Better edge detection

**Implementation:**
```python
# New script: src/walk/01d_extract_spatial_features.py
def extract_spatial_features(client, lat, lon, date, radius_m=500):
    """
    Extract spatial context features around a location.

    Returns:
    - neighborhood_stats: Mean, std, gradients in 500m radius
    - texture_features: GLCM metrics
    - fragmentation: Edge density, connectivity
    - multiscale_embeddings: Features at 30m, 100m, 300m
    """
```

#### 2. Build Fire vs Clearing Classifier
**Why:** 100% miss rate in fire-prone regions indicates confusion between fire and clearing

**Approach:**
- Use MODIS/VIIRS fire detections as additional input
- Extract temporal burn signature (rapid change + recovery)
- Train binary classifier: fire vs clearing

**Expected Impact:**
- Reduce false negatives in fire-prone regions (currently 100% → target < 30%)

#### 3. Implement Model Ensembles
**Why:** Single logistic regression model lacks capacity for complex patterns

**Approach:**
- Combine multiple models: Logistic Regression + Random Forest + XGBoost
- Use different feature subsets for each model
- Ensemble predictions with calibrated voting

**Expected Impact:**
- Improve overall robustness across all validation sets
- Better calibration of probability scores

#### 4. Calibration Improvements
**Why:** Model is overconfident (98.6% predicted → 0% actual)

**Approach:**
- Apply Platt scaling or isotonic regression
- Calibrate on held-out validation data
- Evaluate calibration curves on each hard validation set

**Expected Impact:**
- Enable reliable risk ranking (NDCG > 0.7)
- Support confidence-based alerting

#### 5. Dataset Augmentation
**Why:** Only 173 hard validation samples may be insufficient for training

**Approach:**
- Expand year range to 2010-2023 (currently 2015-2023)
- Increase samples per validation set (target: 100+ per set)
- Add more geographic regions (Central America, West Africa, Borneo)

**Expected Impact:**
- More robust evaluation
- Better coverage of edge cases

### Research Questions to Explore

1. **Feature Importance:** Which spatial vs temporal features matter most for small-scale detection?
2. **Embedding Resolution:** Do higher-resolution embeddings (10m Sentinel-2) outperform 30m Landsat?
3. **Temporal Windows:** Are quarterly snapshots optimal, or should we use monthly/biweekly?
4. **Transfer Learning:** Can we fine-tune AlphaEarth embeddings on deforestation-specific data?

### Success Metrics (Targets for Next Iteration)

| Use Case | Current | Target | Critical Metric |
|----------|---------|--------|-----------------|
| Rapid Response | 36.8% recall | 80% recall | Recall @ 70% precision |
| Comprehensive | 10% recall | 90% recall | Size-invariant recall |
| Risk Ranking | NDCG@10: 0.30 | NDCG@10: 0.70 | Ranking quality |
| Edge Cases | 0% recall | 50% recall | Per-type coverage |

### Timeline Estimate

**Week 1:** Implement spatial feature extraction
**Week 2:** Train models with spatial + temporal features
**Week 3:** Build fire classifier + ensemble models
**Week 4:** Calibration improvements + comprehensive re-evaluation
**Week 5:** Dataset augmentation + final validation

---

## Appendix: Technical Details

### Evaluation Script Usage
```bash
# Extract features for all hard validation sets
uv run python src/walk/01c_extract_features_for_hard_sets.py --set all

# Run comprehensive evaluation
uv run python src/walk/03_evaluate_all_sets.py

# View results
cat results/walk/evaluation_all_sets.json
```

### Sample Data Structure
```python
# Enriched sample format
{
    'lat': -12.34,
    'lon': -56.78,
    'year': 2021,
    'label': 1,  # 0 = intact, 1 = clearing
    'embeddings': {
        'Q1': np.array([...]),  # 64-dim
        'Q2': np.array([...]),
        'Q3': np.array([...]),
        'Q4': np.array([...])
    },
    'features': {
        'distances': {'Q1': 0.0, 'Q2': 0.12, 'Q3': 0.34, 'Q4': 0.89},
        'velocities': {'Q1_Q2': 0.12, 'Q2_Q3': 0.22, 'Q3_Q4': 0.55},
        'accelerations': {'Q1_Q2_Q3': 0.10, 'Q2_Q3_Q4': 0.33},
        'trend_consistency': 1.0
    }
}
```

### Geographic Distribution
```
Amazon (Brazil, Peru): 45% of samples
Congo Basin (DRC, Rep. of Congo): 40% of samples
Southeast Asia (Indonesia): 15% of samples
```

---

**Last Updated:** October 16, 2025
**Next Review:** After spatial feature implementation
