# Progressive Temporal Validation Plan

## Overview

Testing whether the deforestation detection model generalizes to future years using **progressive validation**:

- **Phase 1**: Train on 2020+2021 → Test on 2022
- **Phase 2**: Train on 2020+2021+2022 → Test on 2023
- **Phase 3**: Train on 2020+2021+2022+2023 → Test on 2024

This approach tests the realistic production scenario where models are continuously retrained with new data.

## Workflow

### Step 1: Collect Temporal Validation Samples

For each test year (2022, 2023, 2024), collect ~100 samples (50 clearing + 50 intact):

**Clearing samples:**
- Use FIRMS fire data API to identify fire-cleared areas
- Filter by confidence (>80%) and FRP (>10.0)
- Sample from diverse tropical regions
- Ensure spatial separation (>10km from training/validation)

**Intact samples:**
- Sample from intact forest regions
- Use same spatial grid approach as training data
- Ensure no deforestation through target year

**Data format:**
```python
{
    'lat': float,
    'lon': float,
    'year': int,
    'label': 0/1,  # 0=intact, 1=clearing
    'source': 'temporal_val_2022',
}
```

### Step 2: Extract Features

For collected samples, extract:
1. **Annual features** (3D): pre/post/delta magnitude using `extract_dual_year_features()`
2. **Multiscale features** (66D): coarse landscape embeddings using AlphaEarth API

**Combined feature vector**: 69D (3 + 66)

###  Step 3: Progressive Training & Evaluation

For each phase:

1. **Load training data**:
   - Phase 1: Existing 685 samples (2020+2021)
   - Phase 2: Add 2022 temporal validation samples
   - Phase 3: Add 2023 temporal validation samples

2. **Train model**:
   - Random Forest with GridSearchCV (432 combinations)
   - StratifiedKFold (5-fold)
   - Equal weighting initially (check for temporal drift)

3. **Evaluate**:
   - Use optimal thresholds from threshold optimization:
     - risk_ranking: 0.070
     - rapid_response: 0.608
     - comprehensive: 0.884
     - edge_cases: 0.910
   - Compute comprehensive metrics (precision, recall, F1, F2, F0.5, ROC-AUC, PR-AUC)
   - Assess use-case-specific targets

### Step 4: Temporal Drift Analysis

Compare performance across years:

```
Year    ROC-AUC   Recall   Precision   Target
2022    0.XXX     0.XXX    0.XXX       ✓/✗
2023    0.XXX     0.XXX    0.XXX       ✓/✗
2024    0.XXX     0.XXX    0.XXX       ✓/✗
```

**Drift detection**:
- If >10-20% performance degradation → temporal drift detected
- Recommendations:
  - Temporal weighting (downweight old data)
  - Dropping data >2 years old
  - Year-specific sample weights

## Current State

**Completed**:
- ✓ Threshold optimization (3/4 targets met)
- ✓ Training data ready (685 samples, 2020+2021)
- ✓ Model saved: `walk_model_rf_all_hard_samples.pkl`
- ✓ Optimal thresholds saved
- ✓ FIRMS data available (2000-2025)

**Next Steps**:
1. Collect 2022 samples (~100)
2. Extract features for 2022 samples
3. Run Phase 1 training and evaluation
4. Repeat for 2023 and 2024

## Implementation Scripts

Create three scripts following existing codebase pattern:

### 31a_collect_temporal_samples.py
Collects samples for a specific year (following pattern from `27_collect_all_hard_samples.py`):
- Generate candidate locations
- Check spatial separation
- Extract annual features using `extract_dual_year_features()`
- Save samples (without multiscale features)

### 31b_extract_temporal_features.py
Extracts multiscale features (following pattern from `28_extract_features_all_hard_samples.py`):
- Load collected samples
- Extract 66D coarse multiscale embeddings
- Combine with annual features → 69D
- Save augmented samples

### 31c_progressive_temporal_validation.py
Runs progressive validation:
- Load training data (cumulative)
- Train Random Forest
- Evaluate on test year
- Save results and model
- Analyze temporal drift

## Success Criteria

**Temporal generalization targets**:
- Maintain 3/4 use-case targets across years
- No >20% performance degradation
- ROC-AUC remains ≥ 0.60 for edge_cases

**If drift detected**:
- Try temporal weighting
- Consider retraining frequency
- Investigate systematic shifts in fire patterns
