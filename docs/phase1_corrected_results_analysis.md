# Phase 1 Corrected Results Analysis

**Date**: 2025-10-18
**Status**: Corrected annual features extracted and evaluated
**Critical Discovery**: Buggy quarterly features have been fixed

---

## Summary

After discovering and fixing the AlphaEarth annual embedding bug, we re-extracted Phase 1 features with correct annual approach (3D instead of buggy 17D) and re-evaluated the model.

**Key Finding**: Corrected model performs similarly to buggy model, both UNDERPERFORM significantly compared to temporal generalization baseline.

---

## The Bug (Fixed)

**Problem**: All "quarterly" feature extraction returned IDENTICAL embeddings because AlphaEarth provides ONE embedding per year, not quarterly.

**Buggy Features (17D)**:
- All quarterly features within a year were identical
- Created perfect collinearity (correlation = 1.0)
- Only 1 unique feature value replicated across 17 dimensions

**Corrected Features (3D)**:
```python
delta_1yr = ||emb(Y) - emb(Y-1)||      # Recent annual change
delta_2yr = ||emb(Y-1) - emb(Y-2)||    # Historical annual change
acceleration = delta_1yr - delta_2yr    # Change acceleration
```

---

## Results Comparison

### Buggy (17D) vs Corrected (3D) vs Temporal Generalization

| Validation Set | Buggy 17D | Corrected 3D | Change | Temporal Gen |
|----------------|-----------|--------------|--------|--------------|
| **risk_ranking** | 0.838 | 0.825 | -0.013 | **0.971** |
| **rapid_response** | 0.824 | 0.786 | -0.038 | **0.971** |
| **comprehensive** | 0.756 | 0.747 | -0.009 | **0.971** |
| **edge_cases** | 0.567 | 0.533 | -0.034 | **0.971** |

**Critical Observation**: Both buggy AND corrected Phase 1 models perform MUCH WORSE than temporal generalization (0.97 ROC-AUC).

---

## Why Did Temporal Generalization Work (0.971) But Phase 1 Fails (0.53-0.83)?

### Temporal Generalization Test
- **Train**: 2020 samples
- **Test**: 2021 samples
- **Geography**: SAME regions (within training area)
- **Scenarios**: STANDARD clearings (similar patterns)
- **Result**: 0.946-0.953 ROC-AUC

### Phase 1 Validation Sets
- **Train**: 2020-2022 mixed samples
- **Test**: Out-of-domain validation sets
- **Geography**: DIFFERENT regions (10km exclusion)
- **Scenarios**: CHALLENGING (edge cases, fire-prone, small clearings)
- **Result**: 0.533-0.825 ROC-AUC

### The Gap Explained

**Temporal gen (0.97) vs Phase 1 (0.53-0.83)**: ~20-40% performance drop

**Three factors**:

1. **Geographic generalization** (spatial shift):
   - Temporal gen: Same region, different year → Works well
   - Phase 1: Different region, different year → Harder

2. **Scenario difficulty**:
   - Temporal gen: Standard clearings
   - Phase 1: Edge cases, fire-prone, small (<1ha), fragmented

3. **Sample size**:
   - Temporal gen: 71 test samples (larger)
   - Edge cases: 23 samples (very small)

---

## Feature Importance Analysis

### Corrected Model (3D features)

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| **delta_1yr** | 1.305 | Recent annual change (Y to Y-1) - STRONGEST |
| **delta_2yr** | 1.213 | Historical change (Y-1 to Y-2) - STRONG |
| **acceleration** | 0.190 | Change in change rate - WEAK |

**Insight**: Both recent and historical annual deltas matter nearly equally. Acceleration adds little value.

### Buggy Model (17D features)

| Feature | Importance | Notes |
|---------|-----------|-------|
| delta_q1-q4, delta_mean, delta_max | 0.372 | All identical (perfect collinearity) |
| baseline features | 0.011 | All near-zero (no predictive value) |

**Insight**: Buggy model accidentally replicated the ONE useful feature 6 times, which logistic regression handled gracefully.

---

## Detection vs Prediction Framing

Based on temporal investigation findings (docs/temporal_investigation_findings.md), we confirmed:

**What AlphaEarth DOES**:
- ✓ Detects deforestation with 0-3 month lag (concurrent detection)
- ✓ Detects both fire-based (0.71 signal) and logging-based (0.46 signal) clearings
- ✓ Annual resolution matches the data source capability

**What AlphaEarth CANNOT**:
- ✗ Predict deforestation 4-6+ months in advance (Q4 test: p=0.065, not significant)
- ✗ Provide reliable long-term early warning
- ✗ Sub-annual temporal resolution

**Honest Framing**:
> "Detect forest cover loss with 0-12 month lag using annual cloud-penetrating satellite embeddings, enabling rapid response to deforestation events including both fire-based and logging-based clearing."

---

## Why Does Phase 1 Still Underperform?

Despite fixing the bug, Phase 1 validation performance remains poor. Why?

### Hypothesis 1: Geographic Shift
**Evidence**: Temporal gen (same geography) = 0.97, Phase 1 (different geography) = 0.53-0.83

**Explanation**: Annual delta features capture **local** land cover change patterns that don't generalize well across geographies.

### Hypothesis 2: Scenario Complexity
**Evidence**: Edge cases (0.533) << standard sets (0.747-0.825)

**Breakdown**:
- Small clearings (<1ha): Hard to detect with 10m resolution
- Fire-prone areas: Different change signature than mechanical clearing
- Fragmented landscapes: Mixed signals from surrounding pixels

### Hypothesis 3: Insufficient Features
**Evidence**: Only 3 features (delta_1yr, delta_2yr, acceleration)

**Missing**:
- Spatial context (neighborhood patterns)
- Multi-scale information (different pixel sizes)
- Contextual features (roads, fire history)

---

## Comparison to Original Baseline

**Original Validation Set Performance (Before Phase 1)**:
- risk_ranking: 0.850 ROC-AUC
- rapid_response: 0.824 ROC-AUC
- comprehensive: 0.758 ROC-AUC
- edge_cases: 0.583 ROC-AUC

**Corrected Phase 1 Model**:
- risk_ranking: 0.825 ROC-AUC (-0.025)
- rapid_response: 0.786 ROC-AUC (-0.038)
- comprehensive: 0.747 ROC-AUC (-0.011)
- edge_cases: 0.533 ROC-AUC (-0.050)

**Conclusion**: Phase 1 scaling did NOT improve performance. In fact, it slightly degraded across all sets.

---

## Recommendations

### What We Learned

1. **Annual features are correct but insufficient**: The 3D annual features match AlphaEarth's capability, but don't capture enough information for edge cases.

2. **Detection framing is appropriate**: We should frame this as detection (0-12 month lag), not prediction (4-6 month lead).

3. **Geographic generalization is hard**: Temporal gen (0.97) shows the features WORK, but spatial generalization is much harder.

4. **Edge cases need specialization**: Performance gap (0.97 → 0.53) suggests edge cases require different features.

### Next Steps

**Option 1: Add Spatial + Contextual Features (Recommended)**
- Multi-scale embeddings (100m, 500m, 1km)
- Spatial neighborhood patterns
- Contextual features (roads, fire history)
- **Expected**: 0.70-0.80 ROC-AUC on edge cases

**Option 2: Collect More Diverse Data**
- Scale to 800-1000 samples
- 50% edge cases (small, fire, fragmented)
- Better geographic coverage
- **Expected**: Marginal improvement (0.60-0.65)

**Option 3: Build Specialized Models (Phase 2)**
- Standard model: For normal clearings (0.97 works)
- Edge case model: With spatial/contextual features
- Simple routing: size < 1ha OR fire_prone → edge model
- **Expected**: 0.95+ for standard, 0.70-0.80 for edges

**Recommendation**: **Option 1** - Add spatial + contextual features to the Phase 1 model before specialization.

---

## Technical Details

**Training Data**:
- Samples: 589 (300 clearing, 289 intact)
- Features: 3D annual (delta_1yr, delta_2yr, acceleration)
- Model: Logistic Regression with StandardScaler
- Success rate: 98.2%

**Validation Sets**:
- risk_ranking: 46 samples (6 clearing, 40 intact)
- rapid_response: 27 samples (13 clearing, 14 intact)
- comprehensive: 69 samples (20 clearing, 49 intact)
- edge_cases: 23 samples (8 clearing, 15 intact)

**Output Files**:
- Features: `data/processed/walk_dataset_scaled_phase1_features.pkl`
- Model: `data/processed/walk_model_phase1.pkl`
- Results: `results/walk/phase1_evaluation.json`

---

## Conclusion

**The Bug Fix Was Necessary**: The quarterly feature extraction was fundamentally broken. We now have honest annual features.

**But Phase 1 Still Fails**: Even with corrected features, Phase 1 doesn't achieve the target (0.70+ on edge cases).

**The Gap Is Real**: Temporal generalization (0.97) proves the features WORK for same-geography, standard clearings. The Phase 1 gap (0.53-0.83) is due to geographic shift + scenario complexity.

**Path Forward**: Add spatial and contextual features (multi-scale embeddings, neighborhoods, roads, fire) to bridge the gap.

**Honest Framing**: This is a **detection system** (0-12 month lag), not a **prediction system** (4-6 month lead). AlphaEarth's annual resolution limits us to concurrent detection of year-of-clearing.
