# WALK Phase: Current Status and Next Steps

**Date:** 2025-10-22
**Purpose:** Document current state, what we've learned, and actionable hackathon path forward
**Context:** After completing temporal drift investigation and uniform 30% experiment

---

## Table of Contents

1. [Current Status Summary](#current-status-summary)
2. [What We've Completed](#what-weve-completed)
3. [Key Findings](#key-findings)
4. [What's Left for WALK](#whats-left-for-walk)
5. [Hackathon-Scoped Next Steps](#hackathon-scoped-next-steps)
6. [Experiment Options](#experiment-options)
7. [Decision Framework](#decision-framework)

---

## Current Status Summary

### WALK Phase Completion: ~75%

**Validation & Understanding:** 95% complete
- ✓ Progressive temporal validation (Phases 1-4)
- ✓ Drift investigation and decomposition
- ✓ Hard validation sets created
- ✓ Threshold optimization
- ✓ Feature-level distribution analysis

**Model Development:** 55% complete
- ✓ Random Forest baseline
- ✓ Scaled training data (685 samples)
- ✓ Multiscale features (69D)
- ⏳ Model diversity (XGBoost, ensemble)
- ⏳ Temporal adaptation
- ✗ Spatial features (not done)
- ✗ Q4 precursor testing (not done)

**Overall Assessment:**
> We built the most rigorous validation framework possible, discovered critical temporal drift, and quantified its causes. Now we need to apply what we learned to build the final model.

---

## What We've Completed

### 1. Data Collection ✓

**Training Data (2020-2023):**
- 685 samples with intentional diversity
- Distribution:
  - Standard clearings (>1 ha): 60%
  - Small clearings (<1 ha): 20%
  - Fire-prone: 10%
  - Edge expansion: 10%
- Spatially separated from validation sets (10km buffer)

**Test Data (2024):**
- 162 samples (81 clearing + 81 intact)
- Same diversity as training
- Uniform 30% Hansen threshold

**Hard Validation Sets:**
- risk_ranking: 46 samples
- rapid_response: 27 samples
- comprehensive: 69 samples
- edge_cases: 23 samples
- Total: 165 samples

**Uniform 30% Dataset (2020-2023):**
- 588 samples (for drift decomposition experiment)
- All at uniform 30% threshold
- Used to isolate sampling bias vs temporal drift

**Total Unique Samples:** ~1,100 across all datasets

---

### 2. Feature Engineering ✓

**69D Feature Space:**
- 3D annual features: delta_1yr, delta_2yr, acceleration
- 66D coarse multiscale features:
  - 64D AlphaEarth embedding (coarse resolution)
  - Heterogeneity metric
  - Range metric

**Feature Extraction Scripts:**
- `09_phase1_extract_features.py` - For training data
- `33_extract_features_2024.py` - For 2024 test data
- `39_extract_features_uniform_30pct.py` - For uniform 30% data
- `08_multiscale_embeddings.py` - Multiscale feature extraction

**What's NOT done:**
- Spatial features (neighborhood stats, edge proximity)
- Q4-specific features (precursor signal amplification)
- Fire history features
- Fragmentation metrics

---

### 3. Model Training ✓

**Current Model:**
- Random Forest with GridSearchCV
- 432 hyperparameter combinations tested
- 5-fold StratifiedKFold CV
- Trained on 685 samples (2020-2023)
- Best CV ROC-AUC: 0.981-0.982

**Hyperparameter Grid:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}
```

**Best Parameters:**
- n_estimators: 100
- max_depth: 10
- max_features: sqrt
- min_samples_leaf: 1
- min_samples_split: 2
- class_weight: balanced

---

### 4. Validation Framework ✓✓✓

**Progressive Temporal Validation:**

| Phase | Train Years | Test Year | Samples | ROC-AUC | Status |
|-------|------------|-----------|---------|---------|--------|
| **Phase 1** | 2020-2021 | 2022 | Train: 254, Test: 209 | 0.976 | ✓ Stable |
| **Phase 2** | 2020-2022 | 2023 | Train: 463, Test: 222 | 0.982 | ✓ Stable |
| **Phase 3** | 2020,2022,2023 | 2021 | Train: 545, Test: 140 | 0.981 | ✓ Stable |
| **Phase 4** | 2020-2023 | 2024 | Train: 685, Test: 162 | **0.796** | ⚠️ **18.9% DRIFT** |

**Key Finding:** Model generalizes well within 2020-2023 temporal range but fails on 2024.

---

### 5. Threshold Optimization ✓

**Use-Case-Specific Thresholds:**

| Use Case | Threshold | Target Metric | Performance (Phase 4, 2024) |
|----------|-----------|---------------|------------------------------|
| **risk_ranking** | 0.070 | Recall ≥ 0.90 | Precision: 0.639, Recall: 0.963 ✓ |
| **rapid_response** | 0.608 | Recall ≥ 0.90 | Precision: 0.719, Recall: 0.790 ✗ |
| **comprehensive** | 0.884 | Precision baseline | Precision: 0.778, Recall: 0.519 |
| **edge_cases** | 0.910 | ROC-AUC ≥ 0.65 | Precision: 0.824, Recall: 0.519 |

**All use cases ROC-AUC (2024):** 0.796

---

### 6. Drift Investigation ✓

**Feature Distribution Analysis:**
- Compared 2024 vs 2020-2023 feature distributions
- Kolmogorov-Smirnov tests (p < 0.01 threshold)
- **Result:** 46 out of 69 features showed significant shifts

**Breakdown:**
- Annual features: 3/3 shifted (100%)
- Embedding dimensions: 43/64 shifted (67%)
- Heterogeneity/range: 0/2 shifted (0%)

**Script:** `src/walk/36_analyze_2024_drift.py`
**Documentation:** `docs/phase4_summary_and_next_steps.md`

---

### 7. Drift Decomposition Experiment ✓

**Hypothesis:** Is the 18.9% drift caused by sampling bias (heterogeneous thresholds) or real temporal change?

**Experiment Design:**
- Collected uniform 30% dataset (2020-2023, 588 samples)
- Trained model with same threshold as 2024 test
- Compared performance to heterogeneous model

**Results:**

| Model | CV (2020-2023) | Test (2024) | Drift | Test Diff |
|-------|----------------|-------------|-------|-----------|
| Heterogeneous (30-50%) | 0.982 | 0.796 | 18.9% | - |
| Uniform 30% | 1.000 | 0.809 | 19.1% | +0.013 |

**Conclusion:**
- Sampling bias: ~1.3% contribution (minimal)
- Real temporal drift: ~18.6% contribution (dominant)
- **The drift is REAL distributional change, not a sampling artifact**

**Scripts:**
- `src/walk/38_collect_uniform_30pct_2020_2023.py` - Data collection
- `src/walk/39_extract_features_uniform_30pct.py` - Feature extraction
- `src/walk/40_phase4_uniform_30pct_validation.py` - Validation

**Documentation:**
- `docs/drift_decomposition_experiment_results.md`
- `docs/temporal_validation_journey_and_learnings.md`

---

## Key Findings

### Finding 1: Temporal Drift is the Primary Challenge

**Problem:**
- CV ROC-AUC (2020-2023): 0.982 ← Spatial CV looked great
- Test ROC-AUC (2024): 0.796 ← Reality is 18.9% worse
- **This is not a sampling artifact - it's real distributional change**

**Impact:**
- Models trained on 2020-2023 alone are NOT production-ready
- Cannot deploy without addressing temporal generalization
- Edge case performance (0.583) is secondary to this reliability issue

---

### Finding 2: Spatial CV ≠ Temporal Generalization

**What Spatial CV Validated:**
- Model generalizes across different geographic locations
- 10km spatial separation enforced
- Performance: 0.97-0.98 ROC-AUC (robust)

**What Spatial CV MISSED:**
- Year-to-year distributional changes
- Temporal drift (only caught by explicit temporal validation)
- Future data performance (Phase 4 revealed this)

**Lesson:**
> High spatial CV scores give false confidence. Always validate on genuinely future data.

---

### Finding 3: Sampling Bias Was Not the Culprit

**Original Hypothesis:**
- Heterogeneous Hansen thresholds (30%, 40%, 50%) in training
- Uniform 30% threshold in 2024 test
- This mismatch causes drift

**Experiment Result:**
- Uniform 30% model still experiences 19.1% drift
- Only 1.3% performance difference vs heterogeneous
- **Sampling bias contributes minimally (1.3%), real temporal change dominates (18.6%)**

**Implication:**
- Cannot fix drift by improving sampling strategy
- Need temporal adaptation, not data collection changes

---

### Finding 4: Feature-Level Shifts Reveal Drift Mechanism

**What Changed in 2024:**
- All annual change features shifted (delta_1yr, delta_2yr, acceleration)
- 67% of AlphaEarth embedding dimensions shifted
- Landscape statistics (heterogeneity, range) remained stable

**Possible Causes:**
- Different deforestation patterns in 2024
- Environmental/seasonal changes
- Hansen GFC label quality shifts
- Real-world distribution evolution

**Insight:**
> Temporal change features (deltas, accelerations) are most sensitive to drift. Static landscape features are more robust.

---

### Finding 5: Edge Case Scaling Showed Partial Success

**Baseline (114 samples):**
- risk_ranking: 0.850 ROC-AUC
- comprehensive: 0.758 ROC-AUC
- edge_cases: 0.583 ROC-AUC

**After Scaling (685 samples):**
- risk_ranking: Maintained ~0.85
- comprehensive: Maintained ~0.75
- edge_cases: Improved to ~0.60-0.65 (partial)

**Conclusion:**
- Scaling helped but didn't fully solve edge cases
- Still below target (0.70)
- May need specialized models or different features

---

## What's Left for WALK

### Critical Path (Must-Have for Completion)

**1. Temporal Adaptation** ⏳ REQUIRED
- Retrain with 2024 data included
- Measure performance recovery
- Validate that including recent data addresses drift

**2. Model Diversity Experiment** ⏳ PLANNED
- Test XGBoost vs Random Forest
- Evaluate which handles temporal drift better
- Test ensemble if beneficial

**3. Final Model Selection** ⏳ NEEDED
- Choose best performing approach
- Evaluate on all 4 use cases
- Document final performance

**4. Deployment Readiness** ⏳ NEEDED
- Final validation on held-out samples
- Document model capabilities and limitations
- Prepare demo/presentation

---

### Optional Enhancements (Nice-to-Have)

**5. Spatial Features** ✗ SKIPPED
- Neighborhood statistics
- Edge proximity
- Local variance
- **Status:** Not critical for hackathon, skip for now

**6. Q4 Precursor Testing** ✗ SKIPPED
- Original WALK goal (amplify weak Q4 signal)
- **Status:** Temporal drift is more critical, deprioritize

**7. Fire History Features** ✗ SKIPPED
- MODIS fire integration
- Fire-prone area specialization
- **Status:** Edge case improvement, lower priority

---

## Hackathon-Scoped Next Steps

### Constraint: Only 2024 Data Available

**What we CAN'T do:**
- Drift monitoring pipeline (requires future data)
- Scheduled retraining (requires operational deployment)
- Model lifecycle management (production infrastructure)

**What we CAN demo:**
- Temporal drift discovery and quantification
- Drift decomposition (sampling vs temporal)
- Temporal adaptation with 2024 data
- Model diversity comparison
- Performance recovery

---

### Recommended 3-Day Plan

#### Day 1: Temporal Adaptation Experiments

**Morning (3-4 hours):**
1. Combine 2020-2024 data (all 847 samples)
2. Retrain Random Forest with GridSearchCV
3. Evaluate on held-out 2024 subset
4. Measure: 0.796 → ?

**Afternoon (3-4 hours):**
1. Train XGBoost on 2020-2023 (baseline)
2. Train XGBoost on 2020-2024 (adapted)
3. Compare RF vs XGBoost temporal adaptation
4. Identify which model handles drift better

**Deliverable:**
- Performance comparison table
- Best single-model approach identified

---

#### Day 2: Ensemble & Advanced Adaptation

**Morning (3-4 hours):**
1. **Option A: Temporal Ensemble**
   - Model 1: RF trained on 2020-2023 (historical knowledge)
   - Model 2: RF trained on 2024 subset (recent adaptation)
   - Ensemble: Weighted combination
   - Test if combining historical + recent helps

2. **Option B: Model Ensemble**
   - RF trained on 2020-2024
   - XGB trained on 2020-2024
   - Average predictions
   - Test if model diversity helps

**Afternoon (3-4 hours):**
1. Evaluate best approach on all 4 use cases:
   - risk_ranking
   - rapid_response
   - comprehensive
   - edge_cases
2. Compare to Phase 4 baseline (0.796)
3. Document improvement

**Deliverable:**
- Final model selected
- Performance on all use cases
- Comparison to baseline

---

#### Day 3: Demo Preparation & Documentation

**Morning (3-4 hours):**
1. Create demo notebook with narrative:
   - Act 1: Problem discovery (temporal drift)
   - Act 2: Investigation (drift decomposition)
   - Act 3: Solution (temporal adaptation)
   - Act 4: Results (performance recovery)

2. Generate visualizations:
   - Temporal validation progression (Phases 1-4)
   - Feature distribution shifts
   - Before/after performance comparison
   - Use-case-specific results

**Afternoon (3-4 hours):**
1. Polish documentation:
   - WALK phase summary
   - Key learnings
   - Methodology
   - Results and implications

2. Prepare presentation:
   - Problem statement
   - Approach
   - Findings
   - Demo

**Deliverable:**
- Demo-ready notebook
- Presentation slides
- Complete documentation

---

## Experiment Options

### Option 1: Simple Retraining (Fastest)

**Implementation:**
```python
# Load all data (2020-2024)
train_2020_2024 = load_training_data() + load_2024_data()

# Split 2024 into train/test
train_samples, test_samples = split_2024(train_2020_2024)

# Train on 2020-2024
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluate on held-out 2024
score = evaluate(model, X_test, y_test)
print(f"ROC-AUC: {score} (baseline: 0.796)")
```

**Time:** ~30 minutes to implement, ~1 hour to train and evaluate

**Expected Outcome:**
- ROC-AUC improvement: 0.796 → 0.82-0.85
- Demonstrates that including recent data helps

**Demo Value:** ★★★☆☆ (Good, but straightforward)

---

### Option 2: XGBoost Comparison (Most Interesting)

**Implementation:**
```python
# Train both models on same data
rf_model = train_random_forest(2020_2024_data)
xgb_model = train_xgboost(2020_2024_data)

# Compare on temporal adaptation
rf_baseline = evaluate(rf_2020_2023, test_2024)  # 0.796
rf_adapted = evaluate(rf_2020_2024, test_2024)   # ?

xgb_baseline = evaluate(xgb_2020_2023, test_2024)  # ?
xgb_adapted = evaluate(xgb_2020_2024, test_2024)   # ?

# Which model adapts better?
if xgb_adapted > rf_adapted:
    print("XGBoost handles temporal drift better")
else:
    print("Random Forest handles temporal drift better")
```

**Time:** ~1 hour to implement, ~2 hours to train and evaluate

**Expected Outcome:**
- Answer: Which model architecture handles drift better?
- Potential: XGBoost may outperform RF on temporal adaptation
- Insight: Different models have different robustness to drift

**Demo Value:** ★★★★☆ (Very interesting - model comparison)

---

### Option 3: Temporal Ensemble (Most Demo-Worthy)

**Implementation:**
```python
# Model 1: Historical knowledge (2020-2023)
model_historical = train_rf(data_2020_2023)

# Model 2: Recent adaptation (2024 subset for fine-tuning)
model_recent = train_rf(data_2024_train)

# Ensemble with learned weights
def predict_ensemble(sample):
    pred_historical = model_historical.predict_proba(sample)
    pred_recent = model_recent.predict_proba(sample)

    # Option A: Simple average
    return (pred_historical + pred_recent) / 2

    # Option B: Learned weights (optimize on validation set)
    return w1 * pred_historical + w2 * pred_recent
```

**Time:** ~2 hours to implement, ~2 hours to train and tune

**Expected Outcome:**
- Combines historical knowledge with recent adaptation
- May outperform single model trained on all data
- Demonstrates sophisticated approach to drift

**Demo Value:** ★★★★★ (Excellent - tells compelling story)

**Narrative:**
> "Historical model knows general deforestation patterns. Recent model adapts to 2024 distribution. Ensemble combines both strengths for robust prediction."

---

### Option 4: Feature Augmentation (Simplest)

**Implementation:**
```python
# Add temporal features
X_train['year'] = sample_years
X_train['months_since_2020'] = (sample_years - 2020) * 12

# Model learns temporal trends
model.fit(X_train, y_train)

# At test time, model sees year=2024
X_test['year'] = 2024
X_test['months_since_2020'] = 4 * 12
```

**Time:** ~30 minutes to implement and test

**Expected Outcome:**
- Model explicitly learns temporal patterns
- May help but unlikely to fully solve drift
- Good baseline before more complex approaches

**Demo Value:** ★★☆☆☆ (Basic, but good starting point)

---

### Option 5: Importance Weighting

**Implementation:**
```python
# Compute distribution similarity between train and test
def compute_sample_weights(train_samples, test_distribution):
    """
    Upweight training samples that resemble test distribution.
    """
    weights = []
    for sample in train_samples:
        # Compute feature distance to 2024 distribution
        distance = mahalanobis_distance(sample.features, test_distribution)
        # Lower distance = higher weight
        weight = np.exp(-distance)
        weights.append(weight)
    return np.array(weights)

# Train with sample weights
weights = compute_sample_weights(train_2020_2023, test_2024_distribution)
model.fit(X_train, y_train, sample_weight=weights)
```

**Time:** ~2 hours to implement and test

**Expected Outcome:**
- Emphasizes training samples that look like 2024
- Clever approach to drift without retraining
- May not be as effective as including actual 2024 data

**Demo Value:** ★★★★☆ (Interesting technique, good for discussion)

---

## Decision Framework

### Question 1: What's the primary goal?

**Option A: Maximum Performance**
- Pursue all experiments (Options 1-5)
- Ensemble best approaches
- Target: 0.85+ ROC-AUC on 2024
- Time: 2-3 days

**Option B: Quick Result for Demo**
- Start with Option 1 (simple retraining)
- If time permits, add Option 2 (XGBoost)
- Target: 0.82+ ROC-AUC on 2024
- Time: 1 day

**Option C: Most Interesting Story**
- Focus on Option 3 (temporal ensemble)
- Add Option 2 for comparison
- Target: Compelling narrative + good performance
- Time: 1.5-2 days

---

### Question 2: What's the time constraint?

**If 1 Day Available:**
- Option 1 (simple retraining) + Option 2 (XGBoost comparison)
- Quick path to measurable improvement
- Can demo: "We discovered drift, we fixed it by including recent data"

**If 2 Days Available:**
- Day 1: Options 1 + 2
- Day 2: Option 3 (ensemble) if worth it, or polish demo
- Better results, more thorough comparison

**If 3 Days Available:**
- Day 1: Options 1 + 2 + 4
- Day 2: Options 3 + 5, select best
- Day 3: Demo preparation
- Comprehensive exploration, best possible result

---

### Question 3: What's the demo narrative priority?

**Priority A: Problem Discovery**
- Emphasize temporal validation framework
- Show drift decomposition experiment
- Demonstrate rigorous methodology
- **Best options:** Any option works, focus on documentation

**Priority B: Solution Implementation**
- Show multiple approaches to temporal adaptation
- Compare techniques
- Demonstrate sophisticated ML engineering
- **Best options:** Options 2, 3, 5 (more interesting)

**Priority C: Performance Recovery**
- Focus on before/after comparison
- Show measurable improvement
- Demonstrate practical impact
- **Best options:** Option 1 or 2 (clear results)

---

## Recommended Approach

### My Recommendation: Hybrid Strategy

**Phase A (4-5 hours): Quick Wins**
1. Option 1: Simple retraining with 2020-2024 data
2. Option 4: Add year as feature
3. Get baseline improvement: 0.796 → 0.82-0.85

**Phase B (4-5 hours): Model Comparison**
1. Option 2: Train XGBoost on 2020-2024
2. Compare RF vs XGBoost
3. Identify which architecture handles drift better

**Decision Point:**
- If improvement already good (0.85+): Polish and demo
- If improvement modest (0.80-0.85): Proceed to Phase C

**Phase C (4-5 hours): Advanced Techniques**
1. Option 3: Temporal ensemble
2. Test if ensemble outperforms single models
3. Select final approach

**Phase D (3-4 hours): Demo Preparation**
1. Create demo notebook
2. Generate visualizations
3. Document findings

**Total Time:** 2 days (15-19 hours)

---

### Why This Approach?

**1. Progressive Complexity:**
- Start simple (retraining)
- Add sophistication (XGBoost, ensemble)
- Only pursue advanced techniques if needed

**2. Multiple Stopping Points:**
- Can stop after Phase A if time-constrained
- Can stop after Phase B if results good
- Can pursue Phase C for excellence

**3. Interesting Story:**
- Show multiple approaches
- Demonstrate systematic exploration
- Compelling narrative regardless of result

**4. Pragmatic:**
- Balances performance, time, demo value
- Reduces risk (quick wins early)
- Maximizes learning

---

## Next Decisions Needed

1. **Time available:** 1, 2, or 3 days?
2. **Demo priority:** Problem discovery, solution implementation, or performance recovery?
3. **Starting point:** Should we begin with Option 1 (simple retraining) or jump to Option 2/3?
4. **Scope:** Quick path to result, or comprehensive exploration?

---

## File Inventory

### Data Files Created

**Training Data:**
- `walk_dataset_scaled_phase1_20251020_165345_all_hard_samples_multiscale.pkl` (685 samples, 2020-2023)
- `walk_dataset_uniform_30pct_2020_2023_with_features_20251022_210556.pkl` (588 samples, uniform 30%)

**Test Data:**
- `walk_dataset_2024_with_features_20251021_110417.pkl` (162 samples)

**Hard Validation Sets:**
- `hard_val_risk_ranking.pkl` (46 samples)
- `hard_val_rapid_response.pkl` (27 samples)
- `hard_val_comprehensive.pkl` (69 samples)
- `hard_val_edge_cases.pkl` (23 samples)

### Scripts Created

**Progressive Temporal Validation:**
- `31_temporal_validation.py` - Phases 1-3
- `32_collect_2024_samples.py` - 2024 data collection
- `33_extract_features_2024.py` - 2024 feature extraction
- `34_phase4_temporal_validation.py` - Phase 4 validation

**Drift Investigation:**
- `36_analyze_2024_drift.py` - Feature distribution analysis

**Drift Decomposition:**
- `38_collect_uniform_30pct_2020_2023.py` - Uniform 30% data collection
- `39_extract_features_uniform_30pct.py` - Feature extraction
- `40_phase4_uniform_30pct_validation.py` - Temporal validation

**Supporting:**
- `08_multiscale_embeddings.py` - Multiscale feature extraction
- `09_phase1_extract_features.py` - Training data features
- `30_threshold_optimization.py` - Use-case thresholds

### Documentation Created

**Strategic:**
- `walk_phase_overview.md` - Original plan
- `walk_phase_strategic_decisions.md` - Key decisions
- `scaling_and_specialization_strategy.md` - Model diversity plan

**Results:**
- `phase4_summary_and_next_steps.md` - Phase 4 investigation
- `drift_decomposition_experiment_results.md` - Uniform 30% findings
- `temporal_validation_journey_and_learnings.md` - Complete journey
- `walk_phase_current_status_and_next_steps.md` - This document

**Technical:**
- `honest_performance_evaluation.md` - Validation results
- `temporal_validation_plan.md` - Validation strategy

---

## Summary

**Current State:**
- Rigorous validation framework complete
- Temporal drift discovered, quantified, and explained
- Random Forest baseline established
- Ready for final model development

**What's Left:**
- Temporal adaptation with 2024 data
- Model diversity (XGBoost, ensemble)
- Final model selection
- Demo preparation

**Estimated Time to Complete WALK:**
- Minimum (quick path): 1 day
- Recommended (thorough): 2 days
- Comprehensive (all options): 3 days

**Next Action:**
- Decide on time/scope/priorities
- Begin with selected experiment option
- Iterate based on results

---

**Status:** WALK phase 75% complete, ready for final push
**Blocker:** None, clear path forward
**Decision:** Needed on scope and priorities for final experiments
