# Held-Out Temporal Split Experiment

**Status:** PLANNED (Not yet executed)
**Purpose:** Test whether the model can predict **future** deforestation using patterns learned from the **past**
**Priority:** HIGH - Critical for validating operational deployment readiness

---

## Table of Contents

1. [The Problem: Temporal Leakage in Current Validation](#the-problem-temporal-leakage-in-current-validation)
2. [What is a Held-Out Temporal Split?](#what-is-a-held-out-temporal-split)
3. [Why This Matters](#why-this-matters)
4. [Current Validation vs Temporal Split](#current-validation-vs-temporal-split)
5. [Experimental Design](#experimental-design)
6. [Expected Outcomes](#expected-outcomes)
7. [Risks and Mitigation](#risks-and-mitigation)
8. [Implementation Plan](#implementation-plan)

---

## The Problem: Temporal Leakage in Current Validation

### What We've Done So Far

**Current validation approach:**
- Training set: 87 samples from **2020, 2021, 2022** (mixed years)
- Validation sets: 165 samples from **2020, 2021, 2022** (mixed years)
- Test set: 17 samples from **2020, 2021, 2022** (mixed years)

**What we've validated:**
- ✓ Spatial generalization (different locations)
- ✓ No spatial leakage (10km exclusion buffer)
- ✓ Temporal contamination controlled (dual-year delta approach)
- ✓ Different forest types and clearing scenarios

**What we HAVEN'T validated:**
- ✗ Temporal generalization (predicting future from past)
- ✗ Model robustness to changing deforestation patterns over time
- ✗ True operational scenario (train on history, predict future)

### The Hidden Assumption

By mixing years in both training and validation, we're implicitly assuming:
- Deforestation patterns are **stationary** (don't change over time)
- A sample from 2022 is interchangeable with a sample from 2020
- The model will work equally well on 2023, 2024, 2025...

**But this assumption might not hold:**
- Deforestation drivers change (policies, economics, climate)
- New deforestation frontiers emerge
- Enforcement patterns shift
- Agricultural commodity prices fluctuate

### Example of Temporal Leakage Risk

**Scenario:** Model learns that high delta in specific Amazon sub-region X predicts clearing
- **Why?** 2020-2022 had a deforestation surge in region X due to specific policy change
- **Training:** Model sees samples from 2020, 2021, 2022 in region X
- **Validation:** Model tested on other samples from 2020, 2021, 2022 in region X
- **Result:** High accuracy (but might be overfitting to this specific 2020-2022 pattern)

**2023 reality:** Policy changes, region X is now heavily monitored, deforestation shifts to region Y
- **Deployment:** Model fails because it learned region X patterns, not general precursor signals

**Held-out temporal split would catch this:**
- Train on 2020-2021 only
- Test on 2022 only
- If model still works, it learned transferable patterns, not year-specific artifacts

---

## What is a Held-Out Temporal Split?

### Definition

A **held-out temporal split** separates training and testing data by **time** rather than random sampling:

**Traditional split (what we've done):**
```
Data: [2020 samples, 2021 samples, 2022 samples] (all mixed together)
       ↓ Random 80/20 split
Train: [2020, 2021, 2022] (80% of each year)
Test:  [2020, 2021, 2022] (20% of each year)
```

**Temporal split (what we should do):**
```
Data: [2020 samples, 2021 samples, 2022 samples, 2023 samples]
       ↓ Chronological split
Train: [2020, 2021] (all samples from early years)
Test:  [2022, 2023] (all samples from later years)
```

### Why Chronological Order Matters

In operational deployment:
1. **Today:** You have data up to 2024
2. **Train model:** Using 2020-2024 historical clearings
3. **Deploy:** Model predicts 2025 clearings

**The critical question:** Can patterns learned from 2020-2024 predict 2025?

**Current validation doesn't answer this** because test data comes from the same years as training data.

---

## Why This Matters

### 1. True Operational Scenario

**Real-world deployment:**
- Train on **past** data (2020-2023)
- Predict **future** events (2024-2025)
- You can never train on the future you're trying to predict

**Current validation:**
- Train on **past + present + future** mixed together
- Test on **past + present + future** mixed together
- This is not how the model will be used in production

### 2. Non-Stationarity Detection

**Deforestation is a dynamic process:**

**Drivers that change over time:**
- **Policy:** Brazil's deforestation moratorium (2004), enforcement crackdowns
- **Economics:** Soy prices, beef prices, palm oil demand
- **Climate:** El Niño cycles, drought patterns
- **Technology:** Satellite monitoring improvements, real-time alerts
- **Frontiers:** New road construction, shifting agricultural expansion

**Example from real data:**

| Period | Dominant Pattern |
|--------|------------------|
| **2000-2004** | Large-scale cattle ranching expansion |
| **2005-2012** | Reduction due to policy (soy moratorium) |
| **2013-2019** | Shift to smaller clearings, fragmentation |
| **2019-2022** | Surge in large clearings (policy rollback) |
| **2023+** | Renewed enforcement, pattern shift again |

If your model trains on 2019-2022 (surge period), will it work in 2023+ (enforcement period)?

**Held-out temporal split answers this.**

### 3. Avoiding Overfitting to Temporal Artifacts

**Year-specific patterns that don't generalize:**
- 2020: COVID-19 disruption to enforcement
- 2021: El Niño drought → fire-prone clearings
- 2022: Specific policy change in Brazil

**If model learns these:**
- High accuracy on 2020-2022 validation (same patterns)
- Low accuracy on 2023+ deployment (different patterns)

**Temporal split catches this:**
- Train on 2020-2021
- Test on 2022
- If performance drops, model overfitting to early-year patterns

### 4. Confidence in Long-Term Deployment

**Stakeholder question:** "You have 0.75-0.85 ROC-AUC on validation. Will this hold up in 2025?"

**Current answer:** "We validated on different locations from 2020-2022, so probably?"

**Better answer:** "We tested on held-out 2022-2023 data, trained only on 2020-2021. Performance was 0.72 ROC-AUC, slight drop but still strong. This suggests the model will generalize to 2024-2025."

---

## Current Validation vs Temporal Split

### Current Validation (What We've Done)

**Training Data:**
```
Year 2020: 29 clearings (33%)
Year 2021: 29 clearings (33%)
Year 2022: 29 clearings (33%)
Total: 87 samples
```

**Test Set:**
```
Year 2020: 6 clearings (35%)
Year 2021: 6 clearings (35%)
Year 2022: 5 clearings (30%)
Total: 17 samples
```

**Validation Sets (risk_ranking, comprehensive, etc.):**
```
Years: 2020, 2021, 2022 mixed
Total: 165 samples
```

**What this tests:**
- ✓ Spatial generalization (different lat/lon)
- ✓ Different forest types
- ✓ Different clearing scenarios
- ✗ **NOT** temporal generalization (years are mixed)

### Temporal Split (What We Should Do)

#### Option A: Conservative Split

**Training:**
```
Year 2020: All available clearings (~100)
Year 2021: All available clearings (~100)
Total: ~200 training samples
```

**Validation (hyperparameter tuning):**
```
Year 2022 Q1-Q2: ~30 samples
```

**Test (final evaluation):**
```
Year 2022 Q3-Q4: ~30 samples
Year 2023: All available clearings (~100)
Total: ~130 test samples
```

**Temporal gap:** Train on 2020-2021 → Test on 2022-2023 (1-3 year gap)

#### Option B: Progressive Validation

**Walk-forward approach:**

**Fold 1:**
- Train: 2020
- Test: 2021
- Gap: 1 year

**Fold 2:**
- Train: 2020-2021
- Test: 2022
- Gap: 1-2 years

**Fold 3:**
- Train: 2020-2022
- Test: 2023
- Gap: 1-3 years

**Aggregate:** Average performance across folds

**What this tests:**
- ✓ Temporal generalization at different time scales
- ✓ Performance degradation over time
- ✓ Robustness to changing patterns

---

## Experimental Design

### Proposed Approach: Progressive Temporal Validation

**Step 1: Data Collection**

Collect clearings for each year separately:
```python
clearings_2020 = get_clearings(year=2020, n=100)
clearings_2021 = get_clearings(year=2021, n=100)
clearings_2022 = get_clearings(year=2022, n=100)
clearings_2023 = get_clearings(year=2023, n=100)  # If available

intact_2020 = get_intact(year=2020, n=100)
intact_2021 = get_intact(year=2021, n=100)
intact_2022 = get_intact(year=2022, n=100)
intact_2023 = get_intact(year=2023, n=100)
```

**Step 2: Feature Extraction**

Extract dual-year delta features for all samples:
```python
# For 2021 clearing:
# - Y-1 = 2020 embeddings (Q1, Q2, Q3, Q4)
# - Y = 2021 embeddings (Q1, Q2, Q3, Q4)
# - Delta = Y - Y-1
# - Features = baseline(Y-1) + delta

# For 2022 clearing:
# - Y-1 = 2021 embeddings
# - Y = 2022 embeddings
# - Delta = Y - Y-1
# etc.
```

**Step 3: Temporal Folds**

**Fold 1: 2020 → 2021**
```
Train:
  - 2020 clearings (100 samples)
  - 2020 intact (100 samples)

Test:
  - 2021 clearings (100 samples)
  - 2021 intact (100 samples)

Temporal gap: 1 year
```

**Fold 2: 2020-2021 → 2022**
```
Train:
  - 2020 clearings (100 samples)
  - 2021 clearings (100 samples)
  - 2020-2021 intact (200 samples)

Test:
  - 2022 clearings (100 samples)
  - 2022 intact (100 samples)

Temporal gap: 1-2 years
```

**Fold 3: 2020-2022 → 2023** (if 2023 data available)
```
Train:
  - 2020-2022 clearings (300 samples)
  - 2020-2022 intact (300 samples)

Test:
  - 2023 clearings (100 samples)
  - 2023 intact (100 samples)

Temporal gap: 1-3 years
```

**Step 4: Evaluation**

For each fold:
- Train logistic regression on training set
- Evaluate on test set
- Compute: ROC-AUC, Precision, Recall, Accuracy
- Compare to current validation performance

**Step 5: Analysis**

**Performance degradation analysis:**
```
Fold 1 (2020→2021): ROC-AUC = ?
Fold 2 (2020-2021→2022): ROC-AUC = ?
Fold 3 (2020-2022→2023): ROC-AUC = ?

Current validation (mixed years): ROC-AUC = 0.75-0.85

Degradation = Current - Temporal
```

**Interpretation:**
- **< 5% drop:** Excellent temporal generalization
- **5-10% drop:** Good temporal generalization, acceptable for deployment
- **10-20% drop:** Moderate temporal generalization, need more data or features
- **> 20% drop:** Poor temporal generalization, model overfitting to temporal artifacts

---

## Expected Outcomes

### Scenario 1: Strong Temporal Generalization (Best Case)

**Results:**
- Fold 1 (2020→2021): ROC-AUC = 0.78
- Fold 2 (2020-2021→2022): ROC-AUC = 0.80
- Fold 3 (2020-2022→2023): ROC-AUC = 0.79
- Current validation: ROC-AUC = 0.75-0.85

**Interpretation:**
- ✓ Model generalizes well across years
- ✓ Patterns learned are robust to temporal changes
- ✓ Ready for operational deployment

**Confidence level:** HIGH

### Scenario 2: Moderate Temporal Generalization (Acceptable)

**Results:**
- Fold 1 (2020→2021): ROC-AUC = 0.68
- Fold 2 (2020-2021→2022): ROC-AUC = 0.72
- Fold 3 (2020-2022→2023): ROC-AUC = 0.70
- Current validation: ROC-AUC = 0.75-0.85

**Interpretation:**
- ~ 5-15% performance drop with temporal split
- ~ Model captures some generalizable patterns but also some year-specific artifacts
- ~ Need more training data or engineered features for robustness

**Confidence level:** MODERATE
**Recommendation:** Collect more data, add temporal stability features

### Scenario 3: Poor Temporal Generalization (Concerning)

**Results:**
- Fold 1 (2020→2021): ROC-AUC = 0.55
- Fold 2 (2020-2021→2022): ROC-AUC = 0.60
- Fold 3 (2020-2022→2023): ROC-AUC = 0.58
- Current validation: ROC-AUC = 0.75-0.85

**Interpretation:**
- ✗ Model overfitting to year-specific patterns
- ✗ Does not generalize well to unseen time periods
- ✗ Current validation metrics are misleading

**Confidence level:** LOW
**Recommendation:** Investigate year-specific features, consider different modeling approach

---

## Risks and Mitigation

### Risk 1: Insufficient Data for Temporal Split

**Problem:** We currently have 87 training samples total. Splitting by year reduces sample size:
- 2020 only: ~30 samples
- 2021 only: ~30 samples
- 2022 only: ~30 samples

**Mitigation:**
- Collect more samples (target 100-200 per year)
- Use progressive validation (train on cumulative years) to increase training size
- Accept lower statistical power as trade-off for temporal validity

### Risk 2: Year-Specific Events Confound Results

**Problem:** One year might have anomalous patterns:
- 2020: COVID-19 disruption
- 2021: El Niño drought
- 2022: Policy change

**Mitigation:**
- Use multiple folds (2020→2021, 2020-2021→2022, etc.)
- Average performance across folds
- Analyze year-specific patterns separately

### Risk 3: Temporal Autocorrelation

**Problem:** Deforestation has spatial-temporal autocorrelation:
- Clearing in 2020 at (lat, lon) → likely continued clearing in 2021 nearby
- Model might "cheat" by learning spatial proximity to prior clearings

**Mitigation:**
- Maintain 10km spatial exclusion buffer between train/test
- Check if test clearings are near training clearings
- Add distance-to-prior-clearing as control variable

### Risk 4: Changing Feature Distributions

**Problem:** Embedding distributions might drift over time:
- AlphaEarth model might be updated
- Sensor calibration changes
- Atmospheric correction improvements

**Mitigation:**
- Check feature distribution shifts across years
- Normalize features within each year
- Test with and without normalization

---

## Implementation Plan

### Phase 1: Data Collection (Week 1)

**Tasks:**
1. Collect 100-200 clearings per year (2020, 2021, 2022, 2023)
2. Generate 100-200 intact samples per year
3. Extract dual-year delta features for all samples
4. Verify spatial exclusion (no overlap between years)

**Deliverable:** `data/processed/temporal_split/` with yearly datasets

### Phase 2: Temporal Validation (Week 2)

**Tasks:**
1. Implement progressive temporal validation
2. Train models for each fold (2020→2021, 2020-2021→2022, etc.)
3. Evaluate performance on held-out years
4. Compare to current validation results

**Deliverable:** `results/temporal_generalization_experiment.json`

### Phase 3: Analysis (Week 3)

**Tasks:**
1. Analyze performance degradation
2. Investigate year-specific patterns
3. Check feature distribution shifts
4. Compute confidence intervals

**Deliverable:** `docs/temporal_generalization_results.md`

### Phase 4: Recommendations (Week 4)

**Tasks:**
1. Determine deployment readiness
2. Identify model improvements needed
3. Plan for ongoing temporal validation
4. Document findings for stakeholders

**Deliverable:** Updated `docs/honest_performance_evaluation.md`

---

## Code Skeleton

```python
# src/walk/06_temporal_generalization_experiment.py

def collect_yearly_samples(year, n_clearing=100, n_intact=100):
    """
    Collect clearing and intact samples for a specific year.

    Returns:
        Dict with clearings and intact samples for the year
    """
    pass

def temporal_fold_validation(train_years, test_year):
    """
    Train on train_years, test on test_year.

    Args:
        train_years: List of years to train on (e.g., [2020, 2021])
        test_year: Single year to test on (e.g., 2022)

    Returns:
        Dict with performance metrics
    """
    # Collect data
    train_data = [collect_yearly_samples(y) for y in train_years]
    test_data = collect_yearly_samples(test_year)

    # Extract features
    X_train, y_train = extract_features(train_data)
    X_test, y_test = extract_features(test_data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    return metrics

def progressive_temporal_validation():
    """
    Run multiple temporal folds with increasing training window.
    """
    folds = [
        {'train': [2020], 'test': 2021},
        {'train': [2020, 2021], 'test': 2022},
        {'train': [2020, 2021, 2022], 'test': 2023},
    ]

    results = []
    for fold in folds:
        metrics = temporal_fold_validation(fold['train'], fold['test'])
        results.append(metrics)

    return results
```

---

## Summary

**What:** Test whether model trained on past years can predict future years

**Why:**
- Current validation mixes years (not true operational scenario)
- Need to validate temporal generalization
- Avoid overfitting to year-specific patterns

**How:**
- Train on 2020-2021, test on 2022-2023
- Progressive validation (multiple folds)
- Compare to current validation performance

**When:** Next priority after synthetic contamination validation

**Expected outcome:**
- Best case: <5% performance drop (strong generalization)
- Acceptable: 5-15% drop (moderate generalization)
- Concerning: >20% drop (overfitting to temporal artifacts)

**Decision point:**
- If performance holds: Proceed to deployment
- If moderate drop: Collect more data, add features
- If large drop: Revisit modeling approach

---

## Next Steps

1. **Immediate:** Collect yearly datasets (100+ samples per year)
2. **Week 1:** Implement temporal fold validation
3. **Week 2:** Run experiment and analyze results
4. **Week 3:** Document findings and update honest performance evaluation

**Status:** Ready to implement pending user approval
