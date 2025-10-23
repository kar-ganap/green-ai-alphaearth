# Scaling and Specialization Strategy

**Date:** 2025-10-17
**Purpose:** Strategic plan for addressing performance variability across validation sets
**Status:** Planning - Ready for execution

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Current State](#current-state)
3. [Strategic Options](#strategic-options)
4. [Recommended Phased Approach](#recommended-phased-approach)
5. [Detailed Implementation Plans](#detailed-implementation-plans)
6. [Decision Criteria](#decision-criteria)
7. [Resource Requirements](#resource-requirements)
8. [Success Metrics](#success-metrics)

---

## The Problem

### Performance Variability Across Validation Sets

**Current Results:**

| Validation Set | ROC-AUC | Precision | Recall | Scenario |
|----------------|---------|-----------|--------|----------|
| **Temporal split** | 0.971 | 88-100% | 71-93% | Future years (standard patterns) |
| **risk_ranking** | 0.850 | 67% | 67% | Standard clearings, resource prioritization |
| **comprehensive** | 0.758 | 100% | 45% | Diverse forest types and patterns |
| **edge_cases** | 0.583 | 100% | 25% | Small clearings, fire-prone, forest edges |

**The Gap:** 0.583 vs 0.971 is a **67% performance difference**

### Key Questions

1. **Is this a data scarcity issue?** (Can scaling training data close the gap?)
2. **Are these fundamentally different problems?** (Do edge cases need specialized models?)
3. **Should we use different metrics per use case?** (Precision vs ROC-AUC vs Recall)
4. **Is Mixture of Experts (MoE) warranted?** (Learned routing vs manual specialization)

---

## Current State

### What We've Validated ✓

**Spatial Generalization:**
- Different geographic locations (10km exclusion buffer)
- Performance: 0.58-0.85 ROC-AUC depending on difficulty
- Status: VALIDATED

**Temporal Generalization:**
- Future years (train on past, test on future)
- Performance: 0.97 ROC-AUC (1-3 year gaps)
- Status: VALIDATED

**Temporal Contamination Control:**
- Early vs late year quarters, Q2 vs Q4 clearings
- Performance: Identical across scenarios (0% difference)
- Status: VALIDATED

### What We Haven't Tested ✗

- **Scaled training data** (200-500 samples vs current 87)
- **Intentional edge case representation** in training
- **Specialized models** for different scenarios
- **Threshold optimization** per use case
- **Feature engineering** specifically for edge cases
- **Diagnostic analysis** of edge case failures

### Current Training Data Composition

**87 total samples:**
- Mostly standard clearings (> 1 hectare)
- Limited small clearing representation
- Limited fire-prone area representation
- Limited forest edge representation

**Hypothesis:** Edge case underperformance may be due to **training data composition**, not fundamental model limitations.

---

## Strategic Options

### Option A: Scale Up Training Data (Single Model)

**Approach:** Collect 200-500 samples with intentional diversity, train one model

**Pros:**
- ✓ Simplest deployment (one model for all scenarios)
- ✓ Lowest maintenance burden
- ✓ Temporal validation suggests model CAN generalize (0.97 ROC-AUC)
- ✓ May improve edge cases through better representation
- ✓ Quick to test (2-3 days)

**Cons:**
- May not help if edge cases are fundamentally different
- Could dilute performance on standard cases
- Unclear if data alone closes 0.583 → 0.850 gap

**Data Requirements:** 200-300 samples
**Development Time:** 2-3 days
**Deployment Complexity:** Low (1 model)

### Option B: Specialized Models (Manual Routing)

**Approach:** Build 2-3 models for different scenarios, route with simple rules

**Example:**
```python
Model 1: Standard Clearings
  - Training: Large clearings (> 1 ha), continuous forest
  - Target: 0.85-0.90 ROC-AUC

Model 2: Edge Cases
  - Training: Small clearings, fire-prone, fragmented
  - Features: Add fire history, fragmentation metrics
  - Target: 0.70-0.75 ROC-AUC

Routing Rules:
  if clearing_size < 1 ha or fire_activity_6mo > 5:
      use Model 2
  else:
      use Model 1
```

**Pros:**
- ✓ Each model optimized for its domain
- ✓ Can use different features per model
- ✓ Higher performance ceiling for edge cases
- ✓ Interpretable routing decisions

**Cons:**
- 2-3 models to maintain and deploy
- Need to design routing rules
- More complex infrastructure
- Risk of overfitting to validation sets

**Data Requirements:** 300-400 samples total
**Development Time:** 1-2 weeks
**Deployment Complexity:** Medium (2-3 models, manual routing)

### Option C: Mixture of Experts (MoE)

**Approach:** Multiple specialized models + learned gating network

**Architecture:**
```
Input → Gating Network → Expert weights
     ↓
   Expert 1 (Standard clearings)
   Expert 2 (Edge cases)
   Expert 3 (Generalist fallback)
     ↓
Weighted combination → Final prediction
```

**Pros:**
- ✓ Adaptive routing (learns which expert to use)
- ✓ Specialization benefits
- ✓ Graceful degradation (blend experts if uncertain)
- ✓ Best for truly heterogeneous scenarios

**Cons:**
- ✗ Most complex to implement and deploy
- ✗ Requires 2-3x more data (500-800 samples)
- ✗ Gating network needs ground truth labels
- ✗ Harder to debug and interpret
- ✗ 4 models to maintain (3 experts + gating)

**Data Requirements:** 500-800 samples
**Development Time:** 2-3 weeks
**Deployment Complexity:** High (4 models, learned routing)

### Option D: Hybrid (Single Model + Threshold Tuning)

**Approach:** One model trained on diverse data, different thresholds per use case

**Implementation:**
```python
model = train_on_diverse_data(300_samples)

thresholds = {
    'rapid_response': 0.85,   # High precision for field teams
    'risk_ranking': 0.50,     # Balanced for resource allocation
    'edge_cases': 0.30,       # Lower bar to catch difficult cases
}

def predict(sample, use_case):
    score = model.predict_proba(sample)
    return score > thresholds[use_case]
```

**Pros:**
- ✓ One model (simple deployment)
- ✓ Flexible per-use-case optimization
- ✓ Lower data requirements (200-300 samples)
- ✓ Easy to interpret and debug

**Cons:**
- May not achieve edge case performance of specialized model
- Threshold tuning is simpler than specialized features

**Data Requirements:** 200-300 samples
**Development Time:** 1 week
**Deployment Complexity:** Low (1 model, multiple thresholds)

---

## Recommended Phased Approach

**DON'T jump to MoE.** Follow this sequence:

### Phase 0: Diagnostic Analysis (1-2 days)

**BEFORE scaling or specializing, understand WHY edge cases fail.**

**Objectives:**
1. Analyze feature distributions across validation sets
2. Error analysis: Which edge cases are missed and why?
3. Sample characteristics: Size, forest type, fragmentation, fire history
4. Feature importance: Do edge cases use different features?

**Deliverables:**
- `results/edge_case_diagnostic_analysis.json`
- Understanding of whether gap is data scarcity vs fundamental difference

**Decision Point:**
- If edge cases have **similar feature patterns** → Scale up training (Phase 1)
- If edge cases have **different patterns** → Consider specialization (Phase 2)

---

### Phase 1: Scaled Training with Diversity (2-3 days)

**Hypothesis:** Edge case underperformance is due to insufficient training diversity.

**Approach:**

**Data Collection (300 samples total):**
```
Composition:
  - 60% standard clearings (> 1 ha, continuous forest) = 180 samples
  - 20% small clearings (< 1 ha) = 60 samples
  - 10% fire-prone areas = 30 samples
  - 10% forest edges (fragmentation > 0.7) = 30 samples

Geographic diversity:
  - Amazon: 150 samples
  - Congo Basin: 75 samples (test geographic generalization)
  - SE Asia: 75 samples

Temporal diversity:
  - 2020-2023 distributed evenly
```

**Training:**
```python
# Train single model on all 300 samples
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on all validation sets
results = {
    'risk_ranking': evaluate(model, risk_ranking_set),
    'comprehensive': evaluate(model, comprehensive_set),
    'edge_cases': evaluate(model, edge_cases_set),
}
```

**Success Criteria:**
- Edge cases ROC-AUC improves: 0.583 → **0.70+** (20% improvement)
- Standard cases maintained: risk_ranking stays **0.80+**
- Precision maintained across sets: **85%+**

**If SUCCESS (edge cases ≥ 0.70):**
- ✓ Problem solved with scaling
- Proceed to Phase 1b (threshold optimization)
- No need for specialization

**If PARTIAL (edge cases 0.65-0.70):**
- ~ Moderate improvement
- Proceed to Phase 1b, then reassess
- Consider Phase 2 if threshold tuning insufficient

**If FAILURE (edge cases < 0.65):**
- ✗ Scaling alone insufficient
- Proceed to Phase 2 (specialization)

---

### Phase 1b: Threshold Optimization (1 day)

**Objective:** Optimize decision thresholds for each use case.

**Approach:**
```python
# For each validation set, find optimal threshold
thresholds = {}

for use_case in ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']:
    # Define optimization objective per use case
    if use_case == 'rapid_response':
        # Maximize recall subject to 95% precision
        threshold = optimize_threshold(
            model, validation_set,
            objective='recall',
            constraint='precision >= 0.95'
        )
    elif use_case == 'risk_ranking':
        # Maximize F1 score
        threshold = optimize_threshold(
            model, validation_set,
            objective='f1_score'
        )
    elif use_case == 'edge_cases':
        # Maximize recall subject to 85% precision
        threshold = optimize_threshold(
            model, validation_set,
            objective='recall',
            constraint='precision >= 0.85'
        )

    thresholds[use_case] = threshold
```

**Deliverables:**
- Optimized thresholds per use case
- Performance curves showing precision-recall trade-offs
- Deployment recommendation: which threshold for which scenario

**Success Criteria:**
- Rapid response: 95% precision achieved with ≥ 60% recall
- Risk ranking: F1 ≥ 0.75
- Edge cases: 85% precision achieved with ≥ 40% recall

---

### Phase 2: Manual Specialized Models (1-2 weeks)

**ONLY if Phase 1 fails to achieve edge case performance.**

**Approach:** Build 2 specialized models with manual routing.

**Model 1: Standard Clearings**

**Training Data (200 samples):**
- Large clearings (> 1 ha)
- Continuous forest (fragmentation < 0.5)
- Low fire activity

**Features:**
- 10 baseline features (Y-1 temporal dynamics)
- 7 delta features (Y - Y-1 change)

**Target:** 0.85-0.90 ROC-AUC

**Model 2: Edge Cases**

**Training Data (150 samples):**
- Small clearings (< 1 ha)
- Fire-prone areas (fire activity > 5 events in 6 months)
- Fragmented forests (fragmentation > 0.7)
- Forest edges

**Additional Features:**
- Fire history (6-month, 12-month counts)
- Fragmentation index
- Distance to existing clearings
- Multiscale embeddings (if available)

**Target:** 0.70-0.75 ROC-AUC (realistic ceiling)

**Routing Logic:**
```python
def route_to_expert(sample):
    """Classify sample into standard or edge case."""

    # Extract routing features
    size = estimate_clearing_size(sample)  # From spatial footprint
    fire_activity = get_fire_count_6mo(sample.location)
    fragmentation = compute_fragmentation(sample.location)

    # Simple rule-based routing
    if size < 1.0:  # Small clearing
        return 'edge_model'
    elif fire_activity > 5:  # Fire-prone
        return 'edge_model'
    elif fragmentation > 0.7:  # Fragmented
        return 'edge_model'
    else:
        return 'standard_model'

def predict(sample):
    expert = route_to_expert(sample)
    if expert == 'edge_model':
        return edge_model.predict_proba(sample)
    else:
        return standard_model.predict_proba(sample)
```

**Success Criteria:**
- Standard model: 0.85+ ROC-AUC on risk_ranking
- Edge model: 0.70+ ROC-AUC on edge_cases
- Routing accuracy: 90%+ correct classification

**If SUCCESS:**
- ✓ Deploy with manual routing
- Document routing rules clearly
- Monitor edge case performance

**If FAILURE (routing rules don't work):**
- → Consider Phase 3 (MoE) only if:
  - Can collect 500+ samples
  - Deployment complexity acceptable
  - Simple rules genuinely don't work

---

### Phase 3: Mixture of Experts (2-3 weeks)

**ONLY pursue if Phases 1 and 2 fail AND:**
1. Can collect 500-800 samples
2. Manual routing rules are unreliable
3. Deployment infrastructure supports multi-model complexity
4. Patterns change over time (need adaptive routing)

**Architecture:**

**Expert 1: Standard Clearings**
- Training: 250 samples (large clearings, continuous forest)
- Target: 0.90 ROC-AUC

**Expert 2: Edge Cases**
- Training: 150 samples (small, fire-prone, fragmented)
- Target: 0.75 ROC-AUC

**Expert 3: Generalist**
- Training: 400 samples (all scenarios)
- Target: 0.82 ROC-AUC (fallback)

**Gating Network:**
```python
# Input: Sample features + metadata
gating_features = [
    *sample.features,  # 17 temporal features
    sample.size_estimate,
    sample.fire_history_6mo,
    sample.fragmentation_index,
    sample.forest_type,
]

# Output: Weights for each expert (sum to 1)
weights = gating_network.predict(gating_features)
# weights = [0.1, 0.7, 0.2] = mostly edge expert, some generalist

# Final prediction
prediction = (
    weights[0] * expert1.predict_proba(sample) +
    weights[1] * expert2.predict_proba(sample) +
    weights[2] * expert3.predict_proba(sample)
)
```

**Training Approach:**

**Option A: End-to-End (Harder)**
```python
# Train all experts + gating jointly to optimize final prediction
# Requires differentiable models or RL
```

**Option B: Staged (Easier)**
```python
# 1. Train experts independently on their data
expert1 = train(standard_data)
expert2 = train(edge_data)
expert3 = train(all_data)

# 2. Collect expert predictions on validation set
expert_preds = {
    'expert1': expert1.predict_all(validation_set),
    'expert2': expert2.predict_all(validation_set),
    'expert3': expert3.predict_all(validation_set),
}

# 3. Train gating network to weight experts
# Labels: which expert was most accurate for each sample
gating_network = train_gating(
    features=gating_features,
    labels=expert_performance_labels
)
```

**Success Criteria:**
- Overall ROC-AUC ≥ 0.85 across all sets
- Edge cases: 0.75+ ROC-AUC
- Gating network accuracy: 85%+ correct expert selection

**Data Requirements:**
- Expert 1: 250 samples
- Expert 2: 150 samples
- Expert 3: 400 samples
- Gating training: 200 validation samples
- **Total: 600-800 unique samples**

---

## Detailed Implementation Plans

### Phase 0: Diagnostic Analysis

**Script:** `src/walk/07_edge_case_diagnostic_analysis.py`

**Objectives:**
1. Compare feature distributions across validation sets
2. Identify which edge cases are missed
3. Analyze sample characteristics
4. Test feature importance differences

**Implementation:**

```python
def analyze_feature_distributions():
    """Compare features across validation sets."""

    for val_set in ['risk_ranking', 'comprehensive', 'edge_cases']:
        features = load_features(val_set)

        # Statistical tests
        for feature_idx in range(17):
            # Compare to training distribution
            ks_stat, p_value = ks_test(
                training_features[:, feature_idx],
                features[:, feature_idx]
            )

            if p_value < 0.05:
                print(f"{val_set}: Feature {feature_idx} differs (p={p_value})")

def analyze_failure_modes():
    """Analyze which edge cases are missed."""

    edge_samples = load_validation_set('edge_cases')
    model = load_model()

    predictions = model.predict_proba(edge_samples.features)
    errors = (predictions < 0.5) & (edge_samples.labels == 1)

    # Characterize missed clearings
    missed_samples = edge_samples[errors]

    print(f"Missed {len(missed_samples)}/{len(edge_samples)} edge cases")
    print(f"Average size: {missed_samples.size.mean()}")
    print(f"Fire activity: {missed_samples.fire_6mo.mean()}")
    print(f"Fragmentation: {missed_samples.fragmentation.mean()}")

def test_feature_importance():
    """Test if edge cases use different features."""

    # Train on edge cases only
    edge_model = train_model(edge_case_training_data)

    # Compare feature importance
    standard_importance = standard_model.coef_
    edge_importance = edge_model.coef_

    # Correlation
    corr = np.corrcoef(standard_importance, edge_importance)[0, 1]
    print(f"Feature importance correlation: {corr:.3f}")

    if corr < 0.7:
        print("→ Edge cases use different features, consider specialization")
    else:
        print("→ Similar features, likely data scarcity issue")
```

**Deliverables:**
- Feature distribution comparisons (KS tests)
- Error analysis report
- Feature importance comparison
- Recommendation: Scale vs Specialize

**Time:** 1-2 days

---

### Phase 1: Scaled Training Implementation

**Script:** `src/walk/08_scaled_training_experiment.py`

**Data Collection Strategy:**

```python
def collect_diverse_training_data(n_total=300):
    """Collect 300 samples with intentional diversity."""

    samples = []

    # 60% standard clearings (180 samples)
    standard = collect_clearings(
        min_size=1.0,  # hectares
        max_fragmentation=0.5,
        max_fire_activity=3,
        n=180
    )
    samples.extend(standard)

    # 20% small clearings (60 samples)
    small = collect_clearings(
        min_size=0.1,
        max_size=1.0,
        n=60
    )
    samples.extend(small)

    # 10% fire-prone (30 samples)
    fire_prone = collect_clearings(
        min_fire_activity=5,  # events in 6 months
        n=30
    )
    samples.extend(fire_prone)

    # 10% forest edges (30 samples)
    edges = collect_clearings(
        min_fragmentation=0.7,
        n=30
    )
    samples.extend(edges)

    return samples

def train_and_evaluate_scaled_model():
    """Train on diverse data and evaluate."""

    # Collect data
    training_data = collect_diverse_training_data(300)

    # Extract features
    X_train, y_train = extract_features(training_data)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train)

    # Evaluate on all validation sets
    results = {}
    for val_set in ['risk_ranking', 'comprehensive', 'edge_cases']:
        X_val, y_val = load_validation_set(val_set)
        X_val_scaled = scaler.transform(X_val)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

        results[val_set] = {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred_proba > 0.5),
            'recall': recall_score(y_val, y_pred_proba > 0.5),
        }

    return results
```

**Success Metrics:**
- Edge cases: 0.583 → 0.70+ ROC-AUC
- Risk ranking: Maintain 0.80+ ROC-AUC
- Comprehensive: Maintain 0.75+ ROC-AUC

**Time:** 2-3 days (data collection + training + evaluation)

---

### Phase 1b: Threshold Optimization Implementation

**Script:** `src/walk/09_threshold_optimization.py`

```python
def optimize_threshold_for_use_case(model, val_set, use_case):
    """Find optimal threshold for specific use case."""

    X_val, y_val = load_validation_set(val_set)
    y_scores = model.predict_proba(X_val)[:, 1]

    if use_case == 'rapid_response':
        # Maximize recall subject to 95% precision
        threshold = find_threshold_with_precision_constraint(
            y_val, y_scores,
            min_precision=0.95
        )

    elif use_case == 'risk_ranking':
        # Maximize F1 score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [
            f1_score(y_val, y_scores > t)
            for t in thresholds
        ]
        threshold = thresholds[np.argmax(f1_scores)]

    elif use_case == 'edge_cases':
        # Maximize recall subject to 85% precision
        threshold = find_threshold_with_precision_constraint(
            y_val, y_scores,
            min_precision=0.85
        )

    return threshold

def find_threshold_with_precision_constraint(y_true, y_scores, min_precision):
    """Find highest recall threshold that maintains precision."""

    thresholds = np.linspace(0, 1, 1000)

    for t in sorted(thresholds, reverse=True):  # Start high
        y_pred = y_scores > t
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if prec >= min_precision:
            return t

    return 0.99  # Couldn't find threshold, use very high threshold
```

**Deliverables:**
- Optimal thresholds per use case
- Precision-recall curves
- Performance tables at optimal thresholds

**Time:** 1 day

---

## Decision Criteria

### After Phase 0 (Diagnostic Analysis)

**Proceed to Phase 1 if:**
- ✓ Feature distributions similar across sets
- ✓ No clear pattern in missed edge cases
- ✓ Feature importance similar

**Proceed directly to Phase 2 if:**
- ✗ Edge cases use fundamentally different features
- ✗ Clear pattern: all small clearings missed
- ✗ Fire-prone areas show different delta patterns

### After Phase 1 (Scaled Training)

**SUCCESS - Deploy:**
- ✓ Edge cases ≥ 0.70 ROC-AUC
- ✓ All sets ≥ 0.75 ROC-AUC
- ✓ Precision maintained ≥ 85%

**PARTIAL - Proceed to Phase 1b:**
- ~ Edge cases 0.65-0.70 ROC-AUC
- ~ Other sets maintained
- ~ Try threshold optimization

**FAILURE - Proceed to Phase 2:**
- ✗ Edge cases < 0.65 ROC-AUC
- ✗ Scaling alone insufficient

### After Phase 2 (Manual Specialized)

**SUCCESS - Deploy:**
- ✓ Standard model ≥ 0.85 ROC-AUC
- ✓ Edge model ≥ 0.70 ROC-AUC
- ✓ Routing accuracy ≥ 90%

**FAILURE - Consider Phase 3:**
- ✗ Routing rules unreliable (< 85% accuracy)
- ✗ Can collect 500+ samples
- ✗ Deployment complexity acceptable

### After Phase 3 (MoE)

**SUCCESS - Deploy:**
- ✓ Overall ≥ 0.85 ROC-AUC
- ✓ Edge cases ≥ 0.75 ROC-AUC
- ✓ Gating accuracy ≥ 85%

**FAILURE - Reassess Problem:**
- ✗ MoE doesn't help
- ✗ May need different approach:
  - Additional data sources (radar, weather)
  - Different model architecture (deep learning)
  - Accept edge case limitations

---

## Resource Requirements

### Data Collection

| Phase | Samples Needed | Estimated Time | Cost |
|-------|----------------|----------------|------|
| **Phase 0** | 0 (use existing) | 1-2 days | Low (analysis only) |
| **Phase 1** | 300 total | 1 week | Medium (API calls, labeling) |
| **Phase 1b** | 0 (use Phase 1) | 1 day | Low |
| **Phase 2** | 350 total | 1-2 weeks | Medium-High |
| **Phase 3** | 600-800 total | 2-3 weeks | High |

### Development Time

| Phase | Analysis | Implementation | Testing | Total |
|-------|----------|----------------|---------|-------|
| **Phase 0** | 1-2 days | - | - | 1-2 days |
| **Phase 1** | - | 1-2 days | 1 day | 2-3 days |
| **Phase 1b** | - | 0.5 days | 0.5 days | 1 day |
| **Phase 2** | 1 day | 3-5 days | 2 days | 1-2 weeks |
| **Phase 3** | 2 days | 7-10 days | 3 days | 2-3 weeks |

### Deployment Complexity

| Approach | Models | Infrastructure | Maintenance | Complexity |
|----------|--------|----------------|-------------|------------|
| **Phase 1** | 1 | Simple | Low | ★☆☆☆☆ |
| **Phase 1b** | 1 | Simple + config | Low | ★★☆☆☆ |
| **Phase 2** | 2-3 | Medium | Medium | ★★★☆☆ |
| **Phase 3** | 4 (3 + gating) | Complex | High | ★★★★★ |

---

## Success Metrics

### Overall Project Goals

**Minimum Viable Performance:**
- All validation sets ≥ 0.75 ROC-AUC
- Precision ≥ 85% across all sets
- Recall ≥ 60% for high-priority use cases

**Stretch Goals:**
- Edge cases ≥ 0.75 ROC-AUC
- Standard cases ≥ 0.85 ROC-AUC
- Deployment with single model (simplicity)

### Per-Phase Success

**Phase 0:** Clear diagnosis of edge case issue
- ✓ Understand feature differences
- ✓ Identify failure patterns
- ✓ Recommendation: scale vs specialize

**Phase 1:** Edge case improvement through scaling
- ✓ Edge cases: 0.583 → 0.70+ ROC-AUC (20% improvement)
- ✓ Maintain standard case performance
- ✓ One model sufficient

**Phase 1b:** Use-case-optimized thresholds
- ✓ Rapid response: 95% precision, 60%+ recall
- ✓ Risk ranking: F1 ≥ 0.75
- ✓ Edge cases: 85% precision, 40%+ recall

**Phase 2:** Specialized model effectiveness
- ✓ Standard model: 0.85+ ROC-AUC
- ✓ Edge model: 0.70+ ROC-AUC
- ✓ Routing accuracy: 90%+

**Phase 3:** MoE superior to alternatives
- ✓ Overall: 0.85+ ROC-AUC
- ✓ Edge cases: 0.75+ ROC-AUC
- ✓ Complexity justified by performance

---

## Recommended Immediate Action

**START WITH: Phase 0 (Diagnostic Analysis)**

**Why:**
- 1-2 day investment
- No new data needed
- Critical information for next steps
- Prevents wasted effort on wrong approach

**Next Steps:**
1. Run diagnostic analysis (1-2 days)
2. Review results and make informed decision
3. Most likely: Proceed to Phase 1 (scaled training)
4. Reserve Phases 2-3 for if Phase 1 insufficient

**Expected Outcome (80% confidence):**
- Phase 0 shows data scarcity issue
- Phase 1 (scaled training) solves edge case problem
- Deploy with single model + threshold tuning
- No MoE needed

**Probability Estimates:**
- Phase 1 sufficient: 60%
- Phase 2 needed: 25%
- Phase 3 needed: 10%
- Fundamental limitation: 5%

---

## Conclusion

The **phased approach minimizes risk and resource investment** while systematically addressing performance variability:

1. **Diagnose first** (Phase 0) - understand the problem
2. **Try simplest solution** (Phase 1) - scaled training
3. **Add complexity only if needed** (Phases 2-3) - specialization, MoE

**Most likely outcome:** Scaled training with diverse data (Phase 1) improves edge case performance to acceptable levels (0.70-0.75 ROC-AUC), and threshold tuning (Phase 1b) optimizes for different use cases. MoE (Phase 3) is unlikely to be necessary.

**Recommendation:** Begin with Phase 0 diagnostic analysis immediately.
