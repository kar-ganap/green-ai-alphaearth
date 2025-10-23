# Temporal Validation Journey and Learnings

**Date:** 2025-10-22
**Purpose:** Document the divergence from original WALK plan, what we learned, and how it changes our strategy

---

## Table of Contents

1. [Where We Were Supposed To Be](#where-we-were-supposed-to-be)
2. [Where We Actually Went](#where-we-actually-went)
3. [Why We Diverged](#why-we-diverged)
4. [What We Learned](#what-we-learned)
5. [How This Changes The Path Forward](#how-this-changes-the-path-forward)
6. [Comparison: Plan vs Reality](#comparison-plan-vs-reality)

---

## Where We Were Supposed To Be

### Original WALK Phase Plan

From `docs/walk_phase_overview.md` (Original 12-17 hour estimate):

```
Phase Structure:
├─ 1. Data Preparation ✓ (Complete)
│   └─ Spatial CV splits, temporal features, 100+ samples
├─ 2. Baseline Suite ✓ (Complete)
│   └─ Random, raw embeddings, simple features, Random Forest
├─ 3. Spatial Feature Engineering (Planned)
│   └─ Neighborhood stats, local variance, edge proximity
├─ 4. Q4-Specific Features (Planned)
│   └─ Amplify weak Q4 precursor signal
└─ 5. Validation Protocol (Planned)
    └─ Lock down metrics, leakage checks, error analysis
```

**Key Assumptions:**
- Q4 precursor signal (d=0.81 from CRAWL) could be amplified with features
- Spatial CV sufficient for generalization
- Detection (Q2-Q3) vs Prediction (Q4) distinction primary goal
- Single model trained on ~100-200 samples

**Timeline:** 12-17 hours total for WALK phase

---

### Original Scaling Strategy

From `docs/scaling_and_specialization_strategy.md`:

```
Phased Approach (if baseline insufficient):
├─ Phase 0: Diagnostic Analysis (1-2 days)
│   └─ Why do edge cases fail?
├─ Phase 1: Scaled Training (2-3 days)
│   └─ 300 samples with intentional diversity
├─ Phase 2: Specialized Models (1-2 weeks)
│   └─ Separate models for standard vs edge cases
└─ Phase 3: Mixture of Experts (2-3 weeks)
    └─ Learned gating network (only if Phases 1-2 fail)
```

**Assumption:** Edge case underperformance (0.583 ROC-AUC) was a **data scarcity issue**, solvable by scaling to 300+ samples.

**Expected Outcome:** Phase 1 (scaling) would bring edge cases from 0.583 → 0.70+ ROC-AUC.

---

### Key Missing Element: Temporal Validation

**Original plan did NOT emphasize:**
- Training on past years, testing on future years
- Explicit temporal drift measurement
- Multi-year temporal validation sequence
- Temporal distribution shift analysis

**Why it was missing:**
- Assumed spatial CV would capture generalization
- Focused on detection vs prediction (Q2-Q3 vs Q4) distinction
- Didn't anticipate significant temporal drift

---

## Where We Actually Went

### Actual Implementation Path

What we built instead:

```
Actual WALK Phase (October 2025):
├─ 1. Data Preparation ✓
│   └─ 114 training samples (2020-2023)
│
├─ 2. Baseline Suite ✓
│   └─ Random Forest baseline
│
├─ 3. Phase 1: Scaled Training ✓
│   └─ 685 samples (intentional diversity)
│   └─ Edge cases: 0.583 → 0.60-0.65 (partial improvement)
│
├─ 4. Hard Validation Sets ✓
│   ├─ risk_ranking (46 samples)
│   ├─ rapid_response (27 samples)
│   ├─ comprehensive (69 samples)
│   └─ edge_cases (23 samples)
│
├─ 5. Threshold Optimization ✓
│   └─ Use-case-specific thresholds (0.070 to 0.910)
│
├─ 6. Multiscale Features ✓
│   └─ 69D features (3D annual + 66D coarse multiscale)
│
├─ 7. Progressive Temporal Validation ✓ ← **MAJOR DIVERGENCE**
│   ├─ Phase 1: 2020-2021 → 2022 (0.976 ROC-AUC)
│   ├─ Phase 2: 2020-2022 → 2023 (0.982 ROC-AUC)
│   ├─ Phase 3: 2020,2022,2023 → 2021 (0.981 ROC-AUC)
│   └─ Phase 4: 2020-2023 → 2024 (0.796 ROC-AUC) ⚠️ 18.9% DRIFT
│
├─ 8. Drift Investigation ✓
│   └─ Feature distribution analysis: 46/69 features shifted
│
└─ 9. Drift Decomposition Experiment ✓ ← **THIS WORK**
    └─ Uniform 30% vs Heterogeneous thresholds
    └─ Finding: Drift is REAL, not sampling bias
```

**Timeline:** ~3 weeks of intensive work (not 12-17 hours)

---

## Why We Diverged

### Critical Discovery: Temporal Drift

**What triggered the shift:**

1. **Phase 1 scaling showed partial success:**
   - Edge cases improved: 0.583 → 0.60-0.65
   - But still below target (0.70)
   - Standard cases maintained performance
   - **Conclusion:** Scaling helped but didn't solve the problem

2. **Hard validation sets revealed performance variability:**
   - risk_ranking: 0.850 ROC-AUC
   - comprehensive: 0.758 ROC-AUC
   - edge_cases: 0.583 ROC-AUC
   - **Gap:** 0.583 to 0.850 is substantial

3. **Realized we hadn't tested future generalization:**
   - All validation was on 2020-2023 data (same temporal distribution)
   - No test on genuinely unseen future years
   - Spatial CV ≠ temporal generalization

4. **Phase 4 revealed the problem:**
   - Test on 2024 data: 0.796 ROC-AUC (18.9% drop from CV)
   - This was NOT in the original plan
   - Changed everything

---

### The "Aha" Moment

**Original assumption:**
> "Spatial CV captures generalization. If model performs well across spatially separated regions, it will generalize."

**Reality:**
> "Spatial generalization ≠ temporal generalization. Data distribution shifts over time, and this can't be caught by spatial CV alone."

**Key realization:**
- CV ROC-AUC (2020-2023): 0.982 ← Based on spatial splits
- Test ROC-AUC (2024): 0.796 ← Real future data
- **The 18.9% drop reveals temporal drift that spatial CV didn't catch**

---

### Why This Matters More Than Edge Cases

**Original problem:** Edge cases at 0.583 ROC-AUC

**Discovered problem:** Entire model drops to 0.796 ROC-AUC on 2024 data

**Impact:**
- Edge case issue is a **performance ceiling** problem (can we go from 0.60 → 0.75?)
- Temporal drift is a **reliability** problem (will the model work next year?)

**Priority shift:**
- Original: "How do we improve edge case performance?"
- Now: "How do we ensure the model works on future data?"

---

## What We Learned

### Learning 1: Temporal Drift is Real and Significant

**Finding:**
- 18.9% ROC-AUC drop (0.982 → 0.796) on 2024 data
- 46 out of 69 features showed significant distribution shifts (KS test p < 0.01)
- Both heterogeneous and uniform 30% models experience similar drift

**Implication:**
- Models trained on 2020-2023 do NOT generalize well to 2024
- This is a genuine distributional change, not a sampling artifact
- **Cannot deploy production model based on 2020-2023 training alone**

---

### Learning 2: Sampling Bias is NOT the Primary Driver

**Hypothesis we tested:**
> "The 18.9% drift is caused by heterogeneous Hansen tree cover thresholds (30%, 40%, 50%) in training vs uniform 30% in 2024 test."

**Experiment:**
- Collected uniform 30% dataset (2020-2023, 588 samples)
- Trained model with same thresholds as 2024 test
- Compared performance

**Result:**
```
Heterogeneous (30-50%):
  CV (2020-2023):  0.982
  Test (2024):     0.796
  Drift:           0.186 (18.9%)

Uniform 30%:
  CV (2020-2023):  1.000
  Test (2024):     0.809
  Drift:           0.191 (19.1%)

Test performance difference: +0.013 (only 1.3%)
```

**Conclusion:**
- Sampling bias contributes only ~1.3% to the drift
- Real temporal change accounts for ~18.6% (dominant)
- **The 2024 data distribution has genuinely shifted**

---

### Learning 3: Spatial CV Alone is Insufficient

**What we validated:**
- Spatial generalization: Models generalize across different geographic locations (10km separation)
- Spatial CV ROC-AUC: 0.97-0.98 (robust)

**What spatial CV MISSED:**
- Temporal distribution shift
- Year-to-year environmental changes
- Seasonal or systematic data changes

**Why it matters:**
- Spatial CV gives false confidence
- High CV scores don't guarantee production reliability
- Need explicit temporal validation for deployment

---

### Learning 4: Progressive Temporal Validation Protocol

**What worked:**

Phased temporal validation approach:
1. **Phase 1:** 2020-2021 → 2022 (short-term, 1 year)
2. **Phase 2:** 2020-2022 → 2023 (mid-term, 1-3 years)
3. **Phase 3:** 2020,2022,2023 → 2021 (held-out year)
4. **Phase 4:** 2020-2023 → 2024 (future, 1+ year ahead)

**Key insight:**
- Phases 1-3: All showed 0.97-0.98 ROC-AUC (stable)
- Phase 4: Dropped to 0.796 (unstable)
- **The gap between train years and test year matters:**
  - Within training temporal range (2020-2023): Stable
  - Outside training temporal range (2024): Drift

**Implication:**
- Models interpolate well (test on years within training range)
- Models extrapolate poorly (test on years beyond training range)
- **Need continuous data updates for production**

---

### Learning 5: Feature-Level Analysis Reveals Drift Mechanisms

**Feature distribution shifts (2024 vs 2020-2023):**

| Feature Category | # Shifted / Total | KS p < 0.01 |
|------------------|-------------------|-------------|
| Annual features (delta, accel) | 3 / 3 | 100% |
| Embedding dimensions | 43 / 64 | 67% |
| Heterogeneity/range | 0 / 2 | 0% |

**What this tells us:**
- Annual change features (delta_1yr, delta_2yr, acceleration) ALL shifted
- Most embedding dimensions shifted (AlphaEarth representation changed)
- Heterogeneity/range stable (landscape statistics consistent)

**Possible causes:**
- Environmental changes (different deforestation patterns in 2024)
- Seasonal effects (data collection timing)
- Hansen GFC label quality shifts
- Real-world distribution changes

---

### Learning 6: Experimental Design Rigor Pays Off

**The uniform 30% experiment demonstrated:**

✓ **Controlled experiment design:**
- Single variable isolation (threshold heterogeneity)
- Held other factors constant (features, model, time periods)
- Clear hypothesis and measurable outcome

✓ **Definitive answer:**
- Sampling bias: ~1.3% contribution (minimal)
- Temporal drift: ~18.6% contribution (dominant)
- No ambiguity in interpretation

✓ **Prevented wasted effort:**
- Could have spent weeks "fixing" sampling strategy
- Would not have addressed the real problem
- Experiment saved weeks of misdirected work

**Key lesson:**
> When facing unexplained performance drop, run controlled experiments to isolate causes BEFORE implementing fixes.

---

## How This Changes The Path Forward

### Old Plan (Pre-Temporal Drift Discovery)

```
Original WALK Phase Completion:
├─ Scale to 300+ samples
├─ Optimize thresholds per use case
├─ Specialized models if needed (Phase 2)
├─ Deploy production model
└─ Monitor edge case performance
```

**Assumption:** Model trained on 2020-2023 would work indefinitely.

---

### New Plan (Post-Temporal Drift Discovery)

```
Revised Strategy:
├─ 1. Accept Temporal Drift as Reality
│   └─ 18.9% drift is not fixable with data collection alone
│
├─ 2. Temporal Adaptation Strategy
│   ├─ Option A: Retrain with 2024 data included
│   ├─ Option B: Rolling window training (e.g., last 3 years)
│   ├─ Option C: Online learning / incremental updates
│   └─ Option D: Domain adaptation techniques
│
├─ 3. Production Deployment Considerations
│   ├─ Drift monitoring pipeline (detect distribution shifts)
│   ├─ Retraining schedule (quarterly? annually?)
│   ├─ Model versioning and rollback capability
│   └─ Performance tracking on recent data
│
├─ 4. Research Directions
│   ├─ Investigate drift mechanisms (why did 2024 shift?)
│   ├─ Test temporal domain adaptation methods
│   ├─ Explore causal features (invariant to drift)
│   └─ Ensemble across temporal models
│
└─ 5. Edge Case Improvement (Secondary)
    └─ Revisit Phase 1b-2 if temporal drift addressed
```

---

### Critical Strategic Shifts

#### Shift 1: From Spatial to Temporal Generalization

**Before:**
- Focus: Does model work across different regions?
- Validation: Spatial CV with 10km buffer
- Assumption: Spatial generalization implies deployment readiness

**After:**
- Focus: Does model work across time periods?
- Validation: Progressive temporal validation (Phases 1-4)
- Reality: Temporal drift is the primary deployment risk

---

#### Shift 2: From Static Model to Temporal Adaptation

**Before:**
- Train once on historical data (2020-2023)
- Deploy and monitor
- Update only if performance degrades

**After:**
- Continuous temporal validation required
- Periodic retraining necessary (quarterly/annually)
- Drift detection and model versioning critical
- **Models have shelf life - need maintenance strategy**

---

#### Shift 3: From Performance Optimization to Robustness

**Before:**
- Primary goal: Improve edge case performance (0.583 → 0.70)
- Secondary: Optimize thresholds per use case
- Tertiary: Feature engineering for weak signals

**After:**
- Primary goal: Ensure temporal robustness (prevent 18.9% drift)
- Secondary: Drift monitoring and adaptation
- Tertiary: Performance optimization (edge cases, thresholds)

**Priority inversion:**
- Used to optimize for peak performance
- Now optimize for consistent reliability across time

---

#### Shift 4: From Single Model to Model Lifecycle

**Before:**
```python
# Train once
model = train_on_historical_data(2020-2023)

# Deploy
deploy(model)

# Done
```

**After:**
```python
# Initial training
model_v1 = train(2020-2023_data)

# Temporal validation
temporal_performance = validate_on_future(model_v1, 2024_data)

# Drift detected?
if temporal_performance < threshold:
    # Retrain with recent data
    model_v2 = train(2021-2024_data)

# Continuous monitoring
monitor_drift(production_predictions, ground_truth)

# Scheduled retraining
if time_since_training > 3_months:
    model_vN = retrain_with_latest_data()
```

---

## Comparison: Plan vs Reality

### Data Collection

| Aspect | Original Plan | Reality |
|--------|--------------|---------|
| **Training samples** | 100-200 | 685 (2020-2023) + 162 (2024) |
| **Validation strategy** | Spatial CV only | Spatial + Temporal + Hard sets |
| **Temporal coverage** | Single temporal window | Multi-year progressive validation |
| **Edge case focus** | Intentional diversity | Hard validation sets + scaled training |

---

### Timeline

| Phase | Planned | Actual |
|-------|---------|--------|
| **WALK phase** | 12-17 hours | ~3 weeks |
| **Data collection** | 1-2 days | ~1 week (multiple iterations) |
| **Validation** | 1-2 days | ~1 week (Phases 1-4) |
| **Drift investigation** | Not planned | ~3 days |
| **Drift decomposition** | Not planned | ~2 days |

**Total:** Planned 2-3 days → Actual 3 weeks (10x longer)

---

### Discoveries

| Discovery | Planned | Uncovered |
|-----------|---------|-----------|
| **Spatial generalization** | ✓ Expected | ✓ Validated (0.97 ROC-AUC) |
| **Temporal drift** | ✗ Not anticipated | ⚠️ 18.9% drop (critical) |
| **Sampling bias** | ? Hypothesized cause | ✓ Ruled out (1.3% only) |
| **Feature distribution shifts** | ✗ Not considered | ✓ Quantified (46/69 features) |
| **Edge case scaling** | ✓ Expected to solve | ~ Partial improvement only |

---

### Strategic Pivots

| Aspect | Original Strategy | Revised Strategy |
|--------|------------------|------------------|
| **Primary risk** | Edge case underperformance | Temporal drift |
| **Deployment assumption** | Train once, deploy | Continuous retraining needed |
| **Validation priority** | Spatial CV sufficient | Temporal validation critical |
| **Model lifecycle** | Static model | Dynamic with versioning |
| **Performance target** | 0.70+ edge cases | 0.80+ with temporal robustness |

---

## Key Takeaways for ML Systems

### Takeaway 1: Temporal Validation is Non-Negotiable

**Lesson:**
> Spatial CV alone gives false confidence. Always validate on genuinely unseen future data before claiming production readiness.

**Why it matters:**
- Spatial CV: Tests geographic generalization
- Temporal validation: Tests real-world deployment conditions
- **Production = future data, not different locations**

---

### Takeaway 2: Controlled Experiments Save Time

**Lesson:**
> When facing unexplained performance drop, isolate variables with controlled experiments before implementing fixes.

**Our example:**
- Hypothesis: Sampling bias causes drift
- Experiment: Uniform 30% vs heterogeneous thresholds
- Result: Ruled out hypothesis definitively (1.3% contribution)
- Saved: Weeks of misdirected "fixing" sampling strategy

---

### Takeaway 3: Feature-Level Analysis Guides Diagnosis

**Lesson:**
> Aggregate metrics (ROC-AUC) reveal problems. Feature-level analysis reveals causes.

**What we found:**
- Aggregate: 18.9% drift
- Feature-level: 46/69 features shifted
- Specific: All annual features shifted (delta_1yr, delta_2yr, acceleration)
- **Insight:** Temporal change features are most sensitive to drift

---

### Takeaway 4: Models Have Shelf Life

**Lesson:**
> ML models trained on historical data degrade over time. Plan for continuous updates, not one-time deployment.

**Reality check:**
- CV (2020-2023): 0.982 ROC-AUC ← Looks great
- Test (2024): 0.796 ROC-AUC ← Production reality
- **Gap = temporal drift = maintenance burden**

---

### Takeaway 5: Rigorous Validation Uncovers Truth

**Lesson:**
> The more rigorous your validation, the more uncomfortable truths you'll discover. But better to find them before deployment.

**What rigor revealed:**
- Spatial CV looked good (0.98)
- Temporal validation revealed drift (0.796)
- Controlled experiment isolated cause (sampling vs temporal)
- **Each layer of validation added critical information**

---

## Documentation Value

### Why This Document Matters

**1. Organizational Memory:**
- Future team members can understand why we took this path
- Prevents re-litigating decisions
- Documents experimental reasoning

**2. Methodological Learning:**
- Controlled experiment design
- Progressive temporal validation
- Feature-level drift analysis
- Isolation of causal factors

**3. Strategic Clarity:**
- Why temporal drift matters more than edge cases
- Why production requires continuous updates
- Why sampling bias wasn't the problem

**4. Reproducibility:**
- Clear progression of work
- Hypothesis → Experiment → Finding → Decision
- Can replicate approach for other projects

---

## Next Steps

### Immediate (This Week)

1. **Document experiment results:**
   - ✓ Drift decomposition findings (this doc)
   - ✓ Uniform 30% validation results
   - Update WALK phase summary

2. **Strategic decision:**
   - Retrain with 2024 data included?
   - Implement rolling window training?
   - Test domain adaptation techniques?

3. **Production planning:**
   - Design drift monitoring pipeline
   - Define retraining schedule
   - Set up model versioning

### Short-term (Next 2 Weeks)

1. **Temporal adaptation experiment:**
   - Train on 2021-2024 (including recent data)
   - Test on held-out 2024 samples
   - Measure if drift persists

2. **Drift mechanism investigation:**
   - Why did 2024 shift?
   - Are changes systematic or random?
   - Can we predict future drift?

3. **Edge case revisit:**
   - With temporal drift addressed, revisit edge cases
   - Phase 1b threshold optimization
   - Phase 2 specialization if needed

### Long-term (Next Month+)

1. **Production deployment:**
   - Implement drift monitoring
   - Set up retraining pipeline
   - Deploy with temporal validation protocol

2. **Research directions:**
   - Causal features (invariant to drift)
   - Domain adaptation methods
   - Ensemble across temporal models

3. **Documentation:**
   - Production playbook
   - Drift monitoring guide
   - Retraining procedures

---

## Conclusion

**What we planned:** Simple WALK phase → Scale training → Optimize thresholds → Deploy

**What we discovered:** Temporal drift is real, significant, and the primary deployment risk.

**What we learned:**
1. Spatial CV ≠ temporal generalization
2. Sampling bias was NOT the cause (1.3% vs 18.6%)
3. Models need continuous updates, not one-time training
4. Controlled experiments isolate causes efficiently
5. Rigorous validation reveals uncomfortable truths before deployment

**Strategic pivot:**
- From: "How do we optimize edge case performance?"
- To: "How do we ensure temporal robustness and continuous adaptation?"

**Value of this work:**
> By discovering and quantifying temporal drift BEFORE production deployment, we prevented a model that would have silently degraded in production. The 3-week investment in temporal validation will save months of debugging production failures.

---

**Status:** Temporal validation complete, drift quantified, cause isolated.
**Next:** Design temporal adaptation strategy for production deployment.
**Documentation:** This session's learnings preserved for future reference.
