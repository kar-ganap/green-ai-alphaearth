# WALK Phase - Strategic Decisions

**Date**: October 16, 2025
**Status**: Authoritative design decisions for WALK phase implementation
**Purpose**: Document critical decisions that diverge from or clarify original plan

---

## Context

During WALK phase implementation, we identified gaps between the original CRAWL/WALK plan and current reality. This document records strategic decisions made to address these gaps.

---

## Decision 1: Detection vs Prediction Distinction

### **Decision**: YES, we care about the distinction and will implement it

**Rationale**:
- Detection (0-3 month lag) and prediction (3-12 month lead time) are fundamentally different capabilities
- Different use cases require different lead times
- Scientific integrity requires honest claims about what the model can and cannot predict

**Implementation Strategy**:

### Fire-Related Deforestation
**Separate prediction and detection**:
- Use MODIS MCD64A1 burned area + GLAD alerts (if available) to identify fire-driven clearing
- For fire samples with quarterly dates:
  - **Prediction**: Q4 embedding (0-3 months before) vs Q1-Q3 embeddings
  - **Detection**: Q2-Q3 embeddings (concurrent, 6-9 months before clearing year)
- Test precursor signal hypothesis explicitly for fire-related clearings

### Non-Fire Deforestation
**Detection only** (do not claim prediction):
- Hansen annual loss year provides insufficient temporal resolution
- Cannot reliably determine if clearing happened in Q1 vs Q4 of loss year
- Frame as: "Detection of clearing within the year" (0-12 month window)
- Do NOT claim precursor signal or prediction capability

**Acceptance of Limitations**:
- We acknowledge temporal resolution constraints of Hansen data
- Will do the best we can with available labels
- Fire subset gets quarterly precision via GLAD/MODIS → can test prediction
- Non-fire subset stays at annual precision → detection only

**Model Training**:
- Start with single model for both fire and non-fire
- Train separate models ONLY if:
  - Performance gaps emerge between fire and non-fire
  - Prediction use case requires different architecture/features
  - Evaluation shows specialized models substantially outperform unified model

---

## Decision 2: Q4 Precursor Signal Validation

### **Decision**: YES, must validate precursor signal explicitly

**Rationale**:
- Original CRAWL phase found weak but detectable Q4 signal (d=0.81, p=0.02-0.05)
- **Critical**: Without validating this signal, prediction claims rest on "shaky ground"
- Cannot claim prediction capability without evidence of precursor signal

**Implementation Requirements**:

### 1. Explicit Q4 Hypothesis Testing
```
Null Hypothesis: Q4 embeddings (0-3 months before) show no difference from intact forest
Alternative: Q4 embeddings show detectable precursor signal

Test:
- Q4 clearings vs Q4 intact: Statistical significance (p < 0.05)
- Effect size: Cohen's d > 0.5 (medium effect)
- Model performance: Q4-only features → AUC > 0.70
```

### 2. Compare Q4 vs Q2-Q3 Performance
```
Detection (Q2-Q3):
- Expected: Strong signal (d=2-6 from CRAWL)
- Use case: Rapid response, enforcement

Prediction (Q4):
- Expected: Weak signal (d=0.81 from CRAWL)
- Use case: Early warning, prevention

Decision criteria:
- If Q4 AUC ≥ 0.70 AND significantly > random → Claim prediction
- If Q4 AUC < 0.70 OR not significant → Detection only
```

### 3. Q4-Specific Feature Engineering
Test features designed to amplify weak Q4 signal:
- **Context amplification**: Road proximity + Q4 distance
- **Neighbor context**: Q4 pixels near other changing pixels
- **Trajectory curvature**: Acceleration at Q4
- **Landscape fragmentation**: Edge effects at Q4

**Success Metric**: Keep Q4 features if ΔAUC > 0.02 over baseline

---

## Decision 3: Single Model vs Multiple Models

### **Decision**: Single model first, add specialized models only when needed

**Approach**:

### Phase 1: Single Unified Model
- Train one model on all clearing types (fire + non-fire)
- Use multi-scale features (10m + 30m + 100m)
- Evaluate on all 4 hard validation sets

### Phase 2: Use-Case-Specific Thresholds
Before building separate models, try threshold tuning:
- **Rapid Response**: Lower threshold for higher recall (≥80%)
- **Risk Ranking**: Calibrate probabilities for ranking quality
- **Comprehensive**: Balanced threshold for overall accuracy
- **Edge Cases**: Specialized threshold for small clearings

### Phase 3: Specialized Models (If Needed)
Build separate models ONLY if:
- Threshold tuning insufficient to meet targets
- Fire vs non-fire performance gap > 20% AUC
- Prediction vs detection require different features
- Use case has fundamentally different requirements

**Evidence Required**:
- Must show single model cannot meet targets even with threshold tuning
- Must demonstrate specialized model substantially improves (ΔAUC > 0.05)
- Deployment complexity justified by performance gain

---

## Decision 4: Data Leakage Prevention

### **Decision**: Implement explicit verification checks (spatial AND temporal)

**Rationale**:
- Without explicit checks, entire system is suspect
- Spatial and temporal leakage can inflate performance unrealistically
- Critical for scientific validity and deployment confidence

---

### 4A: Spatial Leakage Verification

**Implementation**:

```python
def verify_no_spatial_leakage(train_samples, val_samples, min_distance_km=10):
    """
    Verify no validation sample is within min_distance_km of any training sample.

    Returns:
        (is_valid, report_dict)
    """
    train_coords = np.array([[s['lat'], s['lon']] for s in train_samples])
    val_coords = np.array([[s['lat'], s['lon']] for s in val_samples])

    from scipy.spatial import cKDTree
    train_tree = cKDTree(train_coords)

    violations = []
    for i, val_coord in enumerate(val_coords):
        # Find nearest training sample
        distance_deg, idx = train_tree.query(val_coord)
        distance_km = distance_deg * 111.0  # Convert degrees to km

        if distance_km < min_distance_km:
            violations.append({
                'val_idx': i,
                'train_idx': idx,
                'distance_km': distance_km,
                'val_coord': val_coord,
                'train_coord': train_coords[idx]
            })

    is_valid = len(violations) == 0

    report = {
        'is_valid': is_valid,
        'n_violations': len(violations),
        'violations': violations,
        'min_distance_km': min_distance_km,
        'min_actual_distance_km': min([v['distance_km'] for v in violations]) if violations else None
    }

    return is_valid, report
```

**Required Checks**:
1. Training set (114 samples) vs All hard validation sets (163 samples)
2. Within training set spatial CV splits (10km buffer enforcement)
3. Between each of 4 hard validation sets (ensure no overlap)

**Enforcement**:
- Run verification before ANY model training
- If violations found → Re-sample violating samples from different regions
- Document verification results in model metadata

---

### 4B: Temporal Causality Verification ⚠️ **CRITICAL ISSUE IDENTIFIED**

**Problem Identified**:

Current approach has potential temporal leakage:

```python
# For a Hansen clearing in year Y (e.g., 2021):
Q1 = f"{Y-1}-06-01"  # Jun 2020 (9-12 months before)
Q2 = f"{Y}-03-01"    # Mar 2021 (6-9 months before)
Q3 = f"{Y}-06-01"    # Jun 2021 (3-6 months before)
Q4 = f"{Y}-09-01"    # Sep 2021 (0-3 months before)

# Problem: Hansen "lossyear=21" means clearing sometime in 2021
# Could be Jan 2021 → Q2, Q3, Q4 are AFTER clearing! (temporal leakage!)
# Could be Dec 2021 → All quarters are before clearing (valid)
```

**We don't know which**, so we might be training on post-clearing data thinking it's pre-clearing!

---

**Solution Options**:

### Option A: Conservative Temporal Windowing (RECOMMENDED)
Only use embeddings that are GUARANTEED to be before clearing:

```python
# For Hansen clearing in year Y:
Q1 = f"{Y-1}-03-01"  # Mar of previous year (15-18 months before)
Q2 = f"{Y-1}-06-01"  # Jun of previous year (12-15 months before)
Q3 = f"{Y-1}-09-01"  # Sep of previous year (9-12 months before)
Q4 = f"{Y-1}-12-01"  # Dec of previous year (6-9 months before)
Clearing = f"{Y+1}-06-01"  # Jun of next year (during/after, for verification)

# All embeddings are from year Y-1, clearing is in year Y
# Guaranteed temporal causality
# Trade-off: Longer lag (9-18 months instead of 0-12 months)
```

**Pros**:
- ✓ Guaranteed no temporal leakage
- ✓ Still tests precursor signal (Q4 at 6-9 months before is still prediction)
- ✓ Works with Hansen annual labels

**Cons**:
- ✗ Longer lag times (9-18 months instead of claimed 0-12 months)
- ✗ More conservative prediction claim

---

### Option B: GLAD Overlay for Quarterly Precision
Use GLAD alerts to get exact clearing quarter:

```python
# Step 1: Sample Hansen clearings
hansen_sample = sample_hansen_loss(year=2021)

# Step 2: Query GLAD alert date
glad_alert = get_glad_alert(lat, lon)
if glad_alert:
    alert_quarter = get_quarter(glad_alert.date)  # e.g., Q3 2021

    # Step 3: Extract embeddings BEFORE alert quarter
    if alert_quarter == "Q3":
        Q1 = "2020-06-01"  # Safe (15 months before)
        Q2 = "2021-03-01"  # Safe (6 months before Q3)
        Q3 = "2021-06-01"  # UNSAFE (concurrent with Q3)
        Q4 = None          # UNSAFE (after clearing)

    # Only use embeddings guaranteed before clearing
```

**Pros**:
- ✓ Can use shorter lag (3-6 months) for GLAD subset
- ✓ Precise temporal ordering
- ✓ Enables detection vs prediction distinction

**Cons**:
- ✗ GLAD covers only ~30-40% of clearings (fire-biased)
- ✗ Non-GLAD samples still need conservative approach
- ✗ More implementation complexity

---

### Option C: Visual Verification (Hybrid)
For validation sets only, manually inspect Sentinel-2 imagery:

```python
# For each validation sample:
# 1. View Sentinel-2 time series in Google Earth Engine
# 2. Manually identify clearing quarter
# 3. Label with precise temporal information
# 4. Use conservative embeddings for training, precise for validation

# Validation samples (163): Feasible to manually inspect
# Training samples (500-1000+): Not feasible at scale
```

**Pros**:
- ✓ Ground truth for validation sets
- ✓ Can verify model claims on precisely-dated samples
- ✓ Gold standard for evaluation

**Cons**:
- ✗ Labor-intensive (5-10 min per sample)
- ✗ Doesn't solve training set temporal leakage
- ✗ Not scalable to large training sets

---

**DECISION: Implement Multi-Pronged Approach**

### Immediate Actions:

**1. Conservative Windowing for All Training**
- Switch all training data to Option A (Y-1 embeddings only)
- Eliminates temporal leakage risk completely
- Document that prediction lag is 6-18 months (not 0-12 months)

**2. GLAD Overlay for Fire Subset**
- Implement Option B for fire-driven clearings
- Use GLAD quarterly dates where available
- Enables true 0-3 month prediction testing for fire subset

**3. Visual Verification for Validation Sets**
- Manually inspect 163 hard validation samples (Option C)
- Create gold-standard temporal labels for evaluation
- Estimate time: 163 samples × 5 min = ~14 hours (spread over 2-3 days)

---

### Temporal Leakage Verification Code

```python
def verify_temporal_causality(samples, embedding_dates_func):
    """
    Verify all embeddings are extracted BEFORE clearing event.

    Args:
        samples: List of samples with year and clearing dates
        embedding_dates_func: Function that returns embedding dates for a sample

    Returns:
        (is_valid, report_dict)
    """
    violations = []

    for i, sample in enumerate(samples):
        clearing_year = sample['year']
        embedding_dates = embedding_dates_func(sample)

        # Conservative check: All embeddings must be from year < clearing_year
        for quarter, date in embedding_dates.items():
            if quarter == 'Clearing':
                continue  # Clearing embedding is expected to be after

            emb_year = int(date.split('-')[0])

            if emb_year >= clearing_year:
                violations.append({
                    'sample_idx': i,
                    'clearing_year': clearing_year,
                    'quarter': quarter,
                    'embedding_date': date,
                    'embedding_year': emb_year,
                    'issue': f'{quarter} embedding from year {emb_year} >= clearing year {clearing_year}'
                })

    is_valid = len(violations) == 0

    report = {
        'is_valid': is_valid,
        'n_violations': len(violations),
        'violations': violations,
        'total_samples': len(samples),
        'violation_rate': len(violations) / len(samples) if samples else 0
    }

    return is_valid, report


def get_conservative_embedding_dates(sample):
    """
    Return embedding dates that GUARANTEE temporal causality.

    All embeddings from year BEFORE clearing year.
    """
    clearing_year = sample['year']

    return {
        'Q1': f"{clearing_year - 1}-03-01",  # Mar Y-1 (15-18 months before)
        'Q2': f"{clearing_year - 1}-06-01",  # Jun Y-1 (12-15 months before)
        'Q3': f"{clearing_year - 1}-09-01",  # Sep Y-1 (9-12 months before)
        'Q4': f"{clearing_year - 1}-12-01",  # Dec Y-1 (6-9 months before)
        'Clearing': f"{clearing_year + 1}-06-01",  # Jun Y+1 (verification)
    }
```

---

## Implementation Checklist

### Spatial Leakage Prevention
- [ ] Implement `verify_no_spatial_leakage()` function
- [ ] Run verification on:
  - [ ] Training (114) vs Hard Validation (163)
  - [ ] Within training spatial CV splits
  - [ ] Between 4 hard validation sets
- [ ] Document results in `docs/data_leakage_verification.md`
- [ ] Re-sample any violating samples from different regions
- [ ] Add verification to model training pipeline (automatic check before train)

### Temporal Causality Prevention
- [ ] Implement `verify_temporal_causality()` function
- [ ] Implement `get_conservative_embedding_dates()` function
- [ ] **Update `extract_quarterly_embeddings()` to use conservative dates**
- [ ] Run verification on all existing datasets
- [ ] Re-extract embeddings for training set with conservative dates
- [ ] Implement GLAD overlay for fire subset (optional but recommended)
- [ ] Visual verification for 163 validation samples (gold standard)
- [ ] Document temporal windowing strategy in all result reports

### Detection vs Prediction Distinction
- [ ] Implement GLAD alert extraction for fire samples
- [ ] Separate fire vs non-fire in evaluation
- [ ] Test Q4 precursor signal hypothesis explicitly
- [ ] Report detection (6-18 months lag) vs prediction (not yet validated) separately
- [ ] Update all performance claims to reflect conservative temporal windowing

### Model Architecture
- [ ] Train single unified model first
- [ ] Evaluate with use-case-specific thresholds
- [ ] Only build specialized models if unified model fails to meet targets

---

## Updated WALK Phase Timeline

Given these decisions, updated timeline:

**Week 1** (Current):
- Implement spatial and temporal leakage verification
- Re-extract training set embeddings with conservative dates
- Verify no leakage in existing datasets

**Week 2**:
- Implement GLAD overlay for fire subset
- Visual verification of 163 validation samples
- Separate fire vs non-fire in evaluation framework

**Week 3**:
- Explicit Q4 precursor signal testing
- Q4-specific feature engineering
- Compare Q4 (prediction) vs Q2-Q3 (detection) performance

**Week 4**:
- Train unified model with verified non-leaking data
- Evaluate on all 4 hard validation sets
- Use-case-specific threshold tuning

**Week 5**:
- Decision point: Specialized models needed?
- Final validation and documentation
- Production readiness assessment

---

## Acceptance Criteria for WALK Phase Completion

### Must Have (Required):
1. ✓ Spatial leakage verification passed (no violations within 10km)
2. ✓ Temporal causality verification passed (all embeddings before clearing)
3. ✓ All 4 hard validation sets meet performance targets:
   - Rapid Response: ≥80% recall, ≥70% precision
   - Risk Ranking: NDCG@10 ≥ 0.70
   - Comprehensive: ≥90% recall (size-invariant)
   - Edge Cases: ≥50% recall per challenge type
4. ✓ Q4 precursor signal validated (if claiming prediction capability)
5. ✓ Fire vs non-fire performance documented separately

### Should Have (Important):
1. Visual verification of validation set clearing dates (gold standard)
2. GLAD overlay for fire subset (enables true prediction testing)
3. Q4-specific feature engineering tested
4. Detection (6-18 months) vs Prediction (if validated) distinction documented

### Nice to Have (Optional):
1. Specialized models for different use cases
2. Expanded training set (500-1000 samples)
3. Transfer learning from AlphaEarth embeddings

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Temporal leakage in current data | 🔴 CRITICAL | Re-extract with conservative dates (Week 1) |
| Spatial leakage unverified | 🔴 HIGH | Implement verification checks (Week 1) |
| Q4 precursor signal not validated | 🟡 MEDIUM | Explicit hypothesis testing (Week 3) |
| GLAD coverage insufficient for fire subset | 🟡 MEDIUM | Visual verification fallback (Week 2) |
| Single model insufficient for all use cases | 🟢 LOW | Specialized models if needed (Week 5) |

---

## References

- Original CRAWL findings: `docs/extended_crawl_findings.md`
- WALK phase overview: `docs/walk_phase_overview.md`
- Hansen-GLAD overlay plan: `docs/three_critical_questions_answered.md`
- Hard validation sets design: `docs/hard_validation_sets_summary.md`

---

**Last Updated**: October 16, 2025
**Next Review**: After Week 1 leakage verification completion
**Approval**: User-confirmed strategic decisions
