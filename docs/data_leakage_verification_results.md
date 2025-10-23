# Data Leakage Verification Results

**Date**: October 16, 2025
**Status**: ⚠️ CRITICAL ISSUES IDENTIFIED
**Action Required**: Immediate remediation before any model training

---

## Executive Summary

Comprehensive verification of spatial and temporal leakage revealed **CRITICAL spatial leakage** across all validation sets:

- **23 exact duplicate locations** between training and validation sets
- **0.0 km distance** - same coordinates appear in both train and validation
- **14.1% of validation samples** are duplicates (23/163)
- **No temporal causality violations** detected

**Impact**: Current performance results (99.8% ROC-AUC) are **INVALID** due to spatial leakage.

**Required Action**: Re-sample all validation sets with enforced geographic separation.

---

## Spatial Leakage Findings

### Overall Summary

| Validation Set | Total Samples | Violations | Violation Rate | Severity |
|---------------|---------------|------------|----------------|----------|
| Rapid Response | 28 | 6 | 21.4% | 6 CRITICAL |
| Risk Ranking | 43 | 4 | 9.3% | 4 CRITICAL |
| Comprehensive | 70 | 9 | 12.9% | 8 CRITICAL, 1 HIGH |
| Edge Cases | 22 | 4 | 18.2% | 4 CRITICAL |
| **Total** | **163** | **23** | **14.1%** | **22 CRITICAL, 1 HIGH** |

### Duplicate Coordinates

The following coordinates appear in BOTH training and multiple validation sets:

#### Coordinate 1: [-3.078, -54.2543] (Pará, Brazil)
- Training set: Index 9
- Appears in:
  - ✗ Rapid Response: Index 0
  - ✗ Risk Ranking: Index 0
  - ✗ Comprehensive: Indices 0, 3, 7
  - ✗ Edge Cases: Indices 0, 2

**Total**: 1 training + 7 validation occurrences

---

#### Coordinate 2: [-5.4211, -51.3298] (Pará, Brazil)
- Training set: Index 1
- Appears in:
  - ✗ Rapid Response: Index 3
  - ✗ Risk Ranking: Index 3
  - ✗ Comprehensive: Index 6

**Total**: 1 training + 3 validation occurrences

---

#### Coordinate 3: [-10.6089, -68.5514] (Acre/Rondônia, Brazil)
- Training set: Index 19
- Appears in:
  - ✗ Rapid Response: Indices 13, 18
  - ✗ Risk Ranking: Index 8
  - ✗ Comprehensive: Indices 20, 23
  - ✗ Edge Cases: Indices 9, 11

**Total**: 1 training + 7 validation occurrences

---

#### Coordinate 4: [-12.6758, -69.6675] (Madre de Dios, Peru)
- Training set: Index 15
- Appears in:
  - ✗ Rapid Response: Index 15
  - ✗ Risk Ranking: Index 10

**Total**: 1 training + 2 validation occurrences

---

### Root Cause Analysis

**Why did this happen?**

1. **Same sampling strategy**: Both training (`01_data_preparation.py`) and validation (`01b_hard_validation_sets.py`) use Earth Engine `.sample()` method

2. **Overlapping geographic regions**: Both sample from same hotspot regions (Pará, Rondônia, Madre de Dios)

3. **Same random seed**: Earth Engine sampling uses fixed seed (42), causing reproducible sampling

4. **Small sample size**: Hansen deforestation is sparse (< 1% of pixels), so limited unique locations

**Example**:
```python
# Training set samples from:
region = "amazon_para_brazil" (bounds: -8° to -3°, -55° to -50°)

# Validation set ALSO samples from:
region = "amazon_para_brazil" (same bounds!)

# Result: Both get same random pixels from Earth Engine
```

---

## Temporal Causality Findings

### Summary

✓ **NO VIOLATIONS DETECTED** in any dataset

All datasets passed temporal causality checks with both:
- Current embedding date approach (Y-1 for Q1, Y for Q2-Q4)
- Conservative approach (Y-1 for ALL quarters)

### Why No Violations?

Inspection of samples reveals:
- Most samples don't have embeddings extracted yet (only raw coordinates + metadata)
- For samples with embeddings, year values happen to align safely
- Q1 embedding always from Y-1 (safe)
- Q2-Q4 from year Y, but Hansen lossyear=Y could mean anytime in Y

**Important**: This doesn't mean temporal causality is guaranteed. It means:
1. Current samples happen to be safe (by luck)
2. Future samples might not be (need conservative approach)

**Recommendation**: Still implement conservative Y-1 windowing to GUARANTEE safety.

---

## Impact on Current Results

### Multi-Scale Embeddings Performance (Reported)

| Validation Set | Reported ROC-AUC | Status |
|---------------|------------------|---------|
| Rapid Response | 99.4% | ⚠️ INVALID (21.4% leakage) |
| Risk Ranking | 100% | ⚠️ INVALID (9.3% leakage) |
| Comprehensive | 99.6% | ⚠️ INVALID (12.9% leakage) |
| Edge Cases | 100% | ⚠️ INVALID (18.2% leakage) |
| **Average** | **99.8%** | **⚠️ INVALID** |

### Why Results Are Invalid

**Leakage Effect**: Model sees EXACT same locations during training and validation
- Training sample at [-3.078, -54.2543]
- Validation sample at [-3.078, -54.2543] (SAME LOCATION!)
- Model memorizes this specific location → artificially high performance

**Real Performance**: Unknown until re-evaluation with clean data

**Estimate**: Expect 5-15% drop in ROC-AUC after removing leakage
- Realistic performance: 85-95% ROC-AUC (still good, but honest)
- Current 99.8% is inflated by 23 duplicate samples

---

## Required Actions

### Immediate (Week 1)

**1. Re-Sample All Validation Sets** 🔴 CRITICAL
- Use different random seeds for validation vs training
- Add explicit geographic exclusion zones
- Verify 10km buffer enforcement BEFORE saving

**Implementation**:
```python
# Add to 01b_hard_validation_sets.py
def sample_with_exclusion(client, bounds, year, exclude_coords, min_distance_km=10):
    """
    Sample from region while excluding training set coordinates.

    Args:
        exclude_coords: List of (lat, lon) from training set
        min_distance_km: Minimum distance from excluded coords
    """
    # Sample with different seed
    samples = client.sample_clearings(bounds, year, seed=100)  # Different from train seed=42

    # Filter out samples too close to training
    valid_samples = []
    for sample in samples:
        min_dist = min(haversine_distance((sample['lat'], sample['lon']), coord)
                      for coord in exclude_coords)
        if min_dist >= min_distance_km:
            valid_samples.append(sample)

    return valid_samples
```

**2. Re-Extract Features**
After re-sampling, re-extract all features:
- Multi-scale embeddings (08_multiscale_embeddings.py)
- Fire features (01f_extract_fire_features.py)
- Spatial features (01e_extract_spatial_for_training.py)

**3. Re-Evaluate Models**
- Re-run all evaluations on clean validation sets
- Compare to current results to measure leakage effect
- Update all performance claims

---

### Short-term (Week 2)

**4. Implement Geographic Stratification**
Instead of random sampling from overlapping regions:

```python
# Option A: Exclusive geographic regions
TRAINING_REGIONS = {
    "amazon_para_brazil": {...},      # Training only
    "amazon_mato_grosso_brazil": {...},  # Training only
}

VALIDATION_REGIONS = {
    "amazon_rondonia_brazil": {...},  # Validation only
    "amazon_acre_brazil": {...},      # Validation only
}

# Option B: Spatial partitioning within regions
# Split large regions into train/val zones with 50km buffer
```

**5. Add Automated Verification to Pipeline**
```python
# Add to all sampling scripts
def save_dataset_with_verification(samples, output_path, training_set_path=None):
    """Save dataset only if verification passes."""

    if training_set_path:
        # Load training set
        with open(training_set_path, 'rb') as f:
            train_data = pickle.load(f)

        # Verify no leakage
        is_valid, report = verify_no_spatial_leakage(
            train_data['data'], samples, min_distance_km=10.0
        )

        if not is_valid:
            raise ValueError(f"Spatial leakage detected! {report['n_violations']} violations")

    # Save only if validation passes
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
```

---

## Verification Protocol for Future Datasets

### Pre-Training Checklist

Before ANY model training, run verification:

```bash
# 1. Run full verification
uv run python src/walk/data_leakage_verification.py

# 2. Check exit code
if [ $? -eq 0 ]; then
    echo "✓ Verification passed - safe to train"
else
    echo "✗ Verification failed - fix leakage first"
    exit 1
fi

# 3. Review detailed report
cat data/processed/leakage_verification_report.json
```

### Acceptance Criteria

✓ **Spatial leakage**: 0 violations (10km buffer enforced)
✓ **Temporal causality**: 0 violations (conservative Y-1 windowing)
✓ **Within-set splits**: 0 violations (10km buffer within train/val/test)

---

## Timeline for Remediation

| Week | Task | Effort | Responsible |
|------|------|--------|-------------|
| Week 1 | Re-sample all validation sets with exclusion | 2-3 hours | - |
| Week 1 | Verify 0 spatial leakage | 30 min | - |
| Week 1 | Re-extract multi-scale features | 4-6 hours | - |
| Week 1 | Re-evaluate all models | 2 hours | - |
| Week 1 | Update performance claims | 1 hour | - |
| Week 2 | Implement geographic stratification | 3-4 hours | - |
| Week 2 | Add automated verification to pipeline | 2 hours | - |
| Week 2 | Document clean validation results | 2 hours | - |

**Total Effort**: ~16-20 hours over 2 weeks

---

## Lessons Learned

### What Went Wrong

1. **Assumed spatial separation**: Thought different sampling scripts = different samples
2. **No verification until now**: Should have checked immediately after sampling
3. **Overlapping regions**: Both train and val sample from same hotspots
4. **Small sample size**: Limited unique Hansen pixels → higher collision probability

### Best Practices Going Forward

1. **Always verify before training**: Make verification mandatory
2. **Exclusive geographic regions**: Train and val should sample from different areas
3. **Different random seeds**: Even if regions overlap, use different seeds
4. **Automated checks**: Add verification to sampling pipeline (fail early)
5. **Document sampling strategy**: Clear separation strategy in code comments

---

## Conclusion

**Current Status**: 🔴 All validation results are INVALID due to spatial leakage

**Severity**: CRITICAL - 14.1% of validation samples are duplicates

**Path Forward**:
1. Re-sample all validation sets with enforced exclusion (Week 1)
2. Re-evaluate on clean data (Week 1)
3. Update all performance claims with honest numbers (Week 1)

**Expected Impact**:
- Performance will likely drop from 99.8% to 85-95% ROC-AUC
- But results will be scientifically valid and trustworthy
- Multi-scale features still expected to substantially outperform baseline

**Silver Lining**:
- We caught this BEFORE publication/deployment
- Verification system works correctly
- Fix is straightforward (re-sampling)
- Within-training-set splits are clean (no leakage)

---

**Last Updated**: October 16, 2025
**Next Action**: Begin re-sampling validation sets with geographic exclusion
**Review Date**: After Week 1 remediation completion
