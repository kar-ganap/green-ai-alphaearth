# Phase 1: GLAD Validation - Summary & Next Steps

**Date**: 2025-10-15
**Status**: ⚠️ INCONCLUSIVE

## Summary

Phase 1 attempted to validate whether we're detecting **true precursor signals** (preparation activities in Y-1 that precede clearing in Y) vs **early detection** (capturing early-year clearing in annual embeddings).

### Methodology

**Key Question**: Does Y-1 AlphaEarth embedding predict Q4 clearings better than Q1?

- **If YES** → True precursor signal (roads/camps in late Y-1, clearing in Q4 of Y)
- **If NO** → Early detection (clearing in Q1 of Y, captured in annual Y embedding)

### Technical Implementation

Successfully implemented GLAD alert access for precise clearing dates:

1. **Dataset Structure**:
   - Archived collections: `projects/glad/alert/{YEAR}final` (ImageCollections)
   - Year-specific bands: `alertDate20`, `conf20` for 2020

2. **Date Encoding**:
   - Julian day of year (1-365/366)
   - alertDate20: 231 = Day 231 of 2020 = August 18, 2020

3. **Access Pattern**:
   ```python
   glad_collection = ee.ImageCollection(f'projects/glad/alert/{year}final')
   glad = glad_collection.select(['alertDate20', 'conf20']).mosaic()
   sample = glad.sample(region=point, scale=30, numPixels=1)
   ```

### Results

**Data Quality**:
- Sample size: 24 clearings (2020-2022)
- GLAD enrichment: **13 / 24 (54%)** - decent success rate
- Missing dates: 11 clearings (46%)

**Quarterly Distribution**:
| Quarter | Count | Percentage | Status |
|---------|-------|------------|--------|
| Q1      | 1     | 7.7%       | ❌ Insufficient (need ≥3) |
| Q2      | 2     | 15.4%      | ❌ Insufficient (need ≥3) |
| Q3      | 8     | 61.5%      | ✅ Sufficient |
| Q4      | 2     | 15.4%      | ❌ Insufficient (need ≥3) |

**Statistical Results**:
- Q3 only: Mean distance = 0.639 ± 0.218, p < 0.0001 (highly significant)
- **Cannot compare Q1 vs Q4** due to insufficient samples

### Interpretation

**Status: INCONCLUSIVE**

We successfully extracted GLAD dates for 54% of clearings, but the quarterly distribution is heavily skewed toward Q3 (mid-year). This prevents us from testing the key hypothesis (Q4 vs Q1 prediction strength).

**Why Q3 dominates**:
- Deforestation in the Amazon peaks during the dry season (July-September)
- Our random sample reflects this natural seasonal pattern
- Q1 (Jan-Mar) and Q4 (Oct-Dec) have fewer clearings

## Key Findings

### What Worked ✅

1. **GLAD Access**: Successfully implemented ImageCollection-based access to archived GLAD data
2. **Date Extraction**: Correctly decoded Julian day encoding (alertDate = day of year)
3. **Enrichment Rate**: 54% success rate is reasonable for GLAD-Landsat (30m resolution)
4. **Q3 Signal**: Strong temporal signal detected for Q3 clearings (p < 0.0001)

### What Didn't Work ❌

1. **Sample Size**: 24 clearings insufficient for quarterly stratification
2. **Seasonal Bias**: Natural deforestation seasonality prevents Q1/Q4 comparison
3. **Validation Approach**: Need alternative strategy to test precursor hypothesis

## Recommendations

### Option A: Increase Sample Size 🎯 **RECOMMENDED**

**Approach**: Scale up to 100-200 clearings to get enough Q1 and Q4 samples

**Pros**:
- Maintains original validation logic
- Addresses root cause (insufficient samples)
- Natural deforestation patterns suggest ~10-15% Q1/Q4, so 100 samples → ~10-15 per quarter

**Cons**:
- Takes longer to run (~30-60 minutes)
- Still subject to seasonal bias

**Implementation**:
```bash
uv run python src/temporal_investigation/phase1_glad_validation.py --n-samples 120
```

### Option B: Alternative Validation ⚡ **FAST ALTERNATIVE**

**Approach**: Compare Q3 clearings to a baseline or use month-level granularity

**Month-level test**:
- Compare early months (Jan-Mar) vs late months (Oct-Dec)
- More granular than quarters, might reveal patterns

**Baseline comparison**:
- Test if Q3 distance is higher than expected by chance
- Use Q2-Q3 transition as control

**Pros**:
- Uses existing data
- Faster to implement
- More statistical power with 8 Q3 samples

**Cons**:
- Different hypothesis than original (less clean)
- Harder to interpret

### Option C: Pivot to Phase 2 🚀 **PRAGMATIC**

**Approach**: Accept temporal ambiguity and proceed with augmentation

**Rationale**:
- AlphaEarth annual embeddings have value regardless of precursor vs early detection
- Adding monthly features (NDVI, radar) addresses temporal resolution directly
- Can revisit precursor validation later with more data

**Pros**:
- Moves project forward
- Addresses core limitation (annual resolution)
- Phase 2 features will help either way

**Cons**:
- Leaves precursor question unanswered
- Might build on wrong assumption

## Decision Criteria

**Choose Option A if**:
- Need definitive answer on precursor signal
- Can afford 1-2 hours compute time
- Want to maintain scientific rigor

**Choose Option B if**:
- Need quick validation
- Willing to accept alternative hypothesis
- Want to use existing data

**Choose Option C if**:
- Temporal ambiguity is acceptable
- Want to proceed with model building
- Value practical progress over perfect understanding

## Next Steps (Recommended: Option A)

1. **Scale up sample size**:
   ```bash
   uv run python src/temporal_investigation/phase1_glad_validation.py --n-samples 120
   ```

2. **Target quarterly distribution** (expected with 120 samples):
   - Q1: ~12-15 samples
   - Q2: ~18-24 samples
   - Q3: ~60-75 samples
   - Q4: ~12-15 samples

3. **If successful**:
   - Document precursor signal status (TRUE_PRECURSOR / EARLY_DETECTION / MIXED)
   - Proceed to WALK phase with clear understanding
   - Add monthly features in Phase 2 if needed

4. **If still inconclusive**:
   - Try 200+ samples OR
   - Pivot to Option C (proceed without definitive precursor validation)

## Technical Notes

### GLAD Dataset Quirks

- **Archived structure**: Year-specific ImageCollections (not single Images)
- **Band naming**: Year suffix (20, 21, 22, not 2020, 2021, 2022)
- **Date encoding**: Julian day (1-365/366), not days since epoch
- **Confidence**: Range 0-100, but even low values (3-5) can be valid
- **Coverage**: ~54% success rate for Amazon clearings (Landsat-based, 30m)

### AlphaEarth Temporal Aggregation

From AlphaEarth paper and our investigation:
- Annual embeddings: Aggregate 5-12 day observations over entire year
- Multi-sensor: Sentinel-2 (5-day), Landsat (16-day), Sentinel-1 (6-12 day)
- Cloud-penetrating: Radar helps ensure coverage in cloudy tropics
- **Temporal ambiguity**: Cannot distinguish late Y-1 precursor from early Y clearing

## Conclusion

Phase 1 successfully implemented GLAD validation infrastructure but yielded inconclusive results due to sample size. **Recommend Option A: Scale to 120+ samples** to get definitive answer on precursor signal before proceeding to WALK phase.

The temporal investigation question remains critical for interpreting model performance:
- **TRUE_PRECURSOR**: 3-9 months warning → High-value early warning system
- **EARLY_DETECTION**: 0-3 months warning → Still valuable but different use case
- **MIXED_SIGNAL**: Variable lead time → Need to characterize uncertainty

---

**Files created during Phase 1**:
- `src/temporal_investigation/phase1_glad_validation.py` - Main validation script
- `src/temporal_investigation/test_glad_access.py` - Dataset diagnostic tool
- `src/temporal_investigation/test_glad_archived.py` - Archive structure inspection
- `src/temporal_investigation/test_glad_sample.py` - Detailed sampling diagnostic
- `results/temporal_investigation/phase1_glad_validation.json` - Results (inconclusive)
