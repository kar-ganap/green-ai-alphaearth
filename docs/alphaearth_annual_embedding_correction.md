# AlphaEarth Annual Embedding Bug Fix

**Date**: 2025-10-18
**Severity**: Critical - Invalidated Phase 1 experiment
**Status**: Fixed

---

## Summary

Discovered that AlphaEarth provides **ONE embedding per year**, not quarterly embeddings. All previous "quarterly" feature extraction code was fundamentally buggy, creating redundant/zero features.

---

## The Bug

### What We Thought

We believed we were extracting quarterly embeddings:
```python
# Buggy code - thought we were getting different embeddings
for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
    date = f"{year}-{month}-01"
    emb = client.get_embedding(lat, lon, date)
    quarterly_embs.append(emb)
```

### What Actually Happened

AlphaEarth returns the **SAME embedding** for ANY date within the same year:

```python
emb_2020_03 = client.get_embedding(lat, lon, "2020-03-01")  # Returns emb_2020
emb_2020_06 = client.get_embedding(lat, lon, "2020-06-01")  # Returns emb_2020 (SAME!)
emb_2020_09 = client.get_embedding(lat, lon, "2020-09-01")  # Returns emb_2020 (SAME!)
emb_2020_12 = client.get_embedding(lat, lon, "2020-12-01")  # Returns emb_2020 (SAME!)
```

**All 4 "quarterly" embeddings are identical!**

---

## Impact on Features

### Buggy Features (17-dimensional)

**Baseline features (indices 0-9):**
```python
d12 = ||Q2 - Q1||  # ALWAYS 0! (same embedding)
d23 = ||Q3 - Q2||  # ALWAYS 0!
d34 = ||Q4 - Q3||  # ALWAYS 0!
# All velocities, accelerations → 0
# Total: 10 features, but all are 0 or constant!
```

**Delta features (indices 10-16):**
```python
delta_q1 = ||emb(Y) - emb(Y-1)||  # Only real feature
delta_q2 = ||emb(Y) - emb(Y-1)||  # SAME! (perfect duplicate)
delta_q3 = ||emb(Y) - emb(Y-1)||  # SAME!
delta_q4 = ||emb(Y) - emb(Y-1)||  # SAME!
delta_mean = mean(above 4)        # SAME! (just one value averaged)
delta_max = max(above 4)          # SAME!
delta_trend = 0                   # ALWAYS 0!
```

**Result**: 17 features, but only **ONE** contains unique information:
- `||emb(Y) - emb(Y-1)||` (the annual year-over-year change)

---

## Why Did Temporal Generalization Work (0.971 ROC-AUC)?

The temporal generalization experiment achieved 0.971 ROC-AUC **despite the bug** because:

1. The single discriminative feature (`||emb(Y) - emb(Y-1)||`) was replicated 6 times
2. Logistic regression handled the perfect collinearity gracefully
3. Year-over-year annual change is **actually very predictive** for standard clearings!

So we accidentally discovered that a single feature works great for temporal generalization.

---

## Why Did Phase 1 Fail (0.567 ROC-AUC)?

Phase 1 performed **worse than baseline** (0.567 vs 0.583) because:

1. All 17 features were redundant copies or zeros
2. The model had **zero degrees of freedom** - no real information
3. StandardScaler created numerical instabilities with perfect correlations
4. Edge cases require more nuanced features (spatial context, multi-year history)

---

## The Fix

### Corrected Annual Features (3-dimensional)

```python
def extract_annual_features(client, sample, year):
    """Use only annual embeddings - AlphaEarth's actual capability."""

    # Get 3 annual snapshots
    emb_y_minus_2 = client.get_embedding(lat, lon, f"{year-2}-06-01")
    emb_y_minus_1 = client.get_embedding(lat, lon, f"{year-1}-06-01")
    emb_y = client.get_embedding(lat, lon, f"{year}-06-01")

    # Compute annual deltas
    delta_1yr = ||emb_y - emb_y_minus_1||         # Recent change
    delta_2yr = ||emb_y_minus_1 - emb_y_minus_2||  # Historical change
    acceleration = delta_1yr - delta_2yr           # Is change accelerating?

    return [delta_1yr, delta_2yr, acceleration]
```

**Benefits:**
- Honest about AlphaEarth's capabilities
- No redundant features
- Clear interpretation
- Numerical stability

---

## Files Updated

### Created

1. `src/walk/annual_features.py` - Clean annual feature extraction module
   - `extract_annual_features()` - Simple 3D features
   - `extract_annual_features_extended()` - Extended 7D features with directional info

### Fixed

2. `src/walk/diagnostic_helpers.py` - Corrected from 17D to 3D
3. `src/walk/10_phase1_train_and_evaluate.py` - Updated FEATURE_NAMES

### Need to Fix

4. `src/walk/09_phase1_extract_features.py` - Re-extract with annual features
5. `src/walk/01_data_preparation.py` - Update for future datasets

---

## Evidence from Code

### EarthEngineClient Implementation

From `src/utils/earth_engine.py` lines 118-123:

```python
# Parse year from date
year = datetime.strptime(date, "%Y-%m-%d").year

# Get image for year (AlphaEarth is annual)
image_collection = ee.ImageCollection(collection)
image = image_collection.filterDate(f"{year}-01-01", f"{year}-12-31").filterBounds(point).first()
```

**The comment literally says "AlphaEarth is annual"!**

The implementation extracts the YEAR, then queries for ANY image between Jan 1 - Dec 31 of that year, returning the first (and only) annual embedding.

### Implementation Blueprint

From `docs/implementation_blueprint.md` line 1857:

```markdown
**AlphaEarth Embeddings:**
- Source: Google Earth Engine
- Collection: `'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'`  <-- ANNUAL!
- Resolution: 10m × 10m
- Dimensions: 64
- Coverage: 2017-2024
```

**The collection ID is literally `ANNUAL`!**

---

## Lessons Learned

1. **Read the documentation** - AlphaEarth collection name included "ANNUAL"
2. **Test assumptions** - Should have verified quarterly embeddings were different
3. **Inspect features** - Perfect correlation (1.0) should have been a red flag
4. **Simple is better** - Annual features match the data source capability

---

## Next Steps

1. ✅ Create corrected annual feature extraction (`annual_features.py`)
2. ✅ Update diagnostic helpers (`diagnostic_helpers.py`)
3. ✅ Fix feature names in training script
4. ⏳ Re-extract Phase 1 features with annual approach
5. ⏳ Re-run Phase 1 training and evaluation
6. ⏳ Update all documentation to reflect annual limitation
7. ⏳ Add tests to prevent similar bugs

---

## Expected Performance

With corrected annual features:

**Hypothesis**: Performance should match or exceed temporal generalization (0.971 ROC-AUC) for standard clearings, but may still struggle with edge cases due to lack of spatial/contextual features.

**Recommendation**: After re-running with corrected features, if edge cases still underperform, proceed with Phase 2 specialization using:
- Annual temporal features (corrected)
- Spatial neighborhood features
- Contextual features (roads, fire history)

---

## Conclusion

This bug taught us an important lesson: AlphaEarth's strength is annual land cover change detection, not sub-annual monitoring. Our features should reflect this reality.

The good news: temporal generalization (0.971 ROC-AUC) shows that annual features work extremely well for standard patterns. We now need to add spatial/contextual features to handle edge cases.
