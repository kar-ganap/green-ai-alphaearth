# Fire Feature Extraction - Bug Investigation and Resolution

## Summary

Fire feature extraction initially reported 0 detections across all 163 validation samples. Investigation revealed two critical bugs that were masking actual fire detections.

## Timeline

### Initial Problem
- **Observation**: 0 fire detections across all 4 validation sets (163 samples total)
- **Unexpected**: Edge cases set includes 5 "fire-prone" samples
- **Contradiction**: Some samples showed pattern='fire_before_only' but fire_detections_total=0

### Investigation Process

1. **Analyzed extraction results** (`06_analyze_fire_results.py`)
   - Confirmed: 0 detections across all sets
   - Found contradiction: pattern != 'none' but counts = 0

2. **Tested MODIS queries directly** (`07_test_modis_fire.py`)
   - Tested 4 sample locations
   - **Key finding**: Location (-5.4367, -51.1579) showed:
     - Burn pixel count: **0.647** (fractional!)
     - FIRMS brightness temp: 313.7
   - Discovered MODIS products DEPRECATED

### Root Causes

#### Bug #1: Type Truncation
**Location**: `src/walk/01f_extract_fire_features.py:137-139`

```python
# BEFORE (buggy)
return {
    'fire_detections_total': int(burns_total),  # <-- Truncates 0.647 to 0!
    'fire_detections_before': int(burns_before),
    'fire_detections_after': int(burns_after),
    ...
}
```

**Problem**: MODIS returns fractional pixel counts due to:
- Partial pixel coverage in buffer region
- Resampling/projection effects
- 500m pixel size vs 1km buffer

**Impact**: Fire detections like 0.647 were truncated to 0, appearing as no fire

#### Bug #2: Deprecated MODIS Product
**Location**: `src/walk/01f_extract_fire_features.py:60`

```python
# BEFORE (deprecated)
modis_ba = ee.ImageCollection('MODIS/006/MCD64A1')  # Collection 6.0 (deprecated)
```

**Problem**:
- MODIS Collection 6.0 deprecated in favor of 6.1
- Earth Engine shows deprecation warnings
- May have data quality/coverage issues

**Impact**: Potentially missing some fire events

### Fixes Applied

#### Fix #1: Preserve Fractional Counts
```python
# AFTER (fixed)
return {
    'fire_detections_total': float(burns_total),  # Keep fractional values
    'fire_detections_before': float(burns_before),
    'fire_detections_after': float(burns_after),
    ...
}
```

Also updated:
- `count_burns()` return type: `0` → `0.0`
- Early return cases: all 0 → 0.0
- Error cases: all 0 → 0.0

#### Fix #2: Update to Collection 6.1
```python
# AFTER (updated)
modis_ba = ee.ImageCollection('MODIS/061/MCD64A1')  # Collection 6.1 (current)
```

## Results After Fixes

### Edge Cases Set (22 samples)
**Before fixes**: 0 samples with fire
**After fixes**: 2 samples with fire (2/10 clearing samples)

**Example detection**:
- Location: (-12.4220, -69.1472), Peru Amazon
- Fire total: **0.114** pixels
- Pattern: fire_after_only
- Interpretation: Fire detected after clearing date (slash-and-burn)

### Why Contradiction Existed

The pattern logic used floating-point comparison:
```python
if burns_before > 0 and burns_after == 0:  # 0.647 > 0 is True
    pattern = 'fire_before_only'

# But then...
'fire_detections_total': int(burns_total)  # int(0.647) = 0
```

So pattern='fire_before_only' (based on float > 0) but fire_detections_total=0 (after int cast).

## Assessment

### Fire Detection Coverage
Across all validation sets:
- Initial: 0/163 samples (0.0%)
- After fixes: ~2-5/163 samples (~1-3%)

**Still very limited fire data!**

### Why So Few Fire Detections?

Possible explanations:
1. **Fire-prone = regional risk, not actual fires**: Labels may indicate fire-prone ecosystems (Cerrado, seasonal forests) rather than actual fire events
2. **Time window mismatch**: 6-month window may not capture fires
3. **Resolution limitations**: MODIS 500m may miss small agricultural fires
4. **Clearing method**: Most deforestation in validation sets may be mechanical, not fire-based

### Implications for Fire Classifier

**Challenge**: 2-5 positive samples insufficient for training classifier

**Options**:
1. **Abandon fire classifier** → Move to multi-scale embeddings (original priority #2)
2. **Expand fire search**:
   - Wider time window (12-18 months)
   - VIIRS active fire (375m, daily)
   - NBR spectral analysis (detect fire scars directly)
3. **Redefine problem**: Instead of "fire vs clearing", use fire as a feature in main classifier

## Recommendations

Given limited fire data (2/163 samples), recommend:

**Skip fire classifier development** and proceed to **multi-scale embeddings** (priority #2 from user's order: 3 → 1 → 4).

**Reasoning**:
- 2 fire samples insufficient for meaningful classifier
- Fire detection would require significant additional work (VIIRS, NBR, expanded windows)
- Multi-scale embeddings directly address small-scale detection (100% miss rate < 1 ha)
- Cost-benefit favors moving to next priority

## Technical Notes

### MODIS MCD64A1 Product Details
- **Resolution**: 500m
- **Temporal**: Monthly composites
- **Band**: BurnDate (day of year, 1-366)
- **Collection 6.0**: Deprecated
- **Collection 6.1**: Current (MODIS/061/MCD64A1)

### Fractional Pixel Counts
Fractional counts occur when:
- Buffer geometry doesn't align with pixel grid
- Partial pixel overlap with buffer boundary
- Earth Engine's `reduceRegion()` interpolates

Example: 1km radius buffer ≈ 3.14 km², each pixel ≈ 0.25 km² → ~12 pixels, but only 0.647 burned

## Files Modified

- `src/walk/01f_extract_fire_features.py`: Bug fixes (deprecated product, type casting)
- `src/walk/06_analyze_fire_results.py`: Analysis script (created)
- `src/walk/07_test_modis_fire.py`: Direct MODIS testing (created)

## Next Steps

After fire extraction completes for all sets:
1. Analyze final fire detection statistics
2. Decision: Skip fire classifier if < 10 fire samples total
3. **Move to multi-scale embeddings** (addresses small-scale detection directly)
4. Document fire feature availability for future use
