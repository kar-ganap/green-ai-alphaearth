# CRAWL Phase Tests

Critical assumption validation tests. **All tests must pass before proceeding to WALK phase.**

## Purpose

The CRAWL phase validates fundamental assumptions about AlphaEarth embeddings and deforestation predictability. These are GO/NO-GO decision gates - if any test fails, we either stop or pivot the approach.

## Tests

### Test 1: Separability
**Question:** Can AlphaEarth embeddings distinguish cleared vs intact forest?

**Method:**
- Get 50 cleared + 50 intact forest locations
- Fetch AlphaEarth embeddings for same date (2023-06-01)
- Test classification with linear SVM + 5-fold CV

**Decision Gate:** Accuracy ≥ 85%

**Run:**
```bash
python src/crawl/test_1_separability.py
```

**Options:**
```bash
# Use more samples for higher confidence
python src/crawl/test_1_separability.py --n-samples 100

# Don't save results
python src/crawl/test_1_separability.py --no-save
```

**Expected Time:** 5-10 minutes (depends on Earth Engine API speed)

**If PASS:** Proceed to Test 2
**If FAIL:** Stop - embeddings don't contain forest cover information

---

### Test 2: Temporal Signal
**Question:** Do embeddings change *before* clearing (not just at/after)?

**Method:**
- Get 20 locations with known clearing dates
- Fetch embedding time series: -6m, -3m, -1m, 0m, +3m
- Test if distance from baseline increases before clearing
- Statistical test: p < 0.05 at -3 months

**Decision Gate:** p-value < 0.05

**Run:**
```bash
python src/crawl/test_2_temporal.py
```

**If PASS:** Proceed to Test 3
**If FAIL:** Pivot to detection (not prediction)

---

### Test 3: Generalization
**Question:** Does the signal work across different regions?

**Method:**
- Test signal in 3 different Amazon regions (north, south, east)
- Calculate coefficient of variation across regions
- High CV = signal not consistent = may need region-specific models

**Decision Gate:** CV < 0.5

**Run:**
```bash
python src/crawl/test_3_generalization.py
```

**If PASS:** Proceed to Test 4
**If WARNING:** Can proceed but may need regional models

---

### Test 4: Minimal Model
**Question:** Can the simplest possible features predict anything?

**Method:**
- Use ONLY 2 features: velocity + distance_to_road
- Train simple logistic regression
- Test with 5-fold CV

**Decision Gate:** AUC ≥ 0.65

**Run:**
```bash
python src/crawl/test_4_minimal_model.py
```

**If PASS:** PROCEED TO WALK PHASE
**If FAIL:** Stop - problem not solvable with this approach

---

## Running All Tests

```bash
# Run all CRAWL tests in sequence
./scripts/run_crawl_tests.sh

# Or manually:
python src/crawl/test_1_separability.py && \
python src/crawl/test_2_temporal.py && \
python src/crawl/test_3_generalization.py && \
python src/crawl/test_4_minimal_model.py
```

## Decision Matrix

| Test | Pass | Fail | Action |
|------|------|------|--------|
| 1. Separability | ✓ → Test 2 | ✗ → STOP | Try different embeddings or abandon |
| 2. Temporal | ✓ → Test 3 | ✗ → PIVOT | Switch to detection (not prediction) |
| 3. Generalization | ✓ → Test 4 | ⚠ → Proceed with caution | May need regional models |
| 4. Minimal Model | ✓ → WALK | ✗ → STOP | Problem not solvable |

## Results

All test results are saved to:
- `results/experiments/crawl_test_N_results.json` - Detailed metrics
- `results/figures/crawl/test_N_*.png` - Visualizations

## Expected Outcomes

Based on the blueprint, we expect:

- **Test 1:** >90% accuracy (AlphaEarth should handle this easily)
- **Test 2:** p < 0.01, distance ratio ~0.3-0.5
- **Test 3:** CV < 0.3 (consistent signal)
- **Test 4:** AUC 0.72-0.78 (strong signal with just 2 features)

## What If Tests Fail?

### Test 1 Fails
AlphaEarth embeddings don't contain forest cover information. Options:
1. Try different embedding model
2. Try raw satellite bands instead of embeddings
3. Abandon approach

### Test 2 Fails
No precursor signal - can't predict in advance. Options:
1. Pivot to real-time detection instead of prediction
2. Try shorter time horizons (7-30 days instead of 90)
3. Focus on post-hoc analysis only

### Test 3 Shows High Variation
Signal not consistent across regions. Options:
1. Train region-specific models
2. Add region as a feature
3. Focus on single region for MVP

### Test 4 Fails
Even simple features don't work. Options:
1. Problem may not be solvable with current data
2. Try different problem framing (e.g., identify high-risk areas, not specific clearings)
3. Abandon predictive modeling, focus on descriptive analysis

## Philosophy

> "Fail fast in CRAWL, build right in WALK, impress in RUN"

The CRAWL phase is designed to waste minimal time on fundamentally flawed approaches.
Better to discover unsolvable problems in 4-6 hours than after 40 hours of work.
