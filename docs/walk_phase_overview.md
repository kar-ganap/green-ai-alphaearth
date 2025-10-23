# WALK Phase Overview

**Status**: In Progress
**Branch**: `walk-phase`
**Goal**: Build methodologically rigorous ML system with proper validation before final model training

---

## Objectives

The WALK phase focuses on establishing a **solid foundation** for model development:

1. **Data Quality**: Spatial CV splits, temporal validation, label filtering
2. **Baseline Benchmarks**: Establish performance targets
3. **Systematic Feature Engineering**: Add features methodically, keep only what improves performance
4. **Validation Protocol**: Lock down evaluation methodology

---

## Phase Structure

### 1. Data Preparation (✅ Complete)

**Script**: `src/walk/01_data_preparation.py`

**What it does**:
- Loads clearing samples from GLAD labels (multiple years: 2020-2023)
- Loads intact forest samples (stable, no recent clearing)
- Extracts quarterly embeddings for each location:
  - Q1 (Y-1, Jun): 9-12 months before clearing
  - Q2 (Y, Mar): 6-9 months before
  - Q3 (Y, Jun): 3-6 months before
  - Q4 (Y, Sep): 0-3 months before (precursor period)
  - Clearing (Y+1, Jun): During/after clearing
- Computes temporal features:
  - **Distances**: L2 distance from Q1 baseline
  - **Velocities**: Change rate between quarters
  - **Accelerations**: Velocity changes
  - **Trend consistency**: Are distances monotonically increasing?
- Creates spatial cross-validation splits:
  - 10km buffer between train/val/test
  - Prevents spatial leakage (nearby pixels stay together)
- Saves to: `data/processed/walk_dataset.pkl`

**Usage**:
```bash
uv run python src/walk/01_data_preparation.py --n-clearing 100 --n-intact 100
```

**Output**:
- Dataset with features and spatial splits
- Ready for baseline testing

---

### 2. Baseline Suite (✅ Complete)

**Script**: `src/walk/02_baseline_suite.py`

**What it tests**:

**Baseline 1: Random**
- Random predictions (should be ~0.5 AUC)
- Sanity check that dataset is reasonable

**Baseline 2: Raw Embeddings**
- Uses only Q4 distance from Q1 baseline
- Simple threshold: higher distance = more likely clearing
- Tests if raw signal has predictive power

**Baseline 3: Simple Features**
- Uses distances + velocities
- Logistic regression
- Tests if basic feature engineering helps

**Baseline 4: All Features**
- Uses all temporal features (distances, velocities, accelerations, trend)
- Random Forest
- Best baseline before sophisticated engineering

**Metrics**:
- ROC-AUC: Overall discrimination ability
- PR-AUC: Precision-recall (better for imbalanced data)
- Precision, Recall, Specificity
- Confusion matrix

**Usage**:
```bash
uv run python src/walk/02_baseline_suite.py
```

**Output**:
- `results/walk/baseline_results.json`: Detailed metrics
- `results/figures/walk/baseline_comparison.png`: Visualization

---

### 3. Spatial Feature Engineering (⏳ Pending)

**Goal**: Extract neighborhood and spatial context features

**Features to implement**:
- **Neighbor distances**: Average embedding distance of nearby pixels
- **Local variance**: How heterogeneous is the neighborhood?
- **Spatial autocorrelation**: Do nearby pixels have similar trajectories?
- **Edge proximity**: Distance to clearing boundaries
- **Neighbor clearing status**: How many neighbors are clearing?

**Implementation approach**:
1. For each pixel, identify neighbors within 1km
2. Compute neighbor statistics
3. Test if these improve model performance (ΔAUC > 0.01)

---

### 4. Q4-Specific Features (⏳ Pending)

**Goal**: Test if additional features amplify weak Q4 precursor signal

From CRAWL findings, we know:
- Q4 has weak but detectable signal (d=0.81, p~0.02-0.05)
- Effect size 2-7x weaker than Q2-Q3 concurrent detection
- ~40% overlap with intact distribution

**Features to test**:
- **Context amplification**: Does road proximity + Q4 distance improve signal?
- **Neighbor context**: Do Q4 pixels near other changing pixels show stronger signal?
- **Trajectory shape**: Does Q4 trajectory curvature help?

**Decision criteria**:
- Keep Q4 features if ΔAUC > 0.02
- Otherwise, confirm detection-only framing

---

### 5. Validation Protocol (⏳ Pending)

**Goal**: Lock down evaluation methodology before final runs

**Components**:

**Metrics Suite**:
- Primary: ROC-AUC, PR-AUC
- Secondary: Calibration curves, precision@recall thresholds
- Breakdown: Q2-Q3 (detection) vs Q4 (prediction) performance

**Leakage Checks**:
- Verify no future information in training
- Confirm spatial buffer enforced (no neighbors in different splits)
- Check temporal ordering

**Reproducibility**:
- Fixed random seeds (42)
- Version-locked dependencies (uv.lock)
- Cached embeddings for consistency

**Error Analysis**:
- Where does model fail?
- False positives: What intact pixels look like clearings?
- False negatives: What clearings look like intact?
- Q4 vs Q2-Q3 performance comparison

---

## Progress Tracking

### Completed ✅
- [x] Data preparation pipeline
- [x] Spatial cross-validation splits (10km buffer)
- [x] Temporal feature extraction (distances, velocities, accelerations)
- [x] Baseline suite implementation (random, raw, simple, all)

### In Progress ⏳
- [ ] Prepare full dataset (100+ samples)
- [ ] Run baseline suite on full data

### Pending 📋
- [ ] Spatial feature engineering
- [ ] Q4-specific feature testing
- [ ] Validation protocol implementation
- [ ] Final model training
- [ ] Error analysis and model interpretation

---

## Key Decisions from CRAWL

The WALK phase builds on CRAWL findings:

**Primary Framing**: Detection System (0-3 month lag)
- Target: Q2-Q3 clearings (26% of GLAD subset)
- Expected: Strong performance (d=2-6 effect sizes)
- Claim: "Reliable concurrent detection"

**Secondary Exploration**: Q4 Precursor Potential
- Finding: Weak but statistically detectable (p=0.02-0.05)
- Effect size: d=0.81 (vs d=2-6 for Q2-Q3)
- Test: Can context features improve weak Q4 signal?
- Decision: Keep Q4 features only if ΔAUC > 0.02

---

## Expected Deliverables

At end of WALK phase:

1. ✅ **Validated dataset**: Spatial CV splits, cleaned labels, no leakage
2. ⏳ **Baseline benchmarks**: Performance targets to beat
3. 📋 **Engineered feature set**: Only features that improve performance
4. 📋 **Evaluation protocol**: Locked-down metrics, reproducible pipeline
5. 📋 **Performance breakdown**: Q2-Q3 vs Q4 results
6. 📋 **Decision point**: Confirm detection framing or revise if Q4 improves

---

## Timeline Estimate

**Completed**: ~4 hours (data prep + baseline suite)
**Remaining**: ~8-13 hours
- Full dataset preparation: 1-2 hours
- Baseline evaluation: 1 hour
- Spatial features: 3-4 hours
- Q4 feature testing: 2-3 hours
- Validation protocol: 2-3 hours
- Error analysis: 1-2 hours

**Total WALK**: 12-17 hours (as planned)

---

## Technical Stack

- **Data**: Google Earth Engine (AlphaEarth, Hansen GFC)
- **Features**: NumPy, SciPy
- **Models**: scikit-learn (LogisticRegression, RandomForest)
- **Validation**: Custom spatial CV with cKDTree
- **Visualization**: Matplotlib, Seaborn

---

## Files

**Scripts**:
- `src/walk/01_data_preparation.py`: Load data, extract features, create splits
- `src/walk/02_baseline_suite.py`: Run baseline benchmarks

**Data**:
- `data/processed/walk_dataset.pkl`: Prepared dataset with features and splits

**Results**:
- `results/walk/baseline_results.json`: Baseline metrics
- `results/figures/walk/baseline_comparison.png`: Baseline visualization

**Documentation**:
- `docs/walk_phase_overview.md`: This file
- `docs/implementation_blueprint.md`: Overall project plan
- `docs/extended_crawl_findings.md`: CRAWL phase conclusions

---

## Next Steps

1. ✅ Wait for full dataset preparation to complete
2. Run baseline suite on full data
3. Analyze baseline results
4. Implement spatial feature extraction
5. Test Q4-specific features
6. Lock down validation protocol
7. Train final model
8. Error analysis and interpretation

---

## Notes

**Data Caching**: All Earth Engine queries are cached in `data/cache/`. Re-running scripts with same parameters will use cached results for speed.

**Branch Strategy**: WALK development on `walk-phase` branch. Will merge to `main` when complete.

**Commit Strategy**: Incremental commits as features are added and tested.
