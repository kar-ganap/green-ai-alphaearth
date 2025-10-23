# Comprehensive Hard Validation Strategy

**Purpose**: Rigorously evaluate temporal generalization across ALL 4 phases on challenging use cases

**Date**: October 23, 2025

---

## Motivation

Current validation sets have critical issues:
1. **Temporal mismatch**: Collected from 2021 (within training range)
2. **Missing labels**: No ground truth labels
3. **Incompatible features**: Old quarterly structure vs current 70D format

**Solution**: Re-collect 2022, 2023, 2024 validation sets that match training methodology exactly.

---

## Temporal Validation Phases

Test model temporal generalization across all phases:

| Phase | Train Years | Test Year | Purpose |
|-------|-------------|-----------|---------|
| **Phase 1** | 2020-2021 | 2022 | 1-year forward prediction |
| **Phase 2** | 2020-2022 | 2023 | Cumulative learning |
| **Phase 3** | 2020-2021 + 2023 | 2024 | Non-consecutive years (gap year) |
| **Phase 4** | 2020-2023 | 2024 | Full recent history |

---

## Use Cases (4 Hard Validation Sets)

### 1. Risk Ranking
**Purpose**: High-priority regions requiring proactive monitoring

**Sampling Strategy**:
- Geographic focus: Amazon (Brazil), DRC, Indonesia
- Deforestation risk: High (recent loss nearby)
- Sample size: 30 per year (90 total: 30×2022, 30×2023, 30×2024)

**Expected Difficulty**: Moderate (clear deforestation in risky areas)

### 2. Rapid Response
**Purpose**: Recent/ongoing deforestation requiring immediate action

**Sampling Strategy**:
- Temporal focus: Last 3-6 months of each year
- Fresh deforestation: Detected in target year
- Sample size: 25 per year (75 total)

**Expected Difficulty**: Hard (small, recent, potentially incomplete clearings)

### 3. Comprehensive
**Purpose**: Diverse deforestation patterns across biomes

**Sampling Strategy**:
- Geographic diversity: All major tropical regions
- Biome diversity: Rainforest, dry forest, tropical woodland
- Deforestation type diversity: Small/large, fire/non-fire, edge/interior
- Sample size: 40 per year (120 total)

**Expected Difficulty**: Moderate-hard (wide variability)

### 4. Edge Cases
**Purpose**: Challenging scenarios at model limits

**Sampling Strategy**:
- Small clearings: <1 ha
- Low tree cover: 30-40% (threshold edge)
- Forest edges: Within 200m of existing clearing
- Gradual degradation: Fire-affected, selective logging
- Sample size: 20 per year (60 total)

**Expected Difficulty**: Very hard (model-breaking scenarios)

---

## Sampling Methodology (Match Training)

### Labeling Strategy: Hansen Global Forest Change

**Clearing (Label = 1)**:
- Hansen loss = 1 (forest loss detected)
- Hansen loss_year matches target year
- Example: For 2024 validation, loss_year = 24

**Intact (Label = 0)**:
- Hansen loss = 0 (no forest loss)
- Tree cover ≥ 30% (forest definition)
- Stable forest (no loss in any year)

### Tree Cover Thresholds (Heterogeneous - Match Training)

Distribution across samples to match training diversity:
- **50% threshold**: 60% of samples (standard clearings)
- **40% threshold**: 20% of samples (small clearings)
- **30% threshold**: 20% of samples (edge/fire clearings)

**Why heterogeneous?**: Training used this distribution to capture diverse forest types.

### Temporal Assignment

**Strict temporal matching**:
- 2022 validation: Hansen loss_year = 22
- 2023 validation: Hansen loss_year = 23
- 2024 validation: Hansen loss_year = 24

**No temporal overlap** with training years for respective phases.

---

## Data Leakage Prevention

### 1. Spatial Leakage

**Rule**: No validation sample within 1km of any training sample

**Implementation**:
```python
from src.walk.data_leakage_verification import verify_no_spatial_leakage

# Load all training samples
train_coords = get_all_training_locations()  # From 2020-2024 training data

# For each validation candidate
for candidate in validation_candidates:
    if has_spatial_overlap(candidate, train_coords, threshold_km=1.0):
        reject(candidate)
```

**Why 1km?**: Conservative buffer to prevent information leakage via spatial autocorrelation.

### 2. Temporal Leakage

**Rule**: Validation year must NOT be in training years for that phase

**Phase-specific rules**:
```
Phase 1: Train [2020, 2021] → Test [2022]
  - Reject any 2022 sample if location appeared in 2020-2021 training

Phase 2: Train [2020-2022] → Test [2023]
  - Reject any 2023 sample if location appeared in 2020-2022 training

Phase 3: Train [2020, 2021, 2023] → Test [2024]
  - Reject any 2024 sample if location appeared in 2020-2021-2023 training

Phase 4: Train [2020-2023] → Test [2024]
  - Reject any 2024 sample if location appeared in 2020-2023 training
```

### 3. Location Uniqueness

**Rule**: Each validation sample location appears in ONLY ONE year

**Implementation**:
- Track all sampled locations across years
- Reject repeat locations even if year differs
- Ensures independent samples across phases

---

## Collection Plan

### Sample Budget

| Use Case | Per Year | Total (3 years) |
|----------|----------|-----------------|
| Risk Ranking | 30 | 90 |
| Rapid Response | 25 | 75 |
| Comprehensive | 40 | 120 |
| Edge Cases | 20 | 60 |
| **Total** | **115** | **345** |

Balanced clearing/intact: ~50/50 split per use case

### Geographic Distribution

**Risk Ranking**:
- Brazil Amazon: 40%
- DRC: 30%
- Indonesia: 20%
- Other hotspots: 10%

**Rapid Response**:
- Follow GLAD alerts (last 3-6 months of each year)
- Global coverage weighted by deforestation activity

**Comprehensive**:
- Even distribution across tropical regions
- Diverse biomes (rainforest, dry forest, woodland)

**Edge Cases**:
- Targeted sampling of challenging scenarios
- Global but focused on known difficult regions

### Collection Strategy by Year

**2022 Samples (Phase 1 test)**:
- Collect from Hansen loss_year = 22
- Avoid training locations (2020-2021 samples)
- Extract features with 2022 AlphaEarth embeddings

**2023 Samples (Phase 2 test)**:
- Collect from Hansen loss_year = 23
- Avoid training locations (2020-2022 samples)
- Extract features with 2023 AlphaEarth embeddings

**2024 Samples (Phases 3-4 test)**:
- Collect from Hansen loss_year = 24
- Avoid training locations (2020-2023 samples)
- Extract features with 2024 AlphaEarth embeddings
- **Note**: Current 162 training samples from 2024 - must exclude these locations!

---

## Implementation Steps

### Step 1: Extract Training Locations (All Years)

```python
# Load all training datasets
datasets = [
    'walk_dataset_scaled_phase1_20251020_165345_all_hard_samples_multiscale.pkl',  # 2020-2023
    'walk_dataset_2024_with_features_20251021_110417.pkl'  # 2024
]

training_locations = {
    (sample['lat'], sample['lon'], sample['year'])
    for dataset in datasets
    for sample in load(dataset)
}

# Spatial index for efficient lookups
training_spatial = build_spatial_index(training_locations)
```

### Step 2: Collect Validation Samples

```python
for year in [2022, 2023, 2024]:
    for use_case in ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']:
        samples = collect_use_case_samples(
            year=year,
            use_case=use_case,
            n_samples=USE_CASE_SIZES[use_case],
            hansen_thresholds=[30, 40, 50],  # Heterogeneous
            excluded_locations=training_spatial,
            min_distance_km=1.0
        )

        # Verify no leakage
        assert verify_no_spatial_leakage(samples, training_locations, threshold_km=1.0)

        save(samples, f'hard_val_{use_case}_{year}.pkl')
```

### Step 3: Extract Features

```python
for year in [2022, 2023, 2024]:
    for use_case in USE_CASES:
        samples = load(f'hard_val_{use_case}_{year}.pkl')

        # Extract 70D features (annual + coarse multiscale)
        samples_with_features = extract_features(samples, year=year)

        save(samples_with_features, f'hard_val_{use_case}_{year}_features.pkl')
```

### Step 4: Temporal Validation (All Phases)

```python
for phase in [1, 2, 3, 4]:
    train_years, test_year = PHASE_CONFIGS[phase]

    # Train model on appropriate years
    model = train_model(years=train_years)

    # Evaluate on all use cases for test year
    for use_case in USE_CASES:
        samples = load(f'hard_val_{use_case}_{test_year}_features.pkl')

        metrics = evaluate_model(model, samples,
                                 threshold=OPTIMAL_THRESHOLDS[use_case])

        save_results(phase, use_case, test_year, metrics)
```

---

## Expected Results Format

### Per-Phase Performance

```json
{
  "phase_1": {
    "train_years": [2020, 2021],
    "test_year": 2022,
    "use_cases": {
      "risk_ranking": {
        "n_samples": 30,
        "roc_auc": 0.XXX,
        "threshold": 0.070,
        "precision": 0.XXX,
        "recall": 0.XXX,
        "target_met": true/false
      },
      "rapid_response": {...},
      "comprehensive": {...},
      "edge_cases": {...}
    }
  },
  "phase_2": {...},
  "phase_3": {...},
  "phase_4": {...}
}
```

### Drift Analysis

Track how performance degrades over time:

```
Phase 1 (2020-2021 → 2022): Minimal drift (1 year gap)
Phase 2 (2020-2022 → 2023): Moderate drift (1 year gap, more data)
Phase 3 (2020-2021+2023 → 2024): High drift (gap year + 3 year span)
Phase 4 (2020-2023 → 2024): Highest drift (4 year span)
```

**Key Question**: Does performance on hard cases degrade faster than on typical cases?

---

## Success Criteria

### Validation Integrity
- ✓ Zero spatial overlap with training (1km buffer)
- ✓ Zero temporal overlap with training years (per phase)
- ✓ Consistent labeling methodology (Hansen loss)
- ✓ Consistent sampling methodology (heterogeneous 30-50%)

### Performance Expectations

**Risk Ranking** (target: recall ≥ 0.90):
- Should perform well (clear deforestation in high-risk areas)
- Expect: 0.85-0.92 ROC-AUC across phases

**Rapid Response** (target: recall ≥ 0.90):
- Harder (recent, potentially incomplete)
- Expect: 0.75-0.85 ROC-AUC across phases

**Comprehensive** (target: precision baseline):
- Diverse scenarios
- Expect: 0.78-0.88 ROC-AUC across phases

**Edge Cases** (target: ROC-AUC ≥ 0.65):
- Hardest
- Expect: 0.60-0.75 ROC-AUC across phases

### Temporal Drift Patterns

Expected performance degradation:
- **Phase 1**: 0-5% drift (1 year forward)
- **Phase 2**: 3-8% drift (more data helps)
- **Phase 3**: 10-18% drift (gap year hurts)
- **Phase 4**: 12-20% drift (long span)

**Key Hypothesis**: Hard cases will show MORE drift than typical cases.

---

## Timeline

**Estimated Time**: 4-6 hours total

1. **Preparation** (30 min):
   - Extract training locations
   - Build spatial index
   - Define collection parameters

2. **Collection** (2-3 hours):
   - Collect 345 samples across 3 years × 4 use cases
   - GEE API rate limits may slow this down
   - Verify no leakage as we go

3. **Feature Extraction** (1-2 hours):
   - Extract 70D features for all samples
   - AlphaEarth API calls (64D embeddings)
   - Annual features (3D)
   - Coarse multiscale features (2D)

4. **Validation Runs** (1 hour):
   - Train 4 models (one per phase)
   - Evaluate on 4 use cases × 3-4 test years = 12-16 evaluations
   - Generate comprehensive report

---

## Output Files

### Data Files
```
data/processed/
  hard_val_risk_ranking_2022.pkl
  hard_val_risk_ranking_2023.pkl
  hard_val_risk_ranking_2024.pkl
  hard_val_risk_ranking_2022_features.pkl
  ...
  (24 files: 4 use cases × 3 years × 2 versions)
```

### Results Files
```
results/walk/
  temporal_validation_hard_cases_phase_1.json
  temporal_validation_hard_cases_phase_2.json
  temporal_validation_hard_cases_phase_3.json
  temporal_validation_hard_cases_phase_4.json
  temporal_validation_hard_cases_summary.json
```

### Documentation
```
docs/
  comprehensive_hard_validation_strategy.md  (this file)
  comprehensive_hard_validation_results.md   (after completion)
```

---

## Risk Assessment

### Potential Issues

1. **Insufficient samples**: Some use cases + years may have few candidates
   - **Mitigation**: Expand geographic search area
   - **Fallback**: Reduce sample sizes proportionally

2. **Feature extraction failures**: API errors, missing embeddings
   - **Mitigation**: Retry logic, batch processing
   - **Fallback**: Skip failed samples, document success rate

3. **Time constraints**: 4-6 hours may exceed available time
   - **Mitigation**: Parallelize collection (multiple use cases at once)
   - **Fallback**: Focus on Phase 4 (2024) only as proof-of-concept

### Contingency Plan

If full collection is infeasible:

**Minimum Viable Evaluation**:
- Collect Phase 4 (2024) only
- Focus on 2 use cases: Edge Cases + Rapid Response
- Sample sizes: 20 + 25 = 45 samples
- Time: ~1-2 hours
- Still demonstrates temporal generalization on hard cases

---

## Conclusion

This comprehensive hard validation strategy ensures:
1. **Scientific rigor**: No data leakage, consistent methodology
2. **Temporal honesty**: True out-of-sample evaluation across all phases
3. **Use-case coverage**: Performance assessment on diverse challenging scenarios
4. **Drift quantification**: Measure degradation on hard cases vs typical cases

**Recommendation**: Proceed with full collection (4-6 hours) for maximum demo impact and scientific credibility.

**Alternative**: If time-constrained, implement minimum viable evaluation (Phase 4 only, 1-2 hours) as proof-of-concept.

---

**Next Step**: Create collection script and begin execution.
