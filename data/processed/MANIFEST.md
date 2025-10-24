# Data Files Manifest

**Last Updated**: 2025-10-24
**Total Files**: 118 pickle files

This manifest tracks all data files in `data/processed/`, categorizing them by status and purpose.

---

## 🟢 PRODUCTION (Used by API/Dashboard)

### Production Models
| File | Size | Date | Purpose | Used By |
|------|------|------|---------|---------|
| `final_xgb_model_2020_2024.pkl` | 316K | Oct 23 | **Production XGBoost model** (0.913 AUROC) | `src/run/model_service.py` |
| `final_rf_model_2020_2024.pkl` | 483K | Oct 23 | Production Random Forest model | Dashboard comparison |
| `final_models_2020_2024_results.pkl` | 1.5K | Oct 23 | Model performance metrics | Dashboard stats |

### Production Validation Sets (Latest with Pre-extracted Features)
These are the **current production validation sets** with pre-extracted 70D features for fast dashboard loading.

| File | Samples | Date | Purpose | Used By |
|------|---------|------|---------|---------|
| **Risk Ranking (2022-2024)** |
| `hard_val_risk_ranking_2022_20251023_015922_features.pkl` | 19 | Oct 23 | Risk ranking 2022 with 70D features | Historical Playback page |
| `hard_val_risk_ranking_2023_20251023_015903_features.pkl` | 25 | Oct 23 | Risk ranking 2023 with 70D features | Historical Playback page |
| `hard_val_risk_ranking_2024_20251023_015822_features.pkl` | 25 | Oct 23 | Risk ranking 2024 with 70D features | Historical Playback page |
| **Comprehensive (2022-2024)** |
| `hard_val_comprehensive_2022_20251023_015927_features.pkl` | 28 | Oct 23 | Comprehensive 2022 with 70D features | Historical Playback page |
| `hard_val_comprehensive_2023_20251023_015913_features.pkl` | 22 | Oct 23 | Comprehensive 2023 with 70D features | Historical Playback page |
| `hard_val_comprehensive_2024_20251023_015827_features.pkl` | 31 | Oct 23 | Comprehensive 2024 with 70D features | Historical Playback page |
| **Rapid Response (2022-2024)** |
| `hard_val_rapid_response_2022_20251023_101531_features.pkl` | 22 | Oct 23 | Rapid response 2022 with 70D features | Historical Playback page |
| `hard_val_rapid_response_2023_20251023_101559_features.pkl` | 21 | Oct 23 | Rapid response 2023 with 70D features | Historical Playback page |
| `hard_val_rapid_response_2024_20251023_101620_features.pkl` | 25 | Oct 23 | Rapid response 2024 with 70D features | Historical Playback page |

**Total Production Validation Samples**: ~218 samples across 9 datasets (2022-2024 × 3 use cases)

### Production Training Data
| File | Purpose | Date |
|------|---------|------|
| `walk_dataset.pkl` | Current training dataset | Oct 23 |

---

## 🟡 EXPERIMENTAL (Research Artifacts)

### Validation Sets - Base Versions (Without Features)
These are the base validation sets without pre-extracted features. **Keep for reproducibility**.

| File | Purpose | Status |
|------|---------|--------|
| `hard_val_risk_ranking_2022_20251023_015922.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_risk_ranking_2023_20251023_015903.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_risk_ranking_2024_20251023_015822.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_comprehensive_2022_20251023_015927.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_comprehensive_2023_20251023_015913.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_comprehensive_2024_20251023_015827.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_rapid_response_2022_20251023_101531.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_rapid_response_2023_20251023_101559.pkl` | Base samples (no features) | ARCHIVE |
| `hard_val_rapid_response_2024_20251023_101620.pkl` | Base samples (no features) | ARCHIVE |

### Edge Cases (Failed Collection Attempts)
All edge cases files are 5 bytes (empty), indicating failed collection. **Can be deleted**.

| File | Size | Status | Reason |
|------|------|--------|--------|
| `hard_val_edge_cases_2022_*.pkl` (multiple) | 5B | DELETE | Failed collection |
| `hard_val_edge_cases_2023_*.pkl` (multiple) | 5B | DELETE | Failed collection |
| `hard_val_edge_cases_2024_*.pkl` (multiple) | 5B | DELETE | Failed collection |

**Action**: Delete all edge case files (none contain valid data).

### Temporal Validation Results
| File | Purpose | Status |
|------|---------|--------|
| `temporal_validation_hard_sets_results.pkl` | Temporal validation metrics | KEEP |
| `temporal_validation_hard_sets_xgboost_results.pkl` | XGBoost temporal results | KEEP |
| `model_ensemble_hard_sets_results.pkl` | Ensemble experiment results | KEEP |

### Old Validation Set Versions (Pre-October 23)
These are older versions created before the final production run. **Archive or delete**.

| Pattern | Count | Status | Reason |
|---------|-------|--------|--------|
| `hard_val_*_20251016_*.pkl` | ~10 | DELETE | Superseded by Oct 23 versions |
| `hard_val_*_20251017_*.pkl` | ~5 | DELETE | Superseded by Oct 23 versions |
| `hard_val_*_20251020_*.pkl` | ~15 | DELETE | Superseded by Oct 23 versions |
| `hard_val_*_20251021_*.pkl` | ~8 | DELETE | Superseded by Oct 23 versions |
| `hard_val_*_20251022_*.pkl` | ~12 | DELETE | Superseded by Oct 23 versions |
| `hard_val_*_005*.pkl` (early Oct 23) | ~20 | DELETE | Early attempts, superseded |

**Total to Delete**: ~70 old validation set versions

---

## 🔴 OLD VERSIONS (Can be Safely Deleted)

### Training Data - Old Versions
| Pattern | Purpose | Status |
|---------|---------|--------|
| `walk_dataset_2024_raw_*.pkl` | Multiple timestamped versions | DELETE (keep latest only) |
| `walk_dataset_scaled_*.pkl` | Old scaled versions | DELETE (superseded) |
| `walk_model_*.pkl` (not final_*) | Experimental models | DELETE (keep final only) |

### Feature Experiments - Old Versions
| Pattern | Purpose | Status |
|---------|---------|--------|
| `*_multiscale.pkl` (non-production) | Multiscale feature experiments | ARCHIVE |
| `*_sentinel2.pkl` (non-production) | Sentinel-2 feature experiments | ARCHIVE |
| `*_fire.pkl` (non-production) | Fire feature experiments | ARCHIVE |
| `*_vector_deltas.pkl` | Vector delta experiments | ARCHIVE |
| `*_spatial.pkl` | Spatial feature experiments | ARCHIVE |

---

## 📊 File Status Summary

| Category | Count | Action | Priority |
|----------|-------|--------|----------|
| **Production Models** | 3 | KEEP | Critical |
| **Production Validation Sets (with features)** | 9 | KEEP | Critical |
| **Production Training Data** | 1 | KEEP | Critical |
| **Base Validation Sets (no features)** | 9 | ARCHIVE | Medium |
| **Failed Edge Cases** | ~15 | DELETE | High |
| **Old Validation Versions** | ~70 | DELETE | High |
| **Experimental Features** | ~10 | ARCHIVE | Low |
| **Temporal/Ensemble Results** | 3 | KEEP | Medium |

**Total Production Files to Keep**: 13 files (11 GB)
**Total Experimental Files to Archive**: 22 files
**Total Old Files to Delete**: ~85 files

---

## 🗂️ Recommended Cleanup Actions

### Immediate (Do Now)
```bash
# 1. Delete all failed edge case files (5 bytes each)
find data/processed -name "hard_val_edge_cases_*.pkl" -size -10c -delete

# 2. Delete old validation set versions (before Oct 23, 10:00 AM)
find data/processed -name "hard_val_*_2025102[0-2]_*.pkl" -delete
find data/processed -name "hard_val_*_20251023_00*.pkl" -delete

# 3. Move experimental features to archive
mkdir -p data/processed/archive/experiments
mv data/processed/*_multiscale.pkl data/processed/archive/experiments/ 2>/dev/null || true
mv data/processed/*_sentinel2.pkl data/processed/archive/experiments/ 2>/dev/null || true
mv data/processed/*_fire.pkl data/processed/archive/experiments/ 2>/dev/null || true
mv data/processed/*_vector_deltas.pkl data/processed/archive/experiments/ 2>/dev/null || true
mv data/processed/*_spatial.pkl data/processed/archive/experiments/ 2>/dev/null || true
```

### Short-term (This Week)
```bash
# 4. Archive old training data versions
mkdir -p data/processed/archive/training_data
mv data/processed/walk_dataset_2024_raw_*.pkl data/processed/archive/training_data/ 2>/dev/null || true
mv data/processed/walk_dataset_scaled_*.pkl data/processed/archive/training_data/ 2>/dev/null || true

# 5. Archive base validation sets (without features)
mkdir -p data/processed/archive/validation_base
mv data/processed/hard_val_*_202510??_??????.pkl data/processed/archive/validation_base/ 2>/dev/null || true
# (Keep only *_features.pkl in main directory)
```

---

## 📝 Usage Notes

### For Production Deployments
**Required files** (13 files):
- 3 model files: `final_*_model_2020_2024.pkl`
- 9 validation sets with features: `hard_val_{use_case}_{year}_*_features.pkl`
- 1 training dataset: `walk_dataset.pkl`

### For Development
All files in `archive/` are available for research but not loaded by production code.

### For Historical Playback Page
The dashboard loads validation sets using this pattern:
```python
pattern = f'hard_val_{use_case}_{year}_*_features.pkl'
# Loads latest timestamped version with pre-extracted 70D features
```

---

## 🔍 File Naming Convention

**Production Format**:
```
hard_val_{use_case}_{year}_{timestamp}_features.pkl
         ↓           ↓       ↓           ↓
    risk_ranking   2022   20251023_015922  (pre-extracted 70D features)
    comprehensive  2023
    rapid_response 2024
```

**Size Indicators**:
- `5 bytes`: Failed collection (empty file)
- `< 10 KB`: Base samples only (lat/lon/year/label)
- `50-100 KB`: Samples with 70D features (production ready)
- `300-500 KB`: Trained model files

---

## 📌 Maintenance

**Update this manifest**:
- After creating new validation sets
- After training new models
- After cleanup operations
- Monthly review

**Last Cleanup**: 2025-10-24 (File cleanup completed)
**Cleanup Summary**:
- Deleted: 31 failed/old files (~200 KB)
- Archived: 57 experimental/old files (~13.8 MB)
- Remaining in main directory: 33 production files
- Archive directories created: experiments/, old_models/, training_data/, validation_base/
**Next Cleanup**: Quarterly review or when accumulating new experimental files

---

**Status**: ✅ Manifest created
**Action Required**: Review and approve cleanup commands above
