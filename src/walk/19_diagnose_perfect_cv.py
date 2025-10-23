"""
Diagnose Perfect CV Score - Investigate 1.000 ROC-AUC

CONTEXT:
- Random Forest achieved 1.000 ROC-AUC on 5-fold stratified CV
- Validation sets show 0.583-0.907 ROC-AUC
- Large gap suggests overfitting or potential leakage

INVESTIGATION:
1. Per-fold CV scores (are all 1.000 or just mean?)
2. Train vs validation gap on each fold
3. Feature importance (any single feature dominating?)
4. Sample similarity (are training samples too similar?)
5. Feature leakage checks (temporal contamination?)

Usage:
    uv run python src/walk/19_diagnose_perfect_cv.py
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.utils import get_config


def combine_alphaearth_features(annual_data, multiscale_data):
    """Combine pre-extracted annual magnitudes with coarse AlphaEarth landscape features.

    (Copied from 11_train_random_forest.py to ensure exact same data loading)
    """
    X_annual = annual_data['X']
    y_annual = annual_data['y']
    annual_samples = annual_data['samples']
    multiscale_samples = multiscale_data['data']

    def get_sample_id(sample):
        return (sample['lat'], sample['lon'], sample['year'])

    annual_id_to_idx = {get_sample_id(s): i for i, s in enumerate(annual_samples)}
    multiscale_id_to_idx = {get_sample_id(s): i for i, s in enumerate(multiscale_samples)}

    common_ids = set(annual_id_to_idx.keys()) & set(multiscale_id_to_idx.keys())

    X_combined = []
    y_combined = []
    samples_combined = []
    coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

    for sample_id in common_ids:
        annual_idx = annual_id_to_idx[sample_id]
        multiscale_idx = multiscale_id_to_idx[sample_id]

        annual_features = X_annual[annual_idx]
        multiscale_sample = multiscale_samples[multiscale_idx]

        if 'multiscale_features' not in multiscale_sample:
            continue

        multiscale_dict = multiscale_sample['multiscale_features']
        missing_features = [k for k in coarse_feature_names if k not in multiscale_dict]
        if missing_features:
            continue

        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
        combined = np.concatenate([annual_features, coarse_features])

        if len(combined) != 69:
            continue

        X_combined.append(combined)
        y_combined.append(y_annual[annual_idx])
        samples_combined.append(annual_samples[annual_idx])

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    feature_names = ['delta_1yr', 'delta_2yr', 'acceleration'] + coarse_feature_names

    return X, y, samples_combined, feature_names


def load_training_data():
    """Load the same training data used by Random Forest."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load annual magnitude features (3D)
    annual_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'
    with open(annual_path, 'rb') as f:
        annual_data = pickle.load(f)

    # Load coarse landscape features (66D)
    multiscale_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale.pkl'
    with open(multiscale_path, 'rb') as f:
        multiscale_data = pickle.load(f)

    # Use exact same combination function as RF training
    X, y, samples, feature_names = combine_alphaearth_features(annual_data, multiscale_data)

    return X, y, samples, feature_names


def analyze_cv_folds(X, y, feature_names):
    """Analyze per-fold performance."""
    print("=" * 80)
    print("PER-FOLD CV ANALYSIS")
    print("=" * 80)

    # Use same CV strategy as training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_train_scores = []
    fold_val_scores = []
    fold_gaps = []

    # Best hyperparameters from training
    best_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_split': 10,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    print(f"\nTraining {skf.n_splits} folds with best hyperparameters...")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    print()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model on this fold
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train, y_train)

        # Predict probabilities
        y_train_pred = rf.predict_proba(X_train)[:, 1]
        y_val_pred = rf.predict_proba(X_val)[:, 1]

        # Calculate scores
        train_score = roc_auc_score(y_train, y_train_pred)
        val_score = roc_auc_score(y_val, y_val_pred)
        gap = train_score - val_score

        fold_train_scores.append(train_score)
        fold_val_scores.append(val_score)
        fold_gaps.append(gap)

        # Predictions
        y_val_pred_class = (y_val_pred >= 0.5).astype(int)
        cm = confusion_matrix(y_val, y_val_pred_class)

        print(f"Fold {fold + 1}:")
        print(f"  Train: {len(y_train)} samples (Clearing={np.sum(y_train)}, Intact={np.sum(1-y_train)})")
        print(f"  Val:   {len(y_val)} samples (Clearing={np.sum(y_val)}, Intact={np.sum(1-y_val)})")
        print(f"  Train ROC-AUC: {train_score:.4f}")
        print(f"  Val ROC-AUC:   {val_score:.4f}")
        print(f"  Gap:           {gap:.4f}")
        print(f"  Confusion Matrix (val):")
        print(f"    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
        print()

    print("=" * 80)
    print("FOLD SUMMARY")
    print("=" * 80)
    print(f"Train scores: {fold_train_scores}")
    print(f"Val scores:   {fold_val_scores}")
    print(f"Gaps:         {fold_gaps}")
    print()
    print(f"Mean train: {np.mean(fold_train_scores):.4f} ± {np.std(fold_train_scores):.4f}")
    print(f"Mean val:   {np.mean(fold_val_scores):.4f} ± {np.std(fold_val_scores):.4f}")
    print(f"Mean gap:   {np.mean(fold_gaps):.4f} ± {np.std(fold_gaps):.4f}")
    print()

    # Check if all folds are 1.000
    all_perfect = all(score >= 0.9999 for score in fold_train_scores)
    if all_perfect:
        print("🚨 WARNING: All folds achieved perfect (1.000) train score!")
        print("   This is highly suspicious and suggests:")
        print("   - Overfitting (model too complex for data)")
        print("   - Feature leakage (features contain target information)")
        print("   - Data contamination (samples too similar)")
    else:
        print("✓ Not all folds are perfect (some variation exists)")
        print("  This is less concerning, but gap is still large.")

    return fold_train_scores, fold_val_scores


def analyze_feature_importance(X, y, feature_names):
    """Check if any single feature dominates."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Train final model
    best_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_split': 10,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    rf = RandomForestClassifier(**best_params)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 20 most important features:")
    total_importance = 0
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        imp = importances[idx]
        total_importance += imp
        print(f"  {i+1:2d}. {feature_names[idx]:30s}  {imp:.4f}")

    print(f"\nTop 20 features account for {total_importance:.1%} of total importance")

    # Check for single feature dominance
    top_feature_imp = importances[indices[0]]
    if top_feature_imp > 0.5:
        print(f"\n🚨 WARNING: Single feature dominates ({top_feature_imp:.1%})!")
        print(f"   Feature: {feature_names[indices[0]]}")
        print("   This suggests potential feature leakage.")
    elif top_feature_imp > 0.3:
        print(f"\n⚠️  CAUTION: Top feature has high importance ({top_feature_imp:.1%})")
        print(f"   Feature: {feature_names[indices[0]]}")
        print("   Review this feature for potential leakage.")
    else:
        print(f"\n✓ No single feature dominates (top: {top_feature_imp:.1%})")

    return importances, indices


def analyze_sample_similarity(X, y, samples):
    """Check if training samples are too similar."""
    print("\n" + "=" * 80)
    print("SAMPLE SIMILARITY ANALYSIS")
    print("=" * 80)

    # Check spatial distribution
    coords = np.array([[s['lat'], s['lon']] for s in samples])

    print(f"\nSpatial distribution:")
    print(f"  Latitude range:  [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Longitude range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  Lat span: {coords[:, 0].max() - coords[:, 0].min():.2f} degrees")
    print(f"  Lon span: {coords[:, 1].max() - coords[:, 1].min():.2f} degrees")

    # Calculate pairwise distances (sample 100 random pairs)
    if len(samples) > 100:
        indices = np.random.choice(len(samples), 100, replace=False)
        sample_coords = coords[indices]
    else:
        sample_coords = coords

    from scipy.spatial.distance import pdist
    distances = pdist(sample_coords, metric='euclidean')

    print(f"\nPairwise spatial distances (degrees):")
    print(f"  Mean:   {distances.mean():.2f}")
    print(f"  Median: {np.median(distances):.2f}")
    print(f"  Min:    {distances.min():.2f}")
    print(f"  Max:    {distances.max():.2f}")

    # Check temporal distribution
    years = [s.get('year', s.get('clearing_year', 0)) for s in samples]
    unique_years = np.unique([y for y in years if y > 0])

    print(f"\nTemporal distribution:")
    print(f"  Years: {sorted(unique_years)}")
    for year in sorted(unique_years):
        count = sum(1 for y in years if y == year)
        print(f"    {year}: {count} samples")

    # Check feature space clustering
    print(f"\nFeature space statistics:")
    print(f"  Mean feature variance: {X.var(axis=0).mean():.4f}")
    print(f"  Max feature variance:  {X.var(axis=0).max():.4f}")
    print(f"  Min feature variance:  {X.var(axis=0).min():.4f}")

    # Check if any features are constant
    constant_features = np.sum(X.var(axis=0) < 1e-10)
    if constant_features > 0:
        print(f"\n⚠️  {constant_features} features are nearly constant!")
        print("   These provide no information and should be removed.")


def check_temporal_leakage(samples):
    """Check for potential temporal leakage."""
    print("\n" + "=" * 80)
    print("TEMPORAL LEAKAGE CHECK")
    print("=" * 80)

    issues = []

    for i, sample in enumerate(samples):
        # Check if embedding year is after clearing year
        clearing_year = sample.get('year', sample.get('clearing_year', 0))

        # Get embedding extraction info if available
        if 'embedding_year' in sample:
            emb_year = sample['embedding_year']
            if emb_year > clearing_year:
                issues.append({
                    'sample_idx': i,
                    'clearing_year': clearing_year,
                    'embedding_year': emb_year,
                    'sample_id': sample.get('id', 'unknown')
                })

    if issues:
        print(f"\n🚨 WARNING: Found {len(issues)} samples with potential temporal leakage!")
        print("\nFirst 10 violations:")
        for issue in issues[:10]:
            print(f"  Sample {issue['sample_idx']} (ID: {issue['sample_id']}):")
            print(f"    Clearing year:  {issue['clearing_year']}")
            print(f"    Embedding year: {issue['embedding_year']}")
        print("\n  This means embeddings were extracted AFTER clearing occurred,")
        print("  potentially leaking future information into features.")
    else:
        print("\n✓ No obvious temporal leakage detected")
        print("  (Note: This check requires 'embedding_year' field in samples)")

    # Check for Y-1 windowing compliance
    print("\n" + "-" * 80)
    print("Y-1 Windowing Check:")
    print("-" * 80)

    # Check if all samples have 'window' field
    has_window = any('window' in s for s in samples)
    if has_window:
        for i, sample in enumerate(samples[:5]):  # Show first 5
            window = sample.get('window', 'unknown')
            clearing = sample.get('year', 'unknown')
            print(f"  Sample {i}: Clearing={clearing}, Window={window}")
        print("  ...")
        print(f"  (Showing 5/{len(samples)} samples)")

        # Check if all use Y-1
        y_minus_1 = sum(1 for s in samples if str(s.get('window', '')).endswith('Y-1'))
        print(f"\n  Samples using Y-1 window: {y_minus_1}/{len(samples)}")
        if y_minus_1 == len(samples):
            print("  ✓ All samples use conservative Y-1 windowing")
        else:
            print("  ⚠️  Not all samples use Y-1 windowing!")
    else:
        print("  (No 'window' field in samples - cannot verify)")


def main():
    print("=" * 80)
    print("PERFECT CV SCORE DIAGNOSIS")
    print("=" * 80)
    print("\nLoading training data...")

    X, y, samples, feature_names = load_training_data()

    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Clearing: {np.sum(y)} samples ({100*np.sum(y)/len(y):.1f}%)")
    print(f"  Intact:   {len(y) - np.sum(y)} samples ({100*(1-np.sum(y)/len(y)):.1f}%)")

    # 1. Analyze CV folds
    fold_train, fold_val = analyze_cv_folds(X, y, feature_names)

    # 2. Analyze feature importance
    importances, indices = analyze_feature_importance(X, y, feature_names)

    # 3. Analyze sample similarity
    analyze_sample_similarity(X, y, samples)

    # 4. Check temporal leakage
    check_temporal_leakage(samples)

    # 5. Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    mean_train = np.mean(fold_train)
    mean_val = np.mean(fold_val)
    mean_gap = mean_train - mean_val

    print(f"\nKey Findings:")
    print(f"  Mean train score: {mean_train:.4f}")
    print(f"  Mean val score:   {mean_val:.4f}")
    print(f"  Mean gap:         {mean_gap:.4f}")

    if mean_train >= 0.9999:
        print(f"\n🚨 CRITICAL: Perfect train performance detected!")
        print("\nLikely causes (in order of probability):")
        print("  1. Model too complex for sample size (589 samples / 69 features)")
        print("     → Try: Increase min_samples_leaf, decrease max_depth")
        print("  2. Training samples are 'easy' cases, validation is 'hard'")
        print("     → Expected: Your validation sets are designed to be challenging")
        print("  3. Feature leakage (unlikely given Y-1 windowing)")
        print("     → Verify: Check embedding extraction timestamps")

        print("\nRecommended actions:")
        print("  1. Train simpler model (logistic regression) as sanity check")
        print("  2. Increase regularization (min_samples_leaf=10, max_depth=5)")
        print("  3. Try nested CV for unbiased performance estimate")
        print("  4. Collect more training samples if possible")

    elif mean_gap > 0.15:
        print(f"\n⚠️  MODERATE: Large train-validation gap detected!")
        print("\nLikely causes:")
        print("  1. Validation sets are intentionally hard (expected)")
        print("  2. Some overfitting due to model complexity")

        print("\nRecommended actions:")
        print("  1. Accept current performance if gap is due to hard validation")
        print("  2. Try moderate regularization increase")
        print("  3. Focus on improving edge_cases specifically")
    else:
        print(f"\n✓ Gap is reasonable for intentionally hard validation sets")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
