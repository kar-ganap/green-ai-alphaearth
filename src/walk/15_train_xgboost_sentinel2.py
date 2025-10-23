"""
Train XGBoost with AlphaEarth + Sentinel-2 Features

Combines 10m Sentinel-2 features with 30m AlphaEarth embeddings to test if
high-resolution satellite imagery can break through the 0.583 performance ceiling:

AlphaEarth (69D):
- 3D annual magnitudes
- 66D coarse landscape

Sentinel-2 (up to 46D):
- Spectral indices: NDVI, NBR, NDMI (12 features)
- GLCM textures: contrast, correlation, entropy, homogeneity, ASM (20+ features)

Total: Up to 115D features

Strategy:
1. Load samples with both AlphaEarth + Sentinel-2 features
2. Handle missing S2 features (138/600 samples failed extraction)
3. Hyperparameter tuning via GridSearchCV with StratifiedKFold
4. L2/L1 regularization to prevent overfitting
5. Compare against 69D XGBoost baseline (0.583 on edge_cases)

Usage:
    uv run python src/walk/15_train_xgboost_sentinel2.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def combine_features_with_sentinel2(annual_data, sentinel2_data):
    """
    Combine annual AlphaEarth features, coarse landscape features, and Sentinel-2 features.

    Returns:
        X: Feature matrix (samples × features)
        y: Labels
        feature_names: List of feature names
        stats: Dict with extraction statistics
    """
    X_annual = annual_data['X']
    y_annual = annual_data['y']
    annual_samples = annual_data['samples']
    sentinel2_samples = sentinel2_data['data']

    def get_sample_id(sample):
        return (sample['lat'], sample['lon'], sample['year'])

    annual_id_to_idx = {get_sample_id(s): i for i, s in enumerate(annual_samples)}
    sentinel2_id_to_idx = {get_sample_id(s): i for i, s in enumerate(sentinel2_samples)}

    common_ids = set(annual_id_to_idx.keys()) & set(sentinel2_id_to_idx.keys())

    X_combined = []
    y_combined = []

    # Feature names
    coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

    # Collect all S2 feature names from first sample with S2 data
    s2_feature_names = None
    for sample in sentinel2_samples:
        if 'multiscale_features' in sample:
            s2_keys = [k for k in sample['multiscale_features'].keys() if k.startswith('s2_')]
            if len(s2_keys) > 0:
                s2_feature_names = sorted(s2_keys)
                break

    if s2_feature_names is None:
        raise ValueError("No Sentinel-2 features found in dataset!")

    stats = {
        'total_samples': len(common_ids),
        'with_s2': 0,
        'without_s2': 0,
        'with_incomplete_s2': 0,
        'used': 0
    }

    for sample_id in common_ids:
        annual_idx = annual_id_to_idx[sample_id]
        sentinel2_idx = sentinel2_id_to_idx[sample_id]

        annual_features = X_annual[annual_idx]
        sentinel2_sample = sentinel2_samples[sentinel2_idx]

        # Check for multiscale features
        if 'multiscale_features' not in sentinel2_sample:
            stats['without_s2'] += 1
            continue

        multiscale_dict = sentinel2_sample['multiscale_features']

        # Check for AlphaEarth coarse features
        missing_coarse = [k for k in coarse_feature_names if k not in multiscale_dict]
        if missing_coarse:
            continue

        # Extract coarse features
        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])

        # Check for Sentinel-2 features
        s2_keys_present = [k for k in s2_feature_names if k in multiscale_dict]

        if len(s2_keys_present) == 0:
            # No S2 features at all
            stats['without_s2'] += 1
            continue
        elif len(s2_keys_present) < len(s2_feature_names):
            # Incomplete S2 features - skip for consistency
            stats['with_incomplete_s2'] += 1
            continue

        # Extract S2 features
        s2_features = np.array([multiscale_dict[k] for k in s2_feature_names])

        # Combine all features: annual (3D) + coarse (66D) + S2 (variable)
        combined = np.concatenate([annual_features, coarse_features, s2_features])

        X_combined.append(combined)
        y_combined.append(y_annual[annual_idx])
        stats['with_s2'] += 1

    stats['used'] = len(X_combined)

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    feature_names = (
        ['delta_1yr', 'delta_2yr', 'acceleration'] +
        coarse_feature_names +
        s2_feature_names
    )

    return X, y, feature_names, stats


def main():
    print("=" * 80)
    print("XGBOOST WITH ALPHAEARTH + SENTINEL-2 FEATURES")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    results_dir = Path('results/walk')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("\nLoading features...")
    annual_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'
    with open(annual_path, 'rb') as f:
        annual_data = pickle.load(f)

    sentinel2_path = processed_dir / 'walk_dataset_scaled_phase1_sentinel2.pkl'
    with open(sentinel2_path, 'rb') as f:
        sentinel2_data = pickle.load(f)

    X_train, y_train, feature_names, stats = combine_features_with_sentinel2(annual_data, sentinel2_data)

    print(f"\n✓ Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Clearing: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Intact: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  Samples/feature ratio: {len(X_train)/X_train.shape[1]:.1f}")

    print(f"\n  Feature extraction stats:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    With S2 features: {stats['with_s2']} ({stats['with_s2']/stats['total_samples']*100:.1f}%)")
    print(f"    Without S2: {stats['without_s2']}")
    print(f"    Incomplete S2: {stats['with_incomplete_s2']}")
    print(f"    Used for training: {stats['used']}")

    # Count feature types
    alphaearth_count = 69  # 3 annual + 66 coarse
    s2_count = X_train.shape[1] - alphaearth_count
    print(f"\n  Feature breakdown:")
    print(f"    AlphaEarth: {alphaearth_count}D (3 annual + 66 coarse)")
    print(f"    Sentinel-2: {s2_count}D")
    print(f"    Total: {X_train.shape[1]}D")

    # Scale features
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING VIA 5-FOLD STRATIFIED CV")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Same grid as 69D experiment for fair comparison
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'reg_lambda': [0.0, 1.0, 5.0],           # L2 regularization
        'reg_alpha': [0.0, 0.5],                  # L1 regularization
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
    }

    print(f"\nSearching {np.prod([len(v) for v in param_grid.values()])} hyperparameter combinations...")
    print(f"Using StratifiedKFold with 5 folds (preserves 50/50 class balance)")

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    xgb = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Check for overfitting
    best_idx = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_idx]
    val_score = grid_search.cv_results_['mean_test_score'][best_idx]

    print(f"\nOverfitting check:")
    print(f"  Mean train score: {train_score:.3f}")
    print(f"  Mean val score:   {val_score:.3f}")
    print(f"  Gap:              {train_score - val_score:.3f}")

    if train_score - val_score > 0.1:
        print(f"  ⚠ Warning: Potential overfitting detected (gap > 0.1)")
    else:
        print(f"  ✓ Good generalization (gap < 0.1)")

    # Train final model
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL MODEL ON {len(X_train)} SAMPLES")
    print(f"{'='*80}\n")

    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X_train_scaled, y_train)
    print("✓ Final model trained")

    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}\n")

    feature_importance = best_xgb.feature_importances_
    importance_idx = np.argsort(feature_importance)[::-1]

    print("Top 30 most important features:")
    for i, idx in enumerate(importance_idx[:30], 1):
        feat_name = feature_names[idx]
        is_s2 = '(S2)' if feat_name.startswith('s2_') else '(AE)'
        print(f"{i:3d}. {feat_name:35s} {is_s2:5s} {feature_importance[idx]:.4f}")

    # Count S2 features in top 20
    top20_s2 = sum(1 for idx in importance_idx[:20] if feature_names[idx].startswith('s2_'))
    top20_ae = 20 - top20_s2
    print(f"\nTop 20 feature breakdown:")
    print(f"  Sentinel-2: {top20_s2}")
    print(f"  AlphaEarth: {top20_ae}")

    # Load validation sets (we'll need to check if they have S2 features extracted)
    print(f"\n{'='*80}")
    print("EVALUATING ON EDGE_CASES VALIDATION SET")
    print("=" * 80)
    print("\nNote: Other validation sets need S2 features extracted first")
    print("Run: uv run python src/walk/14_extract_sentinel2_features.py --dataset validation --set all")

    ee_client = EarthEngineClient(use_cache=True)

    # For now, just evaluate on edge_cases with AlphaEarth-only features
    multiscale_val_path = processed_dir / 'hard_val_edge_cases_multiscale.pkl'

    if not multiscale_val_path.exists():
        print("\n✗ Validation set not found")
        return

    print(f"\nLoading edge_cases (AlphaEarth-only, no S2)...")

    with open(multiscale_val_path, 'rb') as f:
        val_samples = pickle.load(f)

    X_val = []
    y_val = []

    for sample in val_samples:
        # Fix missing 'year' field
        if 'year' not in sample and sample.get('stable', False):
            sample = sample.copy()
            sample['year'] = 2021

        # Extract annual features
        try:
            from src.walk.diagnostic_helpers import extract_dual_year_features
            annual_features = extract_dual_year_features(ee_client, sample)
        except:
            annual_features = None

        if annual_features is None:
            continue

        # Get coarse features
        if 'multiscale_features' not in sample:
            continue

        multiscale_dict = sample['multiscale_features']
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

        missing_features = [k for k in coarse_feature_names if k not in multiscale_dict]
        if missing_features:
            continue

        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])

        # For validation, we only have AlphaEarth features (69D)
        # Pad with zeros for S2 features to match training dimensions
        s2_feature_count = X_train.shape[1] - 69
        s2_placeholder = np.zeros(s2_feature_count)

        combined = np.concatenate([annual_features, coarse_features, s2_placeholder])

        if len(combined) != X_train.shape[1]:
            continue

        X_val.append(combined)
        y_val.append(sample.get('label', 0))

    if len(X_val) == 0:
        print("  ⚠ No valid samples")
        return

    X_val = np.vstack(X_val)
    y_val = np.array(y_val)

    print(f"  ✓ {len(X_val)} samples (using AlphaEarth features only)")

    # Evaluate
    X_val_scaled = scaler.transform(X_val)
    y_pred_proba = best_xgb.predict_proba(X_val_scaled)[:, 1]
    y_pred = best_xgb.predict(X_val_scaled)

    try:
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except ValueError:
        roc_auc = float('nan')

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    baseline_69d = 0.583  # XGBoost 69D baseline

    print(f"\n{'='*80}")
    print("VALIDATION RESULTS (edge_cases)")
    print(f"{'='*80}\n")

    if not np.isnan(roc_auc):
        diff = roc_auc - baseline_69d
        pct_change = (diff / baseline_69d * 100)
        print(f"  ROC-AUC:   {roc_auc:.3f}  (69D baseline: {baseline_69d:.3f}, {diff:+.3f} / {pct_change:+.1f}%)")
    else:
        print(f"  ROC-AUC:   nan")

    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN:  {cm[0, 0]:2d}  FP:  {cm[0, 1]:2d}")
    print(f"    FN:  {cm[1, 0]:2d}  TP:  {cm[1, 1]:2d}")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA")
    print(f"{'='*80}\n")

    target = 0.70

    print(f"Target: edge_cases ROC-AUC ≥ {target:.2f}")
    print(f"Baseline (XGBoost 69D): {baseline_69d:.3f}")
    print(f"Current (XGBoost + S2, val without S2): {roc_auc:.3f}")

    print(f"\n⚠ NOTE: Validation set evaluated WITHOUT Sentinel-2 features!")
    print(f"  This is testing if the model trained with S2 generalizes to AlphaEarth-only data")
    print(f"  For fair comparison, need to extract S2 features for validation sets")

    if not np.isnan(roc_auc):
        if roc_auc >= target:
            print(f"\n✓ TARGET MET: edge_cases ROC-AUC = {roc_auc:.3f} ≥ {target:.2f}")
        else:
            gap = target - roc_auc
            print(f"\n✗ TARGET NOT MET: {gap:.3f} below target")
            if roc_auc > baseline_69d:
                improvement = roc_auc - baseline_69d
                print(f"  ✓ Improved by {improvement:+.3f} over 69D baseline")
            else:
                print(f"  ✗ Did not improve over 69D baseline")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    results = {
        'edge_cases': {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'baseline_69d': baseline_69d,
            'improvement_vs_69d': roc_auc - baseline_69d if not np.isnan(roc_auc) else float('nan'),
            'cv_score': grid_search.best_score_,
            'note': 'Validation WITHOUT Sentinel-2 features'
        },
        'training': {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'alphaearth_features': 69,
            'sentinel2_features': s2_count,
            'cv_score': grid_search.best_score_,
            'train_val_gap': train_score - val_score,
            's2_extraction_stats': stats
        }
    }

    results_file = results_dir / 'xgboost_sentinel2_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    model_file = processed_dir / 'walk_model_xgboost_sentinel2.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': best_xgb,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'metadata': {
                'n_train_samples': len(X_train),
                'n_features': X_train.shape[1],
                'alphaearth_features': 69,
                'sentinel2_features': s2_count,
                'model_type': 'XGBoost',
                'cv_strategy': 'StratifiedKFold_5',
                'regularization': {
                    'reg_lambda': grid_search.best_params_['reg_lambda'],
                    'reg_alpha': grid_search.best_params_['reg_alpha']
                },
                's2_extraction_stats': stats
            }
        }, f)
    print(f"✓ Model saved to: {model_file}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
