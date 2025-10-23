"""
Phase 1: Train and Evaluate Multiscale Model

Trains a model combining:
- Annual delta features (3D): delta_1yr, delta_2yr, acceleration
- Multiscale features (80D): fine-scale Sentinel-2 + coarse-scale landscape

Total: 83D feature space

Usage:
    uv run python src/walk/10a_phase1_train_multiscale.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient
from src.walk.diagnostic_helpers import extract_dual_year_features


ANNUAL_FEATURE_NAMES = ['delta_1yr', 'delta_2yr', 'acceleration']


def combine_features(annual_data, multiscale_data, is_validation=False):
    """
    Combine pre-extracted annual features with multiscale features.

    Args:
        annual_data: Either dict with 'X', 'y', 'samples' keys (training)
                     or list of sample dicts (validation)
        multiscale_data: Dict with 'data' key containing list of samples, or list of samples
        is_validation: Whether this is validation data (different format)

    Returns:
        X: Combined feature matrix (N, 83)
        y: Labels (N,)
        feature_names: List of feature names
        success_count: Number of successfully combined samples
    """
    # Build mapping from sample ID to index in each dataset
    # Use (lat, lon, year) as unique identifier
    def get_sample_id(sample):
        return (sample['lat'], sample['lon'], sample['year'])

    # Handle different data formats
    if is_validation:
        # Validation: annual_data is a list of sample dicts with 'features' field
        annual_samples = annual_data
        annual_id_to_data = {get_sample_id(s): s for s in annual_samples}
    else:
        # Training: annual_data is a dict with 'X', 'y', 'samples'
        X_annual = annual_data['X']  # (N_annual, 3)
        y_annual = annual_data['y']  # (N_annual,)
        annual_samples = annual_data['samples']  # List of sample dicts
        annual_id_to_idx = {get_sample_id(s): i for i, s in enumerate(annual_samples)}

    # Get multiscale data (could be dict with 'data' key or direct list)
    if isinstance(multiscale_data, dict) and 'data' in multiscale_data:
        multiscale_samples = multiscale_data['data']
    elif isinstance(multiscale_data, list):
        multiscale_samples = multiscale_data
    else:
        raise ValueError("multiscale_data must be dict with 'data' key or list")

    multiscale_id_to_idx = {get_sample_id(s): i for i, s in enumerate(multiscale_samples)}

    # Find common samples
    if is_validation:
        annual_ids = set(annual_id_to_data.keys())
    else:
        annual_ids = set(annual_id_to_idx.keys())

    multiscale_ids = set(multiscale_id_to_idx.keys())
    common_ids = annual_ids & multiscale_ids

    print(f"  Annual features: {len(annual_ids)} samples")
    print(f"  Multiscale features: {len(multiscale_ids)} samples")
    print(f"  Common samples: {len(common_ids)} samples")

    # Combine features for common samples
    X_combined = []
    y_combined = []

    # Define expected feature names (14 S2 + 66 coarse = 80)
    expected_s2_keys = ['s2_b11', 's2_b12', 's2_b2', 's2_b3', 's2_b4', 's2_b5', 's2_b6',
                        's2_b7', 's2_b8', 's2_b8a', 's2_evi', 's2_nbr', 's2_ndvi', 's2_ndwi']
    expected_coarse_keys = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
    multiscale_feature_names = expected_s2_keys + expected_coarse_keys

    failed_samples = []
    incomplete_samples = []

    for sample_id in common_ids:
        multiscale_idx = multiscale_id_to_idx[sample_id]

        # Get annual features (3D) from appropriate source
        if is_validation:
            annual_sample = annual_id_to_data[sample_id]
            if 'features' not in annual_sample or annual_sample['features'] is None:
                failed_samples.append(sample_id)
                continue
            annual_features = np.array(annual_sample['features'])  # Should be 3D
            label = annual_sample.get('label', 0)
        else:
            annual_idx = annual_id_to_idx[sample_id]
            annual_features = X_annual[annual_idx]
            label = y_annual[annual_idx]

        # Get multiscale features (80D)
        multiscale_sample = multiscale_samples[multiscale_idx]

        if 'multiscale_features' not in multiscale_sample:
            failed_samples.append(sample_id)
            continue

        multiscale_dict = multiscale_sample['multiscale_features']

        # Check if all required features are present
        missing_features = [k for k in multiscale_feature_names if k not in multiscale_dict]
        if missing_features:
            incomplete_samples.append((sample_id, len(multiscale_dict)))
            continue

        multiscale_features = np.array([multiscale_dict[k] for k in multiscale_feature_names])

        # Combine: 3D annual + 80D multiscale = 83D
        combined = np.concatenate([annual_features, multiscale_features])

        if len(combined) != 83:
            failed_samples.append(sample_id)
            continue

        X_combined.append(combined)
        y_combined.append(label)

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    all_feature_names = list(ANNUAL_FEATURE_NAMES) + multiscale_feature_names

    if incomplete_samples:
        print(f"  ✗ Skipped {len(incomplete_samples)} samples with incomplete multiscale features")
    if failed_samples:
        print(f"  ✗ Failed to combine: {len(failed_samples)} samples")

    return X, y, all_feature_names, len(X)


def main():
    print("=" * 80)
    print("PHASE 1: TRAIN AND EVALUATE MULTISCALE MODEL")
    print("=" * 80)

    # Initialize
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    results_dir = Path('results/walk')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted annual features
    annual_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'
    print(f"\nLoading pre-extracted annual features from: {annual_path}")

    with open(annual_path, 'rb') as f:
        annual_data = pickle.load(f)

    print(f"  Loaded {len(annual_data['X'])} samples with 3D annual features")

    # Load multiscale features
    multiscale_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale.pkl'
    print(f"\nLoading multiscale features from: {multiscale_path}")

    with open(multiscale_path, 'rb') as f:
        multiscale_data = pickle.load(f)

    print(f"  Loaded {len(multiscale_data['data'])} samples with multiscale features")

    # Combine features
    print(f"\n{'='*80}")
    print("COMBINING ANNUAL AND MULTISCALE FEATURES")
    print(f"{'='*80}\n")

    X_train, y_train, all_feature_names, n_combined = combine_features(annual_data, multiscale_data)

    print(f"\n✓ Successfully combined {n_combined} samples")
    print(f"  Feature dimension: {X_train.shape[1]}D (3D annual + 80D multiscale)")
    print(f"  Clearing: {np.sum(y_train == 1)}")
    print(f"  Intact: {np.sum(y_train == 0)}")

    # Train model
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'='*80}\n")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("  ✓ Model trained")

    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}\n")

    feature_importance = np.abs(model.coef_[0])
    importance_idx = np.argsort(feature_importance)[::-1]

    print("Top 20 most important features:")
    for i, idx in enumerate(importance_idx[:20], 1):
        print(f"{i:3d}. {all_feature_names[idx]:30s} {feature_importance[idx]:.4f}")

    # Load validation sets
    print(f"\n{'='*80}")
    print("LOADING VALIDATION SETS")
    print(f"{'='*80}\n")

    val_sets = ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']
    val_data = {}

    # Initialize Earth Engine client for validation annual feature extraction
    print("Initializing Earth Engine client for validation...")
    ee_client = EarthEngineClient(use_cache=True)

    for set_name in val_sets:
        # Check for multiscale features
        multiscale_val_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'

        if not multiscale_val_path.exists():
            print(f"⚠ Skipping {set_name}: multiscale features not found")
            continue

        print(f"Loading {set_name}...")

        # Load multiscale features (list of sample dicts)
        with open(multiscale_val_path, 'rb') as f:
            val_samples = pickle.load(f)

        print(f"  Loaded {len(val_samples)} samples")

        # Extract features on-the-fly for validation (small datasets)
        X_val = []
        y_val = []
        success_count = 0
        failed_count = 0

        for sample in val_samples:
            # Extract 3D annual features
            try:
                annual_features = extract_dual_year_features(ee_client, sample)
            except:
                annual_features = None

            if annual_features is None:
                failed_count += 1
                continue

            # Get 80D multiscale features
            if 'multiscale_features' not in sample:
                failed_count += 1
                continue

            multiscale_dict = sample['multiscale_features']

            # Check if all required features are present (14 S2 + 66 coarse = 80)
            expected_s2_keys = ['s2_b11', 's2_b12', 's2_b2', 's2_b3', 's2_b4', 's2_b5', 's2_b6',
                                's2_b7', 's2_b8', 's2_b8a', 's2_evi', 's2_nbr', 's2_ndvi', 's2_ndwi']
            expected_coarse_keys = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
            multiscale_feature_names = expected_s2_keys + expected_coarse_keys

            missing_features = [k for k in multiscale_feature_names if k not in multiscale_dict]
            if missing_features:
                failed_count += 1
                continue

            multiscale_features = np.array([multiscale_dict[k] for k in multiscale_feature_names])

            # Combine: 3D annual + 80D multiscale = 83D
            combined = np.concatenate([annual_features, multiscale_features])

            if len(combined) != 83:
                failed_count += 1
                continue

            X_val.append(combined)
            y_val.append(sample.get('label', 0))
            success_count += 1

        if len(X_val) == 0:
            print(f"  ⚠ No valid features extracted, skipping")
            continue

        X_val = np.vstack(X_val)
        y_val = np.array(y_val)

        print(f"  Extracted {success_count}/{len(val_samples)} samples ({failed_count} failed)")

        val_data[set_name] = {
            'X': X_val,
            'y': y_val
        }

    # Evaluate on validation sets
    print(f"\n{'='*80}")
    print("EVALUATION ON VALIDATION SETS")
    print(f"{'='*80}\n")

    # Baseline results from Phase 1 (annual-only, 3D)
    baseline_results = {
        'risk_ranking': 0.825,
        'rapid_response': 0.786,
        'comprehensive': 0.747,
        'edge_cases': 0.533
    }

    results = {}

    for set_name, data in val_data.items():
        X_val_scaled = scaler.transform(data['X'])
        y_val = data['y']

        # Predictions
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_pred = model.predict(X_val_scaled)

        # Metrics
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            print(f"⚠ {set_name}: Cannot compute ROC-AUC (only one class present)")
            roc_auc = float('nan')

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)

        baseline = baseline_results.get(set_name, 0.0)
        diff = roc_auc - baseline if not np.isnan(roc_auc) else float('nan')
        pct_change = (diff / baseline * 100) if baseline > 0 and not np.isnan(diff) else float('nan')

        results[set_name] = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'baseline': baseline,
            'improvement': diff,
            'pct_change': pct_change
        }

        print(f"{set_name}:")
        print("=" * 60)
        if not np.isnan(roc_auc):
            print(f"  ROC-AUC:   {roc_auc:.3f}  (baseline: {baseline:.3f}, {diff:+.3f} / {pct_change:+.1f}%)")
        else:
            print(f"  ROC-AUC:   nan  (baseline: {baseline:.3f})")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN:  {cm[0, 0]:2d}  FP:  {cm[0, 1]:2d}")
        print(f"    FN:  {cm[1, 0]:2d}  TP:  {cm[1, 1]:2d}")
        print(f"\n  Class Distribution:")
        print(f"    Clearing (1): {np.sum(y_val == 1)} samples")
        print(f"    Intact (0):   {np.sum(y_val == 0)} samples")

        n_clearing = np.sum(y_val == 1)
        if n_clearing > 0:
            n_detected = np.sum((y_val == 1) & (y_pred == 1))
            print(f"\n  Clearing Detection:")
            print(f"    Detected: {n_detected}/{n_clearing} ({n_detected/n_clearing*100:.1f}%)")
        print()

    # Summary
    print(f"{'='*80}")
    print("MULTISCALE MODEL RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Validation Set':<20s} {'Baseline':>10s} {'Multiscale':>10s} {'Change':>10s} {'% Change':>10s}")
    print("-" * 80)
    for set_name in val_sets:
        if set_name in results:
            r = results[set_name]
            if not np.isnan(r['roc_auc']):
                print(f"{set_name:<20s} {r['baseline']:>10.3f} {r['roc_auc']:>10.3f} "
                      f"{r['improvement']:>+10.3f} {r['pct_change']:>+9.1f}%")
            else:
                print(f"{set_name:<20s} {r['baseline']:>10.3f}        nan          nan        nan")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA")
    print(f"{'='*80}\n")

    edge_roc = results.get('edge_cases', {}).get('roc_auc', 0.0)
    edge_baseline = baseline_results.get('edge_cases', 0.0)
    target = 0.70

    print(f"Target: edge_cases ROC-AUC ≥ {target:.2f}")
    print(f"Baseline (annual-only, 3D): {edge_baseline:.3f}")

    if not np.isnan(edge_roc):
        edge_improvement = edge_roc - edge_baseline
        print(f"Multiscale (83D): {edge_roc:.3f}")
        print(f"Improvement: {edge_improvement:+.3f} ({edge_improvement/edge_baseline*100:+.1f}%)")

        if edge_roc >= target:
            print(f"\n✓ TARGET MET: edge_cases ROC-AUC = {edge_roc:.3f} ≥ {target:.2f}")
            print(f"  Multiscale features successfully bridged the gap!")
        else:
            gap = target - edge_roc
            print(f"\n✗ TARGET NOT MET: {gap:.3f} below target")
            if edge_improvement > 0:
                print(f"  ✓ Multiscale features improved performance by {edge_improvement:.3f}")
                print(f"  Consider: Adding contextual features (roads, fire history)")
            else:
                print(f"  ✗ No improvement from multiscale features")
                print(f"  Consider: Phase 2 with specialized models")
    else:
        print(f"Multiscale (83D): nan")
        print(f"\n✗ Could not evaluate edge_cases (ROC-AUC = nan)")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    results_file = results_dir / 'phase1_multiscale_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    model_file = processed_dir / 'walk_model_phase1_multiscale.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': all_feature_names,
            'metadata': {
                'n_train_samples': len(X_train),
                'n_features': X_train.shape[1],
                'feature_breakdown': {
                    'annual': 3,
                    'multiscale': 80,
                    'total': 83
                }
            }
        }, f)
    print(f"✓ Model saved to: {model_file}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
