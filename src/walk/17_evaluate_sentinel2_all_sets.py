"""
Evaluate XGBoost+Sentinel-2 model on ALL validation sets.

Comprehensive evaluation comparing:
- Baseline (69D): Random Forest performance
- New model (115D): XGBoost + Sentinel-2 performance

Usage:
    uv run python src/walk/17_evaluate_sentinel2_all_sets.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from src.utils import get_config


def combine_features_with_sentinel2(samples):
    """
    Extract and combine AlphaEarth + Sentinel-2 features from validation samples.

    Returns:
        X: Feature matrix (samples × 115 features)
        y: Labels
        feature_names: List of feature names
        stats: Dict with extraction statistics
    """
    X_list = []
    y_list = []

    s2_success = 0
    s2_failed = 0

    # First pass: determine the expected S2 feature set from samples with complete features
    expected_s2_keys = None
    for sample in samples:
        if 'multiscale_features' in sample:
            s2_keys_candidate = sorted([k for k in sample['multiscale_features'].keys() if k.startswith('s2_')])
            # Keep the largest S2 feature set as the template
            if expected_s2_keys is None or len(s2_keys_candidate) > len(expected_s2_keys):
                expected_s2_keys = s2_keys_candidate

    if expected_s2_keys is None:
        expected_s2_keys = []

    # Second pass: extract features only from samples with complete S2 feature sets
    for sample in samples:
        # Get label (validation sets use 'label' field)
        label = sample.get('label', 0)

        # Extract multiscale features
        if 'multiscale_features' not in sample:
            s2_failed += 1
            continue

        multiscale_dict = sample['multiscale_features']

        # Extract annual magnitude features (3D)
        # Check for correct naming: delta_1yr, delta_2yr, acceleration
        annual_keys = ['delta_1yr', 'delta_2yr', 'acceleration']
        if not all(k in multiscale_dict for k in annual_keys):
            s2_failed += 1
            continue

        # Extract coarse landscape features (66D)
        coarse_keys = sorted([k for k in multiscale_dict.keys() if k.startswith('coarse_')])
        if len(coarse_keys) != 66:
            s2_failed += 1
            continue

        # Extract Sentinel-2 features - must have ALL expected features
        s2_keys = sorted([k for k in multiscale_dict.keys() if k.startswith('s2_')])

        # Check if this sample has the complete S2 feature set
        if set(s2_keys) != set(expected_s2_keys):
            s2_failed += 1
            continue

        s2_success += 1

        annual_features = np.array([multiscale_dict[k] for k in annual_keys])
        coarse_features = np.array([multiscale_dict[k] for k in coarse_keys])
        s2_features = np.array([multiscale_dict[k] for k in expected_s2_keys])

        # Combine all features
        combined = np.concatenate([annual_features, coarse_features, s2_features])

        X_list.append(combined)
        y_list.append(label)

    stats = {
        's2_success': s2_success,
        's2_failed': s2_failed,
        'total': len(samples),
        'usable': len(X_list)
    }

    if len(X_list) == 0:
        return np.array([]), np.array([]), [], stats

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Generate feature names (using template from first sample with features)
    feature_names = annual_keys + coarse_keys + expected_s2_keys

    return X, y, feature_names, stats


def main():
    print("=" * 80)
    print("EVALUATE XGBOOST+SENTINEL-2 MODEL ON ALL VALIDATION SETS")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load trained model
    model_path = processed_dir / 'walk_model_xgboost_sentinel2.pkl'

    print(f"\nLoading trained model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    train_feature_names = model_data['feature_names']

    print(f"  Model trained with {len(train_feature_names)} features")
    print(f"  Best CV ROC-AUC: {model_data['cv_score']:.4f}")

    # Validation sets to evaluate
    val_sets = ['risk_ranking', 'rapid_response', 'edge_cases']

    # Baseline scores from Random Forest (69D)
    baseline_scores = {
        'risk_ranking': 0.907,
        'rapid_response': 0.760,
        'edge_cases': 0.583
    }

    results = {}

    # Evaluate each validation set
    for set_name in val_sets:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {set_name}")
        print(f"{'='*80}")

        val_path = processed_dir / f'hard_val_{set_name}_sentinel2.pkl'

        if not val_path.exists():
            print(f"  ✗ File not found: {val_path}")
            continue

        print(f"\nLoading validation set from: {val_path}")
        with open(val_path, 'rb') as f:
            val_samples = pickle.load(f)

        print(f"  Loaded {len(val_samples)} samples")

        # Extract features
        print("\nExtracting features...")
        X_val, y_val, val_feature_names, stats = combine_features_with_sentinel2(val_samples)

        print(f"  Total samples: {stats['total']}")
        print(f"  With S2 features: {stats['s2_success']}")
        print(f"  Without S2 features: {stats['s2_failed']}")
        print(f"  Usable for evaluation: {stats['usable']}")

        if stats['usable'] == 0:
            print(f"\n  ✗ No usable samples, skipping {set_name}")
            continue

        print(f"  Feature dimensionality: {X_val.shape[1]}D")

        # Verify feature alignment
        if val_feature_names != train_feature_names:
            if set(val_feature_names) == set(train_feature_names):
                print("  ✓ Reordering features to match training...")
                feature_indices = [val_feature_names.index(f) for f in train_feature_names]
                X_val = X_val[:, feature_indices]
                val_feature_names = train_feature_names
            else:
                print("  ✗ Feature mismatch - cannot evaluate!")
                continue

        # Make predictions
        print("\nMaking predictions...")
        X_val_scaled = scaler.transform(X_val)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_pred = model.predict(X_val_scaled)

        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            roc_auc = float('nan')

        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', pos_label=1, zero_division=0
        )

        clearing_count = np.sum(y_val)
        intact_count = len(y_val) - clearing_count

        baseline = baseline_scores.get(set_name, 0.0)
        improvement = roc_auc - baseline if not np.isnan(roc_auc) else float('nan')
        improvement_pct = (improvement / baseline) * 100 if baseline > 0 and not np.isnan(improvement) else float('nan')

        results[set_name] = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'baseline': baseline,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'n_samples': len(y_val),
            'n_clearing': int(clearing_count),
            'n_intact': int(intact_count)
        }

        print("\nResults:")
        print(f"  ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "  ROC-AUC:   nan")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nClass distribution:")
        print(f"  Clearing: {clearing_count} samples")
        print(f"  Intact:   {intact_count} samples")

    # Summary table
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Validation Set':<20s} {'Samples':>8s} {'RF (69D)':>10s} {'XGB+S2 (115D)':>15s} {'Δ':>10s}")
    print("-" * 80)

    for set_name in val_sets:
        if set_name in results:
            r = results[set_name]
            if not np.isnan(r['roc_auc']):
                print(f"{set_name:<20s} {r['n_samples']:>8d} {r['baseline']:>10.3f} {r['roc_auc']:>15.3f} {r['improvement']:>+10.3f}")
            else:
                print(f"{set_name:<20s} {r['n_samples']:>8d} {r['baseline']:>10.3f}             nan            nan")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")

    # Calculate average improvement
    valid_improvements = [r['improvement'] for r in results.values() if not np.isnan(r['improvement'])]

    if len(valid_improvements) > 0:
        avg_improvement = np.mean(valid_improvements)
        print(f"Average improvement across {len(valid_improvements)} validation sets: {avg_improvement:+.3f}")

        if avg_improvement > 0.05:
            print(f"\n✓ Sentinel-2 features IMPROVED performance")
        elif avg_improvement > -0.05:
            print(f"\n≈ Sentinel-2 features had NEUTRAL impact")
        else:
            print(f"\n✗ Sentinel-2 features DECREASED performance")
            print(f"\nPossible reasons:")
            print(f"  1. Overfitting to training data (CV: {model_data['cv_score']:.4f})")
            print(f"  2. S2 features may not generalize to validation sets")
            print(f"  3. Feature count increased from 69D to 115D with same sample size")
    else:
        print("✗ Could not compute improvements (no valid results)")

    print()


if __name__ == '__main__':
    main()
