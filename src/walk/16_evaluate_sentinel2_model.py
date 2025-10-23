"""
Evaluate XGBoost+Sentinel-2 model on edge_cases validation set.

This is the moment of truth! We'll test whether adding high-resolution
Sentinel-2 features (10m) breaks through the 0.583 ROC-AUC ceiling.

Baseline:
  - 69D features (AlphaEarth + multiscale): 0.583 ROC-AUC on edge_cases

New approach:
  - 115D features (69D + 46D Sentinel-2): ? ROC-AUC on edge_cases

Usage:
    uv run python src/walk/16_evaluate_sentinel2_model.py
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
    # Expected feature structure:
    # - 3D annual magnitude features (mean annual AlphaEarth)
    # - 66D coarse landscape features (3x3 grid at 100m spacing)
    # - 46D Sentinel-2 features

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

    print(f"  Expected {len(expected_s2_keys)} Sentinel-2 features per sample")

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
    print("EVALUATE XGBOOST+SENTINEL-2 MODEL ON EDGE_CASES")
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

    # Load edge_cases validation set with S2 features
    val_path = processed_dir / 'hard_val_edge_cases_sentinel2.pkl'

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
    print(f"  Feature dimensionality: {X_val.shape[1]}D")

    # Verify feature alignment
    if val_feature_names != train_feature_names:
        print("\n⚠ WARNING: Feature names don't match exactly!")
        print(f"  Training features: {len(train_feature_names)}")
        print(f"  Validation features: {len(val_feature_names)}")

        # Check if same features in different order
        if set(val_feature_names) == set(train_feature_names):
            print("  ✓ Same features, different order - reordering...")
            # Reorder validation features to match training
            feature_indices = [val_feature_names.index(f) for f in train_feature_names]
            X_val = X_val[:, feature_indices]
            val_feature_names = train_feature_names
        else:
            # Show what's different
            print("  ✗ Different features!")
            train_set = set(train_feature_names)
            val_set = set(val_feature_names)

            missing_in_val = train_set - val_set
            extra_in_val = val_set - train_set

            if missing_in_val:
                print(f"\n  Missing in validation ({len(missing_in_val)}): {sorted(list(missing_in_val))[:10]}")
            if extra_in_val:
                print(f"\n  Extra in validation ({len(extra_in_val)}): {sorted(list(extra_in_val))[:10]}")

            print("\n  Attempting to use only common features...")
            common_features = sorted(list(train_set & val_set))
            if len(common_features) > 0:
                print(f"  Using {len(common_features)} common features")
                # Reorder both to use only common features
                train_indices = [train_feature_names.index(f) for f in common_features]
                val_indices = [val_feature_names.index(f) for f in common_features]
                X_val = X_val[:, val_indices]
                val_feature_names = common_features
                print(f"  Proceeding with {len(common_features)}D features")
            else:
                print("  No common features - cannot evaluate!")
                return

    # Make predictions
    print("\nMaking predictions...")
    X_val_scaled = scaler.transform(X_val)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)

    # Calculate metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='binary', pos_label=1
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nXGBoost + Sentinel-2 (115D features):")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\nClass distribution in validation set:")
    clearing_count = np.sum(y_val)
    intact_count = len(y_val) - clearing_count
    print(f"  Clearing: {clearing_count} samples")
    print(f"  Intact:   {intact_count} samples")

    # Compare to baseline
    baseline_roc_auc = 0.583
    improvement = roc_auc - baseline_roc_auc
    improvement_pct = (improvement / baseline_roc_auc) * 100

    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINE")
    print("=" * 80)

    print(f"\nBaseline (69D AlphaEarth + multiscale):")
    print(f"  ROC-AUC: {baseline_roc_auc:.4f}")

    print(f"\nNew model (115D AlphaEarth + multiscale + Sentinel-2):")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {improvement_pct:+.1f}%")

    if roc_auc > baseline_roc_auc:
        print(f"\n✓ SUCCESS! Broke through the 0.583 ceiling!")
    elif roc_auc > baseline_roc_auc - 0.01:
        print(f"\n≈ Approximately same performance (within 0.01)")
    else:
        print(f"\n✗ Performance dropped compared to baseline")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if roc_auc > baseline_roc_auc:
        print("\n1. Analyze which Sentinel-2 features are most important")
        print("2. Consider extracting S2 features for other validation sets")
        print("3. Explore temporal S2 features (multi-date composites)")
    else:
        print("\n1. Analyze why S2 features didn't help")
        print("2. Check feature importance to see if S2 features are used")
        print("3. Consider alternative approaches (temporal features, SAR data, etc.)")


if __name__ == '__main__':
    main()
