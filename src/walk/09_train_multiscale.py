"""
WALK Phase - Train and Evaluate with Multi-Scale Features

Tests whether multi-scale embeddings improve small-scale clearing detection.

Target: Address 100% miss rate on clearings < 1 ha

Usage:
    uv run python src/walk/09_train_multiscale.py
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils import get_config


def prepare_multiscale_features(samples, include_multiscale=True):
    """
    Prepare feature matrix with multi-scale features.

    Args:
        samples: List of sample dicts
        include_multiscale: Whether to include multi-scale features

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    X = []
    y = []
    feature_names = []

    for sample in samples:
        features_vec = []
        label = sample.get('label', 0)

        # Temporal features (baseline)
        temporal_feats = sample.get('features', {})

        # Distances
        for timepoint in ['Q1', 'Q2', 'Q3', 'Q4']:
            dist = temporal_feats.get('distances', {}).get(timepoint, 0.0)
            features_vec.append(dist)
            if len(feature_names) < 200:
                feature_names.append(f'distance_{timepoint}')

        # Velocities
        for transition in ['Q1_Q2', 'Q2_Q3', 'Q3_Q4']:
            vel = temporal_feats.get('velocities', {}).get(transition, 0.0)
            features_vec.append(vel)
            if len(feature_names) < 200:
                feature_names.append(f'velocity_{transition}')

        # Accelerations
        for accel_key in ['Q1_Q2_Q3', 'Q2_Q3_Q4']:
            accel = temporal_feats.get('accelerations', {}).get(accel_key, 0.0)
            features_vec.append(accel)
            if len(feature_names) < 200:
                feature_names.append(f'acceleration_{accel_key}')

        # Trend consistency
        trend = temporal_feats.get('trend_consistency', 0.0)
        features_vec.append(trend)
        if len(feature_names) < 200:
            feature_names.append('trend_consistency')

        # Multi-scale features (if available and requested)
        if include_multiscale and 'multiscale_features' in sample:
            ms_feats = sample['multiscale_features']

            # Sentinel-2 bands
            for band in ['b2', 'b3', 'b4', 'b8', 'b5', 'b6', 'b7', 'b8a', 'b11', 'b12']:
                val = ms_feats.get(f's2_{band}', 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(f's2_{band}')

            # Sentinel-2 indices
            for index in ['ndvi', 'nbr', 'evi', 'ndwi']:
                val = ms_feats.get(f's2_{index}', 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(f's2_{index}')

            # Coarse-scale landscape context (64D embedding)
            for i in range(64):
                key = f'coarse_emb_{i}'
                val = ms_feats.get(key, 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(key)

            # Landscape statistics
            for stat in ['heterogeneity', 'range']:
                key = f'coarse_{stat}'
                val = ms_feats.get(key, 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(key)

        X.append(features_vec)
        y.append(label)

    return np.array(X), np.array(y), feature_names


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate model and print metrics."""
    # Handle case where all predictions are the same class
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # If confusion matrix is not 2x2 (e.g., all predictions same class)
        print(f"\n{model_name} Performance:")
        print(f"  ⚠ Warning: Model predicted only one class")
        print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
        return {
            'roc_auc': 0.0,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': 0.0,
            'recall': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0,
        }

    print(f"\n{model_name} Performance:")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_pred_proba):.3f}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}, FP: {fp}")
    print(f"    FN: {fn}, TN: {tn}")

    return {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def analyze_small_scale_performance(samples, y_true, y_pred):
    """
    Analyze performance specifically on small-scale clearings (< 1 ha).

    Args:
        samples: List of sample dicts
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict with small-scale performance metrics
    """
    # Filter to clearing samples with area < 1 ha
    small_clearing_indices = []
    for i, sample in enumerate(samples):
        if sample.get('label', 0) == 1:  # Clearing
            area_ha = sample.get('area_ha', float('inf'))
            if area_ha < 1.0:
                small_clearing_indices.append(i)

    if len(small_clearing_indices) == 0:
        print("\n  No clearings < 1 ha in dataset")
        return None

    # Compute metrics for small clearings
    y_true_small = y_true[small_clearing_indices]
    y_pred_small = y_pred[small_clearing_indices]

    tp = np.sum((y_true_small == 1) & (y_pred_small == 1))
    fn = np.sum((y_true_small == 1) & (y_pred_small == 0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n  Small-scale clearings (< 1 ha):")
    print(f"    Total: {len(small_clearing_indices)}")
    print(f"    Detected (TP): {tp}")
    print(f"    Missed (FN): {fn}")
    print(f"    Recall: {recall:.1%}")

    return {
        'n_small': len(small_clearing_indices),
        'tp': int(tp),
        'fn': int(fn),
        'recall': float(recall),
    }


def main():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("=" * 80)
    print("TRAIN AND EVALUATE WITH MULTI-SCALE FEATURES")
    print("=" * 80)

    # Load rapid response validation set with multi-scale features
    val_file = processed_dir / "hard_val_rapid_response_multiscale.pkl"
    print(f"\nLoading validation data from {val_file.name}...")

    if not val_file.exists():
        print(f"✗ File not found: {val_file}")
        print(f"\nPlease run first:")
        print(f"  uv run python src/walk/08_multiscale_embeddings.py --set rapid_response")
        return

    with open(val_file, 'rb') as f:
        val_samples = pickle.load(f)

    print(f"Loaded {len(val_samples)} validation samples")
    n_clearing = sum(1 for s in val_samples if s.get('label', 0) == 1)
    n_intact = len(val_samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}")

    # Prepare features
    print(f"\n{'='*80}")
    print("PREPARING FEATURES")
    print(f"{'='*80}")

    X_temporal, y_val, feature_names_temporal = prepare_multiscale_features(
        val_samples,
        include_multiscale=False
    )
    print(f"\nTemporal features: {X_temporal.shape}")

    X_multiscale, _, feature_names_multiscale = prepare_multiscale_features(
        val_samples,
        include_multiscale=True
    )
    print(f"Temporal + Multi-scale features: {X_multiscale.shape}")

    # Check how many samples have multi-scale features
    n_with_multiscale = sum(1 for s in val_samples if 'multiscale_features' in s)
    print(f"\nSamples with multi-scale features: {n_with_multiscale}/{len(val_samples)}")

    # Cross-validation comparison
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPARISON")
    print(f"{'='*80}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline: Temporal only
    print("\nBaseline (Temporal features only):")
    scaler_temporal = StandardScaler()
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)

    lr_temporal = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    temporal_scores = cross_val_score(
        lr_temporal,
        X_temporal_scaled,
        y_val,
        cv=cv,
        scoring='roc_auc'
    )
    print(f"  Mean ROC-AUC: {np.mean(temporal_scores):.3f} (+/- {np.std(temporal_scores):.3f})")

    # Multi-scale
    print("\nMulti-scale (Temporal + Fine + Coarse):")
    scaler_multiscale = StandardScaler()
    X_multiscale_scaled = scaler_multiscale.fit_transform(X_multiscale)

    lr_multiscale = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    multiscale_scores = cross_val_score(
        lr_multiscale,
        X_multiscale_scaled,
        y_val,
        cv=cv,
        scoring='roc_auc'
    )
    print(f"  Mean ROC-AUC: {np.mean(multiscale_scores):.3f} (+/- {np.std(multiscale_scores):.3f})")

    improvement = np.mean(multiscale_scores) - np.mean(temporal_scores)
    print(f"\nImprovement: {improvement:+.1%}")

    # Train final models on full validation set
    print(f"\n{'='*80}")
    print("FULL VALIDATION SET EVALUATION")
    print(f"{'='*80}")

    # Baseline
    lr_temporal.fit(X_temporal_scaled, y_val)
    y_pred_temporal = lr_temporal.predict(X_temporal_scaled)
    y_pred_proba_temporal = lr_temporal.predict_proba(X_temporal_scaled)[:, 1]

    baseline_results = evaluate_model(
        y_val,
        y_pred_temporal,
        y_pred_proba_temporal,
        "BASELINE (Temporal)"
    )

    # Multi-scale
    lr_multiscale.fit(X_multiscale_scaled, y_val)
    y_pred_multiscale = lr_multiscale.predict(X_multiscale_scaled)
    y_pred_proba_multiscale = lr_multiscale.predict_proba(X_multiscale_scaled)[:, 1]

    multiscale_results = evaluate_model(
        y_val,
        y_pred_multiscale,
        y_pred_proba_multiscale,
        "MULTI-SCALE"
    )

    # Small-scale clearing analysis
    print(f"\n{'='*80}")
    print("SMALL-SCALE CLEARING PERFORMANCE")
    print(f"{'='*80}")

    print("\nBaseline (Temporal):")
    baseline_small = analyze_small_scale_performance(
        val_samples,
        y_val,
        y_pred_temporal
    )

    print("\nMulti-scale:")
    multiscale_small = analyze_small_scale_performance(
        val_samples,
        y_val,
        y_pred_multiscale
    )

    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    print(f"\nOverall Performance:")
    print(f"  Metric                  Baseline    Multi-scale    Change")
    print(f"  {'-'*60}")
    print(f"  ROC-AUC                 {baseline_results['roc_auc']:.3f}       {multiscale_results['roc_auc']:.3f}          {multiscale_results['roc_auc'] - baseline_results['roc_auc']:+.3f}")
    print(f"  Recall (most critical)  {baseline_results['recall']:.3f}       {multiscale_results['recall']:.3f}          {multiscale_results['recall'] - baseline_results['recall']:+.3f}")
    print(f"  Precision               {baseline_results['precision']:.3f}       {multiscale_results['precision']:.3f}          {multiscale_results['precision'] - baseline_results['precision']:+.3f}")

    if baseline_small and multiscale_small:
        print(f"\nSmall-scale clearings (< 1 ha):")
        print(f"  Baseline recall:    {baseline_small['recall']:.1%}")
        print(f"  Multi-scale recall: {multiscale_small['recall']:.1%}")
        print(f"  Improvement:        {multiscale_small['recall'] - baseline_small['recall']:+.1%}")

    # Target assessment
    print(f"\n{'='*80}")
    print("TARGET ASSESSMENT")
    print(f"{'='*80}")

    target_recall = 0.80
    target_precision = 0.70

    print(f"\nRapid Response Targets:")

    # Recall
    recall_gap = target_recall - multiscale_results['recall']
    recall_status = '✓ MET' if multiscale_results['recall'] >= target_recall else f'✗ MISSED (need {recall_gap:+.1%})'
    print(f"  Recall ≥ {target_recall:.0%}: {recall_status}")

    # Precision
    precision_gap = target_precision - multiscale_results['precision']
    precision_status = '✓ MET' if multiscale_results['precision'] >= target_precision else f'✗ MISSED (need {precision_gap:+.1%})'
    print(f"  Precision ≥ {target_precision:.0%}: {precision_status}")

    if multiscale_results['recall'] >= target_recall and multiscale_results['precision'] >= target_precision:
        print(f"\n✓ SUCCESS: Multi-scale features enable rapid response deployment!")
    elif improvement > 0:
        print(f"\n→ PROGRESS: Multi-scale improves performance, but targets not yet met")
        print(f"\nNext steps:")
        print(f"  1. Extract multi-scale for training set (currently training on validation)")
        print(f"  2. Try Random Forest (may capture non-linear multi-scale interactions)")
        print(f"  3. Scale up training data (option #4 from original priority list)")
    else:
        print(f"\n⚠ PARTIAL: Multi-scale doesn't improve over baseline")
        print(f"\nRecommendation: Move to scaling up dataset (priority #4)")


if __name__ == "__main__":
    main()
