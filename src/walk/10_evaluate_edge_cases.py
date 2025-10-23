"""
WALK Phase - Evaluate Multi-Scale on Edge Cases

Tests multi-scale performance on hardest samples:
- Small-scale clearings (< 5 ha)
- Fire-prone regions
- Extreme edge cases

Usage:
    uv run python src/walk/10_evaluate_edge_cases.py
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
    """Prepare feature matrix with multi-scale features."""
    X = []
    y = []
    feature_names = []

    for sample in samples:
        features_vec = []
        label = sample.get('label', 0)

        # Temporal features
        temporal_feats = sample.get('features', {})

        for timepoint in ['Q1', 'Q2', 'Q3', 'Q4']:
            dist = temporal_feats.get('distances', {}).get(timepoint, 0.0)
            features_vec.append(dist)
            if len(feature_names) < 200:
                feature_names.append(f'distance_{timepoint}')

        for transition in ['Q1_Q2', 'Q2_Q3', 'Q3_Q4']:
            vel = temporal_feats.get('velocities', {}).get(transition, 0.0)
            features_vec.append(vel)
            if len(feature_names) < 200:
                feature_names.append(f'velocity_{transition}')

        for accel_key in ['Q1_Q2_Q3', 'Q2_Q3_Q4']:
            accel = temporal_feats.get('accelerations', {}).get(accel_key, 0.0)
            features_vec.append(accel)
            if len(feature_names) < 200:
                feature_names.append(f'acceleration_{accel_key}')

        trend = temporal_feats.get('trend_consistency', 0.0)
        features_vec.append(trend)
        if len(feature_names) < 200:
            feature_names.append('trend_consistency')

        # Multi-scale features
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

            # Coarse-scale landscape (64D embedding)
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


def analyze_by_size(samples, y_true, y_pred, y_pred_proba):
    """Analyze performance by clearing size."""
    print("\n" + "="*80)
    print("PERFORMANCE BY CLEARING SIZE")
    print("="*80)

    size_bins = [
        (0, 1, "< 1 ha (tiny)"),
        (1, 5, "1-5 ha (small)"),
        (5, 10, "5-10 ha (medium)"),
        (10, float('inf'), "> 10 ha (large)"),
    ]

    for min_size, max_size, label in size_bins:
        # Find clearings in this size bin
        indices = []
        for i, sample in enumerate(samples):
            if sample.get('label', 0) == 1:  # Clearing
                area_ha = sample.get('area_ha', float('inf'))
                if min_size <= area_ha < max_size:
                    indices.append(i)

        if len(indices) == 0:
            print(f"\n{label}: No samples")
            continue

        y_true_bin = y_true[indices]
        y_pred_bin = y_pred[indices]
        y_pred_proba_bin = y_pred_proba[indices]

        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"\n{label}:")
        print(f"  Total: {len(indices)}")
        print(f"  Detected (TP): {tp}")
        print(f"  Missed (FN): {fn}")
        print(f"  Recall: {recall:.1%}")

        if tp > 0:
            avg_confidence = np.mean(y_pred_proba_bin[y_pred_bin == 1])
            print(f"  Avg confidence (TP): {avg_confidence:.3f}")


def main():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("=" * 80)
    print("EDGE CASES EVALUATION - MULTI-SCALE FEATURES")
    print("=" * 80)

    # Load edge cases with multi-scale features
    val_file = processed_dir / "hard_val_edge_cases_multiscale.pkl"
    print(f"\nLoading edge cases from {val_file.name}...")

    if not val_file.exists():
        print(f"✗ File not found: {val_file}")
        return

    with open(val_file, 'rb') as f:
        val_samples = pickle.load(f)

    print(f"Loaded {len(val_samples)} samples")
    n_clearing = sum(1 for s in val_samples if s.get('label', 0) == 1)
    n_intact = len(val_samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}")

    # Show area distribution
    print(f"\nClearing size distribution:")
    areas = [s.get('area_ha', 0) for s in val_samples if s.get('label', 0) == 1]
    if len(areas) > 0:
        print(f"  Min: {min(areas):.2f} ha")
        print(f"  Max: {max(areas):.2f} ha")
        print(f"  Median: {np.median(areas):.2f} ha")
        small = sum(1 for a in areas if a < 1)
        medium = sum(1 for a in areas if 1 <= a < 5)
        large = sum(1 for a in areas if a >= 5)
        print(f"  < 1 ha: {small}")
        print(f"  1-5 ha: {medium}")
        print(f"  ≥ 5 ha: {large}")

    # Prepare features
    print(f"\n{'='*80}")
    print("PREPARING FEATURES")
    print(f"{'='*80}")

    X_temporal, y_val, _ = prepare_multiscale_features(
        val_samples,
        include_multiscale=False
    )
    print(f"\nTemporal features: {X_temporal.shape}")

    X_multiscale, _, _ = prepare_multiscale_features(
        val_samples,
        include_multiscale=True
    )
    print(f"Temporal + Multi-scale features: {X_multiscale.shape}")

    # Cross-validation comparison
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPARISON")
    print(f"{'='*80}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline
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
    print(f"\nBaseline (Temporal only):")
    print(f"  Mean ROC-AUC: {np.mean(temporal_scores):.3f} (+/- {np.std(temporal_scores):.3f})")

    # Multi-scale
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
    print(f"\nMulti-scale:")
    print(f"  Mean ROC-AUC: {np.mean(multiscale_scores):.3f} (+/- {np.std(multiscale_scores):.3f})")

    improvement = np.mean(multiscale_scores) - np.mean(temporal_scores)
    print(f"\nImprovement: {improvement:+.1%}")

    # Full validation evaluation
    print(f"\n{'='*80}")
    print("FULL VALIDATION SET EVALUATION")
    print(f"{'='*80}")

    # Baseline
    lr_temporal.fit(X_temporal_scaled, y_val)
    y_pred_temporal = lr_temporal.predict(X_temporal_scaled)
    y_pred_proba_temporal = lr_temporal.predict_proba(X_temporal_scaled)[:, 1]

    try:
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_temporal).ravel()
        baseline_roc = roc_auc_score(y_val, y_pred_proba_temporal)
        baseline_recall = recall_score(y_val, y_pred_temporal)
        baseline_precision = precision_score(y_val, y_pred_temporal, zero_division=0)
    except:
        baseline_roc = 0.0
        baseline_recall = 0.0
        baseline_precision = 0.0
        tp = fn = tn = fp = 0

    print(f"\nBASELINE (Temporal):")
    print(f"  ROC-AUC:   {baseline_roc:.3f}")
    print(f"  Recall:    {baseline_recall:.3f}")
    print(f"  Precision: {baseline_precision:.3f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Multi-scale
    lr_multiscale.fit(X_multiscale_scaled, y_val)
    y_pred_multiscale = lr_multiscale.predict(X_multiscale_scaled)
    y_pred_proba_multiscale = lr_multiscale.predict_proba(X_multiscale_scaled)[:, 1]

    try:
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_multiscale).ravel()
        multiscale_roc = roc_auc_score(y_val, y_pred_proba_multiscale)
        multiscale_recall = recall_score(y_val, y_pred_multiscale)
        multiscale_precision = precision_score(y_val, y_pred_multiscale, zero_division=0)
    except:
        multiscale_roc = 0.0
        multiscale_recall = 0.0
        multiscale_precision = 0.0
        tp = fn = tn = fp = 0

    print(f"\nMULTI-SCALE:")
    print(f"  ROC-AUC:   {multiscale_roc:.3f}")
    print(f"  Recall:    {multiscale_recall:.3f}")
    print(f"  Precision: {multiscale_precision:.3f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Size-based analysis
    print(f"\n{'='*80}")
    print("BASELINE - Performance by size")
    print(f"{'='*80}")
    analyze_by_size(val_samples, y_val, y_pred_temporal, y_pred_proba_temporal)

    print(f"\n{'='*80}")
    print("MULTI-SCALE - Performance by size")
    print(f"{'='*80}")
    analyze_by_size(val_samples, y_val, y_pred_multiscale, y_pred_proba_multiscale)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\nMetric              Baseline    Multi-scale    Change")
    print(f"{'-'*60}")
    print(f"ROC-AUC             {baseline_roc:.3f}       {multiscale_roc:.3f}          {multiscale_roc - baseline_roc:+.3f}")
    print(f"Recall              {baseline_recall:.3f}       {multiscale_recall:.3f}          {multiscale_recall - baseline_recall:+.3f}")
    print(f"Precision           {baseline_precision:.3f}       {multiscale_precision:.3f}          {multiscale_precision - baseline_precision:+.3f}")

    if improvement > 0:
        print(f"\n✓ Multi-scale improves edge case performance by {improvement:+.1%}")
    else:
        print(f"\n⚠ Multi-scale does not improve edge case performance")

    print(f"\nEdge cases (hardest samples) results:")
    print(f"  Baseline ROC-AUC: {baseline_roc:.1%}")
    print(f"  Multi-scale ROC-AUC: {multiscale_roc:.1%}")
    print(f"  Previous baseline on edge cases: 0% (random)")
    print(f"  Improvement over original: {multiscale_roc:+.1%}")


if __name__ == "__main__":
    main()
