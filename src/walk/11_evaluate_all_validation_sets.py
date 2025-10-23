"""
WALK Phase - Comprehensive Multi-Scale Evaluation

Evaluates multi-scale performance across all 4 hard validation sets:
- Rapid Response (28 samples)
- Risk Ranking (43 samples)
- Comprehensive (70 samples)
- Edge Cases (22 samples)

Usage:
    uv run python src/walk/11_evaluate_all_validation_sets.py
"""

import pickle
from pathlib import Path

import numpy as np
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

            # Coarse-scale landscape
            for i in range(64):
                key = f'coarse_emb_{i}'
                val = ms_feats.get(key, 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(key)

            for stat in ['heterogeneity', 'range']:
                key = f'coarse_{stat}'
                val = ms_feats.get(key, 0.0)
                features_vec.append(val)
                if len(feature_names) < 200:
                    feature_names.append(key)

        X.append(features_vec)
        y.append(label)

    return np.array(X), np.array(y), feature_names


def evaluate_set(set_name, samples):
    """Evaluate a single validation set."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {set_name.upper()}")
    print(f"{'='*80}")

    n_clearing = sum(1 for s in samples if s.get('label', 0) == 1)
    n_intact = len(samples) - n_clearing
    print(f"\nSamples: {len(samples)} ({n_clearing} clearing, {n_intact} intact)")

    # Prepare features
    X_temporal, y, _ = prepare_multiscale_features(samples, include_multiscale=False)
    X_multiscale, _, _ = prepare_multiscale_features(samples, include_multiscale=True)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)

    # Baseline
    scaler_temporal = StandardScaler()
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)

    lr_temporal = LogisticRegression(random_state=42, max_iter=1000, C=0.1)

    try:
        temporal_scores = cross_val_score(
            lr_temporal,
            X_temporal_scaled,
            y,
            cv=cv,
            scoring='roc_auc'
        )
        temporal_cv = np.mean(temporal_scores)
    except:
        temporal_cv = 0.0

    # Multi-scale
    scaler_multiscale = StandardScaler()
    X_multiscale_scaled = scaler_multiscale.fit_transform(X_multiscale)

    lr_multiscale = LogisticRegression(random_state=42, max_iter=1000, C=0.1)

    try:
        multiscale_scores = cross_val_score(
            lr_multiscale,
            X_multiscale_scaled,
            y,
            cv=cv,
            scoring='roc_auc'
        )
        multiscale_cv = np.mean(multiscale_scores)
    except:
        multiscale_cv = 0.0

    # Full validation
    lr_temporal.fit(X_temporal_scaled, y)
    y_pred_temporal = lr_temporal.predict(X_temporal_scaled)
    y_pred_proba_temporal = lr_temporal.predict_proba(X_temporal_scaled)[:, 1]

    lr_multiscale.fit(X_multiscale_scaled, y)
    y_pred_multiscale = lr_multiscale.predict(X_multiscale_scaled)
    y_pred_proba_multiscale = lr_multiscale.predict_proba(X_multiscale_scaled)[:, 1]

    # Metrics
    try:
        baseline_roc = roc_auc_score(y, y_pred_proba_temporal)
        baseline_recall = recall_score(y, y_pred_temporal)
        baseline_precision = precision_score(y, y_pred_temporal, zero_division=0)
    except:
        baseline_roc = 0.0
        baseline_recall = 0.0
        baseline_precision = 0.0

    try:
        multiscale_roc = roc_auc_score(y, y_pred_proba_multiscale)
        multiscale_recall = recall_score(y, y_pred_multiscale)
        multiscale_precision = precision_score(y, y_pred_multiscale, zero_division=0)
    except:
        multiscale_roc = 0.0
        multiscale_recall = 0.0
        multiscale_precision = 0.0

    print(f"\nCross-Validation ROC-AUC:")
    print(f"  Baseline:    {temporal_cv:.3f}")
    print(f"  Multi-scale: {multiscale_cv:.3f}")
    print(f"  Improvement: {multiscale_cv - temporal_cv:+.3f}")

    print(f"\nFull Validation Performance:")
    print(f"  Metric        Baseline    Multi-scale    Change")
    print(f"  {'-'*50}")
    print(f"  ROC-AUC       {baseline_roc:.3f}       {multiscale_roc:.3f}          {multiscale_roc - baseline_roc:+.3f}")
    print(f"  Recall        {baseline_recall:.3f}       {multiscale_recall:.3f}          {multiscale_recall - baseline_recall:+.3f}")
    print(f"  Precision     {baseline_precision:.3f}       {multiscale_precision:.3f}          {multiscale_precision - baseline_precision:+.3f}")

    return {
        'set_name': set_name,
        'n_samples': len(samples),
        'n_clearing': n_clearing,
        'n_intact': n_intact,
        'baseline_roc': baseline_roc,
        'baseline_recall': baseline_recall,
        'baseline_precision': baseline_precision,
        'multiscale_roc': multiscale_roc,
        'multiscale_recall': multiscale_recall,
        'multiscale_precision': multiscale_precision,
        'cv_baseline': temporal_cv,
        'cv_multiscale': multiscale_cv,
    }


def main():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("=" * 80)
    print("COMPREHENSIVE MULTI-SCALE EVALUATION")
    print("=" * 80)

    validation_sets = [
        'rapid_response',
        'risk_ranking',
        'comprehensive',
        'edge_cases',
    ]

    results = []

    for set_name in validation_sets:
        val_file = processed_dir / f"hard_val_{set_name}_multiscale.pkl"

        if not val_file.exists():
            print(f"\n✗ {set_name}: File not found ({val_file.name})")
            print(f"  Run: uv run python src/walk/08_multiscale_embeddings.py --set {set_name}")
            continue

        with open(val_file, 'rb') as f:
            samples = pickle.load(f)

        result = evaluate_set(set_name, samples)
        results.append(result)

    # Summary table
    if len(results) > 0:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL VALIDATION SETS")
        print(f"{'='*80}")

        print(f"\n{'Set':<20} {'Samples':<10} {'Baseline ROC':<15} {'Multi-scale ROC':<15} {'Improvement'}")
        print(f"{'-'*80}")

        for r in results:
            improvement = r['multiscale_roc'] - r['baseline_roc']
            print(f"{r['set_name']:<20} {r['n_samples']:<10} {r['baseline_roc']:.3f}           {r['multiscale_roc']:.3f}             {improvement:+.3f}")

        # Aggregate statistics
        total_samples = sum(r['n_samples'] for r in results)
        avg_baseline = np.mean([r['baseline_roc'] for r in results])
        avg_multiscale = np.mean([r['multiscale_roc'] for r in results])
        avg_improvement = avg_multiscale - avg_baseline

        print(f"\n{'Average':<20} {total_samples:<10} {avg_baseline:.3f}           {avg_multiscale:.3f}             {avg_improvement:+.3f}")

        # Target achievement
        print(f"\n{'='*80}")
        print("TARGET ACHIEVEMENT")
        print(f"{'='*80}")

        target_recall = 0.80
        target_precision = 0.70

        for r in results:
            recall_met = '✓' if r['multiscale_recall'] >= target_recall else '✗'
            precision_met = '✓' if r['multiscale_precision'] >= target_precision else '✗'

            print(f"\n{r['set_name'].upper()}:")
            print(f"  Recall ≥ 80%:    {recall_met} ({r['multiscale_recall']:.1%})")
            print(f"  Precision ≥ 70%: {precision_met} ({r['multiscale_precision']:.1%})")

        # Overall conclusion
        all_met = all(
            r['multiscale_recall'] >= target_recall and r['multiscale_precision'] >= target_precision
            for r in results
        )

        print(f"\n{'='*80}")
        if all_met:
            print("✓ SUCCESS: All validation sets meet both targets!")
            print("\nMulti-scale features ready for production deployment.")
        else:
            some_met = any(
                r['multiscale_recall'] >= target_recall and r['multiscale_precision'] >= target_precision
                for r in results
            )
            if some_met:
                print("→ PARTIAL: Some validation sets meet targets.")
                print("\nNext steps: Extract multi-scale for training set and retrain.")
            else:
                print("⚠ TARGETS NOT MET: Further improvements needed.")


if __name__ == "__main__":
    main()
