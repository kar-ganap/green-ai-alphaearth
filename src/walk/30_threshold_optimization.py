#!/usr/bin/env python3
"""
Threshold Optimization for Use-Case-Specific Performance

Current problem:
- Using default threshold (0.5) for all use cases
- edge_cases: PR-AUC=0.716 but ROC-AUC=0.617 (threshold suboptimal!)
- risk_ranking: Recall=0.875, only 0.025 short (one FN away)
- comprehensive: Precision=0.389 (over-predicting)

Solution:
- Optimize decision threshold per use case
- rapid_response: Maximize precision subject to recall ≥ 0.90
- risk_ranking: Maximize precision subject to recall ≥ 0.90
- comprehensive: Maximize precision at acceptable recall (≥0.50)
- edge_cases: Maximize F1-score (balanced)

Expected impact:
- edge_cases: 0.617 → 0.65+ ROC-AUC
- risk_ranking: 0.875 → 0.90 recall
- comprehensive: Precision improvement
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix, accuracy_score,
    average_precision_score
)
import json
from datetime import datetime

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient
from src.walk.diagnostic_helpers import extract_dual_year_features


def load_model_and_data():
    """Load the trained model and validation sets."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load model (saved as dict with 'model', 'scaler', etc.)
    model_path = processed_dir / 'walk_model_rf_all_hard_samples.pkl'
    print(f"Loading model: {model_path.name}")
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)

    # Extract model and scaler from dict
    model = model_dict['model']
    scaler = model_dict['scaler']

    # Load validation sets
    validation_sets = {}
    for set_name in ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']:
        # Try multiple file patterns
        patterns = [
            f'walk_dataset_scaled_phase1_*_{set_name}_multiscale.pkl',
            f'hard_val_{set_name}_multiscale.pkl',
            f'hard_val_{set_name}.pkl',
        ]

        found = False
        for pattern in patterns:
            files = list(processed_dir.glob(pattern))
            if files:
                path = sorted(files)[-1]
                print(f"Loading {set_name}: {path.name}")
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'data' in data:
                        validation_sets[set_name] = data['data']
                    else:
                        validation_sets[set_name] = data
                found = True
                break

        if not found:
            print(f"  ⚠️ Warning: No file found for {set_name}")

    return model, scaler, validation_sets


def extract_features_from_samples(samples, ee_client):
    """
    Extract feature matrix from samples (same logic as training script).

    Extracts 69D features:
    - 3D annual features (delta_1yr, delta_2yr, acceleration)
    - 66D coarse landscape features (64 embeddings + 2 stats)
    """
    X = []
    y = []

    for sample in samples:
        # Fix missing 'year' field for intact samples
        if 'year' not in sample and sample.get('stable', False):
            sample = sample.copy()
            sample['year'] = 2021

        # Extract annual features using diagnostic_helpers
        try:
            annual_features = extract_dual_year_features(ee_client, sample)
        except:
            annual_features = None

        if annual_features is None:
            continue

        # Get coarse features from multiscale_features dict
        if 'multiscale_features' not in sample:
            continue

        multiscale_dict = sample['multiscale_features']
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

        missing_features = [k for k in coarse_feature_names if k not in multiscale_dict]
        if missing_features:
            continue

        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])

        # Combine: 3 annual + 66 coarse = 69 features
        combined = np.concatenate([annual_features, coarse_features])

        if len(combined) != 69:
            continue

        X.append(combined)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def find_optimal_threshold_recall_constrained(y_true, y_pred_proba, min_recall=0.90):
    """
    Find threshold that maximizes precision subject to recall >= min_recall.

    Used for: rapid_response, risk_ranking
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find thresholds that achieve minimum recall
    valid_indices = recall >= min_recall

    if not np.any(valid_indices):
        # Can't achieve target recall at any threshold
        return {
            'optimal_threshold': 0.0,
            'achieves_target': False,
            'precision': 0.0,
            'recall': 0.0,
            'reason': f'Cannot achieve recall >= {min_recall}'
        }

    # Among valid thresholds, find the one with maximum precision
    valid_precisions = precision[valid_indices]
    valid_recalls = recall[valid_indices]
    valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds is 1 shorter

    if len(valid_thresholds) == 0:
        # Edge case: only threshold=0 works
        return {
            'optimal_threshold': 0.0,
            'achieves_target': True,
            'precision': valid_precisions[-1],
            'recall': valid_recalls[-1]
        }

    best_idx = np.argmax(valid_precisions[:-1])  # Last element is for threshold=0

    return {
        'optimal_threshold': float(valid_thresholds[best_idx]),
        'achieves_target': True,
        'precision': float(valid_precisions[best_idx]),
        'recall': float(valid_recalls[best_idx])
    }


def find_optimal_threshold_precision_focus(y_true, y_pred_proba, min_recall=0.50):
    """
    Find threshold that maximizes precision subject to recall >= min_recall.

    Used for: comprehensive (minimize false positives)
    """
    return find_optimal_threshold_recall_constrained(y_true, y_pred_proba, min_recall)


def find_optimal_threshold_f1(y_true, y_pred_proba):
    """
    Find threshold that maximizes F1-score.

    Used for: edge_cases (balanced performance)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 for each threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find threshold with maximum F1
    best_idx = np.argmax(f1_scores[:-1])  # Last element is for threshold=0

    if len(thresholds) == 0:
        return {
            'optimal_threshold': 0.5,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'f1': float(f1_scores[best_idx]),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx])
    }


def evaluate_with_threshold(y_true, y_pred_proba, threshold):
    """Evaluate performance at a specific threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        pr_auc = float('nan')

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'threshold': threshold,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'f2': f2,
        'f05': f05,
        'confusion_matrix': cm.tolist(),
        'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
    }


def optimize_thresholds(model, scaler, validation_sets):
    """
    Optimize decision thresholds for each use case.

    Returns:
        Dict with optimal thresholds and performance metrics
    """
    results = {}

    # Initialize EE client for feature extraction
    ee_client = EarthEngineClient(use_cache=True)

    use_case_configs = {
        'rapid_response': {
            'method': 'recall_constrained',
            'min_recall': 0.90,
            'target_metric': 'recall',
            'target_value': 0.90
        },
        'risk_ranking': {
            'method': 'recall_constrained',
            'min_recall': 0.90,
            'target_metric': 'recall',
            'target_value': 0.90
        },
        'comprehensive': {
            'method': 'precision_focus',
            'min_recall': 0.50,
            'target_metric': 'precision',
            'baseline_value': 0.389
        },
        'edge_cases': {
            'method': 'f1_optimization',
            'target_metric': 'roc_auc',
            'target_value': 0.65
        }
    }

    for set_name, samples in validation_sets.items():
        print(f"\n{'='*80}")
        print(f"OPTIMIZING THRESHOLD: {set_name}")
        print('='*80)

        config = use_case_configs[set_name]

        # Extract features
        X, y = extract_features_from_samples(samples, ee_client)

        if len(X) == 0:
            print(f"  ⚠️ No valid samples after feature extraction, skipping")
            continue

        print(f"  Extracted features for {len(X)} samples")

        # Scale features and get predictions
        X_scaled = scaler.transform(X)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        # Baseline: default threshold 0.5
        baseline = evaluate_with_threshold(y, y_pred_proba, 0.5)

        print(f"\nBASELINE (threshold=0.5):")
        print(f"  Precision: {baseline['precision']:.3f}")
        print(f"  Recall:    {baseline['recall']:.3f}")
        print(f"  F1:        {baseline['f1']:.3f}")
        print(f"  ROC-AUC:   {baseline['roc_auc']:.3f}")
        print(f"  PR-AUC:    {baseline['pr_auc']:.3f}")

        # Find optimal threshold
        if config['method'] == 'recall_constrained':
            optimization = find_optimal_threshold_recall_constrained(
                y, y_pred_proba, config['min_recall']
            )
            optimal_threshold = optimization['optimal_threshold']

        elif config['method'] == 'precision_focus':
            optimization = find_optimal_threshold_precision_focus(
                y, y_pred_proba, config['min_recall']
            )
            optimal_threshold = optimization['optimal_threshold']

        elif config['method'] == 'f1_optimization':
            optimization = find_optimal_threshold_f1(y, y_pred_proba)
            optimal_threshold = optimization['optimal_threshold']

        # Evaluate at optimal threshold
        optimized = evaluate_with_threshold(y, y_pred_proba, optimal_threshold)

        print(f"\nOPTIMIZED (threshold={optimal_threshold:.3f}):")
        print(f"  Precision: {optimized['precision']:.3f} (Δ={optimized['precision']-baseline['precision']:+.3f})")
        print(f"  Recall:    {optimized['recall']:.3f} (Δ={optimized['recall']-baseline['recall']:+.3f})")
        print(f"  F1:        {optimized['f1']:.3f} (Δ={optimized['f1']-baseline['f1']:+.3f})")
        print(f"  F2:        {optimized['f2']:.3f} (Δ={optimized['f2']-baseline['f2']:+.3f})")
        print(f"  ROC-AUC:   {optimized['roc_auc']:.3f} (unchanged)")
        print(f"  PR-AUC:    {optimized['pr_auc']:.3f} (unchanged)")

        # Check if target is met
        target_met = False
        if config['target_metric'] == 'recall':
            target_met = optimized['recall'] >= config['target_value']
            print(f"\nTARGET: Recall ≥ {config['target_value']}")
            if target_met:
                print(f"  Status: ✓ MET")
            else:
                gap = config['target_value'] - optimized['recall']
                print(f"  Status: ✗ NOT MET ({gap:.3f} short)")

        elif config['target_metric'] == 'precision':
            improvement = optimized['precision'] - config['baseline_value']
            target_met = improvement > 0.01
            print(f"\nTARGET: Precision improvement")
            print(f"  Baseline: {config['baseline_value']:.3f}")
            print(f"  Current:  {optimized['precision']:.3f}")
            print(f"  Change:   {improvement:+.3f}")
            print(f"  Status:   {'✓ IMPROVED' if target_met else '✗ NO IMPROVEMENT'}")

        elif config['target_metric'] == 'roc_auc':
            target_met = optimized['roc_auc'] >= config['target_value']
            print(f"\nTARGET: ROC-AUC ≥ {config['target_value']}")
            if target_met:
                print(f"  Status: ✓ MET")
            else:
                gap = config['target_value'] - optimized['roc_auc']
                print(f"  Status: ✗ NOT MET ({gap:.3f} short)")

        # Confusion matrix comparison
        print(f"\nCONFUSION MATRIX:")
        print(f"  Baseline  (t=0.5):    TN={baseline['tn']:2d} FP={baseline['fp']:2d} | FN={baseline['fn']:2d} TP={baseline['tp']:2d}")
        print(f"  Optimized (t={optimal_threshold:.3f}): TN={optimized['tn']:2d} FP={optimized['fp']:2d} | FN={optimized['fn']:2d} TP={optimized['tp']:2d}")

        # Store results
        results[set_name] = {
            'config': config,
            'baseline': baseline,
            'optimization_result': optimization,
            'optimized': optimized,
            'target_met': bool(target_met),  # Convert numpy bool to Python bool
            'improvement': {
                'precision': float(optimized['precision'] - baseline['precision']),
                'recall': float(optimized['recall'] - baseline['recall']),
                'f1': float(optimized['f1'] - baseline['f1']),
            }
        }

    return results


def print_summary(results):
    """Print summary comparison of before/after threshold optimization."""
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("="*80)

    print("\nPERFORMANCE BEFORE (threshold=0.5) vs AFTER (optimized):\n")

    # Table header
    print(f"{'Use Case':<20} {'Metric':<12} {'Before':<10} {'After':<10} {'Change':<10} {'Target':<10}")
    print("-" * 80)

    for set_name, result in results.items():
        baseline = result['baseline']
        optimized = result['optimized']
        config = result['config']

        if config['target_metric'] == 'recall':
            metric = 'Recall'
            before = baseline['recall']
            after = optimized['recall']
            target_str = f"≥{config['target_value']:.2f}"

        elif config['target_metric'] == 'precision':
            metric = 'Precision'
            before = baseline['precision']
            after = optimized['precision']
            target_str = "improve"

        elif config['target_metric'] == 'roc_auc':
            metric = 'F1-Score'
            before = baseline['f1']
            after = optimized['f1']
            target_str = f"≥{config['target_value']:.2f}"

        change = after - before
        status = '✓' if result['target_met'] else '✗'

        print(f"{set_name:<20} {metric:<12} {before:<10.3f} {after:<10.3f} {change:+10.3f} {target_str:<10} {status}")

    # Overall summary
    targets_met = sum(1 for r in results.values() if r['target_met'])
    print(f"\n{'='*80}")
    print(f"OVERALL: {targets_met}/{len(results)} use-case targets met")
    print("="*80)

    if targets_met == len(results):
        print("\n🎉 ALL TARGETS MET! Ready to proceed to model diversity experiments.")
    elif targets_met >= len(results) * 0.75:
        print(f"\n✓ {targets_met}/{len(results)} targets met. Good progress with threshold tuning alone.")
    else:
        print(f"\n⚠️ Only {targets_met}/{len(results)} targets met. Will need model diversity to close the gap.")


def save_results(results):
    """Save threshold optimization results."""
    config = get_config()
    results_dir = config.get_path("paths.results_dir") / 'walk'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare for JSON serialization
    output = {
        'timestamp': datetime.now().isoformat(),
        'use_cases': {}
    }

    for set_name, result in results.items():
        output['use_cases'][set_name] = {
            'target_metric': result['config']['target_metric'],
            'optimal_threshold': result['optimized']['threshold'],
            'baseline_performance': {
                'threshold': 0.5,
                'precision': result['baseline']['precision'],
                'recall': result['baseline']['recall'],
                'f1': result['baseline']['f1'],
                'roc_auc': result['baseline']['roc_auc'],
                'pr_auc': result['baseline']['pr_auc'],
            },
            'optimized_performance': {
                'threshold': result['optimized']['threshold'],
                'precision': result['optimized']['precision'],
                'recall': result['optimized']['recall'],
                'f1': result['optimized']['f1'],
                'f2': result['optimized']['f2'],
                'f05': result['optimized']['f05'],
                'roc_auc': result['optimized']['roc_auc'],
                'pr_auc': result['optimized']['pr_auc'],
            },
            'improvement': result['improvement'],
            'target_met': result['target_met'],
            'confusion_matrix_baseline': result['baseline']['confusion_matrix'],
            'confusion_matrix_optimized': result['optimized']['confusion_matrix'],
        }

    output_path = results_dir / 'threshold_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Also save optimal thresholds for easy reference
    thresholds = {
        set_name: result['optimized']['threshold']
        for set_name, result in results.items()
    }

    threshold_path = results_dir / 'optimal_thresholds.json'
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"✓ Optimal thresholds saved to: {threshold_path}")


def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR USE-CASE-SPECIFIC PERFORMANCE")
    print("="*80)
    print("\nGoal: Optimize decision thresholds to maximize use-case-specific metrics")
    print("\nUse-case targets:")
    print("  rapid_response:  Recall ≥ 0.90 (minimize missed clearings)")
    print("  risk_ranking:    Recall ≥ 0.90 (minimize missed high-risk areas)")
    print("  comprehensive:   Precision improvement (minimize false alarms)")
    print("  edge_cases:      Maximize F1 (balanced performance)")
    print()

    # Load model and data
    model, scaler, validation_sets = load_model_and_data()

    print(f"\nValidation sets loaded:")
    for name, samples in validation_sets.items():
        print(f"  {name}: {len(samples)} samples")

    # Optimize thresholds
    results = optimize_thresholds(model, scaler, validation_sets)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    targets_met = sum(1 for r in results.values() if r['target_met'])

    if targets_met == len(results):
        print("\n✓ All targets met with threshold optimization!")
        print("  → Proceed to temporal generalization test")
        print("  → Then move to RUN phase (production system)")
    else:
        print(f"\n⚠️ {targets_met}/{len(results)} targets met")
        print("  → Proceed to model diversity experiments")
        print("  → Try use-case-specific models or XGBoost")
        print("  → Ensemble methods may help close remaining gaps")


if __name__ == '__main__':
    main()
