"""
WALK Phase - Comprehensive Evaluation Framework

Evaluates baseline models on all validation sets with use-case-specific metrics:

1. Original Test Set: Standard metrics baseline
2. Rapid Response: Precision @ confidence thresholds, top-K precision
3. Risk Ranking: NDCG, ranking quality, calibration
4. Comprehensive: Area-weighted accuracy, size-stratified performance
5. Edge Cases: Per-category failure analysis

Usage:
    uv run python src/walk/03_evaluate_all_sets.py
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.utils import get_config


def load_dataset(dataset_path):
    """Load a validation dataset."""
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def extract_features(dataset, feature_type='baseline_delta'):
    """
    Extract features from dataset samples with dual-year temporal structure.

    Args:
        dataset: List of samples or dict with 'data' key
        feature_type: Feature combination to use:
            - 'baseline_only': Y-1 features (landscape susceptibility)
            - 'current_only': Y features (recent state, may have temporal ambiguity)
            - 'delta_only': Delta features (year-over-year change)
            - 'baseline_delta': Y-1 + delta (recommended - safe + change signal)
            - 'all': All three temporal views (baseline + current + delta)

    Returns:
        (X, y) feature matrix and labels
    """
    # Handle both list and dict formats
    if isinstance(dataset, dict):
        data = dataset['data']
    else:
        data = dataset

    X = []
    y = []

    for sample in data:
        if 'features' not in sample:
            # Skip samples without features (validation sets need feature extraction)
            continue

        features_dict = sample['features']
        label = sample.get('label', sample.get('stable', False))
        label = 0 if label is False or label == 0 else 1

        # Build feature vector based on feature_type
        features = []

        if feature_type in ['baseline_only', 'baseline_delta', 'all']:
            # Add baseline (Y-1) features
            baseline = features_dict['baseline']
            dist = baseline['distances']
            vel = baseline['velocities']
            acc = baseline['accelerations']
            features.extend([
                dist['Q1'], dist['Q2'], dist['Q3'], dist['Q4'],
                vel['Q1_Q2'], vel['Q2_Q3'], vel['Q3_Q4'],
                acc['Q1_Q2_Q3'], acc['Q2_Q3_Q4'],
                baseline['trend_consistency'],
            ])

        if feature_type in ['current_only', 'all']:
            # Add current (Y) features
            current = features_dict['current']
            dist = current['distances']
            vel = current['velocities']
            acc = current['accelerations']
            features.extend([
                dist['Q1'], dist['Q2'], dist['Q3'], dist['Q4'],
                vel['Q1_Q2'], vel['Q2_Q3'], vel['Q3_Q4'],
                acc['Q1_Q2_Q3'], acc['Q2_Q3_Q4'],
                current['trend_consistency'],
            ])

        if feature_type in ['delta_only', 'baseline_delta', 'all']:
            # Add delta (change) features
            delta = features_dict['delta']
            features.extend([
                delta['delta_magnitudes']['Q1'],
                delta['delta_magnitudes']['Q2'],
                delta['delta_magnitudes']['Q3'],
                delta['delta_magnitudes']['Q4'],
                delta['mean_delta_magnitude'],
                delta['max_delta_magnitude'],
                delta['delta_trend'],
            ])

        if not features:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def compute_standard_metrics(y_true, y_pred_proba):
    """Compute standard classification metrics."""
    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
    except:
        pr_auc = 0.5

    y_pred = (y_pred_proba >= 0.5).astype(int)

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'precision': float(precision_score),
        'recall': float(recall_score),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
    }


def evaluate_rapid_response(y_true, y_pred_proba, thresholds=[0.7, 0.8, 0.9]):
    """
    Evaluate for rapid response use case.

    Focus: High precision at different confidence levels, top-K precision.
    """
    results = compute_standard_metrics(y_true, y_pred_proba)

    # Precision/Recall at different confidence thresholds
    threshold_metrics = {}
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        n_alerts = int(tp + fp)

        threshold_metrics[f'threshold_{threshold}'] = {
            'precision': float(precision),
            'recall': float(recall),
            'n_alerts': n_alerts,
        }

    results['confidence_thresholds'] = threshold_metrics

    # Top-K precision (for patrol prioritization)
    top_k_metrics = {}
    for k in [5, 10, 20]:
        if len(y_pred_proba) < k:
            continue

        # Get indices of top-K predictions
        top_k_idx = np.argsort(y_pred_proba)[-k:]
        y_true_top_k = y_true[top_k_idx]

        precision_at_k = y_true_top_k.sum() / k
        top_k_metrics[f'top_{k}'] = {
            'precision': float(precision_at_k),
            'n_clearing': int(y_true_top_k.sum()),
        }

    results['top_k'] = top_k_metrics

    return results


def evaluate_risk_ranking(y_true, y_pred_proba, risk_levels=None):
    """
    Evaluate for risk ranking use case.

    Focus: Ranking quality (NDCG), calibration, risk stratification.
    """
    results = compute_standard_metrics(y_true, y_pred_proba)

    # NDCG (Normalized Discounted Cumulative Gain)
    # Sort predictions by score (descending)
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    sorted_relevance = y_true[sorted_idx]

    # DCG@K for different K
    ndcg_metrics = {}
    for k in [5, 10, 20]:
        if len(sorted_relevance) < k:
            continue

        # DCG = sum(rel_i / log2(i+2)) for i in range(k)
        dcg = np.sum(sorted_relevance[:k] / np.log2(np.arange(2, k+2)))

        # IDCG (perfect ranking)
        ideal_relevance = np.sort(y_true)[::-1][:k]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k+2)))

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_metrics[f'ndcg_at_{k}'] = float(ndcg)

    results['ndcg'] = ndcg_metrics

    # Calibration: Do predicted probabilities match actual frequencies?
    # Bin predictions and compare predicted vs actual positive rates
    n_bins = 5
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            predicted_prob = y_pred_proba[mask].mean()
            actual_prob = y_true[mask].mean()
            calibration.append({
                'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                'predicted': float(predicted_prob),
                'actual': float(actual_prob),
                'n_samples': int(mask.sum()),
            })

    results['calibration'] = calibration

    # Risk level performance (if provided)
    if risk_levels is not None:
        risk_performance = {}
        for risk_level in ['high', 'medium', 'low']:
            mask = np.array([r == risk_level for r in risk_levels])
            if mask.sum() > 0:
                risk_metrics = compute_standard_metrics(y_true[mask], y_pred_proba[mask])
                risk_performance[risk_level] = risk_metrics
        results['risk_stratified'] = risk_performance

    return results


def evaluate_comprehensive(y_true, y_pred_proba, size_classes=None):
    """
    Evaluate for comprehensive monitoring use case.

    Focus: Size-stratified performance, balanced metrics.
    """
    results = compute_standard_metrics(y_true, y_pred_proba)

    # Size-stratified performance (if provided)
    if size_classes is not None:
        size_performance = {}
        for size_class in ['small', 'medium', 'large']:
            mask = np.array([s == size_class for s in size_classes])
            if mask.sum() > 0:
                size_metrics = compute_standard_metrics(y_true[mask], y_pred_proba[mask])
                size_performance[size_class] = {
                    'n_samples': int(mask.sum()),
                    'metrics': size_metrics,
                }
        results['size_stratified'] = size_performance

    # Balanced accuracy (equal weight to positive and negative classes)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
        else:
            balanced_acc = 0
    else:
        balanced_acc = 0

    results['balanced_accuracy'] = float(balanced_acc)

    return results


def evaluate_edge_cases(y_true, y_pred_proba, challenge_types):
    """
    Evaluate for edge cases use case.

    Focus: Per-category performance, failure modes.
    """
    results = compute_standard_metrics(y_true, y_pred_proba)

    # Per-challenge-type performance
    type_performance = {}
    for challenge_type in set(challenge_types):
        mask = np.array([c == challenge_type for c in challenge_types])
        if mask.sum() > 0:
            type_metrics = compute_standard_metrics(y_true[mask], y_pred_proba[mask])
            type_performance[challenge_type] = {
                'n_samples': int(mask.sum()),
                'metrics': type_metrics,
            }

    results['challenge_types'] = type_performance

    # Identify hardest cases (highest error rate)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    errors = (y_pred != y_true)
    error_by_type = {}

    for challenge_type in set(challenge_types):
        mask = np.array([c == challenge_type for c in challenge_types])
        if mask.sum() > 0:
            error_rate = errors[mask].mean()
            error_by_type[challenge_type] = float(error_rate)

    # Sort by error rate (descending)
    sorted_types = sorted(error_by_type.items(), key=lambda x: x[1], reverse=True)
    results['hardest_types'] = [{'type': t, 'error_rate': e} for t, e in sorted_types]

    return results


def train_baseline_models(config):
    """
    Train baseline models with different feature combinations.

    Tests multiple temporal views:
    1. baseline_only: Y-1 features (landscape susceptibility)
    2. current_only: Y features (recent state)
    3. delta_only: Delta features (year-over-year change)
    4. baseline_delta: Y-1 + delta (recommended)
    5. all: All three temporal views
    """
    print("=" * 80)
    print("TRAINING BASELINE MODELS (DUAL-YEAR TEMPORAL CONTROL)")
    print("=" * 80)
    print()

    # Load main dataset
    data_dir = config.get_path("paths.data_dir")
    dataset_path = data_dir / "processed" / "walk_dataset.pkl"

    dataset = load_dataset(dataset_path)

    # Get training data
    train_idx = dataset['splits']['train']

    # Get labels
    _, y_full = extract_features(dataset, feature_type='baseline_only')
    y_train = y_full[train_idx]

    print(f"Training on {len(y_train)} samples")
    print(f"  Clearing: {int(y_train.sum())}")
    print(f"  Intact: {int((~y_train.astype(bool)).sum())}")
    print()

    # Train models with different feature combinations
    feature_types = ['baseline_only', 'current_only', 'delta_only', 'baseline_delta', 'all']
    models = {}

    for feature_type in feature_types:
        print(f"Training Logistic Regression ({feature_type})...")

        # Extract features
        X_full, _ = extract_features(dataset, feature_type=feature_type)
        X_train = X_full[train_idx]

        print(f"  Feature dimensions: {X_train.shape[1]}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)

        models[feature_type] = {
            'model': lr_model,
            'scaler': scaler,
            'feature_type': feature_type,
        }

        print(f"  ✓ Trained\n")

    return models


def evaluate_on_validation_set(models, val_samples, set_name, set_type):
    """
    Evaluate models on a validation set with all feature combinations.

    Args:
        models: Dict of trained models (one per feature_type)
        val_samples: List of validation samples
        set_name: Name of validation set
        set_type: Type for use-case-specific evaluation

    Returns:
        Dict with evaluation results for each feature combination
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ON: {set_name}")
    print(f"{'='*80}\n")

    print(f"Total samples: {len(val_samples)}")
    n_clearing = sum(1 for s in val_samples if not s.get('stable', False))
    n_intact = len(val_samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}\n")

    # Check if samples have features
    has_features = all('features' in s for s in val_samples)

    if not has_features:
        print("⚠️  Note: Validation set needs feature extraction")
        print("    Run: uv run python src/walk/01c_extract_features_for_hard_sets.py\n")
        return {
            'set_name': set_name,
            'n_samples': len(val_samples),
            'n_clearing': n_clearing,
            'n_intact': n_intact,
            'status': 'pending_feature_extraction',
        }

    # Evaluate each feature combination
    all_results = {}

    for feature_type, model_info in models.items():
        print(f"Evaluating {feature_type}...")

        # Extract features for this combination
        X, y = extract_features(val_samples, feature_type=feature_type)

        if len(X) == 0:
            print(f"  ⚠️  No valid samples\n")
            continue

        # Get predictions
        scaler = model_info['scaler']
        model = model_info['model']

        X_scaled = scaler.transform(X)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        # Compute use-case-specific metrics
        if set_type == 'rapid_response':
            results = evaluate_rapid_response(y, y_pred_proba)
        elif set_type == 'risk_ranking':
            risk_levels = [s.get('risk_level') for s in val_samples if 'features' in s]
            results = evaluate_risk_ranking(y, y_pred_proba, risk_levels if any(risk_levels) else None)
        elif set_type == 'comprehensive':
            size_classes = [s.get('size_class') for s in val_samples if 'features' in s]
            results = evaluate_comprehensive(y, y_pred_proba, size_classes if any(size_classes) else None)
        elif set_type == 'edge_cases':
            challenge_types = [s.get('challenge_type', 'unknown') for s in val_samples if 'features' in s]
            results = evaluate_edge_cases(y, y_pred_proba, challenge_types)
        else:
            results = compute_standard_metrics(y, y_pred_proba)

        # Print summary
        print(f"  ROC-AUC: {results['roc_auc']:.3f} | PR-AUC: {results['pr_auc']:.3f} | "
              f"Acc: {results['accuracy']:.3f} | Prec: {results['precision']:.3f} | "
              f"Rec: {results['recall']:.3f}\n")

        all_results[feature_type] = results

    return {
        'set_name': set_name,
        'n_samples': len(val_samples),
        'n_clearing': n_clearing,
        'n_intact': n_intact,
        'status': 'evaluated',
        'feature_combinations': all_results,
    }


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION FRAMEWORK")
    print("=" * 80)
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    # Train baseline models
    models = train_baseline_models(config)

    # Load main dataset for original test set evaluation
    print("\n" + "=" * 80)
    print("EVALUATING ON ORIGINAL TEST SET")
    print("=" * 80 + "\n")

    dataset = load_dataset(processed_dir / "walk_dataset.pkl")
    test_idx = dataset['splits']['test']

    # Evaluate all feature combinations on test set
    test_results = {}

    for feature_type, model_info in models.items():
        print(f"Evaluating {feature_type}...")

        # Extract test features
        X_full, y_full = extract_features(dataset, feature_type=feature_type)
        X_test = X_full[test_idx]
        y_test = y_full[test_idx]

        # Get predictions
        X_test_scaled = model_info['scaler'].transform(X_test)
        y_pred_proba = model_info['model'].predict_proba(X_test_scaled)[:, 1]

        results = compute_standard_metrics(y_test, y_pred_proba)

        print(f"  ROC-AUC: {results['roc_auc']:.3f} | PR-AUC: {results['pr_auc']:.3f} | "
              f"Acc: {results['accuracy']:.3f} | Prec: {results['precision']:.3f} | "
              f"Rec: {results['recall']:.3f}\n")

        test_results[feature_type] = results

    # Load and evaluate on hard validation sets
    validation_sets = [
        ('hard_val_rapid_response', 'rapid_response'),
        ('hard_val_risk_ranking', 'risk_ranking'),
        ('hard_val_comprehensive', 'comprehensive'),
        ('hard_val_edge_cases', 'edge_cases'),
    ]

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'original_test': test_results,
        'hard_validation_sets': {},
    }

    for base_name, set_type in validation_sets:
        # Check for enriched version first (*_features.pkl), fallback to original
        enriched_path = processed_dir / f"{base_name}_features.pkl"
        original_path = processed_dir / f"{base_name}.pkl"

        if enriched_path.exists():
            val_path = enriched_path
            print(f"Using enriched dataset: {enriched_path.name}")
        elif original_path.exists():
            val_path = original_path
        else:
            print(f"⚠️  {base_name} not found, skipping")
            continue

        val_samples = load_dataset(val_path)
        results = evaluate_on_validation_set(models, val_samples, base_name, set_type)
        all_results['hard_validation_sets'][set_type] = results

    # Save results
    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "evaluation_all_sets.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Results saved to {output_file}")
    print("\nNext Steps:")
    print("  1. Extract features for hard validation sets using data preparation pipeline")
    print("  2. Re-run this evaluation with full feature extraction")
    print("  3. Analyze use-case-specific metrics for each validation set")
    print("=" * 80)


if __name__ == "__main__":
    main()
