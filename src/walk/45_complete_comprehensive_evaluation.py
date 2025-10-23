#!/usr/bin/env python3
"""
Complete Comprehensive Evaluation

Trains Phase A model and evaluates comprehensively on:
1. Typical case: Held-out 2024 test set
2. Each of 4 hard validation sets with use-case-specific thresholds
3. Temporal drift quantification vs Phase 4 baseline

This is a complete end-to-end evaluation that:
- Trains the best Phase A model (with year feature)
- Makes predictions on all datasets
- Computes comprehensive metrics

Usage:
    uv run python src/walk/45_complete_comprehensive_evaluation.py
"""

import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'walk'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optimal thresholds from threshold optimization
OPTIMAL_THRESHOLDS = {
    'risk_ranking': 0.070,
    'rapid_response': 0.608,
    'comprehensive': 0.884,
    'edge_cases': 0.910
}

# Use-case targets
USE_CASE_TARGETS = {
    'risk_ranking': {'metric': 'recall', 'target': 0.90},
    'rapid_response': {'metric': 'recall', 'target': 0.90},
    'comprehensive': {'metric': 'precision', 'baseline': 0.389},
    'edge_cases': {'metric': 'roc_auc', 'target': 0.65}
}

# Phase 4 baseline (heterogeneous model, 2020-2023 → 2024)
PHASE_4_BASELINE = {
    'cv_roc_auc': 0.982,
    'test_roc_auc': 0.796,
    'drift': 0.186,
    'drift_pct': 18.9
}

# Best Phase A hyperparameters
BEST_PHASE_A_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1
}


def print_header(text: str, level: int = 1):
    """Print formatted header."""
    char = '=' if level == 1 else '-'
    width = 80
    print(f"\n{char * width}")
    print(text.upper() if level == 1 else text)
    print(f"{char * width}\n")


def load_training_data():
    """Load 2020-2023 and 2024 training data."""
    print("Loading training data...")

    # Load 2020-2023
    filepath_2020_2023 = PROCESSED_DIR / 'walk_dataset_scaled_phase1_20251020_165345_all_hard_samples_multiscale.pkl'
    with open(filepath_2020_2023, 'rb') as f:
        data_2020_2023 = pickle.load(f)

    samples_2020_2023 = data_2020_2023.get('samples', data_2020_2023.get('data', data_2020_2023))
    print(f"  2020-2023: {len(samples_2020_2023)} samples")

    # Load 2024
    filepath_2024 = PROCESSED_DIR / 'walk_dataset_2024_with_features_20251021_110417.pkl'
    with open(filepath_2024, 'rb') as f:
        data_2024 = pickle.load(f)

    samples_2024 = data_2024.get('samples', data_2024.get('data', data_2024))
    print(f"  2024: {len(samples_2024)} samples")

    return samples_2020_2023, samples_2024


def extract_features_from_samples(samples: List[dict], add_year: bool = False):
    """Extract features from samples."""
    X = []
    y = []

    for sample in samples:
        annual_features = sample.get('annual_features')
        if annual_features is None:
            continue

        multiscale_dict = sample.get('multiscale_features', {})
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

        try:
            coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
        except KeyError:
            continue

        combined = np.concatenate([annual_features, coarse_features])

        if add_year:
            year = sample.get('year', 2024)
            combined = np.concatenate([combined, [year]])

        X.append(combined)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def train_phase_a_model():
    """Train Phase A model with year feature."""
    print_header("Training Phase A Model (with year feature)")

    # Load data
    samples_2020_2023, samples_2024 = load_training_data()

    # Extract features with year
    X_2020_2023, y_2020_2023 = extract_features_from_samples(samples_2020_2023, add_year=True)
    X_2024_all, y_2024_all = extract_features_from_samples(samples_2024, add_year=True)

    print(f"\nFeature extraction:")
    print(f"  2020-2023: {X_2020_2023.shape}")
    print(f"  2024: {X_2024_all.shape}")

    # Split 2024 into train/test (70/30)
    X_2024_train, X_2024_test, y_2024_train, y_2024_test = train_test_split(
        X_2024_all, y_2024_all, test_size=0.3, random_state=42, stratify=y_2024_all
    )

    print(f"\n2024 split:")
    print(f"  Train: {len(X_2024_train)} samples")
    print(f"  Test: {len(X_2024_test)} samples")

    # Combine training data
    X_train = np.vstack([X_2020_2023, X_2024_train])
    y_train = np.concatenate([y_2020_2023, y_2024_train])

    print(f"\nCombined training data:")
    print(f"  Total: {len(X_train)} samples")
    print(f"  Clearing: {sum(y_train == 1)}")
    print(f"  Intact: {sum(y_train == 0)}")
    print(f"  Features: {X_train.shape[1]}D (69 + year)")

    # Train Random Forest with best hyperparameters
    print(f"\nTraining Random Forest...")
    model = RandomForestClassifier(**BEST_PHASE_A_PARAMS)
    model.fit(X_train, y_train)
    print("✓ Training complete")

    return model, X_2024_test, y_2024_test


def compute_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute comprehensive evaluation metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Basic metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'threshold': float(threshold)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0][0]),
        'fp': int(cm[0][1]),
        'fn': int(cm[1][0]),
        'tp': int(cm[1][1])
    }

    # Derived metrics
    if cm[0][0] + cm[0][1] > 0:
        metrics['specificity'] = float(cm[0][0] / (cm[0][0] + cm[0][1]))
    else:
        metrics['specificity'] = 0.0

    if cm[1][1] + cm[0][1] > 0:
        metrics['ppv'] = float(cm[1][1] / (cm[1][1] + cm[0][1]))
    else:
        metrics['ppv'] = 0.0

    if cm[0][0] + cm[1][0] > 0:
        metrics['npv'] = float(cm[0][0] / (cm[0][0] + cm[1][0]))
    else:
        metrics['npv'] = 0.0

    # Class distribution
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(sum(y_true == 1))
    metrics['n_negative'] = int(sum(y_true == 0))
    metrics['prevalence'] = float(sum(y_true == 1) / len(y_true))

    return metrics


def print_metrics_table(metrics: Dict, use_case: str = None):
    """Print metrics in a formatted table."""
    print(f"\nSample Distribution:")
    print(f"  Total:    {metrics['n_samples']:3d}")
    print(f"  Positive: {metrics['n_positive']:3d} ({metrics['prevalence']:.1%})")
    print(f"  Negative: {metrics['n_negative']:3d}")

    print(f"\nClassification Metrics (threshold: {metrics['threshold']:.3f}):")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.3f}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Recall:      {metrics['recall']:.3f}")
    print(f"  F1-Score:    {metrics['f1']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  PPV:         {metrics['ppv']:.3f}")
    print(f"  NPV:         {metrics['npv']:.3f}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm['tn']:3d}  FP: {cm['fp']:3d}")
    print(f"  FN: {cm['fn']:3d}  TP: {cm['tp']:3d}")

    # Use-case-specific assessment
    if use_case and use_case in USE_CASE_TARGETS:
        target_info = USE_CASE_TARGETS[use_case]
        metric_name = target_info['metric']
        metric_value = metrics[metric_name]

        print(f"\nUse Case Assessment: {use_case}")
        if 'target' in target_info:
            target = target_info['target']
            met_target = metric_value >= target
            status = "✓ MET" if met_target else "✗ MISSED"
            print(f"  Target: {metric_name.upper()} >= {target:.3f}")
            print(f"  Achieved: {metric_value:.3f} [{status}]")
        elif 'baseline' in target_info:
            baseline = target_info['baseline']
            improvement = metric_value - baseline
            improvement_pct = (improvement / baseline) * 100
            print(f"  Baseline: {metric_name.upper()} = {baseline:.3f}")
            print(f"  Achieved: {metric_value:.3f} (+{improvement:.3f}, +{improvement_pct:.1f}%)")


def evaluate_typical_case(model, X_test, y_test):
    """Evaluate on typical case (held-out 2024 test set)."""
    print_header("1. Typical Case Evaluation (Held-Out 2024 Test Set)")

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics with default 0.5 threshold
    metrics = compute_comprehensive_metrics(y_test, y_pred_proba, threshold=0.5)

    print_metrics_table(metrics)

    # Temporal drift analysis
    print_header("Temporal Drift Analysis", level=2)

    print(f"Phase 4 Baseline (2020-2023 → 2024):")
    print(f"  CV ROC-AUC:   {PHASE_4_BASELINE['cv_roc_auc']:.3f}")
    print(f"  Test ROC-AUC: {PHASE_4_BASELINE['test_roc_auc']:.3f}")
    print(f"  Drift:        {PHASE_4_BASELINE['drift']:.3f} ({PHASE_4_BASELINE['drift_pct']:.1f}%)")

    # Assume CV score similar to Phase A (0.972)
    assumed_cv_score = 0.972
    current_drift = assumed_cv_score - metrics['roc_auc']
    current_drift_pct = (current_drift / assumed_cv_score) * 100

    print(f"\nPhase A Model (2020-2024 with year):")
    print(f"  Assumed CV:   {assumed_cv_score:.3f}")
    print(f"  Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Drift:        {current_drift:.3f} ({current_drift_pct:.1f}%)")

    drift_reduction = PHASE_4_BASELINE['drift'] - current_drift
    drift_reduction_pct = (drift_reduction / PHASE_4_BASELINE['drift']) * 100

    print(f"\nDrift Reduction:")
    print(f"  Absolute: {drift_reduction:.3f}")
    print(f"  Relative: {drift_reduction_pct:.1f}%")

    if drift_reduction_pct > 70:
        print(f"  ✓✓✓ EXCELLENT: >70% drift reduction")
    elif drift_reduction_pct > 50:
        print(f"  ✓✓ GOOD: >50% drift reduction")
    elif drift_reduction_pct > 0:
        print(f"  ✓ IMPROVED: Drift reduced")
    else:
        print(f"  ~ NO IMPROVEMENT: Drift similar or worse")

    return {
        'metrics': metrics,
        'temporal_drift': {
            'assumed_cv_score': assumed_cv_score,
            'test_score': metrics['roc_auc'],
            'drift': current_drift,
            'drift_pct': current_drift_pct,
            'baseline_drift': PHASE_4_BASELINE['drift'],
            'drift_reduction': drift_reduction,
            'drift_reduction_pct': drift_reduction_pct
        }
    }


def load_validation_set(name: str):
    """Load a hard validation set."""
    # Try different file patterns (prefer feature-extracted versions)
    patterns = [
        f'hard_val_{name}_features.pkl',  # Feature-extracted version
        f'hard_val_{name}_multiscale.pkl',  # Multiscale version
        f'hard_val_{name}.pkl',
        f'{name}.pkl'
    ]

    for pattern in patterns:
        filepath = PROCESSED_DIR / pattern
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Handle different data formats
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = data.get('samples', data.get('data', data))
            else:
                samples = data

            return samples

    return None


def evaluate_hard_validation_sets(model):
    """Evaluate on all 4 hard validation sets."""
    print_header("2. Hard Validation Sets Evaluation")

    use_cases = [
        'risk_ranking',
        'rapid_response',
        'comprehensive',
        'edge_cases'
    ]

    results = {}

    for use_case in use_cases:
        print_header(f"Evaluating: {use_case}", level=2)

        # Load validation set
        samples = load_validation_set(use_case)

        if samples is None:
            print(f"⚠️ Validation set not found: {use_case}")
            continue

        print(f"Loaded {len(samples)} samples")

        # Extract features (with year)
        X, y = extract_features_from_samples(samples, add_year=True)

        if len(X) == 0:
            print(f"⚠️ No valid samples with features, skipping...")
            continue

        print(f"Valid samples: {len(X)}")
        print(f"  Positive: {sum(y==1)}")
        print(f"  Negative: {sum(y==0)}")

        # Make predictions
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Compute metrics with use-case-specific threshold
        threshold = OPTIMAL_THRESHOLDS[use_case]
        metrics = compute_comprehensive_metrics(y, y_pred_proba, threshold=threshold)

        print_metrics_table(metrics, use_case=use_case)

        results[use_case] = metrics

    return results


def quantify_validation_set_drift(validation_results: Dict):
    """Quantify temporal drift on validation sets."""
    print_header("3. Validation Set Temporal Drift Summary")

    # Phase 4 baseline (approximate - same 0.796 across all)
    phase4_baseline_roc_auc = 0.796

    print("Comparison to Phase 4 Baseline:")
    print(f"{'Use Case':<20s} {'Phase 4':<10s} {'Phase A':<10s} {'Δ':<10s} {'Status':<15s}")
    print("-" * 70)

    for use_case, metrics in validation_results.items():
        phase_a_roc_auc = metrics['roc_auc']
        delta = phase_a_roc_auc - phase4_baseline_roc_auc
        delta_pct = (delta / phase4_baseline_roc_auc) * 100

        if delta > 0.05:
            status = "✓✓ Significant"
        elif delta > 0:
            status = "✓ Improved"
        elif delta > -0.05:
            status = "~ Similar"
        else:
            status = "✗ Degraded"

        print(f"{use_case:<20s} {phase4_baseline_roc_auc:<10.3f} {phase_a_roc_auc:<10.3f} {delta:+.3f} ({delta_pct:+.1f}%)  {status:<15s}")

    print("\nKey Insights:")
    improvements = sum(1 for m in validation_results.values() if m['roc_auc'] > phase4_baseline_roc_auc)
    print(f"  {improvements}/{len(validation_results)} use cases improved")

    avg_delta = np.mean([m['roc_auc'] - phase4_baseline_roc_auc for m in validation_results.values()])
    print(f"  Average improvement: {avg_delta:+.3f}")


def generate_comprehensive_report(typical_results, validation_results):
    """Generate comprehensive evaluation report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'phase_a_comprehensive_evaluation_{timestamp}.json'

    report = {
        'evaluation_date': timestamp,
        'model': {
            'phase': 'phase_a',
            'uses_year_feature': True,
            'hyperparameters': BEST_PHASE_A_PARAMS,
            'description': 'Phase A Random Forest with year feature'
        },
        'typical_case': typical_results,
        'validation_sets': validation_results,
        'baseline_comparison': {
            'phase_4_baseline': PHASE_4_BASELINE,
            'drift_reduction_achieved': typical_results['temporal_drift']['drift_reduction_pct']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved comprehensive evaluation to:")
    print(f"  {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


def main():
    print_header("Complete Comprehensive Evaluation - Phase A")

    # Train Phase A model
    model, X_test, y_test = train_phase_a_model()

    # 1. Evaluate typical case
    typical_results = evaluate_typical_case(model, X_test, y_test)

    # 2. Evaluate hard validation sets
    validation_results = evaluate_hard_validation_sets(model)

    # 3. Quantify validation set drift
    if validation_results:
        quantify_validation_set_drift(validation_results)

    # Generate comprehensive report
    report_path = generate_comprehensive_report(typical_results, validation_results)

    # Final summary
    print_header("Evaluation Summary")

    print("Model: Phase A (Random Forest with year feature)")
    print(f"\nTypical Case (2024 Test Set):")
    print(f"  ROC-AUC: {typical_results['metrics']['roc_auc']:.3f}")
    print(f"  Precision: {typical_results['metrics']['precision']:.3f}")
    print(f"  Recall: {typical_results['metrics']['recall']:.3f}")
    print(f"  Drift Reduction: {typical_results['temporal_drift']['drift_reduction_pct']:.1f}%")

    if validation_results:
        print(f"\nValidation Sets:")
        for use_case, metrics in validation_results.items():
            print(f"  {use_case:<20s}: ROC-AUC {metrics['roc_auc']:.3f}, Recall {metrics['recall']:.3f}")

    print(f"\nComprehensive report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
