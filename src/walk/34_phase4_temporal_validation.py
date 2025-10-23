#!/usr/bin/env python3
"""
Phase 4 Temporal Validation: 2020-2023 → 2024

Final phase of progressive temporal validation:
- Train on 2020-2023 data (685 samples)
- Test on 2024 data (162 samples)
- Evaluate with all 4 use case thresholds
- Compare to Phases 1-3 to check for temporal drift

This validates 1-year future prediction on the most recent data,
completing the temporal validation sequence before production deployment.

Usage:
    uv run python src/walk/34_phase4_temporal_validation.py
    uv run python src/walk/34_phase4_temporal_validation.py --use-case comprehensive
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix
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


def print_header(text: str, level: int = 1):
    """Print formatted header."""
    char = '=' if level == 1 else '-'
    width = 80
    print(f"\n{char * width}")
    print(text.upper() if level == 1 else text)
    print(f"{char * width}\n")


def load_training_data_2020_2023():
    """Load 2020-2023 training data (685 samples)."""
    print("Loading 2020-2023 training data...")

    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find training data matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    if 'data' in data:
        samples = data['data']
    elif 'samples' in data:
        samples = data['samples']
    else:
        samples = data

    print(f"  Loaded {len(samples)} samples")

    # Verify year distribution
    by_year = {}
    for sample in samples:
        year = sample.get('year', 2021)
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(sample)

    print("\n  Samples by year:")
    for year in sorted(by_year.keys()):
        n_clearing = sum(1 for s in by_year[year] if s.get('label', 0) == 1)
        n_intact = sum(1 for s in by_year[year] if s.get('label', 0) == 0)
        print(f"    {year}: {len(by_year[year])} total ({n_clearing} clearing, {n_intact} intact)")

    return samples


def load_2024_test_data():
    """Load 2024 test data (162 samples with features)."""
    print("\nLoading 2024 test data...")

    pattern = 'walk_dataset_2024_with_features_*.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find 2024 data matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('samples', data.get('data', data))
    print(f"  Loaded {len(samples)} samples")

    n_clearing = sum(1 for s in samples if s.get('label', 0) == 1)
    n_intact = sum(1 for s in samples if s.get('label', 0) == 0)
    print(f"    Clearing: {n_clearing}")
    print(f"    Intact: {n_intact}")

    return samples


def extract_features_from_samples(samples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 69D features (3 annual + 66 coarse multiscale) from samples.

    Returns:
        X: Feature matrix (n_samples, 69)
        y: Labels (n_samples,)
    """
    X = []
    y = []

    for sample in samples:
        # Annual features (3D)
        annual_features = sample.get('annual_features')
        if annual_features is None:
            continue

        # Multiscale features (66D coarse)
        multiscale_dict = sample.get('multiscale_features', {})
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

        try:
            coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
        except KeyError:
            continue

        # Combine: 3 annual + 66 coarse = 69 features
        combined = np.concatenate([annual_features, coarse_features])

        X.append(combined)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """Train Random Forest with hyperparameter tuning."""
    print_header("Training model on 2020-2023", level=2)

    print(f"Training samples: {len(X_train)}")
    print(f"  Clearing: {sum(y_train == 1)}")
    print(f"  Intact: {sum(y_train == 0)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Hyperparameter grid (same as Phases 1-3)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nSearching {total_combinations} hyperparameter combinations...")

    # GridSearchCV
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_scaled, y_train)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in sorted(grid_search.best_params_.items()):
        print(f"  {param}: {value}")

    return {
        'model': grid_search.best_estimator_,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'training_years': [2020, 2021, 2022, 2023],
        'n_train_samples': len(X_train)
    }


def evaluate_model(model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray,
                   use_case: str = 'edge_cases') -> Dict:
    """Evaluate model on 2024 test set."""
    print_header(f"Evaluating on 2024 (use_case: {use_case})", level=2)

    model = model_dict['model']
    scaler = model_dict['scaler']
    threshold = OPTIMAL_THRESHOLDS[use_case]

    # Scale test features
    X_scaled = scaler.transform(X_test)

    # Predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'threshold': threshold,
        'test_year': 2024,
        'use_case': use_case
    }

    # Print results
    print(f"\nTest samples: {len(y_test)}")
    print(f"  Clearing: {sum(y_test == 1)}")
    print(f"  Intact: {sum(y_test == 0)}")
    print(f"\nResults (threshold={threshold:.3f}):")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.3f}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0][0]:3d}  FP: {cm[0][1]:3d}")
    print(f"    FN: {cm[1][0]:3d}  TP: {cm[1][1]:3d}")

    # Target assessment
    target_info = USE_CASE_TARGETS[use_case]
    metric_name = target_info['metric']
    metric_value = metrics[metric_name]

    if 'target' in target_info:
        target = target_info['target']
        met_target = metric_value >= target
        status = "✓ MET" if met_target else "✗ MISSED"
        print(f"\n  Target: {metric_name.upper()} >= {target:.3f}")
        print(f"  Achieved: {metric_value:.3f} [{status}]")
    elif 'baseline' in target_info:
        baseline = target_info['baseline']
        improvement = metric_value - baseline
        print(f"\n  Baseline: {metric_name.upper()} = {baseline:.3f}")
        print(f"  Achieved: {metric_value:.3f} (+{improvement:.3f})")

    return metrics


def load_phase_1_3_results():
    """Load Phases 1-3 results for comparison."""
    print("\nLoading Phases 1-3 results for comparison...")

    results_file = RESULTS_DIR / 'progressive_temporal_validation_results.json'

    if not results_file.exists():
        print("  Warning: Could not find previous phase results")
        return None

    with open(results_file, 'r') as f:
        previous_results = json.load(f)

    print("  ✓ Loaded previous phase results")
    return previous_results


def compare_to_previous_phases(phase4_metrics: Dict, previous_results: Dict, use_case: str):
    """Compare Phase 4 results to Phases 1-3."""
    print_header("Temporal Drift Analysis", level=2)

    if previous_results is None:
        print("No previous results available for comparison")
        return

    # Extract Phase 1-3 ROC-AUC scores
    phase_scores = []
    for phase_name in ['phase_1', 'phase_2', 'phase_3']:
        if phase_name in previous_results:
            phase_data = previous_results[phase_name]
            if use_case in phase_data:
                roc_auc = phase_data[use_case].get('roc_auc', 0)
                phase_scores.append(roc_auc)

    if not phase_scores:
        print("Could not extract previous phase scores")
        return

    # Phase 4 score
    phase4_roc_auc = phase4_metrics['roc_auc']

    # Calculate statistics
    mean_prev = np.mean(phase_scores)
    std_prev = np.std(phase_scores)
    min_prev = np.min(phase_scores)
    max_prev = np.max(phase_scores)

    # Temporal drift detection
    drift = mean_prev - phase4_roc_auc
    drift_pct = (drift / mean_prev) * 100

    print(f"ROC-AUC Comparison (use_case: {use_case}):")
    print(f"\n  Phases 1-3:")
    print(f"    Mean:   {mean_prev:.3f} (±{std_prev:.3f})")
    print(f"    Range:  [{min_prev:.3f}, {max_prev:.3f}]")
    print(f"\n  Phase 4 (2024):")
    print(f"    Score:  {phase4_roc_auc:.3f}")
    print(f"\n  Temporal Drift:")
    print(f"    Absolute: {drift:.3f}")
    print(f"    Relative: {drift_pct:.1f}%")

    # Drift assessment (>10% is concerning)
    if abs(drift_pct) > 10:
        print(f"\n  ⚠️  WARNING: Temporal drift detected (>{10}%)")
        print(f"      Model may not generalize well to 2024 data")
    elif abs(drift_pct) > 5:
        print(f"\n  ⚠️  CAUTION: Moderate drift detected (>{5}%)")
        print(f"      Monitor model performance closely")
    else:
        print(f"\n  ✓ STABLE: No significant temporal drift")
        print(f"      Model generalizes well to 2024")


def save_results(model_dict: Dict, metrics_by_use_case: Dict):
    """Save Phase 4 results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'phase4_temporal_validation_{timestamp}.json'

    results = {
        'phase': 4,
        'training_years': model_dict['training_years'],
        'test_year': 2024,
        'n_train_samples': model_dict['n_train_samples'],
        'cv_score': model_dict['cv_score'],
        'best_params': model_dict['best_params'],
        'timestamp': timestamp,
        'metrics_by_use_case': metrics_by_use_case
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Temporal Validation')
    parser.add_argument('--use-case', type=str, default='all',
                       choices=['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases', 'all'],
                       help='Use case threshold to evaluate')

    args = parser.parse_args()

    print_header("Phase 4 Temporal Validation: 2020-2023 → 2024")

    # Load data
    train_samples = load_training_data_2020_2023()
    test_samples = load_2024_test_data()

    # Extract features
    print("\nExtracting features from training data...")
    X_train, y_train = extract_features_from_samples(train_samples)
    print(f"  Training features: {X_train.shape}")

    print("\nExtracting features from test data...")
    X_test, y_test = extract_features_from_samples(test_samples)
    print(f"  Test features: {X_test.shape}")

    # Train model
    model_dict = train_model(X_train, y_train)

    # Evaluate on all use cases or specific one
    use_cases = list(OPTIMAL_THRESHOLDS.keys()) if args.use_case == 'all' else [args.use_case]

    metrics_by_use_case = {}

    for use_case in use_cases:
        metrics = evaluate_model(model_dict, X_test, y_test, use_case=use_case)
        metrics_by_use_case[use_case] = metrics

    # Load and compare to Phases 1-3
    previous_results = load_phase_1_3_results()

    if previous_results:
        for use_case in use_cases:
            compare_to_previous_phases(metrics_by_use_case[use_case], previous_results, use_case)

    # Save results
    save_results(model_dict, metrics_by_use_case)

    # Summary
    print_header("Phase 4 Summary")

    print("Model Performance on 2024:")
    for use_case, metrics in metrics_by_use_case.items():
        print(f"\n  {use_case}:")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1']:.3f}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    print("1. Review temporal drift analysis above")
    print("2. If drift is acceptable (<10%), proceed to production training:")
    print("   uv run python src/walk/35_train_production_model.py")
    print("\n3. If drift is high (>10%), investigate:")
    print("   - Feature distribution changes in 2024")
    print("   - Label quality issues")
    print("   - Environmental/seasonal factors")


if __name__ == '__main__':
    main()
