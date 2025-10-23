#!/usr/bin/env python3
"""
Phase 4 Uniform 30% Temporal Validation: 2020-2023 → 2024

EXPERIMENT: Decompose temporal drift into sampling bias vs real temporal change

Train on uniform 30% threshold data (2020-2023, 588 samples) and test on 2024 (162 samples).
Compare results to heterogeneous Phase 4 to isolate sampling bias contribution.

Heterogeneous Phase 4: 0.981 → 0.796 (18.9% drop)
Uniform 30% Phase 4: 0.98X → ?

Outcomes:
- If ~0.98 ROC-AUC on 2024: Drift was entirely sampling bias
- If ~0.80 ROC-AUC on 2024: Drift is real temporal change
- If intermediate (~0.88-0.90): Both effects present, can quantify each

Usage:
    uv run python src/walk/40_phase4_uniform_30pct_validation.py
    uv run python src/walk/40_phase4_uniform_30pct_validation.py --use-case comprehensive
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


def load_uniform_30pct_training_data():
    """Load uniform 30% threshold training data (2020-2023, 588 samples)."""
    print("Loading uniform 30% training data (2020-2023)...")

    pattern = 'walk_dataset_uniform_30pct_2020_2023_with_features_*.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find training data matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    metadata = data.get('metadata', {})

    print(f"  Loaded {len(samples)} samples")
    print(f"\n  Metadata:")
    print(f"    Threshold: uniform 30% (vs heterogeneous 30-50%)")
    print(f"    Purpose: Isolate temporal drift from sampling bias")

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
    print_header("Training model on uniform 30% data (2020-2023)", level=2)

    print(f"Training samples: {len(X_train)}")
    print(f"  Clearing: {sum(y_train == 1)}")
    print(f"  Intact: {sum(y_train == 0)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Hyperparameter grid (same as Phase 4)
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
        'n_train_samples': len(X_train),
        'dataset_type': 'uniform_30pct'
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


def load_heterogeneous_phase4_results():
    """Load heterogeneous Phase 4 results for comparison."""
    print("\nLoading heterogeneous Phase 4 results for comparison...")

    pattern = 'phase4_temporal_validation_*.json'
    files = list(RESULTS_DIR.glob(pattern))

    if not files:
        print("  Warning: Could not find heterogeneous Phase 4 results")
        return None

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'r') as f:
        hetero_results = json.load(f)

    print("  ✓ Loaded heterogeneous Phase 4 results")
    return hetero_results


def compare_to_heterogeneous(uniform_metrics: Dict, hetero_results: Dict, use_case: str):
    """Compare uniform 30% results to heterogeneous Phase 4 to decompose drift."""
    print_header("Drift Decomposition Analysis", level=2)

    if hetero_results is None:
        print("No heterogeneous results available for comparison")
        return

    # Extract heterogeneous metrics for this use case
    hetero_metrics = hetero_results.get('metrics_by_use_case', {}).get(use_case, {})

    if not hetero_metrics:
        print(f"Could not find heterogeneous results for use_case: {use_case}")
        return

    # Extract key metrics
    hetero_cv_score = hetero_results.get('cv_score', 0)
    hetero_test_roc_auc = hetero_metrics.get('roc_auc', 0)
    hetero_drift = hetero_cv_score - hetero_test_roc_auc
    hetero_drift_pct = (hetero_drift / hetero_cv_score) * 100 if hetero_cv_score > 0 else 0

    # Uniform 30% metrics (from this run)
    uniform_cv_score = 0  # Will be set from model_dict in main()
    uniform_test_roc_auc = uniform_metrics['roc_auc']

    print(f"Drift Decomposition (use_case: {use_case}):")
    print(f"\n  Heterogeneous (30-50% mixed thresholds):")
    print(f"    CV ROC-AUC (2020-2023):  {hetero_cv_score:.3f}")
    print(f"    Test ROC-AUC (2024):     {hetero_test_roc_auc:.3f}")
    print(f"    Temporal drift:          {hetero_drift:.3f} ({hetero_drift_pct:.1f}%)")

    print(f"\n  Uniform 30% (this run):")
    print(f"    Test ROC-AUC (2024):     {uniform_test_roc_auc:.3f}")

    # Drift attribution
    print(f"\n  Drift Attribution:")

    # Compare test performance
    perf_difference = uniform_test_roc_auc - hetero_test_roc_auc

    print(f"    Difference in 2024 test performance:")
    print(f"      Uniform - Heterogeneous = {perf_difference:.3f}")

    if abs(perf_difference) < 0.05:
        print(f"\n  ✓ MINIMAL DIFFERENCE: Both models perform similarly on 2024")
        print(f"    → Temporal drift ({hetero_drift_pct:.1f}%) is REAL temporal change")
        print(f"    → Sampling bias is NOT the primary driver")
    elif perf_difference > 0.05:
        print(f"\n  ✓ UNIFORM BETTER: Uniform 30% outperforms heterogeneous on 2024")
        print(f"    → Part of heterogeneous drift was SAMPLING BIAS")
        print(f"    → Estimated sampling bias contribution: ~{perf_difference:.3f} ROC-AUC")
        print(f"    → Estimated real temporal drift: ~{hetero_drift - perf_difference:.3f} ROC-AUC")
    else:
        print(f"\n  ⚠️  UNEXPECTED: Uniform 30% performs worse on 2024")
        print(f"    → This suggests heterogeneous training may have unintended benefits")
        print(f"    → Further investigation recommended")


def save_results(model_dict: Dict, metrics_by_use_case: Dict, hetero_comparison: Dict):
    """Save uniform 30% Phase 4 results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'phase4_uniform_30pct_validation_{timestamp}.json'

    results = {
        'experiment': 'uniform_30pct_temporal_validation',
        'phase': 4,
        'dataset_type': 'uniform_30pct',
        'training_years': model_dict['training_years'],
        'test_year': 2024,
        'n_train_samples': model_dict['n_train_samples'],
        'cv_score': model_dict['cv_score'],
        'best_params': model_dict['best_params'],
        'timestamp': timestamp,
        'metrics_by_use_case': metrics_by_use_case,
        'heterogeneous_comparison': hetero_comparison
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Uniform 30% Temporal Validation')
    parser.add_argument('--use-case', type=str, default='all',
                       choices=['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases', 'all'],
                       help='Use case threshold to evaluate')

    args = parser.parse_args()

    print_header("Phase 4 Uniform 30% Temporal Validation")
    print("EXPERIMENT: Decompose drift into sampling bias vs temporal change")
    print()

    # Load data
    train_samples = load_uniform_30pct_training_data()
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

    # Load and compare to heterogeneous Phase 4
    hetero_results = load_heterogeneous_phase4_results()

    hetero_comparison = {}
    if hetero_results:
        for use_case in use_cases:
            compare_to_heterogeneous(metrics_by_use_case[use_case], hetero_results, use_case)

            # Store comparison data
            hetero_metrics = hetero_results.get('metrics_by_use_case', {}).get(use_case, {})
            hetero_comparison[use_case] = {
                'heterogeneous_cv_score': hetero_results.get('cv_score', 0),
                'heterogeneous_test_roc_auc': hetero_metrics.get('roc_auc', 0),
                'uniform_cv_score': model_dict['cv_score'],
                'uniform_test_roc_auc': metrics_by_use_case[use_case]['roc_auc'],
                'test_performance_difference': metrics_by_use_case[use_case]['roc_auc'] - hetero_metrics.get('roc_auc', 0)
            }

    # Save results
    save_results(model_dict, metrics_by_use_case, hetero_comparison)

    # Summary
    print_header("Uniform 30% Phase 4 Summary")

    print("Model Performance on 2024:")
    for use_case, metrics in metrics_by_use_case.items():
        print(f"\n  {use_case}:")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1-Score:  {metrics['f1']:.3f}")

    if hetero_comparison:
        print("\n" + "="*80)
        print("DRIFT DECOMPOSITION SUMMARY")
        print("="*80 + "\n")

        for use_case in use_cases:
            comp = hetero_comparison.get(use_case, {})
            if comp:
                hetero_drift = comp['heterogeneous_cv_score'] - comp['heterogeneous_test_roc_auc']
                hetero_drift_pct = (hetero_drift / comp['heterogeneous_cv_score']) * 100 if comp['heterogeneous_cv_score'] > 0 else 0
                perf_diff = comp['test_performance_difference']

                print(f"{use_case}:")
                print(f"  Heterogeneous drift: {hetero_drift:.3f} ({hetero_drift_pct:.1f}%)")
                print(f"  Test performance difference: {perf_diff:+.3f}")

                if abs(perf_diff) < 0.05:
                    print(f"  → Drift is REAL temporal change (not sampling bias)")
                elif perf_diff > 0.05:
                    print(f"  → Drift includes SAMPLING BIAS (~{perf_diff:.3f})")
                    print(f"  → Real temporal drift: ~{hetero_drift - perf_diff:.3f}")
                print()

    print("="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    print("1. Review drift decomposition analysis above")
    print("2. Compare uniform 30% vs heterogeneous performance:")
    print("   uv run python src/walk/41_compare_sampling_strategies.py")
    print("\n3. Document findings:")
    print("   - Quantify sampling bias contribution to drift")
    print("   - Quantify real temporal drift component")
    print("   - Inform future data collection strategy")


if __name__ == '__main__':
    main()
