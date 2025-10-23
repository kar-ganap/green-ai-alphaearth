#!/usr/bin/env python3
"""
Progressive Temporal Validation Using Existing Data

Instead of collecting new samples, this uses the existing 685 training samples
and splits them by year for temporal validation:
  - Phase 1: Train on 2020 → Test on 2021
  - Phase 2: Train on 2020+2021 → Test on 2022
  - Phase 3: Train on 2020+2021+2022 → Test on 2023

Much faster than collecting new samples, and tests the same thing:
can the model generalize to future years?

Usage:
  python src/walk/31b_temporal_validation_from_existing.py --phase 1
  python src/walk/31b_temporal_validation_from_existing.py --all
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

from src.walk.diagnostic_helpers import extract_dual_year_features

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


def load_training_data():
    """Load the 685-sample training dataset with 69D features."""
    print("Loading training dataset...")

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

    # Group by year
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

    return by_year


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


def train_model(X_train: np.ndarray, y_train: np.ndarray, train_years: List[int]) -> Dict:
    """Train Random Forest with hyperparameter tuning."""
    print_header(f"Training model on years: {train_years}", level=2)

    print(f"Training samples: {len(X_train)}")
    print(f"  Clearing: {sum(y_train == 1)}")
    print(f"  Intact: {sum(y_train == 0)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Hyperparameter grid
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
    cv = StratifiedKFold(n_splits=min(5, len(X_train) // 20), shuffle=True, random_state=42)

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
        'training_years': train_years,
        'n_train_samples': len(X_train)
    }


def evaluate_model(model_dict: Dict, X_test: np.ndarray, y_test: np.ndarray,
                   test_year: int, use_case: str = 'edge_cases') -> Dict:
    """Evaluate model on test set."""
    print_header(f"Evaluating on {test_year} (use_case: {use_case})", level=2)

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
        'test_year': test_year,
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
    target_config = USE_CASE_TARGETS[use_case]
    target_metric = target_config['metric']
    if 'target' in target_config:
        target_met = metrics[target_metric] >= target_config['target']
        print(f"\n  Target ({target_metric} ≥ {target_config['target']:.2f}): {'✓ MET' if target_met else '✗ NOT MET'}")
    else:
        target_met = True

    metrics['target_met'] = bool(target_met)

    return metrics


def run_phase(phase: int, by_year: Dict[int, List[dict]], use_case: str = 'edge_cases') -> Dict:
    """Run a single phase of progressive validation."""
    phase_config = {
        1: {'train_years': [2020], 'test_year': 2021},
        2: {'train_years': [2020, 2021], 'test_year': 2022},
        3: {'train_years': [2020, 2021, 2022], 'test_year': 2023}
    }

    if phase not in phase_config:
        raise ValueError(f"Invalid phase: {phase}")

    config = phase_config[phase]
    test_year = config['test_year']
    train_years = config['train_years']

    print_header(f"Phase {phase}: Train on {train_years} → Test on {test_year}")

    # Collect training samples
    train_samples = []
    for year in train_years:
        if year in by_year:
            train_samples.extend(by_year[year])

    # Get test samples
    if test_year not in by_year:
        print(f"Warning: No samples for test year {test_year}")
        return {}

    test_samples = by_year[test_year]

    print(f"\nTraining samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Extract features
    print("\nExtracting training features...")
    X_train, y_train = extract_features_from_samples(train_samples)
    print(f"  ✓ Extracted: {X_train.shape}")

    print("\nExtracting test features...")
    X_test, y_test = extract_features_from_samples(test_samples)
    print(f"  ✓ Extracted: {X_test.shape}")

    # Train model
    model_dict = train_model(X_train, y_train, train_years)

    # Evaluate
    results = evaluate_model(model_dict, X_test, y_test, test_year, use_case)

    results['phase'] = phase
    results['train_years'] = train_years
    results['model_info'] = {
        'cv_score': model_dict['cv_score'],
        'n_train_samples': len(X_train)
    }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'temporal_validation_phase{phase}_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path.name}")

    return results


def detect_temporal_drift(results: List[Dict]) -> Dict:
    """Analyze temporal drift across phases."""
    print_header("Temporal Drift Analysis", level=2)

    if len(results) < 2:
        print("  Not enough phases to assess drift")
        return {'drift_detected': False}

    years = [r['test_year'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]

    print("\nPerformance across years:")
    print(f"{'Year':<8} {'ROC-AUC':<10} {'Recall':<10} {'Precision':<10} {'Target'}")
    print("-" * 60)

    for i, year in enumerate(years):
        target_status = '✓' if results[i]['target_met'] else '✗'
        print(f"{year:<8} {roc_aucs[i]:<10.3f} {recalls[i]:<10.3f} {precisions[i]:<10.3f} {target_status}")

    # Detect degradation
    max_drop = max(roc_aucs[0] - auc for auc in roc_aucs[1:]) if len(roc_aucs) > 1 else 0
    drift_threshold = 0.10
    drift_detected = max_drop > drift_threshold

    print(f"\nDrift detection:")
    print(f"  Max ROC-AUC drop: {max_drop:.3f}")
    print(f"  Threshold: {drift_threshold:.3f}")
    print(f"  Drift detected: {'✓ YES' if drift_detected else '✗ NO'}")

    return {
        'drift_detected': drift_detected,
        'max_drop': max_drop,
        'performance_by_year': {
            str(year): {
                'roc_auc': roc_aucs[i],
                'recall': recalls[i],
                'precision': precisions[i]
            }
            for i, year in enumerate(years)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Progressive Temporal Validation')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help='Which phase to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all phases sequentially')
    parser.add_argument('--use-case', type=str, default='edge_cases',
                       choices=['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases'],
                       help='Which use case threshold to apply')

    args = parser.parse_args()

    # Load data
    by_year = load_training_data()

    if args.all:
        all_results = []
        for phase in [1, 2, 3]:
            results = run_phase(phase, by_year, args.use_case)
            if results:
                all_results.append(results)

        # Analyze drift
        if len(all_results) >= 2:
            drift_analysis = detect_temporal_drift(all_results)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = RESULTS_DIR / f'temporal_validation_all_phases_{timestamp}.json'

            with open(output_path, 'w') as f:
                json.dump({
                    'phases': all_results,
                    'drift_analysis': drift_analysis
                }, f, indent=2)

            print(f"\n✓ Combined results saved to: {output_path.name}")

    elif args.phase:
        run_phase(args.phase, by_year, args.use_case)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
