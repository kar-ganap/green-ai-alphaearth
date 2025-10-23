#!/usr/bin/env python3
"""
Phase B: Model Diversity - XGBoost vs Random Forest

Experiments:
1. XGBoost baseline (2020-2023 only)
2. XGBoost adapted (2020-2024 with year feature)
3. Compare to Random Forest from Phase A
4. Identify which model handles temporal drift better

Goal: Push beyond 0.932 to 0.95+, or identify best model architecture

Usage:
    uv run python src/walk/42_phase_b_model_diversity.py
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
import json

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'walk'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_header(text: str, level: int = 1):
    """Print formatted header."""
    char = '=' if level == 1 else '-'
    width = 80
    print(f"\n{char * width}")
    print(text.upper() if level == 1 else text)
    print(f"{char * width}\n")


def load_2020_2023_data():
    """Load 2020-2023 training data."""
    print("Loading 2020-2023 training data...")

    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find training data")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    print(f"  Loaded {len(samples)} samples")

    return samples


def load_2024_data():
    """Load 2024 test data."""
    print("\nLoading 2024 data...")

    pattern = 'walk_dataset_2024_with_features_*.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find 2024 data")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('samples', data.get('data', data))
    print(f"  Loaded {len(samples)} samples")

    return samples


def extract_features_from_samples(samples: List[dict], add_year: bool = False):
    """Extract features from samples."""
    X = []
    y = []
    years = []

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

        year = sample.get('year', 2021)
        if add_year:
            combined = np.concatenate([combined, [year]])

        X.append(combined)
        y.append(sample.get('label', 0))
        years.append(year)

    return np.array(X), np.array(y), np.array(years)


def experiment_1_xgboost_baseline():
    """
    Experiment 1: XGBoost baseline (2020-2023 only).

    This establishes how well XGBoost handles the temporal drift
    compared to Random Forest baseline (0.796).
    """
    print_header("Experiment 1: XGBoost Baseline (2020-2023)", level=1)

    # Load data
    samples_2020_2023 = load_2020_2023_data()
    samples_2024 = load_2024_data()

    # Extract features (no year)
    X_2020_2023, y_2020_2023, _ = extract_features_from_samples(samples_2020_2023)
    X_2024, y_2024, _ = extract_features_from_samples(samples_2024)

    print(f"\nData summary:")
    print(f"  Training (2020-2023): {len(X_2020_2023)} samples")
    print(f"  Test (2024): {len(X_2024)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_2020_2023)
    X_test_scaled = scaler.transform(X_2024)

    # Train XGBoost with GridSearchCV
    print("\nTraining XGBoost with GridSearchCV...")
    print("Testing temporal drift resilience...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    base_model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_2020_2023)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in sorted(grid_search.best_params_.items()):
        print(f"  {param}: {value}")

    # Evaluate on 2024
    model = grid_search.best_estimator_
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_2024, y_pred)),
        'precision': float(precision_score(y_2024, y_pred, zero_division=0)),
        'recall': float(recall_score(y_2024, y_pred, zero_division=0)),
        'f1': float(f1_score(y_2024, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_2024, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_2024, y_pred_proba)),
        'cv_score': grid_search.best_score_
    }

    print_header("XGBoost Baseline Results (2024 test)", level=2)
    print(f"\nPerformance:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")

    print(f"\nComparison to RF baseline:")
    print(f"  RF baseline (2020-2023): 0.796")
    print(f"  XGB baseline (2020-2023): {metrics['roc_auc']:.3f}")

    diff = metrics['roc_auc'] - 0.796
    print(f"  Difference: {diff:+.3f}")

    if diff > 0.02:
        print(f"  → XGBoost handles temporal drift BETTER than Random Forest")
    elif diff < -0.02:
        print(f"  → XGBoost handles temporal drift WORSE than Random Forest")
    else:
        print(f"  → Similar temporal drift resilience")

    return {
        'experiment': 'xgboost_baseline',
        'metrics': metrics,
        'model': model,
        'scaler': scaler
    }


def experiment_2_xgboost_adapted():
    """
    Experiment 2: XGBoost with 2020-2024 data and year feature.

    This tests if XGBoost can exceed Random Forest's 0.932 performance.
    """
    print_header("Experiment 2: XGBoost Adapted (2020-2024 + Year)", level=1)

    # Load data
    samples_2020_2023 = load_2020_2023_data()
    samples_2024 = load_2024_data()

    # Extract features WITH year
    X_2020_2023, y_2020_2023, _ = extract_features_from_samples(samples_2020_2023, add_year=True)
    X_2024, y_2024, _ = extract_features_from_samples(samples_2024, add_year=True)

    print(f"\nData summary:")
    print(f"  2020-2023: {len(X_2020_2023)} samples (70D with year)")
    print(f"  2024: {len(X_2024)} samples (70D with year)")

    # Split 2024
    X_2024_train, X_2024_test, y_2024_train, y_2024_test = train_test_split(
        X_2024, y_2024, test_size=0.3, random_state=42, stratify=y_2024
    )

    print(f"\n2024 split:")
    print(f"  Train: {len(X_2024_train)} samples")
    print(f"  Test: {len(X_2024_test)} samples")

    # Combine
    X_train_combined = np.vstack([X_2020_2023, X_2024_train])
    y_train_combined = np.concatenate([y_2020_2023, y_2024_train])

    print(f"\nCombined training: {len(X_train_combined)} samples")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_2024_test)

    # Train XGBoost
    print("\nTraining XGBoost on combined 2020-2024 data...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    base_model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train_combined)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in sorted(grid_search.best_params_.items()):
        print(f"  {param}: {value}")

    # Evaluate
    model = grid_search.best_estimator_
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_2024_test, y_pred)),
        'precision': float(precision_score(y_2024_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_2024_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_2024_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_2024_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_2024_test, y_pred_proba)),
        'cv_score': grid_search.best_score_
    }

    print_header("XGBoost Adapted Results", level=2)
    print(f"\nPerformance:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")

    print(f"\nComparison to Phase A:")
    print(f"  RF adapted (Phase A): 0.932")
    print(f"  XGB adapted: {metrics['roc_auc']:.3f}")

    diff = metrics['roc_auc'] - 0.932
    print(f"  Difference: {diff:+.3f}")

    if metrics['roc_auc'] >= 0.95:
        print(f"  ✓✓✓ EXCELLENT: Exceeded 0.95!")
    elif diff > 0.01:
        print(f"  ✓✓ XGBoost outperforms Random Forest")
    elif diff > -0.01:
        print(f"  ✓ Comparable performance")
    else:
        print(f"  ~ Random Forest performs better")

    return {
        'experiment': 'xgboost_adapted',
        'metrics': metrics,
        'model': model,
        'scaler': scaler
    }


def save_results(exp1_results: Dict, exp2_results: Dict):
    """Save Phase B results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'phase_b_model_diversity_{timestamp}.json'

    results = {
        'phase': 'B',
        'description': 'Model Diversity - XGBoost vs Random Forest',
        'baseline_phase4_rf': 0.796,
        'phase_a_rf_adapted': 0.932,
        'timestamp': timestamp,
        'experiments': {
            'experiment_1_xgboost_baseline': {
                'description': 'XGBoost on 2020-2023 (temporal drift test)',
                'metrics': exp1_results['metrics']
            },
            'experiment_2_xgboost_adapted': {
                'description': 'XGBoost on 2020-2024 with year feature',
                'metrics': exp2_results['metrics']
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")
    return output_path


def main():
    print_header("Phase B: Model Diversity - XGBoost vs Random Forest")

    print("Phase A Results (Random Forest):")
    print("  Baseline (2020-2023): 0.796")
    print("  Adapted (2020-2024):  0.932 (+0.136)")
    print("\nPhase B Goal: Test if XGBoost can exceed 0.932")

    # Run experiments
    exp1_results = experiment_1_xgboost_baseline()
    exp2_results = experiment_2_xgboost_adapted()

    # Compare results
    print_header("Phase B Summary", level=1)

    print("Model Comparison:")
    print(f"\n  Random Forest:")
    print(f"    Baseline (2020-2023):  0.796")
    print(f"    Adapted (2020-2024):   0.932")
    print(f"    Improvement:           +0.136 (+17.0%)")

    print(f"\n  XGBoost:")
    print(f"    Baseline (2020-2023):  {exp1_results['metrics']['roc_auc']:.3f}")
    print(f"    Adapted (2020-2024):   {exp2_results['metrics']['roc_auc']:.3f}")

    xgb_improvement = exp2_results['metrics']['roc_auc'] - exp1_results['metrics']['roc_auc']
    xgb_improvement_pct = (xgb_improvement / exp1_results['metrics']['roc_auc']) * 100
    print(f"    Improvement:           {xgb_improvement:+.3f} ({xgb_improvement_pct:+.1f}%)")

    # Determine best model
    best_roc_auc = max(0.932, exp2_results['metrics']['roc_auc'])

    if exp2_results['metrics']['roc_auc'] > 0.932:
        best_model = "XGBoost"
        print(f"\n✓ Winner: XGBoost ({exp2_results['metrics']['roc_auc']:.3f})")
    elif exp2_results['metrics']['roc_auc'] < 0.920:
        best_model = "Random Forest"
        print(f"\n✓ Winner: Random Forest (0.932)")
    else:
        best_model = "Comparable"
        print(f"\n~ Both models comparable (~0.93)")

    # Save results
    save_results(exp1_results, exp2_results)

    # Decision point
    print_header("Decision Point", level=1)

    if best_roc_auc >= 0.95:
        print("✓✓✓ EXCEPTIONAL: ROC-AUC ≥ 0.95")
        print("  → Exceeded target significantly")
        print("  → Ready for Phase C (ensemble) to push even higher")
        print("  → Or finalize current model for demo")
    elif best_roc_auc >= 0.93:
        print("✓✓ EXCELLENT: ROC-AUC ≥ 0.93")
        print("  → Strong performance achieved")
        print("  → Phase C (ensemble) optional for marginal gains")
        print("  → Consider finalizing for demo")
    elif best_roc_auc >= 0.90:
        print("✓ GOOD: ROC-AUC ≥ 0.90")
        print("  → Solid performance")
        print("  → Phase C (ensemble) recommended for improvement")
    else:
        print("~ NEEDS IMPROVEMENT: ROC-AUC < 0.90")
        print("  → Phase C (ensemble) required")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")

    if best_model == "XGBoost":
        print(f"Recommended model: XGBoost ({exp2_results['metrics']['roc_auc']:.3f})")
    elif best_model == "Random Forest":
        print(f"Recommended model: Random Forest (0.932)")
    else:
        print(f"Recommended: Test ensemble in Phase C")

    print("\nPhase C: Advanced Techniques (Ensembles)")
    print("  uv run python src/walk/43_phase_c_ensemble.py")
    print("\nOr finalize current results:")
    print("  uv run python src/walk/44_final_evaluation.py")

    return {
        'best_model': best_model,
        'best_roc_auc': best_roc_auc,
        'xgboost_results': exp2_results,
        'rf_baseline': 0.932
    }


if __name__ == '__main__':
    main()
