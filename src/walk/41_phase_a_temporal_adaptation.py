#!/usr/bin/env python3
"""
Phase A: Quick Wins - Temporal Adaptation with 2020-2024 Data

Experiments:
1. Simple retraining: Include 2024 data in training
2. Feature augmentation: Add year as temporal feature

Goal: Recover from 0.796 baseline to 0.82-0.85+ ROC-AUC

Usage:
    uv run python src/walk/41_phase_a_temporal_adaptation.py
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import json

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
        raise FileNotFoundError(f"Could not find training data matching: {pattern}")

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
        raise FileNotFoundError(f"Could not find 2024 data matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('samples', data.get('data', data))
    print(f"  Loaded {len(samples)} samples")

    return samples


def extract_features_from_samples(samples: List[dict], add_year: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 69D features from samples.

    Args:
        samples: List of samples with features
        add_year: If True, add year as 70th feature

    Returns:
        X: Feature matrix (n_samples, 69 or 70)
        y: Labels (n_samples,)
        years: Sample years (n_samples,)
    """
    X = []
    y = []
    years = []

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

        # Add year if requested
        year = sample.get('year', 2021)
        if add_year:
            combined = np.concatenate([combined, [year]])

        X.append(combined)
        y.append(sample.get('label', 0))
        years.append(year)

    return np.array(X), np.array(y), np.array(years)


def experiment_1_simple_retraining():
    """
    Experiment 1: Simple retraining with 2020-2024 data.

    Returns:
        Dict with results
    """
    print_header("Experiment 1: Simple Retraining (2020-2024)", level=1)

    # Load all data
    samples_2020_2023 = load_2020_2023_data()
    samples_2024 = load_2024_data()

    # Extract features
    X_2020_2023, y_2020_2023, years_2020_2023 = extract_features_from_samples(samples_2020_2023)
    X_2024, y_2024, years_2024 = extract_features_from_samples(samples_2024)

    print(f"\nData summary:")
    print(f"  2020-2023: {len(X_2020_2023)} samples")
    print(f"  2024: {len(X_2024)} samples")

    # Split 2024 into train/test (70/30 split)
    # This gives us ~113 for training, ~49 for testing
    X_2024_train, X_2024_test, y_2024_train, y_2024_test = train_test_split(
        X_2024, y_2024, test_size=0.3, random_state=42, stratify=y_2024
    )

    print(f"\n2024 split:")
    print(f"  Train: {len(X_2024_train)} samples")
    print(f"  Test: {len(X_2024_test)} samples")

    # Combine 2020-2023 + 2024 train
    X_train_combined = np.vstack([X_2020_2023, X_2024_train])
    y_train_combined = np.concatenate([y_2020_2023, y_2024_train])

    print(f"\nCombined training data:")
    print(f"  Total: {len(X_train_combined)} samples")
    print(f"  Clearing: {sum(y_train_combined == 1)}")
    print(f"  Intact: {sum(y_train_combined == 0)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_2024_test)

    # Train Random Forest with GridSearchCV
    print("\nTraining Random Forest with GridSearchCV...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

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

    grid_search.fit(X_train_scaled, y_train_combined)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in sorted(grid_search.best_params_.items()):
        print(f"  {param}: {value}")

    # Evaluate on held-out 2024 test set
    model = grid_search.best_estimator_
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(y_2024_test, y_pred)),
        'precision': float(precision_score(y_2024_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_2024_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_2024_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_2024_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_2024_test, y_pred_proba)),
        'cv_score': grid_search.best_score_,
        'n_train': len(X_train_combined),
        'n_test': len(X_2024_test)
    }

    print_header("Results on held-out 2024 test set", level=2)
    print(f"Test samples: {len(y_2024_test)}")
    print(f"  Clearing: {sum(y_2024_test == 1)}")
    print(f"  Intact: {sum(y_2024_test == 0)}")
    print(f"\nPerformance:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.3f} (baseline: 0.796)")
    print(f"  PR-AUC:     {metrics['pr_auc']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")

    improvement = metrics['roc_auc'] - 0.796
    improvement_pct = (improvement / 0.796) * 100

    print(f"\n  Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

    if improvement > 0.02:
        print(f"  ✓ SIGNIFICANT IMPROVEMENT")
    elif improvement > 0:
        print(f"  ~ MODEST IMPROVEMENT")
    else:
        print(f"  ✗ NO IMPROVEMENT")

    # Store model and scaler for later use
    results = {
        'experiment': 'simple_retraining',
        'metrics': metrics,
        'model': model,
        'scaler': scaler,
        'baseline_roc_auc': 0.796,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

    return results


def experiment_2_with_year_feature():
    """
    Experiment 2: Add year as temporal feature.

    Returns:
        Dict with results
    """
    print_header("Experiment 2: Feature Augmentation (Year as Feature)", level=1)

    # Load all data
    samples_2020_2023 = load_2020_2023_data()
    samples_2024 = load_2024_data()

    # Extract features WITH year
    X_2020_2023, y_2020_2023, years_2020_2023 = extract_features_from_samples(samples_2020_2023, add_year=True)
    X_2024, y_2024, years_2024 = extract_features_from_samples(samples_2024, add_year=True)

    print(f"\nData summary:")
    print(f"  2020-2023: {len(X_2020_2023)} samples (70D features with year)")
    print(f"  2024: {len(X_2024)} samples (70D features with year)")

    # Split 2024
    X_2024_train, X_2024_test, y_2024_train, y_2024_test = train_test_split(
        X_2024, y_2024, test_size=0.3, random_state=42, stratify=y_2024
    )

    # Combine
    X_train_combined = np.vstack([X_2020_2023, X_2024_train])
    y_train_combined = np.concatenate([y_2020_2023, y_2024_train])

    print(f"\nCombined training data:")
    print(f"  Total: {len(X_train_combined)} samples")
    print(f"  Year range: {int(X_train_combined[:, -1].min())} to {int(X_train_combined[:, -1].max())}")

    # Scale features (including year)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_2024_test)

    # Train Random Forest
    print("\nTraining Random Forest with year feature...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

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

    grid_search.fit(X_train_scaled, y_train_combined)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")

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
        'cv_score': grid_search.best_score_,
        'n_train': len(X_train_combined),
        'n_test': len(X_2024_test)
    }

    print_header("Results with year feature", level=2)
    print(f"\nPerformance:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.3f} (baseline: 0.796)")
    print(f"  PR-AUC:     {metrics['pr_auc']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")

    improvement = metrics['roc_auc'] - 0.796
    improvement_pct = (improvement / 0.796) * 100

    print(f"\n  Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

    results = {
        'experiment': 'with_year_feature',
        'metrics': metrics,
        'model': model,
        'scaler': scaler,
        'baseline_roc_auc': 0.796,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

    return results


def save_results(exp1_results: Dict, exp2_results: Dict):
    """Save Phase A results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'phase_a_temporal_adaptation_{timestamp}.json'

    # Extract metrics only (not models/scalers)
    results = {
        'phase': 'A',
        'description': 'Quick Wins - Temporal Adaptation',
        'baseline_phase4': 0.796,
        'timestamp': timestamp,
        'experiments': {
            'experiment_1_simple_retraining': {
                'description': 'Retrain with 2020-2024 data',
                'metrics': exp1_results['metrics'],
                'improvement': exp1_results['improvement'],
                'improvement_pct': exp1_results['improvement_pct']
            },
            'experiment_2_year_feature': {
                'description': 'Add year as temporal feature',
                'metrics': exp2_results['metrics'],
                'improvement': exp2_results['improvement'],
                'improvement_pct': exp2_results['improvement_pct']
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_path}")
    return output_path


def main():
    print_header("Phase A: Quick Wins - Temporal Adaptation")

    print("Baseline (Phase 4): ROC-AUC = 0.796 (2020-2023 → 2024)")
    print("Goal: Recover to 0.82-0.85+ by including 2024 data")

    # Run experiments
    exp1_results = experiment_1_simple_retraining()
    exp2_results = experiment_2_with_year_feature()

    # Compare results
    print_header("Phase A Summary", level=1)

    print("Experiment Comparison:")
    print(f"\n  Baseline (Phase 4):             0.796")
    print(f"  Exp 1 (Simple Retraining):      {exp1_results['metrics']['roc_auc']:.3f} ({exp1_results['improvement']:+.3f})")
    print(f"  Exp 2 (Year Feature):           {exp2_results['metrics']['roc_auc']:.3f} ({exp2_results['improvement']:+.3f})")

    # Determine best approach
    best_exp = exp1_results if exp1_results['metrics']['roc_auc'] > exp2_results['metrics']['roc_auc'] else exp2_results

    print(f"\nBest approach: {best_exp['experiment']}")
    print(f"  ROC-AUC: {best_exp['metrics']['roc_auc']:.3f}")
    print(f"  Improvement: {best_exp['improvement']:+.3f} ({best_exp['improvement_pct']:+.1f}%)")

    # Save results
    save_results(exp1_results, exp2_results)

    # Decision point
    print_header("Decision Point", level=1)

    if best_exp['metrics']['roc_auc'] >= 0.85:
        print("✓ EXCELLENT: ROC-AUC ≥ 0.85")
        print("  → Phase A successful, can proceed to Phase B for further improvement")
        print("  → Or polish current results for demo if time-constrained")
    elif best_exp['metrics']['roc_auc'] >= 0.82:
        print("✓ GOOD: ROC-AUC ≥ 0.82")
        print("  → Significant improvement over baseline")
        print("  → Proceed to Phase B (XGBoost comparison) for potential further gains")
    elif best_exp['metrics']['roc_auc'] >= 0.80:
        print("~ MODEST: ROC-AUC ≥ 0.80")
        print("  → Some improvement but below target")
        print("  → Definitely proceed to Phase B and C for better results")
    else:
        print("⚠️ INSUFFICIENT: ROC-AUC < 0.80")
        print("  → Minimal improvement")
        print("  → Must proceed to Phase B and C")
        print("  → May need to reconsider approach")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    print("Phase B: Model Diversity (XGBoost comparison)")
    print("  uv run python src/walk/42_phase_b_model_diversity.py")
    print("\nPhase C: Advanced Techniques (Ensembles)")
    print("  uv run python src/walk/43_phase_c_ensemble.py")

    return best_exp


if __name__ == '__main__':
    main()
