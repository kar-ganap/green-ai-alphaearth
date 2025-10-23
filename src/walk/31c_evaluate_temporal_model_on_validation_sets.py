#!/usr/bin/env python3
"""
Evaluate Temporal Model on All Validation Sets

Takes the best temporal model (Phase 3: trained on 2020-2022) and evaluates
it on all 4 hard validation sets with their respective optimal thresholds.

Usage:
  python src/walk/31c_evaluate_temporal_model_on_validation_sets.py
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

# Optimal thresholds per validation set
OPTIMAL_THRESHOLDS = {
    'risk_ranking': 0.070,
    'rapid_response': 0.608,
    'comprehensive': 0.884,
    'edge_cases': 0.910
}

USE_CASE_TARGETS = {
    'risk_ranking': {'metric': 'recall', 'target': 0.90},
    'rapid_response': {'metric': 'recall', 'target': 0.90},
    'comprehensive': {'metric': 'precision', 'baseline': 0.389},
    'edge_cases': {'metric': 'roc_auc', 'target': 0.65}
}


def load_training_data():
    """Load training data for 2020-2022."""
    print("Loading training data (2020-2022)...")

    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Could not find training data")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data['data'] if 'data' in data else data.get('samples', data)

    # Filter to 2020-2022 only (like Phase 3)
    train_samples = [s for s in samples if s.get('year', 2021) in [2020, 2021, 2022]]

    print(f"  Loaded {len(train_samples)} samples from 2020-2022")

    return train_samples


def load_validation_set(set_name: str):
    """Load a validation set."""
    val_path = PROCESSED_DIR / f'hard_val_{set_name}_multiscale.pkl'

    if not val_path.exists():
        raise FileNotFoundError(f"Validation set not found: {set_name}")

    with open(val_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  Loaded {len(samples)} {set_name} samples")

    return samples


def extract_features(samples):
    """Extract 69D features from samples."""
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


def train_model(X_train, y_train):
    """Train Random Forest with same hyperparameters as Phase 3."""
    print("\nTraining Random Forest...")
    print(f"  Training samples: {len(X_train)}")
    print(f"    Clearing: {sum(y_train == 1)}")
    print(f"    Intact: {sum(y_train == 0)}")

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

    return grid_search.best_estimator_, scaler


def evaluate_on_validation_set(model, scaler, val_samples, set_name):
    """Evaluate model on a validation set."""
    print(f"\n{'='*80}")
    print(f"Evaluating on {set_name.upper()}")
    print(f"{'='*80}\n")

    # Extract features
    X_val, y_val = extract_features(val_samples)

    if len(X_val) == 0:
        print(f"  Warning: No samples with complete features")
        return None

    print(f"  Test samples: {len(X_val)}")
    print(f"    Clearing: {sum(y_val == 1)}")
    print(f"    Intact: {sum(y_val == 0)}")

    # Scale and predict
    X_scaled = scaler.transform(X_val)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Get optimal threshold for this validation set
    threshold = OPTIMAL_THRESHOLDS[set_name]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    results = {
        'validation_set': set_name,
        'n_samples': len(X_val),
        'n_clearing': int(sum(y_val == 1)),
        'n_intact': int(sum(y_val == 0)),
        'threshold': threshold,
        'roc_auc': float(roc_auc_score(y_val, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_val, y_pred_proba)),
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred, zero_division=0)),
        'recall': float(recall_score(y_val, y_pred, zero_division=0)),
        'f1': float(f1_score(y_val, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
    }

    # Print results
    print(f"\nResults (threshold={threshold:.3f}):")
    print(f"  ROC-AUC:    {results['roc_auc']:.3f}")
    print(f"  PR-AUC:     {results['pr_auc']:.3f}")
    print(f"  Precision:  {results['precision']:.3f}")
    print(f"  Recall:     {results['recall']:.3f}")
    print(f"  F1-Score:   {results['f1']:.3f}")

    cm = results['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0][0]:3d}  FP: {cm[0][1]:3d}")
    print(f"    FN: {cm[1][0]:3d}  TP: {cm[1][1]:3d}")

    # Check target
    target_config = USE_CASE_TARGETS[set_name]
    target_metric = target_config['metric']

    if 'target' in target_config:
        target_met = results[target_metric] >= target_config['target']
        print(f"\n  Target ({target_metric} ≥ {target_config['target']:.2f}): {'✓ MET' if target_met else '✗ NOT MET'}")
    elif 'baseline' in target_config:
        target_met = results['precision'] > target_config['baseline']
        print(f"\n  Target (precision > {target_config['baseline']:.2f}): {'✓ MET' if target_met else '✗ NOT MET'}")
    else:
        target_met = True

    results['target_met'] = bool(target_met)

    return results


def main():
    print("="*80)
    print("TEMPORAL MODEL EVALUATION ON ALL VALIDATION SETS")
    print("="*80)
    print("\nTraining Phase 3 model (2020-2022) and evaluating on all 4 validation sets\n")

    # Load training data
    train_samples = load_training_data()

    # Extract features
    print("\nExtracting training features...")
    X_train, y_train = extract_features(train_samples)
    print(f"  ✓ Extracted: {X_train.shape}")

    # Train model
    model, scaler = train_model(X_train, y_train)

    # Evaluate on all validation sets
    all_results = {}

    for set_name in ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']:
        try:
            val_samples = load_validation_set(set_name)
            results = evaluate_on_validation_set(model, scaler, val_samples, set_name)

            if results:
                all_results[set_name] = results
        except Exception as e:
            print(f"\n  Error evaluating {set_name}: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL VALIDATION SETS")
    print(f"{'='*80}\n")

    print(f"{'Set':<18} {'ROC-AUC':<10} {'Precision':<12} {'Recall':<10} {'Target'}")
    print("-" * 70)

    for set_name in ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']:
        if set_name in all_results:
            r = all_results[set_name]
            target = '✓' if r['target_met'] else '✗'
            print(f"{set_name:<18} {r['roc_auc']:<10.3f} {r['precision']:<12.3f} {r['recall']:<10.3f} {target}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'temporal_model_validation_sets_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump({
            'training_years': [2020, 2021, 2022],
            'training_samples': len(X_train),
            'results_by_set': all_results,
            'timestamp': timestamp
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path.name}")


if __name__ == '__main__':
    main()
