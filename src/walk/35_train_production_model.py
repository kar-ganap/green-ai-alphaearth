#!/usr/bin/env python3
"""
Train Production Model (2020-2024 Combined)

Trains final production model using all available data from 2020-2024.
Combines 685 samples from 2020-2023 with 162 samples from 2024.

Input:
  - walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl (2020-2023)
  - walk_dataset_2024_with_features_*.pkl (2024)

Output:
  - walk_model_production_2020_2024.pkl

Usage:
    uv run python src/walk/35_train_production_model.py
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_config


def load_2020_2023_data():
    """Load 2020-2023 training data (685 samples)."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(processed_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No 2020-2023 data found matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Loading 2020-2023: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    print(f"  Samples: {len(samples)}")
    return samples


def load_2024_data():
    """Load 2024 data (162 samples)."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    pattern = 'walk_dataset_2024_with_features_*.pkl'
    files = list(processed_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No 2024 data found matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Loading 2024: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('samples', data.get('data', data))
    print(f"  Samples: {len(samples)}")
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
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + [
            'coarse_heterogeneity', 'coarse_range'
        ]

        try:
            coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
        except KeyError:
            continue

        # Combine
        combined = np.concatenate([annual_features, coarse_features])
        X.append(combined)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def train_production_model(X_train, y_train):
    """
    Train production Random Forest model with hyperparameter tuning.

    Returns:
        Trained model and best hyperparameters
    """
    print(f"\n{'='*80}")
    print("TRAINING PRODUCTION MODEL")
    print(f"{'='*80}\n")

    print(f"Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}D (3D annual + 66D coarse multiscale)")
    print(f"  Clearing: {sum(y_train)} samples")
    print(f"  Intact:   {sum(y_train == 0)} samples")
    print()

    # Hyperparameter grid (same as Phase 4)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Hyperparameter tuning:")
    print(f"  Grid size: {n_combinations} combinations")
    print(f"  CV strategy: 5-fold StratifiedKFold")
    print(f"  Scoring: ROC-AUC")
    print()

    # Grid search
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Starting grid search (this will take several minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\n✓ Grid search complete")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\n  Best hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model and return metrics."""
    print(f"\n{'-'*80}")
    print(f"EVALUATING ON {dataset_name.upper()}")
    print(f"{'-'*80}\n")

    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Metrics
    roc_auc = roc_auc_score(y, y_pred_proba)
    pr_auc = average_precision_score(y, y_pred_proba)

    print(f"Overall Performance:")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  PR-AUC:  {pr_auc:.3f}")
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Intact', 'Clearing']))

    # Use case thresholds (from Phase 3 optimization)
    use_case_thresholds = {
        'risk_ranking': 0.070,      # Recall 90%+
        'rapid_response': 0.608,    # Recall 90%+
        'comprehensive': 0.884,     # Precision 2x baseline
        'edge_cases': 0.910         # High precision
    }

    print("\nUse Case Performance:")
    print(f"{'Use Case':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)

    for use_case, threshold in use_case_thresholds.items():
        y_pred_uc = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_pred_uc == 1) & (y == 1))
        fp = np.sum((y_pred_uc == 1) & (y == 0))
        fn = np.sum((y_pred_uc == 0) & (y == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{use_case:<20} {threshold:<12.3f} {precision:<12.1%} {recall:<12.1%}")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def save_model(model, best_params, cv_score, train_metrics, metadata):
    """Save production model."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = processed_dir / f'walk_model_production_2020_2024_{timestamp}.pkl'

    model_data = {
        'model': model,
        'best_params': best_params,
        'cv_score': cv_score,
        'train_metrics': train_metrics,
        'metadata': metadata,
        'use_case_thresholds': {
            'risk_ranking': 0.070,
            'rapid_response': 0.608,
            'comprehensive': 0.884,
            'edge_cases': 0.910
        },
        'features': {
            'annual': '3D (delta_1yr, delta_2yr, acceleration)',
            'multiscale': '66D coarse (64D embedding + heterogeneity + range)',
            'total_dims': 69
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Saved production model to:")
    print(f"  {output_path}")

    return output_path


def main():
    print("=" * 80)
    print("PRODUCTION MODEL TRAINING (2020-2024)")
    print("=" * 80)
    print()

    # Load data
    print("Loading training data...")
    samples_2020_2023 = load_2020_2023_data()
    samples_2024 = load_2024_data()

    # Combine
    all_samples = samples_2020_2023 + samples_2024
    print(f"\n✓ Combined: {len(all_samples)} total samples")
    print()

    # Extract features
    print("Extracting features...")
    X_train, y_train = extract_features(all_samples)
    print(f"✓ Extracted features: {X_train.shape}")
    print(f"  Clearing: {sum(y_train)} samples")
    print(f"  Intact:   {sum(y_train == 0)} samples")
    print()

    # Train model
    model, best_params, cv_score = train_production_model(X_train, y_train)

    # Evaluate on training set
    train_metrics = evaluate_model(model, X_train, y_train, "Training Set (2020-2024)")

    # Metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_samples_2020_2023': len(samples_2020_2023),
        'n_samples_2024': len(samples_2024),
        'n_samples_total': len(all_samples),
        'n_features': X_train.shape[1],
        'data_years': '2020-2024',
        'note': 'Mixed tree cover thresholds (30-50%) across years - see docs for details'
    }

    # Save
    model_path = save_model(model, best_params, cv_score, train_metrics, metadata)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"Production Model:")
    print(f"  Training data: 2020-2024 ({len(all_samples)} samples)")
    print(f"  CV ROC-AUC: {cv_score:.3f}")
    print(f"  Training ROC-AUC: {train_metrics['roc_auc']:.3f}")
    print()

    print(f"Next Steps:")
    print(f"  1. Evaluate on validation sets:")
    print(f"     uv run python src/walk/37_evaluate_production_model.py")
    print(f"  2. Review performance across all use cases")
    print(f"  3. Document final model characteristics")
    print()

    print(f"Model saved to:")
    print(f"  {model_path}")


if __name__ == '__main__':
    main()
