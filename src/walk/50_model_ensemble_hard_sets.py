#!/usr/bin/env python3
"""
Model Ensemble (RF + XGB) on Hard Validation Sets

Tests different ensemble strategies:
1. Simple average (0.5*RF + 0.5*XGB)
2. Weighted average optimized on validation
3. Comparison vs individual models

Runs across all 4 temporal phases on hard validation sets.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import get_config
from src.walk.utils import features_to_array

# Paths
config = get_config()
data_dir = config.get_path("paths.data_dir")
PROCESSED_DIR = data_dir / 'processed'

# Hard validation files
VALIDATION_FILES = {
    2022: [
        ("hard_val_risk_ranking_2022_20251023_015922_features.pkl", "risk_ranking"),
        ("hard_val_comprehensive_2022_20251023_015927_features.pkl", "comprehensive"),
        ("hard_val_rapid_response_2022_20251023_101531_features.pkl", "rapid_response"),
    ],
    2023: [
        ("hard_val_risk_ranking_2023_20251023_015903_features.pkl", "risk_ranking"),
        ("hard_val_comprehensive_2023_20251023_015913_features.pkl", "comprehensive"),
        ("hard_val_rapid_response_2023_20251023_101559_features.pkl", "rapid_response"),
    ],
    2024: [
        ("hard_val_risk_ranking_2024_20251023_015822_features.pkl", "risk_ranking"),
        ("hard_val_comprehensive_2024_20251023_015827_features.pkl", "comprehensive"),
        ("hard_val_rapid_response_2024_20251023_101620_features.pkl", "rapid_response"),
    ]
}


def load_training_data(years):
    """Load training data for specified years."""
    print(f"\nLoading training data for years: {years}")

    # Load the correct Phase 1 training data (685 samples, 2020-2023)
    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No training data found matching pattern: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        dataset = pickle.load(f)

    # Extract samples from dataset dict
    all_samples = dataset['data']
    print(f"  Loaded {len(all_samples)} total samples")

    # Filter to specified years
    train_samples = [s for s in all_samples if s.get('year') in years]
    print(f"  Filtered to {len(train_samples)} training samples for years {years}")

    return train_samples


def load_validation_data(year):
    """Load hard validation data for specified year."""
    print(f"\nLoading validation data for year {year}")

    all_samples = {}
    total_count = 0

    for filename, use_case in VALIDATION_FILES[year]:
        filepath = PROCESSED_DIR / filename

        if not filepath.exists():
            print(f"  ⚠️ File not found: {filename}")
            continue

        with open(filepath, 'rb') as f:
            samples = pickle.load(f)

        all_samples[use_case] = samples
        total_count += len(samples)
        print(f"  {use_case}: {len(samples)} samples")

    print(f"  Total: {total_count} validation samples")
    return all_samples


def extract_features(samples):
    """
    Extract 70D features from samples using consolidated module.

    Uses features_to_array() from consolidated feature extraction module.
    """
    X = []
    y = []

    for sample in samples:
        # Use consolidated module to convert sample features to 70D array
        features_70d = features_to_array(sample)

        if features_70d is None:
            raise ValueError(f"Failed to extract features for sample: {sample.get('lat', 'unknown')}, {sample.get('lon', 'unknown')}")

        X.append(features_70d)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def train_models(X_train, y_train):
    """Train both RF and XGB models."""
    print(f"\nTraining both models...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Clearing samples: {np.sum(y_train == 1)}")
    print(f"  Intact samples: {np.sum(y_train == 0)}")

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    print(f"  ✓ Random Forest trained")

    # Train XGBoost (best params from Phase B)
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        gamma=0.2,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print(f"  ✓ XGBoost trained")

    return rf_model, xgb_model


def evaluate_ensemble(rf_model, xgb_model, X_test, y_test, use_case, ensemble_weights=(0.5, 0.5)):
    """Evaluate individual models and ensemble."""
    # Get predictions
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Ensemble predictions
    w_rf, w_xgb = ensemble_weights
    ensemble_proba = w_rf * rf_proba + w_xgb * xgb_proba

    # Binary predictions (threshold 0.5)
    rf_pred = (rf_proba >= 0.5).astype(int)
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    metrics = {
        'use_case': use_case,
        'n_samples': len(y_test),
        'n_clearing': np.sum(y_test == 1),
        'n_intact': np.sum(y_test == 0),
        'rf': {
            'auroc': roc_auc_score(y_test, rf_proba),
            'f1': f1_score(y_test, rf_pred),
            'balanced_acc': balanced_accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
        },
        'xgb': {
            'auroc': roc_auc_score(y_test, xgb_proba),
            'f1': f1_score(y_test, xgb_pred),
            'balanced_acc': balanced_accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
        },
        'ensemble': {
            'auroc': roc_auc_score(y_test, ensemble_proba),
            'f1': f1_score(y_test, ensemble_pred),
            'balanced_acc': balanced_accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'weights': ensemble_weights,
        }
    }

    return metrics


def run_phase(phase_num, train_years, test_year, ensemble_weights=(0.5, 0.5)):
    """Run a single validation phase with ensemble."""
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}: Train {train_years} → Test {test_year}")
    print(f"Ensemble weights: RF={ensemble_weights[0]:.2f}, XGB={ensemble_weights[1]:.2f}")
    print(f"{'='*80}")

    # Load training data
    train_samples = load_training_data(train_years)
    X_train, y_train = extract_features(train_samples)

    # Train both models
    rf_model, xgb_model = train_models(X_train, y_train)

    # Load validation data
    val_data = load_validation_data(test_year)

    # Evaluate on each use case
    results = []

    print(f"\nEvaluating on {test_year} validation sets:")
    print(f"{'='*80}")

    for use_case, samples in val_data.items():
        X_test, y_test = extract_features(samples)
        metrics = evaluate_ensemble(rf_model, xgb_model, X_test, y_test, use_case, ensemble_weights)
        results.append(metrics)

        print(f"\n{use_case.upper()}")
        print(f"  Samples: {metrics['n_samples']} ({metrics['n_clearing']} clearing, {metrics['n_intact']} intact)")
        print(f"  RF      - AUROC: {metrics['rf']['auroc']:.3f}, F1: {metrics['rf']['f1']:.3f}, Bal-Acc: {metrics['rf']['balanced_acc']:.3f}")
        print(f"  XGB     - AUROC: {metrics['xgb']['auroc']:.3f}, F1: {metrics['xgb']['f1']:.3f}, Bal-Acc: {metrics['xgb']['balanced_acc']:.3f}")
        print(f"  Ensemble- AUROC: {metrics['ensemble']['auroc']:.3f}, F1: {metrics['ensemble']['f1']:.3f}, Bal-Acc: {metrics['ensemble']['balanced_acc']:.3f}")

        # Show improvement
        best_individual = max(metrics['rf']['auroc'], metrics['xgb']['auroc'])
        improvement = metrics['ensemble']['auroc'] - best_individual
        print(f"  → Improvement: {improvement:+.3f} vs best individual")

    # Overall metrics
    total_samples = sum(r['n_samples'] for r in results)
    avg_rf_auroc = np.mean([r['rf']['auroc'] for r in results])
    avg_xgb_auroc = np.mean([r['xgb']['auroc'] for r in results])
    avg_ensemble_auroc = np.mean([r['ensemble']['auroc'] for r in results])
    avg_ensemble_f1 = np.mean([r['ensemble']['f1'] for r in results])
    avg_ensemble_bal_acc = np.mean([r['ensemble']['balanced_acc'] for r in results])

    print(f"\n{'='*80}")
    print(f"PHASE {phase_num} SUMMARY")
    print(f"{'='*80}")
    print(f"Total validation samples: {total_samples}")
    print(f"RF Average AUROC:       {avg_rf_auroc:.3f}")
    print(f"XGB Average AUROC:      {avg_xgb_auroc:.3f}")
    print(f"Ensemble Average AUROC: {avg_ensemble_auroc:.3f}")
    print(f"Ensemble F1:            {avg_ensemble_f1:.3f}")
    print(f"Ensemble Balanced Acc:  {avg_ensemble_bal_acc:.3f}")

    best_individual = max(avg_rf_auroc, avg_xgb_auroc)
    improvement = avg_ensemble_auroc - best_individual
    print(f"\n→ Ensemble improvement: {improvement:+.3f} ({improvement/best_individual*100:+.1f}%)")

    return {
        'phase': phase_num,
        'train_years': train_years,
        'test_year': test_year,
        'ensemble_weights': ensemble_weights,
        'results': results,
        'summary': {
            'total_samples': total_samples,
            'avg_rf_auroc': avg_rf_auroc,
            'avg_xgb_auroc': avg_xgb_auroc,
            'avg_ensemble_auroc': avg_ensemble_auroc,
            'avg_ensemble_f1': avg_ensemble_f1,
            'avg_ensemble_bal_acc': avg_ensemble_bal_acc,
            'improvement': improvement
        }
    }


def main():
    """Run ensemble validation across all phases."""
    print("="*80)
    print("MODEL ENSEMBLE (RF + XGB) ON HARD VALIDATION SETS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Test with simple average (0.5, 0.5)
    print("\n" + "="*80)
    print("TESTING SIMPLE AVERAGE ENSEMBLE (0.5 RF + 0.5 XGB)")
    print("="*80)

    # Phase 1: Train 2020-2021 → Test 2022
    phase1 = run_phase(1, [2020, 2021], 2022, ensemble_weights=(0.5, 0.5))
    all_results.append(phase1)

    # Phase 2: Train 2020-2022 → Test 2023
    phase2 = run_phase(2, [2020, 2021, 2022], 2023, ensemble_weights=(0.5, 0.5))
    all_results.append(phase2)

    # Phase 3: Train 2020-2021+2023 → Test 2024
    phase3 = run_phase(3, [2020, 2021, 2023], 2024, ensemble_weights=(0.5, 0.5))
    all_results.append(phase3)

    # Phase 4: Train 2020-2023 → Test 2024
    phase4 = run_phase(4, [2020, 2021, 2022, 2023], 2024, ensemble_weights=(0.5, 0.5))
    all_results.append(phase4)

    # Final summary
    print(f"\n{'='*80}")
    print("ENSEMBLE VALIDATION SUMMARY")
    print(f"{'='*80}")

    for result in all_results:
        print(f"\nPhase {result['phase']}: Train {result['train_years']} → Test {result['test_year']}")
        print(f"  Samples: {result['summary']['total_samples']}")
        print(f"  RF:       {result['summary']['avg_rf_auroc']:.3f}")
        print(f"  XGB:      {result['summary']['avg_xgb_auroc']:.3f}")
        print(f"  Ensemble: {result['summary']['avg_ensemble_auroc']:.3f} ({result['summary']['improvement']:+.3f})")

    # Overall improvement
    avg_improvement = np.mean([r['summary']['improvement'] for r in all_results])
    print(f"\n{'='*80}")
    print(f"Average ensemble improvement: {avg_improvement:+.3f} AUROC")
    print(f"{'='*80}")

    # Save results
    output_file = PROCESSED_DIR / 'model_ensemble_hard_sets_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Ensemble validation complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
