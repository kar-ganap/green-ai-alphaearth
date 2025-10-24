#!/usr/bin/env python3
"""
Final Models Trained on All 2020-2024 Data

Trains both RF and XGB on complete dataset:
- 2020-2023: 685 samples
- 2024: 162 samples
- Total: 847 samples

Evaluates on hard validation sets (340 samples) to compare with Phase 4
which only used 2020-2023 training data.
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

# Hard validation files (separate from training - these are for testing only)
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


def load_all_training_data():
    """Load all 2020-2024 training data (847 samples)."""
    print(f"\nLoading ALL training data (2020-2024)...")

    # Load 2020-2023 data (685 samples)
    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(PROCESSED_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No 2020-2023 data found matching pattern: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"  Loading 2020-2023: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        dataset = pickle.load(f)

    samples_2020_2023 = dataset['data']
    print(f"  Loaded {len(samples_2020_2023)} samples from 2020-2023")

    # Load 2024 data (162 samples)
    file_2024 = PROCESSED_DIR / 'walk_dataset_2024_with_features_20251021_110417.pkl'
    if not file_2024.exists():
        raise FileNotFoundError(f"2024 data not found: {file_2024}")

    print(f"  Loading 2024: {file_2024.name}")

    with open(file_2024, 'rb') as f:
        dataset_2024 = pickle.load(f)

    samples_2024 = dataset_2024['samples']
    print(f"  Loaded {len(samples_2024)} samples from 2024")

    # Combine all data
    all_samples = samples_2020_2023 + samples_2024
    print(f"\n  Total training samples: {len(all_samples)} (2020-2024)")

    return all_samples


def load_all_validation_data():
    """Load all hard validation data (340 samples across all years)."""
    print(f"\nLoading ALL hard validation data...")

    all_samples = []
    by_year = {}

    for year in [2022, 2023, 2024]:
        by_year[year] = {}
        for filename, use_case in VALIDATION_FILES[year]:
            filepath = PROCESSED_DIR / filename

            if not filepath.exists():
                print(f"  ⚠️ File not found: {filename}")
                continue

            with open(filepath, 'rb') as f:
                samples = pickle.load(f)

            by_year[year][use_case] = samples
            all_samples.extend(samples)
            print(f"  {year} {use_case}: {len(samples)} samples")

    print(f"\n  Total validation samples: {len(all_samples)}")
    return all_samples, by_year


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
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL MODELS ON 2020-2024 DATA")
    print(f"{'='*80}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Clearing samples: {np.sum(y_train == 1)}")
    print(f"  Intact samples: {np.sum(y_train == 0)}")

    # Train Random Forest
    print(f"\n  Training Random Forest...")
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
    print(f"\n  Training XGBoost...")
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


def evaluate_models(rf_model, xgb_model, X_test, y_test, samples_by_year):
    """Evaluate models on hard validation sets."""
    print(f"\n{'='*80}")
    print(f"EVALUATING ON HARD VALIDATION SETS (340 SAMPLES)")
    print(f"{'='*80}")

    # Overall predictions
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    rf_pred = (rf_proba >= 0.5).astype(int)
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    # Overall metrics
    overall_metrics = {
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
        }
    }

    print(f"\nOVERALL PERFORMANCE (All 340 hard validation samples):")
    print(f"  RF  - AUROC: {overall_metrics['rf']['auroc']:.3f}, F1: {overall_metrics['rf']['f1']:.3f}, Bal-Acc: {overall_metrics['rf']['balanced_acc']:.3f}")
    print(f"  XGB - AUROC: {overall_metrics['xgb']['auroc']:.3f}, F1: {overall_metrics['xgb']['f1']:.3f}, Bal-Acc: {overall_metrics['xgb']['balanced_acc']:.3f}")

    # By year and use case
    by_year_metrics = {}

    for year in [2022, 2023, 2024]:
        print(f"\n{year} HARD VALIDATION:")
        year_metrics = []

        for use_case, samples in samples_by_year[year].items():
            X_subset, y_subset = extract_features(samples)

            rf_proba_subset = rf_model.predict_proba(X_subset)[:, 1]
            xgb_proba_subset = xgb_model.predict_proba(X_subset)[:, 1]

            rf_pred_subset = (rf_proba_subset >= 0.5).astype(int)
            xgb_pred_subset = (xgb_proba_subset >= 0.5).astype(int)

            metrics = {
                'use_case': use_case,
                'n_samples': len(y_subset),
                'rf_auroc': roc_auc_score(y_subset, rf_proba_subset),
                'xgb_auroc': roc_auc_score(y_subset, xgb_proba_subset),
                'rf_f1': f1_score(y_subset, rf_pred_subset),
                'xgb_f1': f1_score(y_subset, xgb_pred_subset),
            }
            year_metrics.append(metrics)

            print(f"  {use_case:15s} ({metrics['n_samples']:3d} samples) - RF: {metrics['rf_auroc']:.3f}, XGB: {metrics['xgb_auroc']:.3f}")

        # Year average
        avg_rf = np.mean([m['rf_auroc'] for m in year_metrics])
        avg_xgb = np.mean([m['xgb_auroc'] for m in year_metrics])
        by_year_metrics[year] = {
            'avg_rf_auroc': avg_rf,
            'avg_xgb_auroc': avg_xgb,
            'details': year_metrics
        }
        print(f"  {'Average':15s}              - RF: {avg_rf:.3f}, XGB: {avg_xgb:.3f}")

    return overall_metrics, by_year_metrics


def main():
    """Train final models and evaluate on hard validation sets."""
    print("="*80)
    print("FINAL MODELS: TRAINING ON ALL 2020-2024 DATA (847 SAMPLES)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load all training data
    train_samples = load_all_training_data()
    X_train, y_train = extract_features(train_samples)

    # Load all validation data
    val_samples, samples_by_year = load_all_validation_data()
    X_test, y_test = extract_features(val_samples)

    # Train models
    rf_model, xgb_model = train_models(X_train, y_train)

    # Evaluate
    overall_metrics, by_year_metrics = evaluate_models(
        rf_model, xgb_model, X_test, y_test, samples_by_year
    )

    # Comparison to Phase 4 (trained on 2020-2023 only)
    print(f"\n{'='*80}")
    print(f"COMPARISON TO PHASE 4 (2020-2023 training only)")
    print(f"{'='*80}")

    phase4_rf_auroc = 0.692  # From Phase 4 results
    phase4_xgb_auroc = 0.718  # From Phase 4 results

    final_rf_auroc = overall_metrics['rf']['auroc']
    final_xgb_auroc = overall_metrics['xgb']['auroc']

    print(f"\nRandom Forest:")
    print(f"  Phase 4 (2020-2023): {phase4_rf_auroc:.3f}")
    print(f"  Final (2020-2024):   {final_rf_auroc:.3f}")
    print(f"  Improvement:         {final_rf_auroc - phase4_rf_auroc:+.3f} ({(final_rf_auroc - phase4_rf_auroc)/phase4_rf_auroc*100:+.1f}%)")

    print(f"\nXGBoost:")
    print(f"  Phase 4 (2020-2023): {phase4_xgb_auroc:.3f}")
    print(f"  Final (2020-2024):   {final_xgb_auroc:.3f}")
    print(f"  Improvement:         {final_xgb_auroc - phase4_xgb_auroc:+.3f} ({(final_xgb_auroc - phase4_xgb_auroc)/phase4_xgb_auroc*100:+.1f}%)")

    # Save results
    results = {
        'training_samples': len(X_train),
        'validation_samples': len(X_test),
        'overall_metrics': overall_metrics,
        'by_year_metrics': by_year_metrics,
        'phase4_comparison': {
            'rf': {'phase4': phase4_rf_auroc, 'final': final_rf_auroc, 'improvement': final_rf_auroc - phase4_rf_auroc},
            'xgb': {'phase4': phase4_xgb_auroc, 'final': final_xgb_auroc, 'improvement': final_xgb_auroc - phase4_xgb_auroc}
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    output_file = PROCESSED_DIR / 'final_models_2020_2024_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    # Save trained models
    rf_model_file = PROCESSED_DIR / 'final_rf_model_2020_2024.pkl'
    xgb_model_file = PROCESSED_DIR / 'final_xgb_model_2020_2024.pkl'

    with open(rf_model_file, 'wb') as f:
        pickle.dump(rf_model, f)

    with open(xgb_model_file, 'wb') as f:
        pickle.dump(xgb_model, f)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ RF model saved to: {rf_model_file}")
    print(f"✓ XGB model saved to: {xgb_model_file}")
    print(f"✓ Final model training complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
