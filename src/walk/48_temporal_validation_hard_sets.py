#!/usr/bin/env python3
"""
Temporal Validation with Hard Validation Sets

Runs 4 validation phases using collected hard validation samples:
- Phase 1: Train 2020-2021 → Test 2022 (109 samples)
- Phase 2: Train 2020-2022 → Test 2023 (106 samples)
- Phase 3: Train 2020-2021+2023 → Test 2024 (125 samples)
- Phase 4: Train 2020-2023 → Test 2024 (125 samples)

Tests across 3 use cases: risk_ranking, comprehensive, rapid_response
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import get_config

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
    """Extract 70D features from samples using Phase B method."""
    X = []
    y = []

    for sample in samples:
        # Extract annual features (3D)
        annual_features = sample.get('annual_features')
        if annual_features is None:
            raise ValueError(f"Sample missing annual_features: {sample.get('lat', 'unknown')}, {sample.get('lon', 'unknown')}")
        annual_features = np.array(annual_features).flatten()

        # Extract coarse features (66D) from multiscale_features dict
        multiscale_dict = sample.get('multiscale_features', {})
        if not isinstance(multiscale_dict, dict):
            raise ValueError(f"multiscale_features must be a dict, got {type(multiscale_dict)}")

        # Define feature names in correct order: 64 embeddings + 2 stats
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])

        # Extract or compute year feature (1D)
        year = sample.get('year', 2021)
        year_feature = (year - 2020) / 4.0  # Normalize to [0,1] for range 2020-2024

        # Combine: 3D + 66D + 1D = 70D
        combined = np.concatenate([annual_features, coarse_features, [year_feature]])
        X.append(combined)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)


def train_model(X_train, y_train):
    """Train Random Forest model."""
    print(f"\nTraining Random Forest...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Clearing samples: {np.sum(y_train == 1)}")
    print(f"  Intact samples: {np.sum(y_train == 0)}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    print(f"  ✓ Model trained")

    return model


def evaluate_model(model, X_test, y_test, use_case):
    """Evaluate model and return metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'use_case': use_case,
        'n_samples': len(y_test),
        'n_clearing': np.sum(y_test == 1),
        'n_intact': np.sum(y_test == 0),
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }

    return metrics


def run_phase(phase_num, train_years, test_year):
    """Run a single validation phase."""
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}: Train {train_years} → Test {test_year}")
    print(f"{'='*80}")

    # Load training data
    train_samples = load_training_data(train_years)
    X_train, y_train = extract_features(train_samples)

    # Train model
    model = train_model(X_train, y_train)

    # Load validation data
    val_data = load_validation_data(test_year)

    # Evaluate on each use case
    results = []

    print(f"\nEvaluating on {test_year} validation sets:")
    print(f"{'='*80}")

    for use_case, samples in val_data.items():
        X_test, y_test = extract_features(samples)
        metrics = evaluate_model(model, X_test, y_test, use_case)
        results.append(metrics)

        print(f"\n{use_case.upper()}")
        print(f"  Samples: {metrics['n_samples']} ({metrics['n_clearing']} clearing, {metrics['n_intact']} intact)")
        print(f"  AUROC: {metrics['auroc']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Balanced Acc: {metrics['balanced_acc']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")

    # Overall metrics
    total_samples = sum(r['n_samples'] for r in results)
    avg_auroc = np.mean([r['auroc'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_balanced_acc = np.mean([r['balanced_acc'] for r in results])

    print(f"\n{'='*80}")
    print(f"PHASE {phase_num} SUMMARY")
    print(f"{'='*80}")
    print(f"Total validation samples: {total_samples}")
    print(f"Average AUROC: {avg_auroc:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Average Balanced Acc: {avg_balanced_acc:.3f}")

    return {
        'phase': phase_num,
        'train_years': train_years,
        'test_year': test_year,
        'results': results,
        'summary': {
            'total_samples': total_samples,
            'avg_auroc': avg_auroc,
            'avg_f1': avg_f1,
            'avg_balanced_acc': avg_balanced_acc
        }
    }


def main():
    """Run all temporal validation phases."""
    print("="*80)
    print("TEMPORAL VALIDATION WITH HARD VALIDATION SETS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Phase 1: Train 2020-2021 → Test 2022
    phase1 = run_phase(1, [2020, 2021], 2022)
    all_results.append(phase1)

    # Phase 2: Train 2020-2022 → Test 2023
    phase2 = run_phase(2, [2020, 2021, 2022], 2023)
    all_results.append(phase2)

    # Phase 3: Train 2020-2021+2023 → Test 2024
    phase3 = run_phase(3, [2020, 2021, 2023], 2024)
    all_results.append(phase3)

    # Phase 4: Train 2020-2023 → Test 2024
    phase4 = run_phase(4, [2020, 2021, 2022, 2023], 2024)
    all_results.append(phase4)

    # Final summary
    print(f"\n{'='*80}")
    print("TEMPORAL VALIDATION SUMMARY")
    print(f"{'='*80}")

    for result in all_results:
        print(f"\nPhase {result['phase']}: Train {result['train_years']} → Test {result['test_year']}")
        print(f"  Samples: {result['summary']['total_samples']}")
        print(f"  AUROC: {result['summary']['avg_auroc']:.3f}")
        print(f"  F1: {result['summary']['avg_f1']:.3f}")
        print(f"  Balanced Acc: {result['summary']['avg_balanced_acc']:.3f}")

    # Save results
    output_file = PROCESSED_DIR / 'temporal_validation_hard_sets_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Temporal validation complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
