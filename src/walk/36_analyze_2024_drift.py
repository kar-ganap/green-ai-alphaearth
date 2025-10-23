#!/usr/bin/env python3
"""
Analyze 2024 Temporal Drift

Investigates the 18.9% ROC-AUC drop in Phase 4 by comparing feature
distributions between 2020-2023 training data and 2024 test data.

Usage:
    uv run python src/walk/36_analyze_2024_drift.py
"""

import pickle
import numpy as np
from pathlib import Path
from scipy import stats
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_config


def load_training_data():
    """Load 2020-2023 training data."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(processed_dir.glob(pattern))
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    print(f"Loading training data: {latest_file.name}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    print(f"  Loaded {len(samples)} samples\n")
    return samples


def load_2024_data():
    """Load 2024 test data."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    pattern = 'walk_dataset_2024_with_features_*.pkl'
    files = list(processed_dir.glob(pattern))
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    print(f"Loading 2024 data: {latest_file.name}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('samples', data.get('data', data))
    print(f"  Loaded {len(samples)} samples\n")
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


def analyze_feature_distributions(X_train, y_train, X_test, y_test):
    """Compare feature distributions between training and test sets."""
    print("="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)
    print()

    n_features = X_train.shape[1]

    # Overall statistics
    print(f"Training samples: {len(X_train)} (clearing: {sum(y_train)}, intact: {sum(y_train == 0)})")
    print(f"Test samples:     {len(X_test)} (clearing: {sum(y_test)}, intact: {sum(y_test == 0)})")
    print()

    # Feature names
    feature_names = (
        ['annual_delta_1yr', 'annual_delta_2yr', 'annual_accel'] +
        [f'coarse_emb_{i}' for i in range(64)] +
        ['coarse_heterogeneity', 'coarse_range']
    )

    print("-"*80)
    print("DISTRIBUTION SHIFT ANALYSIS (Kolmogorov-Smirnov Test)")
    print("-"*80)
    print()

    significant_shifts = []

    for i in range(n_features):
        # KS test
        statistic, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])

        # Significant shift if p < 0.01
        if p_value < 0.01:
            train_mean = np.mean(X_train[:, i])
            train_std = np.std(X_train[:, i])
            test_mean = np.mean(X_test[:, i])
            test_std = np.std(X_test[:, i])

            significant_shifts.append({
                'feature': feature_names[i],
                'index': i,
                'ks_statistic': statistic,
                'p_value': p_value,
                'train_mean': train_mean,
                'train_std': train_std,
                'test_mean': test_mean,
                'test_std': test_std,
                'mean_shift': test_mean - train_mean,
                'mean_shift_pct': ((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
            })

    # Sort by KS statistic (largest shifts first)
    significant_shifts.sort(key=lambda x: x['ks_statistic'], reverse=True)

    print(f"Found {len(significant_shifts)} features with significant distribution shifts (p < 0.01)")
    print()

    if significant_shifts:
        print("Top 10 features with largest distribution shifts:")
        print()
        print(f"{'Feature':<25} {'KS Stat':<10} {'Train Mean':<12} {'Test Mean':<12} {'Shift %':<10}")
        print("-"*80)

        for shift in significant_shifts[:10]:
            print(f"{shift['feature']:<25} "
                  f"{shift['ks_statistic']:<10.3f} "
                  f"{shift['train_mean']:<12.3f} "
                  f"{shift['test_mean']:<12.3f} "
                  f"{shift['mean_shift_pct']:>9.1f}%")

    print()
    print("-"*80)
    print("SAMPLE QUALITY ANALYSIS")
    print("-"*80)
    print()

    # Check for missing/invalid values
    train_nan = np.isnan(X_train).sum()
    test_nan = np.isnan(X_test).sum()
    train_inf = np.isinf(X_train).sum()
    test_inf = np.isinf(X_test).sum()

    print(f"Training data:")
    print(f"  NaN values: {train_nan}")
    print(f"  Inf values: {train_inf}")
    print()
    print(f"Test data:")
    print(f"  NaN values: {test_nan}")
    print(f"  Inf values: {test_inf}")
    print()

    # Feature magnitude comparison
    print("-"*80)
    print("FEATURE MAGNITUDE COMPARISON")
    print("-"*80)
    print()

    print(f"{'Feature':<25} {'Train Range':<25} {'Test Range':<25} {'Overlap':<10}")
    print("-"*80)

    feature_groups = [
        ('Annual Features', [0, 1, 2]),
        ('Coarse Embeddings (sample)', list(range(3, 13))),  # First 10
        ('Coarse Stats', [67, 68])
    ]

    for group_name, indices in feature_groups:
        print(f"\n{group_name}:")
        for i in indices:
            train_min, train_max = X_train[:, i].min(), X_train[:, i].max()
            test_min, test_max = X_test[:, i].min(), X_test[:, i].max()

            # Calculate overlap
            overlap_min = max(train_min, test_min)
            overlap_max = min(train_max, test_max)
            if overlap_max > overlap_min:
                overlap_pct = ((overlap_max - overlap_min) /
                              (max(train_max, test_max) - min(train_min, test_min)) * 100)
            else:
                overlap_pct = 0

            print(f"  {feature_names[i]:<23} "
                  f"[{train_min:>6.2f}, {train_max:>6.2f}]   "
                  f"[{test_min:>6.2f}, {test_max:>6.2f}]   "
                  f"{overlap_pct:>7.1f}%")

    return significant_shifts


def analyze_class_separation(X_train, y_train, X_test, y_test):
    """Analyze how well classes are separated in training vs test."""
    print()
    print("-"*80)
    print("CLASS SEPARATION ANALYSIS")
    print("-"*80)
    print()

    # For each feature, compute mean difference between classes
    print(f"{'Feature':<25} {'Train Sep':<12} {'Test Sep':<12} {'Ratio':<10}")
    print("-"*80)

    feature_names = (
        ['annual_delta_1yr', 'annual_delta_2yr', 'annual_accel'] +
        [f'coarse_emb_{i}' for i in range(64)] +
        ['coarse_heterogeneity', 'coarse_range']
    )

    separation_changes = []

    for i in range(min(10, X_train.shape[1])):  # First 10 features
        # Training separation
        train_clearing = X_train[y_train == 1, i]
        train_intact = X_train[y_train == 0, i]
        train_sep = abs(train_clearing.mean() - train_intact.mean())

        # Test separation
        test_clearing = X_test[y_test == 1, i]
        test_intact = X_test[y_test == 0, i]
        test_sep = abs(test_clearing.mean() - test_intact.mean())

        ratio = test_sep / train_sep if train_sep > 0 else 0

        separation_changes.append({
            'feature': feature_names[i],
            'train_sep': train_sep,
            'test_sep': test_sep,
            'ratio': ratio
        })

        print(f"{feature_names[i]:<25} {train_sep:<12.3f} {test_sep:<12.3f} {ratio:<10.2f}")

    return separation_changes


def main():
    print("="*80)
    print("2024 TEMPORAL DRIFT ANALYSIS")
    print("="*80)
    print()

    # Load data
    train_samples = load_training_data()
    test_samples = load_2024_data()

    # Extract features
    print("Extracting features...")
    X_train, y_train = extract_features(train_samples)
    X_test, y_test = extract_features(test_samples)
    print(f"  Training features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")
    print()

    # Analyze distributions
    significant_shifts = analyze_feature_distributions(X_train, y_train, X_test, y_test)

    # Analyze class separation
    separation_changes = analyze_class_separation(X_train, y_train, X_test, y_test)

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    print(f"1. Distribution Shifts: {len(significant_shifts)} features show significant drift (p < 0.01)")
    print()

    if significant_shifts:
        top_3 = significant_shifts[:3]
        print("   Top 3 drifted features:")
        for i, shift in enumerate(top_3, 1):
            print(f"   {i}. {shift['feature']}: KS={shift['ks_statistic']:.3f}, "
                  f"mean shift {shift['mean_shift_pct']:+.1f}%")

    print()
    print("2. Potential Root Causes:")
    print()

    if len(significant_shifts) > 30:
        print("   ⚠️  WIDESPREAD DRIFT: >30 features affected")
        print("       → Likely systematic issue (data quality, collection method)")
    elif len(significant_shifts) > 10:
        print("   ⚠️  MODERATE DRIFT: 10-30 features affected")
        print("       → Possible environmental/temporal changes")
    else:
        print("   ✓ LIMITED DRIFT: <10 features affected")
        print("       → Small sample size may be amplifying noise")

    print()
    print("3. Recommendations:")
    print()

    if len(significant_shifts) > 30:
        print("   → Review 2024 data collection process")
        print("   → Check for systematic errors in feature extraction")
        print("   → Consider collecting more 2024 samples to verify patterns")
    else:
        print("   → Small test set (162 samples) may not be representative")
        print("   → Proceed with production model training on full 2020-2024 dataset")
        print("   → Monitor production performance closely")

    print()


if __name__ == '__main__':
    main()
