"""
Temporal Generalization Experiment

Tests whether models trained on past years can predict future deforestation.
This is the critical validation for operational deployment readiness.

Design: Progressive temporal validation
- Fold 1: Train on 2020 → Test on 2021 (1 year gap)
- Fold 2: Train on 2020-2021 → Test on 2022 (1-2 year gap)
- Fold 3: Train on 2020-2022 → Test on 2023 (1-3 year gap)

Usage:
    python src/walk/06_temporal_generalization_experiment.py --n-per-year 80
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from src.utils import EarthEngineClient, get_config


def compute_baseline_features(embeddings_dict):
    """
    Compute baseline features from Y-1 quarterly embeddings.

    Returns:
        10-dimensional feature vector
    """
    Q1, Q2, Q3, Q4 = embeddings_dict['Q1'], embeddings_dict['Q2'], embeddings_dict['Q3'], embeddings_dict['Q4']

    # Distances between consecutive quarters
    d12 = np.linalg.norm(Q2 - Q1)
    d23 = np.linalg.norm(Q3 - Q2)
    d34 = np.linalg.norm(Q4 - Q3)

    # Velocities
    v12 = d12
    v23 = d23
    v34 = d34

    # Accelerations
    a123 = v23 - v12
    a234 = v34 - v23

    # Trend
    trend = 1 if a234 > 0 else (-1 if a234 < 0 else 0)

    # Overall trajectory
    total_distance = d12 + d23 + d34

    return np.array([d12, d23, d34, v12, v23, v34, a123, a234, trend, total_distance])


def compute_delta_features(delta_embeddings_list):
    """
    Compute delta features from quarterly delta embeddings.

    Returns:
        7-dimensional feature vector
    """
    magnitudes = [np.linalg.norm(delta) for delta in delta_embeddings_list]

    mean_magnitude = np.mean(magnitudes)
    max_magnitude = np.max(magnitudes)

    if len(magnitudes) >= 2:
        trend = 1 if magnitudes[-1] > magnitudes[0] else (-1 if magnitudes[-1] < magnitudes[0] else 0)
    else:
        trend = 0

    features = magnitudes + [mean_magnitude, max_magnitude, trend]

    return np.array(features)


def extract_dual_year_features(client, sample, year):
    """
    Extract dual-year delta features for a sample.

    Args:
        client: EarthEngineClient
        sample: Dict with lat, lon
        year: Year for which to extract features (Y)

    Returns:
        17-dimensional feature vector or None
    """
    try:
        lat, lon = sample['lat'], sample['lon']

        # Extract Y-1 quarterly embeddings
        baseline = {}
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year-1}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None:
                return None
            baseline[f'Q{q}'] = emb

        # Extract Y quarterly embeddings
        current = {}
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None:
                return None
            current[f'Q{q}'] = emb

        # Compute delta
        delta = [current[f'Q{q}'] - baseline[f'Q{q}'] for q in [1, 2, 3, 4]]

        # Compute features
        baseline_features = compute_baseline_features(baseline)
        delta_features = compute_delta_features(delta)

        return np.concatenate([baseline_features, delta_features])

    except Exception as e:
        return None


def collect_yearly_clearings(client, year, n_samples=80):
    """
    Collect clearing samples for a specific year.

    Args:
        client: EarthEngineClient
        year: Year to collect clearings from
        n_samples: Target number of samples

    Returns:
        List of clearing dicts with lat, lon, year
    """
    print(f"\n  Collecting clearings for {year}...")

    config = get_config()
    main_bounds = config.study_region_bounds

    # Split region for diversity
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": main_bounds["min_lat"], "max_lat": mid_lat},
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": main_bounds["min_lat"], "max_lat": mid_lat},
    ]

    all_clearings = []

    for bounds in sub_regions:
        try:
            clearings = client.get_deforestation_labels(
                bounds=bounds,
                year=year,
                min_tree_cover=30,
            )
            all_clearings.extend(clearings)
        except Exception as e:
            print(f"    Warning: Failed to get clearings from sub-region: {e}")
            continue

    # Sample if too many
    if len(all_clearings) > n_samples:
        import random
        random.seed(42 + year)  # Different seed per year
        all_clearings = random.sample(all_clearings, n_samples)

    # Add year field
    for clearing in all_clearings:
        clearing['year'] = year

    print(f"    ✓ Collected {len(all_clearings)} clearings")

    return all_clearings


def collect_yearly_intact(year, n_samples=80):
    """
    Generate intact forest samples for a specific year.

    Args:
        year: Year to assign to samples
        n_samples: Number of samples to generate

    Returns:
        List of intact sample dicts
    """
    print(f"  Generating intact samples for {year}...")

    intact_regions = [
        {"name": "Amazon Core", "bounds": {"min_lon": -60, "max_lon": -55,
                                           "min_lat": -5, "max_lat": 0}},
        {"name": "Guiana Shield", "bounds": {"min_lon": -55, "max_lon": -50,
                                             "min_lat": 2, "max_lat": 6}},
        {"name": "Central Amazon", "bounds": {"min_lon": -65, "max_lon": -60,
                                              "min_lat": -2, "max_lat": 2}},
    ]

    intact_samples = []
    samples_per_region = n_samples // len(intact_regions)

    np.random.seed(42 + year)  # Different seed per year

    for region in intact_regions:
        bounds = region['bounds']

        for _ in range(samples_per_region):
            lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
            lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])

            intact_samples.append({
                'lat': lat,
                'lon': lon,
                'year': year,
            })

    # Fill remaining samples
    while len(intact_samples) < n_samples:
        region = intact_regions[len(intact_samples) % len(intact_regions)]
        bounds = region['bounds']
        lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
        lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
        intact_samples.append({'lat': lat, 'lon': lon, 'year': year})

    print(f"    ✓ Generated {len(intact_samples)} intact samples")

    return intact_samples


def extract_features_for_year(client, clearings, intact, year):
    """
    Extract dual-year features for all samples from a specific year.

    Args:
        client: EarthEngineClient
        clearings: List of clearing samples
        intact: List of intact samples
        year: Year (used for Y embeddings, Y-1 for baseline)

    Returns:
        Tuple of (X, y, n_clearing, n_intact)
    """
    print(f"\n  Extracting features for {year} samples...")

    X_clearing = []
    for clearing in clearings:
        features = extract_dual_year_features(client, clearing, year)
        if features is not None:
            X_clearing.append(features)

    X_intact = []
    for sample in intact:
        features = extract_dual_year_features(client, sample, year)
        if features is not None:
            X_intact.append(features)

    print(f"    Clearing: {len(X_clearing)}/{len(clearings)} extracted")
    print(f"    Intact: {len(X_intact)}/{len(intact)} extracted")

    if len(X_clearing) == 0 or len(X_intact) == 0:
        return None, None, 0, 0

    X = np.vstack([X_clearing, X_intact])
    y = np.concatenate([np.ones(len(X_clearing)), np.zeros(len(X_intact))])

    return X, y, len(X_clearing), len(X_intact)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train model and evaluate performance.

    Returns:
        Dict with metrics and model info
    """
    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Feature importance
    feature_importance = np.abs(model.coef_[0])
    baseline_importance = feature_importance[:10]
    delta_importance = feature_importance[10:]

    return {
        'roc_auc': float(roc_auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'feature_importance': feature_importance.tolist(),
        'baseline_importance_mean': float(np.mean(baseline_importance)),
        'delta_importance_mean': float(np.mean(delta_importance)),
    }


def run_temporal_fold(client, train_years, test_year, yearly_data):
    """
    Run one fold of temporal validation.

    Args:
        client: EarthEngineClient
        train_years: List of years to train on
        test_year: Single year to test on
        yearly_data: Dict mapping year -> (clearings, intact)

    Returns:
        Dict with fold results
    """
    fold_name = f"{'_'.join(map(str, train_years))}→{test_year}"
    print(f"\n{'='*60}")
    print(f"FOLD: {fold_name}")
    print(f"{'='*60}")

    # Collect training data from all training years
    X_train_list = []
    y_train_list = []
    train_clearing_counts = []
    train_intact_counts = []

    for year in train_years:
        if year not in yearly_data:
            print(f"Warning: Year {year} not in collected data, skipping")
            continue

        clearings, intact = yearly_data[year]
        X, y, n_clearing, n_intact = extract_features_for_year(client, clearings, intact, year)

        if X is not None:
            X_train_list.append(X)
            y_train_list.append(y)
            train_clearing_counts.append(n_clearing)
            train_intact_counts.append(n_intact)

    if len(X_train_list) == 0:
        print("Error: No training data extracted")
        return None

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    print(f"\nTraining set:")
    print(f"  Years: {train_years}")
    print(f"  Samples: {len(X_train)} ({sum(train_clearing_counts)} clearing, {sum(train_intact_counts)} intact)")

    # Collect test data
    if test_year not in yearly_data:
        print(f"Error: Test year {test_year} not in collected data")
        return None

    test_clearings, test_intact = yearly_data[test_year]
    X_test, y_test, test_clearing_count, test_intact_count = extract_features_for_year(
        client, test_clearings, test_intact, test_year
    )

    if X_test is None:
        print("Error: No test data extracted")
        return None

    print(f"\nTest set:")
    print(f"  Year: {test_year}")
    print(f"  Samples: {len(X_test)} ({test_clearing_count} clearing, {test_intact_count} intact)")

    # Train and evaluate
    print(f"\nTraining and evaluating...")
    metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

    print(f"\nResults:")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")

    return {
        'fold_name': fold_name,
        'train_years': train_years,
        'test_year': test_year,
        'train_samples': {
            'total': len(X_train),
            'clearing': sum(train_clearing_counts),
            'intact': sum(train_intact_counts),
        },
        'test_samples': {
            'total': len(X_test),
            'clearing': test_clearing_count,
            'intact': test_intact_count,
        },
        'metrics': metrics,
    }


def interpret_temporal_generalization(fold_results):
    """
    Interpret temporal generalization results.

    Args:
        fold_results: List of fold result dicts

    Returns:
        Dict with interpretation
    """
    print(f"\n{'='*60}")
    print("TEMPORAL GENERALIZATION ANALYSIS")
    print(f"{'='*60}")

    # Extract ROC-AUC for each fold
    fold_aucs = [(r['fold_name'], r['metrics']['roc_auc']) for r in fold_results]

    print(f"\nFold Performance:")
    for fold_name, auc in fold_aucs:
        print(f"  {fold_name}: {auc:.3f} ROC-AUC")

    # Average performance
    avg_auc = np.mean([auc for _, auc in fold_aucs])
    std_auc = np.std([auc for _, auc in fold_aucs])

    print(f"\nAggregate:")
    print(f"  Mean ROC-AUC: {avg_auc:.3f} (±{std_auc:.3f})")

    # Compare to current validation baseline
    baseline_auc = 0.75  # From honest_performance_evaluation.md
    degradation = baseline_auc - avg_auc
    degradation_pct = (degradation / baseline_auc) * 100

    print(f"\nComparison to Current Validation:")
    print(f"  Current validation (mixed years): ~0.75 ROC-AUC")
    print(f"  Temporal split (held-out years):   {avg_auc:.3f} ROC-AUC")
    print(f"  Degradation: {degradation:+.3f} ({degradation_pct:+.1f}%)")

    # Interpretation
    if abs(degradation_pct) < 5:
        status = "STRONG_GENERALIZATION"
        interpretation = f"""
✓ STRONG TEMPORAL GENERALIZATION (<5% degradation)

The model achieves {avg_auc:.3f} ROC-AUC when trained on past years and tested on future years,
compared to {baseline_auc:.3f} with mixed-year validation. This {degradation_pct:.1f}% difference
is negligible.

Interpretation:
  ✓ Model captures transferable deforestation patterns
  ✓ Not overfitting to year-specific artifacts
  ✓ Ready for operational deployment with confidence
  ✓ Expected to maintain ~{avg_auc:.2f} ROC-AUC on 2024-2025 data

Recommendation: PROCEED TO DEPLOYMENT
"""
    elif abs(degradation_pct) < 15:
        status = "MODERATE_GENERALIZATION"
        interpretation = f"""
~ MODERATE TEMPORAL GENERALIZATION (5-15% degradation)

The model achieves {avg_auc:.3f} ROC-AUC on held-out future years, compared to {baseline_auc:.3f}
with mixed-year validation. This {degradation_pct:.1f}% degradation suggests some overfitting
to year-specific patterns.

Interpretation:
  ~ Model captures some generalizable patterns but also year-specific signals
  ~ Performance likely to degrade 10-15% in operational deployment
  ~ Need more training data or temporal stability features

Recommendation:
  - Collect more samples (target 200+ per year)
  - Add year-invariant features (e.g., spatial context, long-term trends)
  - Consider ensemble models across years
  - Deploy with conservative thresholds
"""
    else:
        status = "POOR_GENERALIZATION"
        interpretation = f"""
✗ POOR TEMPORAL GENERALIZATION (>15% degradation)

The model achieves only {avg_auc:.3f} ROC-AUC on held-out future years, compared to
{baseline_auc:.3f} with mixed-year validation. This {degradation_pct:.1f}% degradation
indicates significant overfitting to year-specific patterns.

Interpretation:
  ✗ Model overfitting to temporal artifacts specific to training years
  ✗ Current validation metrics are misleading
  ✗ NOT ready for operational deployment

Recommendation:
  - Investigate year-specific features (what changed between years?)
  - Add temporal regularization or domain adaptation
  - Consider different modeling approach (e.g., time-aware models)
  - Collect significantly more data
  - DO NOT deploy until temporal generalization improves
"""

    print(f"\n{interpretation}")
    print(f"{'='*60}")

    return {
        'status': status,
        'fold_results': fold_aucs,
        'mean_auc': avg_auc,
        'std_auc': std_auc,
        'baseline_auc': baseline_auc,
        'degradation': degradation,
        'degradation_pct': degradation_pct,
        'interpretation': interpretation.strip(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Temporal generalization experiment: train on past, test on future'
    )
    parser.add_argument('--n-per-year', type=int, default=80,
                        help='Number of samples per year (default: 80)')
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021, 2022],
                        help='Years to collect data for (default: 2020 2021 2022)')

    args = parser.parse_args()

    print(f"{'='*60}")
    print("TEMPORAL GENERALIZATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Samples per year: {args.n_per_year}")
    print(f"  Years: {args.years}")

    # Initialize client
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Step 1: Collect data for each year
    print(f"\n{'='*60}")
    print("STEP 1: COLLECTING YEARLY DATASETS")
    print(f"{'='*60}")

    yearly_data = {}

    for year in args.years:
        print(f"\nYear {year}:")
        clearings = collect_yearly_clearings(client, year, args.n_per_year)
        intact = collect_yearly_intact(year, args.n_per_year)
        yearly_data[year] = (clearings, intact)

    # Step 2: Run progressive temporal folds
    print(f"\n{'='*60}")
    print("STEP 2: PROGRESSIVE TEMPORAL VALIDATION")
    print(f"{'='*60}")

    # Define folds
    folds = []
    if 2020 in args.years and 2021 in args.years:
        folds.append(([2020], 2021))
    if 2020 in args.years and 2021 in args.years and 2022 in args.years:
        folds.append(([2020, 2021], 2022))
    if 2020 in args.years and 2021 in args.years and 2022 in args.years and 2023 in args.years:
        folds.append(([2020, 2021, 2022], 2023))

    if len(folds) == 0:
        print("Error: Need at least 2 consecutive years for temporal validation")
        return

    fold_results = []

    for train_years, test_year in folds:
        result = run_temporal_fold(client, train_years, test_year, yearly_data)
        if result is not None:
            fold_results.append(result)

    if len(fold_results) == 0:
        print("Error: No folds completed successfully")
        return

    # Step 3: Interpret results
    print(f"\n{'='*60}")
    print("STEP 3: INTERPRETATION")
    print(f"{'='*60}")

    interpretation = interpret_temporal_generalization(fold_results)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'temporal_generalization',
        'parameters': {
            'n_per_year': args.n_per_year,
            'years': args.years,
        },
        'fold_results': fold_results,
        'interpretation': interpretation,
    }

    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "temporal_generalization.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
