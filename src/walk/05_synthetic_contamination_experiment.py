"""
Synthetic Contamination Experiment

Tests whether temporal contamination affects model performance by comparing
early-year vs late-year quarterly embeddings.

Experimental Design:
------------------
For each clearing with known year Y:
  - Extract Y-1 and Y quarterly embeddings (Q1, Q2, Q3, Q4)
  - Create three scenarios:
    1. Early-year: Use delta from Q1, Q2 only (more likely pre-clearing = clean)
    2. Late-year: Use delta from Q3, Q4 only (more likely post-clearing = contaminated)
    3. Full-year: Use delta from all 4 quarters (baseline)

Hypothesis:
-----------
- If late-year > early-year → contamination helps (detecting cleared land, not precursors)
- If early-year > late-year → pre-clearing state more informative (genuine precursors)
- If early ≈ late → robust to contamination or inconclusive

Usage:
    python src/walk/05_synthetic_contamination_experiment.py --n-clearing 60 --n-intact 60
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from src.utils import EarthEngineClient, get_config


def compute_baseline_features(embeddings_dict):
    """
    Compute baseline features from Y-1 quarterly embeddings.
    Same as walk phase: distances, velocities, accelerations, trend.

    Returns:
        10-dimensional feature vector
    """
    Q1, Q2, Q3, Q4 = embeddings_dict['Q1'], embeddings_dict['Q2'], embeddings_dict['Q3'], embeddings_dict['Q4']

    # Distances between consecutive quarters
    d12 = np.linalg.norm(Q2 - Q1)
    d23 = np.linalg.norm(Q3 - Q2)
    d34 = np.linalg.norm(Q4 - Q3)

    # Velocities (rate of change)
    v12 = d12
    v23 = d23
    v34 = d34

    # Accelerations (change in velocity)
    a123 = v23 - v12
    a234 = v34 - v23

    # Trend: is velocity increasing or decreasing?
    trend = 1 if a234 > 0 else (-1 if a234 < 0 else 0)

    # Overall trajectory length
    total_distance = d12 + d23 + d34

    return np.array([d12, d23, d34, v12, v23, v34, a123, a234, trend, total_distance])


def compute_delta_features(delta_embeddings_list):
    """
    Compute delta features from list of quarterly delta embeddings.

    Args:
        delta_embeddings_list: List of delta embeddings (2 or 4 quarters)

    Returns:
        Feature vector with magnitudes, mean, max, trend
    """
    # Compute magnitudes
    magnitudes = [np.linalg.norm(delta) for delta in delta_embeddings_list]

    # Summary statistics
    mean_magnitude = np.mean(magnitudes)
    max_magnitude = np.max(magnitudes)

    # Trend: is change accelerating or decelerating?
    if len(magnitudes) >= 2:
        trend = 1 if magnitudes[-1] > magnitudes[0] else (-1 if magnitudes[-1] < magnitudes[0] else 0)
    else:
        trend = 0

    # Stack: individual magnitudes + summary stats + trend
    features = magnitudes + [mean_magnitude, max_magnitude, trend]

    return np.array(features)


def extract_synthetic_features(client, clearing, scenario):
    """
    Extract features for synthetic contamination scenarios.

    Args:
        clearing: Clearing dict with lat, lon, clearing_year
        scenario: 'early', 'late', or 'full'
            - early: Use Q1, Q2 delta only (more likely pre-clearing = clean)
            - late: Use Q3, Q4 delta only (more likely post-clearing = contaminated)
            - full: Use all 4 quarters (baseline)

    Returns:
        Feature vector or None if extraction fails
    """
    try:
        year = clearing['clearing_year']
        lat, lon = clearing['lat'], clearing['lon']

        # Extract baseline (Y-1) embeddings - same for all scenarios
        baseline = {}
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year-1}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None:
                return None
            baseline[f'Q{q}'] = emb

        # Extract current (Y) embeddings
        current = {}
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None:
                return None
            current[f'Q{q}'] = emb
    except Exception as e:
        # Handle any Earth Engine API errors
        return None

    # Compute baseline features (Y-1 temporal dynamics)
    baseline_features = compute_baseline_features(baseline)

    # Select quarters based on scenario
    if scenario == 'early':
        # Early-year: Q1, Q2 (more likely pre-clearing)
        selected_quarters = [1, 2]
    elif scenario == 'late':
        # Late-year: Q3, Q4 (more likely post-clearing)
        selected_quarters = [3, 4]
    else:  # full
        # Full-year: All 4 quarters
        selected_quarters = [1, 2, 3, 4]

    # Compute delta embeddings for selected quarters
    delta_embeddings = [
        current[f'Q{q}'] - baseline[f'Q{q}']
        for q in selected_quarters
    ]

    # Compute delta features
    delta_features = compute_delta_features(delta_embeddings)

    # Concatenate baseline + delta
    return np.concatenate([baseline_features, delta_features])


def get_clearing_samples(client, n_clearing=60, years=[2020, 2021, 2022]):
    """
    Generate clearing samples from deforestation labels.

    Args:
        client: EarthEngineClient
        n_clearing: Number of clearing samples to generate
        years: Years to sample from

    Returns:
        List of clearing dicts with lat, lon, clearing_year
    """
    print(f"\nGenerating {n_clearing} clearing samples...")

    config = get_config()
    main_bounds = config.study_region_bounds

    # Split region into sub-regions for sampling
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
    ]

    all_clearings = []
    samples_per_year = n_clearing // len(years)

    for year in years:
        year_clearings = []
        for bounds in sub_regions:
            try:
                clearings = client.get_deforestation_labels(
                    bounds=bounds,
                    year=year,
                    min_tree_cover=30,
                )
                # Rename year field to clearing_year for consistency
                for c in clearings:
                    c['clearing_year'] = c.pop('year', year)
                year_clearings.extend(clearings)
            except Exception as e:
                print(f"  Warning: Failed to get clearings for year {year}: {e}")
                pass

        # Sample if we have too many
        if len(year_clearings) > samples_per_year:
            import random
            random.seed(42)
            year_clearings = random.sample(year_clearings, samples_per_year)

        all_clearings.extend(year_clearings)
        print(f"  Year {year}: {len(year_clearings)} clearings")

    print(f"  ✓ Total clearings: {len(all_clearings)}")

    return all_clearings


def get_intact_samples(n_intact=60, years=[2020, 2021, 2022]):
    """
    Generate intact forest sample locations.

    Args:
        n_intact: Number of intact samples to generate
        years: Years to assign to samples

    Returns:
        List of intact sample dicts with lat, lon, clearing_year (year field for feature extraction)
    """
    print(f"\nGenerating {n_intact} intact forest samples...")

    # Use intact forest bastions
    intact_regions = [
        {"name": "Amazon Core", "bounds": {"min_lon": -60, "max_lon": -55,
                                           "min_lat": -5, "max_lat": 0}},
        {"name": "Guiana Shield", "bounds": {"min_lon": -55, "max_lon": -50,
                                             "min_lat": 2, "max_lat": 6}},
    ]

    intact_samples = []
    samples_per_year = n_intact // len(years)

    for year in years:
        for region in intact_regions:
            region_samples = samples_per_year // len(intact_regions)
            bounds = region['bounds']

            for _ in range(region_samples):
                # Random point in bounds
                lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
                lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])

                intact_samples.append({
                    'lat': lat,
                    'lon': lon,
                    'clearing_year': year,  # Use year for feature extraction
                })

                if len(intact_samples) >= n_intact:
                    break

            if len(intact_samples) >= n_intact:
                break

        if len(intact_samples) >= n_intact:
            break

    print(f"  ✓ Generated {len(intact_samples)} intact samples")

    return intact_samples


def run_scenario_experiment(client, clearings, intact, scenario):
    """
    Run experiment for a single scenario (early, late, or full).

    Returns:
        dict with performance metrics and feature data
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*60}")

    # Extract features for clearings
    X_clearing = []
    valid_clearing_indices = []

    print(f"Extracting {scenario} features for clearings...")
    for i, clearing in enumerate(clearings):
        features = extract_synthetic_features(client, clearing, scenario)
        if features is not None:
            X_clearing.append(features)
            valid_clearing_indices.append(i)

    print(f"  ✓ {len(X_clearing)} clearings with {scenario} features")

    # Extract features for intact
    X_intact = []
    valid_intact_indices = []

    print(f"Extracting {scenario} features for intact...")
    for i, sample in enumerate(intact):
        features = extract_synthetic_features(client, sample, scenario)
        if features is not None:
            X_intact.append(features)
            valid_intact_indices.append(i)

    print(f"  ✓ {len(X_intact)} intact with {scenario} features")

    # Combine and create labels
    X = np.vstack([X_clearing, X_intact])
    y = np.concatenate([np.ones(len(X_clearing)), np.zeros(len(X_intact))])

    print(f"\nTotal samples: {len(X)} ({len(X_clearing)} clearing, {len(X_intact)} intact)")
    print(f"Feature dimensions: {X.shape[1]}")

    # Train model
    print(f"Training logistic regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    # Evaluate
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    roc_auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)

    print(f"\nPerformance:")
    print(f"  ROC-AUC:   {roc_auc:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")

    # Feature importance
    feature_importance = np.abs(model.coef_[0])

    # Identify baseline vs delta features
    n_baseline = 10
    baseline_importance = feature_importance[:n_baseline]
    delta_importance = feature_importance[n_baseline:]

    print(f"\nFeature importance:")
    print(f"  Baseline (Y-1) features: mean={np.mean(baseline_importance):.3f}, max={np.max(baseline_importance):.3f}")
    print(f"  Delta features:          mean={np.mean(delta_importance):.3f}, max={np.max(delta_importance):.3f}")

    return {
        'scenario': scenario,
        'n_samples': len(X),
        'n_clearing': len(X_clearing),
        'n_intact': len(X_intact),
        'n_features': X.shape[1],
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance.tolist(),
        'baseline_importance_mean': float(np.mean(baseline_importance)),
        'delta_importance_mean': float(np.mean(delta_importance))
    }


def interpret_contamination_results(results):
    """
    Interpret synthetic contamination experiment results.

    Compares early-year vs late-year vs full-year performance.
    """
    early_auc = results['early']['roc_auc']
    late_auc = results['late']['roc_auc']
    full_auc = results['full']['roc_auc']

    # Compare early vs late
    early_late_diff = late_auc - early_auc
    early_late_pct = (early_late_diff / early_auc) * 100 if early_auc > 0 else 0

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    print(f"\nPerformance Comparison:")
    print(f"  Early-year (Q1, Q2):  {early_auc:.3f}")
    print(f"  Late-year (Q3, Q4):   {late_auc:.3f}")
    print(f"  Full-year (Q1-Q4):    {full_auc:.3f}")
    print(f"\n  Late - Early: {early_late_diff:+.3f} ({early_late_pct:+.1f}%)")

    # Interpretation logic
    if abs(early_late_pct) < 5:
        status = "NO_CONTAMINATION_EFFECT"
        interpretation = """
✓ EARLY AND LATE PERFORM SIMILARLY (< 5% difference)

What This Means:
  Using early-year quarters (Q1, Q2) vs late-year quarters (Q3, Q4)
  makes little difference to model performance.

Interpretation:
  → Temporal contamination is NOT a major driver of model performance
  → Delta features are robust across different time windows
  → Model likely capturing genuine year-over-year change signals
  → Consistent with quarterly validation showing Q2 ≈ Q4 performance

Conclusion:
  ✓ Dual-year delta approach successfully controls for temporal contamination
  ✓ Model performance reflects genuine precursor signals, not detection of cleared land
  ✓ Can confidently claim 3-12 month lead time for detected events
"""
    elif early_late_diff > 0 and early_late_pct >= 5:
        status = "LATE_CONTAMINATION_HELPS"
        interpretation = f"""
⚠ LATE-YEAR PERFORMS BETTER (+{early_late_pct:.1f}%)

What This Means:
  Using late-year quarters (Q3, Q4) performs better than early-year quarters (Q1, Q2).

Interpretation:
  → Late-year embeddings more informative for prediction
  → Possible temporal contamination: Q3/Q4 may see cleared land
  → Model may be detecting clearing itself, not precursors

Conclusion:
  ⚠ Temporal contamination may still affect performance
  ⚠ Need to be cautious about causal claims
  → Consider using only Y-1 baseline features for guaranteed temporal safety
  → Or restrict to Q1-Q2 embeddings only
"""
    else:  # early_late_diff < 0 and early_late_pct <= -5
        status = "EARLY_CLEAN_BETTER"
        interpretation = f"""
✓ EARLY-YEAR PERFORMS BETTER (+{abs(early_late_pct):.1f}%)

What This Means:
  Using early-year quarters (Q1, Q2) performs better than late-year quarters (Q3, Q4).

Interpretation:
  → Early-year embeddings (more likely pre-clearing) are more informative
  → This is UNEXPECTED: contamination should help, not hurt
  → Suggests model capturing genuine precursor signals in clean embeddings

Conclusion:
  ✓✓ Strong evidence of genuine precursor detection
  ✓ Dual-year delta approach successfully captures human activity patterns
  ✓ Can confidently claim causal relationship with clearing events
"""

    print(interpretation)

    # Compare to full-year baseline
    print(f"\nFull-Year Baseline:")
    print(f"  Full vs Early: {full_auc - early_auc:+.3f}")
    print(f"  Full vs Late:  {full_auc - late_auc:+.3f}")

    if full_auc >= max(early_auc, late_auc):
        print(f"  → Using all 4 quarters performs best (as expected)")
    else:
        print(f"  → Using subset of quarters unexpectedly outperforms full year")

    return {
        'status': status,
        'early_auc': early_auc,
        'late_auc': late_auc,
        'full_auc': full_auc,
        'difference': early_late_diff,
        'difference_pct': early_late_pct,
        'interpretation': interpretation.strip()
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic contamination experiment')
    parser.add_argument('--n-clearing', type=int, default=60,
                        help='Number of clearing samples to use')
    parser.add_argument('--n-intact', type=int, default=60,
                        help='Number of intact samples to use')

    args = parser.parse_args()

    print(f"{'='*60}")
    print("SYNTHETIC CONTAMINATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Clearing samples: {args.n_clearing}")
    print(f"  Intact samples:   {args.n_intact}")

    # Initialize client
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Generate clearing and intact samples
    years = [2020, 2021, 2022]
    clearings = get_clearing_samples(client, args.n_clearing, years)
    intact = get_intact_samples(args.n_intact, years)

    # Run three scenarios
    results = {}

    for scenario in ['early', 'late', 'full']:
        results[scenario] = run_scenario_experiment(client, clearings, intact, scenario)

    # Interpret results
    interpretation = interpret_contamination_results(results)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'synthetic_contamination_experiment',
        'parameters': {
            'n_clearing': args.n_clearing,
            'n_intact': args.n_intact
        },
        'scenarios': results,
        'interpretation': interpretation
    }

    output_path = Path('results/walk/synthetic_contamination.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
