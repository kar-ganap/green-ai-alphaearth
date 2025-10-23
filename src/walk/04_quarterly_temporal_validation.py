"""
WALK Phase - Quarterly Temporal Validation

Tests whether dual-year delta features successfully reduce temporal contamination
by comparing performance on Q2 vs Q4 clearings.

Key Question: Does dual-year approach change the Q2 vs Q4 pattern?
- Prior finding (single-year): Q2 >> Q4 (early detection of mid-year clearing)
- New test (dual-year delta): Q2 ≈ Q4? (precursor detection?) or Q4 > Q2? (less contamination?)

Usage:
    uv run python src/walk/04_quarterly_temporal_validation.py --n-samples 50
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config
from src.temporal_investigation.phase1_glad_validation import (
    get_glad_clearing_date,
    enrich_clearings_with_dates,
    stratify_by_quarter,
)


def extract_dual_year_features_for_clearing(client, clearing):
    """
    Extract dual-year temporal features for a single clearing.

    Uses the same approach as WALK phase:
    - Baseline (Y-1): 4 quarterly embeddings
    - Current (Y): 4 quarterly embeddings
    - Delta: Year-over-year change

    Args:
        client: EarthEngineClient
        clearing: Dict with 'lat', 'lon', 'year', 'quarter'

    Returns:
        Feature vector (17 features: baseline + delta) or None if extraction fails
    """
    try:
        lat = clearing['lat']
        lon = clearing['lon']
        year = clearing['year']

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

        # Compute delta (year-over-year change)
        delta = {
            f'Q{q}': current[f'Q{q}'] - baseline[f'Q{q}']
            for q in [1, 2, 3, 4]
        }

        # Compute baseline features (10 features)
        baseline_emb_q1 = baseline['Q1']
        baseline_features = []

        # Distances from Q1 baseline
        for q in [1, 2, 3, 4]:
            dist = np.linalg.norm(baseline[f'Q{q}'] - baseline_emb_q1)
            baseline_features.append(float(dist))

        # Velocities (3 features)
        for i in range(3):
            vel = baseline_features[i+1] - baseline_features[i]
            baseline_features.append(float(vel))

        # Accelerations (2 features)
        acc1 = baseline_features[5] - baseline_features[4]
        acc2 = baseline_features[6] - baseline_features[5]
        baseline_features.extend([float(acc1), float(acc2)])

        # Trend consistency (1 feature)
        diffs = np.diff([baseline_features[0], baseline_features[1],
                        baseline_features[2], baseline_features[3]])
        trend = float(np.mean(diffs > 0))
        baseline_features.append(trend)

        # Compute delta features (7 features)
        delta_magnitudes = [
            float(np.linalg.norm(delta[f'Q{q}']))
            for q in [1, 2, 3, 4]
        ]

        delta_features = [
            *delta_magnitudes,  # 4 features
            float(np.mean(delta_magnitudes)),  # mean
            float(np.max(delta_magnitudes)),   # max
            float(delta_magnitudes[3] - delta_magnitudes[0]),  # trend
        ]

        # Return baseline + delta features (17 total)
        return np.array(baseline_features + delta_features)

    except Exception as e:
        return None


def extract_features_for_quarters(client, clearings_by_quarter):
    """
    Extract dual-year features for Q2 and Q4 clearings.

    Args:
        client: EarthEngineClient
        clearings_by_quarter: Dict mapping quarter to list of clearings

    Returns:
        Tuple of (q2_features, q4_features, q2_labels, q4_labels)
    """
    print("\nExtracting dual-year features for quarterly clearings...")

    results = {}

    for quarter in [2, 4]:
        clearings = clearings_by_quarter.get(quarter, [])

        if len(clearings) == 0:
            print(f"  Q{quarter}: No clearings, skipping")
            continue

        print(f"  Q{quarter}: Extracting features for {len(clearings)} clearings...")

        features = []
        labels = []

        for clearing in tqdm(clearings, desc=f"    Q{quarter}", leave=False):
            feature_vec = extract_dual_year_features_for_clearing(client, clearing)

            if feature_vec is not None:
                features.append(feature_vec)
                labels.append(1)  # All are clearings

        results[quarter] = {
            'features': np.array(features) if features else None,
            'labels': np.array(labels) if labels else None,
            'n_extracted': len(features),
            'n_failed': len(clearings) - len(features),
        }

        print(f"    ✓ Extracted {len(features)}/{len(clearings)} samples")

    return results


def generate_intact_forest_samples(client, n_samples, years=[2020, 2021, 2022]):
    """
    Generate intact forest samples for training.

    Args:
        client: EarthEngineClient
        n_samples: Number of intact samples to generate
        years: Years to sample from

    Returns:
        Array of features (n_samples x 17)
    """
    print(f"\nGenerating {n_samples} intact forest samples...")

    config = get_config()

    # Use intact forest bastions
    intact_regions = [
        {"name": "Amazon Core", "bounds": {"min_lon": -60, "max_lon": -55,
                                           "min_lat": -5, "max_lat": 0}},
        {"name": "Guiana Shield", "bounds": {"min_lon": -55, "max_lon": -50,
                                             "min_lat": 2, "max_lat": 6}},
    ]

    features = []

    for year in years:
        year_samples = n_samples // len(years)

        for region in intact_regions:
            region_samples = year_samples // len(intact_regions)

            # Sample random points in region
            bounds = region['bounds']

            for _ in range(region_samples):
                # Random point in bounds
                lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
                lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])

                # Create pseudo-clearing dict for feature extraction
                pseudo_clearing = {'lat': lat, 'lon': lon, 'year': year}

                feature_vec = extract_dual_year_features_for_clearing(client, pseudo_clearing)

                if feature_vec is not None:
                    features.append(feature_vec)

                if len(features) >= n_samples:
                    break

            if len(features) >= n_samples:
                break

        if len(features) >= n_samples:
            break

    print(f"  ✓ Generated {len(features)} intact samples")

    return np.array(features)


def train_and_evaluate_quarterly_models(q2_data, q4_data, intact_features):
    """
    Train separate models for Q2 and Q4 clearings and compare performance.

    Args:
        q2_data: Dict with Q2 features and labels
        q4_data: Dict with Q4 features and labels
        intact_features: Array of intact forest features

    Returns:
        Dict with evaluation results
    """
    print("\nTraining and evaluating quarterly models...")

    results = {}

    # Prepare intact labels
    intact_labels = np.zeros(len(intact_features))

    for quarter, data in [(2, q2_data), (4, q4_data)]:
        if data['features'] is None or len(data['features']) == 0:
            print(f"  Q{quarter}: No features, skipping")
            continue

        print(f"\n  Q{quarter} Model:")

        # Combine clearing and intact samples
        X_clearing = data['features']
        y_clearing = data['labels']

        # Use subset of intact samples (balance classes)
        n_intact = min(len(intact_features), len(X_clearing))
        X_intact = intact_features[:n_intact]
        y_intact = intact_labels[:n_intact]

        X_train = np.vstack([X_clearing, X_intact])
        y_train = np.concatenate([y_clearing, y_intact])

        print(f"    Training samples: {len(X_train)} ({len(X_clearing)} clearing, {len(X_intact)} intact)")

        # Train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Evaluate on training set (in-sample performance indicator)
        y_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        roc_auc = roc_auc_score(y_train, y_pred_proba)

        print(f"    ROC-AUC (in-sample): {roc_auc:.3f}")

        # Compute feature importance (coefficient magnitudes)
        feature_importance = np.abs(model.coef_[0])
        top_features = np.argsort(feature_importance)[-5:][::-1]

        print(f"    Top 5 features (by |coef|): {top_features}")

        results[quarter] = {
            'n_clearing': int(len(X_clearing)),
            'n_intact': int(len(X_intact)),
            'n_total': int(len(X_train)),
            'roc_auc': float(roc_auc),
            'feature_importance': feature_importance.tolist(),
            'top_features': top_features.tolist(),
        }

    return results


def interpret_quarterly_comparison(results):
    """
    Interpret Q2 vs Q4 model performance comparison.

    Args:
        results: Dict with Q2 and Q4 model results

    Returns:
        Dict with interpretation
    """
    print("\n" + "=" * 80)
    print("QUARTERLY COMPARISON INTERPRETATION")
    print("=" * 80)

    if 2 not in results or 4 not in results:
        return {
            'status': 'INCONCLUSIVE',
            'reason': 'Missing Q2 or Q4 results',
        }

    q2_auc = results[2]['roc_auc']
    q4_auc = results[4]['roc_auc']

    diff = q4_auc - q2_auc
    diff_pct = 100 * diff / q2_auc if q2_auc > 0 else 0

    print(f"\n  Q2 ROC-AUC: {q2_auc:.3f} ({results[2]['n_clearing']} clearings)")
    print(f"  Q4 ROC-AUC: {q4_auc:.3f} ({results[4]['n_clearing']} clearings)")
    print(f"  Difference: {diff:+.3f} ({diff_pct:+.1f}%)")

    # Decision criteria
    if q4_auc > q2_auc * 1.05:  # Q4 is 5% better
        status = "REDUCED_CONTAMINATION"
        interpretation = f"""
✓ TEMPORAL CONTAMINATION REDUCED (Q4 > Q2)

Q4 clearings perform {diff_pct:.1f}% better than Q2 clearings.

What This Means:
  The dual-year delta approach successfully reduces temporal contamination
  for late-year clearings (Q4). These clearings have less Y-embedding
  contamination because most quarters are pre-clearing.

  Q4 clearings: Y embeddings mostly clean (Q1-Q3 intact) → Delta captures
                true year-over-year change with less contamination

  Q2 clearings: Y embeddings partially contaminated (Q2-Q4 post-clearing)
                → Delta mixes clearing signal with precursors

Conclusion:
  ✓ Dual-year approach achieves its goal of reducing temporal contamination
  ✓ Late-year clearings show better causal signal (less leakage)
  ✓ Validates the temporal control hypothesis

Recommendation:
  Frame system as providing variable lead time:
  - Q4 clearings: 3-9 months (cleaner signal)
  - Q2 clearings: 0-6 months (mixed signal)
"""

    elif q2_auc > q4_auc * 1.05:  # Q2 is 5% better
        status = "STILL_EARLY_DETECTION"
        interpretation = f"""
⚠️ STILL DETECTING MID-YEAR CLEARINGS (Q2 > Q4)

Q2 clearings perform {abs(diff_pct):.1f}% better than Q4 clearings.

What This Means:
  Even with dual-year delta features, we're still detecting mid-year
  clearing events better than late-year events.

  This suggests:
  - Annual embeddings remain weighted toward mid-year (dry season bias)
  - Delta (Y - Y-1) amplifies mid-year signal
  - Not capturing true precursor activities (roads, camps)

Conclusion:
  ~ Dual-year approach provides value but doesn't fully address temporal contamination
  ~ System is more "annual risk model" than "early warning system"
  ~ Lead time: 0-6 months (detecting clearings, not precursors)

Recommendation:
  Consider monthly embeddings or Sentinel-2 time series for true precursor detection
"""

    else:  # Similar performance
        status = "EQUAL_PERFORMANCE"
        interpretation = f"""
~ Q2 AND Q4 PERFORM SIMILARLY

Q2 and Q4 clearings show comparable performance (within 5%).

What This Means:
  Dual-year delta features work equally well regardless of clearing quarter.
  This could indicate:

  Option A: Successfully capturing precursors for both Q2 and Q4
           → Good news: Temporal contamination not a major issue

  Option B: Detecting both Q2 and Q4 clearings equally well
           → Mixed: System provides value but with uncertain lead time

  Option C: Sample size too small to detect true difference
           → Need more data to make definitive conclusion

Conclusion:
  ~ System provides predictive value
  ~ Temporal dynamics remain somewhat unclear
  ~ May have variable lead time (3-12 months)

Recommendation:
  - Scale to more samples for definitive answer
  - Test on held-out temporal split (train 2020-2021, test 2022-2023)
"""

    print(f"\n{interpretation}")
    print("=" * 80)

    return {
        'status': status,
        'q2_auc': q2_auc,
        'q4_auc': q4_auc,
        'difference': diff,
        'difference_pct': diff_pct,
        'interpretation': interpretation.strip(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test dual-year delta features on quarterly-labeled clearings"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=50,
        help='Number of clearing samples to test (default: 50)'
    )
    parser.add_argument(
        '--n-intact',
        type=int,
        default=30,
        help='Number of intact samples to generate (default: 30)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WALK PHASE - QUARTERLY TEMPORAL VALIDATION")
    print("=" * 80)
    print("\nTesting whether dual-year delta features reduce temporal contamination")
    print("by comparing Q2 vs Q4 clearing performance.\n")
    print(f"Parameters:")
    print(f"  Clearing samples: {args.n_samples}")
    print(f"  Intact samples: {args.n_intact}")
    print()

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Step 1: Get clearing locations with GLAD quarterly labels
    print("Step 1: Getting clearing locations with GLAD quarterly labels...")

    years = [2020, 2021, 2022]
    samples_per_year = args.n_samples // len(years)

    main_bounds = config.study_region_bounds
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
    ]

    all_clearings = []

    for year in years:
        year_clearings = []
        for bounds in sub_regions:
            try:
                clearings = client.get_deforestation_labels(
                    bounds=bounds,
                    year=year,
                    min_tree_cover=30,
                )
                year_clearings.extend(clearings)
            except Exception:
                pass

        if len(year_clearings) > samples_per_year:
            import random
            random.seed(42)
            year_clearings = random.sample(year_clearings, samples_per_year)

        all_clearings.extend(year_clearings)
        print(f"  Year {year}: {len(year_clearings)} clearings")

    print(f"\n  ✓ Total clearings: {len(all_clearings)}")

    # Step 2: Enrich with GLAD dates
    print("\nStep 2: Enriching with GLAD dates...")
    enriched_clearings = enrich_clearings_with_dates(all_clearings)

    if len(enriched_clearings) < 10:
        print(f"\n✗ ERROR: Only {len(enriched_clearings)} clearings with dates")
        print("  Need at least 10 for valid quarterly analysis")
        return None

    # Step 3: Stratify by quarter
    print("\nStep 3: Stratifying by quarter...")
    clearings_by_quarter = stratify_by_quarter(enriched_clearings)

    # Step 4: Extract dual-year features
    print("\nStep 4: Extracting dual-year features...")
    quarterly_features = extract_features_for_quarters(client, clearings_by_quarter)

    # Step 5: Generate intact samples
    print("\nStep 5: Generating intact forest samples...")
    intact_features = generate_intact_forest_samples(
        client,
        n_samples=args.n_intact,
        years=years
    )

    # Step 6: Train and evaluate models
    print("\nStep 6: Training and evaluating quarterly models...")
    model_results = train_and_evaluate_quarterly_models(
        quarterly_features.get(2, {}),
        quarterly_features.get(4, {}),
        intact_features
    )

    # Step 7: Interpret results
    print("\nStep 7: Interpreting quarterly comparison...")
    interpretation = interpret_quarterly_comparison(model_results)

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test': 'quarterly_temporal_validation_dual_year',
        'parameters': {
            'n_clearing_samples': args.n_samples,
            'n_intact_samples': args.n_intact,
            'years': years,
        },
        'data': {
            'n_clearings_total': len(all_clearings),
            'n_clearings_with_dates': len(enriched_clearings),
            'quarterly_counts': {
                q: len(clearings)
                for q, clearings in clearings_by_quarter.items()
            },
            'quarterly_features': {
                q: data.get('n_extracted', 0)
                for q, data in quarterly_features.items()
            },
        },
        'model_results': model_results,
        'interpretation': interpretation,
    }

    # Save results
    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "quarterly_temporal_validation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()

    if results is None:
        exit(1)

    # Exit code based on interpretation
    status = results['interpretation']['status']
    if status == 'REDUCED_CONTAMINATION':
        exit(0)  # Success!
    elif status == 'EQUAL_PERFORMANCE':
        exit(0)  # Acceptable
    else:
        exit(1)  # Still early detection
