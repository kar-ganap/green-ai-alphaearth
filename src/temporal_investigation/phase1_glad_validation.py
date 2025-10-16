"""
Temporal Investigation Phase 1: GLAD-S2 Alert Validation

This script validates whether we're detecting true precursor signals or just
early-year clearing by using GLAD-S2 alerts to get precise clearing dates.

Key Question: Does Y-1 AlphaEarth embedding predict Q4 clearings better than Q1?
- If YES → True precursor signal (preparation in late Y-1, clearing in Q4 Y)
- If NO → Early detection (clearing happened in Q1 Y, captured in annual Y)

Usage:
    python src/temporal_investigation/phase1_glad_validation.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
from scipy.stats import ttest_1samp
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config


def get_glad_clearing_date(lat, lon, year):
    """
    Get precise clearing date from GLAD alerts.

    GLAD dataset uses year-specific bands:
    - conf[YY]: Confidence value for year 20YY
    - alertDate[YY]: Date of alert for year 20YY (days since 2015-01-01)

    Historical data is stored in archived collections:
    - projects/glad/alert/2020final (has bands for 2020 and 2021)
    - projects/glad/alert/2021final (has bands for 2021 and 2022)
    - projects/glad/alert/2022final (has bands for 2022 and 2023)

    Args:
        lat: Latitude
        lon: Longitude
        year: Year we expect clearing (from Hansen GFC)

    Returns:
        dict with 'date', 'year', 'quarter', 'month' or None if no alert found
    """
    try:
        point = ee.Geometry.Point([lon, lat])

        # Use archived year-specific ImageCollection for historical data
        # Current year data (2024-2025) is in projects/glad/alert/UpdResult
        if year >= 2024:
            dataset_id = 'projects/glad/alert/UpdResult'
        else:
            # Use year-specific archived collection
            dataset_id = f'projects/glad/alert/{year}final'

        glad_collection = ee.ImageCollection(dataset_id)

        # Band name for this year (last 2 digits)
        year_suffix = str(year % 100)  # e.g., 2020 → '20', 2021 → '21'

        alert_date_band = f'alertDate{year_suffix}'
        conf_band = f'conf{year_suffix}'

        # Select bands and mosaic to create a single image
        glad = glad_collection.select([alert_date_band, conf_band]).mosaic()

        # Sample at the point
        sample = glad.sample(region=point, scale=10, numPixels=1)
        features = sample.getInfo()['features']

        if len(features) == 0 or len(features[0]['properties']) == 0:
            return None

        props = features[0]['properties']

        # Get date value (Julian day of year: 1-365/366)
        date_value = props.get(alert_date_band)
        conf_value = props.get(conf_band)

        if date_value is None or date_value == 0:
            return None

        # Convert Julian day to actual date
        # date_value is the day of year (1 = Jan 1, 365/366 = Dec 31)
        alert_date = datetime(year, 1, 1) + timedelta(days=int(date_value) - 1)

        # Extract temporal components
        quarter = (alert_date.month - 1) // 3 + 1

        return {
            'date': alert_date.strftime('%Y-%m-%d'),
            'year': alert_date.year,
            'month': alert_date.month,
            'quarter': quarter,
            'day_of_year': alert_date.timetuple().tm_yday,
            'confidence': int(conf_value) if conf_value is not None else None,
            'source': 'GLAD (Landsat 30m)',
        }

    except Exception as e:
        # print(f"  Warning: Could not get GLAD date for ({lat}, {lon}): {e}")
        return None


def enrich_clearings_with_dates(clearings):
    """
    Add precise GLAD dates to clearing locations.

    Args:
        clearings: List of clearing dicts with 'lat', 'lon', 'year'

    Returns:
        List of clearings enriched with date info
    """
    print("Enriching clearing locations with GLAD dates...")
    print(f"  Total clearings: {len(clearings)}")

    enriched = []

    # Track success rate
    glad_success = 0
    no_date_found = 0

    for clearing in tqdm(clearings, desc="Querying GLAD alerts"):
        lat = clearing['lat']
        lon = clearing['lon']
        year = clearing.get('year')

        if year is None:
            print(f"  Warning: No year for clearing at ({lat}, {lon}), skipping")
            continue

        # Get GLAD alert date
        date_info = get_glad_clearing_date(lat, lon, year)

        if date_info is not None:
            glad_success += 1
            # Combine clearing info with date info
            enriched_clearing = {**clearing, **date_info}
            enriched.append(enriched_clearing)
        else:
            no_date_found += 1

    print(f"\n  ✓ Enriched {len(enriched)} / {len(clearings)} clearings")
    print(f"    GLAD alerts found: {glad_success}")
    print(f"    No date found: {no_date_found}")

    return enriched


def stratify_by_quarter(clearings):
    """
    Stratify clearings by quarter.

    Args:
        clearings: List of enriched clearings with 'quarter' field

    Returns:
        Dict mapping quarter to list of clearings
    """
    quarterly = {1: [], 2: [], 3: [], 4: []}

    for clearing in clearings:
        q = clearing['quarter']
        quarterly[q].append(clearing)

    print("\n  Quarterly distribution:")
    for q in [1, 2, 3, 4]:
        count = len(quarterly[q])
        pct = 100 * count / len(clearings) if len(clearings) > 0 else 0
        print(f"    Q{q}: {count:3d} clearings ({pct:5.1f}%)")

    return quarterly


def test_quarterly_prediction(client, clearings_by_quarter):
    """
    Test if Y-1 embedding predicts Q4 clearings better than Q1.

    This is the KEY test for precursor signal validation.

    Args:
        client: EarthEngineClient instance
        clearings_by_quarter: Dict mapping quarter to clearings

    Returns:
        dict with test results and interpretation
    """
    print("\n" + "=" * 80)
    print("QUARTERLY PREDICTION TEST")
    print("=" * 80)
    print("\nHypothesis: Y-1 embedding predicts Q4 > Q1 if true precursor signal exists")
    print("  - Q4 clearings: Precursors built in Y-1, clearing happens late in Y")
    print("  - Q1 clearings: Clearing happens early in Y, might already be in Y-1 embedding")

    results = {}

    for quarter, clearings in clearings_by_quarter.items():
        if len(clearings) < 3:
            print(f"\n  Skipping Q{quarter}: only {len(clearings)} samples")
            continue

        print(f"\n  Testing Q{quarter} clearings ({len(clearings)} samples)...")

        # Get embeddings for Y-1 and Y
        distances = []

        for clearing in tqdm(clearings, desc=f"  Q{quarter} embeddings", leave=False):
            try:
                year = clearing['year']

                # Y-1 embedding (baseline)
                emb_y_minus_1 = client.get_embedding(
                    lat=clearing['lat'],
                    lon=clearing['lon'],
                    date=f"{year - 1}-06-01"
                )

                # Y embedding (clearing year)
                emb_y = client.get_embedding(
                    lat=clearing['lat'],
                    lon=clearing['lon'],
                    date=f"{year}-06-01"
                )

                # Distance from Y-1 to Y
                distance = np.linalg.norm(emb_y - emb_y_minus_1)
                distances.append(distance)

            except Exception as e:
                continue

        if len(distances) < 3:
            print(f"    Only {len(distances)} valid samples, skipping")
            continue

        distances = np.array(distances)

        # Test if distance is significantly > 0
        t_stat, p_value = ttest_1samp(distances, 0, alternative='greater')

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        print(f"    Mean distance: {mean_distance:.4f} ± {std_distance:.4f}")
        print(f"    t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")

        results[quarter] = {
            'n_samples': len(distances),
            'mean_distance': float(mean_distance),
            'std_distance': float(std_distance),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        }

    return results


def interpret_quarterly_results(results):
    """
    Interpret quarterly prediction results.

    Args:
        results: Dict mapping quarter to prediction metrics

    Returns:
        dict with interpretation
    """
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Need at least Q1 and Q4 for comparison
    if 1 not in results or 4 not in results:
        print("\n  ⚠️ WARNING: Need both Q1 and Q4 data for comparison")
        print("  Cannot determine precursor signal status")
        return {
            'status': 'INCONCLUSIVE',
            'reason': 'Insufficient quarterly data',
        }

    q1_distance = results[1]['mean_distance']
    q4_distance = results[4]['mean_distance']

    diff = q4_distance - q1_distance
    diff_pct = 100 * diff / q1_distance if q1_distance > 0 else 0

    print(f"\n  Q1 mean distance: {q1_distance:.4f}")
    print(f"  Q4 mean distance: {q4_distance:.4f}")
    print(f"  Difference: {diff:+.4f} ({diff_pct:+.1f}%)")

    # Decision criteria
    if q4_distance > q1_distance * 1.15:  # Q4 is 15% higher
        status = "TRUE_PRECURSOR"
        interpretation = (
            f"✓ TRUE PRECURSOR SIGNAL DETECTED\n\n"
            f"  Q4 distance ({q4_distance:.4f}) is {diff_pct:.1f}% higher than Q1 ({q1_distance:.4f}).\n"
            f"  This suggests Y-1 embedding captures preparation activities (roads, camps,\n"
            f"  selective logging) that precede Q4 clearing events.\n\n"
            f"  Implication: We can provide ~3-9 months warning for late-year clearings."
        )

    elif q1_distance > q4_distance * 1.15:  # Q1 is 15% higher
        status = "EARLY_DETECTION"
        interpretation = (
            f"⚠️ EARLY DETECTION (NOT TRUE PRECURSOR)\n\n"
            f"  Q1 distance ({q1_distance:.4f}) is higher than Q4 ({q4_distance:.4f}).\n"
            f"  This suggests Y-1 embedding is capturing early-year clearing that\n"
            f"  has already started, not true precursor activities.\n\n"
            f"  Implication: More like 'early detection' than 'prediction'. Annual\n"
            f"  composites include Q1-Q2 clearing in the 'Y' embedding."
        )

    else:  # Similar distances
        status = "MIXED_SIGNAL"
        interpretation = (
            f"~ MIXED SIGNAL\n\n"
            f"  Q1 and Q4 distances are similar ({q1_distance:.4f} vs {q4_distance:.4f}).\n"
            f"  This suggests Y-1 embedding captures a mix of:\n"
            f"    - True precursor signals (roads, camps) for some clearings\n"
            f"    - Early-year clearing detection for others\n\n"
            f"  Implication: System provides value but with variable lead time.\n"
            f"  Some clearings get 3-9 months warning, others get 0-3 months."
        )

    print(f"\n{interpretation}")
    print("=" * 80)

    return {
        'status': status,
        'q1_distance': q1_distance,
        'q4_distance': q4_distance,
        'difference': diff,
        'difference_pct': diff_pct,
        'interpretation': interpretation,
    }


def run_phase1_validation(n_samples=24, save_results=True):
    """
    Run Phase 1: GLAD-S2 validation of precursor signal.

    Args:
        n_samples: Number of clearing samples to test
        save_results: Whether to save results

    Returns:
        dict with validation results
    """
    print("=" * 80)
    print("TEMPORAL INVESTIGATION - PHASE 1: GLAD-S2 VALIDATION")
    print("=" * 80)
    print("\nGoal: Validate whether we're detecting true precursor signals")
    print("Method: Use GLAD-S2 precise dates to stratify by quarter")
    print("Test: Does Y-1 embedding predict Q4 clearings better than Q1?\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Get clearing locations (reuse Test 2 approach)
    print("Step 1: Getting clearing locations...")

    years = [2020, 2021, 2022]
    samples_per_year = n_samples // len(years)

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

    # Step 2: Enrich with GLAD-S2 dates
    print(f"\nStep 2: Enriching with GLAD-S2 dates...")
    enriched_clearings = enrich_clearings_with_dates(all_clearings)

    if len(enriched_clearings) < 10:
        print(f"\n✗ ERROR: Only {len(enriched_clearings)} clearings with dates")
        print("  Need at least 10 for valid quarterly analysis")
        return None

    # Step 3: Stratify by quarter
    print(f"\nStep 3: Stratifying by quarter...")
    clearings_by_quarter = stratify_by_quarter(enriched_clearings)

    # Step 4: Test quarterly prediction
    print(f"\nStep 4: Testing quarterly prediction...")
    quarterly_results = test_quarterly_prediction(client, clearings_by_quarter)

    # Step 5: Interpret results
    print(f"\nStep 5: Interpreting results...")
    interpretation = interpret_quarterly_results(quarterly_results)

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase_1_glad_validation',
        'data': {
            'n_clearings_total': len(all_clearings),
            'n_clearings_with_dates': len(enriched_clearings),
            'quarterly_counts': {q: len(clearings) for q, clearings in clearings_by_quarter.items()},
        },
        'quarterly_results': quarterly_results,
        'interpretation': interpretation,
    }

    # Save results
    if save_results:
        config = get_config()
        results_dir = config.get_path("paths.results_dir") / "temporal_investigation"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "phase1_glad_validation.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: GLAD-S2 Validation")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=24,
        help="Number of clearing samples (default: 24)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )

    args = parser.parse_args()

    results = run_phase1_validation(
        n_samples=args.n_samples,
        save_results=not args.no_save,
    )

    if results is None:
        exit(2)

    # Exit code based on interpretation
    status = results['interpretation']['status']
    if status == 'TRUE_PRECURSOR':
        exit(0)  # Success - proceed to Phase 2
    elif status == 'MIXED_SIGNAL':
        exit(0)  # Can proceed with caveats
    else:
        exit(1)  # Early detection or inconclusive
