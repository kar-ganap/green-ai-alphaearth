"""
CRAWL Test 2: Temporal Signal

Question: Do embeddings change BEFORE clearing (not just at/after)?

Decision Gate: p-value < 0.05 required to proceed

This test validates that AlphaEarth embeddings show precursor signals before
clearing occurs. If embeddings only change after clearing, we cannot predict
in advance - only detect after the fact.

Usage:
    python src/crawl/test_2_temporal.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config, plot_temporal_signal, save_figure


def get_dated_clearings(config, n_samples=20):
    """
    Get locations with known clearing dates.

    Samples from multiple years for robustness.

    Args:
        config: Config instance
        n_samples: Number of clearing events to sample

    Returns:
        List of clearing events with dates
    """
    print("Fetching dated clearing events from Earth Engine...")

    client = EarthEngineClient(use_cache=True)

    # Sample from multiple years
    years = [2020, 2021, 2022]
    samples_per_year = n_samples // len(years)

    print(f"  Strategy: {samples_per_year} samples per year × {len(years)} years = {samples_per_year * len(years)} total")

    # Split region into sub-regions for diversity
    main_bounds = config.study_region_bounds
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},  # NW
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},  # NE
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": main_bounds["min_lat"], "max_lat": mid_lat},  # SW
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": main_bounds["min_lat"], "max_lat": mid_lat},  # SE
    ]

    all_clearings = []

    print(f"\n  Getting clearing events from {len(years)} years × {len(sub_regions)} regions...")
    for year in years:
        year_clearings = []
        for i, bounds in enumerate(sub_regions):
            try:
                clearings = client.get_deforestation_labels(
                    bounds=bounds,
                    year=year,
                    min_tree_cover=30,
                )
                year_clearings.extend(clearings)
            except Exception as e:
                pass

        # Sample from this year
        if len(year_clearings) > samples_per_year:
            import random
            random.seed(42)
            year_clearings = random.sample(year_clearings, samples_per_year)

        all_clearings.extend(year_clearings)
        print(f"    Year {year}: {len(year_clearings)} clearings sampled")

    print(f"\n  ✓ Final sample: {len(all_clearings)} dated clearing events")

    return all_clearings


def get_temporal_trajectory(client, location, clearing_date):
    """
    Get embedding trajectory around a clearing event.

    Fetches embeddings at: -6m, -3m, -1m, 0m, +3m relative to clearing.

    Args:
        client: EarthEngineClient instance
        location: Dict with 'lat' and 'lon'
        clearing_date: Date of clearing event (YYYY-MM-DD string or datetime)

    Returns:
        Dict with embeddings at each timepoint, or None if failed
    """
    if isinstance(clearing_date, str):
        clearing_date = datetime.strptime(clearing_date, "%Y-%m-%d")

    try:
        # Define timepoints (in months relative to clearing)
        timepoints = {
            '-6m': clearing_date - timedelta(days=180),
            '-3m': clearing_date - timedelta(days=90),
            '-1m': clearing_date - timedelta(days=30),
            '0m': clearing_date,
            '+3m': clearing_date + timedelta(days=90),
        }

        embeddings = {}

        for label, date in timepoints.items():
            # AlphaEarth is annual, so use June 1st of that year
            date_str = f"{date.year}-06-01"

            emb = client.get_embedding(
                lat=location["lat"],
                lon=location["lon"],
                date=date_str,
            )
            embeddings[label] = emb

        return embeddings

    except Exception as e:
        print(f"  Warning: Failed to get trajectory for {location}: {e}")
        return None


def test_temporal_signal(trajectories):
    """
    Test if embeddings change significantly before clearing.

    Uses one-sample t-test to check if distance at -3 months is
    significantly greater than baseline (0).

    Args:
        trajectories: List of trajectory dicts, each with embeddings at timepoints

    Returns:
        dict with test results including p-value
    """
    print("\nTesting temporal signal with statistical test...")
    print(f"  Analyzing {len(trajectories)} trajectories")

    # Extract distances from baseline for each trajectory
    times = [-6, -3, -1, 0, 3]  # months relative to clearing
    time_labels = ['-6m', '-3m', '-1m', '0m', '+3m']

    distances = []

    for traj in trajectories:
        baseline = traj['-6m']

        # Calculate Euclidean distance from baseline for each timepoint
        traj_distances = []
        for label in time_labels:
            dist = np.linalg.norm(traj[label] - baseline)
            traj_distances.append(dist)

        distances.append(traj_distances)

    distances = np.array(distances)  # Shape: (n_trajectories, 5)

    # Statistical test: Is -3m distance significantly different from 0?
    # H0: mean distance at -3m = 0 (no change from baseline)
    # H1: mean distance at -3m > 0 (change detected)
    distances_3m = distances[:, 1]  # -3m is index 1

    t_stat, p_value = ttest_1samp(distances_3m, 0, alternative='greater')

    # Calculate mean distances for interpretation
    mean_distances = distances.mean(axis=0)
    std_distances = distances.std(axis=0)

    # Calculate signal ratio: how much change at -3m vs at 0m?
    signal_ratio = mean_distances[1] / (mean_distances[3] + 1e-8)  # -3m / 0m

    print(f"\n  Statistical Test Results:")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"\n  Signal Characteristics:")
    print(f"    Mean distance at -3m: {mean_distances[1]:.4f}")
    print(f"    Mean distance at 0m: {mean_distances[3]:.4f}")
    print(f"    Signal ratio (-3m/0m): {signal_ratio:.2%}")

    result = {
        "p_value": float(p_value),
        "t_statistic": float(t_stat),
        "mean_distances": mean_distances.tolist(),
        "std_distances": std_distances.tolist(),
        "signal_ratio": float(signal_ratio),
        "distances_matrix": distances.tolist(),  # Full data for visualization
        "n_trajectories": len(trajectories),
    }

    return result


def run_test_2(n_samples=20, save_results=True):
    """
    Run CRAWL Test 2: Temporal Signal Test.

    Args:
        n_samples: Number of clearing events to sample
        save_results: Whether to save results to disk

    Returns:
        dict with test results and decision
    """
    print("=" * 80)
    print("CRAWL TEST 2: TEMPORAL SIGNAL")
    print("=" * 80)
    print("\nQuestion: Do embeddings change BEFORE clearing (not just at/after)?")
    print("Decision Gate: p-value < 0.05 required to proceed\n")

    # Load config
    config = get_config()
    threshold = config.crawl_thresholds["test_2_temporal"]["max_p_value"]

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Get dated clearing events
    clearings = get_dated_clearings(config, n_samples=n_samples)

    # Fetch temporal trajectories
    print("\nFetching embedding trajectories around clearing events...")
    print("  Timepoints: -6 months (baseline), -3m, -1m, 0m (clearing), +3m")

    trajectories = []
    for clearing in tqdm(clearings, desc="Fetching trajectories"):
        # Get clearing date
        clearing_year = clearing.get("year")
        if clearing_year:
            # Use mid-year as approximate clearing date
            clearing_date = datetime(clearing_year, 6, 1)
        else:
            continue

        traj = get_temporal_trajectory(client, clearing, clearing_date)

        if traj is not None:
            trajectories.append(traj)

    print(f"\n✓ Fetched {len(trajectories)} complete trajectories")

    if len(trajectories) < 10:
        print(f"\n✗ ERROR: Only {len(trajectories)} trajectories - need at least 10 for valid test")
        print("  This may be due to Earth Engine sampling limitations.")
        print("  Try increasing n_samples or expanding the study region.")
        return None

    # Test temporal signal
    signal_results = test_temporal_signal(trajectories)

    # Visualize
    print("\nGenerating visualization...")
    times = [-6, -3, -1, 0, 3]
    distances_matrix = np.array(signal_results["distances_matrix"])

    fig = plot_temporal_signal(
        times=times,
        distances=distances_matrix,
        p_value=signal_results["p_value"],
    )

    # Save figure
    save_figure(fig, "test_2_temporal.png", subdir="crawl")

    # Make decision
    p_value = signal_results["p_value"]
    passed = p_value < threshold

    print("\n" + "=" * 80)
    print("DECISION GATE")
    print("=" * 80)
    print(f"p-value:                 {p_value:.6f}")
    print(f"Required:                <{threshold:.3f}")
    print(f"Status:                  {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"\nSignal Characteristics:")
    print(f"  Distance at -3 months: {signal_results['mean_distances'][1]:.4f}")
    print(f"  Distance at clearing:  {signal_results['mean_distances'][3]:.4f}")
    print(f"  Signal ratio:          {signal_results['signal_ratio']:.1%}")

    if passed:
        print(f"\nDecision:        GO - Proceed to CRAWL Test 3")
        print(f"Interpretation:  Embeddings show statistically significant change at -3 months")
        print(f"                 before clearing (p={p_value:.6f}). This validates that")
        print(f"                 AlphaEarth contains predictive signals, not just detection.")
    else:
        print(f"\nDecision:        PIVOT - Do not proceed with prediction")
        print(f"Interpretation:  Embeddings do not show significant change before clearing")
        print(f"                 (p={p_value:.6f} >= {threshold}). Without precursor signals,")
        print(f"                 we cannot predict in advance - only detect after clearing.")
        print(f"\nRecommendation:  Pivot to real-time detection instead of early warning.")
        print(f"                 Or try shorter prediction horizons (30-60 days).")

    print("=" * 80 + "\n")

    # Compile results
    results = {
        "test_name": "test_2_temporal",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples_requested": n_samples,
            "n_trajectories_obtained": len(trajectories),
            "years": [2020, 2021, 2022],
            "threshold": threshold,
        },
        "data": {
            "n_clearings": len(clearings),
            "n_trajectories": len(trajectories),
            "timepoints": [-6, -3, -1, 0, 3],
        },
        "results": signal_results,
        "decision": {
            "threshold": threshold,
            "p_value": p_value,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        },
    }

    # Save results
    if save_results:
        results_dir = config.get_path("paths.results_dir") / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "crawl_test_2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAWL Test 2: Temporal Signal Test")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of clearing events to sample (default: 20)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )

    args = parser.parse_args()

    # Run test
    results = run_test_2(
        n_samples=args.n_samples,
        save_results=not args.no_save,
    )

    # Exit with appropriate code
    if results is None:
        exit_code = 2  # Error
    else:
        exit_code = 0 if results["decision"]["passed"] else 1

    exit(exit_code)
