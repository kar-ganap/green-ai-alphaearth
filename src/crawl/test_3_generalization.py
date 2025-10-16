"""
CRAWL Test 3: Generalization Test

Question: Does the signal work consistently across different regions?

Decision Gate: Coefficient of Variation (CV) < 0.5 required to proceed

This test validates that the temporal signal observed in Test 2 generalizes
across different geographic regions. High variation suggests region-specific
behavior that may require separate models.

Usage:
    python src/crawl/test_3_generalization.py
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config, plot_regional_generalization, save_figure


def get_test_regions(config):
    """
    Define test regions covering different parts of the Amazon.

    Returns:
        Dict mapping region name to bounds
    """
    # Use the main study region and divide it into sub-regions
    main_bounds = config.study_region_bounds

    min_lon = main_bounds["min_lon"]
    max_lon = main_bounds["max_lon"]
    min_lat = main_bounds["min_lat"]
    max_lat = main_bounds["max_lat"]

    # Calculate division points
    lon_third = (max_lon - min_lon) / 3
    lat_half = (max_lat - min_lat) / 2

    regions = {
        "West": {
            "min_lon": min_lon,
            "max_lon": min_lon + lon_third,
            "min_lat": min_lat,
            "max_lat": max_lat,
        },
        "Central": {
            "min_lon": min_lon + lon_third,
            "max_lon": min_lon + 2 * lon_third,
            "min_lat": min_lat,
            "max_lat": max_lat,
        },
        "East": {
            "min_lon": min_lon + 2 * lon_third,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
        },
    }

    return regions


def measure_regional_signal(client, region_bounds, n_samples=10):
    """
    Measure temporal signal strength in a region.

    Gets clearings in the region and measures embedding change from
    baseline year to clearing year.

    Args:
        client: EarthEngineClient instance
        region_bounds: Geographic bounds for region
        n_samples: Number of clearings to sample

    Returns:
        List of signal strengths (distances) for this region
    """
    # Get clearings from multiple years
    years = [2020, 2021, 2022]
    all_clearings = []

    for year in years:
        try:
            clearings = client.get_deforestation_labels(
                bounds=region_bounds,
                year=year,
                min_tree_cover=30,
            )
            all_clearings.extend(clearings)
        except Exception as e:
            pass

    if len(all_clearings) == 0:
        return []

    # Sample
    if len(all_clearings) > n_samples:
        import random
        random.seed(42)
        all_clearings = random.sample(all_clearings, n_samples)

    # Measure signal for each clearing
    signals = []

    for clearing in all_clearings:
        try:
            loc = clearing
            year = clearing.get("year")

            if year is None:
                continue

            # Get embeddings: baseline (year-1) vs clearing year
            baseline_date = f"{year - 1}-06-01"
            clearing_date = f"{year}-06-01"

            emb_baseline = client.get_embedding(
                lat=loc["lat"],
                lon=loc["lon"],
                date=baseline_date,
            )

            emb_clearing = client.get_embedding(
                lat=loc["lat"],
                lon=loc["lon"],
                date=clearing_date,
            )

            # Calculate signal strength (Euclidean distance)
            signal = np.linalg.norm(emb_clearing - emb_baseline)
            signals.append(signal)

        except Exception as e:
            continue

    return signals


def test_generalization(regional_signals):
    """
    Test if signal is consistent across regions.

    Uses coefficient of variation (std/mean) to measure consistency.
    Low CV = consistent signal. High CV = region-specific behavior.

    Args:
        regional_signals: Dict mapping region name to list of signal strengths

    Returns:
        dict with test results including CV
    """
    print("\nTesting signal consistency across regions...")

    # Calculate statistics for each region
    region_stats = {}

    for region, signals in regional_signals.items():
        if len(signals) > 0:
            region_stats[region] = {
                "mean": float(np.mean(signals)),
                "std": float(np.std(signals)),
                "n_samples": len(signals),
            }
        else:
            region_stats[region] = {
                "mean": 0.0,
                "std": 0.0,
                "n_samples": 0,
            }

    # Calculate coefficient of variation across regions
    means = [stats["mean"] for stats in region_stats.values() if stats["n_samples"] > 0]

    if len(means) < 2:
        print("  WARNING: Not enough regions with data for CV calculation")
        cv = 1.0  # High CV indicates failure
    else:
        overall_mean = np.mean(means)
        overall_std = np.std(means)
        cv = overall_std / overall_mean if overall_mean > 0 else 1.0

    print(f"\n  Regional Statistics:")
    for region, stats in region_stats.items():
        print(f"    {region}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n_samples']}")

    print(f"\n  Cross-Regional Consistency:")
    print(f"    Coefficient of Variation: {cv:.3f}")
    print(f"    Overall mean: {overall_mean:.4f}")
    print(f"    Overall std: {overall_std:.4f}")

    result = {
        "cv": float(cv),
        "overall_mean": float(overall_mean) if len(means) > 0 else 0.0,
        "overall_std": float(overall_std) if len(means) > 0 else 0.0,
        "region_stats": region_stats,
        "n_regions": len([s for s in region_stats.values() if s["n_samples"] > 0]),
    }

    return result


def run_test_3(n_samples_per_region=10, save_results=True):
    """
    Run CRAWL Test 3: Generalization Test.

    Args:
        n_samples_per_region: Number of clearings to sample per region
        save_results: Whether to save results to disk

    Returns:
        dict with test results and decision
    """
    print("=" * 80)
    print("CRAWL TEST 3: GENERALIZATION")
    print("=" * 80)
    print("\nQuestion: Does the signal work consistently across different regions?")
    print("Decision Gate: Coefficient of Variation (CV) < 0.5 required to proceed\n")

    # Load config
    config = get_config()
    threshold = config.crawl_thresholds["test_3_generalization"]["max_cv"]

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Define test regions
    regions = get_test_regions(config)
    print(f"Testing signal across {len(regions)} regions:")
    for name, bounds in regions.items():
        print(f"  {name}: lon [{bounds['min_lon']:.1f}, {bounds['max_lon']:.1f}], "
              f"lat [{bounds['min_lat']:.1f}, {bounds['max_lat']:.1f}]")

    # Measure signal in each region
    print(f"\nMeasuring signal strength in each region...")
    print(f"  Sampling {n_samples_per_region} clearings per region")

    regional_signals = {}

    for region_name, bounds in tqdm(regions.items(), desc="Processing regions"):
        print(f"\n  {region_name}:")
        signals = measure_regional_signal(client, bounds, n_samples=n_samples_per_region)
        regional_signals[region_name] = signals
        print(f"    Obtained {len(signals)} signal measurements")

    # Test generalization
    generalization_results = test_generalization(regional_signals)

    # Visualize
    print("\nGenerating visualization...")
    fig = plot_regional_generalization(
        region_results=generalization_results["region_stats"],
        cv_threshold=threshold,
    )

    # Save figure
    save_figure(fig, "test_3_generalization.png", subdir="crawl")

    # Make decision
    cv = generalization_results["cv"]
    passed = cv < threshold

    print("\n" + "=" * 80)
    print("DECISION GATE")
    print("=" * 80)
    print(f"Coefficient of Variation: {cv:.3f}")
    print(f"Required:                 <{threshold:.3f}")

    if passed:
        print(f"Status:                   ✓ PASS")
        print(f"\nDecision:        GO - Proceed to CRAWL Test 4")
        print(f"Interpretation:  Signal is consistent across regions (CV={cv:.3f}).")
        print(f"                 A single model should work across the study area.")
    elif cv < 0.75:
        print(f"Status:                   ⚠ WARNING")
        print(f"\nDecision:        PROCEED WITH CAUTION")
        print(f"Interpretation:  Signal shows moderate variation across regions (CV={cv:.3f}).")
        print(f"                 May benefit from region-specific models or region features.")
        print(f"\nRecommendation:  Continue, but consider adding region as a feature.")
    else:
        print(f"Status:                   ✗ HIGH VARIATION")
        print(f"\nDecision:        STOP OR PIVOT")
        print(f"Interpretation:  Signal is highly inconsistent across regions (CV={cv:.3f}).")
        print(f"                 Each region may require separate models.")
        print(f"\nRecommendation:  Focus on single region for MVP, or use regional models.")

    print("=" * 80 + "\n")

    # Compile results
    results = {
        "test_name": "test_3_generalization",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples_per_region": n_samples_per_region,
            "n_regions": len(regions),
            "threshold": threshold,
        },
        "regions": {name: bounds for name, bounds in regions.items()},
        "results": generalization_results,
        "decision": {
            "threshold": threshold,
            "cv": cv,
            "passed": passed,
            "status": "PASS" if passed else ("WARNING" if cv < 0.75 else "FAIL"),
        },
    }

    # Save results
    if save_results:
        results_dir = config.get_path("paths.results_dir") / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "crawl_test_3_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAWL Test 3: Generalization Test")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of clearings per region (default: 10)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )

    args = parser.parse_args()

    # Run test
    results = run_test_3(
        n_samples_per_region=args.n_samples,
        save_results=not args.no_save,
    )

    # Exit with appropriate code
    if results["decision"]["status"] == "PASS":
        exit_code = 0
    elif results["decision"]["status"] == "WARNING":
        exit_code = 0  # Warning still allows proceeding
    else:
        exit_code = 1

    exit(exit_code)
