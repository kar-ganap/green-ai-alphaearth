"""
CRAWL Test 4: Minimal Model Test

Question: Can the simplest possible features predict anything?

Decision Gate: AUC >= 0.65 required to proceed

This test validates that basic features provide predictive signal. We use
ONLY 2 features: embedding velocity and distance to road. If these don't
work, complex features won't help much.

Usage:
    python src/crawl/test_4_minimal_model.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config, plot_minimal_model_results, save_figure


def extract_velocity_feature(client, location, reference_year):
    """
    Extract velocity feature: rate of embedding change.

    Args:
        client: EarthEngineClient instance
        location: Dict with 'lat' and 'lon'
        reference_year: Year to compute velocity for

    Returns:
        float: Velocity (embedding change between consecutive years)
    """
    try:
        # Get embeddings for two consecutive years
        emb_current = client.get_embedding(
            lat=location["lat"],
            lon=location["lon"],
            date=f"{reference_year}-06-01",
        )

        emb_previous = client.get_embedding(
            lat=location["lat"],
            lon=location["lon"],
            date=f"{reference_year - 1}-06-01",
        )

        # Velocity = magnitude of change
        velocity = np.linalg.norm(emb_current - emb_previous)

        return velocity

    except Exception as e:
        return None


def extract_road_distance_feature(client, location):
    """
    Extract distance to nearest road.

    For this CRAWL test, we'll use a simple proxy: distance from center
    of study region (roads tend to be in accessible/central areas).

    In WALK phase, we'll use actual road data from OSM.

    Args:
        client: EarthEngineClient instance
        location: Dict with 'lat' and 'lon'

    Returns:
        float: Distance to nearest road (km)
    """
    # Proxy: distance from region center
    # (Central areas tend to have more roads and clearing)
    # This is a simplification for CRAWL phase
    config = get_config()
    bounds = config.study_region_bounds

    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    # Haversine distance (approximate)
    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1 = radians(location["lat"]), radians(location["lon"])
    lat2, lon2 = radians(center_lat), radians(center_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Earth radius in km
    R = 6371

    distance = R * c

    return distance


def get_labeled_data(config, n_positive=50, n_negative=50):
    """
    Get labeled data: locations that will clear vs stable.

    Positive: Locations that cleared in year Y
    Negative: Locations that remained stable

    Args:
        config: Config instance
        n_positive: Number of positive examples (will clear)
        n_negative: Number of negative examples (stable)

    Returns:
        Tuple of (positive_locations, negative_locations)
    """
    print("Fetching labeled data from Earth Engine...")

    client = EarthEngineClient(use_cache=True)

    # Get positive examples: clearings from multiple years
    years = [2021, 2022]  # Use recent years
    samples_per_year = n_positive // len(years)

    main_bounds = config.study_region_bounds
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

    all_positive = []

    print(f"\n  Getting positive examples (will clear)...")
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

        # Sample
        if len(year_clearings) > samples_per_year:
            import random
            random.seed(42)
            year_clearings = random.sample(year_clearings, samples_per_year)

        all_positive.extend(year_clearings)
        print(f"    Year {year}: {len(year_clearings)} clearings")

    # Get negative examples: stable forest
    print(f"\n  Getting negative examples (stable)...")
    all_negative = []
    for bounds in sub_regions:
        try:
            stable = client.get_stable_forest_locations(
                bounds=bounds,
                n_samples=n_negative,
                min_tree_cover=30,
                max_loss_year=2015,
            )
            all_negative.extend(stable)
        except Exception:
            pass

    # Sample if needed
    if len(all_negative) > n_negative:
        import random
        random.seed(42)
        all_negative = random.sample(all_negative, n_negative)

    print(f"    Total stable: {len(all_negative)}")

    print(f"\n  ✓ Final sample: {len(all_positive)} positive, {len(all_negative)} negative")

    return all_positive, all_negative


def extract_features(client, locations, labels, reference_years):
    """
    Extract 2 features for all locations.

    Args:
        client: EarthEngineClient instance
        locations: List of location dicts
        labels: List of labels (1=will clear, 0=stable)
        reference_years: List of reference years for each location

    Returns:
        Tuple of (X, y) where X is (n_samples, 2) array
    """
    print("\nExtracting features (velocity + distance to road)...")

    X = []
    y = []

    for i, loc in enumerate(tqdm(locations, desc="Extracting features")):
        try:
            ref_year = reference_years[i]

            # Feature 1: Velocity
            velocity = extract_velocity_feature(client, loc, ref_year)

            # Feature 2: Distance to road (proxy: distance from center)
            road_dist = extract_road_distance_feature(client, loc)

            if velocity is not None and road_dist is not None:
                X.append([velocity, road_dist])
                y.append(labels[i])

        except Exception as e:
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"\n✓ Extracted features for {len(X)} locations")
    print(f"  Feature 1 (velocity): mean={X[:, 0].mean():.4f}, std={X[:, 0].std():.4f}")
    print(f"  Feature 2 (road dist): mean={X[:, 1].mean():.2f} km, std={X[:, 1].std():.2f} km")

    return X, y


def test_minimal_model(X, y):
    """
    Test if minimal features can predict deforestation.

    Uses simple logistic regression with stratified cross-validation.

    Args:
        X: Features (n_samples, 2)
        y: Labels (n_samples,)

    Returns:
        dict with test results including AUC
    """
    print("\nTesting minimal model (Logistic Regression)...")
    print(f"  Training data: {len(X)} samples")
    print(f"  Class distribution: {np.sum(y == 1)} positive, {np.sum(y == 0)} negative")

    # Stratified CV
    n_folds = min(5, len(X) // 10)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Simple logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    auc_mean = np.mean(cv_scores)
    auc_std = np.std(cv_scores)

    print(f"\n  Cross-Validation Results:")
    print(f"    Mean AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"    Fold scores: {', '.join([f'{s:.3f}' for s in cv_scores])}")

    # Train final model for interpretation
    model.fit(X, y)
    coefficients = model.coef_[0]

    print(f"\n  Feature Importance (coefficients):")
    print(f"    Velocity:       {coefficients[0]:+.4f}")
    print(f"    Road distance:  {coefficients[1]:+.4f}")

    result = {
        "mean_auc": float(auc_mean),
        "std_auc": float(auc_std),
        "fold_scores": cv_scores.tolist(),
        "coefficients": {
            "velocity": float(coefficients[0]),
            "road_distance": float(coefficients[1]),
        },
        "n_samples": len(X),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
    }

    return result


def run_test_4(n_samples=50, save_results=True):
    """
    Run CRAWL Test 4: Minimal Model Test.

    Args:
        n_samples: Number of samples per class
        save_results: Whether to save results to disk

    Returns:
        dict with test results and decision
    """
    print("=" * 80)
    print("CRAWL TEST 4: MINIMAL MODEL")
    print("=" * 80)
    print("\nQuestion: Can the simplest possible features predict anything?")
    print("Decision Gate: AUC >= 0.65 required to proceed\n")
    print("Features: ONLY 2 features")
    print("  1. Velocity (embedding change rate)")
    print("  2. Distance to road (proxy: distance from center)")

    # Load config
    config = get_config()
    threshold = config.crawl_thresholds["test_4_minimal"]["min_auc"]

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Get labeled data
    positive_locs, negative_locs = get_labeled_data(config, n_positive=n_samples, n_negative=n_samples)

    # Prepare for feature extraction
    all_locations = positive_locs + negative_locs
    all_labels = [1] * len(positive_locs) + [0] * len(negative_locs)

    # Assign reference years (year before clearing for positive, recent year for negative)
    reference_years = []
    for loc in positive_locs:
        year = loc.get("year", 2021)
        reference_years.append(year)  # Use clearing year to compute velocity

    for loc in negative_locs:
        reference_years.append(2022)  # Use recent year for stable locations

    # Extract features
    X, y = extract_features(client, all_locations, all_labels, reference_years)

    if len(X) < 20:
        print(f"\n✗ ERROR: Only {len(X)} samples with features - need at least 20 for valid test")
        return None

    # Test minimal model
    model_results = test_minimal_model(X, y)

    # Visualize
    print("\nGenerating visualization...")
    fig = plot_minimal_model_results(
        X=X,
        y=y,
        feature_names=["Velocity", "Distance to Center (km)"],
        auc=model_results["mean_auc"],
    )

    # Save figure
    save_figure(fig, "test_4_minimal_model.png", subdir="crawl")

    # Make decision
    auc = model_results["mean_auc"]
    passed = auc >= threshold

    print("\n" + "=" * 80)
    print("DECISION GATE")
    print("=" * 80)
    print(f"AUC:                     {auc:.3f}")
    print(f"Required:                ≥{threshold:.3f}")

    if auc >= 0.75:
        print(f"Status:                  ✓ EXCELLENT")
        print(f"\nDecision:        GO - PROCEED TO WALK PHASE")
        print(f"Interpretation:  Strong signal with just 2 features (AUC={auc:.3f})!")
        print(f"                 Complex features should push performance even higher.")
        print(f"                 This validates the approach is fundamentally sound.")
    elif passed:
        print(f"Status:                  ✓ PASS")
        print(f"\nDecision:        GO - Proceed to WALK phase")
        print(f"Interpretation:  Moderate signal with 2 features (AUC={auc:.3f}).")
        print(f"                 Complex features may help, but expect modest improvements.")
    else:
        print(f"Status:                  ✗ FAIL")
        print(f"\nDecision:        STOP - Do not proceed")
        print(f"Interpretation:  Even simple features don't work (AUC={auc:.3f} < {threshold}).")
        print(f"                 Complex features unlikely to help significantly.")
        print(f"\nRecommendation:  Problem may not be solvable with current data.")
        print(f"                 Consider: different embeddings, different problem framing,")
        print(f"                 or focus on descriptive analysis instead of prediction.")

    print("=" * 80 + "\n")

    # Compile results
    results = {
        "test_name": "test_4_minimal_model",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples_requested": n_samples,
            "n_samples_obtained": len(X),
            "features": ["velocity", "distance_to_road"],
            "threshold": threshold,
        },
        "data": {
            "n_positive": model_results["n_positive"],
            "n_negative": model_results["n_negative"],
            "n_total": model_results["n_samples"],
        },
        "results": model_results,
        "decision": {
            "threshold": threshold,
            "auc": auc,
            "passed": passed,
            "status": "EXCELLENT" if auc >= 0.75 else ("PASS" if passed else "FAIL"),
        },
    }

    # Save results
    if save_results:
        results_dir = config.get_path("paths.results_dir") / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "crawl_test_4_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAWL Test 4: Minimal Model Test")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per class (default: 50)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )

    args = parser.parse_args()

    # Run test
    results = run_test_4(
        n_samples=args.n_samples,
        save_results=not args.no_save,
    )

    # Exit with appropriate code
    if results is None:
        exit_code = 2  # Error
    else:
        exit_code = 0 if results["decision"]["passed"] else 1

    exit(exit_code)
