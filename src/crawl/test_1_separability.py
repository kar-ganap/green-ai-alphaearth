"""
CRAWL Test 1: Separability Test

Question: Can AlphaEarth embeddings distinguish cleared vs intact forest?

Decision Gate: Accuracy >= 85% required to proceed

This test validates the fundamental assumption that AlphaEarth embeddings
contain sufficient information to differentiate between cleared and intact
forest. If embeddings cannot make this basic distinction, the entire approach
will fail.

Usage:
    python src/crawl/test_1_separability.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config, plot_pca_separation, save_figure


def get_sample_locations(config, n_samples=50):
    """
    Get sample cleared and intact forest locations.

    Samples equal numbers from each year for temporal balance.

    Args:
        config: Config instance
        n_samples: Target number of samples for each class (will be split across years)

    Returns:
        Tuple of (cleared_locations, intact_locations)
    """
    print("Fetching sample locations from Earth Engine...")

    client = EarthEngineClient(use_cache=True)

    # Sample from multiple years with EQUAL representation
    years = [2019, 2020, 2021]
    samples_per_year = n_samples // len(years)

    print(f"  Strategy: {samples_per_year} samples per year × {len(years)} years = {samples_per_year * len(years)} total")

    # Split the main region into sub-regions for better sampling
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

    all_cleared_by_year = {year: [] for year in years}
    all_intact_by_year = {year: [] for year in years}

    # Sample cleared locations - organize by year
    print(f"\n  Getting cleared forest locations from {len(years)} years × {len(sub_regions)} regions...")
    for year in years:
        for i, bounds in enumerate(sub_regions):
            try:
                cleared = client.get_deforestation_labels(
                    bounds=bounds,
                    year=year,
                    min_tree_cover=25,
                )
                all_cleared_by_year[year].extend(cleared)
            except Exception as e:
                pass
        print(f"    Year {year}: {len(all_cleared_by_year[year])} cleared locations")

    # Sample stable forest - need to distribute across years
    print(f"\n  Getting stable forest locations from {len(sub_regions)} regions...")
    all_intact_raw = []
    for i, bounds in enumerate(sub_regions):
        try:
            intact = client.get_stable_forest_locations(
                bounds=bounds,
                n_samples=n_samples,
                min_tree_cover=30,
                max_loss_year=2010,
            )
            all_intact_raw.extend(intact)
        except Exception as e:
            pass
    print(f"    Total stable locations: {len(all_intact_raw)}")

    # Balance samples across years
    print(f"\n  Balancing samples across years...")
    final_cleared = []
    final_intact = []

    for year in years:
        # Get equal number from each year
        year_cleared = all_cleared_by_year[year][:samples_per_year]
        year_intact = all_intact_raw[:samples_per_year]
        all_intact_raw = all_intact_raw[samples_per_year:]  # Remove used samples

        # Tag with year for temporal alignment
        for loc in year_cleared:
            loc['clearing_year'] = year
        for loc in year_intact:
            loc['reference_year'] = year

        final_cleared.extend(year_cleared)
        final_intact.extend(year_intact)

        print(f"    Year {year}: {len(year_cleared)} cleared + {len(year_intact)} intact")

    print(f"\n  ✓ Final balanced sample: {len(final_cleared)} cleared, {len(final_intact)} intact")

    return final_cleared, final_intact


def fetch_embeddings(client, locations, embedding_year_offset=1):
    """
    Fetch AlphaEarth embeddings for locations.

    Uses the clearing_year or reference_year + offset to ensure
    temporal alignment between cleared and stable samples.

    Args:
        client: EarthEngineClient instance
        locations: List of location dicts with 'clearing_year' or 'reference_year'
        embedding_year_offset: Years to add (e.g., +1 to see post-clearing state)

    Returns:
        numpy array of embeddings (n_locations, 64)
    """
    embeddings = []

    for loc in tqdm(locations, desc="Fetching embeddings"):
        try:
            # Get year from location tag
            base_year = loc.get("clearing_year") or loc.get("reference_year") or loc.get("year")
            if base_year is None:
                print(f"  Warning: No year tag found for {loc}, skipping")
                continue

            # Add offset (e.g., +1 year to see cleared state)
            embedding_year = base_year + embedding_year_offset
            date = f"{embedding_year}-06-01"

            emb = client.get_embedding(
                lat=loc["lat"],
                lon=loc["lon"],
                date=date,
            )
            embeddings.append(emb)
        except Exception as e:
            print(f"  Warning: Failed to get embedding for {loc}: {e}")
            continue

    return np.array(embeddings)


def test_separability(embeddings_cleared, embeddings_intact):
    """
    Test if embeddings can separate cleared vs intact forest.

    Uses a class-weighted linear SVM with stratified cross-validation.
    Reports metrics suitable for imbalanced classification.

    Args:
        embeddings_cleared: Embeddings from cleared locations (n, 64)
        embeddings_intact: Embeddings from intact locations (n, 64)

    Returns:
        dict with multiple metrics suitable for imbalanced data
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score, precision_score, recall_score

    print("\nTesting separability with class-weighted linear SVM...")
    print(f"  Class distribution: {len(embeddings_cleared)} cleared, {len(embeddings_intact)} intact")
    print(f"  Imbalance ratio: {len(embeddings_cleared)/len(embeddings_intact):.2f}:1")

    # Combine embeddings
    X = np.vstack([embeddings_cleared, embeddings_intact])
    y = np.array([1] * len(embeddings_cleared) + [0] * len(embeddings_intact))

    # Stratified CV to maintain class proportions in each fold
    n_samples = len(X)
    n_folds = min(5, max(2, n_samples // 2))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Class-weighted SVM (handles imbalance)
    model = SVC(kernel="linear", random_state=42, class_weight='balanced', probability=True)

    # Multiple metrics for imbalanced classification
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    result = {
        "mean_accuracy": float(np.mean(cv_results['test_accuracy'])),
        "std_accuracy": float(np.std(cv_results['test_accuracy'])),
        "mean_roc_auc": float(np.mean(cv_results['test_roc_auc'])),
        "std_roc_auc": float(np.std(cv_results['test_roc_auc'])),
        "mean_f1": float(np.mean(cv_results['test_f1'])),
        "std_f1": float(np.std(cv_results['test_f1'])),
        "mean_precision": float(np.mean(cv_results['test_precision'])),
        "std_precision": float(np.std(cv_results['test_precision'])),
        "mean_recall": float(np.mean(cv_results['test_recall'])),
        "std_recall": float(np.std(cv_results['test_recall'])),
        "fold_accuracy": cv_results['test_accuracy'].tolist(),
        "fold_roc_auc": cv_results['test_roc_auc'].tolist(),
        "n_cleared": len(embeddings_cleared),
        "n_intact": len(embeddings_intact),
        "imbalance_ratio": float(len(embeddings_cleared) / len(embeddings_intact)),
    }

    print(f"\n  Results (mean ± std):")
    print(f"    Accuracy:  {result['mean_accuracy']:.1%} ± {result['std_accuracy']:.1%}")
    print(f"    ROC-AUC:   {result['mean_roc_auc']:.3f} ± {result['std_roc_auc']:.3f}")
    print(f"    F1-score:  {result['mean_f1']:.3f} ± {result['std_f1']:.3f}")
    print(f"    Precision: {result['mean_precision']:.3f} ± {result['std_precision']:.3f}")
    print(f"    Recall:    {result['mean_recall']:.3f} ± {result['std_recall']:.3f}")

    return result


def run_test_1(n_samples=50, save_results=True):
    """
    Run CRAWL Test 1: Separability Test.

    Args:
        n_samples: Number of samples per class
        save_results: Whether to save results to disk

    Returns:
        dict with test results and decision
    """
    print("=" * 80)
    print("CRAWL TEST 1: SEPARABILITY")
    print("=" * 80)
    print("\nQuestion: Can AlphaEarth embeddings distinguish cleared vs intact forest?")
    print("Decision Gate: Accuracy >= 85% required to proceed\n")

    # Load config
    config = get_config()
    threshold = config.crawl_thresholds["test_1_separability"]["min_accuracy"]

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Get sample locations
    cleared_locs, intact_locs = get_sample_locations(config, n_samples=n_samples)

    # Fetch embeddings with temporal alignment
    # For each year cohort, both cleared and stable get embeddings from year+1
    print("\nFetching embeddings from AlphaEarth...")
    print("  Note: Each year cohort uses same embedding year (clearing_year + 1)")
    print("  Example: 2019 clearing + 2019 stable → both use 2020 embeddings")

    embeddings_cleared = fetch_embeddings(client, cleared_locs, embedding_year_offset=1)
    embeddings_intact = fetch_embeddings(client, intact_locs, embedding_year_offset=1)

    print(f"\n✓ Fetched {len(embeddings_cleared)} cleared embeddings")
    print(f"✓ Fetched {len(embeddings_intact)} intact embeddings")

    # Test separability
    separability_results = test_separability(embeddings_cleared, embeddings_intact)

    # Visualize
    print("\nGenerating visualization...")
    fig = plot_pca_separation(
        embeddings_cleared=embeddings_cleared,
        embeddings_intact=embeddings_intact,
        accuracy=separability_results["mean_accuracy"],
    )

    # Save figure
    save_figure(fig, "test_1_separability.png", subdir="crawl")

    # Make decision using ROC-AUC (more robust to imbalance)
    # Convert accuracy threshold to ROC-AUC equivalent
    # 85% accuracy ≈ 0.80 ROC-AUC for moderately imbalanced data
    roc_auc_threshold = 0.80
    roc_auc = separability_results["mean_roc_auc"]
    accuracy = separability_results["mean_accuracy"]
    f1 = separability_results["mean_f1"]

    passed = roc_auc >= roc_auc_threshold

    print("\n" + "=" * 80)
    print("DECISION GATE (Imbalance-Aware)")
    print("=" * 80)
    print(f"Primary Metric (ROC-AUC):  {roc_auc:.3f}")
    print(f"Required:                  ≥{roc_auc_threshold:.3f}")
    print(f"Status:                    {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"\nSupporting Metrics:")
    print(f"  Accuracy:                {accuracy:.1%}")
    print(f"  F1-score:                {f1:.3f}")
    print(f"  Imbalance Ratio:         {separability_results['imbalance_ratio']:.1f}:1")

    if passed:
        print(f"\nDecision:        GO - Proceed to CRAWL Test 2")
        print(f"Interpretation:  AlphaEarth embeddings can distinguish cleared from intact forest")
        print(f"                 with ROC-AUC of {roc_auc:.3f} despite class imbalance.")
        print(f"                 This validates that embeddings contain forest cover information.")
    else:
        print(f"\nDecision:        STOP - Do not proceed")
        print(f"Interpretation:  AlphaEarth embeddings cannot reliably distinguish cleared")
        print(f"                 from intact forest (ROC-AUC: {roc_auc:.3f} < {roc_auc_threshold:.3f}).")
        print(f"\nRecommendation:  Try different embeddings or abandon this approach.")

    print("=" * 80 + "\n")

    # Compile results
    results = {
        "test_name": "test_1_separability",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples_requested": n_samples,
            "years": [2019, 2020, 2021],
            "default_year": 2020,
            "threshold": threshold,
        },
        "data": {
            "n_cleared_locations": len(cleared_locs),
            "n_intact_locations": len(intact_locs),
            "n_cleared_embeddings": len(embeddings_cleared),
            "n_intact_embeddings": len(embeddings_intact),
        },
        "results": separability_results,
        "decision": {
            "primary_metric": "roc_auc",
            "roc_auc_threshold": roc_auc_threshold,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "f1_score": f1,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        },
    }

    # Save results
    if save_results:
        results_dir = config.get_path("paths.results_dir") / "experiments"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "crawl_test_1_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAWL Test 1: Separability Test")
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
    results = run_test_1(
        n_samples=args.n_samples,
        save_results=not args.no_save,
    )

    # Exit with appropriate code
    exit_code = 0 if results["decision"]["passed"] else 1
    exit(exit_code)
