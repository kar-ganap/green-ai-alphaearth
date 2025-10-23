#!/usr/bin/env python3
"""
Uniform 30% Threshold Collection (2020-2023)

Recollects 2020-2023 training data using a UNIFORM 30% tree cover threshold
across all clearing types, matching the 2024 collection methodology.

Purpose: Isolate temporal drift from sampling bias by ensuring apples-to-apples
         comparison across all years.

Key Difference from Original (08_phase1_scaled_data_collection.py):
  - Original: Mixed 30%, 40%, 50% thresholds
  - This script: Uniform 30% threshold across ALL samples

Sample Distribution (same as original):
- 180 standard clearings (>1 ha, 60%)
- 60 small clearings (<1 ha, 20%)
- 30 fire-prone clearings (10%)
- 30 edge expansion clearings (10%)
- 300 matched intact forest samples

Expected Runtime: ~11 hours (300 clearings × 2-3 min/clearing category)

Output: walk_dataset_uniform_30pct_2020_2023.pkl

Usage:
    uv run python src/walk/38_collect_uniform_30pct_2020_2023.py
    uv run python src/walk/38_collect_uniform_30pct_2020_2023.py --dry-run  # Preview only
"""

import argparse
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import ee

from src.utils import EarthEngineClient, get_config
from src.walk.data_leakage_verification import verify_no_spatial_leakage


# UNIFORM 30% THRESHOLD FOR ALL CLEARING TYPES
UNIFORM_TREE_COVER_THRESHOLD = 30


def get_standard_clearings(client, bounds, year_range, n_samples=180, min_size_ha=1.0):
    """
    Sample standard-size clearings (>1 ha) with UNIFORM 30% threshold.

    CHANGED: tree_cover.gte(50) → tree_cover.gte(30)
    """
    print(f"\nCollecting {n_samples} standard clearings (>1 ha, 30% threshold)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Year range for diversity (2020-2023)
    min_year_code = 20  # 2020
    max_year_code = 23  # 2023

    # UNIFORM 30% THRESHOLD (was 50%)
    mask = (
        tree_cover.gte(UNIFORM_TREE_COVER_THRESHOLD)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # ITERATIVE SAMPLING (matching 2024 methodology)
    print("  Using iterative sampling to accumulate enough samples...")
    all_features = []
    max_iterations = 5
    samples_per_iteration = n_samples * 50

    for iteration in range(max_iterations):
        if len(all_features) >= n_samples:
            break

        print(f"    Iteration {iteration + 1}/{max_iterations}: sampling {samples_per_iteration} pixels...")

        sample = mask.selfMask().sample(
            region=roi,
            scale=30,
            numPixels=samples_per_iteration,
            seed=1000 + iteration,
            geometries=True
        )

        iteration_features = sample.getInfo()["features"]
        print(f"      Got {len(iteration_features)} samples")
        all_features.extend(iteration_features)

    features = all_features
    print(f"  Total found: {len(features)} potential locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} samples, target was {n_samples}")

    if len(features) > n_samples:
        random.seed(1000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year = random.choice(year_range)
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year,
            "date": f"{year}-06-01",
            "source": "GFW",
            "category": "standard",
            "min_size_ha": min_size_ha,
            "tree_cover_threshold": UNIFORM_TREE_COVER_THRESHOLD
        })

    print(f"  ✓ Collected {len(samples)} standard clearings")
    return samples


def get_small_clearings(client, bounds, year_range, n_samples=60, max_size_ha=1.0):
    """
    Sample small-scale clearings (<1 ha) with UNIFORM 30% threshold.

    CHANGED: tree_cover.gte(40) → tree_cover.gte(30)
    """
    print(f"\nCollecting {n_samples} small clearings (<1 ha, 30% threshold)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 20
    max_year_code = 23

    # UNIFORM 30% THRESHOLD (was 40%)
    mask = (
        tree_cover.gte(UNIFORM_TREE_COVER_THRESHOLD)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # ITERATIVE SAMPLING (matching 2024 methodology)
    print("  Using iterative sampling to accumulate enough samples...")
    all_features = []
    max_iterations = 5
    samples_per_iteration = n_samples * 50

    for iteration in range(max_iterations):
        if len(all_features) >= n_samples:
            break

        print(f"    Iteration {iteration + 1}/{max_iterations}: sampling {samples_per_iteration} pixels...")

        sample = mask.selfMask().sample(
            region=roi,
            scale=30,
            numPixels=samples_per_iteration,
            seed=2000 + iteration,
            geometries=True
        )

        iteration_features = sample.getInfo()["features"]
        print(f"      Got {len(iteration_features)} samples")
        all_features.extend(iteration_features)

    features = all_features
    print(f"  Total found: {len(features)} potential locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} samples, target was {n_samples}")

    if len(features) > n_samples:
        random.seed(2000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year = random.choice(year_range)
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year,
            "date": f"{year}-06-01",
            "source": "GFW",
            "category": "small",
            "max_size_ha": max_size_ha,
            "tree_cover_threshold": UNIFORM_TREE_COVER_THRESHOLD
        })

    print(f"  ✓ Collected {len(samples)} small clearings")
    return samples


def get_fire_prone_clearings(client, bounds, year_range, n_samples=30):
    """
    Sample clearings from fire-prone regions with UNIFORM 30% threshold.

    UNCHANGED: Already used 30% threshold
    """
    print(f"\nCollecting {n_samples} fire-prone clearings (30% threshold)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 20
    max_year_code = 23

    # UNIFORM 30% THRESHOLD (unchanged)
    mask = (
        tree_cover.gte(UNIFORM_TREE_COVER_THRESHOLD)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # ITERATIVE SAMPLING (matching 2024 methodology)
    print("  Using iterative sampling to accumulate enough samples...")
    all_features = []
    max_iterations = 5
    samples_per_iteration = n_samples * 50

    for iteration in range(max_iterations):
        if len(all_features) >= n_samples:
            break

        print(f"    Iteration {iteration + 1}/{max_iterations}: sampling {samples_per_iteration} pixels...")

        sample = mask.selfMask().sample(
            region=roi,
            scale=30,
            numPixels=samples_per_iteration,
            seed=3000 + iteration,
            geometries=True
        )

        iteration_features = sample.getInfo()["features"]
        print(f"      Got {len(iteration_features)} samples")
        all_features.extend(iteration_features)

    features = all_features
    print(f"  Total found: {len(features)} potential locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} samples, target was {n_samples}")

    if len(features) > n_samples:
        random.seed(3000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year = random.choice(year_range)
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year,
            "date": f"{year}-06-01",
            "source": "GFW",
            "category": "fire_prone",
            "tree_cover_threshold": UNIFORM_TREE_COVER_THRESHOLD
        })

    print(f"  ✓ Collected {len(samples)} fire-prone clearings")
    return samples


def get_edge_expansion_clearings(client, bounds, year_range, n_samples=30):
    """
    Sample clearings near forest edges with UNIFORM 30% threshold.

    UNCHANGED: Already used 30% threshold
    """
    print(f"\nCollecting {n_samples} edge expansion clearings (30% threshold)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 20
    max_year_code = 23

    # Current loss with UNIFORM 30% THRESHOLD (unchanged)
    current_loss = (
        tree_cover.gte(UNIFORM_TREE_COVER_THRESHOLD)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # Previous loss
    previous_loss = loss.eq(1).And(loss_year.lt(min_year_code))

    # Distance to previous loss
    distance = previous_loss.fastDistanceTransform().sqrt().multiply(30)

    # Edge expansion: current loss near previous loss
    edge_loss = current_loss.updateMask(distance.lte(90))

    # ITERATIVE SAMPLING (matching 2024 methodology)
    print("  Using iterative sampling to accumulate enough samples...")
    all_features = []
    max_iterations = 5
    samples_per_iteration = n_samples * 50

    for iteration in range(max_iterations):
        if len(all_features) >= n_samples:
            break

        print(f"    Iteration {iteration + 1}/{max_iterations}: sampling {samples_per_iteration} pixels...")

        sample = edge_loss.sample(
            region=roi,
            scale=30,
            numPixels=samples_per_iteration,
            seed=4000 + iteration,
            geometries=True
        )

        iteration_features = sample.getInfo()["features"]
        print(f"      Got {len(iteration_features)} samples")
        all_features.extend(iteration_features)

    features = all_features
    print(f"  Total found: {len(features)} potential locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} samples, target was {n_samples}")

    if len(features) > n_samples:
        random.seed(4000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year = random.choice(year_range)
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year,
            "date": f"{year}-06-01",
            "source": "GFW",
            "category": "edge_expansion",
            "tree_cover_threshold": UNIFORM_TREE_COVER_THRESHOLD
        })

    print(f"  ✓ Collected {len(samples)} edge expansion clearings")
    return samples


def get_intact_forest_samples(client, bounds, n_samples=300):
    """
    Generate intact forest samples across diverse regions.

    UNCHANGED: Intact samples don't use tree cover threshold
    """
    print(f"\nCollecting {n_samples} intact forest samples...")

    # Use diverse intact forest regions
    intact_regions = [
        {"name": "Amazon Core", "bounds": {"min_lon": -60, "max_lon": -55, "min_lat": -5, "max_lat": 0}},
        {"name": "Guiana Shield", "bounds": {"min_lon": -55, "max_lon": -50, "min_lat": 2, "max_lat": 6}},
        {"name": "Central Amazon", "bounds": {"min_lon": -65, "max_lon": -60, "min_lat": -2, "max_lat": 2}},
        {"name": "Western Amazon", "bounds": {"min_lon": -75, "max_lon": -70, "min_lat": -5, "max_lat": 0}},
    ]

    samples = []
    samples_per_region = n_samples // len(intact_regions)

    np.random.seed(5000)

    for region in intact_regions:
        region_bounds = region['bounds']

        for _ in range(samples_per_region):
            lat = np.random.uniform(region_bounds['min_lat'], region_bounds['max_lat'])
            lon = np.random.uniform(region_bounds['min_lon'], region_bounds['max_lon'])
            year = np.random.choice([2020, 2021, 2022, 2023])

            samples.append({
                'lat': lat,
                'lon': lon,
                'year': year,
                'stable': True,
                'region': region['name'],
                'category': 'intact'
            })

    # Fill remaining samples
    while len(samples) < n_samples:
        region = intact_regions[len(samples) % len(intact_regions)]
        region_bounds = region['bounds']
        lat = np.random.uniform(region_bounds['min_lat'], region_bounds['max_lat'])
        lon = np.random.uniform(region_bounds['min_lon'], region_bounds['max_lon'])
        year = np.random.choice([2020, 2021, 2022, 2023])

        samples.append({
            'lat': lat,
            'lon': lon,
            'year': year,
            'stable': True,
            'region': region['name'],
            'category': 'intact'
        })

    print(f"  ✓ Generated {len(samples)} intact samples")
    return samples


def filter_spatial_exclusion(samples, exclusion_coords, min_distance_km=10.0):
    """
    Filter samples to exclude those too close to validation sets.

    UNCHANGED: Same spatial exclusion logic
    """
    from scipy.spatial import cKDTree

    if not exclusion_coords:
        return samples

    print(f"\nApplying spatial exclusion (10km buffer)...")
    print(f"  Exclusion set: {len(exclusion_coords)} coordinates")

    # Build tree from exclusion coords
    exclusion_tree = cKDTree(exclusion_coords)

    filtered = []
    for sample in samples:
        sample_coord = (sample['lat'], sample['lon'])
        distance_deg, _ = exclusion_tree.query(sample_coord)
        distance_km = distance_deg * 111.0

        if distance_km >= min_distance_km:
            filtered.append(sample)

    n_before = len(samples)
    n_after = len(filtered)
    n_removed = n_before - n_after

    print(f"  Before: {n_before} samples")
    print(f"  After: {n_after} samples")
    print(f"  Removed: {n_removed} samples ({n_removed/n_before*100:.1f}%)")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='Uniform 30% threshold collection for 2020-2023 (temporal drift experiment)'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview collection without saving')

    args = parser.parse_args()

    print("=" * 80)
    print("UNIFORM 30% THRESHOLD COLLECTION (2020-2023)")
    print("=" * 80)
    print("\n⚠️  EXPERIMENT: Isolating temporal drift from sampling bias")
    print(f"\nUniform tree cover threshold: {UNIFORM_TREE_COVER_THRESHOLD}%")
    print("\nTarget: 600 total samples (300 clearing + 300 intact)")
    print("\nClearing Distribution:")
    print("  - 180 standard clearings (>1 ha, 60%) @ 30%")
    print("  - 60 small clearings (<1 ha, 20%) @ 30%")
    print("  - 30 fire-prone clearings (10%) @ 30%")
    print("  - 30 edge expansion clearings (10%) @ 30%")
    print("\nIntact: 300 diverse forest samples")
    print("\nComparison:")
    print("  - Original: Mixed 50%, 40%, 30%, 30% thresholds")
    print("  - This run: Uniform 30% across all types")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No data will be saved")

    # Initialize
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Main study region
    main_bounds = config.study_region_bounds
    year_range = [2020, 2021, 2022, 2023]

    # Load existing validation sets for spatial exclusion
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    print("\n" + "=" * 80)
    print("LOADING VALIDATION SETS FOR SPATIAL EXCLUSION")
    print("=" * 80)

    exclusion_coords = []

    val_sets = ['hard_val_risk_ranking.pkl', 'hard_val_rapid_response.pkl',
                'hard_val_comprehensive.pkl', 'hard_val_edge_cases.pkl']

    for val_set in val_sets:
        val_path = processed_dir / val_set
        if val_path.exists():
            with open(val_path, 'rb') as f:
                val_data = pickle.load(f)
            for sample in val_data:
                exclusion_coords.append((sample['lat'], sample['lon']))
            print(f"  ✓ Loaded {len(val_data)} samples from {val_set}")

    print(f"\nTotal exclusion coordinates: {len(exclusion_coords)}")

    # Collect clearing samples
    print("\n" + "=" * 80)
    print("COLLECTING CLEARING SAMPLES")
    print("=" * 80)

    all_clearings = []

    # 1. Standard clearings (180) @ 30%
    standard = get_standard_clearings(client, main_bounds, year_range, n_samples=180)
    all_clearings.extend(standard)

    # 2. Small clearings (60) @ 30%
    small = get_small_clearings(client, main_bounds, year_range, n_samples=60)
    all_clearings.extend(small)

    # 3. Fire-prone (30) @ 30%
    fire = get_fire_prone_clearings(client, main_bounds, year_range, n_samples=30)
    all_clearings.extend(fire)

    # 4. Edge expansion (30) @ 30%
    edges = get_edge_expansion_clearings(client, main_bounds, year_range, n_samples=30)
    all_clearings.extend(edges)

    print(f"\n✓ Total clearing samples collected: {len(all_clearings)}")

    # Collect intact samples
    print("\n" + "=" * 80)
    print("COLLECTING INTACT SAMPLES")
    print("=" * 80)

    intact = get_intact_forest_samples(client, main_bounds, n_samples=300)

    # Combine and add labels
    all_samples = []

    for sample in all_clearings:
        sample['label'] = 1  # Clearing
        all_samples.append(sample)

    for sample in intact:
        sample['label'] = 0  # Intact
        all_samples.append(sample)

    print(f"\n✓ Total samples before filtering: {len(all_samples)}")
    print(f"  Clearing: {len(all_clearings)}")
    print(f"  Intact: {len(intact)}")

    # Apply spatial exclusion
    print("\n" + "=" * 80)
    print("SPATIAL EXCLUSION")
    print("=" * 80)

    filtered_samples = filter_spatial_exclusion(all_samples, exclusion_coords)

    # Verify no spatial leakage
    print("\n" + "=" * 80)
    print("VERIFYING SPATIAL SEPARATION")
    print("=" * 80)

    if exclusion_coords:
        is_valid, report = verify_no_spatial_leakage(
            [{'lat': lat, 'lon': lon} for lat, lon in exclusion_coords],
            filtered_samples,
            min_distance_km=10.0
        )

        if not is_valid:
            print(f"\n❌ VERIFICATION FAILED: {report['n_violations']} violations")
            print("   Cannot proceed with spatial leakage")
            return
        else:
            print(f"\n✓ Verification passed: 0 violations")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    n_clearing_final = sum(1 for s in filtered_samples if s['label'] == 1)
    n_intact_final = sum(1 for s in filtered_samples if s['label'] == 0)

    print(f"\nTotal samples: {len(filtered_samples)}")
    print(f"  Clearing: {n_clearing_final}")
    print(f"  Intact: {n_intact_final}")

    # Category breakdown
    categories = {}
    for sample in filtered_samples:
        cat = sample.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save
    if not args.dry_run:
        print("\n" + "=" * 80)
        print("SAVING DATASET")
        print("=" * 80)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = processed_dir / f'walk_dataset_uniform_30pct_2020_2023_{timestamp}.pkl'

        output_data = {
            'data': filtered_samples,
            'metadata': {
                'created': datetime.now().isoformat(),
                'experiment': 'uniform_30pct_temporal_drift',
                'tree_cover_threshold': UNIFORM_TREE_COVER_THRESHOLD,
                'target_samples': 600,
                'actual_samples': len(filtered_samples),
                'clearing_target': 300,
                'clearing_actual': n_clearing_final,
                'intact_target': 300,
                'intact_actual': n_intact_final,
                'clearing_distribution': {
                    'standard': 180,
                    'small': 60,
                    'fire_prone': 30,
                    'edge_expansion': 30
                },
                'year_range': year_range,
                'spatial_exclusion_km': 10.0,
                'categories': categories,
                'note': 'Uniform 30% threshold to isolate temporal drift from sampling bias'
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\n✓ Saved to: {output_path}")
        print(f"  Samples: {len(filtered_samples)}")
        print(f"  Clearing: {n_clearing_final} (target: 300)")
        print(f"  Intact: {n_intact_final} (target: 300)")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Extract features for uniform 30% dataset:")
        print("   uv run python src/walk/39_extract_features_uniform_30pct.py")
        print("\n2. Train uniform 30% model and compare with heterogeneous:")
        print("   uv run python src/walk/40_compare_sampling_strategies.py")
        print("\n3. Temporal validation (2020-2023 → 2024) with uniform 30%:")
        print("   - If ROC-AUC ~0.98: Drift was sampling bias")
        print("   - If ROC-AUC ~0.80: Drift is real temporal change")
        print("   - If ROC-AUC intermediate: Both effects present")
    else:
        print("\n⚠️  DRY RUN COMPLETE - No data saved")


if __name__ == '__main__':
    main()
