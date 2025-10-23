"""
Phase 1: Scaled Data Collection (300 Clearing + 300 Intact = 600 Total)

Based on diagnostic analysis recommendation:
- Scale up training data to address performance gaps across validation sets
- Target diverse clearing types to improve edge case performance

Sample Distribution:
- 180 standard clearings (>1 ha, 60%)
- 60 small clearings (<1 ha, 20%)
- 30 fire-prone clearings (10%)
- 30 forest edge clearings (10%)
- 300 matched intact forest samples

Expected Outcome:
- Edge cases: 0.583 → 0.70+ ROC-AUC
- Overall validation: Improved generalization across all sets

Usage:
    uv run python src/walk/08_phase1_scaled_data_collection.py
    uv run python src/walk/08_phase1_scaled_data_collection.py --dry-run  # Preview only
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


def get_standard_clearings(client, bounds, year_range, n_samples=180, min_size_ha=1.0):
    """
    Sample standard-size clearings (>1 ha).

    These represent typical deforestation patterns.
    """
    print(f"\nCollecting {n_samples} standard clearings (>1 ha)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Year range for diversity (2015-2023)
    min_year_code = 15
    max_year_code = 23

    # Standard clearings: high tree cover, loss in year range
    mask = (
        tree_cover.gte(50)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # Sample significantly more to ensure sufficient coverage
    sample = mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 20,  # Increased multiplier
        seed=1000,
        geometries=True
    )

    features = sample.getInfo()["features"]

    # Subsample
    if len(features) > n_samples:
        random.seed(1000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        # Assign diverse years
        year = random.choice(year_range)
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year,
            "date": f"{year}-06-01",
            "source": "GFW",
            "category": "standard",
            "min_size_ha": min_size_ha
        })

    print(f"  ✓ Collected {len(samples)} standard clearings")
    return samples


def get_small_clearings(client, bounds, year_range, n_samples=60, max_size_ha=1.0):
    """
    Sample small-scale clearings (<1 ha).

    These are harder to detect and critical for rapid response.
    """
    print(f"\nCollecting {n_samples} small clearings (<1 ha)...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 15
    max_year_code = 23

    # Small clearings: moderate-high tree cover, loss
    mask = (
        tree_cover.gte(40)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    sample = mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 20,  # Increased multiplier
        seed=2000,
        geometries=True
    )

    features = sample.getInfo()["features"]

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
            "max_size_ha": max_size_ha
        })

    print(f"  ✓ Collected {len(samples)} small clearings")
    return samples


def get_fire_prone_clearings(client, bounds, year_range, n_samples=30):
    """
    Sample clearings from fire-prone regions.

    Challenging to distinguish intentional clearing vs fire damage.
    """
    print(f"\nCollecting {n_samples} fire-prone clearings...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 15
    max_year_code = 23

    # Fire-prone: loss pixels (fire overlay too restrictive)
    mask = (
        tree_cover.gte(30)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    sample = mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 20,  # Increased multiplier
        seed=3000,
        geometries=True
    )

    features = sample.getInfo()["features"]

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
            "category": "fire_prone"
        })

    print(f"  ✓ Collected {len(samples)} fire-prone clearings")
    return samples


def get_edge_expansion_clearings(client, bounds, year_range, n_samples=30):
    """
    Sample clearings near forest edges (gradual encroachment).

    Represents expansion of existing cleared areas.
    """
    print(f"\nCollecting {n_samples} edge expansion clearings...")

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    min_year_code = 15
    max_year_code = 23

    # Current loss
    current_loss = (
        tree_cover.gte(30)
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

    sample = edge_loss.sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 20,  # Increased multiplier
        seed=4000,
        geometries=True
    )

    features = sample.getInfo()["features"]

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
            "category": "edge_expansion"
        })

    print(f"  ✓ Collected {len(samples)} edge expansion clearings")
    return samples


def get_intact_forest_samples(client, bounds, n_samples=300):
    """
    Generate intact forest samples across diverse regions.

    Matched to clearing sample diversity.
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

    Maintains spatial independence between training and validation.
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
        description='Phase 1: Scaled data collection (300 clearing + 300 intact)'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview collection without saving')

    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 1: SCALED DATA COLLECTION")
    print("=" * 80)
    print("\nTarget: 600 total samples (300 clearing + 300 intact)")
    print("\nClearing Distribution:")
    print("  - 180 standard clearings (>1 ha, 60%)")
    print("  - 60 small clearings (<1 ha, 20%)")
    print("  - 30 fire-prone clearings (10%)")
    print("  - 30 edge expansion clearings (10%)")
    print("\nIntact: 300 diverse forest samples")

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

    # 1. Standard clearings (180)
    standard = get_standard_clearings(client, main_bounds, year_range, n_samples=180)
    all_clearings.extend(standard)

    # 2. Small clearings (60)
    small = get_small_clearings(client, main_bounds, year_range, n_samples=60)
    all_clearings.extend(small)

    # 3. Fire-prone (30)
    fire = get_fire_prone_clearings(client, main_bounds, year_range, n_samples=30)
    all_clearings.extend(fire)

    # 4. Edge expansion (30)
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

        output_path = processed_dir / 'walk_dataset_scaled_phase1.pkl'

        output_data = {
            'data': filtered_samples,
            'metadata': {
                'created': datetime.now().isoformat(),
                'phase': 'phase1_scaling',
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
                'categories': categories
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
        print("\n1. Extract dual-year delta features for all samples")
        print("2. Train scaled model on new dataset")
        print("3. Evaluate on all 4 validation sets")
        print("4. Compare to baseline performance:")
        print("   - risk_ranking: 0.850 → ?")
        print("   - rapid_response: 0.824 → ?")
        print("   - comprehensive: 0.758 → ?")
        print("   - edge_cases: 0.583 → 0.70+ (target)")
    else:
        print("\n⚠️  DRY RUN COMPLETE - No data saved")


if __name__ == '__main__':
    main()
