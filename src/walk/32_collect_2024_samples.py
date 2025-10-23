#!/usr/bin/env python3
"""
Collect 2024 Training Samples

Collects ~170 samples from 2024 to complete the temporal sequence:
- Target: 85 clearing + 85 intact = 170 total
- Matches 2022/2023 sample sizes (172, 165)
- Ensures spatial separation from validation sets (10km buffer)

This enables:
- Phase 4 temporal validation (2020-2023 → 2024)
- Production model training on 2020-2024 (855 total samples)

Usage:
    uv run python src/walk/32_collect_2024_samples.py
    uv run python src/walk/32_collect_2024_samples.py --dry-run
"""

import argparse
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import ee
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.data_leakage_verification import verify_no_spatial_leakage


def load_existing_samples():
    """Load existing training samples to ensure no spatial overlap."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load latest training dataset
    pattern = 'walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'
    files = list(processed_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No training data found matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Loading existing training data: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    print(f"  Existing samples: {len(samples)}")

    # Group by year
    by_year = {}
    for s in samples:
        year = s.get('year', 2021)
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(s)

    print("\n  Samples by year:")
    for year in sorted(by_year.keys()):
        n_clearing = sum(1 for s in by_year[year] if s.get('label', 0) == 1)
        n_intact = sum(1 for s in by_year[year] if s.get('label', 0) == 0)
        print(f"    {year}: {len(by_year[year])} total ({n_clearing} clearing, {n_intact} intact)")

    return samples


def load_validation_sets():
    """Load validation sets to ensure spatial separation."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    val_sets = {}
    for set_name in ['edge_cases', 'rapid_response', 'comprehensive', 'risk_ranking']:
        val_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'
        if val_path.exists():
            with open(val_path, 'rb') as f:
                val_sets[set_name] = pickle.load(f)
                print(f"✓ Loaded {len(val_sets[set_name])} {set_name} validation samples")

    return val_sets


def get_existing_locations(samples):
    """Extract all existing sample locations."""
    locations = []
    for sample in samples:
        lat = sample.get('lat') or sample.get('latitude')
        lon = sample.get('lon') or sample.get('longitude')
        if lat is not None and lon is not None:
            locations.append((lat, lon))
    return np.array(locations)


def check_spatial_separation(new_loc, existing_locs, min_distance_km=10.0):
    """
    Check if new location is at least min_distance_km away from all existing locations.

    Uses haversine distance approximation.
    """
    if len(existing_locs) == 0:
        return True

    new_lat, new_lon = new_loc

    # Vectorized haversine distance (approximate)
    lat1 = np.radians(new_lat)
    lon1 = np.radians(new_lon)
    lat2 = np.radians(existing_locs[:, 0])
    lon2 = np.radians(existing_locs[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distances_km = 6371 * c  # Earth radius in km

    return np.all(distances_km >= min_distance_km)


def collect_2024_clearing_samples(client, n_samples=85, bounds=None):
    """
    Collect clearing samples from 2024.

    Args:
        client: EarthEngineClient instance
        n_samples: Number of clearing samples to collect
        bounds: Geographic bounds (defaults to Amazon basin)

    Returns:
        List of sample dictionaries
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING 2024 CLEARING SAMPLES")
    print(f"{'='*80}\n")
    print(f"Target: {n_samples} samples")

    if bounds is None:
        # Amazon basin default
        bounds = {
            "min_lat": -15,
            "max_lat": 5,
            "min_lon": -75,
            "max_lon": -50
        }

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    # Load Hansen GFC
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Filter for 2024 clearings (lossyear=24)
    # Using lower threshold (30% instead of 50%) to capture more samples
    mask = (
        tree_cover.gte(30)  # Lower tree cover threshold for better sampling
        .And(loss.eq(1))     # Loss occurred
        .And(loss_year.eq(24))  # Loss in 2024
    )

    print("Sampling 2024 Hansen clearings (tree cover >=30%)...")

    # Try multiple sampling iterations if needed
    all_features = []
    max_iterations = 5
    samples_per_iteration = n_samples * 50  # Much larger multiplier

    for iteration in range(max_iterations):
        if len(all_features) >= n_samples:
            break

        print(f"  Iteration {iteration + 1}/{max_iterations}: sampling {samples_per_iteration} pixels...")

        sample = mask.selfMask().sample(
            region=roi,
            scale=30,
            numPixels=samples_per_iteration,
            seed=2024 + iteration,
            geometries=True
        )

        iteration_features = sample.getInfo()["features"]
        print(f"    Got {len(iteration_features)} samples")
        all_features.extend(iteration_features)

    features = all_features
    print(f"  Total found: {len(features)} potential clearing locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} clearings, target was {n_samples}")
        n_samples = len(features)

    # Subsample if needed
    if len(features) > n_samples:
        random.seed(2024)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": 2024,
            "date": "2024-06-01",  # Mid-year date
            "source": "GFW",
            "category": "clearing_2024",
            "label": 1  # Clearing
        })

    print(f"✓ Collected {len(samples)} clearing samples for 2024\n")
    return samples


def collect_2024_intact_samples(client, n_samples=85, bounds=None):
    """
    Collect intact forest samples for 2024.

    Args:
        client: EarthEngineClient instance
        n_samples: Number of intact samples to collect
        bounds: Geographic bounds (defaults to Amazon basin)

    Returns:
        List of sample dictionaries
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING 2024 INTACT SAMPLES")
    print(f"{'='*80}\n")
    print(f"Target: {n_samples} samples")

    if bounds is None:
        bounds = {
            "min_lat": -15,
            "max_lat": 5,
            "min_lon": -75,
            "max_lon": -50
        }

    roi = ee.Geometry.Rectangle([
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"]
    ])

    # Load Hansen GFC
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")

    # Intact: high tree cover, NO loss through 2024
    mask = (
        tree_cover.gte(70)  # High tree cover
        .And(loss.eq(0))     # No loss
    )

    print("Sampling intact forest locations...")
    sample = mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 10,
        seed=2024 + 1000,
        geometries=True
    )

    features = sample.getInfo()["features"]
    print(f"  Found {len(features)} potential intact locations")

    if len(features) < n_samples:
        print(f"  WARNING: Only found {len(features)} intact samples, target was {n_samples}")
        n_samples = len(features)

    if len(features) > n_samples:
        random.seed(2024 + 1000)
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": 2024,
            "date": "2024-06-01",
            "source": "GFW",
            "category": "intact_2024",
            "label": 0  # Intact
        })

    print(f"✓ Collected {len(samples)} intact samples for 2024\n")
    return samples


def filter_spatial_leakage(new_samples, existing_locs, min_distance_km=10.0):
    """
    Filter out samples that are too close to existing training/validation samples.

    Args:
        new_samples: List of new sample dictionaries
        existing_locs: Numpy array of (lat, lon) tuples for existing samples
        min_distance_km: Minimum separation distance in km

    Returns:
        Filtered list of samples
    """
    print(f"\n{'='*80}")
    print(f"SPATIAL LEAKAGE FILTERING")
    print(f"{'='*80}\n")
    print(f"Original samples: {len(new_samples)}")
    print(f"Existing locations: {len(existing_locs)}")
    print(f"Minimum separation: {min_distance_km} km")

    filtered = []
    rejected = 0

    for sample in tqdm(new_samples, desc="Checking spatial separation"):
        new_loc = (sample['lat'], sample['lon'])

        if check_spatial_separation(new_loc, existing_locs, min_distance_km):
            filtered.append(sample)
        else:
            rejected += 1

    print(f"\n✓ Kept: {len(filtered)} samples")
    print(f"✗ Rejected: {rejected} samples (too close to existing)")

    return filtered


def save_samples(samples, dry_run=False):
    """Save collected samples to disk."""
    if dry_run:
        print("\n[DRY RUN] Would save samples, but skipping...")
        return None

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = processed_dir / f'walk_dataset_2024_raw_{timestamp}.pkl'

    data = {
        'samples': samples,
        'metadata': {
            'n_samples': len(samples),
            'n_clearing': sum(1 for s in samples if s.get('label') == 1),
            'n_intact': sum(1 for s in samples if s.get('label') == 0),
            'year': 2024,
            'collection_date': timestamp,
            'min_spatial_separation_km': 10.0
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✓ Saved {len(samples)} samples to:")
    print(f"  {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Collect 2024 training samples')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview collection without saving')
    parser.add_argument('--n-clearing', type=int, default=85,
                       help='Number of clearing samples to collect')
    parser.add_argument('--n-intact', type=int, default=85,
                       help='Number of intact samples to collect')

    args = parser.parse_args()

    print("="*80)
    print("2024 SAMPLE COLLECTION")
    print("="*80)
    print(f"\nTarget samples: {args.n_clearing} clearing + {args.n_intact} intact = {args.n_clearing + args.n_intact} total")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize Earth Engine client
    print("Initializing Earth Engine...")
    client = EarthEngineClient()
    print("✓ Earth Engine initialized\n")

    # Load existing samples for spatial separation
    print("Loading existing training samples...")
    existing_samples = load_existing_samples()

    print("\nLoading validation sets...")
    val_sets = load_validation_sets()

    # Combine all existing locations
    print("\nExtracting existing locations...")
    all_existing_locs = get_existing_locations(existing_samples)

    for set_name, val_samples in val_sets.items():
        val_locs = get_existing_locations(val_samples)
        all_existing_locs = np.vstack([all_existing_locs, val_locs])

    print(f"✓ Total existing locations: {len(all_existing_locs)}")

    # Collect 2024 samples
    clearing_samples = collect_2024_clearing_samples(client, n_samples=args.n_clearing)
    intact_samples = collect_2024_intact_samples(client, n_samples=args.n_intact)

    # Combine
    all_new_samples = clearing_samples + intact_samples
    print(f"\n✓ Total new samples collected: {len(all_new_samples)}")
    print(f"  Clearing: {len(clearing_samples)}")
    print(f"  Intact: {len(intact_samples)}")

    # Filter for spatial separation
    filtered_samples = filter_spatial_leakage(all_new_samples, all_existing_locs, min_distance_km=10.0)

    # Summary
    print(f"\n{'='*80}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*80}\n")

    n_clearing_final = sum(1 for s in filtered_samples if s.get('label') == 1)
    n_intact_final = sum(1 for s in filtered_samples if s.get('label') == 0)

    print(f"Final samples: {len(filtered_samples)}")
    print(f"  Clearing: {n_clearing_final}")
    print(f"  Intact: {n_intact_final}")
    print(f"  Rejection rate: {(1 - len(filtered_samples)/len(all_new_samples))*100:.1f}%")

    # Save
    output_path = save_samples(filtered_samples, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}\n")
        print("1. Extract features:")
        print(f"   uv run python src/walk/33_extract_features_2024.py")
        print("\n2. Run Phase 4 temporal validation:")
        print(f"   uv run python src/walk/34_phase4_temporal_validation.py")
        print("\n3. Retrain production model with 2020-2024:")
        print(f"   uv run python src/walk/35_train_production_model.py")

    return filtered_samples


if __name__ == '__main__':
    main()
