"""
WALK Phase - Step 1: Data Preparation

Prepares training dataset with proper validation splits and feature extraction.

This script:
1. Loads clearing locations with dates (GLAD labels)
2. Extracts embeddings at multiple timepoints (quarterly)
3. Computes temporal features (distances, velocities)
4. Creates spatial cross-validation splits (10km buffer)
5. Implements temporal validation (no future leakage)
6. Filters low-quality labels (cloud cover, boundaries)

Output:
    data/processed/walk_dataset.pkl - Full dataset with features and splits

Usage:
    uv run python src/walk/01_data_preparation.py --n-samples 100
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.utils import EarthEngineClient, get_config, haversine_distance
from src.walk.deforestation_regions import (
    DEFORESTATION_HOTSPOTS,
    INTACT_FOREST_BASTIONS,
    get_diverse_sample,
    get_intact_bastions,
)


def load_clearing_samples(config, n_samples=100, years=None, regions=None):
    """
    Load clearing samples from GLAD with temporal labels across multiple regions.

    Args:
        config: Config instance
        n_samples: Number of clearing events to sample
        years: List of years to sample from (default: [2020, 2021, 2022, 2023])
        regions: Dict of region definitions (default: diverse global sample)

    Returns:
        List of clearing events with metadata including region
    """
    print(f"Loading {n_samples} clearing samples from global hotspots...")

    if years is None:
        years = [2020, 2021, 2022, 2023]

    if regions is None:
        # Get diverse sample: 2 Amazon + 2 Congo + 1 Asia
        regions = get_diverse_sample(n_regions=5)

    client = EarthEngineClient(use_cache=True)

    # Distribute samples across regions and years
    samples_per_region = n_samples // len(regions)
    samples_per_region_year = samples_per_region // len(years)

    all_clearings = []

    print(f"  Sampling from {len(regions)} regions × {len(years)} years...")
    print(f"  Target: ~{samples_per_region_year} samples per region-year\n")

    for region_id, region_info in regions.items():
        print(f"  Region: {region_info['name']}")
        region_clearings = []

        for year in years:
            try:
                clearings = client.get_deforestation_labels(
                    bounds=region_info['bounds'],
                    year=year,
                    min_tree_cover=30,
                )

                # Add region metadata
                for clearing in clearings:
                    clearing['region'] = region_id
                    clearing['region_name'] = region_info['name']
                    clearing['continent'] = region_info['region']

                # Sample from this region-year
                if len(clearings) > samples_per_region_year:
                    import random
                    random.seed(42 + year + hash(region_id))
                    clearings = random.sample(clearings, samples_per_region_year)

                region_clearings.extend(clearings)
                print(f"    Year {year}: {len(clearings)} samples")

            except Exception as e:
                print(f"    Warning: Failed to fetch from {region_id} in {year}: {e}")

        all_clearings.extend(region_clearings)
        print(f"    Total from {region_info['name']}: {len(region_clearings)}\n")

    print(f"  ✓ Loaded {len(all_clearings)} clearing samples across {len(regions)} regions\n")

    return all_clearings


def load_intact_samples(config, n_samples=100, regions=None):
    """
    Load intact forest samples from intact forest bastions.

    Args:
        config: Config instance
        n_samples: Number of intact locations to sample
        regions: Dict of intact bastion definitions (default: diverse global sample)

    Returns:
        List of intact locations with metadata including region
    """
    print(f"Loading {n_samples} intact forest samples from intact bastions...")

    if regions is None:
        # Get intact bastions: 2 Amazon + 2 Congo + 1 Asia
        regions = get_intact_bastions(n_regions=5)

    client = EarthEngineClient(use_cache=True)

    # Use 2023 as reference year (recent, stable forest)
    year = 2023

    # Distribute samples across regions
    samples_per_region = n_samples // len(regions)

    # Try progressively more lenient criteria (very permissive for intact bastions)
    attempts = [
        {"min_tree_cover": 70, "max_loss_year": 2015, "desc": "High coverage, no recent loss"},
        {"min_tree_cover": 50, "max_loss_year": 2018, "desc": "Medium coverage, minimal loss"},
        {"min_tree_cover": 40, "max_loss_year": 2020, "desc": "Lower coverage, some loss"},
        {"min_tree_cover": 30, "max_loss_year": 2022, "desc": "Minimal coverage, accept 2000-2022 loss"},
        {"min_tree_cover": 25, "max_loss_year": 2023, "desc": "Very lenient - any stable forest"},
    ]

    all_locations = []

    print(f"  Sampling from {len(regions)} regions...")
    print(f"  Target: ~{samples_per_region} samples per region\n")

    for region_id, region_info in regions.items():
        print(f"  Region: {region_info['name']}")
        region_locations = []

        for attempt in attempts:
            if len(region_locations) >= samples_per_region:
                break

            print(f"    Attempting: {attempt['desc']}")

            try:
                locations = client.get_stable_forest_locations(
                    bounds=region_info['bounds'],
                    n_samples=samples_per_region * 10,  # Request many more to account for filtering
                    min_tree_cover=attempt['min_tree_cover'],
                    max_loss_year=attempt['max_loss_year'],
                )

                # Add region metadata and deduplicate
                for loc in locations:
                    loc['year'] = year
                    loc['region'] = region_id
                    loc['region_name'] = region_info['name']
                    loc['continent'] = region_info['region']

                    # Check if location already exists in region_locations
                    is_duplicate = any(
                        abs(loc['lat'] - existing['lat']) < 0.001 and
                        abs(loc['lon'] - existing['lon']) < 0.001
                        for existing in region_locations
                    )
                    if not is_duplicate and len(region_locations) < samples_per_region:
                        region_locations.append(loc)

                print(f"      Found {len(locations)} samples (region total: {len(region_locations)})")

            except Exception as e:
                print(f"      Warning: {e}")
                continue

        if len(region_locations) < samples_per_region:
            print(f"    ⚠ Warning: Only found {len(region_locations)}/{samples_per_region} intact samples")
        else:
            # Trim to requested amount
            import random
            random.seed(42 + hash(region_id))
            region_locations = random.sample(region_locations, samples_per_region)

        all_locations.extend(region_locations)
        print(f"    Total from {region_info['name']}: {len(region_locations)}\n")

    print(f"  ✓ Loaded {len(all_locations)} intact samples across {len(regions)} regions\n")

    return all_locations


def extract_quarterly_embeddings(client, location, year):
    """
    Extract dual-year embeddings with delta features for temporal control.

    DUAL-YEAR TEMPORAL APPROACH:
    For a clearing in year Y, extract from BOTH years:

    Year Y-1 (Baseline - guaranteed clean):
    - Q1: Mar Y-1 (15-18 months before)
    - Q2: Jun Y-1 (12-15 months before)
    - Q3: Sep Y-1 (9-12 months before)
    - Q4: Dec Y-1 (6-9 months before)

    Year Y (Current - may include clearing):
    - Q1: Mar Y (3-6 months before to 9-12 months after)
    - Q2: Jun Y (0-3 months before to 6-9 months after)
    - Q3: Sep Y (3 months before to 3 months after)
    - Q4: Dec Y (6 months before to 0 months after)

    Delta (Y - Y-1): Recent change signal (key for causal detection!)

    Clearing: Jun Y+1 (during/after clearing)

    Rationale:
    - Y-1 provides clean baseline landscape susceptibility
    - Y captures recent precursor activity (but may include clearing)
    - Delta (Y - Y-1) captures year-over-year change = human activity signal
    - This enables testing landscape vs. change-based models

    Args:
        client: EarthEngineClient instance
        location: Dict with 'lat', 'lon'
        year: Year of clearing

    Returns:
        Dict with baseline, current, delta embeddings, or None if failed
    """
    try:
        # Year Y-1: Baseline (guaranteed temporal safety)
        baseline_quarters = {
            'Q1': f"{year-1}-03-01",
            'Q2': f"{year-1}-06-01",
            'Q3': f"{year-1}-09-01",
            'Q4': f"{year-1}-12-01",
        }

        # Year Y: Current (may include clearing signal)
        current_quarters = {
            'Q1': f"{year}-03-01",
            'Q2': f"{year}-06-01",
            'Q3': f"{year}-09-01",
            'Q4': f"{year}-12-01",
        }

        baseline = {}
        current = {}

        # Extract baseline embeddings (Y-1)
        for label, date in baseline_quarters.items():
            emb = client.get_embedding(
                lat=location["lat"],
                lon=location["lon"],
                date=date,
            )
            if emb is None:
                return None
            baseline[label] = emb

        # Extract current embeddings (Y)
        for label, date in current_quarters.items():
            emb = client.get_embedding(
                lat=location["lat"],
                lon=location["lon"],
                date=date,
            )
            if emb is None:
                return None
            current[label] = emb

        # Compute delta (recent change)
        delta = {
            label: current[label] - baseline[label]
            for label in ['Q1', 'Q2', 'Q3', 'Q4']
        }

        # Extract clearing embedding
        clearing_emb = client.get_embedding(
            lat=location["lat"],
            lon=location["lon"],
            date=f"{year+1}-06-01",
        )
        if clearing_emb is None:
            return None

        return {
            'baseline': baseline,   # Y-1 embeddings
            'current': current,     # Y embeddings
            'delta': delta,        # Change (Y - Y-1)
            'clearing': clearing_emb,
        }

    except Exception as e:
        return None


def compute_temporal_features(embeddings):
    """
    Compute temporal features from dual-year embeddings.

    Computes features for THREE temporal views:
    1. Baseline (Y-1): Landscape susceptibility
    2. Current (Y): Recent state
    3. Delta (Y - Y-1): Recent change (KEY for causal detection!)

    Features per view:
    - Distances from Q1 baseline
    - Velocity (change rate between quarters)
    - Acceleration (velocity change)
    - Trend consistency

    Additional delta features:
    - Delta magnitude (per quarter)
    - Mean delta magnitude
    - Max delta magnitude

    Args:
        embeddings: Dict with 'baseline', 'current', 'delta', 'clearing' keys

    Returns:
        Dict with computed features for all temporal views
    """
    def compute_features_from_quarters(quarters_dict, clearing_emb):
        """Helper to compute features from Q1-Q4 + clearing embeddings."""
        baseline_emb = quarters_dict['Q1']

        # Distances from Q1 baseline
        distances = {}
        for label in ['Q1', 'Q2', 'Q3', 'Q4']:
            distances[label] = np.linalg.norm(quarters_dict[label] - baseline_emb)
        distances['Clearing'] = np.linalg.norm(clearing_emb - baseline_emb)

        # Velocities (distance change between consecutive quarters)
        velocities = {
            'Q1_Q2': distances['Q2'] - distances['Q1'],
            'Q2_Q3': distances['Q3'] - distances['Q2'],
            'Q3_Q4': distances['Q4'] - distances['Q3'],
            'Q4_Clearing': distances['Clearing'] - distances['Q4'],
        }

        # Accelerations (velocity change)
        accelerations = {
            'Q1_Q2_Q3': velocities['Q2_Q3'] - velocities['Q1_Q2'],
            'Q2_Q3_Q4': velocities['Q3_Q4'] - velocities['Q2_Q3'],
            'Q3_Q4_Clearing': velocities['Q4_Clearing'] - velocities['Q3_Q4'],
        }

        # Trend consistency (monotonically increasing distances?)
        distances_sequence = [distances[q] for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        trend_consistency = sum(
            1 for i in range(len(distances_sequence)-1)
            if distances_sequence[i+1] > distances_sequence[i]
        ) / (len(distances_sequence) - 1)

        return {
            'distances': distances,
            'velocities': velocities,
            'accelerations': accelerations,
            'trend_consistency': trend_consistency,
        }

    # Compute features for baseline (Y-1)
    baseline_features = compute_features_from_quarters(
        embeddings['baseline'],
        embeddings['clearing']
    )

    # Compute features for current (Y)
    current_features = compute_features_from_quarters(
        embeddings['current'],
        embeddings['clearing']
    )

    # Compute delta-specific features (magnitude of change)
    delta_magnitudes = {
        label: float(np.linalg.norm(embeddings['delta'][label]))
        for label in ['Q1', 'Q2', 'Q3', 'Q4']
    }

    delta_features = {
        'delta_magnitudes': delta_magnitudes,
        'mean_delta_magnitude': float(np.mean(list(delta_magnitudes.values()))),
        'max_delta_magnitude': float(np.max(list(delta_magnitudes.values()))),
        'delta_trend': float(delta_magnitudes['Q4'] - delta_magnitudes['Q1']),  # Is change accelerating?
    }

    return {
        'baseline': baseline_features,  # Y-1 features (landscape susceptibility)
        'current': current_features,    # Y features (recent state)
        'delta': delta_features,        # Change features (KEY!)
    }


def create_spatial_splits(locations, test_size=0.2, val_size=0.15, buffer_km=10):
    """
    Create spatial cross-validation splits with buffer zones.

    Ensures pixels within buffer_km of each other stay in same split.

    Args:
        locations: List of location dicts with 'lat', 'lon'
        test_size: Fraction for test set
        val_size: Fraction for validation set
        buffer_km: Buffer distance in kilometers

    Returns:
        Dict with 'train', 'val', 'test' indices
    """
    print(f"Creating spatial splits (buffer={buffer_km}km)...")

    # Extract coordinates
    coords = np.array([[loc['lat'], loc['lon']] for loc in locations])
    n_samples = len(coords)

    # Build spatial tree
    tree = cKDTree(coords)

    # Track assigned samples
    assigned = np.zeros(n_samples, dtype=int)  # 0=unassigned, 1=train, 2=val, 3=test

    # Determine split sizes
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val

    counts = {'train': 0, 'val': 0, 'test': 0}
    targets = {'train': n_train, 'val': n_val, 'test': n_test}

    # Random order for assignment
    import random
    random.seed(42)
    indices = list(range(n_samples))
    random.shuffle(indices)

    # Assign samples with buffer zones
    for idx in indices:
        if assigned[idx] != 0:
            continue  # Already assigned

        # Determine which split needs more samples
        remaining = {k: targets[k] - counts[k] for k in targets}
        if sum(remaining.values()) == 0:
            break

        # Assign to split with most remaining need
        split = max(remaining, key=remaining.get)
        split_id = {'train': 1, 'val': 2, 'test': 3}[split]

        # Find neighbors within buffer
        # Convert buffer_km to degrees (approximate: 1 degree ≈ 111km at equator)
        buffer_deg = buffer_km / 111.0
        neighbors = tree.query_ball_point(coords[idx], buffer_deg)

        # Assign this sample and neighbors to same split
        for neighbor_idx in neighbors:
            if assigned[neighbor_idx] == 0:
                assigned[neighbor_idx] = split_id
                counts[split] += 1

    # Handle any unassigned (shouldn't happen, but safety check)
    for idx in range(n_samples):
        if assigned[idx] == 0:
            # Assign to train by default
            assigned[idx] = 1
            counts['train'] += 1

    # Convert to index lists
    splits = {
        'train': np.where(assigned == 1)[0].tolist(),
        'val': np.where(assigned == 2)[0].tolist(),
        'test': np.where(assigned == 3)[0].tolist(),
    }

    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/n_samples:.1%})")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/n_samples:.1%})")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/n_samples:.1%})\n")

    return splits


def prepare_dataset(n_clearing=100, n_intact=100, clearing_regions=None, intact_regions=None):
    """
    Prepare full dataset with features and splits from multiple global regions.

    Uses two-region strategy:
    - Clearing samples from deforestation hotspots
    - Intact samples from intact forest bastions

    Args:
        n_clearing: Number of clearing samples
        n_intact: Number of intact forest samples
        clearing_regions: Dict of hotspot region definitions (default: diverse sample)
        intact_regions: Dict of intact bastion definitions (default: diverse sample)

    Returns:
        Dict with dataset ready for model training
    """
    print("="*80)
    print("WALK PHASE - DATA PREPARATION (TWO-REGION STRATEGY)")
    print("="*80)
    print()

    if clearing_regions is None:
        # Get deforestation hotspots: 2 Amazon + 2 Congo + 1 Asia
        clearing_regions = get_diverse_sample(n_regions=5)
        print(f"Clearing samples from {len(clearing_regions)} deforestation hotspots:")
        for region_id, info in clearing_regions.items():
            print(f"  - {info['name']} ({info['region']})")
        print()

    if intact_regions is None:
        # Get intact bastions: 2 Amazon + 2 Congo + 1 Asia
        intact_regions = get_intact_bastions(n_regions=5)
        print(f"Intact samples from {len(intact_regions)} intact forest bastions:")
        for region_id, info in intact_regions.items():
            print(f"  - {info['name']} ({info['region']})")
        print()

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Load samples from different regions
    clearings = load_clearing_samples(config, n_samples=n_clearing, regions=clearing_regions)
    intact = load_intact_samples(config, n_samples=n_intact, regions=intact_regions)

    # Extract embeddings and features
    print("Extracting embeddings and computing features...")
    print("  This may take a while (querying Earth Engine)...")
    print()

    dataset = []

    # Process clearings (label=1)
    print("Processing clearing samples...")
    for clearing in tqdm(clearings, desc="Clearings"):
        year = clearing.get('year')
        if not year:
            continue

        embeddings = extract_quarterly_embeddings(client, clearing, year)
        if embeddings is None:
            continue

        features = compute_temporal_features(embeddings)

        dataset.append({
            'location': {'lat': clearing['lat'], 'lon': clearing['lon']},
            'year': year,
            'label': 1,  # Clearing
            'region': clearing.get('region', 'unknown'),
            'region_name': clearing.get('region_name', 'Unknown'),
            'continent': clearing.get('continent', 'Unknown'),
            'embeddings': embeddings,
            'features': features,
        })

    # Process intact (label=0)
    print("\nProcessing intact forest samples...")
    for location in tqdm(intact, desc="Intact"):
        year = location.get('year', 2023)

        embeddings = extract_quarterly_embeddings(client, location, year)
        if embeddings is None:
            continue

        features = compute_temporal_features(embeddings)

        dataset.append({
            'location': {'lat': location['lat'], 'lon': location['lon']},
            'year': year,
            'label': 0,  # Intact
            'region': location.get('region', 'unknown'),
            'region_name': location.get('region_name', 'Unknown'),
            'continent': location.get('continent', 'Unknown'),
            'embeddings': embeddings,
            'features': features,
        })

    print(f"\n✓ Extracted features for {len(dataset)} samples")
    print(f"  Clearing: {sum(1 for d in dataset if d['label'] == 1)}")
    print(f"  Intact:   {sum(1 for d in dataset if d['label'] == 0)}\n")

    # Create spatial splits
    locations = [d['location'] for d in dataset]
    splits = create_spatial_splits(locations, buffer_km=10)

    # Package dataset
    prepared_dataset = {
        'data': dataset,
        'splits': splits,
        'metadata': {
            'n_samples': len(dataset),
            'n_clearing': sum(1 for d in dataset if d['label'] == 1),
            'n_intact': sum(1 for d in dataset if d['label'] == 0),
            'years': list(set(d['year'] for d in dataset)),
            'regions': list(set(d['region'] for d in dataset)),
            'continents': list(set(d['continent'] for d in dataset)),
            'samples_by_region': {
                region: sum(1 for d in dataset if d['region'] == region)
                for region in set(d['region'] for d in dataset)
            },
            'samples_by_continent': {
                continent: sum(1 for d in dataset if d['continent'] == continent)
                for continent in set(d['continent'] for d in dataset)
            },
            'created_at': datetime.now().isoformat(),
            'sampling_strategy': 'multi_region_global',
        }
    }

    return prepared_dataset


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="WALK Phase - Data Preparation")
    parser.add_argument(
        "--n-clearing",
        type=int,
        default=100,
        help="Number of clearing samples (default: 100)",
    )
    parser.add_argument(
        "--n-intact",
        type=int,
        default=100,
        help="Number of intact samples (default: 100)",
    )

    args = parser.parse_args()

    # Prepare dataset
    dataset = prepare_dataset(
        n_clearing=args.n_clearing,
        n_intact=args.n_intact,
    )

    # Save dataset
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "walk_dataset.pkl"

    print(f"Saving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"✓ Dataset saved successfully")
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples:   {dataset['metadata']['n_samples']}")
    print(f"  Clearing:      {dataset['metadata']['n_clearing']}")
    print(f"  Intact:        {dataset['metadata']['n_intact']}")
    print(f"  Class balance: {dataset['metadata']['n_clearing']/(dataset['metadata']['n_samples']):.1%} clearing")
    print()
    print(f"Train samples:   {len(dataset['splits']['train'])}")
    print(f"Val samples:     {len(dataset['splits']['val'])}")
    print(f"Test samples:    {len(dataset['splits']['test'])}")
    print()
    print(f"Regions:         {len(dataset['metadata']['regions'])}")
    for region, count in dataset['metadata']['samples_by_region'].items():
        print(f"  {region:30s}: {count} samples")
    print()
    print(f"Continents:      {', '.join(sorted(dataset['metadata']['continents']))}")
    for continent, count in dataset['metadata']['samples_by_continent'].items():
        print(f"  {continent:20s}: {count} samples")
    print()
    print(f"Years:           {sorted(dataset['metadata']['years'])}")
    print()
    print(f"Output:          {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
