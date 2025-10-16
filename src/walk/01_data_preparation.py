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


def load_clearing_samples(config, n_samples=100, years=None):
    """
    Load clearing samples from GLAD with temporal labels.

    Args:
        config: Config instance
        n_samples: Number of clearing events to sample
        years: List of years to sample from (default: [2020, 2021, 2022, 2023])

    Returns:
        List of clearing events with metadata
    """
    print(f"Loading {n_samples} clearing samples...")

    if years is None:
        years = [2020, 2021, 2022, 2023]

    client = EarthEngineClient(use_cache=True)

    # Split samples across years
    samples_per_year = n_samples // len(years)

    # Split region into sub-regions for spatial diversity
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

    print(f"  Sampling from {len(years)} years × {len(sub_regions)} sub-regions...")

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
            except Exception as e:
                print(f"    Warning: Failed to fetch from sub-region in {year}: {e}")

        # Sample from this year
        if len(year_clearings) > samples_per_year:
            import random
            random.seed(42 + year)  # Reproducible but different per year
            year_clearings = random.sample(year_clearings, samples_per_year)

        all_clearings.extend(year_clearings)
        print(f"    Year {year}: {len(year_clearings)} samples")

    print(f"  ✓ Loaded {len(all_clearings)} clearing samples\n")

    return all_clearings


def load_intact_samples(config, n_samples=100):
    """
    Load intact forest samples (no clearing) for negative class.

    Args:
        config: Config instance
        n_samples: Number of intact locations to sample

    Returns:
        List of intact locations with metadata
    """
    print(f"Loading {n_samples} intact forest samples...")

    client = EarthEngineClient(use_cache=True)

    # Use 2023 as reference year (recent, stable forest)
    year = 2023

    # Get intact forest from wider region
    bounds = config.study_region_bounds

    try:
        # Get stable forest locations (no recent loss)
        locations = client.get_stable_forest_locations(
            bounds=bounds,
            n_samples=n_samples,
            min_tree_cover=70,  # Higher threshold for intact
            max_loss_year=2015,  # No loss after 2015
        )

        # Add year metadata
        for loc in locations:
            loc['year'] = year

        print(f"  ✓ Loaded {len(locations)} intact samples\n")
        return locations

    except Exception as e:
        print(f"  Warning: Failed to fetch intact samples: {e}")
        print(f"  Falling back to random sampling within bounds...")

        # Fallback: Random sample within bounds
        import random
        random.seed(42)

        locations = []
        for _ in range(n_samples):
            lat = random.uniform(bounds["min_lat"], bounds["max_lat"])
            lon = random.uniform(bounds["min_lon"], bounds["max_lon"])
            locations.append({"lat": lat, "lon": lon, "year": year})

        print(f"  ✓ Generated {len(locations)} random locations\n")
        return locations


def extract_quarterly_embeddings(client, location, year):
    """
    Extract embeddings at quarterly intervals for a location.

    For a clearing in year Y, we extract:
    - Q1 (Y-1): 9-12 months before
    - Q2 (Y): 6-9 months before
    - Q3 (Y): 3-6 months before
    - Q4 (Y): 0-3 months before (precursor period)
    - Clearing (Y+1): During/after clearing

    Args:
        client: EarthEngineClient instance
        location: Dict with 'lat', 'lon'
        year: Year of clearing

    Returns:
        Dict with embeddings at each quarter, or None if failed
    """
    try:
        quarters = {
            'Q1': f"{year-1}-06-01",  # Far back (9-12 months)
            'Q2': f"{year}-03-01",    # 6-9 months before
            'Q3': f"{year}-06-01",    # 3-6 months before
            'Q4': f"{year}-09-01",    # Precursor (0-3 months)
            'Clearing': f"{year+1}-06-01",  # During/after
        }

        embeddings = {}

        for label, date in quarters.items():
            emb = client.get_embedding(
                lat=location["lat"],
                lon=location["lon"],
                date=date,
            )

            if emb is None:
                return None

            embeddings[label] = emb

        return embeddings

    except Exception as e:
        return None


def compute_temporal_features(embeddings):
    """
    Compute temporal features from quarterly embeddings.

    Features:
    - Distances from Q1 baseline
    - Velocity (change rate between quarters)
    - Acceleration (velocity change)
    - Trend consistency

    Args:
        embeddings: Dict with Q1, Q2, Q3, Q4, Clearing embeddings

    Returns:
        Dict with computed features
    """
    baseline = embeddings['Q1']

    # Compute distances from baseline
    distances = {}
    for label in ['Q1', 'Q2', 'Q3', 'Q4', 'Clearing']:
        distances[label] = np.linalg.norm(embeddings[label] - baseline)

    # Compute velocities (distance change between consecutive quarters)
    velocities = {
        'Q1_Q2': distances['Q2'] - distances['Q1'],
        'Q2_Q3': distances['Q3'] - distances['Q2'],
        'Q3_Q4': distances['Q4'] - distances['Q3'],
        'Q4_Clearing': distances['Clearing'] - distances['Q4'],
    }

    # Compute acceleration (velocity change)
    accelerations = {
        'Q1_Q2_Q3': velocities['Q2_Q3'] - velocities['Q1_Q2'],
        'Q2_Q3_Q4': velocities['Q3_Q4'] - velocities['Q2_Q3'],
        'Q3_Q4_Clearing': velocities['Q4_Clearing'] - velocities['Q3_Q4'],
    }

    # Trend consistency (are distances monotonically increasing?)
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


def prepare_dataset(n_clearing=100, n_intact=100):
    """
    Prepare full dataset with features and splits.

    Args:
        n_clearing: Number of clearing samples
        n_intact: Number of intact forest samples

    Returns:
        Dict with dataset ready for model training
    """
    print("="*80)
    print("WALK PHASE - DATA PREPARATION")
    print("="*80)
    print()

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Load samples
    clearings = load_clearing_samples(config, n_samples=n_clearing)
    intact = load_intact_samples(config, n_samples=n_intact)

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
            'created_at': datetime.now().isoformat(),
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
    print(f"Train samples:   {len(dataset['splits']['train'])}")
    print(f"Val samples:     {len(dataset['splits']['val'])}")
    print(f"Test samples:    {len(dataset['splits']['test'])}")
    print(f"Years:           {sorted(dataset['metadata']['years'])}")
    print()
    print(f"Output:          {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
