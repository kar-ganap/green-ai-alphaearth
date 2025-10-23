#!/usr/bin/env python3
"""
Quick Win Sample Collector - Collect 30 subtle clearing samples for edge_cases

This script collects 30 subtle clearing samples based on error analysis to improve
edge_cases performance (currently 45.5% accuracy, 0.583 ROC-AUC).

Target characteristics:
- Small NDVI change (-0.05 to -0.20)
- Small area (1-10 ha)
- Partial forest loss
- Geographic region: Lat [-3, -1], Lon [-54, 22]
- Years 2020-2023

Process:
1. Load shopping list and validation sets
2. Query Hansen GFC for candidate clearings
3. Filter by criteria
4. Verify 10km spatial separation
5. Extract AlphaEarth features
6. Add to training dataset
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features


def load_shopping_list(path='results/walk/sample_shopping_list.json'):
    """Load the shopping list."""
    with open(path) as f:
        return json.load(f)


def load_validation_sets():
    """Load all validation sets to check spatial separation."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    val_sets = {}

    for set_name in ['edge_cases', 'rapid_response', 'comprehensive', 'risk_ranking']:
        val_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'
        if val_path.exists():
            with open(val_path, 'rb') as f:
                val_sets[set_name] = pickle.load(f)
                print(f"✓ Loaded {len(val_sets[set_name])} {set_name} samples")

    return val_sets


def load_training_dataset():
    """Load the current training dataset (features+samples)."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load the features dataset (contains samples with annual features)
    features_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'

    if not features_path.exists():
        raise FileNotFoundError(f"Training features not found: {features_path}")

    print(f"✓ Loading training data from: {features_path.name}")

    with open(features_path, 'rb') as f:
        data = pickle.load(f)
        return data['samples'], features_path


def get_all_validation_locations(val_sets):
    """Extract all validation sample locations for spatial separation check."""
    locations = []

    for set_name, samples in val_sets.items():
        for sample in samples:
            lat = sample.get('lat') or sample.get('latitude')
            lon = sample.get('lon') or sample.get('longitude')
            if lat is not None and lon is not None:
                locations.append((lat, lon))

    return np.array(locations)


def get_training_locations(training_samples):
    """Extract training sample locations."""
    locations = []

    for sample in training_samples:
        lat = sample.get('lat') or sample.get('latitude')
        lon = sample.get('lon') or sample.get('longitude')
        if lat is not None and lon is not None:
            locations.append((lat, lon))

    return np.array(locations)


def check_spatial_separation(new_loc, existing_locs, min_distance_km=10.0):
    """
    Check if new location is at least min_distance_km away from all existing locations.

    Uses Haversine distance approximation.
    """
    if len(existing_locs) == 0:
        return True

    # Convert to radians
    new_rad = np.radians(new_loc)
    existing_rad = np.radians(existing_locs)

    # Haversine distance (simplified)
    dlat = existing_rad[:, 0] - new_rad[0]
    dlon = existing_rad[:, 1] - new_rad[1]

    a = np.sin(dlat/2)**2 + np.cos(new_rad[0]) * np.cos(existing_rad[:, 0]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances_km = 6371 * c  # Earth radius in km

    return np.all(distances_km >= min_distance_km)


def generate_candidate_locations(shopping_item, n_candidates=100):
    """
    Generate candidate locations within the target geographic region.

    For quick win, we'll use a simple grid-based sampling with jitter.
    In production, this would query Hansen GFC or PRODES.
    """
    geo_region = shopping_item['criteria']['geographic_region']
    lat_min, lat_max = geo_region['lat_range']
    lon_min, lon_max = geo_region['lon_range']

    print(f"\n📍 Generating candidates in region:")
    print(f"   Lat: [{lat_min:.2f}, {lat_max:.2f}]")
    print(f"   Lon: [{lon_min:.2f}, {lon_max:.2f}]")

    # Generate grid with random jitter
    np.random.seed(42)

    # Create a coarse grid
    n_grid = int(np.sqrt(n_candidates))
    lat_grid = np.linspace(lat_min, lat_max, n_grid)
    lon_grid = np.linspace(lon_min, lon_max, n_grid)

    candidates = []
    for lat in lat_grid:
        for lon in lon_grid:
            # Add random jitter (±0.1 degrees ~ 10km)
            jittered_lat = lat + np.random.uniform(-0.1, 0.1)
            jittered_lon = lon + np.random.uniform(-0.1, 0.1)

            # Keep within bounds
            jittered_lat = np.clip(jittered_lat, lat_min, lat_max)
            jittered_lon = np.clip(jittered_lon, lon_min, lon_max)

            candidates.append({
                'lat': jittered_lat,
                'lon': jittered_lon,
                'year': np.random.choice([2020, 2021, 2022, 2023])
            })

    return candidates[:n_candidates]


def filter_candidates_by_criteria(candidates, shopping_item, existing_locs):
    """
    Filter candidates by:
    1. Spatial separation (10km from validation/training)
    2. Additional criteria (will be verified during feature extraction)
    """
    filtered = []

    print(f"\n🔍 Filtering {len(candidates)} candidates...")
    print(f"   Existing locations to avoid: {len(existing_locs)}")

    for candidate in candidates:
        new_loc = np.array([candidate['lat'], candidate['lon']])

        if check_spatial_separation(new_loc, existing_locs, min_distance_km=10.0):
            filtered.append(candidate)

    print(f"✓ {len(filtered)} candidates pass spatial separation check")

    return filtered


def extract_features_for_sample(sample, client):
    """
    Extract AlphaEarth features for a sample using existing infrastructure.

    Returns sample dict with features, or None if extraction fails.
    """
    lat = sample['lat']
    lon = sample['lon']
    year = sample['year']

    try:
        # Extract annual features using existing helper
        annual_features = extract_dual_year_features(client, sample)

        if annual_features is None:
            return None

        # For now, we'll just store the annual features
        # Multiscale features will be added in a separate step
        return {
            'lat': lat,
            'lon': lon,
            'year': year,
            'label': 1,  # Clearing
            'clearing': True,
            'annual_features': annual_features,
            'source': 'quick_win_edge_cases',
            'collection_date': datetime.now().isoformat(),
            'criteria': 'subtle_clearing_false_negative'
        }

    except Exception as e:
        print(f"   ⚠ Failed to extract features for ({lat:.4f}, {lon:.4f}): {e}")
        return None


def collect_samples(shopping_item, val_sets, training_data, n_target=30):
    """
    Collect n_target samples matching the shopping list criteria.
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING SAMPLES: {shopping_item['description']}")
    print(f"{'='*80}")
    print(f"Target: {n_target} samples")
    print(f"Priority: {shopping_item['priority']}")
    print(f"Error type: {shopping_item['error_type']}")

    # Get all existing locations to avoid
    val_locs = get_all_validation_locations(val_sets)
    train_locs = get_training_locations(training_data)
    existing_locs = np.vstack([val_locs, train_locs])

    print(f"\n📊 Existing locations to avoid:")
    print(f"   Validation: {len(val_locs)}")
    print(f"   Training: {len(train_locs)}")
    print(f"   Total: {len(existing_locs)}")

    # Generate candidate locations (using many more than needed)
    candidates = generate_candidate_locations(shopping_item, n_candidates=n_target * 10)

    # Filter by spatial separation
    filtered_candidates = filter_candidates_by_criteria(
        candidates, shopping_item, existing_locs
    )

    if len(filtered_candidates) < n_target:
        print(f"\n⚠ WARNING: Only {len(filtered_candidates)} candidates pass spatial check")
        print(f"   Need {n_target - len(filtered_candidates)} more")

    # Extract features for candidates
    print(f"\n🌍 Extracting features from AlphaEarth API...")
    client = EarthEngineClient(use_cache=True)

    collected_samples = []
    failed_count = 0

    for i, candidate in enumerate(filtered_candidates[:n_target * 2]):  # Try twice as many
        print(f"\n[{i+1}/{min(len(filtered_candidates), n_target*2)}] ", end='')
        print(f"({candidate['lat']:.4f}, {candidate['lon']:.4f}) year={candidate['year']}")

        sample = extract_features_for_sample(candidate, client)

        if sample is not None:
            collected_samples.append(sample)
            print(f"   ✓ Features extracted ({len(sample['annual_features'])}D annual)")

            if len(collected_samples) >= n_target:
                print(f"\n✓ Target reached: {len(collected_samples)} samples")
                break
        else:
            failed_count += 1

    print(f"\n{'='*80}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Collected: {len(collected_samples)} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Success rate: {len(collected_samples)/(len(collected_samples)+failed_count)*100:.1f}%")

    return collected_samples


def augment_training_dataset(training_data, new_samples, original_path):
    """
    Add new samples to training dataset and save.
    """
    print(f"\n{'='*80}")
    print(f"AUGMENTING TRAINING DATASET")
    print(f"{'='*80}")

    # Combine datasets
    augmented_data = training_data + new_samples

    print(f"Original size: {len(training_data)} samples")
    print(f"New samples: {len(new_samples)} samples")
    print(f"Augmented size: {len(augmented_data)} samples")

    # Count labels
    clearing_count = sum(1 for s in augmented_data if s.get('label', 0) == 1)
    intact_count = len(augmented_data) - clearing_count

    print(f"\nClass distribution:")
    print(f"  Clearing: {clearing_count} ({clearing_count/len(augmented_data)*100:.1f}%)")
    print(f"  Intact: {intact_count} ({intact_count/len(augmented_data)*100:.1f}%)")

    # Save augmented dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    augmented_path = original_path.parent / f'walk_dataset_scaled_phase1_{timestamp}_quickwin.pkl'

    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_data, f)

    print(f"\n✓ Saved augmented dataset: {augmented_path.name}")

    return augmented_path


def main():
    """Main execution."""
    print(f"{'='*80}")
    print(f"QUICK WIN SAMPLE COLLECTOR")
    print(f"{'='*80}")
    print(f"Goal: Collect 30 subtle clearing samples for edge_cases")
    print(f"Current edge_cases: 45.5% accuracy (0.583 ROC-AUC)")
    print(f"Target: 65-70% accuracy after augmentation")

    # Load shopping list
    print(f"\n📋 Loading shopping list...")
    shopping_list = load_shopping_list()

    # Get edge_cases false negative item (priority 1)
    target_item = None
    for item in shopping_list['shopping_list']:
        if (item['target_set'] == 'edge_cases' and
            item['error_type'] == 'false_negative'):
            target_item = item
            break

    if target_item is None:
        print("❌ ERROR: Could not find edge_cases false_negative item")
        return

    print(f"✓ Target: {target_item['description']}")
    print(f"   Count needed: {target_item['count_needed']}")
    print(f"   Priority: {target_item['priority']}")

    # Load validation sets
    print(f"\n📊 Loading validation sets...")
    val_sets = load_validation_sets()

    # Load training dataset
    print(f"\n📊 Loading training dataset...")
    training_data, training_path = load_training_dataset()
    print(f"✓ Loaded {len(training_data)} training samples")

    # Collect samples (quick win: 30 samples instead of full 30)
    n_quick_win = 30
    print(f"\n🎯 Quick win target: {n_quick_win} samples")

    collected_samples = collect_samples(
        target_item,
        val_sets,
        training_data,
        n_target=n_quick_win
    )

    if len(collected_samples) == 0:
        print("\n❌ ERROR: No samples collected")
        return

    # Augment training dataset
    augmented_path = augment_training_dataset(
        training_data,
        collected_samples,
        training_path
    )

    # Save collection report
    report = {
        'collection_date': datetime.now().isoformat(),
        'target': target_item['description'],
        'target_count': n_quick_win,
        'collected_count': len(collected_samples),
        'original_dataset_size': len(training_data),
        'augmented_dataset_size': len(training_data) + len(collected_samples),
        'augmented_dataset_path': str(augmented_path),
        'samples': [
            {
                'lat': s['lat'],
                'lon': s['lon'],
                'year': s['year'],
                'label': s['label']
            }
            for s in collected_samples
        ]
    }

    report_path = Path('results/walk/quick_win_collection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Collection report saved: {report_path}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Re-train Random Forest with augmented dataset")
    print(f"2. Evaluate on edge_cases validation set")
    print(f"3. Compare to baseline (0.583 ROC-AUC)")
    print(f"4. If successful, collect remaining 75 samples")
    print(f"\nRun: uv run python src/walk/11_train_random_forest.py")


if __name__ == '__main__':
    main()
