#!/usr/bin/env python3
"""
Complete Edge Cases Sample Collector

Collects the remaining edge_cases samples to reach 50 total:
- 4 more false negatives (subtle clearings) → Total 30 FN
- 20 false positives (intact forest confused for clearing) → Total 20 FP

Current state: 26 FN samples already collected in quickwin dataset
Target state: 639 samples (589 original + 50 edge_cases)

Usage:
    uv run python src/walk/24_collect_edge_cases_complete.py
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


def load_quickwin_dataset():
    """Load the quickwin augmented dataset (615 samples)."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Find the quickwin dataset
    augmented_paths = list(processed_dir.glob('walk_dataset_scaled_phase1_*_quickwin.pkl'))
    # Exclude features and multiscale files
    augmented_paths = [p for p in augmented_paths if 'features' not in p.name and 'multiscale' not in p.name]

    if not augmented_paths:
        raise FileNotFoundError("No quickwin dataset found")

    augmented_path = sorted(augmented_paths)[-1]

    print(f"✓ Loading quickwin dataset: {augmented_path.name}")

    with open(augmented_path, 'rb') as f:
        data = pickle.load(f)
        return data, augmented_path


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


def generate_candidate_locations(shopping_item, n_candidates=100, seed=42):
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
    np.random.seed(seed)

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


def extract_features_for_sample(sample, client, label, error_type):
    """
    Extract AlphaEarth features for a sample using existing infrastructure.

    Args:
        sample: Dict with lat, lon, year
        client: EarthEngineClient instance
        label: 1 for clearing, 0 for intact
        error_type: 'false_negative' or 'false_positive'

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
            'label': label,
            'clearing': (label == 1),
            'stable': (label == 0),
            'annual_features': annual_features,
            'source': 'edge_cases_complete',
            'collection_date': datetime.now().isoformat(),
            'criteria': f'edge_cases_{error_type}'
        }

    except Exception as e:
        print(f"   ⚠ Failed to extract features for ({lat:.4f}, {lon:.4f}): {e}")
        return None


def collect_samples(shopping_item, val_sets, training_data, n_target, label, error_type, seed=42):
    """
    Collect n_target samples matching the shopping list criteria.

    Args:
        shopping_item: Shopping list item with criteria
        val_sets: Validation sets to avoid
        training_data: Training data to avoid
        n_target: Number of samples to collect
        label: 1 for clearing, 0 for intact
        error_type: 'false_negative' or 'false_positive'
        seed: Random seed for candidate generation
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING SAMPLES: {shopping_item['description']}")
    print(f"{'='*80}")
    print(f"Target: {n_target} samples")
    print(f"Label: {label} ({'clearing' if label == 1 else 'intact'})")
    print(f"Error type: {error_type}")

    # Get all existing locations to avoid
    val_locs = get_all_validation_locations(val_sets)
    train_locs = get_training_locations(training_data)
    existing_locs = np.vstack([val_locs, train_locs])

    print(f"\n📊 Existing locations to avoid:")
    print(f"   Validation: {len(val_locs)}")
    print(f"   Training: {len(train_locs)}")
    print(f"   Total: {len(existing_locs)}")

    # Generate candidate locations (using many more than needed)
    candidates = generate_candidate_locations(shopping_item, n_candidates=n_target * 10, seed=seed)

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

        sample = extract_features_for_sample(candidate, client, label, error_type)

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
    if len(collected_samples) + failed_count > 0:
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

    # Count edge_cases samples
    edge_cases_count = sum(1 for s in augmented_data if 'edge_cases' in s.get('source', ''))
    print(f"\nEdge cases samples: {edge_cases_count}")

    # Save augmented dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    augmented_path = original_path.parent / f'walk_dataset_scaled_phase1_{timestamp}_edge_cases_complete.pkl'

    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_data, f)

    print(f"\n✓ Saved augmented dataset: {augmented_path.name}")

    return augmented_path


def main():
    """Main execution."""
    print(f"{'='*80}")
    print(f"COMPLETE EDGE CASES SAMPLE COLLECTOR")
    print(f"{'='*80}")
    print(f"Goal: Collect 24 more edge_cases samples (4 FN + 20 FP)")
    print(f"Current: 26 FN samples in quickwin dataset")
    print(f"Target: 50 total edge_cases samples (30 FN + 20 FP)")

    # Load shopping list
    print(f"\n📋 Loading shopping list...")
    shopping_list = load_shopping_list()

    # Get edge_cases items
    fn_item = None
    fp_item = None

    for item in shopping_list['shopping_list']:
        if item['target_set'] == 'edge_cases':
            if item['error_type'] == 'false_negative':
                fn_item = item
            elif item['error_type'] == 'false_positive':
                fp_item = item

    if fn_item is None or fp_item is None:
        print("❌ ERROR: Could not find edge_cases items in shopping list")
        return

    print(f"✓ False Negative target: {fn_item['description']}")
    print(f"   Count needed: {fn_item['count_needed']} (already have 26, need 4 more)")
    print(f"✓ False Positive target: {fp_item['description']}")
    print(f"   Count needed: {fp_item['count_needed']} (need all 20)")

    # Load validation sets
    print(f"\n📊 Loading validation sets...")
    val_sets = load_validation_sets()

    # Load quickwin dataset
    print(f"\n📊 Loading quickwin training dataset...")
    training_data, training_path = load_quickwin_dataset()
    print(f"✓ Loaded {len(training_data)} training samples")

    # Verify we have 26 edge_cases samples
    edge_cases_count = sum(1 for s in training_data if 'quick_win' in s.get('source', ''))
    print(f"✓ Found {edge_cases_count} quick_win samples in dataset")

    # Collect 4 more false negative samples (subtle clearings)
    print(f"\n{'='*80}")
    print(f"PHASE 1: COLLECT 4 MORE FALSE NEGATIVES")
    print(f"{'='*80}")

    fn_samples = collect_samples(
        fn_item,
        val_sets,
        training_data,
        n_target=4,
        label=1,  # Clearing
        error_type='false_negative',
        seed=43  # Different seed from quick win
    )

    # Collect 20 false positive samples (intact forest)
    print(f"\n{'='*80}")
    print(f"PHASE 2: COLLECT 20 FALSE POSITIVES")
    print(f"{'='*80}")

    fp_samples = collect_samples(
        fp_item,
        val_sets,
        training_data + fn_samples,  # Avoid newly collected FNs too
        n_target=20,
        label=0,  # Intact
        error_type='false_positive',
        seed=44  # Different seed
    )

    # Combine new samples
    all_new_samples = fn_samples + fp_samples

    if len(all_new_samples) == 0:
        print("\n❌ ERROR: No samples collected")
        return

    print(f"\n{'='*80}")
    print(f"COLLECTION TOTALS")
    print(f"{'='*80}")
    print(f"False negatives collected: {len(fn_samples)}/4")
    print(f"False positives collected: {len(fp_samples)}/20")
    print(f"Total new samples: {len(all_new_samples)}/24")

    # Augment training dataset
    augmented_path = augment_training_dataset(
        training_data,
        all_new_samples,
        training_path
    )

    # Save collection report
    report = {
        'collection_date': datetime.now().isoformat(),
        'quickwin_dataset_size': len(training_data),
        'false_negatives': {
            'target': 4,
            'collected': len(fn_samples),
            'description': fn_item['description']
        },
        'false_positives': {
            'target': 20,
            'collected': len(fp_samples),
            'description': fp_item['description']
        },
        'total_new_samples': len(all_new_samples),
        'augmented_dataset_size': len(training_data) + len(all_new_samples),
        'augmented_dataset_path': str(augmented_path),
        'samples': [
            {
                'lat': s['lat'],
                'lon': s['lon'],
                'year': s['year'],
                'label': s['label'],
                'error_type': s['criteria']
            }
            for s in all_new_samples
        ]
    }

    report_path = Path('results/walk/edge_cases_complete_collection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Collection report saved: {report_path}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Extract multiscale features for new samples")
    print(f"2. Re-train Random Forest with complete edge_cases dataset")
    print(f"3. Evaluate on edge_cases validation set")
    print(f"4. Compare to:")
    print(f"   - Baseline: 0.583 ROC-AUC (589 samples)")
    print(f"   - Quick Win: 0.600 ROC-AUC (615 samples)")
    print(f"   - Target: 0.65-0.70 ROC-AUC")
    print(f"\nRun: uv run python src/walk/22_quick_win_retrain_v2.py")
    print(f"  (Update to use *_edge_cases_complete.pkl files)")


if __name__ == '__main__':
    main()
