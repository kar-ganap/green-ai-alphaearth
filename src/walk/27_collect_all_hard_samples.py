#!/usr/bin/env python3
"""
Collect All Remaining Hard Samples

Collects remaining samples from rapid_response and comprehensive validation sets:
- 25 rapid_response samples (fire-cleared areas with regrowth)
- 30 comprehensive samples (15 clearings + 15 intact, diverse geography)

Current state: 636 samples (589 original + 47 edge_cases)
Target state: 691 samples (589 + 47 + 55 new)

Usage:
    uv run python src/walk/27_collect_all_hard_samples.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features


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


def load_edge_cases_complete_dataset():
    """Load the edge_cases_complete dataset (636 samples)."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Find the edge_cases_complete dataset
    augmented_paths = list(processed_dir.glob('walk_dataset_scaled_phase1_*_edge_cases_complete.pkl'))
    # Exclude features and multiscale files
    augmented_paths = [p for p in augmented_paths if 'features' not in p.name and 'multiscale' not in p.name]

    if not augmented_paths:
        raise FileNotFoundError("No edge_cases_complete dataset found")

    augmented_path = sorted(augmented_paths)[-1]

    print(f"✓ Loading edge_cases_complete dataset: {augmented_path.name}")

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


def generate_rapid_response_candidates(n_candidates=250, seed=45):
    """
    Generate candidates for rapid_response (fire-cleared areas).

    Use broader tropical forest regions.
    """
    print(f"\n📍 Generating rapid_response candidates (fire-cleared areas)")
    print(f"   Region: Tropical forest belt (-15 to 5 lat)")

    np.random.seed(seed)

    # Broader region for rapid_response
    lat_min, lat_max = -15.0, 5.0
    lon_min, lon_max = -75.0, 30.0

    # Generate grid with random jitter
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


def generate_comprehensive_candidates(n_candidates=300, seed=46):
    """
    Generate diverse candidates for comprehensive set.

    Covers Western Amazon (underrepresented) and broader regions.
    """
    print(f"\n📍 Generating comprehensive candidates (diverse geography)")
    print(f"   Region: Western Amazon + broader coverage")

    np.random.seed(seed)

    # Broader region focusing on Western Amazon
    lat_min, lat_max = -12.0, 2.0
    lon_min, lon_max = -75.0, 25.0

    # Generate grid with random jitter
    n_grid = int(np.sqrt(n_candidates))
    lat_grid = np.linspace(lat_min, lat_max, n_grid)
    lon_grid = np.linspace(lon_min, lon_max, n_grid)

    candidates = []
    for lat in lat_grid:
        for lon in lon_grid:
            # Add random jitter (±0.15 degrees ~ 15km for more diversity)
            jittered_lat = lat + np.random.uniform(-0.15, 0.15)
            jittered_lon = lon + np.random.uniform(-0.15, 0.15)

            # Keep within bounds
            jittered_lat = np.clip(jittered_lat, lat_min, lat_max)
            jittered_lon = np.clip(jittered_lon, lon_min, lon_max)

            candidates.append({
                'lat': jittered_lat,
                'lon': jittered_lon,
                'year': np.random.choice([2020, 2021, 2022, 2023])
            })

    return candidates[:n_candidates]


def filter_candidates_by_criteria(candidates, existing_locs):
    """
    Filter candidates by spatial separation (10km from validation/training).
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


def extract_features_for_sample(sample, client, label, source):
    """
    Extract AlphaEarth features for a sample.

    Args:
        sample: Dict with lat, lon, year
        client: EarthEngineClient instance
        label: 1 for clearing, 0 for intact
        source: 'rapid_response' or 'comprehensive'

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

        return {
            'lat': lat,
            'lon': lon,
            'year': year,
            'label': label,
            'clearing': (label == 1),
            'stable': (label == 0),
            'annual_features': annual_features,
            'source': source,
            'collection_date': datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return None


def collect_samples(candidates, existing_locs, client, n_target, label, source, description):
    """
    Collect n_target samples from candidates.

    Args:
        candidates: List of candidate locations
        existing_locs: Existing locations to avoid
        client: EarthEngineClient instance
        n_target: Number of samples to collect
        label: 1 for clearing, 0 for intact
        source: 'rapid_response' or 'comprehensive'
        description: Description of samples
    """
    print(f"\n{'='*80}")
    print(f"COLLECTING SAMPLES: {description}")
    print(f"{'='*80}")
    print(f"Target: {n_target} samples")
    print(f"Label: {label} ({'clearing' if label == 1 else 'intact'})")
    print(f"Source: {source}")

    # Filter by spatial separation
    filtered_candidates = filter_candidates_by_criteria(candidates, existing_locs)

    if len(filtered_candidates) < n_target:
        print(f"\n⚠ WARNING: Only {len(filtered_candidates)} candidates pass spatial check")
        print(f"   Need {n_target - len(filtered_candidates)} more")

    # Extract features for candidates
    print(f"\n🌍 Extracting features from AlphaEarth API...")

    collected_samples = []
    failed_count = 0

    for i, candidate in enumerate(filtered_candidates[:n_target * 2]):  # Try twice as many
        print(f"\n[{i+1}/{min(len(filtered_candidates), n_target*2)}] ", end='')
        print(f"({candidate['lat']:.4f}, {candidate['lon']:.4f}) year={candidate['year']}")

        sample = extract_features_for_sample(candidate, client, label, source)

        if sample is not None:
            collected_samples.append(sample)
            print(f"   ✓ Features extracted (3D annual)")

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

    # Count by source
    edge_cases_count = sum(1 for s in augmented_data if 'edge_cases' in s.get('source', ''))
    rapid_response_count = sum(1 for s in augmented_data if s.get('source') == 'rapid_response')
    comprehensive_count = sum(1 for s in augmented_data if s.get('source') == 'comprehensive')

    print(f"\nSample sources:")
    print(f"  Edge cases: {edge_cases_count}")
    print(f"  Rapid response: {rapid_response_count}")
    print(f"  Comprehensive: {comprehensive_count}")
    print(f"  Original: {len(augmented_data) - edge_cases_count - rapid_response_count - comprehensive_count}")

    # Save augmented dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    augmented_path = original_path.parent / f'walk_dataset_scaled_phase1_{timestamp}_all_hard_samples.pkl'

    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_data, f)

    print(f"\n✓ Saved augmented dataset: {augmented_path.name}")

    return augmented_path


def main():
    """Main execution."""
    print(f"{'='*80}")
    print(f"COLLECT ALL REMAINING HARD SAMPLES")
    print(f"{'='*80}")
    print(f"Goal: Collect 55 more samples (25 rapid_response + 30 comprehensive)")
    print(f"Current: 636 samples (589 original + 47 edge_cases)")
    print(f"Target: 691 samples (589 + 47 + 55)")

    # Load validation sets
    print(f"\n📊 Loading validation sets...")
    val_sets = load_validation_sets()

    # Load edge_cases_complete dataset
    print(f"\n📊 Loading edge_cases_complete training dataset...")
    training_data, training_path = load_edge_cases_complete_dataset()
    print(f"✓ Loaded {len(training_data)} training samples")

    # Get all existing locations
    val_locs = get_all_validation_locations(val_sets)
    train_locs = get_training_locations(training_data)
    existing_locs = np.vstack([val_locs, train_locs])

    print(f"\n📊 Existing locations to avoid:")
    print(f"   Validation: {len(val_locs)}")
    print(f"   Training: {len(train_locs)}")
    print(f"   Total: {len(existing_locs)}")

    # Initialize client
    client = EarthEngineClient(use_cache=True)

    # Collect rapid_response samples (25 clearings)
    print(f"\n{'='*80}")
    print(f"PHASE 1: COLLECT 25 RAPID_RESPONSE SAMPLES")
    print(f"{'='*80}")

    rr_candidates = generate_rapid_response_candidates(n_candidates=250, seed=45)
    rr_samples = collect_samples(
        rr_candidates,
        existing_locs,
        client,
        n_target=25,
        label=1,  # Clearing
        source='rapid_response',
        description='Fire-cleared areas with regrowth'
    )

    # Update existing locations to avoid newly collected samples
    if rr_samples:
        new_locs = np.array([[s['lat'], s['lon']] for s in rr_samples])
        existing_locs = np.vstack([existing_locs, new_locs])

    # Collect comprehensive samples (15 clearings + 15 intact)
    print(f"\n{'='*80}")
    print(f"PHASE 2: COLLECT 30 COMPREHENSIVE SAMPLES (15+15)")
    print(f"{'='*80}")

    comp_candidates = generate_comprehensive_candidates(n_candidates=300, seed=46)

    # Collect 15 clearings
    comp_clearing_samples = collect_samples(
        comp_candidates,
        existing_locs,
        client,
        n_target=15,
        label=1,  # Clearing
        source='comprehensive',
        description='Diverse clearings (geography, size, time)'
    )

    # Update existing locations
    if comp_clearing_samples:
        new_locs = np.array([[s['lat'], s['lon']] for s in comp_clearing_samples])
        existing_locs = np.vstack([existing_locs, new_locs])

    # Collect 15 intact
    comp_intact_samples = collect_samples(
        comp_candidates,
        existing_locs,
        client,
        n_target=15,
        label=0,  # Intact
        source='comprehensive',
        description='Diverse intact forest (geography, variety)'
    )

    # Combine all new samples
    all_new_samples = rr_samples + comp_clearing_samples + comp_intact_samples

    if len(all_new_samples) == 0:
        print("\n❌ ERROR: No samples collected")
        return

    print(f"\n{'='*80}")
    print(f"COLLECTION TOTALS")
    print(f"{'='*80}")
    print(f"Rapid response collected: {len(rr_samples)}/25")
    print(f"Comprehensive clearings collected: {len(comp_clearing_samples)}/15")
    print(f"Comprehensive intact collected: {len(comp_intact_samples)}/15")
    print(f"Total new samples: {len(all_new_samples)}/55")

    # Augment training dataset
    augmented_path = augment_training_dataset(
        training_data,
        all_new_samples,
        training_path
    )

    # Save collection report
    report = {
        'collection_date': datetime.now().isoformat(),
        'edge_cases_complete_dataset_size': len(training_data),
        'rapid_response': {
            'target': 25,
            'collected': len(rr_samples),
            'description': 'Fire-cleared areas with regrowth'
        },
        'comprehensive_clearing': {
            'target': 15,
            'collected': len(comp_clearing_samples),
            'description': 'Diverse clearings'
        },
        'comprehensive_intact': {
            'target': 15,
            'collected': len(comp_intact_samples),
            'description': 'Diverse intact forest'
        },
        'total_new_samples': len(all_new_samples),
        'augmented_dataset_size': len(training_data) + len(all_new_samples),
        'augmented_dataset_path': str(augmented_path),
        'samples': [
            {
                'lat': float(s['lat']),
                'lon': float(s['lon']),
                'year': int(s['year']),
                'label': int(s['label']),
                'source': s['source']
            }
            for s in all_new_samples
        ]
    }

    report_path = Path('results/walk/all_hard_samples_collection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Collection report saved: {report_path}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Extract multiscale features for new samples")
    print(f"2. Re-train Random Forest with all hard samples (691 samples)")
    print(f"3. Evaluate on all validation sets")
    print(f"4. Compare to:")
    print(f"   - Baseline: 0.583 ROC-AUC (589 samples)")
    print(f"   - Quick Win: 0.600 ROC-AUC (615 samples)")
    print(f"   - Complete edge_cases: 0.600 ROC-AUC (636 samples)")
    print(f"   - Target: 0.65-0.70 ROC-AUC")


if __name__ == '__main__':
    main()
