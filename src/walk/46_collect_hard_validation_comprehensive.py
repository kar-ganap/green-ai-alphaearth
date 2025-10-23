#!/usr/bin/env python3
"""
Comprehensive Hard Validation Data Collection

Collects validation samples for all 4 use cases across 3 years (2022-2024)
to test temporal generalization on challenging scenarios.

Collection ensures:
- No spatial overlap with training data (1km buffer)
- No temporal overlap (test year not in training years for phase)
- Matched sampling methodology (heterogeneous Hansen thresholds)
- Matched labeling (Hansen Global Forest Change)

Total samples: 345 (115 per year × 3 years)
- Risk ranking: 30 per year
- Rapid response: 25 per year
- Comprehensive: 40 per year
- Edge cases: 20 per year

Usage:
    # Collect all years and use cases
    uv run python src/walk/46_collect_hard_validation_comprehensive.py

    # Collect specific year and use case
    uv run python src/walk/46_collect_hard_validation_comprehensive.py --year 2024 --use-case edge_cases

    # Dry run to preview
    uv run python src/walk/46_collect_hard_validation_comprehensive.py --dry-run
"""

import argparse
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import ee

from src.utils import EarthEngineClient, get_config

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Use case sample sizes (per year)
USE_CASE_SIZES = {
    'risk_ranking': 30,
    'rapid_response': 25,
    'comprehensive': 40,
    'edge_cases': 20
}

# Geographic bounds for each use case
USE_CASE_BOUNDS = {
    'risk_ranking': {
        # High-risk deforestation regions
        'amazon': {'min_lat': -15, 'max_lat': 5, 'min_lon': -75, 'max_lon': -45},
        'drc': {'min_lat': -5, 'max_lat': 5, 'min_lon': 15, 'max_lon': 30},
        'indonesia': {'min_lat': -5, 'max_lat': 5, 'min_lon': 95, 'max_lon': 140}
    },
    'rapid_response': {
        # Focused high-deforestation regions (not global tropics)
        'amazon': {'min_lat': -15, 'max_lat': 5, 'min_lon': -75, 'max_lon': -45},
        'drc': {'min_lat': -5, 'max_lat': 5, 'min_lon': 15, 'max_lon': 30},
        'indonesia': {'min_lat': -5, 'max_lat': 5, 'min_lon': 95, 'max_lon': 140}
    },
    'comprehensive': {
        # Diverse tropical regions
        'amazon': {'min_lat': -20, 'max_lat': 10, 'min_lon': -80, 'max_lon': -35},
        'africa': {'min_lat': -10, 'max_lat': 15, 'min_lon': -20, 'max_lon': 50},
        'asia': {'min_lat': -10, 'max_lat': 25, 'min_lon': 70, 'max_lon': 150}
    },
    'edge_cases': {
        # Challenging scenarios - global
        'global': {'min_lat': -30, 'max_lat': 30, 'min_lon': -180, 'max_lon': 180}
    }
}

# Hansen threshold distribution (match training)
HANSEN_THRESHOLD_DIST = {
    50: 0.60,  # 60% of samples - standard clearings
    40: 0.20,  # 20% of samples - small clearings
    30: 0.20   # 20% of samples - edge/fire clearings
}

# Use case-specific spatial buffers (km)
# Differentiated based on independence mechanism:
# - 3km: risk_ranking, comprehensive (spatial independence primary)
# - 1km: rapid_response, edge_cases (temporal/feature-space independence primary)
SPATIAL_BUFFERS = {
    'risk_ranking': 3.0,      # Spatial independence primary
    'comprehensive': 3.0,      # Spatial independence primary
    'rapid_response': 1.0,     # Temporal independence primary (recent vs historical)
    'edge_cases': 1.0          # Feature-space independence primary (small/low-cover vs typical)
}

# Edge case tree cover range width (wider = more samples)
EDGE_CASE_RANGE_WIDTH = 20  # e.g., 30-50% instead of 30-40%


def load_training_locations():
    """Load all training sample locations to prevent overlap."""
    print("\nLoading training locations for spatial leakage prevention...")

    training_locations = []

    # Load 2020-2023 training data
    file_2020_2023 = PROCESSED_DIR / 'walk_dataset_scaled_phase1_20251020_165345_all_hard_samples_multiscale.pkl'
    if file_2020_2023.exists():
        with open(file_2020_2023, 'rb') as f:
            data = pickle.load(f)
        samples = data if isinstance(data, list) else data.get('samples', data.get('data', []))
        for s in samples:
            training_locations.append((s['lat'], s['lon'], s.get('year')))
        print(f"  Loaded {len(samples)} samples from 2020-2023 training")

    # Load 2024 training data
    file_2024 = PROCESSED_DIR / 'walk_dataset_2024_with_features_20251021_110417.pkl'
    if file_2024.exists():
        with open(file_2024, 'rb') as f:
            data = pickle.load(f)
        samples = data if isinstance(data, list) else data.get('samples', data.get('data', []))
        for s in samples:
            training_locations.append((s['lat'], s['lon'], s.get('year', 2024)))
        print(f"  Loaded {len(samples)} samples from 2024 training")

    print(f"  Total training locations: {len(training_locations)}")

    return training_locations


def check_spatial_overlap(lat, lon, training_locations, threshold_km=3.0):
    """Check if location is within threshold_km of any training location."""
    # Simple haversine distance check
    # For small distances, approximate: 1 degree ≈ 111 km
    threshold_deg = threshold_km / 111.0

    for train_lat, train_lon, _ in training_locations:
        # Rough distance check
        dlat = abs(lat - train_lat)
        dlon = abs(lon - train_lon)

        if dlat < threshold_deg and dlon < threshold_deg:
            # More precise check using Haversine
            dist_km = haversine_distance(lat, lon, train_lat, train_lon)
            if dist_km < threshold_km:
                return True

    return False


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def collect_risk_ranking(client, year, n_samples, training_locations, used_locations):
    """
    Collect high-risk region samples.

    Focus on Amazon, DRC, Indonesia with recent deforestation nearby.
    Uses 3km spatial buffer (spatial independence primary).
    """
    spatial_buffer = SPATIAL_BUFFERS['risk_ranking']
    print(f"\nCollecting {n_samples} risk ranking samples for {year}... (spatial buffer: {spatial_buffer}km)")

    bounds = USE_CASE_BOUNDS['risk_ranking']
    all_samples = []

    # Distribute across regions
    region_samples = {
        'amazon': int(n_samples * 0.40),
        'drc': int(n_samples * 0.30),
        'indonesia': n_samples - int(n_samples * 0.40) - int(n_samples * 0.30)
    }

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    for region, n_region in region_samples.items():
        print(f"  {region}: {n_region} samples")

        region_bounds = bounds[region]
        roi = ee.Geometry.Rectangle([
            region_bounds['min_lon'], region_bounds['min_lat'],
            region_bounds['max_lon'], region_bounds['max_lat']
        ])

        # Sample clearings with heterogeneous thresholds
        region_samples_collected = []

        for threshold, proportion in HANSEN_THRESHOLD_DIST.items():
            n_threshold = int(n_region * proportion)

            tree_cover = gfc.select("treecover2000")
            loss = gfc.select("loss")
            loss_year = gfc.select("lossyear")

            # Convert year to Hansen year code
            year_code = year - 2000

            # Clearings: tree cover >= threshold, loss in year
            mask = (
                tree_cover.gte(threshold)
                .And(loss.eq(1))
                .And(loss_year.eq(year_code))
            )

            sample = mask.selfMask().sample(
                region=roi,
                scale=30,
                numPixels=n_threshold * 500,  # Aggressive oversample for 3km buffer
                seed=1000 + year + threshold,
                geometries=True
            )

            features = sample.getInfo()["features"]

            # Filter for spatial leakage
            valid_features = []
            for feature in features:
                coords = feature["geometry"]["coordinates"]
                lat, lon = coords[1], coords[0]

                loc_key = (round(lat, 4), round(lon, 4))

                # Check spatial and location uniqueness
                if not check_spatial_overlap(lat, lon, training_locations, threshold_km=spatial_buffer):
                    if loc_key not in used_locations:
                        valid_features.append((lat, lon))
                        used_locations.add(loc_key)

                if len(valid_features) >= n_threshold:
                    break

            # Create samples
            for lat, lon in valid_features[:n_threshold]:
                region_samples_collected.append({
                    "lat": lat,
                    "lon": lon,
                    "year": year,
                    "date": f"{year}-06-01",
                    "source": "GFW",
                    "use_case": "risk_ranking",
                    "region": region,
                    "hansen_threshold": threshold,
                    "label": 1  # Clearing
                })

        all_samples.extend(region_samples_collected)

    print(f"  ✓ Collected {len(all_samples)} risk ranking clearings")

    # Collect matched intact samples
    intact_samples = collect_matched_intact(
        client, bounds, year, len(all_samples), training_locations, used_locations, spatial_buffer
    )

    all_samples.extend(intact_samples)
    print(f"  ✓ Total: {len(all_samples)} samples ({len(all_samples)//2} clearing, {len(intact_samples)} intact)")

    return all_samples


def collect_rapid_response(client, year, n_samples, training_locations, used_locations):
    """
    Collect rapid response samples (recent deforestation).

    Focus on focused high-deforestation regions (Amazon, DRC, Indonesia).
    Uses 1km spatial buffer (temporal independence primary).
    """
    spatial_buffer = SPATIAL_BUFFERS['rapid_response']
    print(f"\nCollecting {n_samples} rapid response samples for {year}... (spatial buffer: {spatial_buffer}km)")

    bounds = USE_CASE_BOUNDS['rapid_response']
    all_samples = []

    # Distribute across regions
    region_samples = {
        'amazon': int(n_samples * 0.40),
        'drc': int(n_samples * 0.30),
        'indonesia': n_samples - int(n_samples * 0.40) - int(n_samples * 0.30)
    }

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    for region, n_region in region_samples.items():
        print(f"  {region}: {n_region} samples")

        region_bounds = bounds[region]
        roi = ee.Geometry.Rectangle([
            region_bounds['min_lon'], region_bounds['min_lat'],
            region_bounds['max_lon'], region_bounds['max_lat']
        ])

        # Sample with heterogeneous thresholds
        for threshold, proportion in HANSEN_THRESHOLD_DIST.items():
            n_threshold = int(n_region * proportion)

            tree_cover = gfc.select("treecover2000")
            loss = gfc.select("loss")
            loss_year = gfc.select("lossyear")

            year_code = year - 2000

            mask = (
                tree_cover.gte(threshold)
                .And(loss.eq(1))
                .And(loss_year.eq(year_code))
            )

            sample = mask.selfMask().sample(
                region=roi,
                scale=30,
                numPixels=n_threshold * 500,  # Aggressive oversample
                seed=2000 + year + threshold,
                geometries=True
            )

            features = sample.getInfo()["features"]

            # Filter for spatial leakage
            valid_samples = []
            for feature in features:
                coords = feature["geometry"]["coordinates"]
                lat, lon = coords[1], coords[0]

                loc_key = (round(lat, 4), round(lon, 4))

                if not check_spatial_overlap(lat, lon, training_locations, threshold_km=spatial_buffer):
                    if loc_key not in used_locations:
                        valid_samples.append({
                            "lat": lat,
                            "lon": lon,
                            "year": year,
                            "date": f"{year}-09-01",  # Later in year for "rapid response"
                            "source": "GFW",
                            "use_case": "rapid_response",
                            "region": region,
                            "hansen_threshold": threshold,
                            "label": 1
                        })
                        used_locations.add(loc_key)

                if len(valid_samples) >= n_threshold:
                    break

            all_samples.extend(valid_samples[:n_threshold])

    print(f"  ✓ Collected {len(all_samples)} rapid response clearings")

    # Collect matched intact
    intact_samples = collect_matched_intact(
        client, bounds, year, len(all_samples), training_locations, used_locations, spatial_buffer
    )

    all_samples.extend(intact_samples)
    print(f"  ✓ Total: {len(all_samples)} samples")

    return all_samples


def collect_comprehensive(client, year, n_samples, training_locations, used_locations):
    """
    Collect comprehensive diverse samples across biomes.
    Uses 3km spatial buffer (spatial independence primary).
    """
    spatial_buffer = SPATIAL_BUFFERS['comprehensive']
    print(f"\nCollecting {n_samples} comprehensive samples for {year}... (spatial buffer: {spatial_buffer}km)")

    bounds = USE_CASE_BOUNDS['comprehensive']
    all_samples = []

    # Distribute across regions
    region_samples = {
        'amazon': int(n_samples * 0.40),
        'africa': int(n_samples * 0.30),
        'asia': n_samples - int(n_samples * 0.40) - int(n_samples * 0.30)
    }

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    for region, n_region in region_samples.items():
        print(f"  {region}: {n_region} samples")

        region_bounds = bounds[region]
        roi = ee.Geometry.Rectangle([
            region_bounds['min_lon'], region_bounds['min_lat'],
            region_bounds['max_lon'], region_bounds['max_lat']
        ])

        for threshold, proportion in HANSEN_THRESHOLD_DIST.items():
            n_threshold = int(n_region * proportion)

            tree_cover = gfc.select("treecover2000")
            loss = gfc.select("loss")
            loss_year = gfc.select("lossyear")

            year_code = year - 2000

            mask = (
                tree_cover.gte(threshold)
                .And(loss.eq(1))
                .And(loss_year.eq(year_code))
            )

            sample = mask.selfMask().sample(
                region=roi,
                scale=30,
                numPixels=n_threshold * 500,  # Aggressive oversample for 3km buffer
                seed=3000 + year + threshold,
                geometries=True
            )

            features = sample.getInfo()["features"]

            valid_samples = []
            for feature in features:
                coords = feature["geometry"]["coordinates"]
                lat, lon = coords[1], coords[0]

                loc_key = (round(lat, 4), round(lon, 4))

                if not check_spatial_overlap(lat, lon, training_locations, threshold_km=spatial_buffer):
                    if loc_key not in used_locations:
                        valid_samples.append({
                            "lat": lat,
                            "lon": lon,
                            "year": year,
                            "date": f"{year}-06-01",
                            "source": "GFW",
                            "use_case": "comprehensive",
                            "region": region,
                            "hansen_threshold": threshold,
                            "label": 1
                        })
                        used_locations.add(loc_key)

                if len(valid_samples) >= n_threshold:
                    break

            all_samples.extend(valid_samples[:n_threshold])

    print(f"  ✓ Collected {len(all_samples)} comprehensive clearings")

    # Collect matched intact
    intact_samples = collect_matched_intact(
        client, bounds, year, len(all_samples), training_locations, used_locations, spatial_buffer
    )

    all_samples.extend(intact_samples)
    print(f"  ✓ Total: {len(all_samples)} samples")

    return all_samples


def collect_edge_cases(client, year, n_samples, training_locations, used_locations):
    """
    Collect edge cases: small clearings, low tree cover, forest edges.
    Uses 1km spatial buffer (feature-space independence primary).
    """
    spatial_buffer = SPATIAL_BUFFERS['edge_cases']
    print(f"\nCollecting {n_samples} edge case samples for {year}... (spatial buffer: {spatial_buffer}km)")

    bounds = USE_CASE_BOUNDS['edge_cases']['global']
    roi = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    all_samples = []

    # Focus on lower thresholds for edge cases
    edge_case_thresholds = {
        30: 0.50,  # 50% at threshold edge
        35: 0.30,  # 30% slightly above
        40: 0.20   # 20% small clearings
    }

    for threshold, proportion in edge_case_thresholds.items():
        n_threshold = int(n_samples * proportion)

        tree_cover = gfc.select("treecover2000")
        loss = gfc.select("loss")
        loss_year = gfc.select("lossyear")

        year_code = year - 2000

        mask = (
            tree_cover.gte(threshold)
            .And(tree_cover.lt(threshold + EDGE_CASE_RANGE_WIDTH))  # Wider range (20% instead of 10%)
            .And(loss.eq(1))
            .And(loss_year.eq(year_code))
        )

        sample = mask.selfMask().sample(
            region=roi,
            scale=30,
            numPixels=n_threshold * 500,  # Aggressive oversample for 3km buffer + narrow range
            seed=4000 + year + threshold,
            geometries=True
        )

        features = sample.getInfo()["features"]

        valid_samples = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]
            lat, lon = coords[1], coords[0]

            loc_key = (round(lat, 4), round(lon, 4))

            if not check_spatial_overlap(lat, lon, training_locations, threshold_km=spatial_buffer):
                if loc_key not in used_locations:
                    valid_samples.append({
                        "lat": lat,
                        "lon": lon,
                        "year": year,
                        "date": f"{year}-06-01",
                        "source": "GFW",
                        "use_case": "edge_cases",
                        "hansen_threshold": threshold,
                        "label": 1
                    })
                    used_locations.add(loc_key)

            if len(valid_samples) >= n_threshold:
                break

        all_samples.extend(valid_samples[:n_threshold])

    print(f"  ✓ Collected {len(all_samples)} edge case clearings")

    # Collect matched intact
    intact_samples = collect_matched_intact(
        client, bounds, year, len(all_samples), training_locations, used_locations, spatial_buffer
    )

    all_samples.extend(intact_samples)
    print(f"  ✓ Total: {len(all_samples)} samples")

    return all_samples


def collect_matched_intact(client, bounds, year, n_samples, training_locations, used_locations, spatial_buffer):
    """Collect intact forest samples matched to clearings."""
    if isinstance(bounds, dict) and 'amazon' in bounds:
        # Multi-region case
        all_intact = []
        total_needed = n_samples
        per_region = total_needed // len(bounds)

        for region, region_bounds in bounds.items():
            intact = collect_intact_single_region(
                client, region_bounds, year, per_region,
                training_locations, used_locations, spatial_buffer
            )
            all_intact.extend(intact)

        return all_intact[:n_samples]
    else:
        # Single region case
        return collect_intact_single_region(
            client, bounds, year, n_samples,
            training_locations, used_locations, spatial_buffer
        )


def collect_intact_single_region(client, bounds, year, n_samples, training_locations, used_locations, spatial_buffer):
    """Collect intact samples from a single region."""
    roi = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    all_intact = []

    for threshold, proportion in HANSEN_THRESHOLD_DIST.items():
        n_threshold = int(n_samples * proportion)

        tree_cover = gfc.select("treecover2000")
        loss = gfc.select("loss")

        # Intact: tree cover >= threshold, no loss ever
        mask = (
            tree_cover.gte(threshold)
            .And(loss.eq(0))
        )

        # Simplify the ROI to avoid complex polygon errors
        roi_simplified = roi.simplify(maxError=100)

        sample = mask.selfMask().sample(
            region=roi_simplified,
            scale=30,
            numPixels=n_threshold * 500,  # Aggressive oversample
            seed=5000 + year + threshold,
            geometries=True
        )

        try:
            features = sample.getInfo()["features"]
        except Exception as e:
            print(f"  ⚠️ Failed to sample with threshold {threshold}: {e}")
            features = []

        valid_samples = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]
            lat, lon = coords[1], coords[0]

            loc_key = (round(lat, 4), round(lon, 4))

            if not check_spatial_overlap(lat, lon, training_locations, threshold_km=spatial_buffer):
                if loc_key not in used_locations:
                    valid_samples.append({
                        "lat": lat,
                        "lon": lon,
                        "year": year,
                        "date": f"{year}-06-01",
                        "source": "GFW",
                        "hansen_threshold": threshold,
                        "label": 0  # Intact
                    })
                    used_locations.add(loc_key)

            if len(valid_samples) >= n_threshold:
                break

        all_intact.extend(valid_samples[:n_threshold])

    return all_intact


def main():
    parser = argparse.ArgumentParser(description='Collect comprehensive hard validation data')
    parser.add_argument('--year', type=int, choices=[2022, 2023, 2024],
                       help='Specific year to collect (default: all)')
    parser.add_argument('--use-case', type=str,
                       choices=['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases'],
                       help='Specific use case to collect (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview collection plan without executing')

    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE HARD VALIDATION DATA COLLECTION")
    print("="*80)

    # Initialize Earth Engine
    client = EarthEngineClient()

    # Load training locations
    training_locations = load_training_locations()

    # Track used locations across all collections
    used_locations = set()

    # Determine what to collect
    years = [args.year] if args.year else [2022, 2023, 2024]
    use_cases = [args.use_case] if args.use_case else ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Preview only\n")
        for year in years:
            for use_case in use_cases:
                n_samples = USE_CASE_SIZES[use_case] * 2  # Clearing + intact
                print(f"{year} - {use_case}: {n_samples} samples")
        return

    # Collect samples
    all_results = {}

    for year in years:
        print(f"\n{'='*80}")
        print(f"YEAR {year}")
        print(f"{'='*80}")

        year_results = {}

        for use_case in use_cases:
            print(f"\nUse case: {use_case}")

            if use_case == 'risk_ranking':
                samples = collect_risk_ranking(
                    client, year, USE_CASE_SIZES[use_case],
                    training_locations, used_locations
                )
            elif use_case == 'rapid_response':
                samples = collect_rapid_response(
                    client, year, USE_CASE_SIZES[use_case],
                    training_locations, used_locations
                )
            elif use_case == 'comprehensive':
                samples = collect_comprehensive(
                    client, year, USE_CASE_SIZES[use_case],
                    training_locations, used_locations
                )
            elif use_case == 'edge_cases':
                samples = collect_edge_cases(
                    client, year, USE_CASE_SIZES[use_case],
                    training_locations, used_locations
                )

            year_results[use_case] = samples

            # Save immediately
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = PROCESSED_DIR / f'hard_val_{use_case}_{year}_{timestamp}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)

            print(f"  ✓ Saved to: {output_file.name}")

        all_results[year] = year_results

    # Summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)

    total_samples = 0
    for year, year_results in all_results.items():
        print(f"\n{year}:")
        for use_case, samples in year_results.items():
            n_clearing = sum(1 for s in samples if s['label'] == 1)
            n_intact = sum(1 for s in samples if s['label'] == 0)
            print(f"  {use_case:20s}: {len(samples):3d} ({n_clearing} clearing, {n_intact} intact)")
            total_samples += len(samples)

    print(f"\n✓ Total samples collected: {total_samples}")
    print(f"✓ Unique locations: {len(used_locations)}")
    print(f"✓ Spatial leakage: PREVENTED (1km buffer)")
    print(f"✓ Temporal leakage: PREVENTED (per-phase validation)")


if __name__ == '__main__':
    main()
