"""
WALK Phase - Extract Fire Features for Validation Sets

Extracts MODIS active fire detections to distinguish fire from clearing.

Fire vs Clearing temporal signatures:
- Fire: Single event → recovery (weeks-months)
- Clearing: Progressive change → persistent (stable)

Usage:
    uv run python src/walk/01f_extract_fire_features.py --set edge_cases
    uv run python src/walk/01f_extract_fire_features.py --set all
"""

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def extract_fire_features(client, lat, lon, year, months_before=6, months_after=6):
    """
    Extract MODIS fire detection features around a location.

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        year: Year of clearing/observation
        months_before: Months before clearing to check for fire (default: 6)
        months_after: Months after clearing to check for fire (default: 6)

    Returns:
        Dict with fire features
    """
    try:
        # Define time windows
        # Clearing assumed to be mid-year (June-September based on Q4)
        clearing_date = datetime(year, 9, 1)  # Q4 timepoint

        start_date = clearing_date - timedelta(days=months_before * 30)
        end_date = clearing_date + timedelta(days=months_after * 30)

        # Create point geometry with buffer (1km radius for fire detection)
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(1000)  # 1km buffer

        # MODIS Burned Area Product (Updated to Collection 6.1)
        # MCD64A1: 500m monthly burned area
        # Using Collection 6.1 (061) instead of deprecated 006
        modis_ba = ee.ImageCollection('MODIS/061/MCD64A1') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filterBounds(buffer)

        # Count of burn detections
        n_images = modis_ba.size().getInfo()

        if n_images == 0:
            return {
                'fire_detections_total': 0.0,
                'fire_detections_before': 0.0,
                'fire_detections_after': 0.0,
                'burn_area_fraction': 0.0,
                'fire_temporal_pattern': 'none'
            }

        # Get burn date band (BurnDate: day of year of burn)
        # Values: 1-366 (day of year), 0 = no burn

        # Split into before/after periods
        modis_before = modis_ba.filterDate(
            start_date.strftime('%Y-%m-%d'),
            clearing_date.strftime('%Y-%m-%d')
        )

        modis_after = modis_ba.filterDate(
            clearing_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Function to count burn pixels
        def count_burns(collection, region):
            """Count number of burned pixels in collection"""
            if collection.size().getInfo() == 0:
                return 0.0

            # Get burn date composite (any burn > 0)
            burn_composite = collection.select('BurnDate').max()

            # Count burned pixels (BurnDate > 0)
            burn_mask = burn_composite.gt(0)

            # Sample in region
            sample = burn_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=1000,  # 1km resolution
                maxPixels=1e9
            )

            burn_count = sample.get('BurnDate')
            result = burn_count.getInfo() if burn_count else 0
            # Keep as float to preserve fractional pixel counts
            return float(result) if result is not None else 0.0

        # Count burns before and after
        burns_before = count_burns(modis_before, buffer)
        burns_after = count_burns(modis_after, buffer)
        burns_total = burns_before + burns_after

        # Compute burn area fraction (rough estimate)
        # Buffer area ≈ π * 1000^2 ≈ 3.14 km²
        # Each pixel ≈ 1 km²
        buffer_pixels = 3  # Approximate number of 1km pixels in 1km radius
        burn_fraction = min(burns_total / buffer_pixels, 1.0) if buffer_pixels > 0 else 0.0

        # Determine temporal pattern
        if burns_total == 0:
            pattern = 'none'
        elif burns_before > 0 and burns_after == 0:
            pattern = 'fire_before_only'  # Fire, then recovery
        elif burns_before == 0 and burns_after > 0:
            pattern = 'fire_after_only'  # Fire after clearing
        elif burns_before > 0 and burns_after > 0:
            pattern = 'fire_before_and_after'  # Repeated fire
        else:
            pattern = 'none'

        return {
            'fire_detections_total': float(burns_total),
            'fire_detections_before': float(burns_before),
            'fire_detections_after': float(burns_after),
            'burn_area_fraction': float(burn_fraction),
            'fire_temporal_pattern': pattern
        }

    except Exception as e:
        print(f"    ✗ Fire extraction failed: {e}")
        return {
            'fire_detections_total': 0.0,
            'fire_detections_before': 0.0,
            'fire_detections_after': 0.0,
            'burn_area_fraction': 0.0,
            'fire_temporal_pattern': 'error'
        }


def enrich_samples_with_fire_features(set_name, config):
    """
    Enrich validation set samples with fire features.

    Args:
        set_name: Name of validation set (e.g., 'edge_cases')
        config: Config object

    Returns:
        Enriched samples with fire features
    """
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    # Load existing enriched dataset (with features)
    input_file = processed_dir / f"hard_val_{set_name}_features.pkl"
    output_file = processed_dir / f"hard_val_{set_name}_fire.pkl"

    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return None

    print(f"\n{'='*80}")
    print(f"EXTRACTING FIRE FEATURES FOR: {set_name}")
    print(f"{'='*80}\n")

    # Load samples
    with open(input_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    n_clearing = sum(1 for s in samples if s.get('label', 0) == 1)
    n_intact = len(samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}\n")

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Extract fire features for each sample
    enriched_samples = []
    failed_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Extracting fire features")):
        try:
            lat = sample['lat']
            lon = sample['lon']
            year = sample.get('year', 2021)

            fire_features = extract_fire_features(client, lat, lon, year)

            # Add to sample
            sample['fire_features'] = fire_features
            enriched_samples.append(sample)

        except Exception as e:
            print(f"\n  ✗ Failed on sample {i}: {e}")
            # Keep sample without fire features
            enriched_samples.append(sample)
            failed_samples.append(i)

    print(f"\n✓ Extracted fire features for {len(enriched_samples)-len(failed_samples)}/{len(samples)} samples")
    if failed_samples:
        print(f"  ✗ Failed: {len(failed_samples)} samples")
        print(f"    Indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")

    # Save enriched dataset
    with open(output_file, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to {output_file}")

    # Print fire detection summary
    fire_detected = sum(
        1 for s in enriched_samples
        if s.get('fire_features', {}).get('fire_detections_total', 0) > 0
    )

    print(f"\n📊 FIRE DETECTION SUMMARY:")
    print(f"  Samples with fire detections: {fire_detected}/{len(enriched_samples)}")

    if fire_detected > 0:
        # Break down by label
        fire_clearing = sum(
            1 for s in enriched_samples
            if s.get('label', 0) == 1 and s.get('fire_features', {}).get('fire_detections_total', 0) > 0
        )
        fire_intact = sum(
            1 for s in enriched_samples
            if s.get('label', 0) == 0 and s.get('fire_features', {}).get('fire_detections_total', 0) > 0
        )

        print(f"  Clearing samples with fire: {fire_clearing}/{n_clearing}")
        print(f"  Intact samples with fire: {fire_intact}/{n_intact}")

        # Break down by temporal pattern
        patterns = {}
        for s in enriched_samples:
            pattern = s.get('fire_features', {}).get('fire_temporal_pattern', 'none')
            patterns[pattern] = patterns.get(pattern, 0) + 1

        print(f"\n  Temporal patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"    {pattern}: {count}")

    return enriched_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set',
        type=str,
        default='edge_cases',
        choices=['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases', 'all'],
        help='Which validation set to extract fire features for'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FIRE FEATURE EXTRACTION FOR VALIDATION SETS")
    print("=" * 80)

    config = get_config()

    if args.set == 'all':
        sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']
    else:
        sets = [args.set]

    for set_name in sets:
        enrich_samples_with_fire_features(set_name, config)

    print("\n" + "=" * 80)
    print("FIRE FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext: Analyze fire patterns and train classifier:")
    print("  uv run python src/walk/06_fire_classifier.py")


if __name__ == "__main__":
    main()
