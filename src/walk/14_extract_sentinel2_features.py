"""
Extract Sentinel-2 Spectral Indices and Texture Features

Adds high-resolution (10m) Sentinel-2 features to complement AlphaEarth:
- Spectral indices: NDVI, NBR, NDMI
- GLCM texture features: Contrast, Correlation, Entropy, Homogeneity, ASM

This should help detect:
- Small-scale clearings (10m vs 30m resolution)
- Fire-affected areas (NBR)
- Edge forests (texture features)

Usage:
    # Extract for training set
    uv run python src/walk/14_extract_sentinel2_features.py --dataset training

    # Extract for validation sets
    uv run python src/walk/14_extract_sentinel2_features.py --dataset validation --set edge_cases
    uv run python src/walk/14_extract_sentinel2_features.py --dataset validation --set all
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import ee
import numpy as np
from tqdm import tqdm

from src.utils import get_config


def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def get_sentinel2_image(lat: float, lon: float, date: str, days_window: int = 30) -> Optional[ee.Image]:
    """
    Get cloud-free Sentinel-2 composite around a date.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)
        days_window: Days before/after to search for imagery

    Returns:
        ee.Image or None if no data available
    """
    try:
        point = ee.Geometry.Point([lon, lat])

        # Parse date
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        start_date = (dt - timedelta(days=days_window)).strftime('%Y-%m-%d')
        end_date = (dt + timedelta(days=days_window)).strftime('%Y-%m-%d')

        # Get Sentinel-2 Surface Reflectance
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        )

        # Check if any images available
        count = collection.size().getInfo()
        if count == 0:
            return None

        # Create cloud-free composite (median)
        image = collection.median()

        return image

    except Exception as e:
        return None


def calculate_spectral_indices(image: ee.Image) -> Dict[str, ee.Image]:
    """
    Calculate spectral indices from Sentinel-2 image.

    Sentinel-2 bands:
    - B4: Red (10m)
    - B8: NIR (10m)
    - B11: SWIR1 (20m)
    - B12: SWIR2 (20m)
    """
    try:
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')

        # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')

        return {
            'NDVI': ndvi,
            'NBR': nbr,
            'NDMI': ndmi
        }
    except Exception:
        return {}


def calculate_glcm_texture(image: ee.Image, band: str = 'B8', size: int = 3) -> Dict[str, ee.Image]:
    """
    Calculate GLCM texture features from NIR band.

    Args:
        image: Sentinel-2 image
        band: Band to use (default: B8 = NIR at 10m)
        size: GLCM kernel size

    Returns:
        Dict of texture feature images
    """
    try:
        # Select NIR band and scale to 8-bit integers for GLCM
        # Sentinel-2 SR values are in range 0-10000, scale to 0-255
        nir = image.select(band).multiply(0.0255).toUint8()

        # Calculate GLCM
        glcm = nir.glcmTexture(size=size)

        # Extract key texture metrics
        return {
            'contrast': glcm.select(f'{band}_contrast'),
            'correlation': glcm.select(f'{band}_corr'),
            'entropy': glcm.select(f'{band}_ent'),
            'homogeneity': glcm.select(f'{band}_idm'),
            'asm': glcm.select(f'{band}_asm')
        }
    except Exception:
        return {}


def extract_features_at_point(image_dict: Dict[str, ee.Image], lat: float, lon: float, buffer: int = 50) -> Dict[str, float]:
    """
    Extract feature statistics at a point with buffer.

    Args:
        image_dict: Dict of named images (indices and textures)
        lat: Latitude
        lon: Longitude
        buffer: Buffer radius in meters

    Returns:
        Dict of feature name -> value
    """
    try:
        point = ee.Geometry.Point([lon, lat]).buffer(buffer)

        features = {}

        for name, img in image_dict.items():
            # Get statistics within buffer
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), '', True
                ).combine(
                    ee.Reducer.minMax(), '', True
                ),
                geometry=point,
                scale=10,  # 10m resolution
                maxPixels=1e9
            ).getInfo()

            # Extract values and clean up naming
            for stat_name, value in stats.items():
                if value is not None:
                    # Remove redundant prefixes from stat_name
                    # e.g., 'NDVI_mean' -> 'mean', 'B8_contrast_mean' -> 'mean'
                    clean_stat = stat_name.split('_')[-1]
                    if clean_stat == 'mean':
                        clean_stat = 'avg'

                    feature_name = f's2_{name}_{clean_stat}'
                    features[feature_name] = float(value)

        return features

    except Exception:
        return {}


def extract_sentinel2_features(lat: float, lon: float, date: str) -> Dict[str, float]:
    """
    Extract all Sentinel-2 features for a single location.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)

    Returns:
        Dict of feature name -> value
    """
    # Get Sentinel-2 image
    s2_image = get_sentinel2_image(lat, lon, date)

    if s2_image is None:
        return {}

    # Calculate spectral indices
    indices = calculate_spectral_indices(s2_image)

    # Calculate texture features
    textures = calculate_glcm_texture(s2_image)

    # Combine all features
    all_features = {**indices, **textures}

    if len(all_features) == 0:
        return {}

    # Extract statistics at point
    features = extract_features_at_point(all_features, lat, lon, buffer=50)

    return features


def extract_for_training_set():
    """Extract Sentinel-2 features for Phase 1 training dataset."""

    print("=" * 80)
    print("EXTRACT SENTINEL-2 FEATURES FOR TRAINING SET")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load existing multiscale training data
    input_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale.pkl'

    print(f"\nLoading training dataset from: {input_path}")

    with open(input_path, 'rb') as f:
        dataset = pickle.load(f)

    samples = dataset['data']
    metadata = dataset['metadata']

    print(f"  Loaded {len(samples)} samples")
    print(f"  Clearing: {metadata['clearing_actual']}")
    print(f"  Intact: {metadata['intact_actual']}")

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    initialize_earth_engine()

    # Extract Sentinel-2 features
    print(f"\nExtracting Sentinel-2 features (10m resolution)...")
    print(f"  Features: NDVI, NBR, NDMI + GLCM textures")
    print(f"  Estimated time: ~3-5 minutes for {len(samples)} samples")

    enriched_samples = []
    failed_count = 0

    for sample in tqdm(samples, desc="Extracting"):
        lat = sample['lat']
        lon = sample['lon']
        year = sample.get('year', 2021)
        date = f'{year}-06-01'  # Mid-year

        try:
            s2_features = extract_sentinel2_features(lat, lon, date)

            if len(s2_features) > 0:
                # Merge with existing features
                if 'multiscale_features' in sample:
                    sample['multiscale_features'].update(s2_features)
                else:
                    sample['multiscale_features'] = s2_features

                enriched_samples.append(sample)
            else:
                enriched_samples.append(sample)
                failed_count += 1

        except Exception:
            enriched_samples.append(sample)
            failed_count += 1

    print(f"\n✓ Extracted Sentinel-2 features for {len(enriched_samples) - failed_count}/{len(samples)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} samples (kept without S2 features)")

    # Check feature count
    samples_with_s2 = [s for s in enriched_samples if 'multiscale_features' in s and any('s2_' in k for k in s['multiscale_features'].keys())]
    if len(samples_with_s2) > 0:
        s2_features = [k for k in samples_with_s2[0]['multiscale_features'].keys() if 's2_' in k]
        print(f"  Added {len(s2_features)} Sentinel-2 features")
        print(f"\n  Sample S2 features:")
        for feat in sorted(s2_features)[:10]:
            print(f"    - {feat}")
        if len(s2_features) > 10:
            print(f"    ... and {len(s2_features) - 10} more")

    # Save enriched dataset
    output_path = processed_dir / 'walk_dataset_scaled_phase1_sentinel2.pkl'

    output_data = {
        'data': enriched_samples,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Extract Sentinel-2 features for validation sets:")
    print("   uv run python src/walk/14_extract_sentinel2_features.py --dataset validation --set all")
    print("\n2. Train XGBoost with AlphaEarth + Sentinel-2 features:")
    print("   uv run python src/walk/15_train_with_sentinel2.py")


def extract_for_validation_set(set_name: str):
    """Extract Sentinel-2 features for a validation set."""

    print("=" * 80)
    print(f"EXTRACT SENTINEL-2 FEATURES FOR: {set_name}")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load existing multiscale validation data
    input_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'

    if not input_path.exists():
        print(f"\n✗ Input file not found: {input_path}")
        return

    print(f"\nLoading validation set from: {input_path}")

    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  Loaded {len(samples)} samples")

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    initialize_earth_engine()

    # Extract Sentinel-2 features
    print(f"\nExtracting Sentinel-2 features (10m resolution)...")

    enriched_samples = []
    failed_count = 0

    for sample in tqdm(samples, desc="Extracting"):
        # Fix missing 'year' field for intact samples
        if 'year' not in sample and sample.get('stable', False):
            sample = sample.copy()
            sample['year'] = 2021

        lat = sample['lat']
        lon = sample['lon']
        year = sample.get('year', 2021)
        date = f'{year}-06-01'

        try:
            s2_features = extract_sentinel2_features(lat, lon, date)

            if len(s2_features) > 0:
                if 'multiscale_features' in sample:
                    sample['multiscale_features'].update(s2_features)
                else:
                    sample['multiscale_features'] = s2_features

                enriched_samples.append(sample)
            else:
                enriched_samples.append(sample)
                failed_count += 1

        except Exception:
            enriched_samples.append(sample)
            failed_count += 1

    print(f"\n✓ Extracted Sentinel-2 features for {len(enriched_samples) - failed_count}/{len(samples)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} samples (kept without S2 features)")

    # Save enriched dataset
    output_path = processed_dir / f'hard_val_{set_name}_sentinel2.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['training', 'validation'],
        help='Which dataset to extract features for'
    )
    parser.add_argument(
        '--set',
        type=str,
        default='edge_cases',
        help='Validation set name (or "all" for all sets)'
    )
    args = parser.parse_args()

    if args.dataset == 'training':
        extract_for_training_set()
    else:  # validation
        if args.set == 'all':
            val_sets = ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']
            for set_name in val_sets:
                extract_for_validation_set(set_name)
                print()
        else:
            extract_for_validation_set(args.set)

    print("\n" + "=" * 80)
    print("SENTINEL-2 FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
