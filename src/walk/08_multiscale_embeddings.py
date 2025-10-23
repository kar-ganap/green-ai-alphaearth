"""
WALK Phase - Multi-Scale Embedding Extraction

Addresses small-scale clearing detection (100% miss rate < 1 ha) by extracting
embeddings at multiple spatial resolutions.

Multi-scale strategy:
- Fine (10m): Sentinel-2 spectral features for small clearing detection
- Medium (30m): AlphaEarth embeddings (current baseline)
- Coarse (100m): Aggregated landscape context

Usage:
    uv run python src/walk/08_multiscale_embeddings.py --set rapid_response
    uv run python src/walk/08_multiscale_embeddings.py --set all
"""

import argparse
import pickle
from pathlib import Path

import ee
import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def extract_sentinel2_features(client, lat, lon, date, scale=10):
    """
    Extract Sentinel-2 spectral features at 10m resolution.

    Sentinel-2 provides higher spatial resolution than Landsat (10m vs 30m),
    crucial for detecting small clearings.

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)
        scale: Resolution in meters (default: 10m)

    Returns:
        Dict with Sentinel-2 features
    """
    try:
        point = ee.Geometry.Point([lon, lat])

        # Get date window (±15 days for cloud-free composite)
        target_date = ee.Date(date)
        start_date = target_date.advance(-15, 'day')
        end_date = target_date.advance(15, 'day')

        # Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        # Check if images available
        n_images = s2.size().getInfo()
        if n_images == 0:
            return None

        # Cloud masking function
        def mask_s2_clouds(image):
            qa = image.select('QA60')
            # Bits 10 and 11 are clouds and cirrus
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0)
            )
            return image.updateMask(mask).divide(10000)

        # Apply cloud mask and get median composite
        s2_clean = s2.map(mask_s2_clouds).median()

        # Select bands (10m and 20m resolution)
        # 10m bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
        # 20m bands: B5 (Red Edge 1), B6 (Red Edge 2), B7 (Red Edge 3),
        #            B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2)
        bands = ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

        # Sample at point
        sample = s2_clean.select(bands).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale,
            maxPixels=1e9
        )

        # Extract band values
        features = {}
        for band in bands:
            val = sample.get(band)
            features[f's2_{band.lower()}'] = float(val.getInfo()) if val else 0.0

        # Compute spectral indices at 10m resolution
        b2 = sample.get('B2').getInfo() or 0.0
        b3 = sample.get('B3').getInfo() or 0.0
        b4 = sample.get('B4').getInfo() or 0.0
        b8 = sample.get('B8').getInfo() or 0.0
        b11 = sample.get('B11').getInfo() or 0.0
        b12 = sample.get('B12').getInfo() or 0.0

        # NDVI (vegetation)
        if (b8 + b4) > 0:
            features['s2_ndvi'] = (b8 - b4) / (b8 + b4)
        else:
            features['s2_ndvi'] = 0.0

        # NBR (burn ratio - for fire detection)
        if (b8 + b12) > 0:
            features['s2_nbr'] = (b8 - b12) / (b8 + b12)
        else:
            features['s2_nbr'] = 0.0

        # EVI (enhanced vegetation index)
        denom = b8 + 6 * b4 - 7.5 * b2 + 1
        if denom != 0:
            features['s2_evi'] = 2.5 * (b8 - b4) / denom
        else:
            features['s2_evi'] = 0.0

        # NDWI (water index)
        if (b3 + b8) > 0:
            features['s2_ndwi'] = (b3 - b8) / (b3 + b8)
        else:
            features['s2_ndwi'] = 0.0

        return features

    except Exception as e:
        print(f"    ✗ Sentinel-2 extraction failed: {e}")
        return None


def extract_coarse_context(client, lat, lon, date, scale=100):
    """
    Extract coarse-scale landscape context features.

    Aggregates AlphaEarth embeddings over larger region to capture
    landscape-level patterns.

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string
        scale: Resolution in meters (default: 100m)

    Returns:
        Dict with coarse-scale features
    """
    try:
        # Sample 3x3 grid around center at 100m spacing
        step = 100 / 111320  # Convert meters to degrees

        embeddings = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                try:
                    emb = client.get_embedding(
                        lat + i * step,
                        lon + j * step,
                        date
                    )
                    embeddings.append(emb)
                except Exception:
                    continue

        if len(embeddings) == 0:
            return None

        embeddings = np.array(embeddings)

        features = {}

        # Mean embedding (landscape average)
        mean_emb = np.mean(embeddings, axis=0)
        for i, val in enumerate(mean_emb):
            features[f'coarse_emb_{i}'] = float(val)

        # Landscape heterogeneity (variance across region)
        variance = np.var(embeddings, axis=0)
        features['coarse_heterogeneity'] = float(np.mean(variance))

        # Landscape range (max - min per dimension)
        ranges = np.max(embeddings, axis=0) - np.min(embeddings, axis=0)
        features['coarse_range'] = float(np.mean(ranges))

        return features

    except Exception as e:
        print(f"    ✗ Coarse context extraction failed: {e}")
        return None


def extract_multiscale_features_for_sample(client, sample, timepoint='annual'):
    """
    Extract multi-scale features for a single sample.

    UPDATED: Uses annual timepoints to match AlphaEarth's capability.

    Combines:
    - Fine scale (10m Sentinel-2)
    - Medium scale (30m AlphaEarth, already extracted)
    - Coarse scale (100m landscape context)

    Args:
        client: EarthEngineClient
        sample: Sample dict with lat, lon, year
        timepoint: Which timepoint to extract for (default: 'annual' uses year-06-01)

    Returns:
        Updated sample dict with 'multiscale_features' key
    """
    lat = sample['lat']
    lon = sample['lon']
    year = sample.get('year', 2021)

    # Use annual date (mid-year to match annual embedding approach)
    # This is consistent with our corrected annual features
    if timepoint == 'annual' or timepoint == 'Y':
        date = f'{year}-06-01'
    elif timepoint == 'Y-1':
        date = f'{year-1}-06-01'
    elif timepoint == 'Y-2':
        date = f'{year-2}-06-01'
    else:
        # Default to annual mid-year
        date = f'{year}-06-01'

    multiscale_features = {}

    # Extract fine-scale Sentinel-2 features
    s2_feats = extract_sentinel2_features(client, lat, lon, date, scale=10)
    if s2_feats is not None:
        multiscale_features.update(s2_feats)

    # Extract coarse-scale context
    coarse_feats = extract_coarse_context(client, lat, lon, date, scale=100)
    if coarse_feats is not None:
        multiscale_features.update(coarse_feats)

    # Medium-scale (30m AlphaEarth) already in sample['features']
    # Will be used together during training

    if len(multiscale_features) == 0:
        return None

    sample['multiscale_features'] = multiscale_features

    return sample


def enrich_dataset_with_multiscale_features(set_name, config):
    """
    Enrich a validation set with multi-scale features.

    Args:
        set_name: Name of validation set (e.g., 'rapid_response')
        config: Config object

    Returns:
        Enriched samples with multi-scale features
    """
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    # Load enriched dataset (with temporal features)
    input_file = processed_dir / f"hard_val_{set_name}_features.pkl"
    output_file = processed_dir / f"hard_val_{set_name}_multiscale.pkl"

    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return None

    print(f"\n{'='*80}")
    print(f"EXTRACTING MULTI-SCALE FEATURES FOR: {set_name}")
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

    # Extract multi-scale features for each sample
    enriched_samples = []
    failed_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Extracting multi-scale features")):
        try:
            enriched_sample = extract_multiscale_features_for_sample(client, sample)
            if enriched_sample is not None:
                enriched_samples.append(enriched_sample)
            else:
                # Keep sample without multi-scale features
                enriched_samples.append(sample)
                failed_samples.append(i)
        except Exception as e:
            print(f"\n  ✗ Failed on sample {i}: {e}")
            # Keep sample without multi-scale features
            enriched_samples.append(sample)
            failed_samples.append(i)

    print(f"\n✓ Extracted multi-scale features for {len(enriched_samples)-len(failed_samples)}/{len(samples)} samples")
    if failed_samples:
        print(f"  ✗ Failed: {len(failed_samples)} samples")
        print(f"    Indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")

    # Save enriched dataset
    with open(output_file, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to {output_file}")

    # Print feature summary
    if len(enriched_samples) > 0:
        sample_with_features = next(
            (s for s in enriched_samples if 'multiscale_features' in s),
            None
        )
        if sample_with_features:
            n_features = len(sample_with_features['multiscale_features'])
            print(f"\n📊 MULTI-SCALE FEATURES:")
            print(f"  Total features per sample: {n_features}")

            # Count by type
            feats = sample_with_features['multiscale_features']
            n_s2 = sum(1 for k in feats.keys() if k.startswith('s2_'))
            n_coarse = sum(1 for k in feats.keys() if k.startswith('coarse_'))

            print(f"  Fine-scale (Sentinel-2): {n_s2}")
            print(f"  Coarse-scale (landscape): {n_coarse}")

    return enriched_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set',
        type=str,
        default='rapid_response',
        choices=['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases', 'all'],
        help='Which validation set to extract multi-scale features for'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-SCALE FEATURE EXTRACTION FOR HARD VALIDATION SETS")
    print("=" * 80)

    config = get_config()

    if args.set == 'all':
        sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']
    else:
        sets = [args.set]

    for set_name in sets:
        enrich_dataset_with_multiscale_features(set_name, config)

    print("\n" + "=" * 80)
    print("MULTI-SCALE FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext: Train model with multi-scale features:")
    print("  uv run python src/walk/09_train_multiscale.py")


if __name__ == "__main__":
    main()
