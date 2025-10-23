"""
Extract Fine-Scale AlphaEarth Features

Adds fine-scale AlphaEarth embeddings at 10m and 20m spatial resolutions
to complement existing coarse-scale (100m) features.

This addresses small-scale clearing detection by providing:
- fine_10m: 3x3 grid sampled at 10m spacing (66D)
- fine_20m: 3x3 grid sampled at 20m spacing (66D)

Total: +132D features (66D × 2 scales)

Combined with existing features:
- Annual: 3D
- Coarse: 66D
- Fine: 132D
Total: 201D

Usage:
    # Extract for training set
    uv run python src/walk/12_extract_fine_scale_features.py --dataset training

    # Extract for validation sets
    uv run python src/walk/12_extract_fine_scale_features.py --dataset validation --set edge_cases
    uv run python src/walk/12_extract_fine_scale_features.py --dataset validation --set all
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def extract_fine_scale_context(client, lat, lon, date, scale=10, grid_size=3):
    """
    Extract fine-scale AlphaEarth features at specified spatial resolution.

    Samples a grid of AlphaEarth embeddings around the center point,
    then aggregates into mean embedding + variability metrics.

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)
        scale: Spacing between grid points in meters (10 or 20)
        grid_size: Grid dimension (default: 3 for 3x3)

    Returns:
        Dict with fine-scale features (66D: 64 mean embeddings + 2 summary stats)
    """
    try:
        # Convert meters to degrees (approximate)
        step = scale / 111320  # 1 degree ≈ 111.32 km at equator

        embeddings = []
        offset = (grid_size - 1) // 2  # For 3x3, offset is 1

        for i in range(-offset, offset + 1):
            for j in range(-offset, offset + 1):
                try:
                    emb = client.get_embedding(
                        lat + i * step,
                        lon + j * step,
                        date
                    )
                    if emb is not None:
                        embeddings.append(emb)
                except Exception:
                    continue

        if len(embeddings) == 0:
            return None

        embeddings = np.array(embeddings)

        features = {}

        # Mean embedding (local average)
        mean_emb = np.mean(embeddings, axis=0)
        prefix = f'fine_{scale}m_emb_'
        for i, val in enumerate(mean_emb):
            features[f'{prefix}{i}'] = float(val)

        # Local heterogeneity (variance across grid)
        variance = np.var(embeddings, axis=0)
        features[f'fine_{scale}m_heterogeneity'] = float(np.mean(variance))

        # Local range (max - min per dimension)
        ranges = np.max(embeddings, axis=0) - np.min(embeddings, axis=0)
        features[f'fine_{scale}m_range'] = float(np.mean(ranges))

        return features

    except Exception as e:
        return None


def extract_fine_scale_features_for_sample(client, sample):
    """
    Extract fine-scale features (10m + 20m) for a single sample.

    Args:
        client: EarthEngineClient
        sample: Sample dict with lat, lon, year

    Returns:
        Updated sample dict with 'fine_scale_features' key
    """
    lat = sample['lat']
    lon = sample['lon']
    year = sample.get('year', 2021)

    # Use annual mid-year date (consistent with other features)
    date = f'{year}-06-01'

    fine_features = {}

    # Extract 10m scale features
    fine_10m = extract_fine_scale_context(client, lat, lon, date, scale=10)
    if fine_10m is not None:
        fine_features.update(fine_10m)

    # Extract 20m scale features
    fine_20m = extract_fine_scale_context(client, lat, lon, date, scale=20)
    if fine_20m is not None:
        fine_features.update(fine_20m)

    if len(fine_features) == 0:
        return None

    # Merge with existing multiscale features if present
    if 'multiscale_features' in sample:
        sample['multiscale_features'].update(fine_features)
    else:
        sample['multiscale_features'] = fine_features

    return sample


def extract_for_training_set():
    """Extract fine-scale features for Phase 1 training dataset."""

    print("=" * 80)
    print("EXTRACT FINE-SCALE FEATURES FOR TRAINING SET")
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

    # Check current feature count
    if 'multiscale_features' in samples[0]:
        current_feats = len(samples[0]['multiscale_features'])
        print(f"  Current multiscale features: {current_feats}D")

    # Initialize Earth Engine client
    print("\nInitializing Earth Engine client...")
    client = EarthEngineClient(use_cache=True)

    # Extract fine-scale features
    print(f"\nExtracting fine-scale features (10m + 20m)...")
    print(f"  Estimated time: ~3-5 minutes for 589 samples (with caching)")

    enriched_samples = []
    failed_count = 0

    for i, sample in enumerate(tqdm(samples, desc="Extracting")):
        try:
            enriched_sample = extract_fine_scale_features_for_sample(client, sample)
            if enriched_sample is not None:
                enriched_samples.append(enriched_sample)
            else:
                enriched_samples.append(sample)
                failed_count += 1
        except Exception as e:
            enriched_samples.append(sample)
            failed_count += 1

    print(f"\n✓ Extracted fine-scale features for {len(enriched_samples) - failed_count}/{len(samples)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} samples (kept without fine-scale features)")

    # Check final feature count
    if 'multiscale_features' in enriched_samples[0]:
        final_feats = len(enriched_samples[0]['multiscale_features'])
        print(f"  Final multiscale features: {final_feats}D")
        print(f"  Added: {final_feats - current_feats}D fine-scale features")

    # Save enriched dataset
    output_path = processed_dir / 'walk_dataset_scaled_phase1_fine_scale.pkl'

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
    print("\n1. Extract fine-scale features for validation sets:")
    print("   uv run python src/walk/12_extract_fine_scale_features.py --dataset validation --set all")
    print("\n2. Train XGBoost with 201D features:")
    print("   uv run python src/walk/13_train_xgboost.py")


def extract_for_validation_set(set_name):
    """Extract fine-scale features for a validation set."""

    print("=" * 80)
    print(f"EXTRACT FINE-SCALE FEATURES FOR: {set_name}")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load existing multiscale validation data
    input_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'

    if not input_path.exists():
        print(f"\n✗ Input file not found: {input_path}")
        print(f"  Run multiscale extraction first:")
        print(f"  uv run python src/walk/08_multiscale_embeddings.py --set {set_name}")
        return

    print(f"\nLoading validation set from: {input_path}")

    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  Loaded {len(samples)} samples")

    # Check current feature count
    samples_with_feats = [s for s in samples if 'multiscale_features' in s]
    if len(samples_with_feats) > 0:
        current_feats = len(samples_with_feats[0]['multiscale_features'])
        print(f"  Current multiscale features: {current_feats}D")

    # Initialize Earth Engine client
    print("\nInitializing Earth Engine client...")
    client = EarthEngineClient(use_cache=True)

    # Extract fine-scale features
    print(f"\nExtracting fine-scale features (10m + 20m)...")

    enriched_samples = []
    failed_count = 0

    for i, sample in enumerate(tqdm(samples, desc="Extracting")):
        # Fix missing 'year' field for intact samples
        if 'year' not in sample and sample.get('stable', False):
            sample = sample.copy()
            sample['year'] = 2021

        try:
            enriched_sample = extract_fine_scale_features_for_sample(client, sample)
            if enriched_sample is not None:
                enriched_samples.append(enriched_sample)
            else:
                enriched_samples.append(sample)
                failed_count += 1
        except Exception as e:
            enriched_samples.append(sample)
            failed_count += 1

    print(f"\n✓ Extracted fine-scale features for {len(enriched_samples) - failed_count}/{len(samples)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} samples (kept without fine-scale features)")

    # Check final feature count
    enriched_with_feats = [s for s in enriched_samples if 'multiscale_features' in s]
    if len(enriched_with_feats) > 0:
        final_feats = len(enriched_with_feats[0]['multiscale_features'])
        print(f"  Final multiscale features: {final_feats}D")
        if len(samples_with_feats) > 0:
            print(f"  Added: {final_feats - current_feats}D fine-scale features")

    # Save enriched dataset
    output_path = processed_dir / f'hard_val_{set_name}_fine_scale.pkl'

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
    print("FINE-SCALE FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
