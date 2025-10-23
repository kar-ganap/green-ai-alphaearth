#!/usr/bin/env python3
"""
Extract Features for Uniform 30% Dataset (2020-2023)

Extracts 69D features for the uniform 30% threshold dataset:
- 3D annual features (delta_1yr, delta_2yr, acceleration)
- 66D coarse multiscale features (64D embedding + heterogeneity + range)

Input: walk_dataset_uniform_30pct_2020_2023_*.pkl (600 samples)
Output: walk_dataset_uniform_30pct_2020_2023_with_features_*.pkl

Usage:
    uv run python src/walk/39_extract_features_uniform_30pct.py
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features

# Import multiscale features extraction function
import importlib.util
spec = importlib.util.spec_from_file_location(
    "multiscale_module",
    Path(__file__).parent / "08_multiscale_embeddings.py"
)
multiscale_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiscale_module)
extract_multiscale_features_for_sample = multiscale_module.extract_multiscale_features_for_sample


def load_uniform_30pct_dataset():
    """Load uniform 30% threshold dataset."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    pattern = 'walk_dataset_uniform_30pct_2020_2023_*.pkl'
    files = list(processed_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No uniform 30% data found matching: {pattern}")

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"✓ Loading: {latest_file.name}")

    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    samples = data.get('data', data.get('samples', data))
    metadata = data.get('metadata', {})

    print(f"  Samples: {len(samples)}")

    return samples, metadata


def extract_features_for_sample(client, sample):
    """
    Extract 69D features for a single sample.

    Returns:
        dict: Sample with added features, or None if extraction fails
    """
    try:
        # Extract annual features using diagnostic_helpers (3D numpy array)
        annual_features = extract_dual_year_features(client, sample)

        if annual_features is None:
            return None

        # Extract multiscale features (returns updated sample)
        updated_sample = extract_multiscale_features_for_sample(client, sample, timepoint='annual')

        if updated_sample is None:
            return None

        # Add annual features to the updated sample
        updated_sample['annual_features'] = annual_features

        return updated_sample

    except Exception as e:
        lat = sample.get('lat', 'unknown')
        lon = sample.get('lon', 'unknown')
        print(f"  Error extracting features for ({lat}, {lon}): {e}")
        return None


def extract_all_features(samples):
    """Extract features for all samples."""
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES")
    print(f"{'='*80}\n")

    print("Initializing Earth Engine...")
    client = EarthEngineClient()
    print("✓ Earth Engine initialized\n")

    print(f"Extracting features for {len(samples)} samples...")

    samples_with_features = []
    failed = 0

    for sample in tqdm(samples, desc="Extracting features"):
        result = extract_features_for_sample(client, sample)

        if result is not None:
            samples_with_features.append(result)
        else:
            failed += 1

    print(f"\n✓ Successfully extracted: {len(samples_with_features)} samples")
    print(f"✗ Failed: {failed} samples")

    return samples_with_features


def save_features(samples, metadata):
    """Save samples with features."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = processed_dir / f'walk_dataset_uniform_30pct_2020_2023_with_features_{timestamp}.pkl'

    # Update metadata
    metadata['n_samples_with_features'] = len(samples)
    metadata['feature_extraction_date'] = timestamp
    metadata['features'] = {
        'annual': '3D (delta_1yr, delta_2yr, acceleration)',
        'multiscale': '66D coarse (64D embedding + heterogeneity + range)',
        'total_dims': 69
    }

    data = {
        'data': samples,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✓ Saved {len(samples)} samples with features to:")
    print(f"  {output_path}")

    return output_path


def main():
    print("=" * 80)
    print("UNIFORM 30% FEATURE EXTRACTION")
    print("=" * 80)
    print()

    # Load uniform 30% samples
    print("Loading uniform 30% dataset...")
    samples, metadata = load_uniform_30pct_dataset()

    print(f"\nSample distribution:")
    n_clearing = sum(1 for s in samples if s.get('label') == 1)
    n_intact = sum(1 for s in samples if s.get('label') == 0)
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}")

    # Extract features
    samples_with_features = extract_all_features(samples)

    # Check distribution after extraction
    n_clearing_final = sum(1 for s in samples_with_features if s.get('label') == 1)
    n_intact_final = sum(1 for s in samples_with_features if s.get('label') == 0)

    print(f"\nFinal distribution:")
    print(f"  Clearing: {n_clearing_final}")
    print(f"  Intact: {n_intact_final}")

    # Save
    output_path = save_features(samples_with_features, metadata)

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    print("1. Run Phase 4 temporal validation with uniform 30%:")
    print(f"   uv run python src/walk/40_phase4_uniform_30pct_validation.py")
    print("\n2. Compare heterogeneous vs uniform models:")
    print(f"   uv run python src/walk/41_compare_sampling_strategies.py")
    print("\n3. Decompose drift (sampling bias vs temporal change):")
    print("   - Heterogeneous: 0.981 → 0.796 (18.9% drop)")
    print("   - Uniform 30%:   0.98X → ?")
    print("   - Difference reveals sampling bias contribution")

    return samples_with_features


if __name__ == '__main__':
    main()
