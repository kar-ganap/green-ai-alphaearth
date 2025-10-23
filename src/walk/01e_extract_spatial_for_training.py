"""
WALK Phase - Extract Spatial Features for Training Set

Extracts spatial features for the full training dataset to enable
proper cross-validation comparison.

Usage:
    uv run python src/walk/01e_extract_spatial_for_training.py
"""

import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# Add walk directory to path to import numbered modules
walk_dir = Path(__file__).parent
sys.path.insert(0, str(walk_dir))

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient

# Import from numbered module using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_spatial",
    walk_dir / "01d_extract_spatial_features.py"
)
extract_spatial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_spatial)
extract_spatial_features_for_sample = extract_spatial.extract_spatial_features_for_sample


def main():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("=" * 80)
    print("EXTRACT SPATIAL FEATURES FOR TRAINING SET")
    print("=" * 80)

    # Load training dataset
    input_file = processed_dir / "walk_dataset.pkl"
    output_file = processed_dir / "walk_dataset_spatial.pkl"

    print(f"\nLoading dataset from {input_file.name}...")
    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)

    all_samples = dataset['data']
    splits = dataset['splits']
    metadata = dataset['metadata']

    print(f"Loaded {len(all_samples)} total samples")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")

    # Initialize Earth Engine client
    print("\nInitializing Earth Engine client...")
    client = EarthEngineClient(use_cache=True)

    # Extract spatial features for all samples
    print("\nExtracting spatial features for all samples...")
    print("(This will take ~8-10 minutes for 114 samples)")

    enriched_samples = []
    failed_indices = []

    for i, sample in enumerate(tqdm(all_samples, desc="Extracting spatial features")):
        try:
            # Handle different sample structures
            # Training samples: {location: {lat, lon}, ...}
            # Validation samples: {lat, lon, ...}
            if 'location' in sample:
                # Training set structure
                sample_copy = sample.copy()
                sample_copy['lat'] = sample['location']['lat']
                sample_copy['lon'] = sample['location']['lon']
                enriched_sample = extract_spatial_features_for_sample(client, sample_copy)
                if enriched_sample is not None:
                    # Remove temporary lat/lon and add spatial features to original
                    del enriched_sample['lat']
                    del enriched_sample['lon']
                    sample['spatial_features'] = enriched_sample['spatial_features']
                    enriched_samples.append(sample)
                else:
                    enriched_samples.append(sample)
                    failed_indices.append(i)
            else:
                # Validation set structure (already has flat lat/lon)
                enriched_sample = extract_spatial_features_for_sample(client, sample)
                if enriched_sample is not None:
                    enriched_samples.append(enriched_sample)
                else:
                    enriched_samples.append(sample)
                    failed_indices.append(i)
        except Exception as e:
            print(f"\n  ✗ Failed on sample {i}: {e}")
            # Keep original sample
            enriched_samples.append(sample)
            failed_indices.append(i)

    success_count = len(all_samples) - len(failed_indices)
    print(f"\n✓ Extracted spatial features for {success_count}/{len(all_samples)} samples")
    if failed_indices:
        print(f"  ⚠ {len(failed_indices)} samples kept without spatial features")
        print(f"    Indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

    # Save enriched dataset
    enriched_dataset = {
        'data': enriched_samples,
        'splits': splits,
        'metadata': metadata
    }

    with open(output_file, 'wb') as f:
        pickle.dump(enriched_dataset, f)

    print(f"\n✓ Saved to {output_file}")
    print("\nNext: Run feature validation analysis:")
    print("  uv run python src/walk/05_spatial_feature_validation.py")


if __name__ == "__main__":
    main()
