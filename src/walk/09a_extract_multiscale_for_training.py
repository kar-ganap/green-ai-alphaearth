"""
Extract Multiscale Features for Phase 1 Training Set

Enriches the Phase 1 training samples with multi-scale features:
- Fine-scale (10m Sentinel-2 spectral features)
- Coarse-scale (100m landscape context)

Combined with annual delta features (3D), this gives us 83D total.

Usage:
    uv run python src/walk/09a_extract_multiscale_for_training.py
"""

import pickle
import importlib.util
from pathlib import Path

from src.utils import get_config

# Import the multiscale embeddings module dynamically (can't import file starting with number normally)
spec = importlib.util.spec_from_file_location(
    "multiscale_module",
    Path(__file__).parent / "08_multiscale_embeddings.py"
)
multiscale_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiscale_module)
enrich_dataset_with_multiscale_features = multiscale_module.enrich_dataset_with_multiscale_features


def extract_for_training_set():
    """Extract multiscale features for Phase 1 training dataset."""

    print("=" * 80)
    print("PHASE 1: EXTRACT MULTISCALE FEATURES FOR TRAINING SET")
    print("=" * 80)

    # Initialize
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # We need to adapt the training dataset to look like a validation set
    # Load Phase 1 dataset
    dataset_path = processed_dir / 'walk_dataset_scaled_phase1.pkl'

    print(f"\nLoading training dataset from: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    samples = dataset['data']
    metadata = dataset['metadata']

    print(f"  Loaded {len(samples)} samples")
    print(f"  Clearing: {metadata['clearing_actual']}")
    print(f"  Intact: {metadata['intact_actual']}")

    # Save as temporary validation-style file for the enrichment function
    # The enrichment function expects format: hard_val_{set_name}_features.pkl
    temp_path = processed_dir / 'hard_val_training_multiscale_features.pkl'
    with open(temp_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"\n✓ Saved temp file: {temp_path}")

    # Use the existing enrichment function
    print("\nEnriching with multiscale features...")
    enriched_samples = enrich_dataset_with_multiscale_features('training_multiscale', config)

    # Save enriched training dataset
    output_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale.pkl'

    output_data = {
        'data': enriched_samples,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved enriched training set to: {output_path}")

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
        print(f"✓ Cleaned up temp file")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Train multiscale model combining:")
    print("   - Annual delta features (3D)")
    print("   - Multiscale features (80D)")
    print("   - Total: 83D")
    print("\n2. Compare to annual-only baseline:")
    print("   - edge_cases: 0.533 → ? (target: 0.70+)")


if __name__ == '__main__':
    extract_for_training_set()
