"""
Add Annual Magnitude Features to Validation Sets

The validation sets are missing the 3D annual magnitude features (delta_1yr, delta_2yr, acceleration)
that are needed for the full 115D feature set (3D annual + 66D coarse + 46D S2).

This script adds those features to validation sets that already have S2 features extracted.

Usage:
    uv run python src/walk/16a_add_annual_features_to_validation.py --set edge_cases
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient
from src.walk.diagnostic_helpers import extract_dual_year_features


def add_annual_features_to_validation(set_name: str):
    """Add annual magnitude features to a validation set."""

    print("=" * 80)
    print(f"ADD ANNUAL FEATURES TO: {set_name}")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load validation set with S2 features
    input_path = processed_dir / f'hard_val_{set_name}_sentinel2.pkl'

    if not input_path.exists():
        print(f"\n✗ Input file not found: {input_path}")
        print(f"  Run S2 extraction first: uv run python src/walk/14_extract_sentinel2_features.py --dataset validation --set {set_name}")
        return

    print(f"\nLoading validation set from: {input_path}")
    with open(input_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"  Loaded {len(samples)} samples")

    # Check current feature structure
    if len(samples) > 0 and 'multiscale_features' in samples[0]:
        keys = list(samples[0]['multiscale_features'].keys())
        annual_keys = [k for k in keys if 'annual_mag_' in k]
        coarse_keys = [k for k in keys if 'coarse_' in k]
        s2_keys = [k for k in keys if 's2_' in k]

        print(f"\n  Current feature structure:")
        print(f"    Annual: {len(annual_keys)}D")
        print(f"    Coarse: {len(coarse_keys)}D")
        print(f"    S2: {len(s2_keys)}D")
        print(f"    Total: {len(keys)}D")

        # Check for CORRECT annual feature names (delta_1yr, delta_2yr, acceleration)
        correct_annual_features = {'delta_1yr', 'delta_2yr', 'acceleration'}
        has_correct_features = correct_annual_features.issubset(set(keys))
        has_old_features = len(annual_keys) > 0

        if has_correct_features and not has_old_features:
            print(f"\n  ✓ Samples already have annual features with correct names!")
            return
        elif has_old_features:
            print(f"\n  ⚠ Found old annual features with incorrect names - will replace them")
        elif has_correct_features and has_old_features:
            print(f"\n  ⚠ Found both old and new features - will clean up old ones")

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    ee_client = EarthEngineClient(use_cache=True)

    # Add annual features
    print(f"\nExtracting annual magnitude features...")
    enriched_samples = []
    success_count = 0
    failed_count = 0

    for sample in tqdm(samples, desc="Processing"):
        # Fix missing 'year' field for intact samples
        if 'year' not in sample and sample.get('stable', False):
            sample = sample.copy()
            sample['year'] = 2021

        try:
            # Extract annual features
            annual_features = extract_dual_year_features(ee_client, sample)

            if annual_features is not None:
                # Add to multiscale_features dict
                # Use the same naming as training set
                if 'multiscale_features' not in sample:
                    sample['multiscale_features'] = {}

                # Remove old annual features if they exist (annual_mag_0/1/2)
                old_keys = [k for k in list(sample['multiscale_features'].keys()) if 'annual_mag_' in k]
                for key in old_keys:
                    del sample['multiscale_features'][key]

                sample['multiscale_features']['delta_1yr'] = float(annual_features[0])
                sample['multiscale_features']['delta_2yr'] = float(annual_features[1])
                sample['multiscale_features']['acceleration'] = float(annual_features[2])

                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            print(f"\n  Error processing sample: {e}")

        enriched_samples.append(sample)

    print(f"\n✓ Extracted annual features for {success_count}/{len(samples)} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} samples (kept without annual features)")

    # Verify final structure
    if len(enriched_samples) > 0 and 'multiscale_features' in enriched_samples[0]:
        keys = list(enriched_samples[0]['multiscale_features'].keys())
        annual_keys = [k for k in keys if 'annual_mag_' in k]
        coarse_keys = [k for k in keys if 'coarse_' in k]
        s2_keys = [k for k in keys if 's2_' in k]

        print(f"\n  Final feature structure:")
        print(f"    Annual: {len(annual_keys)}D")
        print(f"    Coarse: {len(coarse_keys)}D")
        print(f"    S2: {len(s2_keys)}D")
        print(f"    Total: {len(keys)}D")

    # Save enriched dataset
    output_path = input_path  # Overwrite the same file

    with open(output_path, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("NEXT STEP")
    print("=" * 80)
    print("\nNow you can evaluate the model with full 115D features:")
    print(f"  uv run python src/walk/16_evaluate_sentinel2_model.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set',
        type=str,
        required=True,
        help='Validation set name (e.g., edge_cases, risk_ranking, etc.)'
    )
    args = parser.parse_args()

    add_annual_features_to_validation(args.set)


if __name__ == '__main__':
    main()
