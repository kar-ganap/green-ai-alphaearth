#!/usr/bin/env python3
"""
Extract features for missing rapid_response validation samples (2022, 2023).
"""

import pickle
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

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

# Paths
config = get_config()
data_dir = config.get_path("paths.data_dir")
PROCESSED_DIR = data_dir / 'processed'

# Missing files
MISSING_FILES = [
    ("hard_val_rapid_response_2023_20251023_101559.pkl", 2023),
    ("hard_val_rapid_response_2022_20251023_101531.pkl", 2022),
]


def extract_features_for_sample(client, sample, year):
    """Extract 70D features for a single sample."""
    try:
        # 1. Extract annual features (3D)
        annual_features = extract_dual_year_features(client, sample)
        if annual_features is None:
            return None

        # 2. Extract multiscale features (66D)
        updated_sample = extract_multiscale_features_for_sample(client, sample, timepoint='annual')
        if updated_sample is None:
            return None

        # 3. Add annual features
        updated_sample['annual_features'] = annual_features

        # 4. Add year feature (normalized to [0, 1] for 2020-2024)
        year_normalized = (year - 2020) / 4.0
        updated_sample['year_feature'] = year_normalized

        return updated_sample

    except Exception as e:
        print(f"    Error: {e}")
        return None


def process_file(client, filename, year):
    """Process a single validation file."""
    filepath = PROCESSED_DIR / filename

    if not filepath.exists():
        print(f"⚠️ File not found: {filename}")
        return 0, 0

    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"Year: {year}")
    print(f"{'='*80}")

    # Load samples
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")

    # Extract features for each sample
    samples_with_features = []
    failed_count = 0

    for sample in tqdm(samples, desc=f"Extracting {year}"):
        result = extract_features_for_sample(client, sample, year)

        if result is not None:
            samples_with_features.append(result)
        else:
            failed_count += 1

    print(f"\n✓ Success: {len(samples_with_features)}")
    print(f"✗ Failed: {failed_count}")
    print(f"  Success rate: {len(samples_with_features)/len(samples)*100:.1f}%")

    # Save with features
    output_filename = filename.replace('.pkl', '_features.pkl')
    output_path = PROCESSED_DIR / output_filename

    with open(output_path, 'wb') as f:
        pickle.dump(samples_with_features, f)

    print(f"✓ Saved to: {output_filename}")

    return len(samples_with_features), failed_count


def main():
    """Extract features for missing rapid_response files."""
    print("="*80)
    print("MISSING RAPID_RESPONSE FEATURE EXTRACTION")
    print("="*80)

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    client = EarthEngineClient()
    print("✓ Earth Engine initialized")

    # Process missing files
    total_success = 0
    total_failed = 0

    for filename, year in MISSING_FILES:
        success, failed = process_file(client, filename, year)
        total_success += success
        total_failed += failed

    # Summary
    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples processed: {total_success + total_failed}")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_failed}")
    if total_success + total_failed > 0:
        print(f"  Success rate: {total_success/(total_success+total_failed)*100:.1f}%")
    print(f"\n✓ Feature extraction complete!")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
