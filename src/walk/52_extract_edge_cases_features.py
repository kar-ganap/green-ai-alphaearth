#!/usr/bin/env python3
"""
Extract Features for Edge Cases Samples

Extracts 70D features (annual + multiscale + year) for the 23 edge_cases samples.
These samples are mostly from 2021 (8 samples) with 15 having unknown year.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import importlib.util

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features

# Import multiscale features extraction function
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

def determine_year(sample):
    """Determine year from sample, inferring from date if 'unknown'."""
    year = sample.get('year', 'unknown')
    if year == 'unknown':
        # Try to parse from date
        if 'date' in sample and sample['date']:
            date_str = sample['date']
            if isinstance(date_str, str) and len(date_str) >= 4:
                return int(date_str[:4])
        return 2021  # Default fallback
    return year


def main():
    """Extract features for edge_cases samples."""
    print("="*80)
    print("EXTRACTING FEATURES FOR EDGE CASES SAMPLES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load edge_cases samples
    input_file = PROCESSED_DIR / 'hard_val_edge_cases.pkl'
    print(f"\nLoading samples from: {input_file}")

    with open(input_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} edge_cases samples")

    # Initialize Earth Engine client
    ee_client = EarthEngineClient()

    # Extract features for each sample
    samples_with_features = []
    failed_samples = []

    for i, sample in enumerate(samples, 1):
        lat = sample.get('lat')
        lon = sample.get('lon')
        year = determine_year(sample)

        print(f"\n[{i}/{len(samples)}] Processing ({lat:.4f}, {lon:.4f}) year={year}")

        try:
            # Extract annual features (3D) using diagnostic_helpers
            sample_dict = {'lat': lat, 'lon': lon, 'year': year}
            annual_features = extract_dual_year_features(ee_client, sample_dict)
            if annual_features is None or len(annual_features) != 3:
                print(f"  ⚠️ Failed to extract annual features")
                failed_samples.append(sample)
                continue

            # Extract multiscale features (66D) using multiscale module
            multiscale_features = extract_multiscale_features_for_sample(
                sample=sample_dict,
                year=year,
                ee_client=ee_client
            )
            if multiscale_features is None:
                print(f"  ⚠️ Failed to extract multiscale features")
                failed_samples.append(sample)
                continue

            # Add features to sample
            sample_with_features = sample.copy()
            sample_with_features['annual_features'] = annual_features
            sample_with_features['multiscale_features'] = multiscale_features
            sample_with_features['year'] = year  # Update to inferred year

            samples_with_features.append(sample_with_features)
            print(f"  ✓ Features extracted (annual: 3D, multiscale: 66D)")

        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            failed_samples.append(sample)
            continue

    # Summary
    print(f"\n{'='*80}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {len(samples_with_features)}")
    print(f"Failed: {len(failed_samples)}")

    # Save samples with features
    if len(samples_with_features) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = PROCESSED_DIR / f'hard_val_edge_cases_features_{timestamp}.pkl'

        with open(output_file, 'wb') as f:
            pickle.dump(samples_with_features, f)

        print(f"\n✓ Saved {len(samples_with_features)} samples with features to:")
        print(f"  {output_file}")
    else:
        print(f"\n⚠️ No samples with features to save")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
