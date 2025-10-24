#!/usr/bin/env python3
"""
Verification Script: Test Consolidated Feature Extraction

Tests that the new consolidated module produces IDENTICAL results
to the production implementation used to create validation files.

This script:
1. Loads production validation files (with pre-extracted features)
2. Re-extracts features using the new consolidated module
3. Compares results (should be identical)
4. Reports any discrepancies

Usage:
    uv run python src/walk/verify_consolidation.py
"""

import pickle
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.utils import extract_70d_features, features_to_array

# Config
config = get_config()
data_dir = config.get_path("paths.data_dir")
PROCESSED_DIR = data_dir / 'processed'

# Production validation files to test against
TEST_FILES = [
    "hard_val_risk_ranking_2024_20251023_015822_features.pkl",
    "hard_val_comprehensive_2024_20251023_015827_features.pkl",
    "hard_val_rapid_response_2024_20251023_101620_features.pkl",
]


def verify_file(client, filepath):
    """
    Verify that consolidated module produces identical features.

    Returns:
        (total, matches, mismatches, errors)
    """
    print(f"\n{'='*80}")
    print(f"Testing: {filepath.name}")
    print(f"{'='*80}")

    # Load production file
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples with pre-extracted features")

    total = 0
    matches = 0
    mismatches = 0
    errors = 0

    for i, sample in enumerate(tqdm(samples, desc="Verifying")):
        total += 1

        try:
            # Get production features
            prod_annual = sample.get('annual_features')
            prod_multiscale = sample.get('multiscale_features')
            prod_year = sample.get('year_feature')

            if prod_annual is None or prod_multiscale is None or prod_year is None:
                print(f"  Sample {i}: Missing production features")
                errors += 1
                continue

            # Extract using consolidated module
            result = extract_70d_features(client, sample, sample.get('year'))

            if result is None:
                print(f"  Sample {i}: Consolidated extraction failed")
                errors += 1
                continue

            new_annual, new_multiscale, new_year = result

            # Compare annual features (3D numpy array)
            annual_match = np.allclose(new_annual, prod_annual, rtol=1e-5, atol=1e-8)

            # Compare multiscale features (dict with 66 features)
            multiscale_match = True
            for key in ['coarse_heterogeneity', 'coarse_range'] + [f'coarse_emb_{i}' for i in range(64)]:
                prod_val = prod_multiscale.get(key, 0.0)
                new_val = new_multiscale.get(key, 0.0)
                if not np.isclose(prod_val, new_val, rtol=1e-5, atol=1e-8):
                    multiscale_match = False
                    if i < 3:  # Only print first few mismatches
                        print(f"  Sample {i}: Multiscale mismatch on {key}: {prod_val} vs {new_val}")
                    break

            # Compare year feature
            year_match = np.isclose(new_year, prod_year, rtol=1e-5, atol=1e-8)

            if annual_match and multiscale_match and year_match:
                matches += 1
            else:
                mismatches += 1
                if mismatches <= 5:  # Print first 5 mismatches
                    print(f"  Sample {i} MISMATCH:")
                    if not annual_match:
                        print(f"    Annual: {new_annual} vs {prod_annual}")
                    if not year_match:
                        print(f"    Year: {new_year} vs {prod_year}")

        except Exception as e:
            print(f"  Sample {i}: Error - {e}")
            errors += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS: {filepath.name}")
    print(f"{'='*80}")
    print(f"Total samples:   {total}")
    print(f"✓ Matches:       {matches} ({matches/total*100:.1f}%)")
    print(f"✗ Mismatches:    {mismatches} ({mismatches/total*100:.1f}%)")
    print(f"⚠ Errors:        {errors} ({errors/total*100:.1f}%)")

    return total, matches, mismatches, errors


def test_import():
    """Test that imports work correctly."""
    print("\n" + "="*80)
    print("STEP 1: Testing imports")
    print("="*80)

    try:
        from src.walk.utils import (
            extract_70d_features,
            extract_annual_features,
            extract_coarse_multiscale_features,
            enrich_sample_with_features,
            features_to_array,
            FEATURE_NAMES_70D,
            FEATURE_NAMES_ANNUAL,
            FEATURE_NAMES_COARSE,
        )
        print("✓ All imports successful")

        # Check feature names
        assert len(FEATURE_NAMES_70D) == 70, f"Expected 70 features, got {len(FEATURE_NAMES_70D)}"
        assert len(FEATURE_NAMES_ANNUAL) == 3, f"Expected 3 annual features, got {len(FEATURE_NAMES_ANNUAL)}"
        assert len(FEATURE_NAMES_COARSE) == 66, f"Expected 66 coarse features, got {len(FEATURE_NAMES_COARSE)}"
        print("✓ Feature counts correct: 3 annual + 66 coarse + 1 year = 70 total")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run complete verification suite."""
    print("\n" + "="*80)
    print("CONSOLIDATED FEATURE EXTRACTION VERIFICATION")
    print("="*80)

    # Step 1: Test imports
    if not test_import():
        print("\n✗ Import test failed. Exiting.")
        sys.exit(1)

    # Step 2: Initialize Earth Engine
    print("\n" + "="*80)
    print("STEP 2: Initializing Earth Engine")
    print("="*80)
    try:
        client = EarthEngineClient()
        print("✓ Earth Engine initialized")
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        sys.exit(1)

    # Step 3: Verify against production files
    print("\n" + "="*80)
    print("STEP 3: Verifying against production validation files")
    print("="*80)

    all_total = 0
    all_matches = 0
    all_mismatches = 0
    all_errors = 0

    for filename in TEST_FILES:
        filepath = PROCESSED_DIR / filename

        if not filepath.exists():
            print(f"\n⚠ File not found: {filename}")
            continue

        total, matches, mismatches, errors = verify_file(client, filepath)

        all_total += total
        all_matches += matches
        all_mismatches += mismatches
        all_errors += errors

    # Final summary
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Files tested:    {len(TEST_FILES)}")
    print(f"Total samples:   {all_total}")
    print(f"✓ Matches:       {all_matches} ({all_matches/all_total*100:.1f}%)")
    print(f"✗ Mismatches:    {all_mismatches} ({all_mismatches/all_total*100:.1f}%)")
    print(f"⚠ Errors:        {all_errors} ({all_errors/all_total*100:.1f}%)")

    # Pass/Fail
    if all_mismatches == 0 and all_errors == 0:
        print("\n" + "="*80)
        print("✓✓✓ VERIFICATION PASSED ✓✓✓")
        print("="*80)
        print("Consolidated module produces IDENTICAL results to production!")
        print("Safe to migrate scripts and delete old implementations.")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗✗✗ VERIFICATION FAILED ✗✗✗")
        print("="*80)
        print("Consolidated module produces DIFFERENT results!")
        print("DO NOT migrate scripts or delete old implementations.")
        print("Debug and fix consolidation before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
