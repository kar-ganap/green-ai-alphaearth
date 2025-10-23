#!/usr/bin/env python3
"""
Extract Features for All Hard Samples Dataset

Takes the all_hard_samples dataset (685 samples) and:
1. Ensures annual features exist for all samples
2. Extracts multiscale features for samples that don't have them
3. Saves feature-ready dataset for training

Usage:
    uv run python src/walk/28_extract_features_all_hard_samples.py
"""

import pickle
import importlib.util
import signal
from pathlib import Path
from datetime import datetime
from functools import wraps

from src.utils import get_config, EarthEngineClient
from src.walk.diagnostic_helpers import extract_dual_year_features


class TimeoutError(Exception):
    pass


def timeout(seconds=30):
    """Timeout decorator using signal (Unix only)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds} seconds")

            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Reset the alarm and handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result
        return wrapper
    return decorator


def extract_multiscale_with_timeout(client, sample, timeout_seconds=30):
    """
    Extract multiscale features with timeout protection.

    Returns:
        dict or None: Multiscale features or None if failed/timeout
    """
    try:
        # Import the function dynamically
        spec = importlib.util.spec_from_file_location(
            "multiscale_module",
            Path(__file__).parent / "08_multiscale_embeddings.py"
        )
        multiscale_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multiscale_module)

        extract_fn = multiscale_module.extract_multiscale_features_for_sample

        # Wrap with timeout
        @timeout(timeout_seconds)
        def extract_with_timeout():
            return extract_fn(client, sample.copy())

        enriched = extract_with_timeout()

        if enriched and 'multiscale_features' in enriched:
            return enriched['multiscale_features']
        return None

    except TimeoutError as e:
        print(f"   ⏱ Timeout: {e}")
        return None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None


def main():
    print("=" * 80)
    print("EXTRACT FEATURES FOR ALL HARD SAMPLES DATASET")
    print("=" * 80)
    print("Dataset: 685 samples (589 original + 47 edge_cases + 49 new)")
    print("Target: Ensure annual + multiscale features for all")
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Find the all_hard_samples dataset
    augmented_paths = list(processed_dir.glob('walk_dataset_scaled_phase1_*_all_hard_samples.pkl'))

    if not augmented_paths:
        print("❌ ERROR: No all_hard_samples dataset found")
        return

    augmented_path = sorted(augmented_paths)[-1]
    print(f"✓ Loading dataset: {augmented_path.name}")

    with open(augmented_path, 'rb') as f:
        augmented_samples = pickle.load(f)

    print(f"  Total samples: {len(augmented_samples)}")

    # Check annual features
    n_with_annual = sum(1 for s in augmented_samples if 'annual_features' in s)
    print(f"  Samples with annual features: {n_with_annual}/{len(augmented_samples)}")

    # Check multiscale features
    n_with_multiscale = sum(1 for s in augmented_samples if 'multiscale_features' in s)
    print(f"  Samples with multiscale features: {n_with_multiscale}/{len(augmented_samples)}")

    # Extract annual features if needed
    if n_with_annual < len(augmented_samples):
        print("\n" + "=" * 80)
        print("STEP 1: EXTRACT MISSING ANNUAL FEATURES")
        print("=" * 80)

        client = EarthEngineClient(use_cache=True)
        missing_count = 0

        for i, sample in enumerate(augmented_samples):
            if 'annual_features' not in sample:
                if missing_count == 0:
                    print(f"\nExtracting annual features for samples without them...")

                features = extract_dual_year_features(client, sample)
                if features is not None:
                    sample['annual_features'] = features
                    missing_count += 1
                    if missing_count % 10 == 0:
                        print(f"  Extracted: {missing_count}")

        print(f"\n✓ Extracted annual features for {missing_count} samples")
        n_with_annual = sum(1 for s in augmented_samples if 'annual_features' in s)

    # Extract multiscale features
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACT MULTISCALE FEATURES (WITH TIMEOUT)")
    print("=" * 80)

    client = EarthEngineClient(use_cache=True)

    # Find samples that need multiscale features
    samples_needing_multiscale = [
        i for i, s in enumerate(augmented_samples)
        if 'multiscale_features' not in s and 'annual_features' in s
    ]

    if not samples_needing_multiscale:
        print("\n✓ All samples already have multiscale features!")
    else:
        print(f"\nExtracting multiscale features for {len(samples_needing_multiscale)} samples...")
        print(f"Timeout: 30s per sample")

        from tqdm import tqdm

        failed_indices = []
        timeout_indices = []
        success_count = 0

        for i in tqdm(samples_needing_multiscale,
                      desc="Extracting multi-scale features"):

            sample = augmented_samples[i]

            # Extract with timeout
            multiscale_feats = extract_multiscale_with_timeout(client, sample, timeout_seconds=30)

            if multiscale_feats is not None:
                augmented_samples[i]['multiscale_features'] = multiscale_feats
                success_count += 1
            else:
                if "Timeout" in str(multiscale_feats):
                    timeout_indices.append(i)
                else:
                    failed_indices.append(i)

            # Save progress every 25 samples
            if (success_count + len(failed_indices) + len(timeout_indices)) % 25 == 0:
                temp_path = augmented_path.parent / f'temp_{augmented_path.stem}.pkl'
                with open(temp_path, 'wb') as f:
                    pickle.dump(augmented_samples, f)
                print(f"\n  💾 Saved progress")

        print(f"\n✓ Multiscale extraction complete:")
        print(f"  Success: {success_count}/{len(samples_needing_multiscale)} samples")
        print(f"  Failed: {len(failed_indices)} samples")
        print(f"  Timeout: {len(timeout_indices)} samples")

        if failed_indices or timeout_indices:
            print(f"\nProblem indices:")
            if failed_indices:
                print(f"  Failed: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
            if timeout_indices:
                print(f"  Timeout: {timeout_indices[:10]}{'...' if len(timeout_indices) > 10 else ''}")

    # Save final dataset with all features
    final_n_with_multiscale = sum(1 for s in augmented_samples if 'multiscale_features' in s)

    print("\n" + "=" * 80)
    print("SAVING FINAL DATASET")
    print("=" * 80)

    # Update original file with features
    with open(augmented_path, 'wb') as f:
        pickle.dump(augmented_samples, f)
    print(f"✓ Updated dataset: {augmented_path.name}")

    # Also create a multiscale-specific file
    multiscale_path = processed_dir / augmented_path.name.replace('.pkl', '_multiscale.pkl')
    multiscale_data = {
        'data': augmented_samples,
        'metadata': {
            'n_samples': len(augmented_samples),
            'n_with_annual': sum(1 for s in augmented_samples if 'annual_features' in s),
            'n_with_multiscale': final_n_with_multiscale,
            'timestamp': datetime.now().isoformat()
        }
    }

    with open(multiscale_path, 'wb') as f:
        pickle.dump(multiscale_data, f)
    print(f"✓ Saved multiscale dataset: {multiscale_path.name}")

    # Clean up temp file if exists
    temp_path = augmented_path.parent / f'temp_{augmented_path.stem}.pkl'
    if temp_path.exists():
        temp_path.unlink()
        print(f"✓ Cleaned up temp file")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ All hard samples dataset: {len(augmented_samples)} samples")
    print(f"✓ Annual features: {sum(1 for s in augmented_samples if 'annual_features' in s)} samples")
    print(f"✓ Multiscale features: {final_n_with_multiscale} samples")
    print(f"\nNext: Re-train and evaluate to measure improvement")
    print(f"  Baseline:  0.583 ROC-AUC (589 samples)")
    print(f"  Quick Win: 0.600 ROC-AUC (615 samples)")
    print(f"  Edge cases complete: 0.600 ROC-AUC (636 samples)")
    print(f"  Target:    0.65-0.70 ROC-AUC (685 samples)")


if __name__ == '__main__':
    main()
