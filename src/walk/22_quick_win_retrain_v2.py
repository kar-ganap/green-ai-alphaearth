#!/usr/bin/env python3
"""
Quick Win Re-training Script (v2 with timeout and resume)

Takes the augmented dataset (589 + 26 = 615 samples) and:
1. Extracts annual features for all samples
2. Extracts multiscale features with timeout handling and resume capability
3. Re-trains Random Forest
4. Evaluates on edge_cases to measure improvement

Usage:
    uv run python src/walk/22_quick_win_retrain_v2.py
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
    print("QUICK WIN RE-TRAINING V2: WITH TIMEOUT & RESUME")
    print("=" * 80)
    print("Baseline edge_cases: 0.583 ROC-AUC (45.5% accuracy)")
    print("Target: 0.65-0.70 ROC-AUC")
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Find the augmented dataset (exclude features file)
    augmented_paths = list(processed_dir.glob('walk_dataset_scaled_phase1_*_quickwin.pkl'))
    augmented_paths = [p for p in augmented_paths if 'features' not in p.name and 'multiscale' not in p.name]

    if not augmented_paths:
        print("❌ ERROR: No augmented dataset found")
        return

    augmented_path = sorted(augmented_paths)[-1]
    print(f"✓ Loading augmented dataset: {augmented_path.name}")

    with open(augmented_path, 'rb') as f:
        augmented_samples = pickle.load(f)

    print(f"  Total samples: {len(augmented_samples)}")

    # Check if we have partial progress
    temp_path = processed_dir / 'hard_val_training_quickwin_multiscale_features.pkl'
    if temp_path.exists():
        print(f"\n✓ Found partial progress: {temp_path.name}")
        with open(temp_path, 'rb') as f:
            valid_samples = pickle.load(f)

        start_idx = len(valid_samples)
        print(f"  Resuming from sample {start_idx}/{len(augmented_samples)}")
    else:
        # Extract annual features for all samples first
        print("\n" + "=" * 80)
        print("STEP 1: EXTRACT ANNUAL FEATURES")
        print("=" * 80)

        client = EarthEngineClient(use_cache=True)

        X_list = []
        y_list = []
        valid_samples = []
        failed_count = 0

        print(f"\nExtracting features for {len(augmented_samples)} samples...")

        for i, sample in enumerate(augmented_samples):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(augmented_samples)}")

            # Check if features already exist
            if 'annual_features' in sample:
                features = sample['annual_features']
            else:
                features = extract_dual_year_features(client, sample)

            if features is not None:
                X_list.append(features)
                y_list.append(sample.get('label', 0))

                # Ensure sample has the features stored
                sample_copy = sample.copy()
                sample_copy['annual_features'] = features
                valid_samples.append(sample_copy)
            else:
                failed_count += 1

        import numpy as np
        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n✓ Successfully extracted: {len(X)}/{len(augmented_samples)} samples")
        print(f"  Failed: {failed_count} samples ({failed_count/len(augmented_samples)*100:.1f}%)")
        print(f"\nClass distribution:")
        print(f"  Clearing (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"  Intact (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")

        # Save features dataset
        features_path = processed_dir / 'walk_dataset_scaled_phase1_features_quickwin.pkl'
        output_data = {
            'X': X,
            'y': y,
            'samples': valid_samples,
            'metadata': {
                'n_total': len(augmented_samples),
                'n_extracted': len(X),
                'n_failed': failed_count,
                'timestamp': datetime.now().isoformat()
            }
        }

        with open(features_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\n✓ Saved features: {features_path.name}")

        start_idx = 0

    # Extract multiscale features with timeout and resume
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACT MULTISCALE FEATURES (WITH TIMEOUT)")
    print("=" * 80)

    client = EarthEngineClient(use_cache=True)

    print(f"\nExtracting multiscale features (timeout: 30s per sample)...")
    print(f"Starting from sample {start_idx}/{len(valid_samples)}")

    failed_indices = []
    timeout_indices = []

    from tqdm import tqdm

    for i in tqdm(range(start_idx, len(valid_samples)),
                  desc="Extracting multi-scale features",
                  initial=start_idx,
                  total=len(valid_samples)):

        sample = valid_samples[i]

        # Check if already has multiscale features
        if 'multiscale_features' in sample:
            continue

        # Extract with timeout
        multiscale_feats = extract_multiscale_with_timeout(client, sample, timeout_seconds=30)

        if multiscale_feats is not None:
            valid_samples[i]['multiscale_features'] = multiscale_feats
        else:
            # Keep track of failures for logging
            if "Timeout" in str(multiscale_feats):
                timeout_indices.append(i)
            else:
                failed_indices.append(i)

        # Save progress every 50 samples
        if (i + 1) % 50 == 0:
            with open(temp_path, 'wb') as f:
                pickle.dump(valid_samples, f)
            print(f"\n  💾 Saved progress: {i+1}/{len(valid_samples)}")

    # Final save
    with open(temp_path, 'wb') as f:
        pickle.dump(valid_samples, f)

    # Count samples with multiscale features
    n_with_multiscale = sum(1 for s in valid_samples if 'multiscale_features' in s)

    print(f"\n✓ Multiscale extraction complete:")
    print(f"  Success: {n_with_multiscale}/{len(valid_samples)} samples")
    print(f"  Failed: {len(failed_indices)} samples")
    print(f"  Timeout: {len(timeout_indices)} samples")

    if failed_indices or timeout_indices:
        print(f"\nProblem indices:")
        if failed_indices:
            print(f"  Failed: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        if timeout_indices:
            print(f"  Timeout: {timeout_indices[:10]}{'...' if len(timeout_indices) > 10 else ''}")

    # Save final multiscale dataset
    multiscale_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale_quickwin.pkl'
    multiscale_data = {
        'data': valid_samples,
        'metadata': {
            'n_samples': len(valid_samples),
            'n_with_multiscale': n_with_multiscale,
            'timestamp': datetime.now().isoformat()
        }
    }

    with open(multiscale_path, 'wb') as f:
        pickle.dump(multiscale_data, f)

    print(f"\n✓ Saved multiscale features: {multiscale_path.name}")

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
        print(f"✓ Cleaned up temp file")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Augmented dataset: {len(augmented_samples)} samples")
    print(f"✓ Annual features extracted: {len(valid_samples)} samples")
    print(f"✓ Multiscale features extracted: {n_with_multiscale} samples")
    print(f"\nNext: Re-train and evaluate to measure improvement on edge_cases")
    print(f"Baseline: 0.583 ROC-AUC → Target: 0.65-0.70 ROC-AUC")
    print("\nRun: uv run python src/walk/11_train_random_forest.py")
    print("  (Update paths to use *_quickwin.pkl files)")


if __name__ == '__main__':
    main()
