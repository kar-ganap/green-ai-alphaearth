#!/usr/bin/env python3
"""
Quick Win Re-training Script

Takes the augmented dataset (589 + 26 = 615 samples) and:
1. Extracts annual features for all samples
2. Extracts multiscale features for all samples
3. Re-trains Random Forest
4. Evaluates on edge_cases to measure improvement

Usage:
    uv run python src/walk/22_quick_win_retrain.py
"""

import pickle
import importlib.util
from pathlib import Path
from datetime import datetime

from src.utils import get_config, EarthEngineClient
from src.walk.diagnostic_helpers import extract_dual_year_features

# Import multiscale module dynamically
spec = importlib.util.spec_from_file_location(
    "multiscale_module",
    Path(__file__).parent / "08_multiscale_embeddings.py"
)
multiscale_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiscale_module)
enrich_dataset_with_multiscale_features = multiscale_module.enrich_dataset_with_multiscale_features


def main():
    print("=" * 80)
    print("QUICK WIN RE-TRAINING: AUGMENTED DATASET (615 SAMPLES)")
    print("=" * 80)
    print("Baseline edge_cases: 0.583 ROC-AUC (45.5% accuracy)")
    print("Target: 0.65-0.70 ROC-AUC")
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Find the augmented dataset
    augmented_paths = list(processed_dir.glob('walk_dataset_scaled_phase1_*_quickwin.pkl'))
    if not augmented_paths:
        print("❌ ERROR: No augmented dataset found")
        return

    augmented_path = sorted(augmented_paths)[-1]
    print(f"✓ Loading augmented dataset: {augmented_path.name}")

    with open(augmented_path, 'rb') as f:
        augmented_samples = pickle.load(f)

    print(f"  Total samples: {len(augmented_samples)}")

    # Extract annual features for all samples
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

    # Extract multiscale features
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACT MULTISCALE FEATURES")
    print("=" * 80)

    # Save samples in format expected by multiscale enrichment
    temp_path = processed_dir / 'hard_val_training_quickwin_multiscale_features.pkl'
    with open(temp_path, 'wb') as f:
        pickle.dump(valid_samples, f)

    print(f"\n✓ Saved temp file for multiscale extraction")
    print(f"\nEnriching with multiscale features...")

    enriched_samples = enrich_dataset_with_multiscale_features('training_quickwin_multiscale', config)

    # Save enriched multiscale dataset
    multiscale_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale_quickwin.pkl'
    multiscale_data = {
        'data': enriched_samples,
        'metadata': {
            'n_samples': len(enriched_samples),
            'timestamp': datetime.now().isoformat()
        }
    }

    with open(multiscale_path, 'wb') as f:
        pickle.dump(multiscale_data, f)

    print(f"\n✓ Saved multiscale features: {multiscale_path.name}")

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # Re-train Random Forest
    print("\n" + "=" * 80)
    print("STEP 3: RE-TRAIN RANDOM FOREST")
    print("=" * 80)

    print("\nTraining Random Forest with augmented dataset...")
    print("This will use the existing training script with updated data paths.")
    print("\nRun: uv run python src/walk/11_train_random_forest.py")
    print("\nNote: You'll need to update the paths in that script to use:")
    print(f"  - {features_path.name}")
    print(f"  - {multiscale_path.name}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Augmented dataset: {len(augmented_samples)} samples")
    print(f"✓ Annual features extracted: {len(X)} samples")
    print(f"✓ Multiscale features extracted: {len(enriched_samples)} samples")
    print(f"\nNext: Re-train and evaluate to measure improvement on edge_cases")
    print(f"Baseline: 0.583 ROC-AUC → Target: 0.65-0.70 ROC-AUC")


if __name__ == '__main__':
    main()
