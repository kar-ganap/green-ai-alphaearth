"""
Fix Spatial Leakage - Remove Violating Training Sample

CONTEXT:
- Leakage verification (Oct 19) found 8 violations across all validation sets
- All violations trace to training sample 8: [-10.6052, -68.6035]
- This sample is 5.8 km from validation samples at [-10.6089, -68.5514]
- Violates required 10km spatial buffer

SOLUTION:
- Remove training sample 8 from all training data files
- Reduces training set from 87 → 86 samples
- Re-verify to confirm 0 violations

FILES AFFECTED:
- walk_dataset_scaled_phase1.pkl
- walk_dataset_scaled_phase1_features.pkl
- walk_dataset_scaled_phase1_multiscale.pkl
- walk_dataset_scaled_phase1_sentinel2.pkl (if exists)
- walk_dataset_scaled_phase1_vector_deltas.pkl (if exists)

Usage:
    uv run python src/walk/18_fix_spatial_leakage.py
"""

import pickle
import shutil
from pathlib import Path
from datetime import datetime

from src.utils import get_config


def remove_sample_from_list(samples, index_to_remove):
    """Remove sample at index from list."""
    if index_to_remove < 0 or index_to_remove >= len(samples):
        raise ValueError(f"Index {index_to_remove} out of range for {len(samples)} samples")

    cleaned = samples[:index_to_remove] + samples[index_to_remove + 1:]
    return cleaned


def remove_sample_from_dict(data_dict, index_to_remove):
    """Remove sample at index from dataset dictionary (with X, y, samples structure)."""
    if 'samples' in data_dict:
        # Has samples list
        data_dict['samples'] = remove_sample_from_list(data_dict['samples'], index_to_remove)

    if 'data' in data_dict:
        # Has data list (used in some files)
        data_dict['data'] = remove_sample_from_list(data_dict['data'], index_to_remove)

    if 'X' in data_dict and data_dict['X'] is not None:
        # Has feature matrix
        import numpy as np
        X = data_dict['X']
        if len(X) > 0:
            mask = np.ones(len(X), dtype=bool)
            mask[index_to_remove] = False
            data_dict['X'] = X[mask]

    if 'y' in data_dict and data_dict['y'] is not None:
        # Has labels
        import numpy as np
        y = data_dict['y']
        if len(y) > 0:
            mask = np.ones(len(y), dtype=bool)
            mask[index_to_remove] = False
            data_dict['y'] = y[mask]

    return data_dict


def main():
    print("=" * 80)
    print("FIX SPATIAL LEAKAGE - REMOVE VIOLATING TRAINING SAMPLE")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Sample to remove (from verification report)
    VIOLATING_INDEX = 8
    VIOLATING_COORD = [-10.6052, -68.6035]

    print(f"\n📍 Target: Remove sample {VIOLATING_INDEX}")
    print(f"   Coordinates: {VIOLATING_COORD}")
    print(f"   Reason: 5.8 km from validation samples (violates 10km buffer)")

    # Files to fix
    training_files = [
        'walk_dataset_scaled_phase1.pkl',
        'walk_dataset_scaled_phase1_features.pkl',
        'walk_dataset_scaled_phase1_multiscale.pkl',
        'walk_dataset_scaled_phase1_sentinel2.pkl',
        'walk_dataset_scaled_phase1_vector_deltas.pkl',
    ]

    print(f"\n📂 Processing {len(training_files)} training data files...")

    backup_dir = processed_dir / 'backup_before_leakage_fix'
    backup_dir.mkdir(exist_ok=True)

    fixed_count = 0
    skipped_count = 0

    for filename in training_files:
        filepath = processed_dir / filename

        if not filepath.exists():
            print(f"\n  ⊘ {filename}")
            print(f"     File not found, skipping")
            skipped_count += 1
            continue

        print(f"\n  📄 {filename}")

        # Backup original
        backup_path = backup_dir / f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"     Creating backup: {backup_path.name}")
        shutil.copy2(filepath, backup_path)

        # Load data
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Check structure
        if isinstance(data, dict):
            # Dictionary structure (most files)
            samples_key = 'samples' if 'samples' in data else 'data' if 'data' in data else None

            if samples_key:
                original_count = len(data[samples_key])
                print(f"     Original samples: {original_count}")

                # Verify the sample we're removing
                sample_to_remove = data[samples_key][VIOLATING_INDEX]
                actual_coord = [sample_to_remove.get('lat'), sample_to_remove.get('lon')]

                # Check coordinates match (within tolerance for float precision)
                if abs(actual_coord[0] - VIOLATING_COORD[0]) < 0.001 and \
                   abs(actual_coord[1] - VIOLATING_COORD[1]) < 0.001:
                    print(f"     ✓ Verified coordinates match: {actual_coord}")
                else:
                    print(f"     ⚠ WARNING: Coordinates don't match!")
                    print(f"       Expected: {VIOLATING_COORD}")
                    print(f"       Found: {actual_coord}")
                    response = input("     Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        print("     Skipping file")
                        skipped_count += 1
                        continue

                # Remove sample
                data = remove_sample_from_dict(data, VIOLATING_INDEX)
                new_count = len(data[samples_key])

                print(f"     Cleaned samples: {new_count}")
                print(f"     Removed: {original_count - new_count}")

            else:
                print(f"     ⚠ Unknown structure, skipping")
                skipped_count += 1
                continue

        elif isinstance(data, list):
            # List structure (rare)
            original_count = len(data)
            print(f"     Original samples: {original_count}")

            data = remove_sample_from_list(data, VIOLATING_INDEX)
            new_count = len(data)

            print(f"     Cleaned samples: {new_count}")
            print(f"     Removed: {original_count - new_count}")
        else:
            print(f"     ⚠ Unexpected structure type: {type(data)}, skipping")
            skipped_count += 1
            continue

        # Save cleaned data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"     ✓ Saved cleaned file")
        fixed_count += 1

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Fixed: {fixed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Backups: {backup_dir}")

    if fixed_count > 0:
        print(f"\n✓ Training sample {VIOLATING_INDEX} removed from all files")
        print(f"  New training set size: 86 samples (was 87)")

        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print(f"{'='*80}")
        print(f"\n1. Run verification to confirm 0 violations:")
        print(f"   uv run python src/walk/data_leakage_verification.py")
        print(f"\n2. Re-train Random Forest on clean 86-sample dataset:")
        print(f"   uv run python src/walk/11_train_random_forest.py")
        print(f"\n3. Document clean baseline performance")
    else:
        print(f"\n⚠ No files were fixed!")

    print()


if __name__ == '__main__':
    main()
