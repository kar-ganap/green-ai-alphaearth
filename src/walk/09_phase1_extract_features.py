"""
Phase 1: Extract Features for Scaled Dataset

CORRECTED: Extracts annual delta features (3 dimensions) for all 600 Phase 1 samples.

IMPORTANT: AlphaEarth provides ONE embedding per year, not quarterly.
Previous quarterly extraction was buggy - all quarters returned identical embeddings.

Features (CORRECTED):
- delta_1yr: ||emb(Y) - emb(Y-1)|| - Recent annual change
- delta_2yr: ||emb(Y-1) - emb(Y-2)|| - Historical annual change
- acceleration: delta_1yr - delta_2yr - Is change accelerating?

Usage:
    uv run python src/walk/09_phase1_extract_features.py
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features


def main():
    """Extract features for Phase 1 scaled dataset."""

    print("=" * 80)
    print("PHASE 1: FEATURE EXTRACTION")
    print("=" * 80)

    # Initialize
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load Phase 1 dataset
    dataset_path = processed_dir / 'walk_dataset_scaled_phase1.pkl'

    print(f"\nLoading dataset from: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    samples = dataset['data']
    metadata = dataset['metadata']

    print(f"  Loaded {len(samples)} samples")
    print(f"  Clearing: {metadata['clearing_actual']}")
    print(f"  Intact: {metadata['intact_actual']}")

    # Extract features
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURES")
    print("=" * 80)

    X_list = []
    y_list = []
    valid_samples = []
    failed_indices = []

    for idx, sample in enumerate(tqdm(samples, desc="Extracting features")):
        features = extract_dual_year_features(client, sample)

        if features is not None:
            X_list.append(features)
            y_list.append(sample['label'])
            valid_samples.append(sample)
        else:
            failed_indices.append(idx)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n✓ Successfully extracted: {len(X)}/{len(samples)} samples")
    print(f"  Failed: {len(failed_indices)} samples ({len(failed_indices)/len(samples)*100:.1f}%)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  Clearing (label=1): {np.sum(y == 1)}")
    print(f"  Intact (label=0): {np.sum(y == 0)}")

    print(f"\nFeature ranges:")
    for i in range(X.shape[1]):
        feat_min, feat_max = np.min(X[:, i]), np.max(X[:, i])
        feat_mean, feat_std = np.mean(X[:, i]), np.std(X[:, i])
        print(f"  Feature {i}: [{feat_min:.3f}, {feat_max:.3f}], "
              f"mean={feat_mean:.3f}, std={feat_std:.3f}")

    # Save features
    print("\n" + "=" * 80)
    print("SAVING FEATURES")
    print("=" * 80)

    output_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'

    output_data = {
        'X': X,
        'y': y,
        'samples': valid_samples,
        'failed_indices': failed_indices,
        'metadata': {
            **metadata,
            'feature_extraction': {
                'timestamp': datetime.now().isoformat(),
                'n_total': len(samples),
                'n_extracted': len(X),
                'n_failed': len(failed_indices),
                'success_rate': len(X) / len(samples) * 100,
                'feature_dim': X.shape[1]
            }
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Success rate: {len(X)/len(samples)*100:.1f}%")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Train scaled model on Phase 1 features (CORRECTED: 3D annual features)")
    print("2. Evaluate on all 4 validation sets")
    print("3. Compare to baseline:")
    print("   - risk_ranking: 0.850 → ?")
    print("   - rapid_response: 0.824 → ?")
    print("   - comprehensive: 0.758 → ?")
    print("   - edge_cases: 0.583 → 0.70+ (target)")
    print("\nNOTE: With corrected annual features, we expect:")
    print("  - Match temporal generalization (~0.95 ROC-AUC) for standard clearings")
    print("  - Detection framing (0-12 month lag), not prediction (4-6 month lead)")
    print("  - Edge cases may still need spatial/contextual features (Phase 2)")


if __name__ == '__main__':
    main()
