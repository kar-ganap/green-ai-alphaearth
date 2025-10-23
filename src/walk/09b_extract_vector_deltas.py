"""
Extract Raw Annual Vector Deltas for Training

Instead of computing magnitude-only features (3D), this script extracts
the full vector differences between annual embeddings (128D):

- delta_1yr_vec: emb(Y) - emb(Y-1) [64D]
- delta_2yr_vec: emb(Y-1) - emb(Y-2) [64D]

Total: 128D temporal features (preserves directional information)

Usage:
    uv run python src/walk/09b_extract_vector_deltas.py
"""

import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def extract_vector_deltas(client, sample: dict) -> np.ndarray:
    """
    Extract full vector delta features (128D) for a sample.

    Features:
    - delta_1yr_vec: emb(Y) - emb(Y-1) [64D]
    - delta_2yr_vec: emb(Y-1) - emb(Y-2) [64D]

    Returns:
        128-dimensional feature vector or None if extraction fails
    """
    lat, lon = sample['lat'], sample['lon']
    year = sample['year']

    try:
        # Get annual embeddings
        emb_y_minus_2 = client.get_embedding(lat, lon, f"{year-2}-06-01")
        emb_y_minus_1 = client.get_embedding(lat, lon, f"{year-1}-06-01")
        emb_y = client.get_embedding(lat, lon, f"{year}-06-01")

        if emb_y_minus_2 is None or emb_y_minus_1 is None or emb_y is None:
            return None

        emb_y_minus_2 = np.array(emb_y_minus_2)
        emb_y_minus_1 = np.array(emb_y_minus_1)
        emb_y = np.array(emb_y)

        # Vector differences (preserve directional information!)
        delta_1yr_vec = emb_y - emb_y_minus_1  # 64D
        delta_2yr_vec = emb_y_minus_1 - emb_y_minus_2  # 64D

        # Concatenate into 128D feature vector
        return np.concatenate([delta_1yr_vec, delta_2yr_vec])

    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("EXTRACTING RAW ANNUAL VECTOR DELTAS (128D)")
    print("=" * 80)

    # Initialize
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load training samples
    annual_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'
    print(f"\nLoading training samples from: {annual_path}")

    with open(annual_path, 'rb') as f:
        annual_data = pickle.load(f)

    samples = annual_data['samples']
    y = annual_data['y']

    print(f"  Loaded {len(samples)} samples")
    print(f"  Clearing: {np.sum(y == 1)}")
    print(f"  Intact: {np.sum(y == 0)}")

    # Initialize Earth Engine client
    print("\nInitializing Earth Engine client...")
    ee_client = EarthEngineClient(use_cache=True)

    # Extract vector deltas
    print(f"\nExtracting 128D vector deltas for {len(samples)} samples...")
    print("  (This will take ~73 minutes for 600 samples)")

    X_vectors = []
    y_vectors = []
    valid_samples = []
    failed_count = 0

    for i, sample in enumerate(tqdm(samples, desc="Extracting")):
        vector_deltas = extract_vector_deltas(ee_client, sample)

        if vector_deltas is not None and len(vector_deltas) == 128:
            X_vectors.append(vector_deltas)
            y_vectors.append(y[i])
            valid_samples.append(sample)
        else:
            failed_count += 1

    X = np.vstack(X_vectors)
    y_final = np.array(y_vectors)

    print(f"\n✓ Extraction complete")
    print(f"  Success: {len(X)}/{len(samples)} samples ({len(X)/len(samples)*100:.1f}%)")
    print(f"  Failed: {failed_count}")
    print(f"  Feature dimension: {X.shape[1]}D")
    print(f"  Clearing: {np.sum(y_final == 1)}")
    print(f"  Intact: {np.sum(y_final == 0)}")

    # Feature names
    feature_names = []
    for i in range(64):
        feature_names.append(f'delta_1yr_dim_{i}')
    for i in range(64):
        feature_names.append(f'delta_2yr_dim_{i}')

    # Save
    output_path = processed_dir / 'walk_dataset_scaled_phase1_vector_deltas.pkl'

    result = {
        'X': X,
        'y': y_final,
        'samples': valid_samples,
        'feature_names': feature_names,
        'metadata': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_type': 'vector_deltas',
            'feature_breakdown': {
                'delta_1yr_vec': 64,
                'delta_2yr_vec': 64,
                'total': 128
            }
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✓ Saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
