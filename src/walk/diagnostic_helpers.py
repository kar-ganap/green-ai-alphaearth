"""
Diagnostic Helper Functions

Shared functions for feature extraction and analysis across
diagnostic and evaluation scripts.

IMPORTANT: AlphaEarth provides annual embeddings, not quarterly.
All "quarterly" features extracted previously were redundant/zero.
"""

import numpy as np


def extract_dual_year_features(client, sample: dict) -> np.ndarray:
    """
    Extract annual delta features for a sample.

    CORRECTED: Uses annual embeddings (AlphaEarth limitation).
    Previous "quarterly" approach was buggy - all quarters within
    a year return identical embeddings.

    Uses 3 annual snapshots:
    - Y-2: Two years before clearing
    - Y-1: One year before clearing (baseline)
    - Y: Year of clearing

    Features:
    1. delta_1yr: ||emb(Y) - emb(Y-1)||
    2. delta_2yr: ||emb(Y-1) - emb(Y-2)||
    3. acceleration: delta_1yr - delta_2yr

    Returns:
        3-dimensional feature vector or None if extraction fails
    """
    lat, lon = sample['lat'], sample['lon']
    year = sample['year']

    try:
        # Get annual embeddings (any date within year works - they're all the same!)
        emb_y_minus_2 = client.get_embedding(lat, lon, f"{year-2}-06-01")
        emb_y_minus_1 = client.get_embedding(lat, lon, f"{year-1}-06-01")
        emb_y = client.get_embedding(lat, lon, f"{year}-06-01")

        if emb_y_minus_2 is None or emb_y_minus_1 is None or emb_y is None:
            return None

        emb_y_minus_2 = np.array(emb_y_minus_2)
        emb_y_minus_1 = np.array(emb_y_minus_1)
        emb_y = np.array(emb_y)

        # Annual deltas
        delta_1yr = np.linalg.norm(emb_y - emb_y_minus_1)
        delta_2yr = np.linalg.norm(emb_y_minus_1 - emb_y_minus_2)

        # Acceleration (is recent change faster than historical?)
        acceleration = delta_1yr - delta_2yr

        return np.array([delta_1yr, delta_2yr, acceleration])

    except Exception as e:
        return None


# Feature names for interpretation
FEATURE_NAMES = [
    'delta_1yr',        # Recent annual change (Y to Y-1)
    'delta_2yr',        # Historical annual change (Y-1 to Y-2)
    'acceleration'      # Change in change rate
]
