"""
Annual Feature Extraction for AlphaEarth

DEPRECATED: This module is deprecated. Use src/walk/utils/feature_extraction instead.

IMPORTANT: AlphaEarth provides ONE embedding per year, not quarterly.
This module extracts annual features that match AlphaEarth's actual capabilities.

The 0.971 ROC-AUC from temporal generalization came from a single feature:
||emb(Y) - emb(Y-1)|| - the annual year-over-year change magnitude.

All "quarterly" features were redundant/zero because AlphaEarth embeddings
don't change within a year.
"""

import warnings
import numpy as np

# Import from consolidated module
from .utils.feature_extraction import extract_annual_features as _extract_annual_features


def extract_annual_features(client, sample: dict, year: int) -> np.ndarray:
    """
    DEPRECATED: Use src.walk.utils.extract_annual_features() instead.

    Extract annual features from AlphaEarth embeddings.

    Uses 3 annual embeddings:
    - Y-2: Two years before clearing
    - Y-1: One year before clearing (baseline)
    - Y: Year of clearing

    Computes simple annual deltas:
    - delta_1yr = ||emb(Y) - emb(Y-1)||: Recent change
    - delta_2yr = ||emb(Y-1) - emb(Y-2)||: Historical change
    - acceleration = delta_1yr - delta_2yr: Is change speeding up?

    Args:
        client: EarthEngineClient
        sample: Dict with 'lat', 'lon'
        year: Year of clearing

    Returns:
        3-dimensional feature vector or None if extraction fails
        [delta_1yr, delta_2yr, acceleration]
    """
    warnings.warn(
        "annual_features.extract_annual_features() is deprecated. "
        "Use src.walk.utils.extract_annual_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _extract_annual_features(client, sample, year)


def extract_annual_features_extended(client, sample: dict, year: int) -> np.ndarray:
    """
    Extended annual features with directional information.

    Adds:
    - Directional consistency: Is change in consistent direction?
    - Baseline magnitude: ||emb(Y-1)||
    - Current magnitude: ||emb(Y)||

    Args:
        client: EarthEngineClient
        sample: Dict with 'lat', 'lon'
        year: Year of clearing

    Returns:
        7-dimensional feature vector or None
        [delta_1yr, delta_2yr, acceleration,
         baseline_mag, current_mag, direction_consistency, total_change]
    """
    try:
        lat, lon = sample['lat'], sample['lon']

        # Get annual embeddings
        emb_y_minus_2 = client.get_embedding(lat, lon, f"{year-2}-06-01")
        emb_y_minus_1 = client.get_embedding(lat, lon, f"{year-1}-06-01")
        emb_y = client.get_embedding(lat, lon, f"{year}-06-01")

        if emb_y_minus_2 is None or emb_y_minus_1 is None or emb_y is None:
            return None

        emb_y_minus_2 = np.array(emb_y_minus_2)
        emb_y_minus_1 = np.array(emb_y_minus_1)
        emb_y = np.array(emb_y)

        # Delta magnitudes
        delta_1yr = np.linalg.norm(emb_y - emb_y_minus_1)
        delta_2yr = np.linalg.norm(emb_y_minus_1 - emb_y_minus_2)
        acceleration = delta_1yr - delta_2yr

        # Embedding magnitudes
        baseline_mag = np.linalg.norm(emb_y_minus_1)
        current_mag = np.linalg.norm(emb_y)

        # Directional consistency (cosine similarity of delta vectors)
        delta_vec_1yr = emb_y - emb_y_minus_1
        delta_vec_2yr = emb_y_minus_1 - emb_y_minus_2

        norm_1 = np.linalg.norm(delta_vec_1yr)
        norm_2 = np.linalg.norm(delta_vec_2yr)

        if norm_1 > 1e-8 and norm_2 > 1e-8:
            direction_consistency = np.dot(delta_vec_1yr, delta_vec_2yr) / (norm_1 * norm_2)
        else:
            direction_consistency = 0.0

        # Total change from Y-2 to Y
        total_change = np.linalg.norm(emb_y - emb_y_minus_2)

        return np.array([
            delta_1yr,
            delta_2yr,
            acceleration,
            baseline_mag,
            current_mag,
            direction_consistency,
            total_change
        ])

    except Exception as e:
        return None


FEATURE_NAMES_SIMPLE = [
    'delta_1yr',           # Recent annual change (Y to Y-1)
    'delta_2yr',           # Historical annual change (Y-1 to Y-2)
    'acceleration'         # Change in change rate
]

FEATURE_NAMES_EXTENDED = [
    'delta_1yr',
    'delta_2yr',
    'acceleration',
    'baseline_magnitude',  # ||emb(Y-1)||
    'current_magnitude',   # ||emb(Y)||
    'direction_consistency',  # Cosine similarity of delta vectors
    'total_change'         # ||emb(Y) - emb(Y-2)||
]
