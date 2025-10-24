"""
Canonical Feature Extraction Module

Single source of truth for all feature extraction in the WALK phase.
This module consolidates 30+ scattered implementations into one place.

Production 70D Feature Vector:
- 3D Annual features (temporal deltas)
- 66D Coarse multiscale features (landscape context)
- 1D Year feature (normalized)

Based on production implementation from:
- src/walk/47_extract_hard_validation_features.py (Oct 23, 2025)
- src/walk/diagnostic_helpers.py
- src/walk/08_multiscale_embeddings.py

IMPORTANT: AlphaEarth provides annual embeddings, not quarterly.
All quarterly features were redundant (same embedding within a year).
"""

import numpy as np
import ee
from typing import Dict, Optional, Tuple


# ============================================================================
# ANNUAL FEATURES (3D)
# ============================================================================

def extract_annual_features(client, sample: dict, year: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Extract annual temporal delta features from AlphaEarth embeddings.

    Uses 3 annual snapshots:
    - Y-2: Two years before clearing
    - Y-1: One year before clearing (baseline)
    - Y: Year of clearing

    Computes:
    1. delta_1yr = ||emb(Y) - emb(Y-1)||: Recent change magnitude
    2. delta_2yr = ||emb(Y-1) - emb(Y-2)||: Historical change magnitude
    3. acceleration = delta_1yr - delta_2yr: Is change accelerating?

    Args:
        client: EarthEngineClient instance
        sample: Dict with 'lat', 'lon', and optionally 'year'
        year: Year of clearing (if not in sample)

    Returns:
        3D numpy array [delta_1yr, delta_2yr, acceleration] or None if extraction fails

    Example:
        >>> features = extract_annual_features(client, {'lat': -3.5, 'lon': -62.0, 'year': 2023})
        >>> print(features)  # [0.12, 0.08, 0.04]
    """
    try:
        lat = sample['lat']
        lon = sample['lon']
        year = year or sample.get('year')

        if year is None:
            return None

        # Get annual embeddings (mid-year dates - all dates within a year are identical)
        emb_y_minus_2 = client.get_embedding(lat, lon, f"{year-2}-06-01")
        emb_y_minus_1 = client.get_embedding(lat, lon, f"{year-1}-06-01")
        emb_y = client.get_embedding(lat, lon, f"{year}-06-01")

        if emb_y_minus_2 is None or emb_y_minus_1 is None or emb_y is None:
            return None

        # Convert to numpy arrays
        emb_y_minus_2 = np.array(emb_y_minus_2)
        emb_y_minus_1 = np.array(emb_y_minus_1)
        emb_y = np.array(emb_y)

        # Compute annual deltas
        delta_1yr = np.linalg.norm(emb_y - emb_y_minus_1)
        delta_2yr = np.linalg.norm(emb_y_minus_1 - emb_y_minus_2)

        # Acceleration: is recent change faster than historical?
        acceleration = delta_1yr - delta_2yr

        return np.array([delta_1yr, delta_2yr, acceleration])

    except Exception as e:
        return None


# ============================================================================
# COARSE MULTISCALE FEATURES (66D)
# ============================================================================

def extract_coarse_multiscale_features(
    client,
    lat: float,
    lon: float,
    date: str,
    scale: int = 100
) -> Optional[Dict[str, float]]:
    """
    Extract coarse-scale landscape context features.

    Samples a 3x3 grid of AlphaEarth embeddings around the target location
    at 100m spacing to capture landscape-level patterns.

    Returns 66D features:
    - 64D: Mean embedding across the 3x3 grid (landscape average)
    - 1D: Heterogeneity (variance across region)
    - 1D: Range (diversity across region)

    Args:
        client: EarthEngineClient instance
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)
        scale: Resolution in meters (default: 100m)

    Returns:
        Dict with 66 features:
        - coarse_emb_0 through coarse_emb_63: Mean embedding
        - coarse_heterogeneity: Landscape variance
        - coarse_range: Landscape diversity

    Example:
        >>> features = extract_coarse_multiscale_features(client, -3.5, -62.0, "2023-06-01")
        >>> len(features)  # 66
        >>> 'coarse_emb_0' in features  # True
        >>> 'coarse_heterogeneity' in features  # True
    """
    try:
        # Sample 3x3 grid around center at 100m spacing
        step = scale / 111320  # Convert meters to degrees

        embeddings = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                try:
                    emb = client.get_embedding(
                        lat + i * step,
                        lon + j * step,
                        date
                    )
                    if emb is not None:
                        embeddings.append(emb)
                except Exception:
                    continue

        if len(embeddings) == 0:
            return None

        embeddings = np.array(embeddings)

        features = {}

        # Mean embedding (landscape average)
        mean_emb = np.mean(embeddings, axis=0)
        for i, val in enumerate(mean_emb):
            features[f'coarse_emb_{i}'] = float(val)

        # Landscape heterogeneity (variance across region)
        variance = np.var(embeddings, axis=0)
        features['coarse_heterogeneity'] = float(np.mean(variance))

        # Landscape range (max - min per dimension)
        range_vals = np.max(embeddings, axis=0) - np.min(embeddings, axis=0)
        features['coarse_range'] = float(np.mean(range_vals))

        return features

    except Exception as e:
        return None


# ============================================================================
# COMPLETE 70D FEATURE EXTRACTION
# ============================================================================

def extract_70d_features(
    client,
    sample: dict,
    year: Optional[int] = None
) -> Optional[Tuple[np.ndarray, Dict[str, float], float]]:
    """
    Extract complete 70D feature vector for a sample.

    This is the canonical production feature extraction used by all models.

    Returns 70D features:
    - 3D Annual features (temporal deltas)
    - 66D Coarse multiscale features (landscape context)
    - 1D Year feature (normalized)

    Args:
        client: EarthEngineClient instance
        sample: Dict with 'lat', 'lon', and optionally 'year'
        year: Year of clearing (if not in sample)

    Returns:
        Tuple of (annual_features, multiscale_features, year_feature) or None
        - annual_features: 3D numpy array
        - multiscale_features: Dict with 66 features
        - year_feature: Float (normalized year)

    Example:
        >>> result = extract_70d_features(client, {'lat': -3.5, 'lon': -62.0, 'year': 2023})
        >>> if result:
        ...     annual, multiscale, year_feat = result
        ...     print(f"Annual: {annual.shape}, Multiscale: {len(multiscale)}, Year: {year_feat}")
        Annual: (3,), Multiscale: 66, Year: 0.75
    """
    try:
        lat = sample['lat']
        lon = sample['lon']
        year = year or sample.get('year')

        if year is None:
            return None

        # 1. Extract annual features (3D)
        annual_features = extract_annual_features(client, sample, year)
        if annual_features is None:
            return None

        # 2. Extract coarse multiscale features (66D)
        date = f"{year}-06-01"  # Mid-year date
        multiscale_features = extract_coarse_multiscale_features(client, lat, lon, date)
        if multiscale_features is None:
            return None

        # 3. Normalized year feature (0 to 1 for 2020-2024)
        year_feature = (year - 2020) / 4.0

        return (annual_features, multiscale_features, year_feature)

    except Exception as e:
        return None


def enrich_sample_with_features(
    client,
    sample: dict,
    year: Optional[int] = None
) -> Optional[dict]:
    """
    Enrich a sample dict with extracted features.

    Adds three keys to the sample:
    - 'annual_features': 3D numpy array
    - 'multiscale_features': Dict with 66 features
    - 'year_feature': Float

    Args:
        client: EarthEngineClient instance
        sample: Dict with 'lat', 'lon', and optionally 'year'
        year: Year of clearing (if not in sample)

    Returns:
        Enriched sample dict or None if extraction fails

    Example:
        >>> enriched = enrich_sample_with_features(client, {'lat': -3.5, 'lon': -62.0, 'year': 2023})
        >>> 'annual_features' in enriched
        True
        >>> 'multiscale_features' in enriched
        True
    """
    result = extract_70d_features(client, sample, year)
    if result is None:
        return None

    annual_features, multiscale_features, year_feature = result

    enriched_sample = sample.copy()
    enriched_sample['annual_features'] = annual_features
    enriched_sample['multiscale_features'] = multiscale_features
    enriched_sample['year_feature'] = year_feature

    return enriched_sample


def features_to_array(
    annual_features: np.ndarray,
    multiscale_features: Dict[str, float],
    year_feature: float
) -> np.ndarray:
    """
    Convert extracted features to a single 70D numpy array for model input.

    Args:
        annual_features: 3D numpy array
        multiscale_features: Dict with 66 coarse features
        year_feature: Float (normalized year)

    Returns:
        70D numpy array ready for model prediction

    Example:
        >>> annual = np.array([0.12, 0.08, 0.04])
        >>> multiscale = {f'coarse_emb_{i}': 0.5 for i in range(64)}
        >>> multiscale['coarse_heterogeneity'] = 0.02
        >>> multiscale['coarse_range'] = 0.15
        >>> year_feat = 0.75
        >>> X = features_to_array(annual, multiscale, year_feat)
        >>> X.shape
        (70,)
    """
    # Extract coarse features in correct order
    coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + \
                          ['coarse_heterogeneity', 'coarse_range']
    coarse_values = [multiscale.get(k, 0.0) for k in coarse_feature_names]
    multiscale_array = np.array(coarse_values)

    # Combine: 3D + 66D + 1D = 70D
    return np.concatenate([annual_features, multiscale_array, [year_feature]])


# ============================================================================
# FEATURE NAMES (FOR INTERPRETABILITY)
# ============================================================================

FEATURE_NAMES_ANNUAL = [
    'delta_1yr',        # Recent annual change (Y to Y-1)
    'delta_2yr',        # Historical annual change (Y-1 to Y-2)
    'acceleration'      # Change in change rate
]

FEATURE_NAMES_COARSE = (
    [f'coarse_emb_{i}' for i in range(64)] +  # 64D mean embedding
    ['coarse_heterogeneity', 'coarse_range']   # 2D landscape stats
)

FEATURE_NAMES_70D = FEATURE_NAMES_ANNUAL + FEATURE_NAMES_COARSE + ['year_normalized']
