"""
Diagnostic Helper Functions

DEPRECATED: This module is deprecated. Use src/walk/utils/feature_extraction instead.

Shared functions for feature extraction and analysis across
diagnostic and evaluation scripts.

IMPORTANT: AlphaEarth provides annual embeddings, not quarterly.
All "quarterly" features extracted previously were redundant/zero.
"""

import warnings
import numpy as np

# Import from consolidated module
from .utils.feature_extraction import extract_annual_features as _extract_annual_features


def extract_dual_year_features(client, sample: dict) -> np.ndarray:
    """
    DEPRECATED: Use src.walk.utils.extract_annual_features() instead.

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
    warnings.warn(
        "diagnostic_helpers.extract_dual_year_features() is deprecated. "
        "Use src.walk.utils.extract_annual_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Extract year from sample and delegate to consolidated module
    year = sample.get('year')
    if year is None:
        return None

    return _extract_annual_features(client, sample, year)


# Feature names for interpretation
FEATURE_NAMES = [
    'delta_1yr',        # Recent annual change (Y to Y-1)
    'delta_2yr',        # Historical annual change (Y-1 to Y-2)
    'acceleration'      # Change in change rate
]
