"""
WALK Phase - Spatial Feature Extraction for Rapid Response

Extracts spatial context features to improve small-scale clearing detection.

Based on hard validation results showing:
- 100% miss rate on clearings < 1 ha
- 63% miss rate on rapid response cases
- Need: Neighborhood context, edge detection, texture features

Usage:
    uv run python src/walk/01d_extract_spatial_features.py --set rapid_response
    uv run python src/walk/01d_extract_spatial_features.py --set all
"""

import argparse
import pickle
from pathlib import Path

import ee
import numpy as np
from scipy.ndimage import sobel
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def extract_neighborhood_features(client, lat, lon, date, radius_m=500):
    """
    Extract neighborhood statistics around a location.

    For small-scale clearing detection, we need to understand:
    - Is this pixel different from its neighbors?
    - Is the neighborhood heterogeneous (patchy clearing)?
    - What's the gradient/edge strength?

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string (YYYY-MM-DD)
        radius_m: Neighborhood radius in meters (default: 500m)

    Returns:
        Dict with neighborhood features
    """
    try:
        # Get center pixel embedding
        center_emb = client.get_embedding(lat, lon, date)

        # Sample neighbors in a grid around the center
        # Using 8 neighbors in cardinal + diagonal directions
        offsets_m = radius_m / 111320  # Convert meters to degrees (approx)

        neighbor_locs = [
            (lat + offsets_m, lon),           # N
            (lat + offsets_m, lon + offsets_m),  # NE
            (lat, lon + offsets_m),           # E
            (lat - offsets_m, lon + offsets_m),  # SE
            (lat - offsets_m, lon),           # S
            (lat - offsets_m, lon - offsets_m),  # SW
            (lat, lon - offsets_m),           # W
            (lat + offsets_m, lon - offsets_m),  # NW
        ]

        # Get embeddings for all neighbors
        neighbor_embs = []
        for nlat, nlon in neighbor_locs:
            try:
                emb = client.get_embedding(nlat, nlon, date)
                neighbor_embs.append(emb)
            except Exception:
                # Skip neighbors that fail (e.g., outside image bounds)
                continue

        if len(neighbor_embs) == 0:
            return None

        neighbor_embs = np.array(neighbor_embs)

        # Compute neighborhood statistics
        features = {}

        # 1. Center-to-neighbor distances
        distances = np.linalg.norm(neighbor_embs - center_emb, axis=1)
        features['neighbor_mean_distance'] = float(np.mean(distances))
        features['neighbor_std_distance'] = float(np.std(distances))
        features['neighbor_max_distance'] = float(np.max(distances))

        # 2. Neighborhood heterogeneity (variance among neighbors)
        # High heterogeneity suggests patchy/mixed land cover
        neighbor_variance = np.var(neighbor_embs, axis=0)  # Variance per dimension
        features['neighbor_heterogeneity'] = float(np.mean(neighbor_variance))

        # 3. Gradient strength (how different is center from average neighbor?)
        avg_neighbor = np.mean(neighbor_embs, axis=0)
        gradient = np.linalg.norm(center_emb - avg_neighbor)
        features['gradient_strength'] = float(gradient)

        # 4. Edge likelihood (is center an outlier from neighbors?)
        # Z-score of center distance relative to neighbor-to-neighbor distances
        neighbor_to_neighbor = []
        for i in range(len(neighbor_embs)):
            for j in range(i+1, len(neighbor_embs)):
                dist = np.linalg.norm(neighbor_embs[i] - neighbor_embs[j])
                neighbor_to_neighbor.append(dist)

        if len(neighbor_to_neighbor) > 0:
            center_dist_mean = np.mean(distances)
            n2n_mean = np.mean(neighbor_to_neighbor)
            n2n_std = np.std(neighbor_to_neighbor)

            if n2n_std > 0:
                edge_score = (center_dist_mean - n2n_mean) / n2n_std
                features['edge_score'] = float(edge_score)
            else:
                features['edge_score'] = 0.0
        else:
            features['edge_score'] = 0.0

        return features

    except Exception as e:
        print(f"    ✗ Neighborhood extraction failed: {e}")
        return None


def extract_texture_features(client, lat, lon, date, window_size=5):
    """
    Extract texture features using GLCM (Gray Level Co-occurrence Matrix).

    Texture helps detect patterns in small clearings:
    - Homogeneous: Intact forest or large clearing
    - Heterogeneous: Fragmented clearing, edge areas

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string
        window_size: Size of window for GLCM (default: 5 = 150m)

    Returns:
        Dict with texture features
    """
    try:
        # Sample a small grid around the center
        step = 30 / 111320  # 30m pixel size in degrees
        half_window = window_size // 2

        grid = []
        for i in range(-half_window, half_window + 1):
            row = []
            for j in range(-half_window, half_window + 1):
                try:
                    emb = client.get_embedding(
                        lat + i * step,
                        lon + j * step,
                        date
                    )
                    # Use first principal component as intensity
                    intensity = emb[0]
                    row.append(intensity)
                except Exception:
                    row.append(np.nan)
            grid.append(row)

        grid = np.array(grid)

        # Check if we have enough valid pixels
        valid_pixels = np.sum(~np.isnan(grid))
        if valid_pixels < (window_size * window_size * 0.5):
            return None

        # Interpolate missing values
        if np.any(np.isnan(grid)):
            mask = ~np.isnan(grid)
            grid[~mask] = np.nanmean(grid)

        # Normalize to 0-255 for GLCM
        grid_min = np.min(grid)
        grid_max = np.max(grid)
        if grid_max > grid_min:
            grid_normalized = ((grid - grid_min) / (grid_max - grid_min) * 255).astype(np.uint8)
        else:
            grid_normalized = np.zeros_like(grid, dtype=np.uint8)

        # Compute GLCM
        # distances=[1] means look at adjacent pixels
        # angles=[0, np.pi/4, np.pi/2, 3*np.pi/4] means 4 directions
        glcm = graycomatrix(
            grid_normalized,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract texture properties
        features = {}

        # Contrast: Local variations (high for edges)
        contrast = graycoprops(glcm, 'contrast')
        features['texture_contrast'] = float(np.mean(contrast))

        # Dissimilarity: How different are neighboring pixels
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        features['texture_dissimilarity'] = float(np.mean(dissimilarity))

        # Homogeneity: How uniform is the texture (low for fragmented areas)
        homogeneity = graycoprops(glcm, 'homogeneity')
        features['texture_homogeneity'] = float(np.mean(homogeneity))

        # Energy: Uniformity (low for complex patterns)
        energy = graycoprops(glcm, 'energy')
        features['texture_energy'] = float(np.mean(energy))

        # Correlation: Linear dependencies
        correlation = graycoprops(glcm, 'correlation')
        features['texture_correlation'] = float(np.mean(correlation))

        return features

    except Exception as e:
        print(f"    ✗ Texture extraction failed: {e}")
        return None


def extract_edge_features(client, lat, lon, date, window_size=5):
    """
    Extract edge detection features using Sobel gradients.

    Edge features help detect:
    - Clearing boundaries
    - Forest fragmentation
    - Transition zones

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        date: Date string
        window_size: Size of window (default: 5 = 150m)

    Returns:
        Dict with edge features
    """
    try:
        # Sample grid (same as texture features)
        step = 30 / 111320
        half_window = window_size // 2

        grid = []
        for i in range(-half_window, half_window + 1):
            row = []
            for j in range(-half_window, half_window + 1):
                try:
                    emb = client.get_embedding(
                        lat + i * step,
                        lon + j * step,
                        date
                    )
                    intensity = emb[0]
                    row.append(intensity)
                except Exception:
                    row.append(np.nan)
            grid.append(row)

        grid = np.array(grid)

        # Check validity
        valid_pixels = np.sum(~np.isnan(grid))
        if valid_pixels < (window_size * window_size * 0.5):
            return None

        # Interpolate missing
        if np.any(np.isnan(grid)):
            mask = ~np.isnan(grid)
            grid[~mask] = np.nanmean(grid)

        # Apply Sobel filters
        sobel_h = sobel(grid, axis=0)  # Horizontal edges
        sobel_v = sobel(grid, axis=1)  # Vertical edges

        # Edge magnitude
        edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)

        features = {}

        # Edge statistics
        features['edge_mean'] = float(np.mean(edge_magnitude))
        features['edge_std'] = float(np.std(edge_magnitude))
        features['edge_max'] = float(np.max(edge_magnitude))

        # Edge density (fraction of high-gradient pixels)
        threshold = np.mean(edge_magnitude) + np.std(edge_magnitude)
        edge_density = np.sum(edge_magnitude > threshold) / edge_magnitude.size
        features['edge_density'] = float(edge_density)

        return features

    except Exception as e:
        print(f"    ✗ Edge extraction failed: {e}")
        return None


def extract_spatial_features_for_sample(client, sample, timepoint='Q4'):
    """
    Extract all spatial features for a single sample.

    Args:
        client: EarthEngineClient
        sample: Sample dict with lat, lon, year
        timepoint: Which timepoint to extract spatial features for (default: Q4)

    Returns:
        Updated sample dict with 'spatial_features' key
    """
    lat = sample['lat']
    lon = sample['lon']
    year = sample.get('year', 2021)

    # Use Q4 date (most recent before clearing)
    if timepoint == 'Q4':
        date = f'{year}-09-01'
    elif timepoint == 'Q3':
        date = f'{year}-06-01'
    elif timepoint == 'Q2':
        date = f'{year}-03-01'
    elif timepoint == 'Q1':
        date = f'{year-1}-06-01'
    else:
        date = f'{year}-09-01'

    spatial_features = {}

    # Extract neighborhood features
    neighbor_feats = extract_neighborhood_features(client, lat, lon, date)
    if neighbor_feats is not None:
        spatial_features.update(neighbor_feats)

    # Extract texture features
    texture_feats = extract_texture_features(client, lat, lon, date)
    if texture_feats is not None:
        spatial_features.update(texture_feats)

    # Extract edge features
    edge_feats = extract_edge_features(client, lat, lon, date)
    if edge_feats is not None:
        spatial_features.update(edge_feats)

    if len(spatial_features) == 0:
        return None

    sample['spatial_features'] = spatial_features

    return sample


def enrich_dataset_with_spatial_features(set_name, config):
    """
    Enrich a validation set with spatial features.

    Args:
        set_name: Name of validation set (e.g., 'rapid_response')
        config: Config object

    Returns:
        Enriched samples with spatial features
    """
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    # Load enriched dataset (with temporal features)
    input_file = processed_dir / f"hard_val_{set_name}_features.pkl"
    output_file = processed_dir / f"hard_val_{set_name}_spatial.pkl"

    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return None

    print(f"\n{'='*80}")
    print(f"EXTRACTING SPATIAL FEATURES FOR: {set_name}")
    print(f"{'='*80}\n")

    # Load samples
    with open(input_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    n_clearing = sum(1 for s in samples if s.get('label', 0) == 1)
    n_intact = len(samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}\n")

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Extract spatial features for each sample
    enriched_samples = []
    failed_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Extracting spatial features")):
        try:
            enriched_sample = extract_spatial_features_for_sample(client, sample)
            if enriched_sample is not None:
                enriched_samples.append(enriched_sample)
            else:
                failed_samples.append(i)
        except Exception as e:
            print(f"\n  ✗ Failed on sample {i}: {e}")
            failed_samples.append(i)

    print(f"\n✓ Extracted spatial features for {len(enriched_samples)}/{len(samples)} samples")
    if failed_samples:
        print(f"  ✗ Failed: {len(failed_samples)} samples")
        print(f"    Indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")

    # Save enriched dataset
    with open(output_file, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to {output_file}")

    return enriched_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set',
        type=str,
        default='rapid_response',
        choices=['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases', 'all'],
        help='Which validation set to extract spatial features for'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SPATIAL FEATURE EXTRACTION FOR HARD VALIDATION SETS")
    print("=" * 80)

    config = get_config()

    if args.set == 'all':
        sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']
    else:
        sets = [args.set]

    for set_name in sets:
        enrich_dataset_with_spatial_features(set_name, config)

    print("\n" + "=" * 80)
    print("SPATIAL FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext: Train model with spatial features:")
    print("  uv run python src/walk/04_train_with_spatial_features.py")


if __name__ == "__main__":
    main()
