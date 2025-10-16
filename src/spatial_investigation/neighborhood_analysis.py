"""
Spatial Investigation: Neighborhood Analysis

Tests the hypothesis that cleared pixels have different SPATIAL neighborhoods in Y-1
compared to intact pixels (not temporal changes in same pixel).

Key Question: Are cleared pixels near roads/edges/clearings in Y-1?

Method:
  1. Sample cleared pixels (from 2020) and intact pixels
  2. For each pixel, extract Y-1 (2019) embeddings for:
     - Center pixel
     - 8 neighbors (3x3 grid, 30m radius)
  3. Calculate spatial features:
     - Gradient magnitude (edge detection)
     - Heterogeneity (neighborhood variance)
     - Mean distance from center
  4. Compare cleared vs intact distributions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import math

from src.utils import EarthEngineClient, get_config


def get_neighbor_offsets(distance_m, lat):
    """
    Calculate lat/lon offsets for neighbors at given distance.

    Args:
        distance_m: Distance in meters
        lat: Latitude (for longitude correction)

    Returns:
        List of (lat_offset, lon_offset) tuples for 8 neighbors
    """
    # Earth radius and degree conversions
    # 1 degree latitude ≈ 111.32 km everywhere
    lat_per_m = 1 / 111320.0

    # 1 degree longitude varies with latitude
    lon_per_m = 1 / (111320.0 * math.cos(math.radians(lat)))

    # 8-neighbor offsets (3x3 grid, excluding center)
    # N, NE, E, SE, S, SW, W, NW
    offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue  # Skip center

            lat_offset = dy * distance_m * lat_per_m
            lon_offset = dx * distance_m * lon_per_m
            offsets.append((lat_offset, lon_offset))

    return offsets


def extract_neighborhood_embeddings(client, lat, lon, year, distance_m=30):
    """
    Extract embeddings for a pixel and its 8 neighbors.

    Args:
        client: EarthEngineClient
        lat: Center latitude
        lon: Center longitude
        year: Year to extract embeddings
        distance_m: Distance to neighbors (default 30m for 3x3 grid)

    Returns:
        dict with 'center' and 'neighbors' embeddings, or None if fails
    """
    try:
        # Get center embedding
        center_emb = client.get_embedding(
            lat=lat,
            lon=lon,
            date=f"{year}-06-01"
        )

        # Get neighbor embeddings
        offsets = get_neighbor_offsets(distance_m, lat)
        neighbor_embs = []

        for lat_offset, lon_offset in offsets:
            neighbor_lat = lat + lat_offset
            neighbor_lon = lon + lon_offset

            try:
                neighbor_emb = client.get_embedding(
                    lat=neighbor_lat,
                    lon=neighbor_lon,
                    date=f"{year}-06-01"
                )
                neighbor_embs.append(neighbor_emb)
            except Exception:
                # Skip if neighbor fails (might be out of bounds)
                continue

        if len(neighbor_embs) < 6:  # Need at least 6 of 8 neighbors
            return None

        return {
            'center': center_emb,
            'neighbors': np.array(neighbor_embs),
            'n_neighbors': len(neighbor_embs)
        }

    except Exception as e:
        return None


def calculate_spatial_features(neighborhood):
    """
    Calculate spatial features from neighborhood embeddings.

    Features:
      - gradient_magnitude: Mean distance from center to neighbors (edge detection)
      - heterogeneity: Std dev of distances (how varied the neighborhood is)
      - neighbor_variance: Variance within neighbors (internal heterogeneity)

    Args:
        neighborhood: Dict with 'center' and 'neighbors' arrays

    Returns:
        dict with spatial features
    """
    center = neighborhood['center']
    neighbors = neighborhood['neighbors']

    # Distance from center to each neighbor
    distances = np.array([
        np.linalg.norm(neighbor - center)
        for neighbor in neighbors
    ])

    # Feature 1: Gradient magnitude (mean distance to neighbors)
    # High gradient = center is different from neighbors (edge!)
    gradient_magnitude = np.mean(distances)

    # Feature 2: Heterogeneity (std dev of distances)
    # High heterogeneity = varied neighborhood (mixed land use)
    heterogeneity = np.std(distances)

    # Feature 3: Neighbor variance (how different neighbors are from each other)
    # Calculate mean pairwise distance among neighbors
    if len(neighbors) > 1:
        pairwise_distances = []
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                pairwise_distances.append(
                    np.linalg.norm(neighbors[i] - neighbors[j])
                )
        neighbor_variance = np.mean(pairwise_distances)
    else:
        neighbor_variance = 0.0

    # Feature 4: Max distance (most different neighbor)
    max_distance = np.max(distances)

    return {
        'gradient_magnitude': gradient_magnitude,
        'heterogeneity': heterogeneity,
        'neighbor_variance': neighbor_variance,
        'max_distance': max_distance,
    }


def run_neighborhood_analysis(n_samples=50):
    """
    Main analysis: Compare spatial neighborhoods of cleared vs intact pixels.

    Args:
        n_samples: Number of samples per class (cleared, intact)

    Returns:
        dict with results
    """
    print("=" * 80)
    print("SPATIAL INVESTIGATION: NEIGHBORHOOD ANALYSIS")
    print("=" * 80)
    print("\nHypothesis: Cleared pixels have different Y-1 neighborhoods than intact pixels")
    print("  - Higher gradients (edges nearby)")
    print("  - More heterogeneous (mixed forest/clearing)")
    print("  - Spatial precursors (roads, recent clearing) visible in neighborhood\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Sample pixels
    print(f"Step 1: Sampling {n_samples} cleared and {n_samples} intact pixels...")

    main_bounds = config.study_region_bounds
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    # Use multiple sub-regions for better sampling
    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},  # NW
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},  # NE
    ]

    # Get cleared pixels (2020) from all regions
    cleared_pixels = []
    for bounds in sub_regions:
        try:
            cleared = client.get_deforestation_labels(
                bounds=bounds,
                year=2020,
                min_tree_cover=30,
            )
            cleared_pixels.extend(cleared)
        except Exception:
            pass

    # Get intact pixels (stable forest, no clearing) from all regions
    intact_pixels = []
    for bounds in sub_regions:
        try:
            intact = client.get_stable_forest_locations(
                bounds=bounds,
                n_samples=n_samples,
                min_tree_cover=50,
                max_loss_year=2015,  # No loss since 2015
            )
            intact_pixels.extend(intact)
        except Exception:
            pass

    # Sample randomly
    import random
    random.seed(42)

    if len(cleared_pixels) > n_samples:
        cleared_pixels = random.sample(cleared_pixels, n_samples)
    if len(intact_pixels) > n_samples:
        intact_pixels = random.sample(intact_pixels, n_samples)

    print(f"  ✓ Sampled {len(cleared_pixels)} cleared pixels")
    print(f"  ✓ Sampled {len(intact_pixels)} intact pixels")

    # Extract neighborhood embeddings for Y-1 (2019)
    print("\nStep 2: Extracting Y-1 (2019) neighborhood embeddings...")
    print("  For each pixel: center + 8 neighbors (3x3 grid, 30m radius)")

    cleared_neighborhoods = []
    intact_neighborhoods = []

    # Process cleared pixels
    print("\n  Processing cleared pixels...")
    for pixel in tqdm(cleared_pixels, desc="  Cleared"):
        neighborhood = extract_neighborhood_embeddings(
            client=client,
            lat=pixel['lat'],
            lon=pixel['lon'],
            year=2019,  # Y-1
            distance_m=30
        )
        if neighborhood is not None:
            cleared_neighborhoods.append(neighborhood)

    # Process intact pixels
    print("\n  Processing intact pixels...")
    for pixel in tqdm(intact_pixels, desc="  Intact"):
        neighborhood = extract_neighborhood_embeddings(
            client=client,
            lat=pixel['lat'],
            lon=pixel['lon'],
            year=2019,  # Y-1
            distance_m=30
        )
        if neighborhood is not None:
            intact_neighborhoods.append(neighborhood)

    print(f"\n  ✓ Extracted {len(cleared_neighborhoods)} cleared neighborhoods")
    print(f"  ✓ Extracted {len(intact_neighborhoods)} intact neighborhoods")

    if len(cleared_neighborhoods) < 5 or len(intact_neighborhoods) < 5:
        print(f"\n✗ ERROR: Insufficient neighborhoods extracted (need ≥5 each)")
        print(f"  Cleared: {len(cleared_neighborhoods)}, Intact: {len(intact_neighborhoods)}")
        return None

    if len(cleared_neighborhoods) < 10 or len(intact_neighborhoods) < 10:
        print(f"\n  ⚠️  WARNING: Small sample sizes may reduce statistical power")
        print(f"  Cleared: {len(cleared_neighborhoods)}, Intact: {len(intact_neighborhoods)}")
        print(f"  Proceeding with available samples...")

    # Calculate spatial features
    print("\nStep 3: Calculating spatial features...")

    cleared_features = [
        calculate_spatial_features(nbhd)
        for nbhd in cleared_neighborhoods
    ]

    intact_features = [
        calculate_spatial_features(nbhd)
        for nbhd in intact_neighborhoods
    ]

    # Convert to arrays for analysis
    feature_names = ['gradient_magnitude', 'heterogeneity', 'neighbor_variance', 'max_distance']

    cleared_arrays = {
        name: np.array([f[name] for f in cleared_features])
        for name in feature_names
    }

    intact_arrays = {
        name: np.array([f[name] for f in intact_features])
        for name in feature_names
    }

    # Statistical comparison
    print("\nStep 4: Statistical comparison...")
    print("-" * 80)

    results = {}

    for name in feature_names:
        cleared_vals = cleared_arrays[name]
        intact_vals = intact_arrays[name]

        # Summary statistics
        cleared_mean = np.mean(cleared_vals)
        cleared_std = np.std(cleared_vals)
        intact_mean = np.mean(intact_vals)
        intact_std = np.std(intact_vals)

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(cleared_vals, intact_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((cleared_std**2 + intact_std**2) / 2)
        cohens_d = (cleared_mean - intact_mean) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        significant = p_value < 0.05

        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Cleared: {cleared_mean:.4f} ± {cleared_std:.4f}")
        print(f"  Intact:  {intact_mean:.4f} ± {intact_std:.4f}")
        print(f"  Difference: {cleared_mean - intact_mean:+.4f} ({100*(cleared_mean - intact_mean)/intact_mean:+.1f}%)")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        if significant:
            print(f"  ✓ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  ✗ Not significant")

        results[name] = {
            'cleared_mean': float(cleared_mean),
            'cleared_std': float(cleared_std),
            'intact_mean': float(intact_mean),
            'intact_std': float(intact_std),
            'difference': float(cleared_mean - intact_mean),
            'difference_pct': float(100 * (cleared_mean - intact_mean) / intact_mean) if intact_mean != 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(significant),
        }

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Check if gradient magnitude is higher for cleared
    gradient_higher = results['gradient_magnitude']['cleared_mean'] > results['gradient_magnitude']['intact_mean']
    gradient_sig = results['gradient_magnitude']['significant']

    heterogeneity_higher = results['heterogeneity']['cleared_mean'] > results['heterogeneity']['intact_mean']
    heterogeneity_sig = results['heterogeneity']['significant']

    if gradient_sig and gradient_higher:
        interpretation = """
✓ SPATIAL PRECURSOR SIGNAL DETECTED

Cleared pixels show significantly higher gradient magnitude in Y-1:
  - Gradients indicate edges/boundaries in the neighborhood
  - Higher gradients = center pixel is different from neighbors
  - This suggests roads, recent clearing, or forest edges nearby

Mechanism: Spatial precursors (roads/edges) in Y-1 → clearing in Y

Implication:
  - Deforestation spreads from existing clearing and roads
  - "Precursor signal" is SPATIAL (nearby features) not TEMPORAL (same pixel)
  - Lead time: 3-6 months (realistic for spatial spread)
  - Actionable: Monitor frontier edges and road corridors

Recommendation: Add spatial features to WALK phase
  - Distance to Y-1 clearing
  - Distance to forest edge
  - Neighborhood gradients
  - Use spatial CV (critical!)
"""
    elif not gradient_sig and not heterogeneity_sig:
        interpretation = """
✗ NO SIGNIFICANT SPATIAL SIGNAL

Cleared and intact pixels have similar Y-1 neighborhoods:
  - No difference in gradients (edge detection)
  - No difference in heterogeneity (mixed land use)

Possible explanations:
  1. Spatial precursors exist but not captured at 30m scale (try 100m, 500m)
  2. Signal is in the embeddings themselves, not spatial statistics
  3. Different clearing mechanisms (interior vs edge)

Recommendation: Investigate further
  - Try larger neighborhood radius (100m, 500m)
  - Test distance to clearings/edges directly
  - May need to pivot approach
"""
    else:
        interpretation = """
~ MIXED SPATIAL SIGNAL

Some spatial features differ, but pattern is unclear:
  - Need more investigation to understand mechanism
  - May be detecting some spatial patterns

Recommendation: Continue with Investigation 2 (Distance Features)
"""

    print(interpretation)

    # Create visualizations
    print("\nStep 5: Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spatial Neighborhood Analysis: Cleared vs Intact Pixels (Y-1)', fontsize=14, fontweight='bold')

    for idx, name in enumerate(feature_names):
        ax = axes[idx // 2, idx % 2]

        cleared_vals = cleared_arrays[name]
        intact_vals = intact_arrays[name]

        # Violin plot
        parts = ax.violinplot(
            [cleared_vals, intact_vals],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )

        # Styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Cleared (2020)', 'Intact'])
        ax.set_ylabel('Value')
        ax.set_title(f'{name.replace("_", " ").title()}\n(p={results[name]["p_value"]:.4f})')
        ax.grid(axis='y', alpha=0.3)

        # Add significance marker
        if results[name]['significant']:
            y_max = max(np.max(cleared_vals), np.max(intact_vals))
            ax.plot([1, 2], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=2)
            ax.text(1.5, y_max * 1.15, '***' if results[name]['p_value'] < 0.001 else '*',
                   ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/spatial_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "neighborhood_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    # Save results
    output = {
        'analysis': 'neighborhood_spatial_features',
        'n_cleared': len(cleared_neighborhoods),
        'n_intact': len(intact_neighborhoods),
        'features': results,
        'interpretation': interpretation.strip(),
    }

    results_file = output_dir / "neighborhood_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  ✓ Saved results: {results_file}")
    print("=" * 80)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spatial Neighborhood Analysis")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per class (default: 50)",
    )

    args = parser.parse_args()

    results = run_neighborhood_analysis(n_samples=args.n_samples)

    if results is None:
        exit(1)
