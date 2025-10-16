"""
Spatial Investigation: Distance Features Analysis

Tests the hypothesis that cleared pixels are closer to spatial precursors
(roads, edges, recent clearing) in Y-1 compared to intact pixels.

This is a more direct test than neighborhood gradients.

Key Question: Are Y clearings closer to Y-1 clearings/edges than intact pixels are?

Method:
  1. Sample cleared pixels (from 2020) and intact pixels
  2. For each pixel, calculate in Y-1 (2019):
     - Distance to nearest Y-1 clearing
     - Distance to nearest Y-2 clearing (2018)
     - Distance to forest edge
     - Clearing density within 100m, 500m, 1km
  3. Compare cleared vs intact distributions
  4. Test which features predict clearing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import ee

from src.utils import EarthEngineClient, get_config


def calculate_distance_to_nearest_clearing(lat, lon, year, client):
    """
    Calculate distance from a pixel to nearest clearing in a given year.

    Uses Hansen GFC loss year to find clearings.

    Args:
        lat: Latitude
        lon: Longitude
        year: Year to check for clearings
        client: EarthEngineClient

    Returns:
        Distance in meters, or None if error
    """
    try:
        # Create point
        point = ee.Geometry.Point([lon, lat])

        # Search radius (start small, expand if no clearings found)
        search_radii = [500, 1000, 2000, 5000]  # meters

        for radius in search_radii:
            # Create buffer
            buffer = point.buffer(radius)

            # Get Hansen GFC data
            gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

            # Get pixels that were cleared in the target year
            loss_year = gfc.select('lossyear')

            # Hansen loss year encoding: 1 = 2001, 20 = 2020, etc.
            target_year_code = year - 2000

            # Create mask for clearings in target year
            clearings = loss_year.eq(target_year_code)

            # Sample clearings in buffer
            sample = clearings.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=buffer,
                scale=30,
                maxPixels=1e8
            )

            clearing_count = sample.get('lossyear').getInfo()

            if clearing_count > 0:
                # Found clearings within this radius
                # Get actual clearing pixels
                clearing_pixels = clearings.selfMask()

                # Calculate distance
                distance_img = clearing_pixels.distance(ee.Kernel.euclidean(radius, 'meters'))

                # Sample at point
                distance = distance_img.reduceRegion(
                    reducer=ee.Reducer.min(),
                    geometry=point,
                    scale=30
                ).get('lossyear').getInfo()

                if distance is not None:
                    return float(distance)

        # No clearings found within max radius
        return search_radii[-1]  # Return max distance as upper bound

    except Exception as e:
        print(f"  Warning: Error calculating distance for ({lat}, {lon}): {e}")
        return None


def calculate_distance_to_edge(lat, lon, year, client):
    """
    Calculate distance from pixel to forest edge in given year.

    Edge = boundary between forest (tree cover > 30%) and non-forest.

    Args:
        lat: Latitude
        lon: Longitude
        year: Year to check
        client: EarthEngineClient

    Returns:
        Distance in meters, or None if error
    """
    try:
        point = ee.Geometry.Point([lon, lat])

        # Get Hansen GFC tree cover and loss
        gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
        tree_cover_2000 = gfc.select('treecover2000')
        loss_year = gfc.select('lossyear')

        # Calculate forest mask for target year
        # Start with 2000 tree cover > 30%
        forest_2000 = tree_cover_2000.gt(30)

        # Subtract any loss that happened before target year
        target_year_code = year - 2000
        loss_before_year = loss_year.gt(0).And(loss_year.lte(target_year_code))

        # Forest in target year = forest in 2000 minus loss before year
        forest_year = forest_2000.And(loss_before_year.Not())

        # Edge detection: forest pixels adjacent to non-forest
        # Use morphological operations
        # Erode forest by 1 pixel, difference gives edge
        kernel = ee.Kernel.circle(radius=1, units='pixels')
        forest_eroded = forest_year.reduceNeighborhood(
            reducer=ee.Reducer.min(),
            kernel=kernel
        )

        # Edge = forest but not in eroded version
        edge = forest_year.And(forest_eroded.Not())

        # Calculate distance to edge
        search_radius = 5000  # 5km max
        edge_pixels = edge.selfMask()

        distance_img = edge_pixels.distance(ee.Kernel.euclidean(search_radius, 'meters'))

        distance = distance_img.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=point,
            scale=30
        ).get('treecover2000').getInfo()

        if distance is not None:
            return float(distance)
        else:
            return search_radius  # Upper bound

    except Exception as e:
        print(f"  Warning: Error calculating edge distance for ({lat}, {lon}): {e}")
        return None


def calculate_clearing_density(lat, lon, year, radius_m, client):
    """
    Calculate percentage of area cleared within radius in given year.

    Args:
        lat: Latitude
        lon: Longitude
        year: Year to check
        radius_m: Radius in meters
        client: EarthEngineClient

    Returns:
        Percentage cleared (0-100), or None if error
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(radius_m)

        # Get Hansen GFC
        gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
        loss_year = gfc.select('lossyear')

        # Clearings in target year
        target_year_code = year - 2000
        clearings = loss_year.eq(target_year_code)

        # Calculate fraction cleared
        stats = clearings.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=30,
            maxPixels=1e8
        )

        fraction = stats.get('lossyear').getInfo()

        if fraction is not None:
            return float(fraction * 100)  # Convert to percentage
        else:
            return 0.0

    except Exception as e:
        print(f"  Warning: Error calculating clearing density for ({lat}, {lon}): {e}")
        return None


def extract_spatial_features(client, lat, lon, reference_year):
    """
    Extract all spatial features for a pixel.

    Features calculated relative to reference_year (typically Y-1).

    Args:
        client: EarthEngineClient
        lat: Latitude
        lon: Longitude
        reference_year: Year to calculate features (e.g., 2019 for 2020 clearing)

    Returns:
        dict with spatial features, or None if error
    """
    features = {}

    # Distance to nearest Y-1 clearing
    dist_y1 = calculate_distance_to_nearest_clearing(lat, lon, reference_year, client)
    if dist_y1 is None:
        return None
    features['distance_to_clearing_y1'] = dist_y1

    # Distance to nearest Y-2 clearing (control)
    dist_y2 = calculate_distance_to_nearest_clearing(lat, lon, reference_year - 1, client)
    if dist_y2 is None:
        return None
    features['distance_to_clearing_y2'] = dist_y2

    # Distance to forest edge in Y-1
    dist_edge = calculate_distance_to_edge(lat, lon, reference_year, client)
    if dist_edge is None:
        return None
    features['distance_to_edge_y1'] = dist_edge

    # Clearing density within different radii in Y-1
    for radius in [100, 500, 1000]:
        density = calculate_clearing_density(lat, lon, reference_year, radius, client)
        if density is None:
            return None
        features[f'clearing_density_{radius}m_y1'] = density

    # Also get Y-2 clearing density as control
    density_y2 = calculate_clearing_density(lat, lon, reference_year - 1, 500, client)
    if density_y2 is None:
        return None
    features['clearing_density_500m_y2'] = density_y2

    return features


def run_distance_features_analysis(n_samples=30):
    """
    Main analysis: Compare spatial distance features for cleared vs intact pixels.

    Args:
        n_samples: Number of samples per class

    Returns:
        dict with results
    """
    print("=" * 80)
    print("SPATIAL INVESTIGATION: DISTANCE FEATURES ANALYSIS")
    print("=" * 80)
    print("\nHypothesis: Cleared pixels are closer to Y-1 clearings/edges than intact pixels")
    print("  - Lower distance to Y-1 clearings")
    print("  - Lower distance to forest edges")
    print("  - Higher clearing density in neighborhood")
    print("  - Spatial precursors (roads, recent clearing) predict future clearing\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Sample pixels
    print(f"Step 1: Sampling {n_samples} cleared and {n_samples} intact pixels...")

    main_bounds = config.study_region_bounds
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

    # Use multiple sub-regions
    sub_regions = [
        {"min_lon": main_bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
        {"min_lon": mid_lon, "max_lon": main_bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": main_bounds["max_lat"]},
    ]

    # Get cleared pixels (2020)
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

    # Get intact pixels
    intact_pixels = []
    for bounds in sub_regions:
        try:
            intact = client.get_stable_forest_locations(
                bounds=bounds,
                n_samples=n_samples,
                min_tree_cover=50,
                max_loss_year=2015,
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

    # Extract spatial features for Y-1 (2019)
    print("\nStep 2: Extracting Y-1 (2019) spatial features...")
    print("  For each pixel:")
    print("    - Distance to nearest 2019 clearing")
    print("    - Distance to forest edge")
    print("    - Clearing density within 100m, 500m, 1km")

    cleared_features = []
    intact_features = []

    # Process cleared pixels
    print("\n  Processing cleared pixels...")
    for pixel in tqdm(cleared_pixels, desc="  Cleared"):
        features = extract_spatial_features(
            client=client,
            lat=pixel['lat'],
            lon=pixel['lon'],
            reference_year=2019  # Y-1
        )
        if features is not None:
            cleared_features.append(features)

    # Process intact pixels
    print("\n  Processing intact pixels...")
    for pixel in tqdm(intact_pixels, desc="  Intact"):
        features = extract_spatial_features(
            client=client,
            lat=pixel['lat'],
            lon=pixel['lon'],
            reference_year=2019  # Y-1
        )
        if features is not None:
            intact_features.append(features)

    print(f"\n  ✓ Extracted features for {len(cleared_features)} cleared pixels")
    print(f"  ✓ Extracted features for {len(intact_features)} intact pixels")

    if len(cleared_features) < 5 or len(intact_features) < 5:
        print(f"\n✗ ERROR: Insufficient features extracted (need ≥5 each)")
        return None

    # Statistical comparison
    print("\nStep 3: Statistical comparison...")
    print("-" * 80)

    feature_names = list(cleared_features[0].keys())
    results = {}

    for name in feature_names:
        cleared_vals = np.array([f[name] for f in cleared_features])
        intact_vals = np.array([f[name] for f in intact_features])

        # Summary statistics
        cleared_mean = np.mean(cleared_vals)
        cleared_std = np.std(cleared_vals)
        cleared_median = np.median(cleared_vals)
        intact_mean = np.mean(intact_vals)
        intact_std = np.std(intact_vals)
        intact_median = np.median(intact_vals)

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(cleared_vals, intact_vals)

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, p_value_u = stats.mannwhitneyu(cleared_vals, intact_vals, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((cleared_std**2 + intact_std**2) / 2)
        cohens_d = (cleared_mean - intact_mean) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        significant = p_value < 0.05

        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Cleared: {cleared_mean:.1f}m ± {cleared_std:.1f}m (median: {cleared_median:.1f}m)")
        print(f"  Intact:  {intact_mean:.1f}m ± {intact_std:.1f}m (median: {intact_median:.1f}m)")
        print(f"  Difference: {cleared_mean - intact_mean:+.1f}m ({100*(cleared_mean - intact_mean)/intact_mean:+.1f}%)")
        print(f"  t-test: t={t_stat:.3f}, p={p_value:.6f}")
        print(f"  Mann-Whitney U: p={p_value_u:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        if significant:
            print(f"  ✓ SIGNIFICANT (p < 0.05)")
            if cleared_mean < intact_mean:
                print(f"    → Cleared pixels are CLOSER (supports hypothesis)")
            else:
                print(f"    → Cleared pixels are FARTHER (contradicts hypothesis)")
        else:
            print(f"  ✗ Not significant")

        results[name] = {
            'cleared_mean': float(cleared_mean),
            'cleared_std': float(cleared_std),
            'cleared_median': float(cleared_median),
            'intact_mean': float(intact_mean),
            'intact_std': float(intact_std),
            'intact_median': float(intact_median),
            'difference': float(cleared_mean - intact_mean),
            'difference_pct': float(100 * (cleared_mean - intact_mean) / intact_mean) if intact_mean != 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'p_value_mann_whitney': float(p_value_u),
            'cohens_d': float(cohens_d),
            'significant': bool(significant),
            'supports_hypothesis': bool(significant and (cleared_mean < intact_mean)),
        }

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Check key features
    dist_y1_sig = results['distance_to_clearing_y1']['significant']
    dist_y1_supports = results['distance_to_clearing_y1']['supports_hypothesis']

    dist_edge_sig = results['distance_to_edge_y1']['significant']
    dist_edge_supports = results['distance_to_edge_y1']['supports_hypothesis']

    density_500_sig = results['clearing_density_500m_y1']['significant']

    if dist_y1_supports or dist_edge_supports:
        interpretation = f"""
✓ SPATIAL PRECURSOR SIGNAL DETECTED

Cleared pixels show significantly {'closer proximity to Y-1 clearings' if dist_y1_supports else 'closer proximity to forest edges'} in Y-1.

Key findings:
  - Distance to Y-1 clearing: {results['distance_to_clearing_y1']['cleared_mean']:.0f}m (cleared) vs {results['distance_to_clearing_y1']['intact_mean']:.0f}m (intact)
    p = {results['distance_to_clearing_y1']['p_value']:.4f}, {'SIGNIFICANT' if dist_y1_sig else 'not significant'}

  - Distance to forest edge: {results['distance_to_edge_y1']['cleared_mean']:.0f}m (cleared) vs {results['distance_to_edge_y1']['intact_mean']:.0f}m (intact)
    p = {results['distance_to_edge_y1']['p_value']:.4f}, {'SIGNIFICANT' if dist_edge_sig else 'not significant'}

Mechanism: Spatial diffusion from clearings/edges
  - Deforestation spreads from existing clearings and forest edges
  - Cleared pixels in Y are near Y-1 clearings/edges
  - Timeline: 6-12 months (Y-1 features → Y clearing)

Implication:
  - "Precursor signal" is SPATIAL (proximity to features), not temporal (same pixel change)
  - Actionable: Target enforcement near frontiers and recent clearings
  - Lead time: 6-12 months (realistic for spatial spread)

Recommendation:
  - Add spatial features to WALK phase (distance to clearing, edge proximity)
  - Use spatial CV (critical to avoid overfitting due to autocorrelation)
  - Test larger neighborhoods (100m, 500m) to confirm scale
"""
    elif not any([r['significant'] for r in results.values()]):
        interpretation = """
✗ NO SIGNIFICANT SPATIAL SIGNAL

Cleared and intact pixels show similar distances to Y-1 features:
  - No difference in proximity to Y-1 clearings
  - No difference in proximity to forest edges
  - No difference in clearing density

Possible explanations:
  1. Spatial precursors exist at even larger scales (>5km)
  2. Different clearing mechanisms (illegal vs legal, interior vs edge)
  3. Model uses something else (embedding patterns, other features)

Recommendation:
  - Re-examine Test 4 "distance to center" feature (why did it work?)
  - Consider that model may be using embedding semantics, not spatial proximity
  - May need to pivot understanding of what model detects
"""
    else:
        interpretation = """
~ MIXED SPATIAL SIGNAL

Some spatial features differ, but pattern is unclear.
Need more investigation to understand mechanism.

Recommendation: Test larger neighborhood scales (Option A)
"""

    print(interpretation)

    # Visualizations
    print("\nStep 4: Creating visualizations...")

    # Plot top 4 features
    top_features = ['distance_to_clearing_y1', 'distance_to_edge_y1',
                    'clearing_density_500m_y1', 'distance_to_clearing_y2']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spatial Distance Features: Cleared vs Intact Pixels (Y-1)',
                 fontsize=14, fontweight='bold')

    for idx, name in enumerate(top_features):
        ax = axes[idx // 2, idx % 2]

        cleared_vals = np.array([f[name] for f in cleared_features])
        intact_vals = np.array([f[name] for f in intact_features])

        # Box plot
        bp = ax.boxplot([cleared_vals, intact_vals],
                        labels=['Cleared (2020)', 'Intact'],
                        patch_artist=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')

        # Styling
        ax.set_ylabel('Distance (m)' if 'distance' in name else 'Density (%)')
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

    plot_file = output_dir / "distance_features_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    # Save results
    output = {
        'analysis': 'distance_features',
        'n_cleared': len(cleared_features),
        'n_intact': len(intact_features),
        'features': results,
        'interpretation': interpretation.strip(),
    }

    results_file = output_dir / "distance_features_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  ✓ Saved results: {results_file}")
    print("=" * 80)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spatial Distance Features Analysis")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples per class (default: 30)",
    )

    args = parser.parse_args()

    results = run_distance_features_analysis(n_samples=args.n_samples)

    if results is None:
        exit(1)
