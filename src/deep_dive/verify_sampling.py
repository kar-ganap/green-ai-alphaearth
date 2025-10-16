"""
Deep-Dive Investigation: Verify Sampling

This script verifies that our pixel sampling is working correctly:
  1. Visualize pixel locations on a map
  2. Check tree cover and loss year for each pixel
  3. Verify cleared pixels are actually cleared
  4. Verify intact pixels are actually intact
  5. Look for any systematic biases

If sampling is broken, all downstream analyses are invalid.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ee

from src.utils import EarthEngineClient, get_config


def verify_pixel_labels(client, pixels, label_type):
    """
    Verify that pixels match their expected labels.

    Args:
        client: EarthEngineClient
        pixels: List of pixel dicts with lat/lon
        label_type: 'cleared' or 'intact'

    Returns:
        dict with verification results
    """
    print(f"\nVerifying {len(pixels)} {label_type} pixels...")

    gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    tree_cover_2000 = gfc.select('treecover2000')
    loss_year = gfc.select('lossyear')

    results = []

    for i, pixel in enumerate(pixels):
        lat, lon = pixel['lat'], pixel['lon']
        point = ee.Geometry.Point([lon, lat])

        # Sample Hansen GFC
        sample = gfc.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=30
        ).getInfo()

        tree_cover = sample.get('treecover2000', None)
        loss_year_val = sample.get('lossyear', None)

        # Convert loss year (1=2001, 20=2020)
        actual_loss_year = None
        if loss_year_val and loss_year_val > 0:
            actual_loss_year = 2000 + loss_year_val

        # Determine if matches expected label
        if label_type == 'cleared':
            expected_loss_year = 2020
            matches = (actual_loss_year == expected_loss_year)
        else:  # intact
            expected_loss_year = None
            matches = (actual_loss_year is None or actual_loss_year <= 2015)

        result = {
            'index': i,
            'lat': lat,
            'lon': lon,
            'tree_cover_2000': tree_cover,
            'loss_year': actual_loss_year,
            'expected': expected_loss_year if label_type == 'cleared' else 'no loss or ≤2015',
            'matches': matches,
        }

        results.append(result)

        status = "✓" if matches else "✗"
        print(f"  {status} Pixel {i:2d}: ({lat:.4f}, {lon:.4f}) | "
              f"Tree cover: {tree_cover}% | Loss: {actual_loss_year or 'none'}")

    # Summary
    num_matches = sum(1 for r in results if r['matches'])
    pct_matches = 100 * num_matches / len(results) if results else 0

    print(f"\n  Summary: {num_matches}/{len(results)} ({pct_matches:.1f}%) match expected labels")

    return results


def check_spatial_distribution(cleared_pixels, intact_pixels):
    """
    Check spatial distribution of sampled pixels.

    Are they clustered? Spread out? Any biases?
    """
    print("\n" + "="*80)
    print("SPATIAL DISTRIBUTION")
    print("="*80)

    # Extract coordinates
    cleared_lats = [p['lat'] for p in cleared_pixels]
    cleared_lons = [p['lon'] for p in cleared_pixels]
    intact_lats = [p['lat'] for p in intact_pixels]
    intact_lons = [p['lon'] for p in intact_pixels]

    # Basic statistics
    print("\nCleared pixels:")
    print(f"  Lat range: {min(cleared_lats):.4f} to {max(cleared_lats):.4f}")
    print(f"  Lon range: {min(cleared_lons):.4f} to {max(cleared_lons):.4f}")
    print(f"  Lat std: {np.std(cleared_lats):.4f}")
    print(f"  Lon std: {np.std(cleared_lons):.4f}")

    print("\nIntact pixels:")
    print(f"  Lat range: {min(intact_lats):.4f} to {max(intact_lats):.4f}")
    print(f"  Lon range: {min(intact_lons):.4f} to {max(intact_lons):.4f}")
    print(f"  Lat std: {np.std(intact_lats):.4f}")
    print(f"  Lon std: {np.std(intact_lons):.4f}")

    # Check if they overlap spatially
    cleared_lat_center = np.mean(cleared_lats)
    cleared_lon_center = np.mean(cleared_lons)
    intact_lat_center = np.mean(intact_lats)
    intact_lon_center = np.mean(intact_lons)

    # Distance between centroids (rough)
    lat_diff = abs(cleared_lat_center - intact_lat_center)
    lon_diff = abs(cleared_lon_center - intact_lon_center)

    print(f"\nCentroid separation:")
    print(f"  Lat difference: {lat_diff:.4f} degrees ({lat_diff * 111:.1f} km)")
    print(f"  Lon difference: {lon_diff:.4f} degrees")

    if lat_diff > 1.0 or lon_diff > 1.0:
        print("\n  ⚠️  WARNING: Cleared and intact pixels are far apart (>100km)")
        print("     This could explain lack of spatial signal if they're in different regions")


def create_map_visualization(cleared_pixels, intact_pixels, cleared_results, intact_results):
    """
    Create a map visualization of sampled pixels.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot cleared pixels
    cleared_lats = [p['lat'] for p in cleared_pixels]
    cleared_lons = [p['lon'] for p in cleared_pixels]

    # Color by whether they match expected label
    cleared_colors = ['green' if r['matches'] else 'red' for r in cleared_results]

    ax.scatter(cleared_lons, cleared_lats,
              c=cleared_colors, s=100, alpha=0.6,
              marker='o', edgecolors='black', linewidths=1,
              label='Cleared (2020)')

    # Plot intact pixels
    intact_lats = [p['lat'] for p in intact_pixels]
    intact_lons = [p['lon'] for p in intact_pixels]

    intact_colors = ['green' if r['matches'] else 'red' for r in intact_results]

    ax.scatter(intact_lons, intact_lats,
              c=intact_colors, s=100, alpha=0.6,
              marker='s', edgecolors='black', linewidths=1,
              label='Intact')

    # Styling
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Sampled Pixel Locations\n(Green=Matches Expected Label, Red=Mismatch)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    num_cleared_match = sum(1 for r in cleared_results if r['matches'])
    num_intact_match = sum(1 for r in intact_results if r['matches'])

    ax.text(0.02, 0.98,
           f"Cleared: {num_cleared_match}/{len(cleared_results)} match\n"
           f"Intact: {num_intact_match}/{len(intact_results)} match",
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "pixel_sampling_verification.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    return plot_file


def check_2019_clearing_presence(client, cleared_pixels):
    """
    Check if there are ANY 2019 clearings in the study region.

    This tests if the "no clearings within 5km" finding is because:
      a) Our pixels are in pristine areas, OR
      b) There are genuinely no 2019 clearings in the entire region
    """
    print("\n" + "="*80)
    print("CHECKING 2019 CLEARING PRESENCE IN REGION")
    print("="*80)

    config = get_config()
    bounds = config.study_region_bounds

    # Create bounding box
    bbox = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])

    # Get Hansen GFC
    gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    loss_year = gfc.select('lossyear')

    # 2019 clearings (loss_year = 19)
    clearings_2019 = loss_year.eq(19)

    # Count pixels
    stats = clearings_2019.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=bbox,
        scale=30,
        maxPixels=1e9
    )

    num_2019_clearings = stats.get('lossyear').getInfo()

    print(f"\n2019 clearings in study region:")
    print(f"  Count: {num_2019_clearings} pixels")
    print(f"  Area: {num_2019_clearings * 30 * 30 / 10000:.1f} hectares")

    if num_2019_clearings == 0:
        print("\n  ⚠️  WARNING: NO 2019 clearings found in entire study region!")
        print("     This explains why distance-to-clearing was 5000m for all pixels")
        print("     May need to expand region or choose different years")
    else:
        print(f"\n  ✓ 2019 clearings exist in region")
        print(f"    Our sampled pixels just happen to be far from them")


def run_sampling_verification():
    """
    Main verification workflow.
    """
    print("="*80)
    print("DEEP-DIVE INVESTIGATION: VERIFY SAMPLING")
    print("="*80)
    print("\nGoal: Verify that pixel sampling is working correctly")
    print("  - Check if cleared pixels are actually cleared in 2020")
    print("  - Check if intact pixels are actually intact")
    print("  - Visualize spatial distribution")
    print("  - Check for any systematic biases\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Sample pixels (same as distance analysis)
    print("Step 1: Sampling pixels...")

    main_bounds = config.study_region_bounds
    mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
    mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

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
        except Exception as e:
            print(f"  Warning: Failed to get cleared pixels from region: {e}")

    # Get intact pixels
    intact_pixels = []
    for bounds in sub_regions:
        try:
            intact = client.get_stable_forest_locations(
                bounds=bounds,
                n_samples=30,
                min_tree_cover=50,
                max_loss_year=2015,
            )
            intact_pixels.extend(intact)
        except Exception as e:
            print(f"  Warning: Failed to get intact pixels from region: {e}")

    # Sample randomly
    import random
    random.seed(42)

    n_samples = 30
    if len(cleared_pixels) > n_samples:
        cleared_pixels = random.sample(cleared_pixels, n_samples)
    if len(intact_pixels) > n_samples:
        intact_pixels = random.sample(intact_pixels, n_samples)

    print(f"  ✓ Sampled {len(cleared_pixels)} cleared pixels")
    print(f"  ✓ Sampled {len(intact_pixels)} intact pixels")

    # Verify labels
    print("\n" + "="*80)
    print("STEP 2: VERIFYING PIXEL LABELS")
    print("="*80)

    cleared_results = verify_pixel_labels(client, cleared_pixels, 'cleared')
    intact_results = verify_pixel_labels(client, intact_pixels, 'intact')

    # Check spatial distribution
    check_spatial_distribution(cleared_pixels, intact_pixels)

    # Check 2019 clearing presence
    check_2019_clearing_presence(client, cleared_pixels)

    # Create visualization
    plot_file = create_map_visualization(
        cleared_pixels, intact_pixels,
        cleared_results, intact_results
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    num_cleared_match = sum(1 for r in cleared_results if r['matches'])
    num_intact_match = sum(1 for r in intact_results if r['matches'])

    pct_cleared = 100 * num_cleared_match / len(cleared_results) if cleared_results else 0
    pct_intact = 100 * num_intact_match / len(intact_results) if intact_results else 0

    print(f"\nLabel Verification:")
    print(f"  Cleared: {num_cleared_match}/{len(cleared_results)} ({pct_cleared:.1f}%) match")
    print(f"  Intact:  {num_intact_match}/{len(intact_results)} ({pct_intact:.1f}%) match")

    if pct_cleared < 90 or pct_intact < 90:
        print("\n  ✗ WARNING: Label mismatch detected!")
        print("    Sampling may be broken - need to fix before continuing")
    else:
        print("\n  ✓ Labels verified - sampling is working correctly")

    # Save results
    output = {
        'cleared_verification': cleared_results,
        'intact_verification': intact_results,
        'summary': {
            'n_cleared': len(cleared_results),
            'n_intact': len(intact_results),
            'cleared_match_rate': pct_cleared / 100,
            'intact_match_rate': pct_intact / 100,
        }
    }

    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "sampling_verification.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Saved results: {results_file}")
    print("="*80)

    return output


if __name__ == "__main__":
    results = run_sampling_verification()
