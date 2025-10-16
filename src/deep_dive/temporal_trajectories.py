"""
Deep-Dive Investigation: Temporal Embedding Trajectories

Goal: Test if embeddings show progressive changes before clearing.

Hypothesis: If AlphaEarth detects forest degradation/vulnerability:
  - Cleared pixels: Y-2 → Y-1 → Y embeddings drift toward "cleared" state
  - Intact pixels: Y-2 → Y-1 → Y embeddings remain stable

This would prove a TEMPORAL PRECURSOR SIGNAL exists in embeddings,
even if we don't understand what physical process it represents.

Method:
  1. Extract Y-2 (2018), Y-1 (2019), Y (2020) embeddings for each pixel
  2. Calculate embedding distances/changes over time
  3. Test if cleared pixels show larger Y-1→Y changes than Y-2→Y-1
  4. Compare to intact pixels (which should be stable across all years)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from src.utils import EarthEngineClient, get_config


def extract_multi_year_embeddings(client, pixels, label):
    """
    Extract embeddings for pixels across 3 years: Y-2, Y-1, Y.

    For cleared pixels (2020): 2018, 2019, 2020
    For intact pixels: 2018, 2019, 2020 (all should be similar)

    Args:
        client: EarthEngineClient
        pixels: List of pixel dicts
        label: 'cleared' or 'intact'

    Returns:
        List of dicts with embeddings and trajectories
    """
    print(f"\nExtracting 3-year embeddings for {len(pixels)} {label} pixels...")

    years = [2018, 2019, 2020]
    results = []

    for pixel in tqdm(pixels, desc=f"  {label.capitalize()}"):
        lat, lon = pixel['lat'], pixel['lon']

        pixel_data = {
            'lat': lat,
            'lon': lon,
            'label': label,
            'embeddings': {},
            'distances': {},
        }

        # Extract embeddings for all years
        success = True
        for year in years:
            try:
                emb = client.get_embedding(lat=lat, lon=lon, date=f"{year}-06-01")
                pixel_data['embeddings'][year] = emb
            except Exception as e:
                print(f"\n  Warning: Failed to get embedding for ({lat:.4f}, {lon:.4f}), year {year}")
                success = False
                break

        if not success:
            continue

        # Calculate distances between consecutive years
        emb_2018 = pixel_data['embeddings'][2018]
        emb_2019 = pixel_data['embeddings'][2019]
        emb_2020 = pixel_data['embeddings'][2020]

        # Y-2 → Y-1 distance (2018 → 2019)
        pixel_data['distances']['2018_to_2019'] = np.linalg.norm(emb_2019 - emb_2018)

        # Y-1 → Y distance (2019 → 2020)
        pixel_data['distances']['2019_to_2020'] = np.linalg.norm(emb_2020 - emb_2019)

        # Total change: Y-2 → Y (2018 → 2020)
        pixel_data['distances']['2018_to_2020'] = np.linalg.norm(emb_2020 - emb_2018)

        # Acceleration: Is Y-1→Y change larger than Y-2→Y-1?
        pixel_data['acceleration'] = (
            pixel_data['distances']['2019_to_2020'] -
            pixel_data['distances']['2018_to_2019']
        )

        results.append(pixel_data)

    print(f"  ✓ Successfully extracted {len(results)}/{len(pixels)} pixel trajectories")

    return results


def analyze_temporal_patterns(cleared_trajectories, intact_trajectories):
    """
    Analyze temporal embedding changes.

    Key tests:
      1. Do cleared pixels show larger Y-1→Y changes than intact?
      2. Do cleared pixels show acceleration (Y-1→Y > Y-2→Y-1)?
      3. Are intact pixels stable across all years?
    """
    print("\n" + "="*80)
    print("TEMPORAL TRAJECTORY ANALYSIS")
    print("="*80)

    # Extract distances
    cleared_2018_2019 = np.array([t['distances']['2018_to_2019'] for t in cleared_trajectories])
    cleared_2019_2020 = np.array([t['distances']['2019_to_2020'] for t in cleared_trajectories])
    cleared_2018_2020 = np.array([t['distances']['2018_to_2020'] for t in cleared_trajectories])
    cleared_acceleration = np.array([t['acceleration'] for t in cleared_trajectories])

    intact_2018_2019 = np.array([t['distances']['2018_to_2019'] for t in intact_trajectories])
    intact_2019_2020 = np.array([t['distances']['2019_to_2020'] for t in intact_trajectories])
    intact_2018_2020 = np.array([t['distances']['2018_to_2020'] for t in intact_trajectories])
    intact_acceleration = np.array([t['acceleration'] for t in intact_trajectories])

    # Test 1: Do cleared pixels show larger Y-1→Y (2019→2020) changes?
    print("\n" + "-"*80)
    print("TEST 1: Y-1 → Y (2019 → 2020) embedding change")
    print("-"*80)

    print(f"\nCleared pixels (approaching clearing):")
    print(f"  Mean: {np.mean(cleared_2019_2020):.4f}")
    print(f"  Std:  {np.std(cleared_2019_2020):.4f}")
    print(f"  Median: {np.median(cleared_2019_2020):.4f}")

    print(f"\nIntact pixels (stable forest):")
    print(f"  Mean: {np.mean(intact_2019_2020):.4f}")
    print(f"  Std:  {np.std(intact_2019_2020):.4f}")
    print(f"  Median: {np.median(intact_2019_2020):.4f}")

    t_stat, p_value = stats.ttest_ind(cleared_2019_2020, intact_2019_2020)
    u_stat, p_value_u = stats.mannwhitneyu(cleared_2019_2020, intact_2019_2020, alternative='two-sided')

    pooled_std = np.sqrt((np.std(cleared_2019_2020)**2 + np.std(intact_2019_2020)**2) / 2)
    cohens_d = (np.mean(cleared_2019_2020) - np.mean(intact_2019_2020)) / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical test:")
    print(f"  t-test: t = {t_stat:.3f}, p = {p_value:.6f}")
    print(f"  Mann-Whitney U: p = {p_value_u:.6f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    if p_value < 0.05:
        if np.mean(cleared_2019_2020) > np.mean(intact_2019_2020):
            print(f"\n  ✓ SIGNIFICANT: Cleared pixels show LARGER Y-1→Y changes!")
            print(f"    → Embeddings change dramatically in year of clearing")
        else:
            print(f"\n  ✓ SIGNIFICANT: Cleared pixels show SMALLER Y-1→Y changes")
            print(f"    → Unexpected - needs investigation")
    else:
        print(f"\n  ✗ Not significant: No difference in Y-1→Y changes")

    # Test 2: Do cleared pixels show acceleration?
    print("\n" + "-"*80)
    print("TEST 2: Acceleration (is Y-1→Y change > Y-2→Y-1 change?)")
    print("-"*80)

    print(f"\nCleared pixels:")
    print(f"  Y-2→Y-1 (2018→2019): {np.mean(cleared_2018_2019):.4f} ± {np.std(cleared_2018_2019):.4f}")
    print(f"  Y-1→Y   (2019→2020): {np.mean(cleared_2019_2020):.4f} ± {np.std(cleared_2019_2020):.4f}")
    print(f"  Acceleration: {np.mean(cleared_acceleration):.4f} ± {np.std(cleared_acceleration):.4f}")

    # Paired t-test: is 2019→2020 significantly larger than 2018→2019 for cleared pixels?
    t_stat_paired, p_value_paired = stats.ttest_rel(cleared_2019_2020, cleared_2018_2019)

    print(f"\n  Paired t-test (within cleared pixels):")
    print(f"    t = {t_stat_paired:.3f}, p = {p_value_paired:.6f}")

    if p_value_paired < 0.05:
        if np.mean(cleared_2019_2020) > np.mean(cleared_2018_2019):
            print(f"\n  ✓ SIGNIFICANT ACCELERATION!")
            print(f"    → Cleared pixels show faster embedding changes in Y-1→Y")
            print(f"    → This is a TEMPORAL PRECURSOR SIGNAL!")
        else:
            print(f"\n  ✓ SIGNIFICANT DECELERATION")
            print(f"    → Cleared pixels actually slow down before clearing")
    else:
        print(f"\n  ✗ No significant acceleration")
        print(f"    → Embedding changes are similar Y-2→Y-1 vs Y-1→Y")

    print(f"\nIntact pixels (for comparison):")
    print(f"  Y-2→Y-1 (2018→2019): {np.mean(intact_2018_2019):.4f} ± {np.std(intact_2018_2019):.4f}")
    print(f"  Y-1→Y   (2019→2020): {np.mean(intact_2019_2020):.4f} ± {np.std(intact_2019_2020):.4f}")
    print(f"  Acceleration: {np.mean(intact_acceleration):.4f} ± {np.std(intact_acceleration):.4f}")

    # Test 3: Are intact pixels stable?
    print("\n" + "-"*80)
    print("TEST 3: Stability of intact pixels")
    print("-"*80)

    # Are intact pixel changes small and similar across years?
    intact_mean_change = np.mean([np.mean(intact_2018_2019), np.mean(intact_2019_2020)])

    print(f"\nIntact pixel mean change: {intact_mean_change:.4f}")
    print(f"Cleared pixel Y-1→Y change: {np.mean(cleared_2019_2020):.4f}")
    print(f"Ratio: {np.mean(cleared_2019_2020) / intact_mean_change:.2f}x")

    if np.mean(cleared_2019_2020) > 2 * intact_mean_change:
        print(f"\n  ✓ Intact pixels are STABLE (cleared changes >{2:.0f}x larger)")
    else:
        print(f"\n  ~ Intact pixels show similar variability to cleared")

    # Summary statistics
    results = {
        'test1_y1_to_y_change': {
            'cleared_mean': float(np.mean(cleared_2019_2020)),
            'cleared_std': float(np.std(cleared_2019_2020)),
            'intact_mean': float(np.mean(intact_2019_2020)),
            'intact_std': float(np.std(intact_2019_2020)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05),
        },
        'test2_acceleration': {
            'cleared_2018_2019_mean': float(np.mean(cleared_2018_2019)),
            'cleared_2019_2020_mean': float(np.mean(cleared_2019_2020)),
            'acceleration_mean': float(np.mean(cleared_acceleration)),
            't_statistic_paired': float(t_stat_paired),
            'p_value_paired': float(p_value_paired),
            'significant': bool(p_value_paired < 0.05),
        },
        'test3_stability': {
            'intact_mean_change': float(intact_mean_change),
            'cleared_y1_to_y_change': float(np.mean(cleared_2019_2020)),
            'ratio': float(np.mean(cleared_2019_2020) / intact_mean_change) if intact_mean_change > 0 else None,
        },
    }

    return results


def create_trajectory_visualizations(cleared_trajectories, intact_trajectories):
    """
    Create visualizations of temporal trajectories.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))

    # Extract data
    cleared_2018_2019 = np.array([t['distances']['2018_to_2019'] for t in cleared_trajectories])
    cleared_2019_2020 = np.array([t['distances']['2019_to_2020'] for t in cleared_trajectories])
    cleared_acceleration = np.array([t['acceleration'] for t in cleared_trajectories])

    intact_2018_2019 = np.array([t['distances']['2018_to_2019'] for t in intact_trajectories])
    intact_2019_2020 = np.array([t['distances']['2019_to_2020'] for t in intact_trajectories])
    intact_acceleration = np.array([t['acceleration'] for t in intact_trajectories])

    # 1. Y-2 → Y-1 changes
    ax1 = plt.subplot(2, 3, 1)
    bp1 = ax1.boxplot([cleared_2018_2019, intact_2018_2019],
                       labels=['Cleared (2020)', 'Intact'],
                       patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightcoral')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Embedding Distance', fontsize=11)
    ax1.set_title('Y-2 → Y-1 Change (2018 → 2019)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Y-1 → Y changes
    ax2 = plt.subplot(2, 3, 2)
    bp2 = ax2.boxplot([cleared_2019_2020, intact_2019_2020],
                       labels=['Cleared (2020)', 'Intact'],
                       patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('Embedding Distance', fontsize=11)
    ax2.set_title('Y-1 → Y Change (2019 → 2020)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Acceleration
    ax3 = plt.subplot(2, 3, 3)
    bp3 = ax3.boxplot([cleared_acceleration, intact_acceleration],
                       labels=['Cleared (2020)', 'Intact'],
                       patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightcoral')
    bp3['boxes'][1].set_facecolor('lightgreen')
    ax3.set_ylabel('Acceleration', fontsize=11)
    ax3.set_title('Acceleration (Y-1→Y minus Y-2→Y-1)', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Scatter: Y-2→Y-1 vs Y-1→Y for cleared pixels
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(cleared_2018_2019, cleared_2019_2020,
               c='coral', s=80, alpha=0.6, edgecolors='black', linewidths=0.5,
               label='Cleared (2020)')
    ax4.plot([0, max(cleared_2018_2019.max(), cleared_2019_2020.max())],
             [0, max(cleared_2018_2019.max(), cleared_2019_2020.max())],
             'k--', linewidth=1, alpha=0.5, label='No acceleration')
    ax4.set_xlabel('Y-2 → Y-1 Change (2018 → 2019)', fontsize=10)
    ax4.set_ylabel('Y-1 → Y Change (2019 → 2020)', fontsize=10)
    ax4.set_title('Cleared Pixels: Acceleration Test', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Scatter: Y-2→Y-1 vs Y-1→Y for intact pixels
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(intact_2018_2019, intact_2019_2020,
               c='lightgreen', s=80, alpha=0.6, edgecolors='black', linewidths=0.5,
               label='Intact')
    ax5.plot([0, max(intact_2018_2019.max(), intact_2019_2020.max())],
             [0, max(intact_2018_2019.max(), intact_2019_2020.max())],
             'k--', linewidth=1, alpha=0.5, label='No acceleration')
    ax5.set_xlabel('Y-2 → Y-1 Change (2018 → 2019)', fontsize=10)
    ax5.set_ylabel('Y-1 → Y Change (2019 → 2020)', fontsize=10)
    ax5.set_title('Intact Pixels: Stability Test', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Mean trajectory comparison
    ax6 = plt.subplot(2, 3, 6)

    # Cleared pixels trajectory
    cleared_means = [
        0,  # Baseline at Y-2
        np.mean(cleared_2018_2019),  # Y-2 → Y-1
        np.mean(cleared_2018_2019) + np.mean(cleared_2019_2020)  # Y-2 → Y-1 → Y
    ]

    # Intact pixels trajectory
    intact_means = [
        0,  # Baseline at Y-2
        np.mean(intact_2018_2019),  # Y-2 → Y-1
        np.mean(intact_2018_2019) + np.mean(intact_2019_2020)  # Y-2 → Y-1 → Y
    ]

    years = [2018, 2019, 2020]
    ax6.plot(years, cleared_means, marker='o', linewidth=2, markersize=8,
            color='coral', label='Cleared (2020)')
    ax6.plot(years, intact_means, marker='s', linewidth=2, markersize=8,
            color='green', label='Intact')

    ax6.set_xlabel('Year', fontsize=11)
    ax6.set_ylabel('Cumulative Embedding Distance from 2018', fontsize=11)
    ax6.set_title('Mean Temporal Trajectory', fontsize=12, fontweight='bold')
    ax6.set_xticks(years)
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.suptitle('Temporal Embedding Trajectories: Cleared vs Intact Pixels',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "temporal_trajectories.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    return plot_file


def run_temporal_trajectory_analysis(n_samples=30):
    """
    Main analysis workflow.
    """
    print("="*80)
    print("DEEP-DIVE INVESTIGATION: TEMPORAL EMBEDDING TRAJECTORIES")
    print("="*80)
    print("\nGoal: Test if embeddings show progressive changes before clearing")
    print("  - Extract Y-2 (2018), Y-1 (2019), Y (2020) embeddings")
    print("  - Test for acceleration in cleared pixels")
    print("  - Compare to intact pixel stability\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Sample pixels
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

    import random
    random.seed(42)

    if len(cleared_pixels) > n_samples:
        cleared_pixels = random.sample(cleared_pixels, n_samples)
    if len(intact_pixels) > n_samples:
        intact_pixels = random.sample(intact_pixels, n_samples)

    print(f"  ✓ Sampled {len(cleared_pixels)} cleared pixels")
    print(f"  ✓ Sampled {len(intact_pixels)} intact pixels")

    # Extract multi-year embeddings
    print("\nStep 2: Extracting multi-year embeddings (2018, 2019, 2020)...")

    cleared_trajectories = extract_multi_year_embeddings(client, cleared_pixels, 'cleared')
    intact_trajectories = extract_multi_year_embeddings(client, intact_pixels, 'intact')

    if len(cleared_trajectories) < 5 or len(intact_trajectories) < 5:
        print(f"\n✗ ERROR: Insufficient trajectories extracted")
        return None

    # Analyze temporal patterns
    print("\nStep 3: Analyzing temporal patterns...")
    results = analyze_temporal_patterns(cleared_trajectories, intact_trajectories)

    # Create visualizations
    print("\nStep 4: Creating visualizations...")
    plot_file = create_trajectory_visualizations(cleared_trajectories, intact_trajectories)

    # Save results
    output = {
        'n_cleared': len(cleared_trajectories),
        'n_intact': len(intact_trajectories),
        'statistical_tests': results,
    }

    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    results_file = output_dir / "temporal_trajectories.json"

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Saved results: {results_file}")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Temporal Trajectory Analysis")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples per class (default: 30)",
    )

    args = parser.parse_args()

    results = run_temporal_trajectory_analysis(n_samples=args.n_samples)

    if results is None:
        exit(1)
