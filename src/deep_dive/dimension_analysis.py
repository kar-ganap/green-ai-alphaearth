"""
Deep-Dive Investigation: Dimension Analysis

Goal: Understand what the discriminative embedding dimensions encode.

We know:
  - Dimensions 56, 49, 3, 52, 1, 50 are top discriminators (Cohen's d > 2.0)
  - 48% of dimensions show significant separation
  - Embeddings show temporal acceleration before clearing

Question: What do these dimensions represent?

Since we can't decompose AlphaEarth back to source modalities, we analyze:
  1. Temporal behavior of top dimensions
  2. Correlation structure among dimensions
  3. Change patterns (which dims accelerate most?)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from src.utils import EarthEngineClient, get_config


def analyze_dimension_temporal_behavior(cleared_trajectories, intact_trajectories, top_dims):
    """
    Analyze how top discriminative dimensions change over time.

    Args:
        cleared_trajectories: Cleared pixel trajectories with Y-2, Y-1, Y embeddings
        intact_trajectories: Intact pixel trajectories
        top_dims: List of dimension indices to analyze

    Returns:
        dict with temporal behavior analysis
    """
    print("\n" + "="*80)
    print("DIMENSION TEMPORAL BEHAVIOR ANALYSIS")
    print("="*80)

    results = {}

    for dim_idx in top_dims[:10]:  # Analyze top 10
        print(f"\n--- Dimension {dim_idx} ---")

        # Extract values for this dimension across years
        cleared_2018 = np.array([t['embeddings'][2018][dim_idx] for t in cleared_trajectories])
        cleared_2019 = np.array([t['embeddings'][2019][dim_idx] for t in cleared_trajectories])
        cleared_2020 = np.array([t['embeddings'][2020][dim_idx] for t in cleared_trajectories])

        intact_2018 = np.array([t['embeddings'][2018][dim_idx] for t in intact_trajectories])
        intact_2019 = np.array([t['embeddings'][2019][dim_idx] for t in intact_trajectories])
        intact_2020 = np.array([t['embeddings'][2020][dim_idx] for t in intact_trajectories])

        # Temporal changes
        cleared_change_2018_2019 = cleared_2019 - cleared_2018
        cleared_change_2019_2020 = cleared_2020 - cleared_2019

        intact_change_2018_2019 = intact_2019 - intact_2018
        intact_change_2019_2020 = intact_2020 - intact_2019

        # Mean values and changes
        cleared_mean_2019 = np.mean(cleared_2019)
        intact_mean_2019 = np.mean(intact_2019)

        cleared_mean_change_y1_y = np.mean(cleared_change_2019_2020)
        intact_mean_change_y1_y = np.mean(intact_change_2019_2020)

        # Test for directional change (increasing vs decreasing)
        direction = "increasing" if cleared_mean_change_y1_y > 0 else "decreasing"

        # Test acceleration
        cleared_acceleration = np.mean(cleared_change_2019_2020) - np.mean(cleared_change_2018_2019)

        print(f"  Cleared Y-1 mean: {cleared_mean_2019:.4f}")
        print(f"  Intact Y-1 mean:  {intact_mean_2019:.4f}")
        print(f"  Cleared Y-1→Y change: {cleared_mean_change_y1_y:.4f} ({direction})")
        print(f"  Intact Y-1→Y change:  {intact_mean_change_y1_y:.4f}")
        print(f"  Cleared acceleration: {cleared_acceleration:.4f}")

        results[f"dim_{dim_idx}"] = {
            'dimension': int(dim_idx),
            'cleared_y1_mean': float(cleared_mean_2019),
            'intact_y1_mean': float(intact_mean_2019),
            'cleared_y1_to_y_change': float(cleared_mean_change_y1_y),
            'intact_y1_to_y_change': float(intact_mean_change_y1_y),
            'direction': direction,
            'acceleration': float(cleared_acceleration),
        }

    return results


def create_dimension_visualizations(cleared_trajectories, intact_trajectories, top_dims):
    """
    Visualize top dimensions' temporal behavior.
    """
    print("\n" + "="*80)
    print("CREATING DIMENSION VISUALIZATIONS")
    print("="*80)

    # Select top 6 dimensions to visualize
    dims_to_plot = top_dims[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    years = [2018, 2019, 2020]

    for idx, dim_idx in enumerate(dims_to_plot):
        ax = axes[idx]

        # Extract values for this dimension
        cleared_2018 = [t['embeddings'][2018][dim_idx] for t in cleared_trajectories]
        cleared_2019 = [t['embeddings'][2019][dim_idx] for t in cleared_trajectories]
        cleared_2020 = [t['embeddings'][2020][dim_idx] for t in cleared_trajectories]

        intact_2018 = [t['embeddings'][2018][dim_idx] for t in intact_trajectories]
        intact_2019 = [t['embeddings'][2019][dim_idx] for t in intact_trajectories]
        intact_2020 = [t['embeddings'][2020][dim_idx] for t in intact_trajectories]

        # Plot individual trajectories (faded)
        for i in range(len(cleared_trajectories)):
            ax.plot(years, [cleared_2018[i], cleared_2019[i], cleared_2020[i]],
                   color='coral', alpha=0.2, linewidth=1)

        for i in range(len(intact_trajectories)):
            ax.plot(years, [intact_2018[i], intact_2019[i], intact_2020[i]],
                   color='green', alpha=0.2, linewidth=1)

        # Plot mean trajectories (bold)
        cleared_means = [np.mean(cleared_2018), np.mean(cleared_2019), np.mean(cleared_2020)]
        intact_means = [np.mean(intact_2018), np.mean(intact_2019), np.mean(intact_2020)]

        ax.plot(years, cleared_means, color='coral', linewidth=3, marker='o',
               markersize=8, label='Cleared (2020)')
        ax.plot(years, intact_means, color='darkgreen', linewidth=3, marker='s',
               markersize=8, label='Intact')

        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel(f'Dimension {dim_idx} Value', fontsize=10)
        ax.set_title(f'Dimension {dim_idx} Temporal Evolution', fontsize=11, fontweight='bold')
        ax.set_xticks(years)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Top Discriminative Dimensions: Temporal Evolution',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "dimension_temporal_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    return plot_file


def run_dimension_analysis(n_samples=30):
    """
    Main analysis workflow.
    """
    print("="*80)
    print("DEEP-DIVE INVESTIGATION: DIMENSION ANALYSIS")
    print("="*80)
    print("\nGoal: Understand what discriminative dimensions encode")
    print("  - Analyze temporal behavior of top dimensions")
    print("  - Identify directional patterns (increasing/decreasing)")
    print("  - Test which dimensions show strongest acceleration\n")

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

    # Extract multi-year embeddings (reuse from temporal_trajectories.py logic)
    print("\nStep 2: Extracting multi-year embeddings (2018, 2019, 2020)...")

    years = [2018, 2019, 2020]

    def extract_multi_year(pixels, label):
        results = []
        for pixel in tqdm(pixels, desc=f"  {label.capitalize()}"):
            lat, lon = pixel['lat'], pixel['lon']
            pixel_data = {'lat': lat, 'lon': lon, 'label': label, 'embeddings': {}}
            success = True
            for year in years:
                try:
                    emb = client.get_embedding(lat=lat, lon=lon, date=f"{year}-06-01")
                    pixel_data['embeddings'][year] = emb
                except Exception:
                    success = False
                    break
            if success:
                results.append(pixel_data)
        return results

    cleared_trajectories = extract_multi_year(cleared_pixels, 'cleared')
    intact_trajectories = extract_multi_year(intact_pixels, 'intact')

    print(f"\n  ✓ Extracted {len(cleared_trajectories)} cleared trajectories")
    print(f"  ✓ Extracted {len(intact_trajectories)} intact trajectories")

    if len(cleared_trajectories) < 5 or len(intact_trajectories) < 5:
        print(f"\n✗ ERROR: Insufficient trajectories")
        return None

    # Identify top discriminative dimensions from Y-1 data
    print("\nStep 3: Identifying top discriminative dimensions...")

    cleared_embs_y1 = np.array([t['embeddings'][2019] for t in cleared_trajectories])
    intact_embs_y1 = np.array([t['embeddings'][2019] for t in intact_trajectories])

    n_dims = cleared_embs_y1.shape[1]
    dimension_scores = []

    for dim in range(n_dims):
        cleared_vals = cleared_embs_y1[:, dim]
        intact_vals = intact_embs_y1[:, dim]

        pooled_std = np.sqrt((np.std(cleared_vals)**2 + np.std(intact_vals)**2) / 2)
        cohens_d = abs((np.mean(cleared_vals) - np.mean(intact_vals)) / pooled_std) if pooled_std > 0 else 0

        dimension_scores.append({'dimension': dim, 'cohens_d': cohens_d})

    dimension_scores.sort(key=lambda x: x['cohens_d'], reverse=True)

    top_dims = [d['dimension'] for d in dimension_scores[:10]]

    print(f"\n  Top 10 discriminative dimensions (by Cohen's d):")
    for i, d in enumerate(dimension_scores[:10]):
        print(f"    {i+1}. Dimension {d['dimension']}: Cohen's d = {d['cohens_d']:.3f}")

    # Analyze temporal behavior
    print("\nStep 4: Analyzing temporal behavior of top dimensions...")
    dim_results = analyze_dimension_temporal_behavior(
        cleared_trajectories, intact_trajectories, top_dims
    )

    # Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_file = create_dimension_visualizations(
        cleared_trajectories, intact_trajectories, top_dims
    )

    # Save results
    output = {
        'n_cleared': len(cleared_trajectories),
        'n_intact': len(intact_trajectories),
        'embedding_dim': n_dims,
        'top_dimensions': dimension_scores[:20],
        'temporal_analysis': dim_results,
    }

    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    results_file = output_dir / "dimension_analysis.json"

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Saved results: {results_file}")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dimension Analysis")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples per class (default: 30)",
    )

    args = parser.parse_args()

    results = run_dimension_analysis(n_samples=args.n_samples)

    if results is None:
        exit(1)
