"""
Deep-Dive Investigation: Analyze Embedding Structure

Goal: Understand what the 256-dimensional AlphaEarth embeddings encode.

Questions:
  1. Do cleared vs intact pixels cluster differently in embedding space?
  2. What dimensions drive the separation?
  3. Are embeddings temporally smooth (Y-2 → Y-1 → Y)?
  4. Can we identify what semantic information is encoded?

Methods:
  - PCA: Linear dimensionality reduction
  - t-SNE: Nonlinear visualization
  - Dimension analysis: Which dimensions discriminate best?
  - Temporal trajectories: How do embeddings change over time?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.utils import EarthEngineClient, get_config


def extract_embeddings(client, pixels, years, label):
    """
    Extract embeddings for pixels across multiple years.

    Args:
        client: EarthEngineClient
        pixels: List of pixel dicts
        years: List of years to extract (e.g., [2018, 2019, 2020])
        label: 'cleared' or 'intact'

    Returns:
        dict with embeddings and metadata
    """
    print(f"\nExtracting embeddings for {len(pixels)} {label} pixels across {len(years)} years...")

    results = []

    for pixel in tqdm(pixels, desc=f"  {label.capitalize()}"):
        lat, lon = pixel['lat'], pixel['lon']

        pixel_data = {
            'lat': lat,
            'lon': lon,
            'label': label,
            'embeddings': {}
        }

        # Extract for each year
        success = True
        for year in years:
            try:
                emb = client.get_embedding(lat=lat, lon=lon, date=f"{year}-06-01")
                pixel_data['embeddings'][year] = emb
            except Exception as e:
                print(f"\n  Warning: Failed to get embedding for {label} pixel ({lat:.4f}, {lon:.4f}), year {year}: {e}")
                success = False
                break

        if success:
            results.append(pixel_data)

    print(f"  ✓ Successfully extracted embeddings for {len(results)}/{len(pixels)} pixels")

    return results


def analyze_dimension_separation(cleared_embs, intact_embs):
    """
    Analyze which embedding dimensions best separate cleared vs intact.

    Args:
        cleared_embs: Array of shape (n_cleared, 256)
        intact_embs: Array of shape (n_intact, 256)

    Returns:
        dict with dimension analysis
    """
    print("\n" + "="*80)
    print("DIMENSION SEPARATION ANALYSIS")
    print("="*80)

    n_dims = cleared_embs.shape[1]

    # For each dimension, calculate effect size (Cohen's d)
    dimension_scores = []

    for dim in range(n_dims):
        cleared_vals = cleared_embs[:, dim]
        intact_vals = intact_embs[:, dim]

        # Cohen's d
        pooled_std = np.sqrt((np.std(cleared_vals)**2 + np.std(intact_vals)**2) / 2)
        if pooled_std > 0:
            cohens_d = abs((np.mean(cleared_vals) - np.mean(intact_vals)) / pooled_std)
        else:
            cohens_d = 0

        # T-test
        t_stat, p_value = stats.ttest_ind(cleared_vals, intact_vals)

        dimension_scores.append({
            'dimension': dim,
            'cohens_d': cohens_d,
            'p_value': p_value,
            't_statistic': abs(t_stat),
        })

    # Sort by effect size
    dimension_scores.sort(key=lambda x: x['cohens_d'], reverse=True)

    # Report top discriminative dimensions
    print(f"\nTop 10 most discriminative dimensions (by Cohen's d):")
    cohens_label = "Cohen's d"
    print(f"{'Dim':<6} {cohens_label:<12} {'|t|':<12} {'p-value':<12}")
    print("-" * 50)

    for i, score in enumerate(dimension_scores[:10]):
        dim = score['dimension']
        d = score['cohens_d']
        t = score['t_statistic']
        p = score['p_value']
        print(f"{dim:<6} {d:<12.4f} {t:<12.4f} {p:<12.6f}")

    # Summary statistics
    num_significant = sum(1 for s in dimension_scores if s['p_value'] < 0.05)
    num_large_effect = sum(1 for s in dimension_scores if s['cohens_d'] > 0.8)

    print(f"\nSummary:")
    print(f"  Dimensions with p < 0.05: {num_significant}/{n_dims} ({100*num_significant/n_dims:.1f}%)")
    print(f"  Dimensions with |d| > 0.8: {num_large_effect}/{n_dims} ({100*num_large_effect/n_dims:.1f}%)")

    return dimension_scores


def perform_pca_analysis(cleared_embs, intact_embs):
    """
    Perform PCA and visualize embedding space.

    Args:
        cleared_embs: Array of shape (n_cleared, 256)
        intact_embs: Array of shape (n_intact, 256)

    Returns:
        dict with PCA results
    """
    print("\n" + "="*80)
    print("PCA ANALYSIS")
    print("="*80)

    # Combine data
    X = np.vstack([cleared_embs, intact_embs])
    y = np.array([0]*len(cleared_embs) + [1]*len(intact_embs))  # 0=cleared, 1=intact

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_components = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])  # Can't exceed n_samples or n_features
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    print(f"\nExplained variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
    if len(cumsum_variance) > 1:
        print(f"  PC1-2: {cumsum_variance[1]:.3f}")
    if len(cumsum_variance) > 4:
        print(f"  PC1-5: {cumsum_variance[4]:.3f}")
    if len(cumsum_variance) > 9:
        print(f"  PC1-10: {cumsum_variance[9]:.3f}")
    if len(cumsum_variance) > 19:
        print(f"  PC1-20: {cumsum_variance[19]:.3f}")

    # Separate by class
    cleared_pca = X_pca[y == 0]
    intact_pca = X_pca[y == 1]

    # Test separation on first few PCs
    print(f"\nSeparation on principal components:")
    for pc in range(min(5, X_pca.shape[1])):
        cleared_vals = cleared_pca[:, pc]
        intact_vals = intact_pca[:, pc]

        t_stat, p_value = stats.ttest_ind(cleared_vals, intact_vals)
        cohens_d = abs((np.mean(cleared_vals) - np.mean(intact_vals)) /
                      np.sqrt((np.std(cleared_vals)**2 + np.std(intact_vals)**2) / 2))

        print(f"  PC{pc+1}: Cohen's d = {cohens_d:.3f}, p = {p_value:.6f}")

    return {
        'pca_model': pca,
        'X_pca': X_pca,
        'y': y,
        'scaler': scaler,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumsum_variance,
    }


def perform_tsne_analysis(cleared_embs, intact_embs, pca_result):
    """
    Perform t-SNE for nonlinear visualization.

    Args:
        cleared_embs: Array of shape (n_cleared, 256)
        intact_embs: Array of shape (n_intact, 256)
        pca_result: PCA results (use first 50 PCs as input to t-SNE)

    Returns:
        dict with t-SNE results
    """
    print("\n" + "="*80)
    print("t-SNE ANALYSIS")
    print("="*80)

    # Use all available PCs as input
    X_pca_all = pca_result['X_pca']
    y = pca_result['y']

    # t-SNE
    print(f"\nRunning t-SNE on {X_pca_all.shape[1]} principal components...")
    perplexity = min(5, max(2, len(X_pca_all)//4))  # Adjust for small samples
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_pca_all)

    print("  ✓ t-SNE complete")

    return {
        'X_tsne': X_tsne,
        'y': y,
    }


def create_visualizations(pca_result, tsne_result):
    """
    Create comprehensive visualizations of embedding space.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))

    # Layout: 2x3 grid
    # Row 1: PCA plots
    # Row 2: t-SNE and variance plots

    # 1. PCA: PC1 vs PC2
    ax1 = plt.subplot(2, 3, 1)
    X_pca = pca_result['X_pca']
    y = pca_result['y']

    cleared_pca = X_pca[y == 0]
    intact_pca = X_pca[y == 1]

    ax1.scatter(cleared_pca[:, 0], cleared_pca[:, 1],
               c='coral', s=80, alpha=0.6, label='Cleared (2020)', edgecolors='black', linewidths=0.5)
    ax1.scatter(intact_pca[:, 0], intact_pca[:, 1],
               c='lightgreen', s=80, alpha=0.6, label='Intact', edgecolors='black', linewidths=0.5)
    ax1.set_xlabel(f'PC1 ({pca_result["explained_variance_ratio"][0]:.1%} variance)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({pca_result["explained_variance_ratio"][1]:.1%} variance)', fontsize=10)
    ax1.set_title('PCA: PC1 vs PC2', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. PCA: PC1 vs PC3
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(cleared_pca[:, 0], cleared_pca[:, 2],
               c='coral', s=80, alpha=0.6, label='Cleared (2020)', edgecolors='black', linewidths=0.5)
    ax2.scatter(intact_pca[:, 0], intact_pca[:, 2],
               c='lightgreen', s=80, alpha=0.6, label='Intact', edgecolors='black', linewidths=0.5)
    ax2.set_xlabel(f'PC1 ({pca_result["explained_variance_ratio"][0]:.1%} variance)', fontsize=10)
    ax2.set_ylabel(f'PC3 ({pca_result["explained_variance_ratio"][2]:.1%} variance)', fontsize=10)
    ax2.set_title('PCA: PC1 vs PC3', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. PCA: PC2 vs PC3
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(cleared_pca[:, 1], cleared_pca[:, 2],
               c='coral', s=80, alpha=0.6, label='Cleared (2020)', edgecolors='black', linewidths=0.5)
    ax3.scatter(intact_pca[:, 1], intact_pca[:, 2],
               c='lightgreen', s=80, alpha=0.6, label='Intact', edgecolors='black', linewidths=0.5)
    ax3.set_xlabel(f'PC2 ({pca_result["explained_variance_ratio"][1]:.1%} variance)', fontsize=10)
    ax3.set_ylabel(f'PC3 ({pca_result["explained_variance_ratio"][2]:.1%} variance)', fontsize=10)
    ax3.set_title('PCA: PC2 vs PC3', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. t-SNE
    ax4 = plt.subplot(2, 3, 4)
    X_tsne = tsne_result['X_tsne']
    y_tsne = tsne_result['y']

    cleared_tsne = X_tsne[y_tsne == 0]
    intact_tsne = X_tsne[y_tsne == 1]

    ax4.scatter(cleared_tsne[:, 0], cleared_tsne[:, 1],
               c='coral', s=80, alpha=0.6, label='Cleared (2020)', edgecolors='black', linewidths=0.5)
    ax4.scatter(intact_tsne[:, 0], intact_tsne[:, 1],
               c='lightgreen', s=80, alpha=0.6, label='Intact', edgecolors='black', linewidths=0.5)
    ax4.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax4.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax4.set_title('t-SNE Visualization', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Explained variance (scree plot)
    ax5 = plt.subplot(2, 3, 5)
    n_pcs = len(pca_result['explained_variance_ratio'])
    ax5.plot(range(1, n_pcs+1), pca_result['explained_variance_ratio'],
            marker='o', linewidth=2, markersize=4)
    ax5.set_xlabel('Principal Component', fontsize=10)
    ax5.set_ylabel('Explained Variance Ratio', fontsize=10)
    ax5.set_title('PCA Scree Plot', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    # 6. Cumulative explained variance
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(range(1, n_pcs+1), pca_result['cumulative_variance'],
            linewidth=2, color='green')
    ax6.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='95% threshold')
    ax6.set_xlabel('Number of Components', fontsize=10)
    ax6.set_ylabel('Cumulative Explained Variance', fontsize=10)
    ax6.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.suptitle('Embedding Space Analysis: Cleared vs Intact Pixels (Y-1)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_file = output_dir / "embedding_structure_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {plot_file}")

    return plot_file


def run_embedding_analysis(n_samples=30):
    """
    Main analysis workflow.
    """
    print("="*80)
    print("DEEP-DIVE INVESTIGATION: EMBEDDING STRUCTURE ANALYSIS")
    print("="*80)
    print("\nGoal: Understand what AlphaEarth embeddings encode")
    print("  - Do cleared vs intact cluster differently?")
    print("  - What dimensions drive separation?")
    print("  - PCA and t-SNE visualization\n")

    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Sample pixels (same as before)
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

    # Extract embeddings for Y-1 (2019)
    # We focus on Y-1 since that's what Test 4 used
    print("\nStep 2: Extracting Y-1 (2019) embeddings...")

    years = [2019]  # Focus on Y-1

    cleared_data = extract_embeddings(client, cleared_pixels, years, 'cleared')
    intact_data = extract_embeddings(client, intact_pixels, years, 'intact')

    if len(cleared_data) < 5 or len(intact_data) < 5:
        print(f"\n✗ ERROR: Insufficient embeddings extracted")
        return None

    # Extract embedding arrays
    cleared_embs = np.array([d['embeddings'][2019] for d in cleared_data])
    intact_embs = np.array([d['embeddings'][2019] for d in intact_data])

    print(f"\nEmbedding arrays:")
    print(f"  Cleared: {cleared_embs.shape}")
    print(f"  Intact: {intact_embs.shape}")

    # Analyze dimension separation
    print("\nStep 3: Analyzing dimension separation...")
    dimension_scores = analyze_dimension_separation(cleared_embs, intact_embs)

    # PCA analysis
    print("\nStep 4: PCA analysis...")
    pca_result = perform_pca_analysis(cleared_embs, intact_embs)

    # t-SNE analysis
    print("\nStep 5: t-SNE analysis...")
    tsne_result = perform_tsne_analysis(cleared_embs, intact_embs, pca_result)

    # Visualizations
    print("\nStep 6: Creating visualizations...")
    plot_file = create_visualizations(pca_result, tsne_result)

    # Save results
    pca_summary = {
        'n_components': len(pca_result['explained_variance_ratio']),
        'variance_pc1': float(pca_result['explained_variance_ratio'][0]),
    }
    if len(pca_result['cumulative_variance']) > 4:
        pca_summary['variance_pc1_5'] = float(pca_result['cumulative_variance'][4])
    if len(pca_result['cumulative_variance']) > 9:
        pca_summary['variance_pc1_10'] = float(pca_result['cumulative_variance'][9])

    output = {
        'n_cleared': len(cleared_data),
        'n_intact': len(intact_data),
        'embedding_dim': cleared_embs.shape[1],
        'top_discriminative_dimensions': dimension_scores[:20],
        'pca_summary': pca_summary,
    }

    output_dir = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/deep_dive")
    results_file = output_dir / "embedding_structure_analysis.json"

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✓ Saved results: {results_file}")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Structure Analysis")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples per class (default: 30)",
    )

    args = parser.parse_args()

    results = run_embedding_analysis(n_samples=args.n_samples)

    if results is None:
        exit(1)
