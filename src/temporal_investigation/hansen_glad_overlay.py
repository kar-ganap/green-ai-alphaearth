"""
Hansen-GLAD Overlay Analysis

OBJECTIVE: Combine Hansen (complete coverage) with GLAD (temporal precision)

Strategy:
1. Sample Hansen cleared pixels (e.g., 200 pixels cleared in 2020)
2. For each Hansen pixel, check if GLAD alert exists at same location
3. Group into:
   - WITH GLAD: Has quarterly timing (fire-based or detectable)
   - WITHOUT GLAD: No timing (likely logging, degradation, or missed)
4. Test AlphaEarth signal for both groups
5. Determine if we can predict BOTH groups (all deforestation) or just GLAD subset

This answers:
- What % of Hansen clearings have GLAD alerts? (overlap rate)
- Does AlphaEarth detect non-GLAD clearings? (logging capability)
- Can we determine lead time for GLAD subset while validating on all Hansen?
"""

import ee
import numpy as np
import json
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.earth_engine import EarthEngineClient
from src.utils import get_config

def sample_hansen_clearings_from_hotspots(client, hotspots, year, target_per_hotspot=50):
    """
    Sample Hansen cleared pixels from deforestation hotspots

    Args:
        hotspots: List of smaller bounds dicts (deforestation hotspots)
        year: Year to sample clearings from
        target_per_hotspot: Target number of pixels per hotspot

    Returns:
        List of Hansen clearing locations
    """
    print(f"\n{'='*80}")
    print(f"SAMPLING HANSEN CLEARED PIXELS FROM HOTSPOTS")
    print(f"{'='*80}\n")

    print(f"Target: {target_per_hotspot} Hansen clearings per hotspot")
    print(f"Hotspots: {len(hotspots)}")
    print(f"Total target: ~{len(hotspots) * target_per_hotspot} pixels\n")

    all_clearings = []

    for i, hotspot_bounds in enumerate(hotspots):
        print(f"{'='*60}")
        print(f"HOTSPOT {i+1}/{len(hotspots)}: {hotspot_bounds.get('name', 'Unnamed')}")
        print(f"{'='*60}")
        print(f"Bounds: lon [{hotspot_bounds['min_lon']}, {hotspot_bounds['max_lon']}], "
              f"lat [{hotspot_bounds['min_lat']}, {hotspot_bounds['max_lat']}]")

        try:
            clearings = client.get_deforestation_labels(
                bounds=hotspot_bounds,
                year=year,
                min_tree_cover=30
            )

            print(f"  Found {len(clearings)} clearings")

            # Sample target count for this hotspot
            if len(clearings) > target_per_hotspot:
                import random
                random.seed(42 + i)  # Different seed per hotspot
                clearings = random.sample(clearings, target_per_hotspot)
                print(f"  Sampled {len(clearings)} clearings (target: {target_per_hotspot})")

            all_clearings.extend(clearings)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"TOTAL HANSEN CLEARINGS SAMPLED: {len(all_clearings)}")
    print(f"{'='*80}\n")

    return all_clearings

def check_glad_overlay(hansen_pixels, year):
    """
    For each Hansen pixel, check if GLAD alert exists

    Returns:
        List of Hansen pixels enriched with GLAD info (if available)
    """
    print(f"\n{'='*80}")
    print(f"CHECKING GLAD OVERLAY")
    print(f"{'='*80}\n")

    print(f"Checking {len(hansen_pixels)} Hansen pixels for GLAD alerts...")

    # GLAD dataset
    if year >= 2024:
        dataset_id = 'projects/glad/alert/UpdResult'
    else:
        dataset_id = f'projects/glad/alert/{year}final'

    glad_collection = ee.ImageCollection(dataset_id)

    year_suffix = str(year % 100)
    alert_date_band = f'alertDate{year_suffix}'
    conf_band = f'conf{year_suffix}'

    glad = glad_collection.select([alert_date_band, conf_band]).mosaic()

    enriched = []

    for i, pixel in enumerate(hansen_pixels):
        if (i + 1) % 50 == 0:
            print(f"  Processing pixel {i+1}/{len(hansen_pixels)}...")

        lat = pixel['lat']
        lon = pixel['lon']

        try:
            # Sample GLAD at this location
            point = ee.Geometry.Point([lon, lat])
            sample = glad.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30
            ).getInfo()

            date_value = sample.get(alert_date_band)
            conf_value = sample.get(conf_band)

            # Determine if GLAD alert exists
            if date_value is not None and date_value > 0:
                # GLAD alert exists - convert to date
                alert_date = datetime(year, 1, 1) + timedelta(days=int(date_value) - 1)
                quarter = (alert_date.month - 1) // 3 + 1

                pixel_data = {
                    **pixel,
                    'has_glad': True,
                    'glad_date': alert_date.isoformat(),
                    'glad_quarter': quarter,
                    'glad_month': alert_date.month,
                    'glad_julian_day': int(date_value),
                    'glad_confidence': conf_value
                }
            else:
                # No GLAD alert
                pixel_data = {
                    **pixel,
                    'has_glad': False,
                    'glad_date': None,
                    'glad_quarter': None
                }

            enriched.append(pixel_data)

        except Exception as e:
            print(f"    Error at ({lat}, {lon}): {e}")
            # Add without GLAD
            enriched.append({
                **pixel,
                'has_glad': False,
                'glad_date': None,
                'glad_quarter': None
            })

    # Summary
    with_glad = sum(1 for p in enriched if p['has_glad'])
    without_glad = len(enriched) - with_glad
    overlap_rate = 100 * with_glad / len(enriched) if len(enriched) > 0 else 0

    print(f"\n{'='*80}")
    print(f"OVERLAY RESULTS")
    print(f"{'='*80}\n")

    print(f"Total Hansen clearings: {len(enriched)}")
    print(f"  WITH GLAD alert: {with_glad} ({overlap_rate:.1f}%)")
    print(f"  WITHOUT GLAD alert: {without_glad} ({100-overlap_rate:.1f}%)")

    # Quarterly distribution for GLAD subset
    if with_glad > 0:
        print(f"\nQuarterly distribution (GLAD subset only):")
        quarter_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for p in enriched:
            if p['has_glad']:
                quarter_counts[p['glad_quarter']] += 1

        for q in [1, 2, 3, 4]:
            count = quarter_counts[q]
            pct = 100 * count / with_glad if with_glad > 0 else 0
            print(f"  Q{q}: {count:3d} ({pct:5.1f}%)")

    return enriched

def extract_embeddings_for_both_groups(client, pixels):
    """
    Extract AlphaEarth embeddings for both GLAD and non-GLAD groups

    CRITICAL FIX: Use BEFORE vs DURING/AFTER embeddings (detection), not Y-2 vs Y-1 (precursor)

    For 2020 clearing:
    - emb_before = 2019-06-01 (year BEFORE clearing)
    - emb_during = 2020-06-01 (year OF clearing)
    - distance = change from before to during/after

    This measures DETECTION (can we detect the clearing), not prediction (precursor signals)

    Returns:
        Two lists: with_glad_embeddings, without_glad_embeddings
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING EMBEDDINGS (DETECTION: BEFORE vs DURING/AFTER)")
    print(f"{'='*80}\n")

    print("Strategy: For each clearing year Y,")
    print("  - emb_before = (Y-1)-06-01 (before clearing)")
    print("  - emb_during = Y-06-01 (during/after clearing)")
    print("  - distance = ||emb_during - emb_before||")
    print()

    with_glad = []
    without_glad = []

    for i, pixel in enumerate(pixels):
        if (i + 1) % 20 == 0:
            print(f"  Processing pixel {i+1}/{len(pixels)}...")

        lat = pixel['lat']
        lon = pixel['lon']
        year = pixel['year']

        try:
            # CORRECTED: Extract BEFORE vs DURING/AFTER embeddings
            emb_before = client.get_embedding(lat=lat, lon=lon, date=f"{year-1}-06-01")
            emb_during = client.get_embedding(lat=lat, lon=lon, date=f"{year}-06-01")

            # Calculate distance (before → during change)
            distance = np.linalg.norm(np.array(emb_during) - np.array(emb_before))

            pixel_data = {
                **pixel,
                'embedding_before': emb_before,
                'embedding_during': emb_during,
                'distance_before_to_during': distance
            }

            if pixel['has_glad']:
                with_glad.append(pixel_data)
            else:
                without_glad.append(pixel_data)

        except Exception as e:
            print(f"    Error at ({lat}, {lon}): {e}")
            continue

    print(f"\nSuccessfully extracted embeddings:")
    print(f"  WITH GLAD: {len(with_glad)} pixels")
    print(f"  WITHOUT GLAD: {len(without_glad)} pixels")

    return with_glad, without_glad

def analyze_quarterly_signals(with_glad_pixels, intact_pixels):
    """
    Analyze signal strength by quarter

    CRITICAL TEST: Does Q4 show precursor signals?
    """
    print("Testing quarterly precursor signals...")
    print("  Q2-Q3 (Apr-Sep): Concurrent with June embedding → detection")
    print("  Q4 (Oct-Dec): 4-6 months after June → PRECURSOR TEST")
    print()

    # Group by quarter
    by_quarter = {1: [], 2: [], 3: [], 4: []}

    for p in with_glad_pixels:
        q = p.get('glad_quarter')
        if q and q in by_quarter:
            by_quarter[q].append(p)

    # Intact baseline
    intact_dists = [p['distance_before_to_during'] for p in intact_pixels]
    intact_mean = np.mean(intact_dists)
    intact_std = np.std(intact_dists, ddof=1) if len(intact_dists) > 1 else 0

    print(f"Intact baseline: {intact_mean:.3f} ± {intact_std:.3f} (n={len(intact_pixels)})\n")

    results = {}

    for q in [1, 2, 3, 4]:
        pixels = by_quarter[q]

        if len(pixels) == 0:
            print(f"Q{q}: No data")
            results[f'Q{q}'] = {'n': 0}
            continue

        dists = [p['distance_before_to_during'] for p in pixels]
        mean = np.mean(dists)
        std = np.std(dists, ddof=1) if len(dists) > 1 else 0

        # Statistical test vs intact
        if len(dists) >= 2 and len(intact_dists) >= 2:
            t_stat, p_value = stats.ttest_ind(dists, intact_dists, equal_var=False)
            pooled_std = np.sqrt((std**2 + intact_std**2) / 2)
            cohen_d = (mean - intact_mean) / pooled_std if pooled_std > 0 else 0
        else:
            t_stat, p_value, cohen_d = np.nan, np.nan, np.nan

        significant = p_value < 0.05 if not np.isnan(p_value) else False

        # Interpretation
        if q in [2, 3]:
            interpretation = "Concurrent detection (0-3mo)"
        elif q == 4:
            interpretation = "PRECURSOR TEST (4-6mo)"
        else:
            interpretation = "Too far (9-12mo)"

        results[f'Q{q}'] = {
            'n': len(pixels),
            'mean': float(mean),
            'std': float(std),
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'cohens_d': float(cohen_d) if not np.isnan(cohen_d) else None,
            'significant': bool(significant),  # Convert to native Python bool
            'interpretation': interpretation,
            'distances': [float(d) for d in dists],  # Save actual distances!
        }

        print(f"{'='*70}")
        print(f"Q{q}: {interpretation}")
        print(f"{'='*70}")
        print(f"  n = {len(pixels)}")
        print(f"  Distance: {mean:.3f} ± {std:.3f}")
        print(f"  vs Intact: {mean/intact_mean:.2f}x")
        print(f"  Statistics: t={t_stat:.2f}, p={p_value:.6f}, d={cohen_d:.2f}")
        print(f"  Result: {'✓ SIGNIFICANT' if significant else '✗ NOT SIGNIFICANT'}")

        if q == 4:
            print()
            print("  🔍 CRITICAL: Q4 Precursor Test!")
            if significant:
                print(f"  ✅ Q4 SHOWS SIGNAL → PRECURSOR CAPABILITY EXISTS!")
                print(f"     Distance: {mean:.3f} >> Intact: {intact_mean:.3f}")
                print(f"     Effect size: d={cohen_d:.2f}")
                print(f"     → 4-6 month lead time for {len(pixels)} pixels ({len(pixels)/len(with_glad_pixels)*100:.1f}%)")
            else:
                print(f"  ❌ Q4 shows NO signal → No precursor capability")
                print(f"     Distance: {mean:.3f} ≈ Intact: {intact_mean:.3f}")
                print(f"     p={p_value:.3f} (not significant)")
        print()

    results['intact_baseline'] = {
        'n': len(intact_pixels),
        'mean': float(intact_mean),
        'std': float(intact_std),
        'distances': [float(d) for d in intact_dists],
    }

    return results


def compare_glad_vs_noglad_signals(with_glad, without_glad, intact):
    """
    Test if AlphaEarth signal differs between GLAD and non-GLAD groups

    Key questions:
    1. Do non-GLAD clearings show temporal signal? (detects logging?)
    2. Is signal stronger for GLAD clearings? (fire vs logging difference?)
    3. Are both groups separable from intact? (both detected?)
    """
    print(f"\n{'='*80}")
    print(f"GLAD vs NON-GLAD SIGNAL COMPARISON")
    print(f"{'='*80}\n")

    # Extract distances (CORRECTED: using before_to_during distances)
    glad_distances = [p['distance_before_to_during'] for p in with_glad]
    noglad_distances = [p['distance_before_to_during'] for p in without_glad]
    intact_distances = [p['distance_before_to_during'] for p in intact]

    # Statistics
    glad_mean = np.mean(glad_distances) if len(glad_distances) > 0 else 0
    glad_std = np.std(glad_distances, ddof=1) if len(glad_distances) > 1 else 0

    noglad_mean = np.mean(noglad_distances) if len(noglad_distances) > 0 else 0
    noglad_std = np.std(noglad_distances, ddof=1) if len(noglad_distances) > 1 else 0

    intact_mean = np.mean(intact_distances) if len(intact_distances) > 0 else 0
    intact_std = np.std(intact_distances, ddof=1) if len(intact_distances) > 1 else 0

    print(f"{'Group':<20} {'N':<6} {'Mean ± Std':<20} {'vs Intact':<15}")
    print("-" * 70)
    print(f"{'WITH GLAD (fire?)':<20} {len(glad_distances):<6} {glad_mean:.4f} ± {glad_std:.4f}   "
          f"{glad_mean - intact_mean:+.4f}")
    print(f"{'WITHOUT GLAD (log?)':<20} {len(noglad_distances):<6} {noglad_mean:.4f} ± {noglad_std:.4f}   "
          f"{noglad_mean - intact_mean:+.4f}")
    print(f"{'Intact (control)':<20} {len(intact_distances):<6} {intact_mean:.4f} ± {intact_std:.4f}   "
          f"(baseline)")

    # Statistical tests
    print(f"\n{'='*80}")
    print(f"STATISTICAL TESTS")
    print(f"{'='*80}\n")

    results = {
        'glad': {'n': len(glad_distances), 'mean': float(glad_mean), 'std': float(glad_std)},
        'noglad': {'n': len(noglad_distances), 'mean': float(noglad_mean), 'std': float(noglad_std)},
        'intact': {'n': len(intact_distances), 'mean': float(intact_mean), 'std': float(intact_std)}
    }

    # Test 1: WITH GLAD vs Intact
    if len(glad_distances) >= 3 and len(intact_distances) >= 3:
        t_stat, p_value = stats.ttest_ind(glad_distances, intact_distances, equal_var=False)
        cohens_d = (glad_mean - intact_mean) / np.sqrt((glad_std**2 + intact_std**2) / 2)

        print(f"1. WITH GLAD vs Intact:")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")

        results['glad_vs_intact'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }

    # Test 2: WITHOUT GLAD vs Intact
    if len(noglad_distances) >= 3 and len(intact_distances) >= 3:
        t_stat, p_value = stats.ttest_ind(noglad_distances, intact_distances, equal_var=False)
        cohens_d = (noglad_mean - intact_mean) / np.sqrt((noglad_std**2 + intact_std**2) / 2)

        print(f"\n2. WITHOUT GLAD vs Intact:")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")

        if p_value < 0.05:
            print(f"\n   ✓ AlphaEarth DETECTS non-GLAD clearings (likely logging/degradation)!")
        else:
            print(f"\n   ✗ AlphaEarth does NOT detect non-GLAD clearings")
            print(f"     → Model may be fire-specific")

        results['noglad_vs_intact'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }

    # Test 3: WITH GLAD vs WITHOUT GLAD
    if len(glad_distances) >= 3 and len(noglad_distances) >= 3:
        t_stat, p_value = stats.ttest_ind(glad_distances, noglad_distances, equal_var=False)
        cohens_d = (glad_mean - noglad_mean) / np.sqrt((glad_std**2 + noglad_std**2) / 2)

        print(f"\n3. WITH GLAD vs WITHOUT GLAD:")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")

        if p_value < 0.05:
            if glad_mean > noglad_mean:
                print(f"\n   ✓ Fire-based clearings show STRONGER signal than logging")
                print(f"     → Fire detection may be easier than logging detection")
            else:
                print(f"\n   ✓ Logging shows STRONGER signal than fire-based")
                print(f"     → Unexpected! Investigate further")
        else:
            print(f"\n   ✗ No significant difference between fire and logging signals")
            print(f"     → AlphaEarth detects both equally well")

        results['glad_vs_noglad'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }

    return results

def create_visualizations(with_glad, without_glad, intact, stats_results):
    """
    Create visualizations comparing GLAD vs non-GLAD groups
    """
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Embedding distances box plot
    ax1 = plt.subplot(2, 3, 1)

    glad_dists = [p['distance_before_to_during'] for p in with_glad]
    noglad_dists = [p['distance_before_to_during'] for p in without_glad]
    intact_dists = [p['distance_before_to_during'] for p in intact]

    data = [glad_dists, noglad_dists, intact_dists]
    labels = ['WITH GLAD\n(Fire?)', 'WITHOUT GLAD\n(Logging?)', 'Intact\n(Control)']
    colors = ['#d62728', '#ff7f0e', '#7f7f7f']

    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Embedding Distance (Y-1 to Y-2)', fontsize=12, fontweight='bold')
    ax1.set_title('AlphaEarth Signal: GLAD vs Non-GLAD vs Intact', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Sample sizes
    ax2 = plt.subplot(2, 3, 2)

    sizes = [len(glad_dists), len(noglad_dists), len(intact_dists)]
    bars = ax2.bar(range(3), sizes, color=colors, alpha=0.7, edgecolor='black')

    ax2.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Sizes by Group', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['WITH GLAD', 'WITHOUT GLAD', 'Intact'])
    ax2.grid(axis='y', alpha=0.3)

    # Annotate bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.02,
                f'n={size}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 3: GLAD overlap rate
    ax3 = plt.subplot(2, 3, 3)

    total_hansen = len(with_glad) + len(without_glad)
    overlap_rate = 100 * len(with_glad) / total_hansen if total_hansen > 0 else 0
    non_overlap_rate = 100 - overlap_rate

    wedges, texts, autotexts = ax3.pie(
        [overlap_rate, non_overlap_rate],
        labels=['WITH GLAD\n(Fire-detectable)', 'WITHOUT GLAD\n(Logging/missed)'],
        colors=['#d62728', '#ff7f0e'],
        autopct='%1.1f%%',
        startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax3.set_title(f'Hansen-GLAD Overlap\n(Total Hansen: {total_hansen})',
                 fontsize=13, fontweight='bold')

    # Plot 4: Quarterly distribution (GLAD subset only)
    ax4 = plt.subplot(2, 3, 4)

    if len(with_glad) > 0:
        quarter_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for p in with_glad:
            if p.get('glad_quarter'):
                quarter_counts[p['glad_quarter']] += 1

        quarters = [1, 2, 3, 4]
        counts = [quarter_counts[q] for q in quarters]
        percentages = [100 * c / len(with_glad) for c in counts]

        bars = ax4.bar(quarters, percentages, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
                      alpha=0.7, edgecolor='black')

        ax4.set_xlabel('Quarter', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Quarterly Distribution\n(GLAD subset only)', fontsize=13, fontweight='bold')
        ax4.set_xticks(quarters)
        ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax4.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 5: Mean comparison
    ax5 = plt.subplot(2, 3, 5)

    means = [stats_results['glad']['mean'],
             stats_results['noglad']['mean'],
             stats_results['intact']['mean']]
    stds = [stats_results['glad']['std'],
            stats_results['noglad']['std'],
            stats_results['intact']['std']]

    bars = ax5.bar(range(3), means, yerr=stds, color=colors, alpha=0.7,
                  edgecolor='black', capsize=10)

    ax5.set_ylabel('Mean Embedding Distance', fontsize=12, fontweight='bold')
    ax5.set_title('Mean ± Std by Group', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(3))
    ax5.set_xticklabels(['WITH GLAD', 'WITHOUT GLAD', 'Intact'])
    ax5.grid(axis='y', alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Build summary text
    summary = f"""
HANSEN-GLAD OVERLAY SUMMARY

Total Hansen Clearings: {total_hansen}
  WITH GLAD alerts: {len(with_glad)} ({overlap_rate:.1f}%)
  WITHOUT GLAD alerts: {len(without_glad)} ({non_overlap_rate:.1f}%)

AlphaEarth Signal Strength:
  WITH GLAD: {stats_results['glad']['mean']:.4f} ± {stats_results['glad']['std']:.4f}
  WITHOUT GLAD: {stats_results['noglad']['mean']:.4f} ± {stats_results['noglad']['std']:.4f}
  Intact: {stats_results['intact']['mean']:.4f} ± {stats_results['intact']['std']:.4f}

Statistical Tests:
"""

    if 'glad_vs_intact' in stats_results:
        sig = stats_results['glad_vs_intact']['significant']
        p_val = stats_results['glad_vs_intact']['p_value']
        summary += f"  WITH GLAD vs Intact: {'✓ SIG' if sig else '✗ NOT SIG'} (p={p_val:.4f})\n"

    if 'noglad_vs_intact' in stats_results:
        sig = stats_results['noglad_vs_intact']['significant']
        p_val = stats_results['noglad_vs_intact']['p_value']
        summary += f"  WITHOUT GLAD vs Intact: {'✓ SIG' if sig else '✗ NOT SIG'} (p={p_val:.4f})\n"

        if sig:
            summary += f"\n  ✓ AlphaEarth DETECTS non-GLAD clearings!\n"
            summary += f"    → Likely detects logging/degradation\n"
        else:
            summary += f"\n  ✗ AlphaEarth does NOT detect non-GLAD\n"
            summary += f"    → May be fire-specific\n"

    if 'glad_vs_noglad' in stats_results:
        sig = stats_results['glad_vs_noglad']['significant']
        p_val = stats_results['glad_vs_noglad']['p_value']
        summary += f"  WITH vs WITHOUT GLAD: {'✓ SIG' if sig else '✗ NOT SIG'} (p={p_val:.4f})\n"

    ax6.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation/hansen_glad_overlay.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

def main():
    """
    Main execution
    """
    print(f"\n{'='*80}")
    print(f"HANSEN-GLAD OVERLAY ANALYSIS")
    print(f"{'='*80}\n")

    print("OBJECTIVE: Combine Hansen (complete) + GLAD (temporal)")
    print("  1. Sample Hansen cleared pixels (all deforestation)")
    print("  2. Check which have GLAD alerts (fire-detectable subset)")
    print("  3. Test AlphaEarth signal for BOTH groups")
    print("  4. Determine: Does AlphaEarth detect logging (non-GLAD)?")
    print()

    # Initialize
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Define deforestation hotspots (smaller regions with high clearing rates)
    hotspots = [
        {
            "name": "Rondônia",
            "min_lon": -63,
            "max_lon": -60,
            "min_lat": -12,
            "max_lat": -9
        },
        {
            "name": "Pará",
            "min_lon": -54,
            "max_lon": -51,
            "min_lat": -7,
            "max_lat": -4
        },
        {
            "name": "Mato Grosso",
            "min_lon": -58,
            "max_lon": -55,
            "min_lat": -13,
            "max_lat": -10
        }
    ]

    # Use multiple years for larger sample size
    years = [2019, 2020, 2021, 2022, 2023, 2024]

    print(f"Strategy: Sample from {len(years)} years × {len(hotspots)} hotspots")
    print(f"Target: ~{len(years) * len(hotspots) * 70} total Hansen pixels\n")

    all_hansen_pixels = []

    for year in years:
        print(f"\n{'='*80}")
        print(f"SAMPLING YEAR {year}")
        print(f"{'='*80}\n")

        hansen_pixels_year = sample_hansen_clearings_from_hotspots(client, hotspots, year, target_per_hotspot=70)
        all_hansen_pixels.extend(hansen_pixels_year)

        print(f"Year {year}: {len(hansen_pixels_year)} pixels")

    hansen_pixels = all_hansen_pixels
    print(f"\n{'='*80}")
    print(f"COMBINED TOTAL: {len(hansen_pixels)} Hansen pixels across {len(years)} years")
    print(f"{'='*80}\n")

    if len(hansen_pixels) < 20:
        print(f"\n⚠️  WARNING: Only {len(hansen_pixels)} Hansen pixels")
        print("Need at least 20 for meaningful analysis")
        return

    # Step 2: Check GLAD overlay (need to handle multiple years)
    print(f"\n{'='*80}")
    print(f"CHECKING GLAD OVERLAY FOR MULTIPLE YEARS")
    print(f"{'='*80}\n")

    enriched_pixels = []
    for year in years:
        year_pixels = [p for p in hansen_pixels if p['year'] == year]
        if len(year_pixels) > 0:
            print(f"\nProcessing {year} pixels ({len(year_pixels)} total)...")
            enriched_year = check_glad_overlay(year_pixels, year)
            enriched_pixels.extend(enriched_year)

    print(f"\n{'='*80}")
    print(f"COMBINED GLAD OVERLAY RESULTS")
    print(f"{'='*80}\n")

    with_glad_count = sum(1 for p in enriched_pixels if p['has_glad'])
    without_glad_count = len(enriched_pixels) - with_glad_count
    overlap_rate = 100 * with_glad_count / len(enriched_pixels) if len(enriched_pixels) > 0 else 0

    print(f"Total Hansen clearings: {len(enriched_pixels)}")
    print(f"  WITH GLAD alert: {with_glad_count} ({overlap_rate:.1f}%)")
    print(f"  WITHOUT GLAD alert: {without_glad_count} ({100-overlap_rate:.1f}%)")

    # Step 3: Extract embeddings
    with_glad, without_glad = extract_embeddings_for_both_groups(client, enriched_pixels)

    # Also sample intact pixels for control (from same hotspots, with year distribution)
    print(f"\n{'='*80}")
    print(f"SAMPLING INTACT FOREST (CONTROL)")
    print(f"{'='*80}\n")

    print("Strategy: Sample intact pixels and assign to years to match cleared distribution")

    intact_pixels_raw = []
    for hotspot in hotspots:
        try:
            pixels = client.get_stable_forest_locations(
                bounds=hotspot,
                n_samples=30  # More samples to distribute across years
            )
            intact_pixels_raw.extend(pixels)
            print(f"  {hotspot['name']}: {len(pixels)} intact pixels")
        except Exception as e:
            print(f"  {hotspot['name']}: Error - {e}")
            continue

    # Distribute intact pixels across years to match cleared distribution
    intact_per_year = len(intact_pixels_raw) // len(years)
    intact_pixels_by_year = {}

    for i, year in enumerate(years):
        start_idx = i * intact_per_year
        end_idx = (i + 1) * intact_per_year if i < len(years) - 1 else len(intact_pixels_raw)
        year_pixels = intact_pixels_raw[start_idx:end_idx]

        # Tag with year for temporal alignment
        for p in year_pixels:
            p['reference_year'] = year

        intact_pixels_by_year[year] = year_pixels
        print(f"  Year {year}: {len(year_pixels)} intact pixels")

    print(f"\nTotal sampled: {len(intact_pixels_raw)} intact pixels")

    # Extract embeddings for intact (use matching year-pairs)
    intact_enriched = []
    for year in years:
        year_pixels = intact_pixels_by_year.get(year, [])

        for pixel in year_pixels:
            try:
                # Use same year-pair as cleared pixels: (year-1) → year
                emb_before = client.get_embedding(lat=pixel['lat'], lon=pixel['lon'], date=f"{year-1}-06-01")
                emb_during = client.get_embedding(lat=pixel['lat'], lon=pixel['lon'], date=f"{year}-06-01")

                distance = np.linalg.norm(np.array(emb_during) - np.array(emb_before))

                intact_enriched.append({
                    **pixel,
                    'year': year,
                    'embedding_before': emb_before,
                    'embedding_during': emb_during,
                    'distance_before_to_during': distance
                })
            except Exception as e:
                continue

    print(f"Successfully extracted: {len(intact_enriched)} intact with embeddings")

    # Show distribution by year
    intact_by_year_counts = {}
    for year in years:
        count = sum(1 for p in intact_enriched if p['year'] == year)
        intact_by_year_counts[year] = count
        print(f"  Year {year}: {count} intact with embeddings")


    # Step 4: Compare signals
    stats_results = compare_glad_vs_noglad_signals(with_glad, without_glad, intact_enriched)

    # Step 5: Create visualizations
    create_visualizations(with_glad, without_glad, intact_enriched, stats_results)

    # Save results
    output_dir = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation'
    os.makedirs(output_dir, exist_ok=True)

    # QUARTERLY ANALYSIS
    print(f"\n{'='*80}")
    print(f"QUARTERLY PRECURSOR SIGNAL ANALYSIS")
    print(f"{'='*80}\n")

    quarterly_stats = analyze_quarterly_signals(with_glad, intact_enriched)

    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'hotspots': hotspots,
            'years': years,
            'embedding_strategy': 'before_vs_during (detection, not precursor)',
            'total_hansen_pixels': len(enriched_pixels),
            'with_glad': len(with_glad),
            'without_glad': len(without_glad),
            'intact': len(intact_enriched),
            'overlap_rate': 100 * len(with_glad) / len(enriched_pixels) if len(enriched_pixels) > 0 else 0
        },
        'statistics': stats_results,
        'quarterly_analysis': quarterly_stats
    }

    output_path = os.path.join(output_dir, 'hansen_glad_overlay.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results: {output_path}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    print(f"Hansen-GLAD Overlap: {results['metadata']['overlap_rate']:.1f}%")
    print(f"  WITH GLAD: {len(with_glad)}")
    print(f"  WITHOUT GLAD: {len(without_glad)}")

    if 'noglad_vs_intact' in stats_results and stats_results['noglad_vs_intact']['significant']:
        print(f"\n✓ CRITICAL FINDING: AlphaEarth DETECTS non-GLAD clearings!")
        print(f"  → Model likely detects ALL deforestation (including logging)")
        print(f"  → GLAD provides temporal precision for {results['metadata']['overlap_rate']:.0f}% of clearings")
        print(f"  → Can use GLAD for lead time analysis on subset")
    else:
        print(f"\n✗ AlphaEarth may be fire-specific")
        print(f"  → Non-GLAD clearings show NO signal")

if __name__ == "__main__":
    main()
