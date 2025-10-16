"""
Quarterly Precursor Signal Test

CRITICAL QUESTION: Does Q4 show precursor signals?

Q2-Q3 clearings (61.5%): June embedding overlaps → concurrent detection (0-3mo)
Q4 clearings (25.6%): June embedding precedes → potential precursor (4-6mo)

If Q4 shows significant signal vs intact → precursors exist!
If Q4 shows no signal → only concurrent detection works
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import json
import os

from src.utils import EarthEngineClient, get_config


def get_quarter_from_date(date_str):
    """
    Extract quarter from date string (julian or ISO format)

    Args:
        date_str: Date string (e.g., '2020-08-15' or julian day)

    Returns:
        Quarter (1-4)
    """
    # Handle different date formats from GLAD
    if isinstance(date_str, str):
        if '-' in date_str:
            # ISO format: 2020-08-15
            month = int(date_str.split('-')[1])
        else:
            # Could be other format, skip for now
            return None
    else:
        return None

    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    else:  # 10, 11, 12
        return 4


def sample_and_enrich_pixels_by_quarter(client, hotspots, years=[2019, 2020, 2021]):
    """
    Sample Hansen clearings, check GLAD overlay, extract embeddings

    Returns pixels with quarterly information
    """
    print(f"\n{'='*80}")
    print(f"SAMPLING HANSEN CLEARINGS BY QUARTER")
    print(f"{'='*80}\n")

    all_pixels = []

    for year in years:
        print(f"\nYear {year}:")

        for hotspot in hotspots:
            try:
                # Get Hansen clearings
                clearings = client.get_deforestation_labels(
                    bounds=hotspot,
                    year=year,
                    min_tree_cover=30
                )

                print(f"  {hotspot['name']}: {len(clearings)} Hansen clearings")

                # For each clearing, check GLAD
                for clearing in clearings:
                    lat = clearing['lat']
                    lon = clearing['lon']

                    # Check for GLAD alert
                    try:
                        glad_alerts = client.get_glad_alerts(
                            lat=lat,
                            lon=lon,
                            year=year
                        )

                        if len(glad_alerts) > 0:
                            # Has GLAD alert - get date and quarter
                            alert = glad_alerts[0]
                            alert_date = alert.get('date', alert.get('alertDate', None))
                            quarter = get_quarter_from_date(alert_date)

                            pixel = {
                                'lat': lat,
                                'lon': lon,
                                'year': year,
                                'has_glad': True,
                                'glad_date': alert_date,
                                'quarter': quarter,
                            }
                        else:
                            # No GLAD alert
                            pixel = {
                                'lat': lat,
                                'lon': lon,
                                'year': year,
                                'has_glad': False,
                                'glad_date': None,
                                'quarter': None,
                            }

                        all_pixels.append(pixel)

                    except Exception as e:
                        continue

            except Exception as e:
                print(f"  {hotspot['name']}: Error - {e}")
                continue

    print(f"\nTotal pixels: {len(all_pixels)}")
    with_glad = [p for p in all_pixels if p['has_glad']]
    print(f"  WITH GLAD: {len(with_glad)}")

    # Extract embeddings for all pixels
    print(f"\nExtracting embeddings...")

    enriched = []
    for i, pixel in enumerate(all_pixels):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_pixels)}...")

        try:
            year = pixel['year']
            emb_before = client.get_embedding(
                lat=pixel['lat'],
                lon=pixel['lon'],
                date=f"{year-1}-06-01"
            )
            emb_during = client.get_embedding(
                lat=pixel['lat'],
                lon=pixel['lon'],
                date=f"{year}-06-01"
            )

            distance = np.linalg.norm(np.array(emb_during) - np.array(emb_before))

            enriched.append({
                **pixel,
                'embedding_before': emb_before,
                'embedding_during': emb_during,
                'distance': distance,
            })
        except Exception as e:
            continue

    print(f"\nEnriched: {len(enriched)} pixels with embeddings")

    return enriched


def test_quarterly_signals(pixels_with_glad, intact_pixels):
    """
    Test signal strength by quarter

    KEY TEST: Does Q4 show significant signal vs intact?
    """
    print(f"\n{'='*80}")
    print(f"QUARTERLY PRECURSOR SIGNAL TEST")
    print(f"{'='*80}\n")

    # Split by quarter
    by_quarter = {1: [], 2: [], 3: [], 4: []}

    for pixel in pixels_with_glad:
        if pixel['quarter'] is not None:
            by_quarter[pixel['quarter']].append(pixel)

    intact_dists = [p['distance'] for p in intact_pixels]
    intact_mean = np.mean(intact_dists)
    intact_std = np.std(intact_dists, ddof=1)

    print(f"Intact baseline: {intact_mean:.3f} ± {intact_std:.3f} (n={len(intact_pixels)})")
    print()

    results = {}

    for q in [1, 2, 3, 4]:
        pixels = by_quarter[q]

        if len(pixels) == 0:
            print(f"Q{q}: No data")
            continue

        dists = [p['distance'] for p in pixels]
        mean = np.mean(dists)
        std = np.std(dists, ddof=1) if len(dists) > 1 else 0

        # Statistical test vs intact
        if len(dists) >= 2 and len(intact_dists) >= 2:
            t_stat, p_value = stats.ttest_ind(dists, intact_dists)
            cohen_d = (mean - intact_mean) / np.sqrt((std**2 + intact_std**2) / 2)
        else:
            t_stat, p_value, cohen_d = np.nan, np.nan, np.nan

        significant = p_value < 0.05 if not np.isnan(p_value) else False

        # Interpretation
        if q in [2, 3]:
            interpretation = "Concurrent detection (0-3 months)"
        elif q == 4:
            interpretation = "PRECURSOR? (4-6 months lead)" if significant else "No precursor signal"
        else:
            interpretation = "Too far from June embedding"

        results[f'Q{q}'] = {
            'n': len(pixels),
            'mean': mean,
            'std': std,
            'vs_intact_p': p_value,
            'vs_intact_t': t_stat,
            'cohen_d': cohen_d,
            'significant': significant,
            'interpretation': interpretation,
        }

        print(f"{'='*70}")
        print(f"Q{q} ({interpretation})")
        print(f"{'='*70}")
        print(f"  n: {len(pixels)}")
        print(f"  Distance: {mean:.3f} ± {std:.3f}")
        print(f"  vs Intact: t={t_stat:.2f}, p={p_value:.6f}, d={cohen_d:.2f}")
        print(f"  Significant: {'✓ YES' if significant else '✗ NO'}")

        if q == 4:
            print()
            print(f"  🔍 CRITICAL: Q4 is the precursor test!")
            if significant:
                print(f"  ✅ Q4 shows signal → PRECURSOR CAPABILITY EXISTS")
                print(f"     → 4-6 month lead time for ~25% of clearings")
            else:
                print(f"  ❌ Q4 shows no signal → Only concurrent detection")
                print(f"     → No precursor capability with annual embeddings")
        print()

    return results


def visualize_quarterly_signals(results, intact_mean, intact_std):
    """
    Visualize quarterly signal comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Signal strength by quarter
    ax1 = axes[0]

    quarters = []
    means = []
    stds = []
    colors = []

    for q in [1, 2, 3, 4]:
        key = f'Q{q}'
        if key in results:
            quarters.append(f'Q{q}')
            means.append(results[key]['mean'])
            stds.append(results[key]['std'])

            # Color by significance
            if results[key]['significant']:
                colors.append('#2ecc71' if q in [2, 3] else '#e74c3c')  # Green for detection, red for precursor
            else:
                colors.append('#95a5a6')  # Gray for not significant

    # Add intact baseline
    quarters.append('Intact')
    means.append(intact_mean)
    stds.append(intact_std)
    colors.append('#3498db')

    bars = ax1.bar(quarters, means, yerr=stds, color=colors, alpha=0.7, capsize=5)

    # Add significance markers
    for i, q in enumerate([1, 2, 3, 4]):
        key = f'Q{q}'
        if key in results and results[key]['significant']:
            ax1.text(i, means[i] + stds[i] + 0.05, '***' if results[key]['vs_intact_p'] < 0.001 else '*',
                    ha='center', fontsize=14, fontweight='bold')

    ax1.axhline(intact_mean, color='#3498db', linestyle='--', linewidth=2, alpha=0.5, label='Intact baseline')
    ax1.set_ylabel('Embedding Distance (Y-1 → Y)', fontsize=12)
    ax1.set_xlabel('Quarter of Clearing', fontsize=12)
    ax1.set_title('Signal Strength by Quarter', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effect sizes (Cohen's d)
    ax2 = axes[1]

    quarters_d = []
    cohens_d = []
    colors_d = []

    for q in [1, 2, 3, 4]:
        key = f'Q{q}'
        if key in results and not np.isnan(results[key]['cohen_d']):
            quarters_d.append(f'Q{q}')
            cohens_d.append(results[key]['cohen_d'])
            colors_d.append(colors[[1, 2, 3, 4].index(q)])

    bars = ax2.bar(quarters_d, cohens_d, color=colors_d, alpha=0.7)

    # Add effect size interpretation lines
    ax2.axhline(0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax2.axhline(0.8, color='gray', linestyle='-', alpha=0.5, label='Large effect')

    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax2.set_xlabel('Quarter of Clearing', fontsize=12)
    ax2.set_title('Effect Size vs Intact', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def main():
    """
    Main execution: Test quarterly precursor signals
    """
    print(f"\n{'='*80}")
    print(f"QUARTERLY PRECURSOR SIGNAL ANALYSIS")
    print(f"{'='*80}\n")

    print("CRITICAL QUESTION: Does Q4 show precursor signals?")
    print()
    print("Framework:")
    print("  Q2 (Apr-Jun): June embedding overlaps → concurrent detection")
    print("  Q3 (Jul-Sep): June embedding overlaps → concurrent detection")
    print("  Q4 (Oct-Dec): June embedding precedes → PRECURSOR TEST (4-6mo)")
    print()

    # Initialize
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Hotspots
    hotspots = [
        {"name": "Rondônia", "min_lon": -63, "max_lon": -60, "min_lat": -12, "max_lat": -9},
        {"name": "Pará", "min_lon": -54, "max_lon": -51, "min_lat": -7, "max_lat": -4},
        {"name": "Mato Grosso", "min_lon": -58, "max_lon": -55, "min_lat": -13, "max_lat": -10}
    ]

    years = [2019, 2020, 2021]

    # Sample and enrich pixels
    all_pixels = sample_and_enrich_pixels_by_quarter(client, hotspots, years)

    # Split into GLAD and intact
    with_glad = [p for p in all_pixels if p['has_glad']]
    without_glad = [p for p in all_pixels if not p['has_glad']]

    # Sample intact pixels
    print(f"\n{'='*80}")
    print(f"SAMPLING INTACT PIXELS")
    print(f"{'='*80}\n")

    intact_pixels = []
    for hotspot in hotspots:
        try:
            pixels = client.get_stable_forest_locations(bounds=hotspot, n_samples=20)
            for pixel in pixels:
                # Use 2020 as reference year
                emb_before = client.get_embedding(lat=pixel['lat'], lon=pixel['lon'], date='2019-06-01')
                emb_during = client.get_embedding(lat=pixel['lat'], lon=pixel['lon'], date='2020-06-01')
                distance = np.linalg.norm(np.array(emb_during) - np.array(emb_before))

                intact_pixels.append({
                    **pixel,
                    'distance': distance,
                })
        except Exception as e:
            continue

    print(f"Intact pixels: {len(intact_pixels)}")

    # Test quarterly signals
    results = test_quarterly_signals(with_glad, intact_pixels)

    # Visualize
    intact_dists = [p['distance'] for p in intact_pixels]
    fig = visualize_quarterly_signals(results, np.mean(intact_dists), np.std(intact_dists, ddof=1))

    # Save
    output_dir = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(os.path.join(output_dir, 'quarterly_precursor_test.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {os.path.join(output_dir, 'quarterly_precursor_test.png')}")

    # Save results
    output_data = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'years': years,
            'hotspots': hotspots,
            'total_pixels': len(all_pixels),
            'with_glad': len(with_glad),
            'intact': len(intact_pixels),
        },
        'quarterly_results': results,
        'interpretation': {
            'q4_significant': results.get('Q4', {}).get('significant', False),
            'precursor_capability': 'YES - 4-6 month lead time for Q4 clearings' if results.get('Q4', {}).get('significant', False) else 'NO - Only concurrent detection (0-3 months)',
        }
    }

    with open(os.path.join(output_dir, 'quarterly_precursor_test.json'), 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"✓ Saved: {os.path.join(output_dir, 'quarterly_precursor_test.json')}")

    # Final verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}\n")

    if results.get('Q4', {}).get('significant', False):
        print("✅ Q4 SHOWS SIGNIFICANT SIGNAL!")
        print()
        print("CONCLUSION: Precursor capability EXISTS")
        print(f"  - Q4 distance: {results['Q4']['mean']:.3f} >> Intact: {np.mean(intact_dists):.3f}")
        print(f"  - Effect size: Cohen's d = {results['Q4']['cohen_d']:.2f}")
        print(f"  - Statistical significance: p = {results['Q4']['vs_intact_p']:.6f}")
        print()
        print("SYSTEM CAPABILITY:")
        print("  - 60% clearings: Concurrent detection (Q2-Q3, 0-3 months)")
        print("  - 25% clearings: Short-term prediction (Q4, 4-6 months)")
        print("  - 15% clearings: Unclear (Q1, too far from June)")
        print()
        print("VALUE PROPOSITION:")
        print("  - Majority: Early detection (0-3 month lead)")
        print("  - Minority: Precursor prediction (4-6 month lead)")
        print("  - Both faster than optical-only methods (6-12 month lag)")
    else:
        print("❌ Q4 shows NO significant signal")
        print()
        print("CONCLUSION: No precursor capability with annual embeddings")
        print(f"  - Q4 distance: {results.get('Q4', {}).get('mean', 0):.3f} ≈ Intact: {np.mean(intact_dists):.3f}")
        print(f"  - Statistical significance: p = {results.get('Q4', {}).get('vs_intact_p', 1):.6f}")
        print()
        print("SYSTEM CAPABILITY:")
        print("  - 60% clearings: Concurrent detection (Q2-Q3, 0-3 months)")
        print("  - 25% clearings: No signal (Q4)")
        print("  - 15% clearings: Unclear (Q1)")
        print()
        print("PIVOT REQUIRED:")
        print("  - Focus on early/concurrent detection only")
        print("  - Don't claim prediction capability")
        print("  - Value is speed: 0-3 months vs 6-12 month lag (optical)")


if __name__ == "__main__":
    main()
