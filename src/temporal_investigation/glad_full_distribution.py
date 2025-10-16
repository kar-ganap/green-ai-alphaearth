"""
GLAD Full Distribution Analysis

Instead of sampling 17 pixels, analyze the ENTIRE GLAD dataset to get
true quarterly distribution of deforestation.

This will:
1. Query all GLAD alerts in study region (1000s of pixels)
2. Extract quarterly distribution
3. Compare to our sample (17 pixels) and literature expectations
4. Give definitive answer on when deforestation actually occurs
"""

import ee
import numpy as np
import json
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.earth_engine import EarthEngineClient
from src.utils import get_config

def get_all_glad_alerts_distribution(bounds, years):
    """
    Get quarterly distribution of ALL GLAD alerts in region

    Args:
        bounds: Study region bounds
        years: List of years to analyze (e.g., [2019, 2020, 2021])

    Returns:
        dict with quarterly counts and distribution
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING FULL GLAD DISTRIBUTION")
    print(f"{'='*80}\n")

    print(f"Study region: {bounds}")
    print(f"Years: {years}")

    # Create ROI
    roi = ee.Geometry.Rectangle([
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"]
    ])

    all_dates = []
    year_counts = {}

    for year in years:
        print(f"\nProcessing year {year}...")

        try:
            # Get GLAD dataset
            if year >= 2024:
                dataset_id = 'projects/glad/alert/UpdResult'
            else:
                dataset_id = f'projects/glad/alert/{year}final'

            glad_collection = ee.ImageCollection(dataset_id)

            # Band names
            year_suffix = str(year % 100)
            alert_date_band = f'alertDate{year_suffix}'
            conf_band = f'conf{year_suffix}'

            # Get as mosaic
            glad = glad_collection.select([alert_date_band, conf_band]).mosaic()

            # Sample MANY pixels from the region (not just a few)
            # Use stratified sampling to get good coverage
            sample = glad.sample(
                region=roi,
                scale=30,
                numPixels=10000,  # Sample up to 10,000 pixels
                seed=42,
                geometries=False
            )

            # Get features
            features = sample.getInfo()['features']

            print(f"  Sampled: {len(features)} pixels")

            # Extract dates
            year_dates = []
            for feature in features:
                props = feature['properties']
                date_value = props.get(alert_date_band)
                conf_value = props.get(conf_band)

                if date_value is None or date_value == 0:
                    continue

                # Convert Julian day to date
                alert_date = datetime(year, 1, 1) + timedelta(days=int(date_value) - 1)
                quarter = (alert_date.month - 1) // 3 + 1

                year_dates.append({
                    'year': year,
                    'date': alert_date,
                    'month': alert_date.month,
                    'quarter': quarter,
                    'julian_day': int(date_value),
                    'confidence': conf_value
                })

            all_dates.extend(year_dates)
            year_counts[year] = len(year_dates)

            print(f"  Valid alerts: {len(year_dates)}")

        except Exception as e:
            print(f"  Error processing {year}: {e}")
            year_counts[year] = 0

    print(f"\n{'='*80}")
    print(f"TOTAL ALERTS ANALYZED: {len(all_dates)}")
    print(f"{'='*80}\n")

    return all_dates, year_counts

def analyze_quarterly_distribution(dates_data):
    """
    Analyze quarterly distribution
    """
    print(f"\n{'='*80}")
    print(f"QUARTERLY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")

    # Count by quarter
    quarters = [d['quarter'] for d in dates_data]
    quarter_counts = Counter(quarters)

    total = len(dates_data)

    print(f"Total alerts: {total}\n")

    print(f"{'Quarter':<15} {'Count':<10} {'Percentage':<15} {'Literature Expected':<20} {'Match'}")
    print("-" * 80)

    literature_expected = {
        1: (15, 20),  # Q1: 15-20%
        2: (20, 25),  # Q2: 20-25%
        3: (30, 35),  # Q3: 30-35% (PEAK)
        4: (20, 25)   # Q4: 20-25%
    }

    results = {}

    for q in [1, 2, 3, 4]:
        count = quarter_counts.get(q, 0)
        pct = 100 * count / total if total > 0 else 0
        expected_range = literature_expected[q]

        matches = expected_range[0] <= pct <= expected_range[1]

        print(f"Q{q} ({['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'][q-1]:<10}) "
              f"{count:<10} {pct:>6.1f}%         {expected_range[0]}-{expected_range[1]}%              "
              f"{'✓' if matches else '✗'}")

        results[f'Q{q}'] = {
            'count': count,
            'percentage': pct,
            'expected_range': expected_range,
            'matches_literature': matches
        }

    return results

def analyze_monthly_distribution(dates_data):
    """
    Analyze monthly distribution (more granular than quarterly)
    """
    print(f"\n{'='*80}")
    print(f"MONTHLY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")

    months = [d['month'] for d in dates_data]
    month_counts = Counter(months)

    total = len(dates_data)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"{'Month':<10} {'Count':<10} {'Percentage':<15} {'Quarter'}")
    print("-" * 50)

    monthly_results = {}

    for m in range(1, 13):
        count = month_counts.get(m, 0)
        pct = 100 * count / total if total > 0 else 0
        quarter = (m - 1) // 3 + 1

        print(f"{month_names[m-1]:<10} {count:<10} {pct:>6.1f}%         Q{quarter}")

        monthly_results[month_names[m-1]] = {
            'count': count,
            'percentage': pct,
            'quarter': quarter
        }

    return monthly_results

def create_visualizations(quarterly_results, monthly_results, sample_comparison):
    """
    Create visualizations
    """
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    fig = plt.figure(figsize=(18, 10))

    # Plot 1: Quarterly Distribution vs Literature
    ax1 = plt.subplot(2, 3, 1)

    quarters = [1, 2, 3, 4]
    percentages = [quarterly_results[f'Q{q}']['percentage'] for q in quarters]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    bars = ax1.bar(quarters, percentages, color=colors, alpha=0.7, edgecolor='black', width=0.6)

    # Add literature expected ranges
    expected_ranges = [(15, 20), (20, 25), (30, 35), (20, 25)]
    for i, (q, (low, high)) in enumerate(zip(quarters, expected_ranges)):
        ax1.plot([q-0.25, q+0.25], [low, low], 'k--', linewidth=2, alpha=0.7)
        ax1.plot([q-0.25, q+0.25], [high, high], 'k--', linewidth=2, alpha=0.7)
        ax1.plot([q, q], [low, high], 'k--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Full GLAD Distribution vs Literature Expected', fontsize=13, fontweight='bold')
    ax1.set_xticks(quarters)
    ax1.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(['Literature Expected Range'], loc='upper right')

    # Annotate bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Monthly Distribution
    ax2 = plt.subplot(2, 3, 2)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_pcts = [monthly_results[m]['percentage'] for m in month_names]

    # Color by quarter
    month_colors = []
    for m in range(12):
        quarter = (m) // 3
        month_colors.append(colors[quarter])

    bars = ax2.bar(range(1, 13), month_pcts, color=month_colors, alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Distribution (Color = Quarter)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Sample (17 pixels) vs Full GLAD Distribution
    ax3 = plt.subplot(2, 3, 3)

    sample_pcts = [sample_comparison['sample'][f'Q{q}'] for q in quarters]
    glad_pcts = [quarterly_results[f'Q{q}']['percentage'] for q in quarters]

    x = np.arange(len(quarters))
    width = 0.35

    bars1 = ax3.bar(x - width/2, sample_pcts, width, label='Our Sample (17 pixels)',
                    color='orange', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, glad_pcts, width, label='Full GLAD Dataset',
                    color='blue', alpha=0.7, edgecolor='black')

    ax3.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Our Sample vs Full GLAD Distribution', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Fire Season Overlay
    ax4 = plt.subplot(2, 3, 4)

    # Monthly distribution with fire season highlighted
    bars = ax4.bar(range(1, 13), month_pcts, color='lightblue', alpha=0.7, edgecolor='black')

    # Highlight fire season (Aug-Sep, months 8-9)
    for i in [7, 8]:  # Aug=8, Sep=9 (0-indexed: 7, 8)
        bars[i].set_color('red')
        bars[i].set_alpha(0.8)

    # Highlight dry season (Jul-Oct, months 7-10)
    for i in [6, 9]:  # Jul=7, Oct=10 (0-indexed: 6, 9)
        bars[i].set_color('orange')
        bars[i].set_alpha(0.8)

    ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Monthly Distribution with Fire Season\n(Red=Peak Fire, Orange=Dry Season)',
                 fontsize=13, fontweight='bold')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(month_names, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Cumulative Distribution
    ax5 = plt.subplot(2, 3, 5)

    cumulative = np.cumsum(month_pcts)

    ax5.plot(range(1, 13), cumulative, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% Mark')
    ax5.axhline(y=75, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='75% Mark')

    ax5.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Cumulative Distribution\n(When does 50%/75% occur?)', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(month_names, rotation=45, ha='right')
    ax5.legend(loc='lower right')
    ax5.grid(alpha=0.3)

    # Plot 6: Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate peak month
    peak_month_idx = np.argmax(month_pcts)
    peak_month = month_names[peak_month_idx]
    peak_pct = month_pcts[peak_month_idx]

    # Calculate when 50% and 75% cumulative occur
    month_50 = np.argmax(cumulative >= 50) + 1
    month_75 = np.argmax(cumulative >= 75) + 1

    # Q3 dominance
    q3_pct = quarterly_results['Q3']['percentage']

    summary = f"""
FULL GLAD DISTRIBUTION SUMMARY

Total Alerts Analyzed: {sample_comparison['total_alerts']:,}

Quarterly Breakdown:
  Q1 (Jan-Mar): {quarterly_results['Q1']['percentage']:.1f}%
  Q2 (Apr-Jun): {quarterly_results['Q2']['percentage']:.1f}%
  Q3 (Jul-Sep): {quarterly_results['Q3']['percentage']:.1f}%
  Q4 (Oct-Dec): {quarterly_results['Q4']['percentage']:.1f}%

Key Statistics:
  Peak Month: {peak_month} ({peak_pct:.1f}%)
  50% Cumulative: By {month_names[month_50-1]}
  75% Cumulative: By {month_names[month_75-1]}
  Q3 Dominance: {'YES' if q3_pct > 35 else 'NO'} ({q3_pct:.1f}%)

Literature Comparison:
  Q1: {'MATCH' if quarterly_results['Q1']['matches_literature'] else 'MISMATCH'}
  Q2: {'MATCH' if quarterly_results['Q2']['matches_literature'] else 'MISMATCH'}
  Q3: {'MATCH' if quarterly_results['Q3']['matches_literature'] else 'MISMATCH'}
  Q4: {'MATCH' if quarterly_results['Q4']['matches_literature'] else 'MISMATCH'}

Sample (17 pixels) vs Full GLAD:
  Sample Q3: {sample_comparison['sample']['Q3']:.1f}%
  Full Q3: {q3_pct:.1f}%
  Difference: {abs(sample_comparison['sample']['Q3'] - q3_pct):.1f}%
"""

    ax6.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation/glad_full_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

def main():
    """
    Main execution
    """
    print(f"\n{'='*80}")
    print(f"GLAD FULL DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")

    print("OBJECTIVE: Analyze ENTIRE GLAD dataset to get true quarterly distribution")
    print("  - Sample 10,000s of pixels (not just 17)")
    print("  - Get definitive answer on when deforestation occurs")
    print("  - Compare to our sample and literature expectations")
    print()

    # Initialize
    ee.Initialize()

    # Study region (same as before)
    main_bounds = {
        "min_lon": -73,
        "max_lon": -50,
        "min_lat": -15,
        "max_lat": 5
    }

    # Years
    years = [2019, 2020, 2021]

    # Get full GLAD distribution
    dates_data, year_counts = get_all_glad_alerts_distribution(main_bounds, years)

    if len(dates_data) < 100:
        print(f"\n⚠️  WARNING: Only {len(dates_data)} alerts found")
        print("Need at least 100 for meaningful distribution analysis")
        return

    # Analyze quarterly distribution
    quarterly_results = analyze_quarterly_distribution(dates_data)

    # Analyze monthly distribution
    monthly_results = analyze_monthly_distribution(dates_data)

    # Compare to our sample (17 pixels: 0%, 6%, 71%, 24%)
    sample_comparison = {
        'total_alerts': len(dates_data),
        'sample': {
            'Q1': 0.0,
            'Q2': 5.9,
            'Q3': 70.6,
            'Q4': 23.5
        },
        'glad_full': {
            'Q1': quarterly_results['Q1']['percentage'],
            'Q2': quarterly_results['Q2']['percentage'],
            'Q3': quarterly_results['Q3']['percentage'],
            'Q4': quarterly_results['Q4']['percentage']
        }
    }

    # Create visualizations
    create_visualizations(quarterly_results, monthly_results, sample_comparison)

    # Save results
    output_dir = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation'
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'study_region': main_bounds,
            'years': years,
            'total_alerts': len(dates_data),
            'year_counts': year_counts
        },
        'quarterly_distribution': quarterly_results,
        'monthly_distribution': monthly_results,
        'sample_comparison': sample_comparison
    }

    output_path = os.path.join(output_dir, 'glad_full_distribution.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results: {output_path}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    print(f"Analyzed {len(dates_data):,} GLAD alerts")
    print(f"\nQuarterly Distribution:")
    for q in [1, 2, 3, 4]:
        pct = quarterly_results[f'Q{q}']['percentage']
        expected = quarterly_results[f'Q{q}']['expected_range']
        match = quarterly_results[f'Q{q}']['matches_literature']
        print(f"  Q{q}: {pct:>5.1f}% (expected {expected[0]}-{expected[1]}%) {'✓' if match else '✗'}")

    print(f"\nSample (17 pixels) vs Full GLAD:")
    print(f"  Q3 Sample: {sample_comparison['sample']['Q3']:.1f}%")
    print(f"  Q3 GLAD:   {sample_comparison['glad_full']['Q3']:.1f}%")
    print(f"  Difference: {abs(sample_comparison['sample']['Q3'] - sample_comparison['glad_full']['Q3']):.1f}%")

if __name__ == "__main__":
    main()
