"""
Comprehensive Quarterly Validation

GOAL: Resolve detection vs prediction ambiguity with larger sample
- Test 1: Larger Q2 vs Q4 comparison (100+ pixels target)
- Test 2: Test Q3 SEPARATELY (critical missing piece!)
- Test 3: Full quarterly distribution validation

CRITICAL QUESTION: Does Q3 dominate as literature suggests (peak fire season)?
- Literature: Q3 (Jul-Sep) should be 30-35% of clearings (peak: Aug-Sep)
- Our findings: Q2 showed strongest signal (0.78 vs 0.38 for Q4)
- But Q3 NOT TESTED separately (8 out of 13 pixels assumed similar to Q2)

This test will determine if we have:
- Early detection system (0-3 months) if Q2 dominant
- Prediction system (3-6 months) if Q3-Q4 dominant
- Mixed system if both show signal
"""

import ee
import numpy as np
import json
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.earth_engine import EarthEngineClient
from src.utils import get_config

def get_glad_clearing_quarter(lat, lon, year, client):
    """
    Extract GLAD clearing date and determine quarter

    Returns:
        quarter (int): 1, 2, 3, or 4
        date (datetime): actual clearing date
        None if not found or error
    """
    try:
        # Determine dataset ID
        if year >= 2024:
            dataset_id = 'projects/glad/alert/UpdResult'
        else:
            dataset_id = f'projects/glad/alert/{year}final'

        # Load as ImageCollection (not Image!)
        glad_collection = ee.ImageCollection(dataset_id)

        # Band names
        year_suffix = str(year % 100)
        alert_date_band = f'alertDate{year_suffix}'
        conf_band = f'conf{year_suffix}'

        # Get as mosaic
        glad = glad_collection.select([alert_date_band, conf_band]).mosaic()

        # Sample at point
        point = ee.Geometry.Point([lon, lat])
        sample = glad.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=30
        ).getInfo()

        date_value = sample.get(alert_date_band)
        conf_value = sample.get(conf_band)

        if date_value is None or date_value == 0:
            return None

        # CRITICAL: GLAD encoding is Julian day of year (1-365)
        # NOT days since epoch!
        alert_date = datetime(year, 1, 1) + timedelta(days=int(date_value) - 1)

        # Determine quarter (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)
        quarter = (alert_date.month - 1) // 3 + 1

        return {
            'quarter': quarter,
            'date': alert_date.isoformat(),
            'confidence': conf_value,
            'julian_day': int(date_value)
        }

    except Exception as e:
        print(f"Error getting GLAD date for ({lat}, {lon}): {e}")
        return None

def sample_cleared_pixels_with_quarters(client, bounds, year, target_count=150):
    """
    Sample cleared pixels and extract GLAD quarters

    Target: 150 pixels to hopefully get 100+ with valid GLAD dates

    Returns:
        list of dicts with pixel info + quarter + embedding distances
    """
    print(f"\n{'='*80}")
    print(f"SAMPLING CLEARED PIXELS WITH GLAD QUARTERS")
    print(f"{'='*80}\n")

    print(f"Target: {target_count} pixels from {year}")
    print(f"Bounds: {bounds}")

    # Sample cleared locations from multiple sub-regions to get more samples
    # (get_deforestation_labels samples up to 1000 pixels per call)
    print("\nSampling cleared pixels from Hansen GFC...")

    # Divide region into quadrants to get more samples
    mid_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
    mid_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2

    sub_regions = [
        {"min_lon": bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": mid_lat, "max_lat": bounds["max_lat"]},  # NW
        {"min_lon": mid_lon, "max_lon": bounds["max_lon"],
         "min_lat": mid_lat, "max_lat": bounds["max_lat"]},  # NE
        {"min_lon": bounds["min_lon"], "max_lon": mid_lon,
         "min_lat": bounds["min_lat"], "max_lat": mid_lat},  # SW
        {"min_lon": mid_lon, "max_lon": bounds["max_lon"],
         "min_lat": bounds["min_lat"], "max_lat": mid_lat},  # SE
    ]

    cleared_pixels = []

    for i, sub_bounds in enumerate(sub_regions):
        print(f"  Sampling from sub-region {i+1}/4...")
        try:
            pixels = client.get_deforestation_labels(
                bounds=sub_bounds,
                year=year,
                min_tree_cover=30
            )
            cleared_pixels.extend(pixels)
            print(f"    Found {len(pixels)} pixels")
        except Exception as e:
            print(f"    Error sampling sub-region: {e}")
            continue

    # Randomly sample if we have too many
    if len(cleared_pixels) > target_count:
        import random
        random.seed(42)
        cleared_pixels = random.sample(cleared_pixels, target_count)

    print(f"\nTotal sampled: {len(cleared_pixels)} pixels")

    # For each pixel:
    # 1. Get GLAD quarter
    # 2. Extract Y-1 (2019) and Y-2 (2018) embeddings
    # 3. Calculate embedding distance

    results = []

    for i, pixel in enumerate(cleared_pixels):
        lat = pixel['lat']
        lon = pixel['lon']

        if (i + 1) % 10 == 0:
            print(f"Processing pixel {i+1}/{len(cleared_pixels)}...")

        # Get GLAD quarter
        glad_info = get_glad_clearing_quarter(lat, lon, year, client)

        if glad_info is None:
            # Skip if no GLAD date
            continue

        # Extract embeddings
        try:
            emb_y2 = client.get_embedding(lat=lat, lon=lon, date=f"{year-2}-06-01")  # 2018
            emb_y1 = client.get_embedding(lat=lat, lon=lon, date=f"{year-1}-06-01")  # 2019

            # Calculate distance Y-1 to Y-2 (baseline variability)
            distance_y1_to_y2 = np.linalg.norm(np.array(emb_y1) - np.array(emb_y2))

            results.append({
                'lat': lat,
                'lon': lon,
                'quarter': glad_info['quarter'],
                'clearing_date': glad_info['date'],
                'julian_day': glad_info['julian_day'],
                'confidence': glad_info['confidence'],
                'embedding_y2': emb_y2,
                'embedding_y1': emb_y1,
                'distance_y1_to_y2': distance_y1_to_y2
            })

        except Exception as e:
            print(f"  Error extracting embeddings for ({lat}, {lon}): {e}")
            continue

    print(f"\nSuccessfully extracted: {len(results)} pixels with GLAD dates and embeddings")

    # Summary by quarter
    quarter_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in results:
        quarter_counts[r['quarter']] += 1

    print("\nQuarterly Distribution:")
    for q in [1, 2, 3, 4]:
        count = quarter_counts[q]
        pct = 100 * count / len(results) if len(results) > 0 else 0
        print(f"  Q{q}: {count:3d} pixels ({pct:5.1f}%)")

    return results

def sample_intact_pixels(client, bounds, sample_size=30):
    """
    Sample intact forest pixels for control comparison

    Returns:
        list of dicts with pixel info + embedding distances
    """
    print(f"\n{'='*80}")
    print(f"SAMPLING INTACT FOREST PIXELS (CONTROL)")
    print(f"{'='*80}\n")

    print(f"Target: {sample_size} intact pixels")

    # Sample intact locations
    intact_pixels = client.get_stable_forest_locations(
        bounds=bounds,
        n_samples=sample_size
    )

    print(f"Sampled: {len(intact_pixels)} pixels")

    results = []

    for i, pixel in enumerate(intact_pixels):
        lat = pixel['lat']
        lon = pixel['lon']

        if (i + 1) % 10 == 0:
            print(f"Processing pixel {i+1}/{len(intact_pixels)}...")

        try:
            emb_y2 = client.get_embedding(lat=lat, lon=lon, date="2018-06-01")
            emb_y1 = client.get_embedding(lat=lat, lon=lon, date="2019-06-01")

            distance_y1_to_y2 = np.linalg.norm(np.array(emb_y1) - np.array(emb_y2))

            results.append({
                'lat': lat,
                'lon': lon,
                'embedding_y2': emb_y2,
                'embedding_y1': emb_y1,
                'distance_y1_to_y2': distance_y1_to_y2
            })

        except Exception as e:
            print(f"  Error extracting embeddings for ({lat}, {lon}): {e}")
            continue

    print(f"\nSuccessfully extracted: {len(results)} intact pixels with embeddings")

    return results

def test_1_q2_vs_q4_larger_sample(cleared_pixels):
    """
    TEST 1: Re-run Q2 vs Q4 with larger sample

    Previous result (13 pixels):
    - Q2: 0.78 ± 0.14 (STRONG)
    - Q4: 0.38 ± 0.10 (WEAK)
    - p = 0.0016 (significant)

    Does this hold with larger sample?
    """
    print(f"\n{'='*80}")
    print(f"TEST 1: Q2 vs Q4 COMPARISON (LARGER SAMPLE)")
    print(f"{'='*80}\n")

    # Filter by quarter
    q2_pixels = [p for p in cleared_pixels if p['quarter'] == 2]
    q4_pixels = [p for p in cleared_pixels if p['quarter'] == 4]

    print(f"Q2 (Apr-Jun) pixels: {len(q2_pixels)}")
    print(f"Q4 (Oct-Dec) pixels: {len(q4_pixels)}")

    if len(q2_pixels) < 3 or len(q4_pixels) < 3:
        print("⚠️  Insufficient sample for statistical test (need ≥3 per group)")
        return None

    # Extract distances
    q2_distances = [p['distance_y1_to_y2'] for p in q2_pixels]
    q4_distances = [p['distance_y1_to_y2'] for p in q4_pixels]

    # Statistics
    q2_mean = np.mean(q2_distances)
    q2_std = np.std(q2_distances, ddof=1)
    q4_mean = np.mean(q4_distances)
    q4_std = np.std(q4_distances, ddof=1)

    print(f"\nQ2 embedding distance: {q2_mean:.4f} ± {q2_std:.4f}")
    print(f"Q4 embedding distance: {q4_mean:.4f} ± {q4_std:.4f}")
    print(f"Difference: {q2_mean - q4_mean:+.4f} ({100*(q2_mean - q4_mean)/q4_mean:+.1f}%)")

    # Statistical test (one-tailed: Q2 > Q4)
    t_stat, p_value_two_tailed = stats.ttest_ind(q2_distances, q4_distances, equal_var=False)
    p_value_one_tailed = p_value_two_tailed / 2 if t_stat > 0 else 1 - p_value_two_tailed / 2

    # Effect size
    pooled_std = np.sqrt((q2_std**2 + q4_std**2) / 2)
    cohens_d = (q2_mean - q4_mean) / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical Test (Welch's t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value (one-tailed, Q2 > Q4): {p_value_one_tailed:.6f}")
    print(f"  Significant at α=0.05: {'✓ YES' if p_value_one_tailed < 0.05 else '✗ NO'}")
    print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

    result = {
        'q2': {
            'n': len(q2_pixels),
            'mean': float(q2_mean),
            'std': float(q2_std),
            'distances': q2_distances
        },
        'q4': {
            'n': len(q4_pixels),
            'mean': float(q4_mean),
            'std': float(q4_std),
            'distances': q4_distances
        },
        'difference': float(q2_mean - q4_mean),
        'percent_difference': float(100 * (q2_mean - q4_mean) / q4_mean),
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value_one_tailed': float(p_value_one_tailed),
            'significant': bool(p_value_one_tailed < 0.05)
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    }

    # Interpretation
    print(f"\nInterpretation:")
    if p_value_one_tailed < 0.05 and q2_mean > q4_mean:
        print("✓ Q2 clearings show SIGNIFICANTLY STRONGER signal than Q4")
        print("  → Supports EARLY DETECTION (0-3 months) interpretation")
        print("  → June embedding captures Apr-Jun clearings concurrently")
    elif p_value_one_tailed < 0.05 and q4_mean > q2_mean:
        print("✓ Q4 clearings show SIGNIFICANTLY STRONGER signal than Q2")
        print("  → Supports PREDICTION (3-6 months) interpretation")
        print("  → June embedding predicts Oct-Dec clearings")
    else:
        print("✗ NO significant difference between Q2 and Q4")
        print("  → Signal may be uniform across quarters")

    return result

def test_2_q3_separate(cleared_pixels):
    """
    TEST 2: Test Q3 SEPARATELY - critical missing piece!

    Literature predicts: Q3 (Jul-Sep) should be DOMINANT (30-35%)
    - Peak fire season: August-September
    - GLAD detects burning

    Our previous finding: Q3 assumed similar to Q2 (not tested separately)

    This is the KEY test to resolve detection vs prediction!
    """
    print(f"\n{'='*80}")
    print(f"TEST 2: Q3 (JULY-SEPTEMBER) SEPARATE TEST ⭐ CRITICAL")
    print(f"{'='*80}\n")

    print("Literature expectation:")
    print("  Q3 (Jul-Sep) = PEAK FIRE SEASON (Aug-Sep)")
    print("  Should be DOMINANT quarter (30-35% of clearings)")
    print("  GLAD detects burning/clearing in real-time\n")

    # Filter Q3
    q3_pixels = [p for p in cleared_pixels if p['quarter'] == 3]

    print(f"Q3 (Jul-Sep) pixels: {len(q3_pixels)}")

    if len(q3_pixels) < 3:
        print("⚠️  Insufficient Q3 sample for statistical test")
        return None

    # Extract distances
    q3_distances = [p['distance_y1_to_y2'] for p in q3_pixels]

    # Statistics
    q3_mean = np.mean(q3_distances)
    q3_std = np.std(q3_distances, ddof=1)

    print(f"\nQ3 embedding distance: {q3_mean:.4f} ± {q3_std:.4f}")

    # Compare to Q2 and Q4
    q2_pixels = [p for p in cleared_pixels if p['quarter'] == 2]
    q4_pixels = [p for p in cleared_pixels if p['quarter'] == 4]

    if len(q2_pixels) >= 3:
        q2_distances = [p['distance_y1_to_y2'] for p in q2_pixels]
        q2_mean = np.mean(q2_distances)

        # Test Q3 vs Q2
        t_stat_q3_q2, p_value_q3_q2 = stats.ttest_ind(q3_distances, q2_distances, equal_var=False)

        print(f"\nQ3 vs Q2 comparison:")
        print(f"  Q3 mean: {q3_mean:.4f}")
        print(f"  Q2 mean: {q2_mean:.4f}")
        print(f"  Difference: {q3_mean - q2_mean:+.4f} ({100*(q3_mean - q2_mean)/q2_mean:+.1f}%)")
        print(f"  p-value: {p_value_q3_q2:.6f}")
        print(f"  Significant: {'✓ YES' if p_value_q3_q2 < 0.05 else '✗ NO'}")

    if len(q4_pixels) >= 3:
        q4_distances = [p['distance_y1_to_y2'] for p in q4_pixels]
        q4_mean = np.mean(q4_distances)

        # Test Q3 vs Q4
        t_stat_q3_q4, p_value_q3_q4 = stats.ttest_ind(q3_distances, q4_distances, equal_var=False)

        print(f"\nQ3 vs Q4 comparison:")
        print(f"  Q3 mean: {q3_mean:.4f}")
        print(f"  Q4 mean: {q4_mean:.4f}")
        print(f"  Difference: {q3_mean - q4_mean:+.4f} ({100*(q3_mean - q4_mean)/q4_mean:+.1f}%)")
        print(f"  p-value: {p_value_q3_q4:.6f}")
        print(f"  Significant: {'✓ YES' if p_value_q3_q4 < 0.05 else '✗ NO'}")

    result = {
        'q3': {
            'n': len(q3_pixels),
            'mean': float(q3_mean),
            'std': float(q3_std),
            'distances': q3_distances
        }
    }

    if len(q2_pixels) >= 3:
        result['q3_vs_q2'] = {
            't_statistic': float(t_stat_q3_q2),
            'p_value': float(p_value_q3_q2),
            'significant': bool(p_value_q3_q2 < 0.05)
        }

    if len(q4_pixels) >= 3:
        result['q3_vs_q4'] = {
            't_statistic': float(t_stat_q3_q4),
            'p_value': float(p_value_q3_q4),
            'significant': bool(p_value_q3_q4 < 0.05)
        }

    # Interpretation
    print(f"\nInterpretation:")
    print(f"Q3 mean distance: {q3_mean:.4f}")

    if q3_mean > 0.6:
        print("✓ Q3 shows STRONG signal (>0.6)")
        print("  → BUT: June embedding is BEFORE Jul-Sep clearing")
        print("  → This would support TRUE PRECURSOR (1-3 months before)")
    elif q3_mean > 0.4:
        print("⚠ Q3 shows MODERATE signal (0.4-0.6)")
        print("  → Mixed detection/precursor")
    else:
        print("✗ Q3 shows WEAK signal (<0.4)")
        print("  → Contradicts literature (Q3 should be peak fire season)")

    return result

def test_3_quarterly_distribution(cleared_pixels, intact_pixels):
    """
    TEST 3: Full Quarterly Distribution Validation

    Literature expectation (Amazon):
    - Q1 (Jan-Mar): 15-20% (wet season cutting)
    - Q2 (Apr-Jun): 20-25% (late wet, drying starts)
    - Q3 (Jul-Sep): 30-35% ⭐ PEAK FIRE SEASON
    - Q4 (Oct-Dec): 20-25% (late burning)

    Our hypothesis (based on Q2 vs Q4 test):
    - Q2-Q3 combined: ~60% (dominant)
    - Q1, Q4: Lower

    This test will:
    1. Compare observed distribution to literature
    2. Test if embedding distances correlate with quarter
    3. Compare all quarters to intact control
    """
    print(f"\n{'='*80}")
    print(f"TEST 3: FULL QUARTERLY DISTRIBUTION VALIDATION")
    print(f"{'='*80}\n")

    # Group by quarter
    quarterly_data = {q: [] for q in [1, 2, 3, 4]}

    for pixel in cleared_pixels:
        q = pixel['quarter']
        quarterly_data[q].append(pixel['distance_y1_to_y2'])

    # Intact control
    intact_distances = [p['distance_y1_to_y2'] for p in intact_pixels]
    intact_mean = np.mean(intact_distances)
    intact_std = np.std(intact_distances, ddof=1)

    print(f"Intact forest control: {intact_mean:.4f} ± {intact_std:.4f} (n={len(intact_pixels)})")
    print()

    # Statistics by quarter
    print("Quarterly Analysis:")
    print(f"{'Quarter':<10} {'N':<6} {'Mean':<10} {'Std':<10} {'vs Intact':<15} {'p-value':<10} {'Sig':<5}")
    print("-" * 80)

    results = {
        'distribution': {},
        'vs_intact': {},
        'intact_control': {
            'n': len(intact_pixels),
            'mean': float(intact_mean),
            'std': float(intact_std)
        }
    }

    for q in [1, 2, 3, 4]:
        distances = quarterly_data[q]
        n = len(distances)

        if n == 0:
            print(f"Q{q}        {n:<6} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'N/A':<10} {'N/A':<5}")
            continue

        mean = np.mean(distances)
        std = np.std(distances, ddof=1) if n > 1 else 0

        # Test vs intact
        if n >= 3:
            t_stat, p_value = stats.ttest_ind(distances, intact_distances, equal_var=False)
            significant = p_value < 0.05

            diff = mean - intact_mean
            pct_diff = 100 * diff / intact_mean if intact_mean != 0 else 0

            print(f"Q{q}        {n:<6} {mean:<10.4f} {std:<10.4f} {diff:+.4f} ({pct_diff:+.1f}%) {p_value:<10.6f} {'✓' if significant else '✗':<5}")

            results['vs_intact'][f'Q{q}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(significant)
            }
        else:
            print(f"Q{q}        {n:<6} {mean:<10.4f} {std:<10.4f} {'N/A':<15} {'N/A':<10} {'N/A':<5}")

        results['distribution'][f'Q{q}'] = {
            'n': int(n),
            'mean': float(mean),
            'std': float(std),
            'distances': distances
        }

    # Observed vs Expected Distribution
    print(f"\n{'='*80}")
    print("OBSERVED vs LITERATURE EXPECTED DISTRIBUTION")
    print(f"{'='*80}\n")

    total_pixels = len(cleared_pixels)

    print(f"{'Quarter':<15} {'Observed':<20} {'Literature Expected':<25} {'Match':<10}")
    print("-" * 80)

    literature_expected = {
        'Q1': (15, 20),  # 15-20%
        'Q2': (20, 25),  # 20-25%
        'Q3': (30, 35),  # 30-35% ⭐ PEAK
        'Q4': (20, 25)   # 20-25%
    }

    for q in [1, 2, 3, 4]:
        n = quarterly_data[q].__len__()
        observed_pct = 100 * n / total_pixels if total_pixels > 0 else 0
        expected_range = literature_expected[f'Q{q}']

        # Check if within expected range
        matches = expected_range[0] <= observed_pct <= expected_range[1]

        print(f"Q{q}             {n:3d} ({observed_pct:5.1f}%)      {expected_range[0]}-{expected_range[1]}%                  {'✓' if matches else '✗'}")

        # Only add these fields if quarter has data in results
        if f'Q{q}' in results['distribution']:
            results['distribution'][f'Q{q}']['observed_percent'] = float(observed_pct)
            results['distribution'][f'Q{q}']['expected_range'] = list(expected_range)
            results['distribution'][f'Q{q}']['matches_literature'] = bool(matches)
        else:
            # Initialize empty quarter
            results['distribution'][f'Q{q}'] = {
                'n': 0,
                'observed_percent': 0.0,
                'expected_range': list(expected_range),
                'matches_literature': bool(matches)
            }

    # ANOVA test (are means different across quarters?)
    print(f"\n{'='*80}")
    print("ANOVA: Are embedding distances different across quarters?")
    print(f"{'='*80}\n")

    # Filter quarters with sufficient samples
    quarterly_distances = [quarterly_data[q] for q in [1, 2, 3, 4] if len(quarterly_data[q]) >= 3]

    if len(quarterly_distances) >= 2:
        f_stat, p_value_anova = stats.f_oneway(*quarterly_distances)

        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value_anova:.6f}")
        print(f"Significant at α=0.05: {'✓ YES' if p_value_anova < 0.05 else '✗ NO'}")

        results['anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value_anova),
            'significant': bool(p_value_anova < 0.05)
        }

        if p_value_anova < 0.05:
            print("\n✓ Quarters show SIGNIFICANTLY different embedding distances")
            print("  → Temporal resolution matters!")
        else:
            print("\n✗ Quarters show NO significant difference")
            print("  → Signal may be uniform across year")
    else:
        print("⚠️  Insufficient quarters with ≥3 samples for ANOVA")

    return results

def create_visualizations(test1_result, test2_result, test3_result, cleared_pixels, intact_pixels):
    """
    Create comprehensive visualizations
    """
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    fig = plt.figure(figsize=(16, 10))

    # Figure 1: Quarterly Distribution (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)

    quarter_counts = {q: 0 for q in [1, 2, 3, 4]}
    for p in cleared_pixels:
        quarter_counts[p['quarter']] += 1

    quarters = [1, 2, 3, 4]
    counts = [quarter_counts[q] for q in quarters]
    total = sum(counts)
    percentages = [100 * c / total if total > 0 else 0 for c in counts]

    bars = ax1.bar(quarters, percentages, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.7, edgecolor='black')

    # Literature expected ranges
    expected_ranges = [(15, 20), (20, 25), (30, 35), (20, 25)]
    for i, (q, (low, high)) in enumerate(zip(quarters, expected_ranges)):
        ax1.plot([q-0.3, q+0.3], [low, low], 'k--', linewidth=1.5, alpha=0.5)
        ax1.plot([q-0.3, q+0.3], [high, high], 'k--', linewidth=1.5, alpha=0.5)
        ax1.plot([q, q], [low, high], 'k--', linewidth=1.5, alpha=0.5)

    ax1.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Clearings (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Observed vs Literature Expected\nQuarterly Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(quarters)
    ax1.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Expected Range')
    ax1.legend(['Literature Expected'], loc='upper right')

    # Annotate bars with counts
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={count}\n{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Figure 2: Embedding Distances by Quarter (Box Plot)
    ax2 = plt.subplot(2, 3, 2)

    quarterly_distances = []
    labels = []
    colors_box = []

    for q in [1, 2, 3, 4]:
        distances = [p['distance_y1_to_y2'] for p in cleared_pixels if p['quarter'] == q]
        if len(distances) > 0:
            quarterly_distances.append(distances)
            labels.append(f'Q{q}')
            colors_box.append(['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'][q-1])

    # Add intact control
    intact_distances = [p['distance_y1_to_y2'] for p in intact_pixels]
    quarterly_distances.append(intact_distances)
    labels.append('Intact')
    colors_box.append('#7f7f7f')

    bp = ax2.boxplot(quarterly_distances, labels=labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Embedding Distance (Y-1 to Y-2)', fontsize=12, fontweight='bold')
    ax2.set_title('Embedding Distances by Quarter\n(vs Intact Control)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Figure 3: Q2 vs Q4 Comparison (from Test 1)
    ax3 = plt.subplot(2, 3, 3)

    if test1_result:
        q2_mean = test1_result['q2']['mean']
        q2_std = test1_result['q2']['std']
        q4_mean = test1_result['q4']['mean']
        q4_std = test1_result['q4']['std']

        x = [1, 2]
        means = [q2_mean, q4_mean]
        stds = [q2_std, q4_std]

        bars = ax3.bar(x, means, yerr=stds, color=['#ff7f0e', '#1f77b4'], alpha=0.7,
                      edgecolor='black', capsize=10, width=0.6)

        ax3.set_ylabel('Embedding Distance', fontsize=12, fontweight='bold')
        ax3.set_title(f"Test 1: Q2 vs Q4 Comparison\n(p={test1_result['statistical_test']['p_value_one_tailed']:.4f})",
                     fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Q2\n(Apr-Jun)', 'Q4\n(Oct-Dec)'])
        ax3.set_ylim(0, max(means) + max(stds) + 0.1)
        ax3.grid(axis='y', alpha=0.3)

        # Annotate with significance
        if test1_result['statistical_test']['significant']:
            ax3.text(1.5, max(means) + max(stds) + 0.05, '***',
                    ha='center', fontsize=20, fontweight='bold')

    # Figure 4: Q3 Analysis (from Test 2)
    ax4 = plt.subplot(2, 3, 4)

    if test2_result and test3_result:
        # Compare Q2, Q3, Q4
        q2_mean = test3_result['distribution']['Q2']['mean'] if test3_result['distribution']['Q2']['n'] > 0 else 0
        q3_mean = test2_result['q3']['mean']
        q4_mean = test3_result['distribution']['Q4']['mean'] if test3_result['distribution']['Q4']['n'] > 0 else 0

        x = [1, 2, 3]
        means = [q2_mean, q3_mean, q4_mean]

        bars = ax4.bar(x, means, color=['#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.7,
                      edgecolor='black', width=0.6)

        ax4.set_ylabel('Embedding Distance', fontsize=12, fontweight='bold')
        ax4.set_title('Test 2: Q3 Separate Analysis\n(Peak Fire Season)', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Q2\n(Apr-Jun)', 'Q3⭐\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
        ax4.grid(axis='y', alpha=0.3)

        # Annotate with literature expectation
        ax4.text(2, max(means) * 1.1, 'Literature:\nSHOULD be\nDOMINANT',
                ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Figure 5: Cleared vs Intact (All Quarters)
    ax5 = plt.subplot(2, 3, 5)

    if test3_result:
        quarters_with_data = []
        means_cleared = []

        for q in [1, 2, 3, 4]:
            if test3_result['distribution'][f'Q{q}']['n'] > 0:
                quarters_with_data.append(q)
                means_cleared.append(test3_result['distribution'][f'Q{q}']['mean'])

        intact_mean = test3_result['intact_control']['mean']

        x = list(range(len(quarters_with_data)))

        ax5.plot(x, means_cleared, 'o-', color='#d62728', linewidth=2, markersize=10,
                label='Cleared (by quarter)', alpha=0.8)
        ax5.axhline(y=intact_mean, color='#7f7f7f', linestyle='--', linewidth=2,
                   label=f'Intact Control ({intact_mean:.3f})', alpha=0.8)

        ax5.set_xlabel('Quarter', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Embedding Distance', fontsize=12, fontweight='bold')
        ax5.set_title('Test 3: Cleared vs Intact Control\n(All Quarters)', fontsize=13, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'Q{q}' for q in quarters_with_data])
        ax5.legend(loc='best')
        ax5.grid(alpha=0.3)

    # Figure 6: Temporal Timeline Schematic
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Draw timeline
    timeline_text = """
    TEMPORAL INTERPRETATION SUMMARY

    June 2019 Embedding captures:

    Q1 (Jan-Mar 2020): 9-12 months before
        ↓ PREDICTION

    Q2 (Apr-Jun 2020): 0-3 months overlap
        ↓ EARLY DETECTION

    Q3 (Jul-Sep 2020): 1-3 months after
        ↓ ??? (Literature: should be PEAK)

    Q4 (Oct-Dec 2020): 4-6 months after
        ↓ NO SIGNAL (expected)

    """

    # Determine interpretation based on results
    if test1_result and test1_result['statistical_test']['significant']:
        if test1_result['q2']['mean'] > test1_result['q4']['mean']:
            interpretation = "✓ EARLY DETECTION system (0-3 months)\n    Q2 dominant, Q4 weak"
        else:
            interpretation = "✓ PREDICTION system (3-6 months)\n    Q4 dominant, Q2 weak"
    else:
        interpretation = "? Mixed or uniform signal"

    if test2_result:
        q3_mean = test2_result['q3']['mean']
        if q3_mean > 0.6:
            q3_interp = "✓ Q3 STRONG (supports precursor)"
        elif q3_mean > 0.4:
            q3_interp = "⚠ Q3 MODERATE"
        else:
            q3_interp = "✗ Q3 WEAK (contradicts literature)"
    else:
        q3_interp = "? Q3 not tested"

    summary_text = f"""
COMPREHENSIVE QUARTERLY VALIDATION
RESULTS SUMMARY

Test 1: Q2 vs Q4 (Larger Sample)
{interpretation}

Test 2: Q3 Separate Analysis
{q3_interp}

Test 3: Distribution vs Literature
"""

    if test3_result and 'anova' in test3_result:
        if test3_result['anova']['significant']:
            summary_text += "✓ Quarters DIFFER significantly (ANOVA)\n"
        else:
            summary_text += "✗ Quarters do NOT differ (ANOVA)\n"

    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation/quarterly_validation_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

def main():
    """
    Main execution
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE QUARTERLY VALIDATION")
    print(f"{'='*80}\n")

    print("GOAL: Resolve detection vs prediction ambiguity")
    print("  1. Re-test Q2 vs Q4 with larger sample")
    print("  2. Test Q3 SEPARATELY (critical missing piece!)")
    print("  3. Validate quarterly distribution vs literature")
    print()

    # Initialize client
    print("Initializing Earth Engine...")
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Study region (same as before)
    main_bounds = {
        "min_lon": -73,
        "max_lon": -50,
        "min_lat": -15,
        "max_lat": 5
    }

    # Sample cleared pixels from MULTIPLE YEARS to get larger sample
    # (single year 2020 only gave us 5 pixels)
    print("\nSampling strategy: Using multiple years (2019-2021) to increase sample size\n")

    all_cleared_pixels = []

    for year in [2019, 2020, 2021]:
        print(f"\nSampling year {year}...")
        pixels = sample_cleared_pixels_with_quarters(
            client=client,
            bounds=main_bounds,
            year=year,
            target_count=60  # ~60 per year → ~180 total
        )
        all_cleared_pixels.extend(pixels)
        print(f"  Year {year}: {len(pixels)} pixels with GLAD dates")

    cleared_pixels = all_cleared_pixels
    print(f"\n✓ Combined total: {len(cleared_pixels)} pixels across all years")

    if len(cleared_pixels) < 10:
        print(f"\n⚠️  WARNING: Only {len(cleared_pixels)} pixels with GLAD dates")
        print("May not have sufficient statistical power")
        print("Proceeding anyway...\n")

    # Sample intact pixels (control)
    intact_pixels = sample_intact_pixels(
        client=client,
        bounds=main_bounds,
        sample_size=30
    )

    # Run tests
    test1_result = test_1_q2_vs_q4_larger_sample(cleared_pixels)
    test2_result = test_2_q3_separate(cleared_pixels)
    test3_result = test_3_quarterly_distribution(cleared_pixels, intact_pixels)

    # Save results
    output_dir = '/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation'
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'study_region': main_bounds,
            'cleared_years': [2019, 2020, 2021],
            'total_cleared_pixels': len(cleared_pixels),
            'total_intact_pixels': len(intact_pixels),
            'note': 'Multi-year sampling used to increase sample size'
        },
        'test1_q2_vs_q4': test1_result,
        'test2_q3_separate': test2_result,
        'test3_quarterly_distribution': test3_result
    }

    output_path = os.path.join(output_dir, 'quarterly_validation_comprehensive.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results: {output_path}")

    # Create visualizations
    create_visualizations(test1_result, test2_result, test3_result, cleared_pixels, intact_pixels)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total pixels analyzed: {len(cleared_pixels)} cleared + {len(intact_pixels)} intact")
    print()

    if test1_result and test1_result['statistical_test']['significant']:
        if test1_result['q2']['mean'] > test1_result['q4']['mean']:
            print("✓ Test 1: Q2 >> Q4 (early detection, 0-3 months)")
        else:
            print("✓ Test 1: Q4 >> Q2 (prediction, 3-6 months)")
    else:
        print("✗ Test 1: No significant Q2 vs Q4 difference")

    if test2_result:
        q3_mean = test2_result['q3']['mean']
        print(f"✓ Test 2: Q3 mean = {q3_mean:.4f}")
        if q3_mean > 0.6:
            print("  → Q3 shows STRONG signal (supports precursor)")
        elif q3_mean > 0.4:
            print("  → Q3 shows MODERATE signal")
        else:
            print("  → Q3 shows WEAK signal (contradicts literature)")

    if test3_result and 'anova' in test3_result:
        if test3_result['anova']['significant']:
            print("✓ Test 3: Quarters show SIGNIFICANT differences (ANOVA)")
        else:
            print("✗ Test 3: Quarters show NO significant differences")

    print(f"\n{'='*80}")
    print("COMPREHENSIVE QUARTERLY VALIDATION COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
