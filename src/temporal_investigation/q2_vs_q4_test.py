"""
Statistical Test: Q2 vs Q4 Temporal Signal Analysis

Tests whether Q2 clearings show larger Y-1 to Y embedding distance than Q4 clearings.
This distinguishes between:
- Early detection (Q2 > Q4): Annual embeddings weight mid-year
- Precursor signal (Q4 > Q2): Detecting preparation activities before clearing
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# Load Phase 1 results
results_file = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation/phase1_glad_validation.json")

with open(results_file, 'r') as f:
    results = json.load(f)

print("=" * 80)
print("Q2 vs Q4 STATISTICAL TEST")
print("=" * 80)
print("\nHypothesis:")
print("  H0: Q2 and Q4 have equal mean distances")
print("  H1: Q2 and Q4 have different mean distances")
print("\nInterpretation:")
print("  - Q2 > Q4 (significant): Early detection of mid-year clearing")
print("  - Q4 > Q2 (significant): True precursor signal")
print("  - Q2 ≈ Q4 (not significant): Mixed signal")

# Extract Q2 and Q4 results
q2_results = results['quarterly_results']['2']
q4_results = results['quarterly_results']['4']

print("\n" + "-" * 80)
print("DATA SUMMARY")
print("-" * 80)
print(f"\nQ2 (April-June):")
print(f"  n = {q2_results['n_samples']}")
print(f"  Mean distance = {q2_results['mean_distance']:.4f}")
print(f"  Std deviation = {q2_results['std_distance']:.4f}")
print(f"  Std error = {q2_results['std_distance'] / np.sqrt(q2_results['n_samples']):.4f}")

print(f"\nQ4 (October-December):")
print(f"  n = {q4_results['n_samples']}")
print(f"  Mean distance = {q4_results['mean_distance']:.4f}")
print(f"  Std deviation = {q4_results['std_distance']:.4f}")
print(f"  Std error = {q4_results['std_distance'] / np.sqrt(q4_results['n_samples']):.4f}")

print(f"\nDifference:")
diff = q2_results['mean_distance'] - q4_results['mean_distance']
diff_pct = 100 * diff / q4_results['mean_distance']
print(f"  Q2 - Q4 = {diff:+.4f} ({diff_pct:+.1f}%)")
print(f"  Q2 is {diff_pct:.1f}% {'higher' if diff > 0 else 'lower'} than Q4")

# For the statistical test, we need the actual samples
# We'll reconstruct approximate samples from mean and std
# This is not ideal but necessary given we only have summary stats

print("\n" + "-" * 80)
print("STATISTICAL TESTS")
print("-" * 80)

# Method 1: Two-sample t-test with Welch's correction (unequal variances)
# Using summary statistics
n1, mean1, std1 = q2_results['n_samples'], q2_results['mean_distance'], q2_results['std_distance']
n2, mean2, std2 = q4_results['n_samples'], q4_results['mean_distance'], q4_results['std_distance']

# Calculate pooled standard error
se1 = std1 / np.sqrt(n1)
se2 = std2 / np.sqrt(n2)
se_diff = np.sqrt(se1**2 + se2**2)

# Welch's t-statistic
t_welch = (mean1 - mean2) / se_diff

# Welch-Satterthwaite degrees of freedom
num = (std1**2/n1 + std2**2/n2)**2
denom = (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
df_welch = num / denom

# Two-tailed p-value
p_welch_two = 2 * (1 - stats.t.cdf(abs(t_welch), df_welch))

# One-tailed p-value (Q2 > Q4)
p_welch_one = 1 - stats.t.cdf(t_welch, df_welch)

print("\n1. Welch's Two-Sample t-Test (unequal variances)")
print(f"   t-statistic = {t_welch:.4f}")
print(f"   Degrees of freedom = {df_welch:.2f}")
print(f"   Two-tailed p-value = {p_welch_two:.6f}")
print(f"   One-tailed p-value (Q2 > Q4) = {p_welch_one:.6f}")

significance_level = 0.05
if p_welch_two < significance_level:
    print(f"   ✓ SIGNIFICANT at α = {significance_level} (two-tailed)")
else:
    print(f"   ✗ NOT SIGNIFICANT at α = {significance_level} (two-tailed)")

if p_welch_one < significance_level:
    print(f"   ✓ SIGNIFICANT at α = {significance_level} (one-tailed: Q2 > Q4)")
else:
    print(f"   ✗ NOT SIGNIFICANT at α = {significance_level} (one-tailed: Q2 > Q4)")

# Calculate effect size (Cohen's d)
# Pooled standard deviation for unequal sample sizes
pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

print(f"\n2. Effect Size (Cohen's d)")
print(f"   Cohen's d = {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"

print(f"   Interpretation: {effect_interpretation.upper()} effect size")
print(f"   (Cohen's d: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large)")

# Calculate confidence interval for the difference
# 95% CI: diff ± t_crit * SE_diff
t_crit = stats.t.ppf(0.975, df_welch)  # Two-tailed 95% CI
ci_lower = diff - t_crit * se_diff
ci_upper = diff + t_crit * se_diff

print(f"\n3. 95% Confidence Interval for Difference (Q2 - Q4)")
print(f"   [{ci_lower:.4f}, {ci_upper:.4f}]")
if ci_lower > 0:
    print(f"   ✓ CI excludes zero: Q2 is significantly higher than Q4")
elif ci_upper < 0:
    print(f"   ✓ CI excludes zero: Q4 is significantly higher than Q2")
else:
    print(f"   ✗ CI includes zero: Difference not significant")

# Power analysis (post-hoc)
print(f"\n4. Statistical Power (Post-hoc)")
print(f"   Sample sizes: n1={n1}, n2={n2}")
print(f"   Effect size: Cohen's d = {cohens_d:.4f}")
print(f"   Note: Small sample sizes (n<10) reduce statistical power")
print(f"   Even large effects may not reach significance with n=3, n=6")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Determine conclusion based on statistical test
if p_welch_one < 0.05 and diff > 0:
    status = "EARLY_DETECTION"
    interpretation = f"""
✓ EARLY DETECTION CONFIRMED (Q2 > Q4)

Statistical Evidence:
  - Q2 mean distance ({mean1:.4f}) is {diff_pct:.1f}% higher than Q4 ({mean2:.4f})
  - Difference is statistically significant (p = {p_welch_one:.4f} < 0.05, one-tailed)
  - Effect size is {effect_interpretation} (Cohen's d = {cohens_d:.4f})

What This Means:
  Annual AlphaEarth embeddings are temporally weighted toward mid-year (Q2-Q3).

  Q2-Q3 clearings show STRONG signal in annual embeddings because:
    - Dry season (May-September) has more clear-sky observations
    - 2020 embedding captures 8-10 months of cleared land
    - Large difference from 2019 intact forest

  Q4 clearings show WEAK signal in annual embeddings because:
    - Only 3 months of clearing captured in 2020 annual aggregate
    - 2020 embedding still looks mostly forested (9 months intact)
    - Smaller difference from 2019 intact forest

Conclusion:
  We are detecting MID-YEAR CLEARING EVENTS, not precursor activities.

  Lead Time: 0-6 months (detecting clearings that happen Q2-Q3)
  System Type: Annual risk model, not early warning system

  Value Proposition:
    ✓ Predicts annual clearing risk from previous year
    ✓ Useful for resource allocation and planning
    ✓ Identifies high-risk areas for monitoring

    ✗ Not a true "early warning" with 9-15 month lead time
    ✗ Not detecting precursor activities (roads, camps)

  Recommendation: Proceed to WALK phase with honest framing
"""

elif p_welch_one < 0.05 and diff < 0:
    status = "TRUE_PRECURSOR"
    interpretation = f"""
✓ TRUE PRECURSOR SIGNAL DETECTED (Q4 > Q2)

Statistical Evidence:
  - Q4 mean distance ({mean2:.4f}) is {abs(diff_pct):.1f}% higher than Q2 ({mean1:.4f})
  - Difference is statistically significant (p < 0.05)
  - Effect size is {effect_interpretation} (Cohen's d = {abs(cohens_d):.4f})

What This Means:
  Y-1 embeddings capture preparation activities that precede late-year clearing.

  Q4 clearings show STRONG signal because:
    - 2019 embeddings capture late-year precursors (roads, camps, selective logging)
    - Large change from precursors to cleared state

  Q2 clearings show WEAKER signal because:
    - No precursor activities in 2019
    - Just detecting the clearing itself

Conclusion:
  We have a TRUE EARLY WARNING SYSTEM.

  Lead Time: 9-15 months (late 2019 precursors → late 2020 clearing)
  System Type: Precursor detection for early warning

  Value Proposition:
    ✓ Can predict clearings well in advance
    ✓ Provides time for intervention
    ✓ Detects preparation activities before clearing

  Recommendation: Proceed to WALK phase with confidence
"""

else:
    status = "MIXED_OR_INCONCLUSIVE"
    interpretation = f"""
~ MIXED SIGNAL or INCONCLUSIVE

Statistical Evidence:
  - Q2 mean distance ({mean1:.4f}) vs Q4 ({mean2:.4f})
  - Difference: {diff:+.4f} ({diff_pct:+.1f}%)
  - NOT statistically significant (p = {p_welch_two:.4f} > 0.05)
  - Sample sizes are small (n=6, n=3) → low statistical power

What This Means:
  We cannot definitively distinguish between early detection and precursor signal.

  Possible interpretations:
  1. Mixed signal: Both precursor detection and early detection occurring
  2. Insufficient power: True difference exists but sample too small to detect
  3. No difference: Q2 and Q4 clearings predicted equally well

Conclusion:
  System provides predictive value but temporal dynamics unclear.

  Lead Time: Variable 3-12 months (uncertain)
  System Type: Annual risk prediction with variable lead time

  Recommendation:
  - Option A: Scale to more samples for definitive answer
  - Option B: Proceed to WALK with honest framing about uncertainty
"""

print(interpretation)

# Save results
output = {
    'timestamp': results['timestamp'],
    'test': 'q2_vs_q4_comparison',
    'data': {
        'q2': {
            'n': n1,
            'mean': mean1,
            'std': std1,
            'se': se1,
        },
        'q4': {
            'n': n2,
            'mean': mean2,
            'std': std2,
            'se': se2,
        },
        'difference': {
            'absolute': diff,
            'percent': diff_pct,
        }
    },
    'statistical_test': {
        'test_name': 'Welch Two-Sample t-Test',
        't_statistic': t_welch,
        'degrees_of_freedom': df_welch,
        'p_value_two_tailed': p_welch_two,
        'p_value_one_tailed_q2_gt_q4': p_welch_one,
        'significant_at_0.05': p_welch_one < 0.05,
    },
    'effect_size': {
        'cohens_d': cohens_d,
        'interpretation': effect_interpretation,
        'pooled_std': pooled_std,
    },
    'confidence_interval_95': {
        'lower': ci_lower,
        'upper': ci_upper,
    },
    'conclusion': {
        'status': status,
        'interpretation': interpretation.strip(),
    }
}

output_file = Path("/Users/kartikganapathi/Documents/Personal/random_projects/green-ai-alphaearth/results/temporal_investigation/q2_vs_q4_test.json")
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("=" * 80)
