"""
WALK Phase - Comprehensive Spatial Feature Validation

Runs both Option A (CV performance) and Option B (feature understanding)
to validate whether spatial features help detect small-scale clearing.

Usage:
    uv run python src/walk/05_spatial_feature_validation.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils import get_config


def extract_feature_matrix(samples, include_spatial=False):
    """
    Extract feature matrix from samples.

    Args:
        samples: List of sample dicts
        include_spatial: Whether to include spatial features

    Returns:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
    """
    X = []
    feature_names = []

    for sample in samples:
        features_vec = []

        # Temporal features
        temporal_feats = sample.get('features', {})

        # Distances (4 features)
        for tp in ['Q1', 'Q2', 'Q3', 'Q4']:
            val = temporal_feats.get('distances', {}).get(tp, 0.0)
            features_vec.append(val)
            if len(feature_names) < 100:
                feature_names.append(f'distance_{tp}')

        # Velocities (3 features)
        for trans in ['Q1_Q2', 'Q2_Q3', 'Q3_Q4']:
            val = temporal_feats.get('velocities', {}).get(trans, 0.0)
            features_vec.append(val)
            if len(feature_names) < 100:
                feature_names.append(f'velocity_{trans}')

        # Accelerations (2 features)
        for acc in ['Q1_Q2_Q3', 'Q2_Q3_Q4']:
            val = temporal_feats.get('accelerations', {}).get(acc, 0.0)
            features_vec.append(val)
            if len(feature_names) < 100:
                feature_names.append(f'accel_{acc}')

        # Trend consistency (1 feature)
        val = temporal_feats.get('trend_consistency', 0.0)
        features_vec.append(val)
        if len(feature_names) < 100:
            feature_names.append('trend_consistency')

        # Spatial features (15 features)
        if include_spatial and 'spatial_features' in sample:
            spatial_feats = sample['spatial_features']

            # Neighborhood (6 features)
            for feat in [
                'neighbor_mean_distance',
                'neighbor_std_distance',
                'neighbor_max_distance',
                'neighbor_heterogeneity',
                'gradient_strength',
                'edge_score'
            ]:
                val = spatial_feats.get(feat, 0.0)
                features_vec.append(val)
                if len(feature_names) < 100:
                    feature_names.append(feat)

            # Texture (5 features)
            for feat in [
                'texture_contrast',
                'texture_dissimilarity',
                'texture_homogeneity',
                'texture_energy',
                'texture_correlation'
            ]:
                val = spatial_feats.get(feat, 0.0)
                features_vec.append(val)
                if len(feature_names) < 100:
                    feature_names.append(feat)

            # Edge (4 features)
            for feat in [
                'edge_mean',
                'edge_std',
                'edge_max',
                'edge_density'
            ]:
                val = spatial_feats.get(feat, 0.0)
                features_vec.append(val)
                if len(feature_names) < 100:
                    feature_names.append(feat)

        X.append(features_vec)

    return np.array(X), feature_names


def option_b_feature_understanding(X, y, feature_names):
    """
    Option B: Understand what spatial features capture.

    Args:
        X: Feature matrix with all features
        y: Labels
        feature_names: List of feature names

    Returns:
        Dict with analysis results
    """
    print("\n" + "="*80)
    print("OPTION B: FEATURE UNDERSTANDING")
    print("="*80)

    # Train Random Forest to get feature importances
    print("\nTraining Random Forest for feature importance...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X, y)

    # Feature importance ranking
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n📊 FEATURE IMPORTANCE RANKING")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Type'}")
    print("-" * 80)

    temporal_features = feature_names[:10]  # First 10 are temporal

    top_features = []
    for i, idx in enumerate(indices[:15]):
        feat_name = feature_names[idx]
        importance = importances[idx]
        feat_type = 'Temporal' if feat_name in temporal_features else 'Spatial'
        print(f"{i+1:<6} {feat_name:<35} {importance:<12.4f} {feat_type}")
        top_features.append({
            'rank': i+1,
            'feature': feat_name,
            'importance': importance,
            'type': feat_type
        })

    # Count spatial features in top-K
    spatial_in_top5 = sum(1 for f in top_features[:5] if f['type'] == 'Spatial')
    spatial_in_top10 = sum(1 for f in top_features[:10] if f['type'] == 'Spatial')

    print(f"\n📈 SPATIAL FEATURES IN TOP RANKS:")
    print(f"  Top-5:  {spatial_in_top5}/5 features are spatial")
    print(f"  Top-10: {spatial_in_top10}/10 features are spatial")

    # Correlation analysis
    print("\n🔗 CORRELATION ANALYSIS")
    print("-" * 80)

    # Split features
    X_temporal = X[:, :10]
    X_spatial = X[:, 10:]

    # Compute correlation between temporal and spatial
    temporal_spatial_corr = np.corrcoef(X_temporal.T, X_spatial.T)[:10, 10:]
    max_corr = np.max(np.abs(temporal_spatial_corr))
    mean_corr = np.mean(np.abs(temporal_spatial_corr))

    print(f"  Max correlation (temporal ↔ spatial): {max_corr:.3f}")
    print(f"  Mean correlation (temporal ↔ spatial): {mean_corr:.3f}")

    if mean_corr < 0.3:
        print("  ✓ Spatial features are ORTHOGONAL (low correlation)")
    elif mean_corr < 0.6:
        print("  ⚠ Spatial features are PARTIALLY CORRELATED")
    else:
        print("  ✗ Spatial features are REDUNDANT (high correlation)")

    return {
        'top_features': top_features,
        'spatial_in_top5': spatial_in_top5,
        'spatial_in_top10': spatial_in_top10,
        'max_correlation': max_corr,
        'mean_correlation': mean_corr
    }


def option_a_performance_validation(X_temporal, X_spatial, y):
    """
    Option A: Cross-validation performance comparison.

    Args:
        X_temporal: Feature matrix with temporal features only
        X_spatial: Feature matrix with temporal + spatial features
        y: Labels

    Returns:
        Dict with comparison results
    """
    print("\n" + "="*80)
    print("OPTION A: PERFORMANCE VALIDATION")
    print("="*80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Temporal only
    print("\n🔍 Model 1: Temporal Features Only (10 features)")
    print("-" * 80)

    lr_temporal = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=0.1  # Heavy regularization
    )

    # Normalize features
    scaler_temporal = StandardScaler()
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal)

    scores_temporal = cross_val_score(
        lr_temporal,
        X_temporal_scaled,
        y,
        cv=cv,
        scoring='roc_auc'
    )

    print(f"  ROC-AUC: {np.mean(scores_temporal):.3f} ± {np.std(scores_temporal):.3f}")
    print(f"  Folds: {[f'{s:.3f}' for s in scores_temporal]}")

    # Model 2: Temporal + Spatial
    print("\n🔍 Model 2: Temporal + Spatial Features (25 features)")
    print("-" * 80)

    lr_spatial = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=0.1  # Heavy regularization
    )

    scaler_spatial = StandardScaler()
    X_spatial_scaled = scaler_spatial.fit_transform(X_spatial)

    scores_spatial = cross_val_score(
        lr_spatial,
        X_spatial_scaled,
        y,
        cv=cv,
        scoring='roc_auc'
    )

    print(f"  ROC-AUC: {np.mean(scores_spatial):.3f} ± {np.std(scores_spatial):.3f}")
    print(f"  Folds: {[f'{s:.3f}' for s in scores_spatial]}")

    # Statistical comparison
    print("\n📊 STATISTICAL COMPARISON")
    print("-" * 80)

    improvement = np.mean(scores_spatial) - np.mean(scores_temporal)
    improvement_pct = (improvement / np.mean(scores_temporal)) * 100

    # Paired t-test
    t_stat, p_value = ttest_rel(scores_spatial, scores_temporal)

    print(f"  Improvement: {improvement:+.3f} ROC-AUC ({improvement_pct:+.1f}%)")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

    if p_value < 0.05 and improvement > 0:
        print(f"  ✓ SIGNIFICANT IMPROVEMENT (p < 0.05)")
    elif improvement > 0.05:
        print(f"  ⚠ IMPROVEMENT DETECTED but not statistically significant")
    else:
        print(f"  ✗ NO SIGNIFICANT IMPROVEMENT")

    # Variance analysis
    var_temporal = np.std(scores_temporal)
    var_spatial = np.std(scores_spatial)

    print(f"\n  Variance (temporal): {var_temporal:.3f}")
    print(f"  Variance (spatial):  {var_spatial:.3f}")

    if var_temporal > 0.10 or var_spatial > 0.10:
        print(f"  ⚠ HIGH VARIANCE - Results may be unstable with {len(y)} samples")

    return {
        'temporal_auc': np.mean(scores_temporal),
        'temporal_std': np.std(scores_temporal),
        'spatial_auc': np.mean(scores_spatial),
        'spatial_std': np.std(scores_spatial),
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        'significant': p_value < 0.05 and improvement > 0
    }


def synthesize_results(option_a_results, option_b_results):
    """
    Synthesize results from both analyses into clear recommendation.
    """
    print("\n" + "="*80)
    print("🎯 SYNTHESIS AND RECOMMENDATION")
    print("="*80)

    # Determine scenario
    improvement = option_a_results['improvement']
    significant = option_a_results['significant']
    spatial_in_top5 = option_b_results['spatial_in_top5']
    mean_corr = option_b_results['mean_correlation']

    print(f"\n📋 DECISION MATRIX:")
    print(f"  Performance improvement:  {improvement:+.3f} ROC-AUC ({option_a_results['improvement_pct']:+.1f}%)")
    print(f"  Statistical significance: {'YES' if significant else 'NO'} (p={option_a_results['p_value']:.4f})")
    print(f"  Spatial in top-5:         {spatial_in_top5}/5 features")
    print(f"  Feature orthogonality:    {'HIGH' if mean_corr < 0.3 else 'MODERATE' if mean_corr < 0.6 else 'LOW'} (corr={mean_corr:.3f})")

    print(f"\n🎯 RECOMMENDATION:")
    print("-" * 80)

    # Scenario 1: Strong evidence for spatial features
    if improvement > 0.05 and significant and spatial_in_top5 >= 2 and mean_corr < 0.6:
        print("✅ STRONG VALIDATION - Proceed with spatial features")
        print("\n  Evidence:")
        print(f"    • {improvement:+.1%} improvement in ROC-AUC (significant, p<0.05)")
        print(f"    • {spatial_in_top5} spatial features in top-5 importance")
        print(f"    • Low correlation with temporal features (orthogonal signal)")
        print("\n  Next Steps:")
        print("    1. Extract spatial features for full training set (if not already done)")
        print("    2. Extract spatial features for all validation sets")
        print("    3. Re-evaluate on rapid response set")
        print("    4. Expected improvement: 30-40% → 50-60% recall")
        recommendation = "ADOPT_SPATIAL"

    # Scenario 2: Weak evidence - marginal improvement
    elif improvement > 0.02 and spatial_in_top5 >= 1:
        print("⚠️  WEAK VALIDATION - Spatial features help marginally")
        print("\n  Evidence:")
        print(f"    • {improvement:+.1%} improvement (small)")
        print(f"    • {spatial_in_top5} spatial feature(s) in top-5")
        print(f"    • {'Orthogonal' if mean_corr < 0.3 else 'Partially correlated'} with temporal")
        print("\n  Interpretation:")
        print("    Spatial features add signal but improvement may not justify complexity.")
        print("\n  Next Steps:")
        print("    1. Test on rapid response set to see if improvement holds")
        print("    2. Consider simpler spatial features (just neighborhood stats)")
        print("    3. If improvement < 5% on validation, skip spatial features")
        recommendation = "TEST_ON_VALIDATION"

    # Scenario 3: Redundant features
    elif improvement < 0.02 and spatial_in_top5 == 0:
        print("❌ NO VALIDATION - Spatial features don't help")
        print("\n  Evidence:")
        print(f"    • {improvement:+.1%} improvement (negligible)")
        print(f"    • No spatial features in top-5 importance")
        print(f"    • Features may be {'redundant' if mean_corr > 0.6 else 'noisy'}")
        print("\n  Interpretation:")
        print("    Spatial context not captured by current features OR")
        print("    Temporal features already capture the signal.")
        print("\n  Next Steps:")
        print("    1. Try different spatial features:")
        print("       - Multi-scale embeddings (30m, 100m, 300m)")
        print("       - Spectral bands directly (NDVI, NBR)")
        print("       - Road/settlement proximity")
        print("    2. Or focus on fire classifier instead")
        print("    3. Or scale up dataset size")
        recommendation = "TRY_DIFFERENT_FEATURES"

    # Scenario 4: Surprising case - improvement despite low importance
    else:
        print("🤔 UNEXPECTED - Results are mixed")
        print("\n  Evidence:")
        print(f"    • Performance: {improvement:+.1%}")
        print(f"    • Importance: {spatial_in_top5}/5 in top-5")
        print(f"    • Correlation: {mean_corr:.3f}")
        print("\n  Interpretation:")
        print("    Results don't fit clear pattern. May indicate:")
        print("    - Small dataset causing noise")
        print("    - Feature interactions (ensemble effects)")
        print("    - Overfitting")
        print("\n  Next Steps:")
        print("    1. Increase dataset size (current: 114 samples too small)")
        print("    2. Re-run analysis with 300+ samples")
        recommendation = "NEED_MORE_DATA"

    return {
        'recommendation': recommendation,
        'improvement': improvement,
        'significant': significant,
        'spatial_important': spatial_in_top5 >= 2,
        'orthogonal': mean_corr < 0.3
    }


def main():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("="*80)
    print("SPATIAL FEATURE VALIDATION - COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load dataset with spatial features
    dataset_file = processed_dir / "walk_dataset_spatial.pkl"

    if not dataset_file.exists():
        print(f"\n✗ Error: {dataset_file} not found")
        print("  Run: uv run python src/walk/01e_extract_spatial_for_training.py")
        return

    print(f"\nLoading dataset from {dataset_file.name}...")
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

    all_samples = dataset['data']
    train_indices = dataset['splits']['train']

    # Get training samples
    train_samples = [all_samples[i] for i in train_indices]

    print(f"Loaded {len(train_samples)} training samples")

    # Count samples with spatial features
    n_with_spatial = sum(1 for s in train_samples if 'spatial_features' in s)
    print(f"  Samples with spatial features: {n_with_spatial}/{len(train_samples)}")

    if n_with_spatial < len(train_samples) * 0.8:
        print(f"\n⚠ Warning: Only {n_with_spatial/len(train_samples)*100:.1f}% have spatial features")
        print("  Some samples may have failed extraction.")

    # Extract features
    print("\nExtracting feature matrices...")
    X_temporal, temporal_names = extract_feature_matrix(train_samples, include_spatial=False)
    X_spatial, all_names = extract_feature_matrix(train_samples, include_spatial=True)
    y = np.array([s['label'] for s in train_samples])

    print(f"  Temporal features: {X_temporal.shape}")
    print(f"  Temporal + Spatial: {X_spatial.shape}")
    print(f"  Labels: {y.shape} ({np.sum(y)} clearing, {len(y)-np.sum(y)} intact)")

    # Run Option B: Feature Understanding
    option_b_results = option_b_feature_understanding(X_spatial, y, all_names)

    # Run Option A: Performance Validation
    option_a_results = option_a_performance_validation(X_temporal, X_spatial, y)

    # Synthesize and recommend
    synthesis = synthesize_results(option_a_results, option_b_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print(f"\nResults saved for reference:")
    print(f"  • Dataset: {dataset_file}")
    print(f"  • Recommendation: {synthesis['recommendation']}")


if __name__ == "__main__":
    main()
