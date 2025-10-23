"""
Comprehensive Diagnostic Analysis for All Validation Sets

Analyzes all 4 validation sets to understand root causes of performance variability:
- risk_ranking: 0.850 ROC-AUC (46 samples)
- rapid_response: 0.824 ROC-AUC (27 samples)
- comprehensive: 0.758 ROC-AUC (69 samples)
- edge_cases: 0.583 ROC-AUC (23 samples)

Outputs:
1. Feature distribution comparison (KS tests, overlap metrics)
2. Error pattern analysis (characterize missed clearings)
3. Feature importance comparison (per-set models)
4. Learning curve analysis (estimate data needs)
5. Comparative analysis (gradient vs clusters vs unique patterns)
6. Recommendations (Phase 1 scaling vs Phase 2 specialization)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple

from src.utils import EarthEngineClient, get_config


# Feature names for interpretation
FEATURE_NAMES = [
    # Baseline features (Y-1) - indices 0-9
    "baseline_q1_magnitude", "baseline_q2_magnitude",
    "baseline_q3_magnitude", "baseline_q4_magnitude",
    "baseline_dist_mean", "baseline_dist_max",
    "baseline_vel_mean", "baseline_vel_max",
    "baseline_accel_mean", "baseline_trend",
    # Delta features (Y - Y-1) - indices 10-16
    "delta_q1_magnitude", "delta_q2_magnitude",
    "delta_q3_magnitude", "delta_q4_magnitude",
    "delta_mean", "delta_max", "delta_trend"
]


def extract_dual_year_features(client, sample: dict) -> np.ndarray:
    """
    Extract dual-year delta features for a sample.

    Returns:
        17-dimensional feature vector (10 baseline + 7 delta)
        or None if extraction fails
    """
    lat, lon = sample['lat'], sample['lon']
    year = sample['year']

    try:
        # Get Y-1 quarterly embeddings (baseline)
        y_minus_1_embeddings = []
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year-1}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None or len(emb) == 0:
                return None
            y_minus_1_embeddings.append(np.array(emb))

        # Get Y quarterly embeddings (current year)
        y_embeddings = []
        for q, month in [(1, '03'), (2, '06'), (3, '09'), (4, '12')]:
            date = f"{year}-{month}-01"
            emb = client.get_embedding(lat, lon, date)
            if emb is None or len(emb) == 0:
                return None
            y_embeddings.append(np.array(emb))

        # Baseline features (Y-1): 10 features
        baseline_features = []

        # 1. Quarterly magnitudes (4 features)
        for emb in y_minus_1_embeddings:
            baseline_features.append(np.linalg.norm(emb))

        # 2. Inter-quarter distances (2 features: mean, max)
        distances = [
            np.linalg.norm(y_minus_1_embeddings[i+1] - y_minus_1_embeddings[i])
            for i in range(3)
        ]
        baseline_features.extend([np.mean(distances), np.max(distances)])

        # 3. Velocities (2 features: mean, max)
        velocities = distances  # Same as distances for quarterly data
        baseline_features.extend([np.mean(velocities), np.max(velocities)])

        # 4. Accelerations (1 feature: mean)
        accelerations = [velocities[i+1] - velocities[i] for i in range(2)]
        baseline_features.append(np.mean(accelerations))

        # 5. Trend (1 feature)
        trend = np.linalg.norm(y_minus_1_embeddings[-1] - y_minus_1_embeddings[0])
        baseline_features.append(trend)

        # Delta features (Y - Y-1): 7 features
        delta_features = []

        # 1. Quarterly delta magnitudes (4 features)
        for y_emb, y1_emb in zip(y_embeddings, y_minus_1_embeddings):
            delta = y_emb - y1_emb
            delta_features.append(np.linalg.norm(delta))

        # 2. Mean delta magnitude (1 feature)
        delta_features.append(np.mean(delta_features[:4]))

        # 3. Max delta magnitude (1 feature)
        delta_features.append(np.max(delta_features[:4]))

        # 4. Delta trend (1 feature)
        delta_trend = np.linalg.norm(
            (y_embeddings[-1] - y_minus_1_embeddings[-1]) -
            (y_embeddings[0] - y_minus_1_embeddings[0])
        )
        delta_features.append(delta_trend)

        # Combine all features
        all_features = baseline_features + delta_features

        return np.array(all_features)

    except Exception as e:
        print(f"Error extracting features for {lat}, {lon}, {year}: {e}")
        return None


def load_validation_set(path: Path) -> List[dict]:
    """Load a validation set from pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Add 'label' field based on 'stable' field
    # stable=True means intact (0), stable=False or missing means clearing (1)
    # Use numeric labels for consistency with sklearn: 1=clearing, 0=intact
    for sample in data:
        if 'label' not in sample:
            sample['label'] = 0 if sample.get('stable', False) else 1

        # Ensure year field exists (default to 2021 as used in the construction scripts)
        if 'year' not in sample:
            sample['year'] = sample.get('date', '2021-06-01').split('-')[0]
            sample['year'] = int(sample['year'])

    return data


def extract_features_for_set(client, samples: List[dict], set_name: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Extract features for all samples in a validation set.

    Returns:
        X: Feature matrix (n_samples, 17)
        y: Labels (n_samples,)
        valid_samples: Samples that had successful feature extraction
    """
    print(f"\nExtracting features for {set_name}...")

    X_list = []
    y_list = []
    valid_samples = []

    for sample in samples:
        features = extract_dual_year_features(client, sample)
        if features is not None:
            X_list.append(features)
            # Handle both numeric (0/1) and string ('clearing'/'intact') labels
            label = sample.get('label')
            if isinstance(label, str):
                y_list.append(1 if label == 'clearing' else 0)
            else:
                y_list.append(int(label))  # Numeric label: 1=clearing, 0=intact
            valid_samples.append(sample)

    print(f"  Successfully extracted: {len(X_list)}/{len(samples)} samples")

    return np.array(X_list), np.array(y_list), valid_samples


def analyze_feature_distributions(
    train_X: np.ndarray,
    val_sets: Dict[str, np.ndarray]
) -> Dict:
    """
    Compare feature distributions between training and validation sets.

    Uses Kolmogorov-Smirnov tests and overlap metrics.
    """
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)

    results = {}

    for set_name, val_X in val_sets.items():
        print(f"\n{set_name}:")

        set_results = {
            'ks_tests': [],
            'mean_differences': [],
            'std_ratios': []
        }

        for i, feature_name in enumerate(FEATURE_NAMES):
            # KS test
            ks_stat, ks_pval = ks_2samp(train_X[:, i], val_X[:, i])

            # Mean difference (normalized by training std)
            train_mean = np.mean(train_X[:, i])
            val_mean = np.mean(val_X[:, i])
            train_std = np.std(train_X[:, i])
            mean_diff = (val_mean - train_mean) / (train_std + 1e-10)

            # Std ratio
            val_std = np.std(val_X[:, i])
            std_ratio = val_std / (train_std + 1e-10)

            set_results['ks_tests'].append({
                'feature': feature_name,
                'statistic': float(ks_stat),
                'p_value': float(ks_pval),
                'significant': ks_pval < 0.05
            })
            set_results['mean_differences'].append({
                'feature': feature_name,
                'difference': float(mean_diff)
            })
            set_results['std_ratios'].append({
                'feature': feature_name,
                'ratio': float(std_ratio)
            })

            # Print significant differences
            if ks_pval < 0.05:
                print(f"  ⚠ {feature_name}: KS={ks_stat:.3f}, p={ks_pval:.4f}, "
                      f"mean_diff={mean_diff:.2f}σ, std_ratio={std_ratio:.2f}")

        # Summary statistics
        n_significant = sum(1 for t in set_results['ks_tests'] if t['significant'])
        mean_ks = np.mean([t['statistic'] for t in set_results['ks_tests']])

        set_results['summary'] = {
            'n_significant_features': n_significant,
            'pct_significant': n_significant / len(FEATURE_NAMES) * 100,
            'mean_ks_statistic': float(mean_ks)
        }

        print(f"\n  Summary: {n_significant}/{len(FEATURE_NAMES)} features "
              f"significantly different (mean KS={mean_ks:.3f})")

        results[set_name] = set_results

    return results


def analyze_error_patterns(
    client,
    val_sets_data: Dict[str, dict],
    models: Dict[str, LogisticRegression]
) -> Dict:
    """
    Analyze error patterns: characterize missed clearings.

    For each validation set, identify which clearings were missed
    and extract their characteristics.
    """
    print("\n" + "="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)

    results = {}

    for set_name, data in val_sets_data.items():
        print(f"\n{set_name}:")

        X = data['X']
        y = data['y']
        samples = data['samples']
        model = models[set_name]

        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Find clearing indices
        clearing_indices = np.where(y == 1)[0]

        # Find missed clearings (false negatives)
        missed = [i for i in clearing_indices if y_pred[i] == 0]
        detected = [i for i in clearing_indices if y_pred[i] == 1]

        print(f"  Clearings: {len(clearing_indices)} total")
        print(f"  Detected: {len(detected)} ({len(detected)/len(clearing_indices)*100:.1f}%)")
        print(f"  Missed: {len(missed)} ({len(missed)/len(clearing_indices)*100:.1f}%)")

        # Characterize missed clearings
        if len(missed) > 0:
            missed_features = X[missed]
            detected_features = X[detected] if len(detected) > 0 else None

            # Compare feature distributions
            print(f"\n  Missed clearing characteristics:")

            # Focus on delta features (indices 10-16) - most important
            delta_indices = list(range(10, 17))

            for idx in delta_indices:
                missed_mean = np.mean(missed_features[:, idx])
                if detected_features is not None:
                    detected_mean = np.mean(detected_features[:, idx])
                    diff = missed_mean - detected_mean
                    print(f"    {FEATURE_NAMES[idx]}: "
                          f"missed={missed_mean:.3f}, detected={detected_mean:.3f}, "
                          f"diff={diff:.3f}")
                else:
                    print(f"    {FEATURE_NAMES[idx]}: missed={missed_mean:.3f}")

            # Store results
            results[set_name] = {
                'n_clearings': len(clearing_indices),
                'n_detected': len(detected),
                'n_missed': len(missed),
                'recall': len(detected) / len(clearing_indices) if len(clearing_indices) > 0 else 0,
                'missed_samples': [samples[i] for i in missed],
                'missed_predictions': [float(y_proba[i]) for i in missed],
                'detected_predictions': [float(y_proba[i]) for i in detected]
            }
        else:
            results[set_name] = {
                'n_clearings': len(clearing_indices),
                'n_detected': len(detected),
                'n_missed': 0,
                'recall': 1.0
            }

    return results


def analyze_feature_importance(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_sets: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict:
    """
    Train separate models on each validation set and compare feature importance.

    If all sets use similar features → scaling helps
    If different features → specialization needed
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    results = {}

    # Train model on training set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)

    train_model = LogisticRegression(random_state=42, max_iter=1000)
    train_model.fit(X_scaled, train_y)

    train_importance = np.abs(train_model.coef_[0])

    print(f"\nTraining set feature importance (top 5):")
    top_indices = np.argsort(train_importance)[-5:][::-1]
    for idx in top_indices:
        print(f"  {FEATURE_NAMES[idx]}: {train_importance[idx]:.3f}")

    results['training'] = {
        'importance': train_importance.tolist(),
        'top_features': [FEATURE_NAMES[i] for i in top_indices]
    }

    # Train separate models for each validation set
    for set_name, (val_X, val_y) in val_sets.items():
        print(f"\n{set_name}:")

        # Need enough samples to train
        if len(val_X) < 10:
            print(f"  Too few samples ({len(val_X)}) to train model")
            continue

        # Scale and train
        val_scaler = StandardScaler()
        val_X_scaled = val_scaler.fit_transform(val_X)

        val_model = LogisticRegression(random_state=42, max_iter=1000)
        val_model.fit(val_X_scaled, val_y)

        val_importance = np.abs(val_model.coef_[0])

        # Print top features
        top_indices = np.argsort(val_importance)[-5:][::-1]
        print(f"  Top 5 features:")
        for idx in top_indices:
            print(f"    {FEATURE_NAMES[idx]}: {val_importance[idx]:.3f}")

        # Compute correlation with training importance
        correlation = np.corrcoef(train_importance, val_importance)[0, 1]
        print(f"  Correlation with training: {correlation:.3f}")

        results[set_name] = {
            'importance': val_importance.tolist(),
            'top_features': [FEATURE_NAMES[i] for i in top_indices],
            'correlation_with_training': float(correlation)
        }

    return results


def generate_learning_curves(
    client,
    train_samples: List[dict],
    val_sets_data: Dict[str, dict],
    sample_sizes: List[int] = [50, 100, 150, 200, 250, 300]
) -> Dict:
    """
    Generate learning curves to estimate how much more data would help.

    Sample different amounts of training data and measure validation performance.
    """
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)

    results = {}

    # Extract all training features
    print("\nExtracting training features...")
    train_X_list = []
    train_y_list = []

    for sample in train_samples:
        features = extract_dual_year_features(client, sample)
        if features is not None:
            train_X_list.append(features)
            # Handle both numeric (0/1) and string ('clearing'/'intact') labels
            label = sample.get('label')
            if isinstance(label, str):
                train_y_list.append(1 if label == 'clearing' else 0)
            else:
                train_y_list.append(int(label))  # Numeric label: 1=clearing, 0=intact

    train_X_full = np.array(train_X_list)
    train_y_full = np.array(train_y_list)

    print(f"  Total training samples: {len(train_X_full)}")

    # For each validation set
    for set_name, data in val_sets_data.items():
        print(f"\n{set_name}:")

        val_X = data['X']
        val_y = data['y']

        set_results = []

        for n in sample_sizes:
            if n > len(train_X_full):
                continue

            # Sample training data
            indices = np.random.choice(len(train_X_full), size=n, replace=False)
            X_sample = train_X_full[indices]
            y_sample = train_y_full[indices]

            # Train and evaluate
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)

            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y_sample)

            # Evaluate on validation set
            val_X_scaled = scaler.transform(val_X)
            val_proba = model.predict_proba(val_X_scaled)[:, 1]

            auc = roc_auc_score(val_y, val_proba)

            set_results.append({
                'n_samples': n,
                'roc_auc': float(auc)
            })

            print(f"  n={n}: ROC-AUC={auc:.3f}")

        results[set_name] = set_results

    return results


def comparative_analysis(
    distribution_results: Dict,
    error_results: Dict,
    importance_results: Dict
) -> Dict:
    """
    Perform comparative analysis across all validation sets.

    Key questions:
    1. Is performance gap a gradient or are there clusters?
    2. Do low-performing sets have unique characteristics?
    3. Are feature importance patterns consistent or different?
    """
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)

    # Sort sets by performance
    performance_order = [
        ('risk_ranking', 0.850),
        ('rapid_response', 0.824),
        ('comprehensive', 0.758),
        ('edge_cases', 0.583)
    ]

    print("\nPerformance spectrum:")
    for set_name, auc in performance_order:
        print(f"  {set_name}: {auc:.3f} ROC-AUC")

    # 1. Distribution shift analysis
    print("\n" + "-"*60)
    print("1. Distribution Shift Pattern")
    print("-"*60)

    print("\nPercentage of significantly different features:")
    for set_name, _ in performance_order:
        if set_name in distribution_results:
            pct = distribution_results[set_name]['summary']['pct_significant']
            print(f"  {set_name}: {pct:.1f}%")

    # 2. Feature importance correlation
    print("\n" + "-"*60)
    print("2. Feature Importance Correlation with Training")
    print("-"*60)

    for set_name, _ in performance_order:
        if set_name in importance_results and 'correlation_with_training' in importance_results[set_name]:
            corr = importance_results[set_name]['correlation_with_training']
            print(f"  {set_name}: {corr:.3f}")

    # 3. Error rate analysis
    print("\n" + "-"*60)
    print("3. Recall Rates (Clearing Detection)")
    print("-"*60)

    for set_name, _ in performance_order:
        if set_name in error_results:
            recall = error_results[set_name]['recall']
            print(f"  {set_name}: {recall:.3f}")

    # Generate insights
    insights = []

    # Check if gradient or clusters
    aucs = [auc for _, auc in performance_order]
    gaps = [aucs[i] - aucs[i+1] for i in range(len(aucs)-1)]
    max_gap = max(gaps)
    avg_gap = np.mean(gaps)

    if max_gap > 2 * avg_gap:
        insights.append({
            'type': 'CLUSTER_PATTERN',
            'description': f'Large gap ({max_gap:.3f}) suggests clusters rather than gradient',
            'implication': 'May need specialized models for distinct groups'
        })
    else:
        insights.append({
            'type': 'GRADIENT_PATTERN',
            'description': f'Relatively uniform gaps (max={max_gap:.3f}, avg={avg_gap:.3f})',
            'implication': 'Scaling may help all sets uniformly'
        })

    # Check feature importance consistency
    if len(importance_results) >= 3:
        correlations = [
            importance_results[name]['correlation_with_training']
            for name, _ in performance_order
            if name in importance_results and 'correlation_with_training' in importance_results[name]
        ]

        if len(correlations) > 0:
            mean_corr = np.mean(correlations)

            if mean_corr > 0.7:
                insights.append({
                    'type': 'CONSISTENT_FEATURES',
                    'description': f'High feature importance correlation (mean={mean_corr:.3f})',
                    'implication': 'All sets use similar features → single model appropriate'
                })
            else:
                insights.append({
                    'type': 'DIVERGENT_FEATURES',
                    'description': f'Low feature importance correlation (mean={mean_corr:.3f})',
                    'implication': 'Sets use different features → specialization may help'
                })

    return {
        'performance_order': performance_order,
        'insights': insights
    }


def generate_recommendations(
    comparative_results: Dict,
    learning_curve_results: Dict
) -> Dict:
    """
    Generate actionable recommendations based on all analyses.

    Decides: Phase 1 (scaling) vs Phase 2 (specialization)
    """
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    insights = comparative_results['insights']

    # Decision logic
    recommendation = None
    confidence = None
    reasoning = []

    # Check for gradient pattern
    has_gradient = any(i['type'] == 'GRADIENT_PATTERN' for i in insights)
    has_cluster = any(i['type'] == 'CLUSTER_PATTERN' for i in insights)

    # Check for consistent features
    has_consistent = any(i['type'] == 'CONSISTENT_FEATURES' for i in insights)
    has_divergent = any(i['type'] == 'DIVERGENT_FEATURES' for i in insights)

    # Count evidence
    evidence_for_scaling = 0
    evidence_for_specialization = 0

    if has_gradient:
        evidence_for_scaling += 2
        reasoning.append("✓ Gradient pattern suggests scaling helps uniformly")

    if has_cluster:
        evidence_for_specialization += 2
        reasoning.append("⚠ Cluster pattern suggests distinct groups")

    if has_consistent:
        evidence_for_scaling += 2
        reasoning.append("✓ Consistent features across sets")

    if has_divergent:
        evidence_for_specialization += 2
        reasoning.append("⚠ Divergent features suggest different patterns")

    # Check learning curves (if available)
    if learning_curve_results:
        # Check if edge_cases improves with more data
        if 'edge_cases' in learning_curve_results:
            curves = learning_curve_results['edge_cases']
            if len(curves) >= 2:
                first_auc = curves[0]['roc_auc']
                last_auc = curves[-1]['roc_auc']
                improvement = last_auc - first_auc

                if improvement > 0.05:
                    evidence_for_scaling += 1
                    reasoning.append(f"✓ Edge cases improve with more data (+{improvement:.3f} AUC)")
                else:
                    evidence_for_specialization += 1
                    reasoning.append(f"⚠ Edge cases plateau with more data (+{improvement:.3f} AUC)")

    # Make recommendation
    if evidence_for_scaling > evidence_for_specialization:
        recommendation = "PHASE_1_SCALING"
        confidence = "HIGH" if evidence_for_scaling >= 4 else "MEDIUM"
        action = (
            "Proceed with Phase 1: Scale up training data to 300 samples\n"
            "  - Target 60% standard clearings (>1 ha)\n"
            "  - Target 20% small clearings (<1 ha)\n"
            "  - Target 10% fire-prone areas\n"
            "  - Target 10% forest edges\n"
            "  - Expected: Edge cases 0.583 → 0.70+ ROC-AUC"
        )
    elif evidence_for_specialization > evidence_for_scaling:
        recommendation = "PHASE_2_SPECIALIZATION"
        confidence = "MEDIUM"
        action = (
            "Consider Phase 2: Build specialized models\n"
            "  - Standard model (risk_ranking, rapid_response, comprehensive)\n"
            "  - Edge case model (small, fire, fragmented)\n"
            "  - Simple routing: size < 1ha OR fire_prone OR fragmented → edge model\n"
            "  - Data needs: 300-400 samples total"
        )
    else:
        recommendation = "PHASE_1_SCALING"
        confidence = "LOW"
        action = (
            "Start with Phase 1: Scale up training data (less risky)\n"
            "  - Evidence is mixed, but scaling is simpler and faster\n"
            "  - Can always specialize later if needed\n"
            "  - Target: 300 samples with diversity"
        )
        reasoning.append("⚠ Mixed evidence → default to simpler approach (scaling)")

    # Print recommendation
    print(f"\nRecommendation: {recommendation}")
    print(f"Confidence: {confidence}")
    print(f"\n{action}")

    print(f"\nReasoning:")
    for r in reasoning:
        print(f"  {r}")

    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'action': action,
        'reasoning': reasoning,
        'evidence_for_scaling': evidence_for_scaling,
        'evidence_for_specialization': evidence_for_specialization
    }


def main():
    """Run comprehensive diagnostic analysis."""

    print("="*60)
    print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("="*60)
    print("\nAnalyzing all 4 validation sets:")
    print("  - risk_ranking: 0.850 ROC-AUC (46 samples)")
    print("  - rapid_response: 0.824 ROC-AUC (27 samples)")
    print("  - comprehensive: 0.758 ROC-AUC (69 samples)")
    print("  - edge_cases: 0.583 ROC-AUC (23 samples)")

    # Initialize client
    config = get_config()
    client = EarthEngineClient(use_cache=True)

    # Load validation sets
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    base_path = data_dir / 'processed'

    val_paths = {
        'risk_ranking': base_path / 'hard_val_risk_ranking.pkl',
        'rapid_response': base_path / 'hard_val_rapid_response.pkl',
        'comprehensive': base_path / 'hard_val_comprehensive.pkl',
        'edge_cases': base_path / 'hard_val_edge_cases.pkl'
    }

    val_samples = {}
    for name, path in val_paths.items():
        val_samples[name] = load_validation_set(path)
        print(f"\nLoaded {name}: {len(val_samples[name])} samples")

    # Load training set
    train_path = base_path / 'walk_dataset.pkl'
    print(f"\nLoading training set from: {train_path}")

    with open(train_path, 'rb') as f:
        training_data = pickle.load(f)

    # Extract samples from training data
    if isinstance(training_data, dict) and 'data' in training_data:
        train_samples = training_data['data']
    else:
        train_samples = training_data

    # Ensure each sample has required fields
    for sample in train_samples:
        # Extract location if nested
        if 'location' in sample and isinstance(sample['location'], dict):
            if 'lat' not in sample:
                sample['lat'] = sample['location']['lat']
            if 'lon' not in sample:
                sample['lon'] = sample['location']['lon']

        # Ensure numeric label (don't overwrite if it exists!)
        # Training set uses 1=clearing, 0=intact (numeric)
        if 'label' not in sample:
            # If no label but has 'stable' field, convert it
            sample['label'] = 0 if sample.get('stable', False) else 1

        # Add year field (use date if available, otherwise default)
        if 'year' not in sample:
            if 'date' in sample:
                sample['year'] = int(sample['date'].split('-')[0])
            else:
                sample['year'] = 2021  # Default

    print(f"Loaded training set: {len(train_samples)} samples")

    # Extract features for all sets
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)

    # Training set
    train_X, train_y, _ = extract_features_for_set(client, train_samples, 'training')

    # Validation sets
    val_sets_data = {}
    for name, samples in val_samples.items():
        X, y, valid_samples = extract_features_for_set(client, samples, name)
        val_sets_data[name] = {
            'X': X,
            'y': y,
            'samples': valid_samples
        }

    # Train models for each validation set
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    models = {}
    for name, data in val_sets_data.items():
        print(f"\nTraining model for {name}...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_X)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, train_y)

        # Evaluate
        val_X_scaled = scaler.transform(data['X'])
        val_proba = model.predict_proba(val_X_scaled)[:, 1]
        auc = roc_auc_score(data['y'], val_proba)

        print(f"  ROC-AUC: {auc:.3f}")

        models[name] = model

    # Run analyses

    # 1. Feature distribution analysis
    val_X_dict = {name: data['X'] for name, data in val_sets_data.items()}
    distribution_results = analyze_feature_distributions(train_X, val_X_dict)

    # 2. Error pattern analysis
    error_results = analyze_error_patterns(client, val_sets_data, models)

    # 3. Feature importance analysis
    val_Xy_dict = {name: (data['X'], data['y']) for name, data in val_sets_data.items()}
    importance_results = analyze_feature_importance(train_X, train_y, val_Xy_dict)

    # 4. Learning curve analysis (sample sizes based on current training set size)
    current_train_size = len(train_X)
    sample_sizes = [
        int(current_train_size * 0.5),
        int(current_train_size * 0.75),
        current_train_size,
        min(int(current_train_size * 1.5), 300),
        min(int(current_train_size * 2.0), 300)
    ]
    sample_sizes = sorted(list(set([s for s in sample_sizes if s <= 300])))

    learning_curve_results = generate_learning_curves(
        client, train_samples, val_sets_data, sample_sizes
    )

    # 5. Comparative analysis
    comparative_results = comparative_analysis(
        distribution_results,
        error_results,
        importance_results
    )

    # 6. Generate recommendations
    recommendations = generate_recommendations(
        comparative_results,
        learning_curve_results
    )

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'validation_sets': {
            name: {
                'n_samples': len(data['samples']),
                'n_clearing': int(np.sum(data['y'])),
                'n_intact': int(len(data['y']) - np.sum(data['y']))
            }
            for name, data in val_sets_data.items()
        },
        'distribution_analysis': distribution_results,
        'error_analysis': error_results,
        'importance_analysis': importance_results,
        'learning_curves': learning_curve_results,
        'comparative_analysis': comparative_results,
        'recommendations': recommendations
    }

    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "diagnostic_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
