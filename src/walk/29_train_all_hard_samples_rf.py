#!/usr/bin/env python3
"""
Train Random Forest on Complete All Hard Samples Dataset

Uses the all_hard_samples dataset (685 samples = 589 original + 96 hard samples)
with enhanced use-case-specific evaluation metrics.

Different validation sets have different operational requirements:
- rapid_response & risk_ranking: RECALL priority (minimize missed clearings)
- comprehensive: PRECISION priority (minimize false alarms)
- edge_cases: ROC-AUC (balanced testing)

Baseline: 0.583 ROC-AUC (589 samples)
Quick Win: 0.600 ROC-AUC (615 samples)
Complete edge_cases: 0.600 ROC-AUC (636 samples)
Target: 0.65-0.70 ROC-AUC (685 samples)

Usage:
    uv run python src/walk/29_train_all_hard_samples_rf.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, fbeta_score, average_precision_score

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def combine_alphaearth_features(annual_data, multiscale_data):
    """Combine pre-extracted annual magnitudes with coarse AlphaEarth landscape features."""
    X_annual = annual_data['X']
    y_annual = annual_data['y']
    annual_samples = annual_data['samples']
    multiscale_samples = multiscale_data['data']

    def get_sample_id(sample):
        return (sample['lat'], sample['lon'], sample['year'])

    annual_id_to_idx = {get_sample_id(s): i for i, s in enumerate(annual_samples)}
    multiscale_id_to_idx = {get_sample_id(s): i for i, s in enumerate(multiscale_samples)}

    common_ids = set(annual_id_to_idx.keys()) & set(multiscale_id_to_idx.keys())

    X_combined = []
    y_combined = []
    coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

    for sample_id in common_ids:
        annual_idx = annual_id_to_idx[sample_id]
        multiscale_idx = multiscale_id_to_idx[sample_id]

        annual_features = X_annual[annual_idx]
        multiscale_sample = multiscale_samples[multiscale_idx]

        if 'multiscale_features' not in multiscale_sample:
            continue

        multiscale_dict = multiscale_sample['multiscale_features']
        missing_features = [k for k in coarse_feature_names if k not in multiscale_dict]
        if missing_features:
            continue

        coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
        combined = np.concatenate([annual_features, coarse_features])

        if len(combined) != 69:
            continue

        X_combined.append(combined)
        y_combined.append(y_annual[annual_idx])

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    feature_names = ['delta_1yr', 'delta_2yr', 'acceleration'] + coarse_feature_names

    return X, y, feature_names


def print_use_case_specific_results(set_name, metrics, baseline_scores):
    """
    Print results with comprehensive metrics for dev-phase understanding.

    Shows multiple metric perspectives:
    - F-beta scores (F2 for recall-focus, F0.5 for precision-focus, F1 balanced)
    - Precision-Recall tradeoff
    - ROC-AUC
    - Confusion matrix context
    """
    roc_auc = metrics['roc_auc']
    pr_auc = metrics['pr_auc']
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    f2 = metrics['f2']
    f05 = metrics['f05']
    cm = metrics['confusion_matrix']

    # Baseline comparisons
    baseline_orig = baseline_scores['baseline_original'].get(set_name, 0.0)
    quickwin = baseline_scores['quickwin'].get(set_name, 0.0)
    edge_complete = baseline_scores['edge_complete'].get(set_name, 0.0)

    print(f"\n{set_name}:")
    print("=" * 80)

    # Show comprehensive metrics for all use cases
    print(f"\n  CLASSIFICATION METRICS:")
    print(f"    Precision:  {precision:.3f}  (true positives / predicted positives)")
    print(f"    Recall:     {recall:.3f}  (true positives / actual positives)")
    print(f"    F1-Score:   {f1:.3f}  (harmonic mean of precision & recall)")
    print(f"    F2-Score:   {f2:.3f}  (weights recall 2x more)")
    print(f"    F0.5-Score: {f05:.3f}  (weights precision 2x more)")

    print(f"\n  RANKING METRICS:")
    if not np.isnan(roc_auc):
        print(f"    ROC-AUC:    {roc_auc:.3f}  (area under ROC curve)")
    if not np.isnan(pr_auc):
        print(f"    PR-AUC:     {pr_auc:.3f}  (area under precision-recall curve)")

    print(f"\n  BASELINE COMPARISON (ROC-AUC):")
    print(f"    Baseline:       {baseline_orig:.3f}")
    print(f"    Quick Win:      {quickwin:.3f}")
    print(f"    Edge Complete:  {edge_complete:.3f}")
    if not np.isnan(roc_auc):
        print(f"    Current:        {roc_auc:.3f}  ({roc_auc - baseline_orig:+.3f} vs baseline)")

    # Use-case-specific interpretation
    print(f"\n  USE-CASE INTERPRETATION:")
    if set_name in ['rapid_response', 'risk_ranking']:
        print(f"    Priority: RECALL (minimize missed clearings)")
        print(f"    Target:   Recall ≥ 0.90")
        print(f"    Status:   {'✓ MET' if recall >= 0.90 else f'✗ NOT MET ({0.90 - recall:.3f} short)'}")
        print(f"    Best metric: F2-Score = {f2:.3f} (balances recall priority)")

    elif set_name == 'comprehensive':
        print(f"    Priority: PRECISION (minimize false alarms)")
        print(f"    Baseline precision: 0.389")
        print(f"    Status:   {'✓ IMPROVED' if precision > 0.389 else f'✗ NO IMPROVEMENT ({precision - 0.389:+.3f})'}")
        print(f"    Best metric: F0.5-Score = {f05:.3f} (balances precision priority)")

    elif set_name == 'edge_cases':
        print(f"    Priority: BALANCED (ROC-AUC)")
        print(f"    Target:   ROC-AUC ≥ 0.65")
        if not np.isnan(roc_auc):
            print(f"    Status:   {'✓ MET' if roc_auc >= 0.65 else f'✗ NOT MET ({0.65 - roc_auc:.3f} short)'}")
        print(f"    Best metric: F1-Score = {f1:.3f} (balanced)")

    # Common: Confusion matrix and class distribution
    print(f"\n  CONFUSION MATRIX:")
    print(f"    TN:  {cm[0, 0]:2d}  FP:  {cm[0, 1]:2d}  |  Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0:.3f}")
    print(f"    FN:  {cm[1, 0]:2d}  TP:  {cm[1, 1]:2d}  |  Sensitivity: {cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0:.3f}")

    y_val = metrics['y_true']
    print(f"\n  CLASS DISTRIBUTION:")
    print(f"    Clearing (1): {np.sum(y_val == 1):2d} samples  ({np.sum(y_val == 1)/len(y_val)*100:.1f}%)")
    print(f"    Intact (0):   {np.sum(y_val == 0):2d} samples  ({np.sum(y_val == 0)/len(y_val)*100:.1f}%)")
    print()


def main():
    print("=" * 80)
    print("ALL HARD SAMPLES: RANDOM FOREST WITH 685 SAMPLES")
    print("=" * 80)
    print("Dataset composition:")
    print("  589 original samples")
    print("  47 edge_cases samples (28 FN + 19 FP)")
    print("  49 new hard samples (25 rapid_response + 24 comprehensive)")
    print()
    print("Performance history:")
    print("  Baseline:           0.583 ROC-AUC (589 samples)")
    print("  Quick Win:          0.600 ROC-AUC (615 samples, +2.9%)")
    print("  Edge Complete:      0.600 ROC-AUC (636 samples, +0.0%)")
    print("  Target (All Hard):  0.65-0.70 ROC-AUC (685 samples)")
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    results_dir = Path('results/walk')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load all_hard_samples features
    print("\nLoading all_hard_samples dataset features...")

    # First check for the multiscale file (final output of script 28)
    multiscale_path = sorted(processed_dir.glob('walk_dataset_scaled_phase1_*_all_hard_samples_multiscale.pkl'))

    if not multiscale_path:
        print("❌ ERROR: No all_hard_samples_multiscale dataset found")
        print("\nPlease ensure script 28 (extract_features_all_hard_samples.py) has completed.")
        return

    multiscale_path = multiscale_path[-1]
    print(f"✓ Loading: {multiscale_path.name}")

    with open(multiscale_path, 'rb') as f:
        multiscale_data = pickle.load(f)

    # The multiscale file contains {'data': samples, 'metadata': {...}}
    samples = multiscale_data['data']
    metadata = multiscale_data.get('metadata', {})

    print(f"\n  Metadata:")
    print(f"    Total samples:        {metadata.get('n_samples', 'N/A')}")
    print(f"    With annual features: {metadata.get('n_with_annual', 'N/A')}")
    print(f"    With multiscale:      {metadata.get('n_with_multiscale', 'N/A')}")

    # Convert to format expected by combine_alphaearth_features
    # Extract annual features and build annual_data structure
    X_list = []
    y_list = []
    valid_samples = []

    for sample in samples:
        if 'annual_features' not in sample or 'multiscale_features' not in sample:
            continue

        X_list.append(sample['annual_features'])
        y_list.append(sample.get('label', 0))
        valid_samples.append(sample)

    if not X_list:
        print("❌ ERROR: No samples with both annual and multiscale features")
        return

    X_annual = np.array(X_list)
    y_annual = np.array(y_list)

    annual_data = {
        'X': X_annual,
        'y': y_annual,
        'samples': valid_samples
    }

    # Build multiscale_data structure (already have it, but need to match format)
    multiscale_data_formatted = {
        'data': valid_samples
    }

    X_train, y_train, feature_names = combine_alphaearth_features(annual_data, multiscale_data_formatted)

    print(f"\n✓ All hard samples training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Clearing: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Intact: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  New samples since baseline: {len(X_train) - 589}")

    # Scale features
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING VIA 5-FOLD STRATIFIED CV")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define hyperparameter grid (same as baseline)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    print(f"\nSearching {np.prod([len(v) for v in param_grid.values()])} hyperparameter combinations...")
    print(f"Using StratifiedKFold with 5 folds (preserves class balance)")

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n✓ Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    print(f"\nBest hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Check for overfitting
    best_idx = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_idx]
    val_score = grid_search.cv_results_['mean_test_score'][best_idx]

    print(f"\nOverfitting check:")
    print(f"  Mean train score: {train_score:.3f}")
    print(f"  Mean val score:   {val_score:.3f}")
    print(f"  Gap:              {train_score - val_score:.3f}")

    if train_score - val_score > 0.1:
        print(f"  ⚠ Warning: Potential overfitting detected (gap > 0.1)")
    else:
        print(f"  ✓ Good generalization (gap < 0.1)")

    # Train final model on all data with best hyperparameters
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL MODEL ON ALL {len(X_train)} SAMPLES")
    print(f"{'='*80}\n")

    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train_scaled, y_train)
    print("✓ Final model trained")

    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}\n")

    feature_importance = best_rf.feature_importances_
    importance_idx = np.argsort(feature_importance)[::-1]

    print("Top 20 most important features:")
    for i, idx in enumerate(importance_idx[:20], 1):
        print(f"{i:3d}. {feature_names[idx]:30s} {feature_importance[idx]:.4f}")

    # Load validation sets
    print(f"\n{'='*80}")
    print("EVALUATING ON HELD-OUT VALIDATION SETS")
    print("=" * 80)
    print("\nUse-Case-Specific Evaluation Criteria:")
    print("  rapid_response:  PRIMARY = Recall ≥ 0.90 (minimize missed fire clearings)")
    print("  risk_ranking:    PRIMARY = Recall ≥ 0.90 (minimize missed high-risk areas)")
    print("  comprehensive:   PRIMARY = Precision improvement (minimize false alarms)")
    print("  edge_cases:      PRIMARY = ROC-AUC ≥ 0.65 (balanced performance)")
    print()

    val_sets = ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']
    val_data = {}

    ee_client = EarthEngineClient(use_cache=True)

    for set_name in val_sets:
        multiscale_val_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'

        if not multiscale_val_path.exists():
            continue

        print(f"Loading {set_name}...")

        with open(multiscale_val_path, 'rb') as f:
            val_samples = pickle.load(f)

        X_val = []
        y_val = []

        for sample in val_samples:
            # Fix missing 'year' field for intact samples
            if 'year' not in sample and sample.get('stable', False):
                sample = sample.copy()
                sample['year'] = 2021

            # Extract annual features
            try:
                from src.walk.diagnostic_helpers import extract_dual_year_features
                annual_features = extract_dual_year_features(ee_client, sample)
            except:
                annual_features = None

            if annual_features is None:
                continue

            # Get coarse features
            if 'multiscale_features' not in sample:
                continue

            multiscale_dict = sample['multiscale_features']
            coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

            missing_features = [k for k in coarse_feature_names if k not in multiscale_dict]
            if missing_features:
                continue

            coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
            combined = np.concatenate([annual_features, coarse_features])

            if len(combined) != 69:
                continue

            X_val.append(combined)
            y_val.append(sample.get('label', 0))

        if len(X_val) == 0:
            print(f"  ⚠ No valid features, skipping")
            continue

        X_val = np.vstack(X_val)
        y_val = np.array(y_val)

        print(f"  ✓ {len(X_val)} samples")

        val_data[set_name] = {'X': X_val, 'y': y_val}

    # Baseline scores for comparison
    baseline_scores = {
        'baseline_original': {
            'risk_ranking': 0.950,
            'rapid_response': 0.778,
            'comprehensive': 0.711,
            'edge_cases': 0.583
        },
        'quickwin': {
            'risk_ranking': 0.911,
            'rapid_response': 0.757,
            'comprehensive': 0.702,
            'edge_cases': 0.600
        },
        'edge_complete': {
            'risk_ranking': 0.907,
            'rapid_response': 0.700,
            'comprehensive': 0.700,
            'edge_cases': 0.600
        }
    }

    results = {}

    print(f"\n{'='*80}")
    print("VALIDATION SET RESULTS (USE-CASE-SPECIFIC)")
    print(f"{'='*80}")

    for set_name, data in val_data.items():
        X_val_scaled = scaler.transform(data['X'])
        y_val = data['y']

        y_pred_proba = best_rf.predict_proba(X_val_scaled)[:, 1]
        y_pred = best_rf.predict(X_val_scaled)

        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            roc_auc = float('nan')

        try:
            pr_auc = average_precision_score(y_val, y_pred_proba)
        except ValueError:
            pr_auc = float('nan')

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)  # Recall-focused
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)  # Precision-focused
        cm = confusion_matrix(y_val, y_pred)

        baseline_orig = baseline_scores['baseline_original'].get(set_name, 0.0)
        quickwin = baseline_scores['quickwin'].get(set_name, 0.0)
        edge_complete = baseline_scores['edge_complete'].get(set_name, 0.0)

        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'f05': f05,
            'confusion_matrix': cm,
            'y_true': y_val,
            'baseline_original': baseline_orig,
            'quickwin': quickwin,
            'edge_complete': edge_complete,
            'cv_score': grid_search.best_score_
        }

        results[set_name] = metrics

        # Print use-case-specific results
        print_use_case_specific_results(set_name, metrics, baseline_scores)

    # Summary table
    print(f"{'='*80}")
    print("RESULTS SUMMARY TABLE")
    print(f"{'='*80}\n")

    print(f"{'Validation Set':<20s} {'Baseline':>10s} {'QuickWin':>10s} {'EdgeCplt':>10s} {'AllHard':>10s} {'vs Base':>10s}")
    print("-" * 88)
    for set_name in val_sets:
        if set_name in results:
            r = results[set_name]
            if not np.isnan(r['roc_auc']):
                improvement = r['roc_auc'] - r['baseline_original']
                print(f"{set_name:<20s} {r['baseline_original']:>10.3f} {r['quickwin']:>10.3f} "
                      f"{r['edge_complete']:>10.3f} {r['roc_auc']:>10.3f} {improvement:>+10.3f}")
            else:
                print(f"{set_name:<20s} {r['baseline_original']:>10.3f} {r['quickwin']:>10.3f} "
                      f"{r['edge_complete']:>10.3f}        nan           nan")

    # Use-case-specific success criteria summary
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA SUMMARY")
    print(f"{'='*80}\n")

    success_count = 0
    total_criteria = 0

    print("Use-Case-Specific Targets:")
    print()

    # rapid_response: Recall ≥ 0.90
    if 'rapid_response' in results:
        recall = results['rapid_response']['recall']
        print(f"  rapid_response (Recall ≥ 0.90):")
        print(f"    Current:  {recall:.3f}")
        if recall >= 0.90:
            print(f"    Status:   ✓ MET")
            success_count += 1
        else:
            print(f"    Status:   ✗ NOT MET ({0.90 - recall:.3f} below target)")
        total_criteria += 1
        print()

    # risk_ranking: Recall ≥ 0.90
    if 'risk_ranking' in results:
        recall = results['risk_ranking']['recall']
        print(f"  risk_ranking (Recall ≥ 0.90):")
        print(f"    Current:  {recall:.3f}")
        if recall >= 0.90:
            print(f"    Status:   ✓ MET")
            success_count += 1
        else:
            print(f"    Status:   ✗ NOT MET ({0.90 - recall:.3f} below target)")
        total_criteria += 1
        print()

    # comprehensive: Precision improvement
    if 'comprehensive' in results:
        precision = results['comprehensive']['precision']
        baseline_precision = 0.389  # From baseline
        improvement = precision - baseline_precision
        print(f"  comprehensive (Precision improvement):")
        print(f"    Current:    {precision:.3f}")
        print(f"    Baseline:   {baseline_precision:.3f}")
        print(f"    Change:     {improvement:+.3f}")
        if improvement > 0:
            print(f"    Status:     ✓ IMPROVED")
            success_count += 1
        else:
            print(f"    Status:     ✗ NO IMPROVEMENT")
        total_criteria += 1
        print()

    # edge_cases: ROC-AUC ≥ 0.65
    if 'edge_cases' in results:
        roc_auc = results['edge_cases']['roc_auc']
        if not np.isnan(roc_auc):
            print(f"  edge_cases (ROC-AUC ≥ 0.65):")
            print(f"    Current:    {roc_auc:.3f}")
            print(f"    Target:     0.65")
            print(f"    Baseline:   0.583")
            if roc_auc >= 0.65:
                print(f"    Status:     ✓ TARGET MET")
                success_count += 1
            else:
                gap = 0.65 - roc_auc
                print(f"    Status:     ✗ NOT MET ({gap:.3f} below target)")
            total_criteria += 1
            print()

    print(f"{'='*80}")
    print(f"OVERALL: {success_count}/{total_criteria} use-case criteria met")
    print(f"{'='*80}\n")

    if success_count == total_criteria:
        print("✓ ALL USE-CASE REQUIREMENTS MET!")
        print("  Active learning approach successfully validated across all use cases.")
    else:
        print(f"✗ {total_criteria - success_count} use-case requirement(s) not met")
        print("\nNext steps:")
        print("  - Review failure modes for unmet criteria")
        print("  - Consider collecting more targeted samples for specific use cases")
        print("  - Explore alternative models or feature engineering")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    # Convert numpy types to Python types for JSON serialization
    results_json = {}
    for set_name, metrics in results.items():
        results_json[set_name] = {
            'roc_auc': float(metrics['roc_auc']) if not np.isnan(metrics['roc_auc']) else None,
            'pr_auc': float(metrics['pr_auc']) if not np.isnan(metrics['pr_auc']) else None,
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'f2': float(metrics['f2']),
            'f05': float(metrics['f05']),
            'confusion_matrix': [[int(x) for x in row] for row in metrics['confusion_matrix']],
            'baseline_original': float(metrics['baseline_original']),
            'quickwin': float(metrics['quickwin']),
            'edge_complete': float(metrics['edge_complete']),
            'cv_score': float(metrics['cv_score'])
        }

    results_file = results_dir / 'all_hard_samples_rf_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    model_file = processed_dir / 'walk_model_rf_all_hard_samples.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': best_rf,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'metadata': {
                'n_train_samples': len(X_train),
                'n_features': X_train.shape[1],
                'model_type': 'RandomForest',
                'cv_strategy': 'StratifiedKFold_5',
                'dataset': 'all_hard_samples_685_samples',
                'dataset_composition': {
                    'original': 589,
                    'edge_cases': 47,
                    'rapid_response': 25,
                    'comprehensive': 24
                }
            }
        }, f)
    print(f"✓ Model saved to: {model_file}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
