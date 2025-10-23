"""
Train XGBoost with L2 Regularization

Uses 201D fine-scale features (3D annual + 66D coarse + 132D fine)
with aggressive regularization to prevent overfitting on high-dimensional data.

Strategy:
1. Hyperparameter tuning via GridSearchCV with StratifiedKFold
2. L2 (reg_lambda) and L1 (reg_alpha) regularization
3. Conservative tree depth and child weight
4. Train final model on all 589 samples with best hyperparameters
5. Evaluate ONCE on 4 held-out validation sets (no peeking during tuning)

Usage:
    uv run python src/walk/13_train_xgboost.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def combine_fine_scale_features(annual_data, fine_scale_data):
    """
    Combine pre-extracted annual magnitudes with fine-scale AlphaEarth features.

    Feature breakdown:
    - Annual: 3D (delta_1yr, delta_2yr, acceleration)
    - Coarse (100m): 66D (64 embeddings + heterogeneity + range)
    - Fine 10m: 66D (64 embeddings + heterogeneity + range)
    - Fine 20m: 66D (64 embeddings + heterogeneity + range)
    Total: 201D
    """
    X_annual = annual_data['X']
    y_annual = annual_data['y']
    annual_samples = annual_data['samples']
    fine_scale_samples = fine_scale_data['data']

    def get_sample_id(sample):
        return (sample['lat'], sample['lon'], sample['year'])

    annual_id_to_idx = {get_sample_id(s): i for i, s in enumerate(annual_samples)}
    fine_scale_id_to_idx = {get_sample_id(s): i for i, s in enumerate(fine_scale_samples)}

    common_ids = set(annual_id_to_idx.keys()) & set(fine_scale_id_to_idx.keys())

    X_combined = []
    y_combined = []

    # Define expected features
    coarse_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
    fine_10m_names = [f'fine_10m_emb_{i}' for i in range(64)] + ['fine_10m_heterogeneity', 'fine_10m_range']
    fine_20m_names = [f'fine_20m_emb_{i}' for i in range(64)] + ['fine_20m_heterogeneity', 'fine_20m_range']

    all_feature_names = coarse_names + fine_10m_names + fine_20m_names

    for sample_id in common_ids:
        annual_idx = annual_id_to_idx[sample_id]
        fine_scale_idx = fine_scale_id_to_idx[sample_id]

        annual_features = X_annual[annual_idx]
        fine_scale_sample = fine_scale_samples[fine_scale_idx]

        if 'multiscale_features' not in fine_scale_sample:
            continue

        multiscale_dict = fine_scale_sample['multiscale_features']

        # Check for missing features
        missing_features = [k for k in all_feature_names if k not in multiscale_dict]
        if missing_features:
            continue

        # Extract all multiscale features in order
        multiscale_features = np.array([multiscale_dict[k] for k in all_feature_names])
        combined = np.concatenate([annual_features, multiscale_features])

        if len(combined) != 201:
            continue

        X_combined.append(combined)
        y_combined.append(y_annual[annual_idx])

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    feature_names = ['delta_1yr', 'delta_2yr', 'acceleration'] + all_feature_names

    return X, y, feature_names


def main():
    print("=" * 80)
    print("XGBOOST WITH L2 REGULARIZATION (201D FEATURES)")
    print("=" * 80)

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    results_dir = Path('results/walk')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("\nLoading features...")
    annual_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'
    with open(annual_path, 'rb') as f:
        annual_data = pickle.load(f)

    fine_scale_path = processed_dir / 'walk_dataset_scaled_phase1_fine_scale.pkl'

    if not fine_scale_path.exists():
        print(f"\n✗ Fine-scale features not found: {fine_scale_path}")
        print(f"  Run extraction first:")
        print(f"  uv run python src/walk/12_extract_fine_scale_features.py --dataset training")
        return

    with open(fine_scale_path, 'rb') as f:
        fine_scale_data = pickle.load(f)

    X_train, y_train, feature_names = combine_fine_scale_features(annual_data, fine_scale_data)

    print(f"\n✓ Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Clearing: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Intact: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  Samples/feature ratio: {len(X_train)/X_train.shape[1]:.1f}")

    # Scale features
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING VIA 5-FOLD STRATIFIED CV")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define hyperparameter grid with AGGRESSIVE regularization
    param_grid = {
        'max_depth': [3, 5, 7],                     # Shallow trees to prevent overfitting
        'learning_rate': [0.01, 0.05, 0.1],         # Lower learning rate
        'n_estimators': [100, 200, 300],
        'reg_lambda': [1.0, 5.0, 10.0],             # L2 regularization (KEY!)
        'reg_alpha': [0.0, 0.5, 1.0],               # L1 regularization
        'subsample': [0.7, 0.8],                    # Row sampling
        'colsample_bytree': [0.7, 0.8],             # Column sampling per tree
        'gamma': [0, 0.1, 0.5],                     # Min loss reduction for split
        'min_child_weight': [3, 5, 10],             # Min sum of weights in child
        'scale_pos_weight': [1.0],                  # Class balance (1:1 ratio)
    }

    print(f"\nSearching {np.prod([len(v) for v in param_grid.values()])} hyperparameter combinations...")
    print(f"Using StratifiedKFold with 5 folds (preserves 50/50 class balance)")
    print(f"\nRegularization settings:")
    print(f"  L2 (reg_lambda): {param_grid['reg_lambda']}")
    print(f"  L1 (reg_alpha): {param_grid['reg_alpha']}")
    print(f"  Max depth: {param_grid['max_depth']}")
    print(f"  Min child weight: {param_grid['min_child_weight']}")

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    xgb = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    grid_search = GridSearchCV(
        xgb,
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
        print(f"     Consider increasing regularization or using PCA")
    else:
        print(f"  ✓ Good generalization (gap < 0.1)")

    # Train final model on all data with best hyperparameters
    print(f"\n{'='*80}")
    print("TRAINING FINAL MODEL ON ALL 589 SAMPLES")
    print(f"{'='*80}\n")

    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X_train_scaled, y_train)
    print("✓ Final model trained")

    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}\n")

    feature_importance = best_xgb.feature_importances_
    importance_idx = np.argsort(feature_importance)[::-1]

    print("Top 20 most important features:")
    for i, idx in enumerate(importance_idx[:20], 1):
        print(f"{i:3d}. {feature_names[idx]:30s} {feature_importance[idx]:.4f}")

    # Load validation sets
    print(f"\n{'='*80}")
    print("EVALUATING ON HELD-OUT VALIDATION SETS")
    print(f"{'='*80}\n")

    val_sets = ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']
    val_data = {}

    ee_client = EarthEngineClient(use_cache=True)

    for set_name in val_sets:
        fine_scale_val_path = processed_dir / f'hard_val_{set_name}_fine_scale.pkl'

        if not fine_scale_val_path.exists():
            print(f"⚠ {set_name}: fine-scale features not found, skipping")
            continue

        print(f"Loading {set_name}...")

        with open(fine_scale_val_path, 'rb') as f:
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

            # Get fine-scale features
            if 'multiscale_features' not in sample:
                continue

            multiscale_dict = sample['multiscale_features']

            # Define expected features
            coarse_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
            fine_10m_names = [f'fine_10m_emb_{i}' for i in range(64)] + ['fine_10m_heterogeneity', 'fine_10m_range']
            fine_20m_names = [f'fine_20m_emb_{i}' for i in range(64)] + ['fine_20m_heterogeneity', 'fine_20m_range']
            all_feature_names = coarse_names + fine_10m_names + fine_20m_names

            missing_features = [k for k in all_feature_names if k not in multiscale_dict]
            if missing_features:
                continue

            multiscale_features = np.array([multiscale_dict[k] for k in all_feature_names])
            combined = np.concatenate([annual_features, multiscale_features])

            if len(combined) != 201:
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

    # Evaluate
    baseline_rf = {
        'risk_ranking': 0.907,
        'rapid_response': 0.760,
        'comprehensive': 0.713,
        'edge_cases': 0.583
    }

    results = {}

    print(f"\n{'='*80}")
    print("VALIDATION SET RESULTS")
    print(f"{'='*80}\n")

    for set_name, data in val_data.items():
        X_val_scaled = scaler.transform(data['X'])
        y_val = data['y']

        y_pred_proba = best_xgb.predict_proba(X_val_scaled)[:, 1]
        y_pred = best_xgb.predict(X_val_scaled)

        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            roc_auc = float('nan')

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred)

        baseline = baseline_rf.get(set_name, 0.0)
        diff = roc_auc - baseline if not np.isnan(roc_auc) else float('nan')
        pct_change = (diff / baseline * 100) if baseline > 0 and not np.isnan(diff) else float('nan')

        results[set_name] = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'baseline_rf': baseline,
            'improvement_vs_rf': diff,
            'pct_change_vs_rf': pct_change,
            'cv_score': grid_search.best_score_
        }

        print(f"{set_name}:")
        print("=" * 60)
        if not np.isnan(roc_auc):
            print(f"  ROC-AUC:   {roc_auc:.3f}  (RF: {baseline:.3f}, {diff:+.3f} / {pct_change:+.1f}%)")
        else:
            print(f"  ROC-AUC:   nan  (RF: {baseline:.3f})")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN:  {cm[0, 0]:2d}  FP:  {cm[0, 1]:2d}")
        print(f"    FN:  {cm[1, 0]:2d}  TP:  {cm[1, 1]:2d}")
        print(f"\n  Class Distribution:")
        print(f"    Clearing (1): {np.sum(y_val == 1)} samples")
        print(f"    Intact (0):   {np.sum(y_val == 0)} samples")
        print()

    # Summary
    print(f"{'='*80}")
    print("XGBOOST RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Validation Set':<20s} {'RandomForest':>13s} {'XGBoost':>10s} {'Improvement':>12s}")
    print("-" * 80)
    for set_name in val_sets:
        if set_name in results:
            r = results[set_name]
            if not np.isnan(r['roc_auc']):
                print(f"{set_name:<20s} {r['baseline_rf']:>13.3f} {r['roc_auc']:>10.3f} "
                      f"{r['improvement_vs_rf']:>+12.3f} ({r['pct_change_vs_rf']:>+5.1f}%)")
            else:
                print(f"{set_name:<20s} {r['baseline_rf']:>13.3f}        nan              nan")

    # Success criteria
    print(f"\n{'='*80}")
    print("SUCCESS CRITERIA")
    print(f"{'='*80}\n")

    edge_roc = results.get('edge_cases', {}).get('roc_auc', 0.0)
    target = 0.70

    print(f"Target: edge_cases ROC-AUC ≥ {target:.2f}")
    print(f"Random Forest (69D): 0.583")

    if not np.isnan(edge_roc):
        print(f"XGBoost (201D): {edge_roc:.3f}")
        print(f"  5-Fold CV score: {grid_search.best_score_:.3f}")
        print(f"  Train-val gap: {train_score - val_score:.3f}")

        if edge_roc >= target:
            print(f"\n✓ TARGET MET: edge_cases ROC-AUC = {edge_roc:.3f} ≥ {target:.2f}")
            print(f"  XGBoost with fine-scale features successfully improved performance!")
        else:
            gap = target - edge_roc
            print(f"\n✗ TARGET NOT MET: {gap:.3f} below target")
            improvement = edge_roc - 0.583
            if improvement > 0:
                print(f"  ✓ XGBoost improved by {improvement:+.3f} over Random Forest")
                print(f"  Next step: Try PCA compression (201D → 94D)")
            else:
                print(f"  ✗ XGBoost did not improve over Random Forest")
                print(f"  Next step: Apply PCA compression or move to Path A specialized models")
    else:
        print(f"XGBoost (201D): nan")
        print(f"\n✗ Could not evaluate edge_cases")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    results_file = results_dir / 'xgboost_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    model_file = processed_dir / 'walk_model_xgboost.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': best_xgb,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'metadata': {
                'n_train_samples': len(X_train),
                'n_features': X_train.shape[1],
                'model_type': 'XGBoost',
                'cv_strategy': 'StratifiedKFold_5',
                'regularization': {
                    'reg_lambda': grid_search.best_params_['reg_lambda'],
                    'reg_alpha': grid_search.best_params_['reg_alpha']
                }
            }
        }, f)
    print(f"✓ Model saved to: {model_file}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
