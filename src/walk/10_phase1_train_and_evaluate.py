"""
Phase 1: Train and Evaluate Scaled Model

Trains logistic regression model on Phase 1 scaled dataset (600 samples)
and evaluates on all 4 validation sets.

Compares to baseline performance:
- risk_ranking: 0.850 ROC-AUC (46 samples)
- rapid_response: 0.824 ROC-AUC (27 samples)
- comprehensive: 0.758 ROC-AUC (69 samples)
- edge_cases: 0.583 ROC-AUC (23 samples)

Target: edge_cases 0.583 → 0.70+ ROC-AUC

Usage:
    uv run python src/walk/10_phase1_train_and_evaluate.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

from src.utils import get_config


# Feature names for interpretation (CORRECTED to annual features)
FEATURE_NAMES = [
    "delta_1yr",      # Recent annual change (Y to Y-1)
    "delta_2yr",      # Historical annual change (Y-1 to Y-2)
    "acceleration"    # Change in change rate
]


def load_validation_features(val_set_name: str, processed_dir: Path):
    """
    Load pre-extracted validation set features.

    Assumes features were extracted by diagnostic analysis script.
    """
    # Check if features exist from diagnostic analysis
    # If not, we'll need to extract them

    # For now, return None - will be loaded from validation set .pkl files
    # and features extracted on-the-fly if needed
    return None


def extract_val_features_if_needed(client, val_samples, set_name):
    """Extract features for validation set if not already cached."""
    from src.walk.diagnostic_helpers import extract_dual_year_features

    print(f"\nExtracting features for {set_name}...")

    X_list = []
    y_list = []
    valid_samples = []

    for sample in val_samples:
        features = extract_dual_year_features(client, sample)
        if features is not None:
            X_list.append(features)

            # Handle labels
            label = sample.get('label')
            if label is None:
                # Use stable field: True=intact (0), False=clearing (1)
                label = 0 if sample.get('stable', False) else 1
            elif isinstance(label, str):
                label = 1 if label == 'clearing' else 0
            else:
                label = int(label)

            y_list.append(label)
            valid_samples.append(sample)

    print(f"  Successfully extracted: {len(X_list)}/{len(val_samples)} samples")

    return np.array(X_list), np.array(y_list), valid_samples


def evaluate_model(model, scaler, X, y, set_name, baseline_auc=None):
    """Evaluate model on a validation set and print detailed metrics."""

    print(f"\n{set_name}:")
    print("=" * 60)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    auc = roc_auc_score(y, y_proba)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Print metrics
    print(f"  ROC-AUC:   {auc:.3f}", end="")
    if baseline_auc is not None:
        improvement = auc - baseline_auc
        pct_change = improvement / baseline_auc * 100
        sign = "+" if improvement >= 0 else ""
        print(f"  (baseline: {baseline_auc:.3f}, {sign}{improvement:.3f} / {sign}{pct_change:.1f}%)")
    else:
        print()

    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")

    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn:3d}  FP: {fp:3d}")
    print(f"    FN: {fn:3d}  TP: {tp:3d}")

    # Class-wise breakdown
    n_clearing = np.sum(y == 1)
    n_intact = np.sum(y == 0)

    print(f"\n  Class Distribution:")
    print(f"    Clearing (1): {n_clearing} samples")
    print(f"    Intact (0):   {n_intact} samples")

    # Detection rate for clearings
    clearing_detected = tp
    clearing_total = tp + fn
    clearing_detection_rate = clearing_detected / clearing_total if clearing_total > 0 else 0

    print(f"\n  Clearing Detection:")
    print(f"    Detected: {clearing_detected}/{clearing_total} ({clearing_detection_rate*100:.1f}%)")

    return {
        'roc_auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'n_clearing': int(n_clearing),
        'n_intact': int(n_intact),
        'baseline_auc': baseline_auc,
        'improvement': float(auc - baseline_auc) if baseline_auc else None,
        'improvement_pct': float((auc - baseline_auc) / baseline_auc * 100) if baseline_auc else None
    }


def main():
    """Train and evaluate Phase 1 scaled model."""

    print("=" * 80)
    print("PHASE 1: TRAIN AND EVALUATE SCALED MODEL")
    print("=" * 80)

    # Initialize
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load Phase 1 features
    features_path = processed_dir / 'walk_dataset_scaled_phase1_features.pkl'

    print(f"\nLoading Phase 1 features from: {features_path}")

    with open(features_path, 'rb') as f:
        feature_data = pickle.load(f)

    train_X = feature_data['X']
    train_y = feature_data['y']

    print(f"  Loaded {len(train_X)} samples")
    print(f"  Feature dimension: {train_X.shape[1]}")
    print(f"  Clearing: {np.sum(train_y == 1)}")
    print(f"  Intact: {np.sum(train_y == 0)}")

    # Train model
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    print("\nScaling features...")
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    print("Training logistic regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(train_X_scaled, train_y)

    print("  ✓ Model trained")

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance = np.abs(model.coef_[0])
    top_indices = np.argsort(importance)[-10:][::-1]

    print("\nTop 10 most important features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:25s} {importance[idx]:.4f}")

    # Load validation sets
    print("\n" + "=" * 80)
    print("LOADING VALIDATION SETS")
    print("=" * 80)

    val_paths = {
        'risk_ranking': processed_dir / 'hard_val_risk_ranking.pkl',
        'rapid_response': processed_dir / 'hard_val_rapid_response.pkl',
        'comprehensive': processed_dir / 'hard_val_comprehensive.pkl',
        'edge_cases': processed_dir / 'hard_val_edge_cases.pkl'
    }

    baseline_aucs = {
        'risk_ranking': 0.850,
        'rapid_response': 0.824,
        'comprehensive': 0.758,
        'edge_cases': 0.583
    }

    val_data = {}

    # We need to extract features for validation sets
    # Import the extraction function from diagnostic script
    from src.utils import EarthEngineClient
    from src.walk.diagnostic_helpers import extract_dual_year_features

    client = EarthEngineClient(use_cache=True)

    for name, path in val_paths.items():
        print(f"\nLoading {name}...")
        with open(path, 'rb') as f:
            samples = pickle.load(f)

        # Add labels if missing
        for sample in samples:
            if 'label' not in sample:
                sample['label'] = 0 if sample.get('stable', False) else 1
            if 'year' not in sample:
                year_str = sample.get('date', '2021-06-01').split('-')[0]
                sample['year'] = int(year_str)

        print(f"  Loaded {len(samples)} samples")

        # Extract features
        X_list = []
        y_list = []
        valid_samples = []

        for sample in samples:
            features = extract_dual_year_features(client, sample)
            if features is not None:
                X_list.append(features)

                label = sample.get('label')
                if isinstance(label, str):
                    y_list.append(1 if label == 'clearing' else 0)
                else:
                    y_list.append(int(label))

                valid_samples.append(sample)

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Extracted features: {len(X)}/{len(samples)} samples")

        val_data[name] = {
            'X': X,
            'y': y,
            'samples': valid_samples,
            'baseline_auc': baseline_aucs[name]
        }

    # Evaluate on all validation sets
    print("\n" + "=" * 80)
    print("EVALUATION ON VALIDATION SETS")
    print("=" * 80)

    results = {}

    for name, data in val_data.items():
        result = evaluate_model(
            model, scaler,
            data['X'], data['y'],
            name,
            baseline_auc=data['baseline_auc']
        )
        results[name] = result

    # Summary comparison
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 80)

    print("\n{:<20s} {:>10s} {:>10s} {:>12s} {:>10s}".format(
        "Validation Set", "Baseline", "Phase 1", "Change", "% Change"
    ))
    print("-" * 80)

    for name in ['risk_ranking', 'rapid_response', 'comprehensive', 'edge_cases']:
        if name in results:
            r = results[name]
            baseline = r['baseline_auc']
            phase1 = r['roc_auc']
            change = r['improvement']
            pct = r['improvement_pct']

            sign = "+" if change >= 0 else ""

            print("{:<20s} {:>10.3f} {:>10.3f} {:>11s} {:>9.1f}%".format(
                name, baseline, phase1, f"{sign}{change:.3f}", pct
            ))

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)

    edge_case_auc = results['edge_cases']['roc_auc']
    edge_case_target = 0.70

    print(f"\nTarget: edge_cases ROC-AUC ≥ {edge_case_target:.2f}")
    print(f"Actual: edge_cases ROC-AUC = {edge_case_auc:.3f}")

    if edge_case_auc >= edge_case_target:
        print("\n✓ SUCCESS: Phase 1 target achieved!")
        print("  Edge case performance improved to acceptable level")
        recommendation = "DEPLOY_PHASE1"
    else:
        gap = edge_case_target - edge_case_auc
        print(f"\n✗ TARGET NOT MET: {gap:.3f} below target")

        # Check if progress was made
        edge_improvement = results['edge_cases']['improvement']
        if edge_improvement > 0.05:
            print(f"  However, significant improvement ({edge_improvement:+.3f}) was achieved")
            recommendation = "PHASE_1B_MORE_DATA"
        else:
            print(f"  Limited improvement ({edge_improvement:+.3f}) suggests need for specialization")
            recommendation = "PHASE_2_SPECIALIZATION"

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if recommendation == "DEPLOY_PHASE1":
        print("\nDeploy Phase 1 model:")
        print("  - All validation sets show acceptable performance")
        print("  - Edge cases target achieved")
        print("  - Single model appropriate for all scenarios")

    elif recommendation == "PHASE_1B_MORE_DATA":
        print("\nProceed with Phase 1B: Collect more diverse data:")
        print("  - Good progress on edge cases (+{:.3f})".format(edge_improvement))
        print("  - Target: 800-1000 samples with emphasis on edge cases")
        print("  - Collect 50% edge cases (small, fire, fragmented)")
        print("  - Expected: edge_cases 0.70+ ROC-AUC")

    elif recommendation == "PHASE_2_SPECIALIZATION":
        print("\nProceed with Phase 2: Build specialized models:")
        print("  - Limited improvement suggests fundamental differences")
        print("  - Build 2 models: standard + edge case")
        print("  - Simple routing: size < 1ha OR fire_prone → edge model")
        print("  - Data needs: 300-400 samples per model")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase1_scaled',
        'training': {
            'n_samples': len(train_X),
            'n_clearing': int(np.sum(train_y == 1)),
            'n_intact': int(np.sum(train_y == 0)),
            'feature_dim': train_X.shape[1]
        },
        'feature_importance': {
            'top_10': [
                {
                    'feature': FEATURE_NAMES[idx],
                    'importance': float(importance[idx])
                }
                for idx in top_indices
            ]
        },
        'validation_results': results,
        'recommendation': recommendation,
        'success_criteria': {
            'target': edge_case_target,
            'achieved': edge_case_auc >= edge_case_target,
            'edge_cases_auc': edge_case_auc,
            'gap': float(edge_case_target - edge_case_auc) if edge_case_auc < edge_case_target else 0
        }
    }

    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "phase1_evaluation.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Save model
    model_path = processed_dir / 'walk_model_phase1.pkl'

    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'metadata': {
            'created': datetime.now().isoformat(),
            'phase': 'phase1_scaled',
            'n_training_samples': len(train_X),
            'validation_performance': {
                name: r['roc_auc']
                for name, r in results.items()
            }
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to: {model_path}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
