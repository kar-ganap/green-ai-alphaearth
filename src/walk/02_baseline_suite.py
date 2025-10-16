"""
WALK Phase - Step 2: Baseline Suite

Establishes performance benchmarks that engineered models must beat.

Baselines tested:
1. Random: 50% AUC (sanity check)
2. Raw embeddings: Simple L2 distance from baseline
3. Simple engineered: Basic temporal features (distances, velocities)
4. Context-only: (Future) Roads + proximity features without embeddings

Each baseline provides insights:
- Random: Confirms dataset is reasonable
- Raw embeddings: Tests if raw signal has predictive power
- Simple engineered: Tests if basic feature engineering helps
- Context-only: Tests how much context contributes

Output:
    results/walk/baseline_results.json
    results/figures/walk/baseline_comparison.png

Usage:
    uv run python src/walk/02_baseline_suite.py
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.utils import get_config, save_figure


def load_dataset(dataset_path):
    """
    Load prepared dataset.

    Args:
        dataset_path: Path to walk_dataset.pkl

    Returns:
        Dict with dataset
    """
    print(f"Loading dataset from {dataset_path}...")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"  ✓ Loaded {dataset['metadata']['n_samples']} samples")
    print(f"    Clearing: {dataset['metadata']['n_clearing']}")
    print(f"    Intact:   {dataset['metadata']['n_intact']}")
    print(f"    Train:    {len(dataset['splits']['train'])}")
    print(f"    Val:      {len(dataset['splits']['val'])}")
    print(f"    Test:     {len(dataset['splits']['test'])}\n")

    return dataset


def extract_features(dataset, feature_type='simple'):
    """
    Extract features from dataset for baseline models.

    Args:
        dataset: Prepared dataset dict
        feature_type: Type of features to extract
            - 'raw': Raw Q4 distance from Q1 baseline only
            - 'simple': Distances, velocities, accelerations
            - 'all': All available features

    Returns:
        Tuple of (X, y) where X is feature matrix, y is labels
    """
    data = dataset['data']

    X = []
    y = []

    for sample in data:
        features_dict = sample['features']
        label = sample['label']

        if feature_type == 'raw':
            # Just Q4 distance (simplest possible feature)
            features = [features_dict['distances']['Q4']]

        elif feature_type == 'simple':
            # Distances + velocities
            dist = features_dict['distances']
            vel = features_dict['velocities']

            features = [
                dist['Q1'],
                dist['Q2'],
                dist['Q3'],
                dist['Q4'],
                dist['Clearing'],
                vel['Q1_Q2'],
                vel['Q2_Q3'],
                vel['Q3_Q4'],
                vel['Q4_Clearing'],
            ]

        elif feature_type == 'all':
            # All temporal features
            dist = features_dict['distances']
            vel = features_dict['velocities']
            acc = features_dict['accelerations']
            trend = features_dict['trend_consistency']

            features = [
                # Distances
                dist['Q1'],
                dist['Q2'],
                dist['Q3'],
                dist['Q4'],
                dist['Clearing'],
                # Velocities
                vel['Q1_Q2'],
                vel['Q2_Q3'],
                vel['Q3_Q4'],
                vel['Q4_Clearing'],
                # Accelerations
                acc['Q1_Q2_Q3'],
                acc['Q2_Q3_Q4'],
                acc['Q3_Q4_Clearing'],
                # Trend
                trend,
            ]

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred_proba):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dict with metrics
    """
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except:
        roc_auc = 0.5  # Fallback for degenerate cases

    # Precision-Recall AUC (better for imbalanced data)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
    except:
        pr_auc = 0.5

    # Threshold-based metrics at 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'precision': float(ppv),
        'recall': float(tpr),
        'specificity': float(tnr),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }
    }


def baseline_random(y_train, y_test):
    """
    Baseline 1: Random predictions.

    Should give ~0.5 AUC. Sanity check.

    Args:
        y_train: Train labels (not used)
        y_test: Test labels

    Returns:
        Random predictions
    """
    np.random.seed(42)
    y_pred_proba = np.random.rand(len(y_test))
    return y_pred_proba


def baseline_raw_embeddings(X_train, y_train, X_test, y_test):
    """
    Baseline 2: Raw Q4 distance only.

    Uses simple threshold on Q4 distance from Q1 baseline.

    Args:
        X_train: Train features (Q4 distance)
        y_train: Train labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Predicted probabilities
    """
    # Find optimal threshold on train set
    # Higher distance = more likely clearing
    distances_train = X_train[:, 0]  # Q4 distance is first (and only) feature

    # Normalize to [0, 1] using train min/max
    min_dist = distances_train.min()
    max_dist = distances_train.max()

    distances_test = X_test[:, 0]
    y_pred_proba = (distances_test - min_dist) / (max_dist - min_dist + 1e-8)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    return y_pred_proba


def baseline_simple_features(X_train, y_train, X_test, y_test):
    """
    Baseline 3: Simple engineered features with logistic regression.

    Uses distances + velocities with linear model.

    Args:
        X_train: Train features
        y_train: Train labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Predicted probabilities
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return y_pred_proba


def baseline_all_features(X_train, y_train, X_test, y_test):
    """
    Baseline 4: All temporal features with Random Forest.

    Uses all features (distances, velocities, accelerations, trend) with RF.

    Args:
        X_train: Train features
        y_train: Train labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Predicted probabilities
    """
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_pred_proba


def run_baseline_suite(dataset):
    """
    Run full baseline suite.

    Args:
        dataset: Prepared dataset

    Returns:
        Dict with baseline results
    """
    print("="*80)
    print("BASELINE SUITE")
    print("="*80)
    print()

    results = {}

    # Get splits
    train_idx = dataset['splits']['train']
    test_idx = dataset['splits']['test']

    # Baseline 1: Random
    print("Baseline 1: Random predictions...")
    y_train = np.array([dataset['data'][i]['label'] for i in train_idx])
    y_test = np.array([dataset['data'][i]['label'] for i in test_idx])

    y_pred_random = baseline_random(y_train, y_test)
    results['random'] = evaluate_model(y_test, y_pred_random)
    print(f"  ROC-AUC: {results['random']['roc_auc']:.3f}")
    print(f"  PR-AUC:  {results['random']['pr_auc']:.3f}\n")

    # Baseline 2: Raw embeddings (Q4 distance only)
    print("Baseline 2: Raw embeddings (Q4 distance)...")
    X_full, y_full = extract_features(dataset, feature_type='raw')
    X_train = X_full[train_idx]
    X_test = X_full[test_idx]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    y_pred_raw = baseline_raw_embeddings(X_train, y_train, X_test, y_test)
    results['raw_embeddings'] = evaluate_model(y_test, y_pred_raw)
    print(f"  ROC-AUC: {results['raw_embeddings']['roc_auc']:.3f}")
    print(f"  PR-AUC:  {results['raw_embeddings']['pr_auc']:.3f}\n")

    # Baseline 3: Simple features (distances + velocities)
    print("Baseline 3: Simple features (distances + velocities)...")
    X_full, y_full = extract_features(dataset, feature_type='simple')
    X_train = X_full[train_idx]
    X_test = X_full[test_idx]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    y_pred_simple = baseline_simple_features(X_train, y_train, X_test, y_test)
    results['simple_features'] = evaluate_model(y_test, y_pred_simple)
    print(f"  ROC-AUC: {results['simple_features']['roc_auc']:.3f}")
    print(f"  PR-AUC:  {results['simple_features']['pr_auc']:.3f}\n")

    # Baseline 4: All temporal features
    print("Baseline 4: All temporal features (RF)...")
    X_full, y_full = extract_features(dataset, feature_type='all')
    X_train = X_full[train_idx]
    X_test = X_full[test_idx]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    y_pred_all = baseline_all_features(X_train, y_train, X_test, y_test)
    results['all_features'] = evaluate_model(y_test, y_pred_all)
    print(f"  ROC-AUC: {results['all_features']['roc_auc']:.3f}")
    print(f"  PR-AUC:  {results['all_features']['pr_auc']:.3f}\n")

    return results


def plot_baseline_comparison(results, output_path):
    """
    Plot baseline comparison.

    Args:
        results: Dict with baseline results
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    baselines = list(results.keys())
    roc_aucs = [results[b]['roc_auc'] for b in baselines]
    pr_aucs = [results[b]['pr_auc'] for b in baselines]

    # ROC-AUC comparison
    ax = axes[0]
    bars = ax.barh(baselines, roc_aucs, color=sns.color_palette("viridis", len(baselines)))
    ax.axvline(0.5, color='red', linestyle='--', label='Random')
    ax.set_xlabel('ROC-AUC', fontsize=12)
    ax.set_title('Baseline Comparison: ROC-AUC', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, roc_aucs)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)

    # PR-AUC comparison
    ax = axes[1]
    bars = ax.barh(baselines, pr_aucs, color=sns.color_palette("viridis", len(baselines)))
    ax.set_xlabel('PR-AUC', fontsize=12)
    ax.set_title('Baseline Comparison: PR-AUC', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pr_aucs)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved figure to {output_path}")

    plt.close()


def main():
    """Main entry point."""
    print("="*80)
    print("WALK PHASE - BASELINE SUITE")
    print("="*80)
    print()

    config = get_config()

    # Load dataset
    data_dir = config.get_path("paths.data_dir")
    dataset_path = data_dir / "processed" / "walk_dataset.pkl"

    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        print("  Run: uv run python src/walk/01_data_preparation.py")
        return

    dataset = load_dataset(dataset_path)

    # Check if we have enough samples
    if len(dataset['splits']['test']) < 2:
        print("✗ Insufficient test samples for baseline evaluation")
        print("  Run data preparation with more samples:")
        print("  uv run python src/walk/01_data_preparation.py --n-clearing 50 --n-intact 50")
        return

    # Run baseline suite
    results = run_baseline_suite(dataset)

    # Save results
    results_dir = config.get_path("paths.results_dir") / "walk"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "baseline_results.json"

    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'path': str(dataset_path),
            'n_samples': dataset['metadata']['n_samples'],
            'n_train': len(dataset['splits']['train']),
            'n_test': len(dataset['splits']['test']),
        },
        'baselines': results,
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {results_file}\n")

    # Plot comparison
    figures_dir = config.get_path("paths.figures_dir") / "walk"
    plot_path = figures_dir / "baseline_comparison.png"
    plot_baseline_comparison(results, plot_path)

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("\nBaseline Performance (ROC-AUC):")
    for baseline, metrics in results.items():
        print(f"  {baseline:20s}: {metrics['roc_auc']:.3f}")

    print("\n✓ Baseline suite complete")
    print(f"  Results: {results_file}")
    print(f"  Figures: {plot_path}")
    print("="*80)


if __name__ == "__main__":
    main()
