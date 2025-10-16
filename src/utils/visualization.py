"""Visualization utilities for plots and maps."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from .config import get_config


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def save_figure(fig: Figure, filename: str, subdir: str = None, dpi: int = 300):
    """
    Save figure to results directory.

    Args:
        fig: Matplotlib figure
        filename: Filename (with .png extension)
        subdir: Subdirectory in figures/ (e.g., "crawl", "walk", "run")
        dpi: Resolution

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_figure(fig, "test_plot.png", subdir="crawl")
    """
    config = get_config()
    figures_dir = config.get_path("paths.results_dir") / "figures"

    if subdir:
        figures_dir = figures_dir / subdir

    figures_dir.mkdir(parents=True, exist_ok=True)

    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {filepath}")


def plot_pca_separation(
    embeddings_cleared: np.ndarray,
    embeddings_intact: np.ndarray,
    accuracy: float,
    output_path: str = None,
) -> Figure:
    """
    Plot PCA visualization of cleared vs intact forest embeddings.

    Args:
        embeddings_cleared: Embeddings from cleared locations (n, 64)
        embeddings_intact: Embeddings from intact locations (n, 64)
        accuracy: Classification accuracy to display
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> cleared = np.random.randn(50, 64)
        >>> intact = np.random.randn(50, 64) + 2
        >>> fig = plot_pca_separation(cleared, intact, accuracy=0.89)
    """
    from sklearn.decomposition import PCA

    # Combine embeddings
    X = np.vstack([embeddings_cleared, embeddings_intact])
    y = np.array([1] * len(embeddings_cleared) + [0] * len(embeddings_intact))

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter_cleared = ax.scatter(
        X_2d[y == 1, 0],
        X_2d[y == 1, 1],
        c="red",
        alpha=0.6,
        s=50,
        label=f"Cleared (n={len(embeddings_cleared)})",
        edgecolors="darkred",
    )

    scatter_intact = ax.scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        c="green",
        alpha=0.6,
        s=50,
        label=f"Intact (n={len(embeddings_intact)})",
        edgecolors="darkgreen",
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Embedding Separability Test\nAccuracy: {accuracy:.1%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with results
    result_text = f"✓ PASS" if accuracy >= 0.85 else "✗ FAIL"
    color = "green" if accuracy >= 0.85 else "red"

    ax.text(
        0.02,
        0.98,
        f"{result_text}\nAccuracy: {accuracy:.1%}\nRequired: ≥85%",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_temporal_signal(
    times: List[int],
    distances: np.ndarray,
    p_value: float,
    output_path: str = None,
) -> Figure:
    """
    Plot temporal signal showing embedding changes before clearing.

    Args:
        times: Time points (e.g., [-6, -3, -1, 0, 3] months)
        distances: Mean distances from baseline, shape (n_trajectories, len(times))
        p_value: Statistical significance of -3 month signal
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> times = [-6, -3, -1, 0, 3]
        >>> distances = np.random.rand(20, 5).cumsum(axis=1)
        >>> fig = plot_temporal_signal(times, distances, p_value=0.02)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot individual trajectories
    for i in range(distances.shape[0]):
        ax.plot(
            times,
            distances[i],
            color="gray",
            alpha=0.3,
            linewidth=1,
        )

    # Plot mean trajectory
    mean_distances = distances.mean(axis=0)
    std_distances = distances.std(axis=0)

    ax.plot(
        times,
        mean_distances,
        color="red",
        linewidth=3,
        label="Mean trajectory",
        marker="o",
        markersize=8,
    )

    # Add confidence interval
    ax.fill_between(
        times,
        mean_distances - std_distances,
        mean_distances + std_distances,
        color="red",
        alpha=0.2,
        label="±1 std",
    )

    # Mark clearing event
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Clearing event")

    ax.set_xlabel("Months relative to clearing")
    ax.set_ylabel("Distance from baseline embedding")
    ax.set_title(f"Temporal Signal Test\np-value at -3 months: {p_value:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add result box
    result_text = f"✓ PASS" if p_value < 0.05 else "✗ FAIL"
    color = "green" if p_value < 0.05 else "red"

    ax.text(
        0.02,
        0.98,
        f"{result_text}\np-value: {p_value:.4f}\nRequired: <0.05",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_regional_generalization(
    region_results: Dict[str, Dict[str, float]],
    cv_threshold: float = 0.5,
    output_path: str = None,
) -> Figure:
    """
    Plot signal consistency across regions.

    Args:
        region_results: Dict mapping region name to {"mean": float, "std": float}
        cv_threshold: Maximum acceptable coefficient of variation
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> results = {
        ...     "north": {"mean": 0.5, "std": 0.1},
        ...     "south": {"mean": 0.6, "std": 0.15},
        ...     "east": {"mean": 0.55, "std": 0.12},
        ... }
        >>> fig = plot_regional_generalization(results)
    """
    regions = list(region_results.keys())
    means = [region_results[r]["mean"] for r in regions]
    stds = [region_results[r]["std"] for r in regions]

    # Calculate CV
    cv = np.std(means) / np.mean(means)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(regions, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue")

    ax.set_ylabel("Mean signal strength")
    ax.set_xlabel("Region")
    ax.set_title(f"Regional Generalization Test\nCoefficient of Variation: {cv:.3f}")
    ax.grid(True, alpha=0.3, axis="y")

    # Add result box
    result_text = f"✓ PASS" if cv < cv_threshold else "⚠ WARNING"
    color = "green" if cv < cv_threshold else "orange"

    ax.text(
        0.02,
        0.98,
        f"{result_text}\nCV: {cv:.3f}\nRequired: <{cv_threshold}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_minimal_model_results(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    auc: float,
    output_path: str = None,
) -> Figure:
    """
    Plot minimal model results (2 features).

    Args:
        X: Features (n_samples, 2)
        y: Labels (n_samples,)
        feature_names: List of 2 feature names
        auc: ROC-AUC score
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> X = np.random.rand(200, 2)
        >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
        >>> fig = plot_minimal_model_results(X, y, ["velocity", "dist_to_road"], 0.75)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter_neg = ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="green",
        alpha=0.5,
        s=50,
        label=f"Stable (n={np.sum(y == 0)})",
        edgecolors="darkgreen",
    )

    scatter_pos = ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="red",
        alpha=0.5,
        s=50,
        label=f"Cleared (n={np.sum(y == 1)})",
        edgecolors="darkred",
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"Minimal Model Test (2 features)\nAUC: {auc:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add result box
    if auc >= 0.75:
        result_text = "✓ EXCELLENT"
        color = "green"
    elif auc >= 0.65:
        result_text = "✓ PASS"
        color = "lightgreen"
    else:
        result_text = "✗ FAIL"
        color = "red"

    ax.text(
        0.02,
        0.98,
        f"{result_text}\nAUC: {auc:.3f}\nRequired: ≥0.65",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "ROC Curve",
    output_path: str = None,
) -> Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> fig = plot_roc_curve(y_true, y_pred)
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    output_path: str = None,
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels (not probabilities)
        labels: Class labels
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = ["Stable", "Cleared"]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def create_decision_gate_summary(
    test_results: Dict[str, Dict[str, any]],
    output_path: str = None,
) -> Figure:
    """
    Create summary visualization for CRAWL decision gate.

    Args:
        test_results: Dict mapping test name to results dict with "passed" and "value" keys
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> results = {
        ...     "Test 1: Separability": {"passed": True, "value": 0.89, "threshold": 0.85},
        ...     "Test 2: Temporal": {"passed": True, "value": 0.02, "threshold": 0.05},
        ...     "Test 3: Generalization": {"passed": True, "value": 0.3, "threshold": 0.5},
        ...     "Test 4: Minimal Model": {"passed": True, "value": 0.75, "threshold": 0.65},
        ... }
        >>> fig = create_decision_gate_summary(results)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    tests = list(test_results.keys())
    passed = [test_results[t]["passed"] for t in tests]
    colors = ["green" if p else "red" for p in passed]

    y_pos = np.arange(len(tests))

    ax.barh(y_pos, [1] * len(tests), color=colors, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests)
    ax.set_xlim([0, 1])
    ax.set_xticks([])

    # Add value annotations
    for i, test in enumerate(tests):
        result = test_results[test]
        text = f"Value: {result['value']:.3f} | Threshold: {result['threshold']:.3f}"
        ax.text(0.5, i, text, ha="center", va="center", fontsize=11, fontweight="bold")

    # Overall result
    all_passed = all(passed)
    decision = "GO TO WALK PHASE" if all_passed else "STOP / PIVOT"
    decision_color = "green" if all_passed else "red"

    ax.set_title(
        f"CRAWL Phase Decision Gate\n{decision}",
        fontsize=16,
        fontweight="bold",
        color=decision_color,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
