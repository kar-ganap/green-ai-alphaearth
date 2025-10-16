"""Unit tests for visualization utilities."""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from src.utils.visualization import (
    plot_confusion_matrix,
    plot_minimal_model_results,
    plot_pca_separation,
    plot_regional_generalization,
    plot_roc_curve,
    plot_temporal_signal,
)


class TestPlotPCASeparation:
    """Test PCA separation plot."""

    def test_basic_plot(self):
        """Test that basic plot generation works."""
        # Generate test data
        np.random.seed(42)
        cleared = np.random.randn(50, 64)
        intact = np.random.randn(50, 64) + 2  # Shifted for separation

        fig = plot_pca_separation(cleared, intact, accuracy=0.89)

        assert fig is not None
        assert len(fig.axes) == 1  # Should have one subplot

    def test_plot_with_output_path(self):
        """Test saving plot to file."""
        np.random.seed(42)
        cleared = np.random.randn(50, 64)
        intact = np.random.randn(50, 64) + 2

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"

            fig = plot_pca_separation(cleared, intact, accuracy=0.89, output_path=str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File has content


class TestPlotTemporalSignal:
    """Test temporal signal plot."""

    def test_basic_plot(self):
        """Test basic temporal plot generation."""
        times = [-6, -3, -1, 0, 3]
        np.random.seed(42)
        distances = np.random.rand(20, 5).cumsum(axis=1)

        fig = plot_temporal_signal(times, distances, p_value=0.02)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_plot_with_significant_pvalue(self):
        """Test plot with significant p-value."""
        times = [-6, -3, -1, 0, 3]
        distances = np.random.rand(20, 5)

        fig = plot_temporal_signal(times, distances, p_value=0.001)
        assert fig is not None

    def test_plot_with_nonsignificant_pvalue(self):
        """Test plot with non-significant p-value."""
        times = [-6, -3, -1, 0, 3]
        distances = np.random.rand(20, 5)

        fig = plot_temporal_signal(times, distances, p_value=0.9)
        assert fig is not None


class TestPlotRegionalGeneralization:
    """Test regional generalization plot."""

    def test_basic_plot(self):
        """Test basic regional plot generation."""
        region_results = {
            "north": {"mean": 0.5, "std": 0.1},
            "south": {"mean": 0.6, "std": 0.15},
            "east": {"mean": 0.55, "std": 0.12},
        }

        fig = plot_regional_generalization(region_results, cv_threshold=0.5)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_plot_with_low_cv(self):
        """Test plot with low coefficient of variation (good)."""
        region_results = {
            "north": {"mean": 0.5, "std": 0.1},
            "south": {"mean": 0.51, "std": 0.1},
            "east": {"mean": 0.49, "std": 0.1},
        }

        fig = plot_regional_generalization(region_results, cv_threshold=0.5)
        assert fig is not None

    def test_plot_with_high_cv(self):
        """Test plot with high coefficient of variation (warning)."""
        region_results = {
            "north": {"mean": 0.2, "std": 0.1},
            "south": {"mean": 0.8, "std": 0.1},
        }

        fig = plot_regional_generalization(region_results, cv_threshold=0.5)
        assert fig is not None


class TestPlotMinimalModelResults:
    """Test minimal model results plot."""

    def test_basic_plot(self):
        """Test basic minimal model plot."""
        np.random.seed(42)
        X = np.random.rand(200, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        feature_names = ["velocity", "dist_to_road"]

        fig = plot_minimal_model_results(X, y, feature_names, auc=0.75)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_plot_with_excellent_auc(self):
        """Test plot with excellent AUC."""
        X = np.random.rand(200, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        feature_names = ["feature1", "feature2"]

        fig = plot_minimal_model_results(X, y, feature_names, auc=0.85)
        assert fig is not None

    def test_plot_with_failing_auc(self):
        """Test plot with failing AUC."""
        X = np.random.rand(200, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        feature_names = ["feature1", "feature2"]

        fig = plot_minimal_model_results(X, y, feature_names, auc=0.55)
        assert fig is not None


class TestPlotROCCurve:
    """Test ROC curve plot."""

    def test_basic_roc_plot(self):
        """Test basic ROC curve generation."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15])

        fig = plot_roc_curve(y_true, y_pred)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_perfect_predictions(self):
        """Test ROC curve with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        fig = plot_roc_curve(y_true, y_pred)
        assert fig is not None

    def test_random_predictions(self):
        """Test ROC curve with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)

        fig = plot_roc_curve(y_true, y_pred)
        assert fig is not None


class TestPlotConfusionMatrix:
    """Test confusion matrix plot."""

    def test_basic_confusion_matrix(self):
        """Test basic confusion matrix generation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        fig = plot_confusion_matrix(y_true, y_pred)

        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar

    def test_confusion_matrix_with_labels(self):
        """Test confusion matrix with custom labels."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        fig = plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"])
        assert fig is not None

    def test_perfect_confusion_matrix(self):
        """Test confusion matrix with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
