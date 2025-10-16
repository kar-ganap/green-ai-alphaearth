"""Unit tests for config utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config


@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    config_data = {
        "project": {
            "name": "test-project",
            "version": "1.0.0",
        },
        "data": {
            "alphaearth": {
                "collection": "TEST/COLLECTION",
                "dimensions": 64,
            },
            "region": {
                "bounds": {
                    "min_lat": -4.0,
                    "max_lat": -3.0,
                    "min_lon": -63.0,
                    "max_lon": -62.0,
                },
            },
        },
        "model": {
            "type": "xgboost",
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 6,
            },
        },
        "validation": {
            "requirements": {
                "min_roc_auc": 0.75,
                "min_precision_at_50_recall": 0.70,
            },
        },
        "crawl_tests": {
            "test_1_separability": {
                "min_accuracy": 0.85,
            },
        },
        "paths": {
            "data_dir": "data",
            "results_dir": "results",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestConfig:
    """Test Config class."""

    def test_load_config(self, temp_config):
        """Test that config loads correctly."""
        config = Config(temp_config)
        assert config._config is not None
        assert isinstance(config._config, dict)

    def test_get_simple_value(self, temp_config):
        """Test getting simple config values."""
        config = Config(temp_config)
        assert config.get("project.name") == "test-project"
        assert config.get("project.version") == "1.0.0"

    def test_get_nested_value(self, temp_config):
        """Test getting nested config values."""
        config = Config(temp_config)
        assert config.get("model.xgboost.n_estimators") == 200
        assert config.get("model.xgboost.max_depth") == 6

    def test_get_default_value(self, temp_config):
        """Test default value when key doesn't exist."""
        config = Config(temp_config)
        assert config.get("nonexistent.key", default="default") == "default"
        assert config.get("nonexistent.key") is None

    def test_get_path(self, temp_config):
        """Test path resolution."""
        config = Config(temp_config)
        data_path = config.get_path("paths.data_dir")

        assert isinstance(data_path, Path)
        assert data_path.name == "data"

    def test_alphaearth_collection_property(self, temp_config):
        """Test AlphaEarth collection property."""
        config = Config(temp_config)
        assert config.alphaearth_collection == "TEST/COLLECTION"

    def test_embedding_dimensions_property(self, temp_config):
        """Test embedding dimensions property."""
        config = Config(temp_config)
        assert config.embedding_dimensions == 64

    def test_study_region_bounds_property(self, temp_config):
        """Test study region bounds property."""
        config = Config(temp_config)
        bounds = config.study_region_bounds

        assert isinstance(bounds, dict)
        assert bounds["min_lat"] == -4.0
        assert bounds["max_lat"] == -3.0
        assert bounds["min_lon"] == -63.0
        assert bounds["max_lon"] == -62.0

    def test_model_params_property(self, temp_config):
        """Test model params property."""
        config = Config(temp_config)
        params = config.model_params

        assert isinstance(params, dict)
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 6

    def test_validation_requirements_property(self, temp_config):
        """Test validation requirements property."""
        config = Config(temp_config)
        reqs = config.validation_requirements

        assert isinstance(reqs, dict)
        assert reqs["min_roc_auc"] == 0.75
        assert reqs["min_precision_at_50_recall"] == 0.70

    def test_crawl_thresholds_property(self, temp_config):
        """Test CRAWL test thresholds property."""
        config = Config(temp_config)
        thresholds = config.crawl_thresholds

        assert isinstance(thresholds, dict)
        assert "test_1_separability" in thresholds
        assert thresholds["test_1_separability"]["min_accuracy"] == 0.85

    def test_repr(self, temp_config):
        """Test string representation."""
        config = Config(temp_config)
        repr_str = repr(config)

        assert "Config" in repr_str
        assert temp_config in repr_str


def test_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config("/nonexistent/config.yaml")
