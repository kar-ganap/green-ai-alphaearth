"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml. If None, searches in project root.
        """
        if config_path is None:
            # Find project root (where config.yaml lives)
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "model.xgboost.n_estimators")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> config.get("model.xgboost.n_estimators")
            200
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, path_key: str) -> Path:
        """
        Get absolute path from config.

        Args:
            path_key: Key for path (e.g., "paths.data_dir")

        Returns:
            Absolute Path object

        Example:
            >>> config = Config()
            >>> config.get_path("paths.data_dir")
            Path("/path/to/project/data")
        """
        relative_path = self.get(path_key)
        if relative_path is None:
            raise ValueError(f"Path key not found: {path_key}")

        return self.project_root / relative_path

    @property
    def alphaearth_collection(self) -> str:
        """Get AlphaEarth collection ID."""
        return self.get("data.alphaearth.collection")

    @property
    def embedding_dimensions(self) -> int:
        """Get AlphaEarth embedding dimensions."""
        return self.get("data.alphaearth.dimensions", 64)

    @property
    def study_region_bounds(self) -> Dict[str, float]:
        """Get study region bounding box."""
        return self.get("data.region.bounds")

    @property
    def model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        model_type = self.get("model.type", "xgboost")
        return self.get(f"model.{model_type}", {})

    @property
    def validation_requirements(self) -> Dict[str, float]:
        """Get validation requirements."""
        return self.get("validation.requirements", {})

    @property
    def crawl_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get CRAWL test thresholds."""
        return self.get("crawl_tests", {})

    def ensure_directories(self):
        """Create all configured directories if they don't exist."""
        paths_config = self.get("paths", {})

        for path_key, path_value in paths_config.items():
            full_path = self.project_root / path_value
            full_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file (optional, uses default if None)

    Returns:
        Config instance

    Example:
        >>> from src.utils.config import get_config
        >>> config = get_config()
        >>> config.get("model.xgboost.n_estimators")
        200
    """
    global _config

    if _config is None:
        _config = Config(config_path)

    return _config
