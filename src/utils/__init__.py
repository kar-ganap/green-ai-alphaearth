"""Utility modules for the deforestation early warning system."""

from .config import Config, get_config
from .earth_engine import EarthEngineClient, initialize_earth_engine
from .geo import (
    Location,
    buffer_zone,
    distance_matrix,
    geographic_bounds,
    get_neighbors,
    grid_sample_region,
    haversine_distance,
    point_in_bounds,
)
from .visualization import (
    create_decision_gate_summary,
    plot_confusion_matrix,
    plot_minimal_model_results,
    plot_pca_separation,
    plot_regional_generalization,
    plot_roc_curve,
    plot_temporal_signal,
    save_figure,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    # Earth Engine
    "EarthEngineClient",
    "initialize_earth_engine",
    # Geo
    "Location",
    "buffer_zone",
    "distance_matrix",
    "geographic_bounds",
    "get_neighbors",
    "grid_sample_region",
    "haversine_distance",
    "point_in_bounds",
    # Visualization
    "create_decision_gate_summary",
    "plot_confusion_matrix",
    "plot_minimal_model_results",
    "plot_pca_separation",
    "plot_regional_generalization",
    "plot_roc_curve",
    "plot_temporal_signal",
    "save_figure",
]
