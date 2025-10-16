"""Geospatial utilities for distance calculations and coordinate handling."""

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great circle distance between two points on Earth.

    Uses the Haversine formula.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters

    Example:
        >>> distance = haversine_distance(-3.5, -62.5, -3.6, -62.6)
        >>> distance
        15313.7  # meters
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Earth radius in meters
    earth_radius = 6371000

    return earth_radius * c


def distance_matrix(
    locations1: List[Tuple[float, float]],
    locations2: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Calculate pairwise distances between locations.

    Args:
        locations1: List of (lat, lon) tuples
        locations2: Optional second list. If None, computes distances within locations1

    Returns:
        Distance matrix in meters, shape (len(locations1), len(locations2))

    Example:
        >>> locs1 = [(-3.5, -62.5), (-3.6, -62.6)]
        >>> locs2 = [(-3.7, -62.7), (-3.8, -62.8)]
        >>> dists = distance_matrix(locs1, locs2)
        >>> dists.shape
        (2, 2)
    """
    if locations2 is None:
        locations2 = locations1

    # Convert to numpy arrays
    coords1 = np.array(locations1)
    coords2 = np.array(locations2)

    # Vectorized haversine distance
    def haversine_vectorized(coords1, coords2):
        # Convert to radians
        coords1_rad = np.radians(coords1)
        coords2_rad = np.radians(coords2)

        # Compute distances
        distances = np.zeros((len(coords1), len(coords2)))

        for i, (lat1, lon1) in enumerate(coords1_rad):
            for j, (lat2, lon2) in enumerate(coords2_rad):
                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                c = 2 * np.arcsin(np.sqrt(a))

                distances[i, j] = 6371000 * c  # Earth radius in meters

        return distances

    return haversine_vectorized(coords1, coords2)


def get_neighbors(
    center_lat: float,
    center_lon: float,
    distance_m: float = 1000,
    n_neighbors: int = 8,
) -> List[Tuple[float, float]]:
    """
    Get N neighboring locations around a center point.

    Creates a regular grid of neighbors at specified distance.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        distance_m: Distance to neighbors in meters
        n_neighbors: Number of neighbors (4 or 8)

    Returns:
        List of (lat, lon) tuples for neighbor locations

    Example:
        >>> neighbors = get_neighbors(-3.5, -62.5, distance_m=1000, n_neighbors=8)
        >>> len(neighbors)
        8
    """
    # Approximate conversion: 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude varies with latitude
    meters_per_degree_lat = 111000
    meters_per_degree_lon = 111000 * math.cos(math.radians(center_lat))

    # Convert distance to degrees
    delta_lat = distance_m / meters_per_degree_lat
    delta_lon = distance_m / meters_per_degree_lon

    if n_neighbors == 4:
        # Cardinal directions: N, E, S, W
        offsets = [
            (delta_lat, 0),  # North
            (0, delta_lon),  # East
            (-delta_lat, 0),  # South
            (0, -delta_lon),  # West
        ]
    elif n_neighbors == 8:
        # Include diagonals
        diagonal_dist = delta_lat / math.sqrt(2)
        offsets = [
            (delta_lat, 0),  # N
            (diagonal_dist, diagonal_dist),  # NE
            (0, delta_lon),  # E
            (-diagonal_dist, diagonal_dist),  # SE
            (-delta_lat, 0),  # S
            (-diagonal_dist, -diagonal_dist),  # SW
            (0, -delta_lon),  # W
            (diagonal_dist, -diagonal_dist),  # NW
        ]
    else:
        raise ValueError("n_neighbors must be 4 or 8")

    neighbors = [
        (center_lat + dlat, center_lon + dlon)
        for dlat, dlon in offsets
    ]

    return neighbors


def buffer_zone(
    locations: List[Tuple[float, float]],
    buffer_distance_m: float,
) -> List[Tuple[float, float]]:
    """
    Remove locations within buffer distance of each other.

    Useful for spatial cross-validation to prevent leakage.

    Args:
        locations: List of (lat, lon) tuples
        buffer_distance_m: Minimum distance between locations

    Returns:
        Filtered list of locations with minimum spacing

    Example:
        >>> locs = [(-3.5, -62.5), (-3.5001, -62.5001), (-4.0, -63.0)]
        >>> filtered = buffer_zone(locs, buffer_distance_m=1000)
        >>> len(filtered) < len(locs)
        True
    """
    if len(locations) == 0:
        return []

    # Convert to numpy array
    coords = np.array(locations)

    # Compute pairwise distances
    dist_matrix = distance_matrix(locations)

    # Greedy selection: keep locations with minimum spacing
    selected = [0]  # Start with first location

    for i in range(1, len(locations)):
        # Check distance to all selected locations
        distances_to_selected = dist_matrix[i, selected]

        if np.all(distances_to_selected >= buffer_distance_m):
            selected.append(i)

    return [locations[i] for i in selected]


def geographic_bounds(
    locations: List[Tuple[float, float]],
    padding_degrees: float = 0.1,
) -> Dict[str, float]:
    """
    Get bounding box for a list of locations.

    Args:
        locations: List of (lat, lon) tuples
        padding_degrees: Extra padding around bounds

    Returns:
        Dict with keys: min_lat, max_lat, min_lon, max_lon

    Example:
        >>> locs = [(-3.5, -62.5), (-4.0, -63.0)]
        >>> bounds = geographic_bounds(locs, padding_degrees=0.1)
        >>> bounds
        {'min_lat': -4.1, 'max_lat': -3.4, 'min_lon': -63.1, 'max_lon': -62.4}
    """
    coords = np.array(locations)

    return {
        "min_lat": float(coords[:, 0].min() - padding_degrees),
        "max_lat": float(coords[:, 0].max() + padding_degrees),
        "min_lon": float(coords[:, 1].min() - padding_degrees),
        "max_lon": float(coords[:, 1].max() + padding_degrees),
    }


def point_in_bounds(
    lat: float,
    lon: float,
    bounds: Dict[str, float],
) -> bool:
    """
    Check if a point is within geographic bounds.

    Args:
        lat: Latitude
        lon: Longitude
        bounds: Dict with min_lat, max_lat, min_lon, max_lon

    Returns:
        True if point is within bounds

    Example:
        >>> bounds = {"min_lat": -4, "max_lat": -3, "min_lon": -63, "max_lon": -62}
        >>> point_in_bounds(-3.5, -62.5, bounds)
        True
        >>> point_in_bounds(-5.0, -65.0, bounds)
        False
    """
    return (
        bounds["min_lat"] <= lat <= bounds["max_lat"]
        and bounds["min_lon"] <= lon <= bounds["max_lon"]
    )


def grid_sample_region(
    bounds: Dict[str, float],
    spacing_degrees: float = 0.01,
) -> List[Tuple[float, float]]:
    """
    Create a regular grid of sample points within bounds.

    Args:
        bounds: Geographic bounds
        spacing_degrees: Grid spacing in degrees (~1.1 km at equator)

    Returns:
        List of (lat, lon) grid points

    Example:
        >>> bounds = {"min_lat": -3.5, "max_lat": -3.4, "min_lon": -62.5, "max_lon": -62.4}
        >>> grid = grid_sample_region(bounds, spacing_degrees=0.05)
        >>> len(grid)
        9  # 3x3 grid
    """
    lats = np.arange(
        bounds["min_lat"],
        bounds["max_lat"],
        spacing_degrees,
    )
    lons = np.arange(
        bounds["min_lon"],
        bounds["max_lon"],
        spacing_degrees,
    )

    grid_points = [
        (lat, lon)
        for lat in lats
        for lon in lons
    ]

    return grid_points


class Location:
    """Wrapper for geographic location with useful methods."""

    def __init__(self, lat: float, lon: float, metadata: Dict = None):
        """
        Initialize location.

        Args:
            lat: Latitude
            lon: Longitude
            metadata: Optional metadata dict
        """
        self.lat = lat
        self.lon = lon
        self.metadata = metadata or {}

    def distance_to(self, other: "Location") -> float:
        """Calculate distance to another location in meters."""
        return haversine_distance(self.lat, self.lon, other.lat, other.lon)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lat, lon) tuple."""
        return (self.lat, self.lon)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "lat": self.lat,
            "lon": self.lon,
            **self.metadata,
        }

    def __repr__(self) -> str:
        return f"Location(lat={self.lat:.4f}, lon={self.lon:.4f})"
