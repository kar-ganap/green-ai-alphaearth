"""Unit tests for geospatial utilities."""

import math

import numpy as np
import pytest

from src.utils.geo import (
    Location,
    buffer_zone,
    distance_matrix,
    geographic_bounds,
    get_neighbors,
    grid_sample_region,
    haversine_distance,
    point_in_bounds,
)


class TestHaversineDistance:
    """Test haversine distance calculations."""

    def test_same_point(self):
        """Distance from point to itself should be 0."""
        dist = haversine_distance(-3.5, -62.5, -3.5, -62.5)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        """Test against a known distance."""
        # Approximate 1 degree at equator ≈ 111 km
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(111000, rel=0.01)  # 1% tolerance

    def test_symmetry(self):
        """Distance should be symmetric."""
        dist1 = haversine_distance(-3.5, -62.5, -3.6, -62.6)
        dist2 = haversine_distance(-3.6, -62.6, -3.5, -62.5)
        assert dist1 == pytest.approx(dist2)

    def test_positive_distance(self):
        """Distance should always be positive."""
        dist = haversine_distance(-3.5, -62.5, -3.6, -62.6)
        assert dist > 0


class TestDistanceMatrix:
    """Test distance matrix calculations."""

    def test_single_location(self):
        """Test with single location."""
        locations = [(-3.5, -62.5)]
        dists = distance_matrix(locations)

        assert dists.shape == (1, 1)
        assert dists[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_multiple_locations(self):
        """Test with multiple locations."""
        locations = [
            (-3.5, -62.5),
            (-3.6, -62.6),
            (-3.7, -62.7),
        ]
        dists = distance_matrix(locations)

        assert dists.shape == (3, 3)
        # Diagonal should be 0
        assert np.allclose(np.diag(dists), 0.0, atol=1e-6)
        # Matrix should be symmetric
        assert np.allclose(dists, dists.T)

    def test_two_sets_of_locations(self):
        """Test distance between two different sets."""
        locs1 = [(-3.5, -62.5), (-3.6, -62.6)]
        locs2 = [(-3.7, -62.7), (-3.8, -62.8), (-3.9, -62.9)]
        dists = distance_matrix(locs1, locs2)

        assert dists.shape == (2, 3)


class TestGetNeighbors:
    """Test neighbor location generation."""

    def test_four_neighbors(self):
        """Test 4 cardinal neighbors."""
        neighbors = get_neighbors(-3.5, -62.5, distance_m=1000, n_neighbors=4)

        assert len(neighbors) == 4
        # All neighbors should be tuples of (lat, lon)
        assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)

    def test_eight_neighbors(self):
        """Test 8 neighbors (cardinal + diagonal)."""
        neighbors = get_neighbors(-3.5, -62.5, distance_m=1000, n_neighbors=8)

        assert len(neighbors) == 8

    def test_neighbor_distance(self):
        """Test that neighbors are approximately at correct distance."""
        center = (-3.5, -62.5)
        distance_m = 1000
        neighbors = get_neighbors(center[0], center[1], distance_m=distance_m, n_neighbors=4)

        for neighbor in neighbors:
            dist = haversine_distance(center[0], center[1], neighbor[0], neighbor[1])
            # Allow 1% error due to Earth curvature approximation
            assert dist == pytest.approx(distance_m, rel=0.01)

    def test_invalid_n_neighbors(self):
        """Test error with invalid number of neighbors."""
        with pytest.raises(ValueError):
            get_neighbors(-3.5, -62.5, distance_m=1000, n_neighbors=5)


class TestBufferZone:
    """Test buffer zone filtering."""

    def test_empty_list(self):
        """Test with empty location list."""
        result = buffer_zone([], buffer_distance_m=1000)
        assert result == []

    def test_single_location(self):
        """Test with single location."""
        locs = [(-3.5, -62.5)]
        result = buffer_zone(locs, buffer_distance_m=1000)
        assert len(result) == 1

    def test_filter_nearby_locations(self):
        """Test that nearby locations are filtered."""
        locs = [
            (-3.5, -62.5),
            (-3.5001, -62.5001),  # Very close to first
            (-4.0, -63.0),  # Far from others
        ]
        result = buffer_zone(locs, buffer_distance_m=1000)

        # Should keep first and third, filter second
        assert len(result) == 2
        assert result[0] == locs[0]
        assert result[1] == locs[2]

    def test_all_locations_kept(self):
        """Test when all locations are far enough apart."""
        locs = [
            (-3.0, -62.0),
            (-4.0, -63.0),
            (-5.0, -64.0),
        ]
        result = buffer_zone(locs, buffer_distance_m=1000)

        assert len(result) == 3


class TestGeographicBounds:
    """Test geographic bounds calculation."""

    def test_single_location(self):
        """Test bounds for single location."""
        locs = [(-3.5, -62.5)]
        bounds = geographic_bounds(locs, padding_degrees=0.1)

        assert bounds["min_lat"] == pytest.approx(-3.6)
        assert bounds["max_lat"] == pytest.approx(-3.4)
        assert bounds["min_lon"] == pytest.approx(-62.6)
        assert bounds["max_lon"] == pytest.approx(-62.4)

    def test_multiple_locations(self):
        """Test bounds for multiple locations."""
        locs = [
            (-3.0, -62.0),
            (-4.0, -63.0),
        ]
        bounds = geographic_bounds(locs, padding_degrees=0.0)

        assert bounds["min_lat"] == -4.0
        assert bounds["max_lat"] == -3.0
        assert bounds["min_lon"] == -63.0
        assert bounds["max_lon"] == -62.0


class TestPointInBounds:
    """Test point in bounds checking."""

    def test_point_inside(self):
        """Test point inside bounds."""
        bounds = {
            "min_lat": -4.0,
            "max_lat": -3.0,
            "min_lon": -63.0,
            "max_lon": -62.0,
        }
        assert point_in_bounds(-3.5, -62.5, bounds) is True

    def test_point_outside(self):
        """Test point outside bounds."""
        bounds = {
            "min_lat": -4.0,
            "max_lat": -3.0,
            "min_lon": -63.0,
            "max_lon": -62.0,
        }
        assert point_in_bounds(-5.0, -65.0, bounds) is False

    def test_point_on_boundary(self):
        """Test point exactly on boundary."""
        bounds = {
            "min_lat": -4.0,
            "max_lat": -3.0,
            "min_lon": -63.0,
            "max_lon": -62.0,
        }
        assert point_in_bounds(-4.0, -63.0, bounds) is True


class TestGridSampleRegion:
    """Test grid sampling."""

    def test_grid_generation(self):
        """Test that grid is generated correctly."""
        bounds = {
            "min_lat": -3.5,
            "max_lat": -3.4,
            "min_lon": -62.5,
            "max_lon": -62.4,
        }
        grid = grid_sample_region(bounds, spacing_degrees=0.05)

        # Should have approximately 3x3 = 9 points
        assert len(grid) >= 4  # At least 2x2

        # All points should be tuples
        assert all(isinstance(p, tuple) and len(p) == 2 for p in grid)

        # All points should be within bounds
        for lat, lon in grid:
            assert point_in_bounds(lat, lon, bounds)


class TestLocation:
    """Test Location class."""

    def test_initialization(self):
        """Test Location initialization."""
        loc = Location(lat=-3.5, lon=-62.5)
        assert loc.lat == -3.5
        assert loc.lon == -62.5
        assert loc.metadata == {}

    def test_initialization_with_metadata(self):
        """Test Location with metadata."""
        loc = Location(lat=-3.5, lon=-62.5, metadata={"name": "Site A"})
        assert loc.metadata["name"] == "Site A"

    def test_distance_to(self):
        """Test distance_to method."""
        loc1 = Location(lat=-3.5, lon=-62.5)
        loc2 = Location(lat=-3.6, lon=-62.6)

        dist = loc1.distance_to(loc2)
        expected = haversine_distance(-3.5, -62.5, -3.6, -62.6)

        assert dist == pytest.approx(expected)

    def test_to_tuple(self):
        """Test to_tuple method."""
        loc = Location(lat=-3.5, lon=-62.5)
        assert loc.to_tuple() == (-3.5, -62.5)

    def test_to_dict(self):
        """Test to_dict method."""
        loc = Location(lat=-3.5, lon=-62.5, metadata={"name": "Site A"})
        d = loc.to_dict()

        assert d["lat"] == -3.5
        assert d["lon"] == -62.5
        assert d["name"] == "Site A"

    def test_repr(self):
        """Test string representation."""
        loc = Location(lat=-3.5, lon=-62.5)
        repr_str = repr(loc)

        assert "Location" in repr_str
        assert "-3.5" in repr_str
        assert "-62.5" in repr_str
