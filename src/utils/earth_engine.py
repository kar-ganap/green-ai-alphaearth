"""Google Earth Engine utilities for AlphaEarth embeddings and data access."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ee
import numpy as np
from tqdm import tqdm

from .config import get_config


class EarthEngineClient:
    """Client for Google Earth Engine operations."""

    def __init__(self, use_cache: bool = True):
        """
        Initialize Earth Engine client.

        Args:
            use_cache: Whether to cache API responses to disk
        """
        self.config = get_config()
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir = self.config.get_path("paths.cache_dir")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            print(f"Earth Engine initialization failed: {e}")
            print("Run: earthengine authenticate")
            raise

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[any]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def _save_to_cache(self, cache_key: str, data: any):
        """Save data to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")

    def get_embedding(
        self,
        lat: float,
        lon: float,
        date: str,
        collection: str = None,
    ) -> np.ndarray:
        """
        Get AlphaEarth embedding for a location and date.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date string (YYYY-MM-DD)
            collection: AlphaEarth collection ID (uses config default if None)

        Returns:
            64-dimensional embedding as numpy array

        Example:
            >>> client = EarthEngineClient()
            >>> emb = client.get_embedding(-3.5, -62.5, "2023-01-01")
            >>> emb.shape
            (64,)
        """
        # Create cache key
        cache_key = f"embedding_{lat}_{lon}_{date}_{collection}"

        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Get from Earth Engine
        if collection is None:
            collection = self.config.alphaearth_collection

        point = ee.Geometry.Point([lon, lat])

        # Parse year from date
        year = datetime.strptime(date, "%Y-%m-%d").year

        # Get image for year (AlphaEarth is annual)
        image_collection = ee.ImageCollection(collection)
        image = image_collection.filterDate(f"{year}-01-01", f"{year}-12-31").filterBounds(point).first()

        # Sample embedding at point
        # Use scale=30 to match Hansen GFC resolution (aggregates 3x3 grid of 10m AlphaEarth pixels)
        sample = image.sample(region=point, scale=30, numPixels=1)
        features = sample.getInfo()["features"]

        if len(features) == 0:
            raise ValueError(f"No embedding found at ({lat}, {lon}) on {date}")

        # Extract embedding values
        properties = features[0]["properties"]
        embedding_dims = self.config.embedding_dimensions

        # AlphaEarth embeddings are stored as A00, A01, ..., A63
        embedding = np.array([properties.get(f"A{i:02d}", 0.0) for i in range(embedding_dims)])

        # Cache result
        self._save_to_cache(cache_key, embedding)

        return embedding

    def get_embedding_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        interval_days: int = 30,
    ) -> Tuple[List[datetime], np.ndarray]:
        """
        Get time series of embeddings for a location.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval_days: Interval between samples in days

        Returns:
            Tuple of (dates, embeddings) where embeddings is shape (n_timesteps, 64)

        Example:
            >>> client = EarthEngineClient()
            >>> dates, embeddings = client.get_embedding_timeseries(
            ...     -3.5, -62.5, "2022-01-01", "2023-01-01", interval_days=30
            ... )
            >>> embeddings.shape
            (12, 64)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dates = []
        embeddings = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")

            try:
                emb = self.get_embedding(lat, lon, date_str)
                dates.append(current)
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Failed to get embedding for {date_str}: {e}")

            current += timedelta(days=interval_days)

        return dates, np.array(embeddings)

    def get_deforestation_labels(
        self,
        bounds: Dict[str, float],
        year: int,
        min_tree_cover: int = 30,
        min_loss_area: float = 0.5,
    ) -> List[Dict]:
        """
        Get deforestation events from Global Forest Watch.

        Args:
            bounds: Dict with keys: min_lat, max_lat, min_lon, max_lon
            year: Year of forest loss
            min_tree_cover: Minimum tree cover % in 2000
            min_loss_area: Minimum loss area in hectares

        Returns:
            List of deforestation events, each with location and metadata

        Example:
            >>> client = EarthEngineClient()
            >>> bounds = {"min_lat": -4, "max_lat": -3, "min_lon": -63, "max_lon": -62}
            >>> events = client.get_deforestation_labels(bounds, year=2023)
        """
        # Create region of interest
        roi = ee.Geometry.Rectangle([
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"],
        ])

        # Load Hansen Global Forest Change dataset
        gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

        # Get tree cover, loss, and loss year
        tree_cover = gfc.select("treecover2000")
        loss = gfc.select("loss")
        loss_year = gfc.select("lossyear")

        # Filter to specified year and tree cover threshold
        # Note: lossyear is encoded as 0-24 for years 2000-2024
        year_code = year - 2000

        mask = (
            tree_cover.gte(min_tree_cover)
            .And(loss.eq(1))
            .And(loss_year.eq(year_code))
        )

        # Get loss pixels
        loss_pixels = mask.selfMask()

        # Sample points from loss pixels
        # For computational efficiency, sample up to 1000 points
        sample = loss_pixels.sample(
            region=roi,
            scale=30,  # 30m resolution
            numPixels=1000,
            seed=42,
            geometries=True,
        )

        # Get sample points
        features = sample.getInfo()["features"]

        events = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]
            events.append({
                "lat": coords[1],
                "lon": coords[0],
                "year": year,
                "date": f"{year}-06-01",  # Approximate mid-year
                "source": "GFW",
            })

        return events

    def get_stable_forest_locations(
        self,
        bounds: Dict[str, float],
        n_samples: int = 100,
        min_tree_cover: int = 80,
        max_loss_year: int = 2015,
    ) -> List[Dict]:
        """
        Get locations with stable forest (no recent clearing).

        Args:
            bounds: Geographic bounds
            n_samples: Number of sample locations
            min_tree_cover: Minimum tree cover % required
            max_loss_year: Latest acceptable loss year (earlier = more stable)

        Returns:
            List of stable forest locations

        Example:
            >>> client = EarthEngineClient()
            >>> bounds = {"min_lat": -4, "max_lat": -3, "min_lon": -63, "max_lon": -62}
            >>> stable_locations = client.get_stable_forest_locations(bounds, n_samples=50)
        """
        roi = ee.Geometry.Rectangle([
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"],
        ])

        # Load Hansen dataset
        gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

        tree_cover = gfc.select("treecover2000")
        loss = gfc.select("loss")
        loss_year = gfc.select("lossyear")

        # Stable forest: high tree cover AND no loss
        # Note: We only use loss.eq(0) to avoid EE sampling quirks with OR clauses
        # The max_loss_year parameter is kept for API compatibility but not used
        stable_mask = tree_cover.gte(min_tree_cover).And(loss.eq(0))

        stable_pixels = stable_mask.selfMask()

        # Sample points - request 5x more due to EE sampling behavior
        # EE's sample() often returns fewer pixels than requested
        sample = stable_pixels.sample(
            region=roi,
            scale=30,
            numPixels=n_samples * 5,
            seed=42,
            geometries=True,
        )

        features = sample.getInfo()["features"]

        # Subsample to requested amount if we got more
        if len(features) > n_samples:
            import random
            random.seed(42)
            features = random.sample(features, n_samples)

        locations = []
        for feature in features:
            coords = feature["geometry"]["coordinates"]
            locations.append({
                "lat": coords[1],
                "lon": coords[0],
                "stable": True,
            })

        return locations


def initialize_earth_engine():
    """Initialize Google Earth Engine (convenience function)."""
    try:
        ee.Initialize()
        print("✓ Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        print("  Run: earthengine authenticate")
        return False
