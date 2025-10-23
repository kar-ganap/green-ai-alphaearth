#!/usr/bin/env python3
"""
Sample Collection Helpers for Temporal Validation

Provides utilities for collecting clearing and intact samples for specific years,
with spatial separation from existing training/validation sets.
"""

import ee
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial import cKDTree


def sample_fires_for_clearing_validation(
    ee_client,
    target_year: int,
    n_samples: int = 50,
    min_confidence: int = 80,
    min_frp: float = 10.0,
    sample_strategy: str = 'stratified'
) -> List[Dict]:
    """
    Sample fire-cleared areas for a specific year using FIRMS data.

    Args:
        ee_client: EarthEngineClient instance
        target_year: Year to sample clearings from
        n_samples: Number of clearing samples to collect
        min_confidence: Minimum FIRMS confidence (0-100)
        min_frp: Minimum Fire Radiative Power
        sample_strategy: 'stratified' or 'random'

    Returns:
        List of clearing samples with lat, lon, year
    """
    # Load Hansen GFC to get forest loss
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Convert target year to Hansen year code (e.g., 2022 -> 22)
    year_code = target_year - 2000

    # Create mask for forest loss in target year
    mask = (
        tree_cover.gte(30)
        .And(loss.eq(1))
        .And(loss_year.eq(year_code))
    )

    loss_pixels = mask.selfMask()

    # Define sampling regions (tropical forests)
    tropical_regions = [
        # Amazon
        {"min_lon": -75, "max_lon": -45, "min_lat": -15, "max_lat": 5},
        # Congo Basin
        {"min_lon": 10, "max_lon": 30, "min_lat": -10, "max_lat": 5},
        # SE Asia
        {"min_lon": 95, "max_lon": 120, "min_lat": -5, "max_lat": 10},
    ]

    all_samples = []
    samples_per_region = n_samples // len(tropical_regions)

    for i, region in enumerate(tropical_regions):
        region_name = ['Amazon', 'Congo Basin', 'SE Asia'][i]
        roi = ee.Geometry.Rectangle([
            region["min_lon"],
            region["min_lat"],
            region["max_lon"],
            region["max_lat"],
        ])

        # Sample clearing pixels
        try:
            # Increase samples requested to account for potential filtering
            sample = loss_pixels.sample(
                region=roi,
                scale=30,
                numPixels=samples_per_region * 5,  # Request 5x more
                seed=target_year,  # Use year as seed for reproducibility
                geometries=True,
            )

            features = sample.getInfo()["features"]
            print(f"    {region_name}: Found {len(features)} clearings, taking {min(len(features), samples_per_region)}")

            # Convert to sample dicts
            for feature in features[:samples_per_region]:
                coords = feature["geometry"]["coordinates"]
                all_samples.append({
                    "lat": coords[1],
                    "lon": coords[0],
                    "year": target_year,
                })
        except Exception as e:
            print(f"  Warning: Failed to sample from {region_name}: {e}")
            continue

    print(f"  Collected {len(all_samples)} clearing samples for {target_year}")

    # If we got very few samples, try a wider year range
    if len(all_samples) < n_samples // 2:
        print(f"  Warning: Only got {len(all_samples)} samples, trying wider year range...")
        # Try year-1 to year+1
        for offset in [-1, 1]:
            alt_year_code = year_code + offset
            if alt_year_code < 0 or alt_year_code > 24:
                continue

            mask_alt = (
                tree_cover.gte(30)
                .And(loss.eq(1))
                .And(loss_year.eq(alt_year_code))
            )
            loss_pixels_alt = mask_alt.selfMask()

            needed = n_samples - len(all_samples)
            if needed <= 0:
                break

            samples_per_region_alt = needed // len(tropical_regions)

            for i, region in enumerate(tropical_regions):
                region_name = ['Amazon', 'Congo Basin', 'SE Asia'][i]
                roi = ee.Geometry.Rectangle([
                    region["min_lon"],
                    region["min_lat"],
                    region["max_lon"],
                    region["max_lat"],
                ])

                try:
                    sample = loss_pixels_alt.sample(
                        region=roi,
                        scale=30,
                        numPixels=samples_per_region_alt * 3,
                        seed=target_year + offset,
                        geometries=True,
                    )

                    features = sample.getInfo()["features"]
                    print(f"    {region_name} ({2000 + alt_year_code}): Found {len(features)} clearings")

                    for feature in features[:samples_per_region_alt]:
                        coords = feature["geometry"]["coordinates"]
                        all_samples.append({
                            "lat": coords[1],
                            "lon": coords[0],
                            "year": 2000 + alt_year_code,  # Use actual year
                        })
                except Exception as e:
                    continue

        print(f"  Final total: {len(all_samples)} clearing samples")

    return all_samples


def sample_intact_forest(
    ee_client,
    reference_year: int,
    n_samples: int = 50,
    sample_strategy: str = 'representative'
) -> List[Dict]:
    """
    Sample intact forest areas (no clearing through reference year).

    Args:
        ee_client: EarthEngineClient instance
        reference_year: Sample forests intact through this year
        n_samples: Number of intact samples to collect
        sample_strategy: 'representative' or 'random'

    Returns:
        List of intact samples with lat, lon, year
    """
    # Load Hansen GFC
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    year_code = reference_year - 2000

    # Mask: Forest in 2000, no loss through reference year
    mask = tree_cover.gte(30).And(loss.eq(0))

    intact_pixels = mask.selfMask()

    # Same tropical regions as clearing samples
    tropical_regions = [
        {"min_lon": -75, "max_lon": -45, "min_lat": -15, "max_lat": 5},
        {"min_lon": 10, "max_lon": 30, "min_lat": -10, "max_lat": 5},
        {"min_lon": 95, "max_lon": 120, "min_lat": -5, "max_lat": 10},
    ]

    all_samples = []
    samples_per_region = n_samples // len(tropical_regions)

    for region in tropical_regions:
        roi = ee.Geometry.Rectangle([
            region["min_lon"],
            region["min_lat"],
            region["max_lon"],
            region["max_lat"],
        ])

        try:
            sample = intact_pixels.sample(
                region=roi,
                scale=30,
                numPixels=samples_per_region * 3,  # Request extra
                seed=reference_year + 1000,  # Different seed from clearing
                geometries=True,
            )

            features = sample.getInfo()["features"]

            # Convert to sample dicts
            for feature in features[:samples_per_region]:
                coords = feature["geometry"]["coordinates"]
                all_samples.append({
                    "lat": coords[1],
                    "lon": coords[0],
                    "year": reference_year,
                    "stable": True,
                })
        except Exception as e:
            print(f"  Warning: Failed to sample intact from region {region}: {e}")
            continue

    print(f"  Collected {len(all_samples)} intact samples for {reference_year}")
    return all_samples


def check_spatial_separation(
    samples: List[Dict],
    existing_coords: np.ndarray,
    min_distance_km: float = 10.0
) -> Tuple[List[Dict], int]:
    """
    Filter samples to ensure spatial separation from existing locations.

    Args:
        samples: New samples to filter
        existing_coords: Array of (lat, lon) from existing training/validation
        min_distance_km: Minimum required distance

    Returns:
        (filtered_samples, n_removed)
    """
    if len(existing_coords) == 0:
        return samples, 0

    # Build KDTree from existing coordinates
    tree = cKDTree(existing_coords)

    filtered = []
    removed = 0

    for sample in samples:
        coord = (sample['lat'], sample['lon'])

        # Find nearest existing point
        distance_deg, _ = tree.query(coord)
        distance_km = distance_deg * 111.0  # Convert degrees to km

        if distance_km >= min_distance_km:
            filtered.append(sample)
        else:
            removed += 1

    return filtered, removed
