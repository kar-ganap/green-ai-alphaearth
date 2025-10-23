"""
Test MODIS Fire Product Directly

Tests MODIS MCD64A1 queries for sample locations to diagnose
why we're getting 0 fire detections.
"""

import ee
from datetime import datetime, timedelta

from src.utils.earth_engine import EarthEngineClient


def test_modis_fire_for_location(lat, lon, year):
    """Test MODIS fire query for a specific location."""

    print(f"\n{'='*80}")
    print(f"Testing MODIS for location: ({lat:.4f}, {lon:.4f}), Year: {year}")
    print("=" * 80)

    # Initialize EE
    client = EarthEngineClient(use_cache=False)

    # Define dates (same logic as extraction script)
    clearing_date = datetime(year, 9, 1)
    start_date = clearing_date - timedelta(days=6 * 30)
    end_date = clearing_date + timedelta(days=6 * 30)

    print(f"\nTime window:")
    print(f"  Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"  End: {end_date.strftime('%Y-%m-%d')}")

    # Create point and buffer
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(1000)  # 1km buffer

    print(f"\nGeometry: 1km buffer around point")

    # Query MODIS MCD64A1
    print(f"\n{'='*80}")
    print("TESTING: MODIS/006/MCD64A1 (Burned Area)")
    print("=" * 80)

    try:
        modis_ba = ee.ImageCollection('MODIS/006/MCD64A1') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filterBounds(buffer)

        n_images = modis_ba.size().getInfo()
        print(f"  Images found: {n_images}")

        if n_images > 0:
            # Get image dates
            image_dates = modis_ba.aggregate_array('system:time_start').getInfo()
            print(f"\n  Image dates:")
            for ts in image_dates[:5]:  # Show first 5
                dt = datetime.fromtimestamp(ts / 1000)
                print(f"    {dt.strftime('%Y-%m-%d')}")
            if len(image_dates) > 5:
                print(f"    ... and {len(image_dates) - 5} more")

            # Try to get burn data
            burn_composite = modis_ba.select('BurnDate').max()

            # Sample at point
            sample = burn_composite.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=point,
                scale=500,
                maxPixels=1e9
            )

            burn_date = sample.get('BurnDate').getInfo()
            print(f"\n  BurnDate at point: {burn_date}")

            # Count burns in buffer
            burn_mask = burn_composite.gt(0)
            burn_count = burn_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=buffer,
                scale=1000,
                maxPixels=1e9
            )

            count = burn_count.get('BurnDate')
            count_val = count.getInfo() if count else 0
            print(f"  Burn pixel count in buffer: {count_val}")

    except Exception as e:
        print(f"  ✗ Error querying MCD64A1: {e}")

    # Try alternative: MODIS Terra Thermal Anomalies (MOD14A1)
    print(f"\n{'='*80}")
    print("TESTING: MODIS/006/MOD14A1 (Terra Thermal Anomalies)")
    print("=" * 80)

    try:
        mod14a1 = ee.ImageCollection('MODIS/006/MOD14A1') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filterBounds(buffer)

        n_images = mod14a1.size().getInfo()
        print(f"  Images found: {n_images}")

        if n_images > 0:
            # Get MaxFRP (Maximum Fire Radiative Power)
            max_frp = mod14a1.select('MaxFRP').max()

            sample = max_frp.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=point,
                scale=1000,
                maxPixels=1e9
            )

            frp = sample.get('MaxFRP').getInfo()
            print(f"  Max FRP at point: {frp}")

    except Exception as e:
        print(f"  ✗ Error querying MOD14A1: {e}")

    # Try VIIRS
    print(f"\n{'='*80}")
    print("TESTING: FIRMS (VIIRS Active Fire)")
    print("=" * 80)

    try:
        firms = ee.ImageCollection('FIRMS') \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filterBounds(buffer)

        n_images = firms.size().getInfo()
        print(f"  Images found: {n_images}")

        if n_images > 0:
            # Get confidence
            max_conf = firms.select('T21').max()  # Brightness temperature

            sample = max_conf.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=point,
                scale=375,
                maxPixels=1e9
            )

            t21 = sample.get('T21').getInfo()
            print(f"  Max brightness temp: {t21}")

    except Exception as e:
        print(f"  ✗ Error querying FIRMS: {e}")


def main():
    print("=" * 80)
    print("MODIS FIRE PRODUCT TESTING")
    print("=" * 80)

    # Test a few sample locations
    test_locations = [
        # Amazon samples from edge cases
        (-3.0780, -54.2543, 2021),
        (-6.6170, -53.5217, 2021),
        (-0.9541, 21.9598, 2021),  # Congo

        # Sample that showed 'fire_before_only' pattern
        (-5.4367, -51.1579, 2021),
    ]

    for lat, lon, year in test_locations:
        test_modis_fire_for_location(lat, lon, year)

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    print("\nIf all queries return 0 images or 0 burns:")
    print("  → MODIS MCD64A1 may not detect fires at our sample locations")
    print("  → Fire events may be outside our time window")
    print("  → 'Fire-prone' may refer to regional risk, not actual fires")
    print("\nIf queries return data but extraction script got 0:")
    print("  → Bug in extraction logic")
    print("\nRecommendations:")
    print("  1. If no fire data exists → Skip fire classifier, move to multi-scale")
    print("  2. If fire data exists → Debug extraction script")
    print("  3. Consider using spectral indices (NBR) instead")


if __name__ == "__main__":
    main()
