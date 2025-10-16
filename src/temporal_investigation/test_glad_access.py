"""
Test script to diagnose GLAD dataset access and availability.
"""

import ee
from src.utils import get_config

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✓ Earth Engine initialized successfully")
except Exception as e:
    print(f"✗ Earth Engine initialization failed: {e}")
    exit(1)

print("\n" + "=" * 80)
print("GLAD DATASET DIAGNOSTICS")
print("=" * 80)

# Test 1: Try to access the GLAD dataset
print("\n1. Testing GLAD dataset access...")
try:
    # GLAD is an ImageCollection, not an Image
    glad_collection = ee.ImageCollection('projects/glad/alert/UpdResult')
    print("✓ Can access 'projects/glad/alert/UpdResult' as ImageCollection")

    # Get the first image to inspect bands
    first_image = glad_collection.first()
    band_names = first_image.bandNames().getInfo()
    print(f"\n  Available bands ({len(band_names)} total):")

    # Group by year
    by_year = {}
    for band in band_names:
        if 'conf' in band or 'alertDate' in band:
            # Extract year
            year_suffix = ''.join(filter(str.isdigit, band))
            if len(year_suffix) == 2:
                year = 2000 + int(year_suffix)
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(band)

    for year in sorted(by_year.keys()):
        bands = by_year[year]
        print(f"    {year}: {', '.join(bands)}")

except Exception as e:
    print(f"✗ Cannot access GLAD dataset: {e}")
    print("\nTrying alternative dataset IDs...")

    alternatives = [
        'projects/glad/alert/2019final',
        'projects/glad/alert/2020final',
        'projects/glad/GLADS2/alert',
    ]

    for alt_id in alternatives:
        try:
            test = ee.Image(alt_id)
            print(f"✓ Can access: {alt_id}")
            bands = test.bandNames().getInfo()
            print(f"  Bands: {bands[:10]}..." if len(bands) > 10 else f"  Bands: {bands}")
        except Exception as e2:
            print(f"✗ Cannot access: {alt_id}")

# Test 2: Try to sample GLAD at a known clearing location
print("\n2. Testing GLAD sampling at a clearing location...")

config = get_config()
test_lat, test_lon = -9.0, -57.0  # Pará, Brazil

try:
    # Load as ImageCollection and mosaic
    glad_collection = ee.ImageCollection('projects/glad/alert/UpdResult')
    point = ee.Geometry.Point([test_lon, test_lat])

    # Try years 2020, 2021, 2022
    for year in [2020, 2021, 2022]:
        year_suffix = str(year % 100)
        alert_band = f'alertDate{year_suffix}'
        conf_band = f'conf{year_suffix}'

        # Select bands and mosaic before sampling
        glad = glad_collection.select([alert_band, conf_band]).mosaic()
        sample = glad.sample(region=point, scale=30, numPixels=1)
        features = sample.getInfo()['features']

        if len(features) > 0:
            props = features[0]['properties']
            alert_val = props.get(alert_band)
            conf_val = props.get(conf_band)

            print(f"  Year {year}:")
            print(f"    {alert_band}: {alert_val}")
            print(f"    {conf_band}: {conf_val}")
        else:
            print(f"  Year {year}: No features returned")

except Exception as e:
    print(f"✗ Sampling failed: {e}")

# Test 3: Check if GLAD covers our study region
print("\n3. Checking GLAD coverage in study region...")

config = get_config()
bounds = config.study_region_bounds

roi = ee.Geometry.Rectangle([
    bounds["min_lon"],
    bounds["min_lat"],
    bounds["max_lon"],
    bounds["max_lat"],
])

try:
    # Load as ImageCollection and mosaic
    glad_collection = ee.ImageCollection('projects/glad/alert/UpdResult')
    glad = glad_collection.select(['conf20', 'alertDate20']).mosaic()

    # Check if any alerts exist in the region for 2020
    sample = glad.sample(
        region=roi,
        scale=1000,  # Coarse sampling
        numPixels=100,
        seed=42
    )

    features = sample.getInfo()['features']
    print(f"  Found {len(features)} sample points in region")

    if len(features) > 0:
        # Check how many have actual alerts
        with_alerts = 0
        for feat in features:
            props = feat['properties']
            if props.get('alertDate20') is not None and props.get('alertDate20') > 0:
                with_alerts += 1

        print(f"  Points with 2020 alerts: {with_alerts} / {len(features)}")

        if with_alerts > 0:
            # Show example
            for feat in features:
                props = feat['properties']
                if props.get('alertDate20') is not None and props.get('alertDate20') > 0:
                    print(f"\n  Example alert:")
                    print(f"    alertDate20: {props.get('alertDate20')}")
                    print(f"    conf20: {props.get('conf20')}")
                    print(f"    Location: {feat['geometry']['coordinates']}")
                    break

except Exception as e:
    print(f"✗ Regional sampling failed: {e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
