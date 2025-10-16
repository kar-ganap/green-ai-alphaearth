"""
Detailed diagnostic: Sample GLAD at specific clearing location and inspect all values.
"""

import ee
from src.utils import EarthEngineClient, get_config

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✓ Earth Engine initialized successfully\n")
except Exception as e:
    print(f"✗ Earth Engine initialization failed: {e}")
    exit(1)

print("=" * 80)
print("GLAD SAMPLING DIAGNOSTIC")
print("=" * 80)

# Get a real clearing location from our data
config = get_config()
client = EarthEngineClient(use_cache=True)

# Get one clearing location from 2020
main_bounds = config.study_region_bounds
mid_lon = (main_bounds["min_lon"] + main_bounds["max_lon"]) / 2
mid_lat = (main_bounds["min_lat"] + main_bounds["max_lat"]) / 2

sub_bounds = {
    "min_lon": main_bounds["min_lon"],
    "max_lon": mid_lon,
    "min_lat": mid_lat,
    "max_lat": main_bounds["max_lat"]
}

clearings = client.get_deforestation_labels(
    bounds=sub_bounds,
    year=2020,
    min_tree_cover=30,
)

if len(clearings) == 0:
    print("No clearings found!")
    exit(1)

# Take first clearing
clearing = clearings[0]
lat = clearing['lat']
lon = clearing['lon']
year = 2020

print(f"\nTest Location:")
print(f"  Lat: {lat}")
print(f"  Lon: {lon}")
print(f"  Year: {year}")
print(f"  Loss: {clearing.get('loss', 'N/A')}")
print(f"  Tree cover: {clearing.get('tree_cover', 'N/A')}")

# Test GLAD sampling at this location
print(f"\n{'='*80}")
print("TESTING GLAD SAMPLING")
print(f"{'='*80}")

point = ee.Geometry.Point([lon, lat])
year_suffix = str(year % 100)
dataset_id = f'projects/glad/alert/{year}final'

print(f"\nDataset: {dataset_id}")
print(f"Bands: alertDate{year_suffix}, conf{year_suffix}")

try:
    glad_collection = ee.ImageCollection(dataset_id)

    print(f"\n✓ Loaded ImageCollection")
    print(f"  Collection size: {glad_collection.size().getInfo()} images")

    # Get first image to inspect
    first = glad_collection.first()
    all_bands = first.bandNames().getInfo()
    print(f"  Available bands: {all_bands}")

    # Try different sampling scales
    for scale in [30, 100, 500]:
        print(f"\n--- Sampling at scale {scale}m ---")

        # Sample ALL bands (not just alert bands) to see what's there
        glad = glad_collection.mosaic()
        sample = glad.sample(region=point, scale=scale, numPixels=1)
        features = sample.getInfo()['features']

        if len(features) == 0:
            print(f"  No features returned at scale {scale}m")
            continue

        if len(features[0]['properties']) == 0:
            print(f"  Empty properties at scale {scale}m")
            continue

        props = features[0]['properties']
        print(f"  ✓ Got {len(props)} properties:")

        # Print all property values
        for key, value in sorted(props.items()):
            print(f"    {key}: {value}")

        # Specifically check alert bands
        alert_val = props.get(f'alertDate{year_suffix}')
        conf_val = props.get(f'conf{year_suffix}')

        print(f"\n  Target bands:")
        print(f"    alertDate{year_suffix}: {alert_val}")
        print(f"    conf{year_suffix}: {conf_val}")

        if alert_val is not None and alert_val > 0:
            from datetime import datetime, timedelta
            # Correct interpretation: Julian day of year
            alert_date = datetime(year, 1, 1) + timedelta(days=int(alert_val) - 1)
            print(f"    -> Alert date (corrected): {alert_date.strftime('%Y-%m-%d')}")
            print(f"       Quarter: Q{(alert_date.month - 1) // 3 + 1}")

        # Only test first scale if we got data
        if alert_val is not None:
            break

except Exception as e:
    print(f"\n✗ Error: {e}")

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
