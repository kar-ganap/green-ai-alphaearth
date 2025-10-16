"""
Test script to inspect archived GLAD datasets for 2020-2022.
"""

import ee
from src.utils import get_config

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✓ Earth Engine initialized successfully\n")
except Exception as e:
    print(f"✗ Earth Engine initialization failed: {e}")
    exit(1)

print("=" * 80)
print("ARCHIVED GLAD DATASET INSPECTION")
print("=" * 80)

# Test different possible dataset IDs for archived years
test_datasets = [
    # Year-specific archived datasets
    'projects/glad/alert/2020final',
    'projects/glad/alert/2021final',
    'projects/glad/alert/2022final',
    'projects/glad/alert/2023final',

    # Try without "final" suffix
    'projects/glad/alert/2020',
    'projects/glad/alert/2021',
    'projects/glad/alert/2022',

    # Try ImageCollection paths
    'projects/glad/UpdResult',
]

for dataset_id in test_datasets:
    print(f"\nTesting: {dataset_id}")
    print("-" * 80)

    # Try as Image
    try:
        asset = ee.Image(dataset_id)
        bands = asset.bandNames().getInfo()
        print(f"✓ Accessible as Image")
        print(f"  Bands ({len(bands)}): {bands}")
        continue
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"✗ Not found as Image")
        elif "is not an Image" in error_msg:
            print(f"  Not an Image (might be ImageCollection)")
        else:
            print(f"  Error as Image: {error_msg}")

    # Try as ImageCollection
    try:
        asset = ee.ImageCollection(dataset_id)
        # Get info about the collection
        size = asset.size().getInfo()
        print(f"✓ Accessible as ImageCollection")
        print(f"  Size: {size} images")

        if size > 0:
            # Get first image bands
            first = asset.first()
            bands = first.bandNames().getInfo()
            print(f"  First image bands ({len(bands)}): {bands}")
        continue
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"✗ Not found as ImageCollection")
        else:
            print(f"  Error as ImageCollection: {error_msg}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
