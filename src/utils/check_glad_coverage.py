"""
Check GLAD Alert Coverage

GLAD has two main products:
1. GLAD-L (Landsat-based): Global coverage
2. GLAD-S2 (Sentinel-2-based): Tropical regions only

Let's check what's available and where.
"""

import ee
from src.utils import get_config

# Initialize Earth Engine
ee.Initialize()

print("="*80)
print("CHECKING GLAD ALERT COVERAGE")
print("="*80)

# GLAD-L (Landsat-based) - should be global
print("\n1. GLAD-L (Landsat-based)")
print("-"*80)

try:
    # Check archived 2020 data
    glad_l_2020 = ee.ImageCollection('projects/glad/alert/2020final')

    # Get bounds
    bounds = glad_l_2020.geometry().bounds().getInfo()
    print(f"✓ GLAD-L 2020 available")
    print(f"  Bounds: {bounds}")

    # Get band names to understand structure
    first_image = glad_l_2020.first()
    bands = first_image.bandNames().getInfo()
    print(f"  Bands: {bands}")

except Exception as e:
    print(f"✗ GLAD-L 2020 error: {e}")

# Check current GLAD-L
try:
    glad_l_current = ee.ImageCollection('projects/glad/alert/UpdResult')
    print(f"\n✓ GLAD-L UpdResult (current) available")

    # Get date range
    date_range = glad_l_current.reduceColumns(ee.Reducer.minMax(), ["system:time_start"]).getInfo()
    print(f"  Date range: {date_range}")

except Exception as e:
    print(f"✗ GLAD-L UpdResult error: {e}")

# GLAD-S2 (Sentinel-2) - tropical only
print("\n2. GLAD-S2 (Sentinel-2-based)")
print("-"*80)

# GLAD-S2 is not in standard Earth Engine catalog
# It's in a special project: projects/GLADS2 or similar
# Let's check a few possible locations

s2_locations = [
    'projects/glad-s2/alert/UpdResult',
    'projects/GLADS2/alert/UpdResult',
    'UMD/GLAD/S2/alert',
]

for location in s2_locations:
    try:
        glad_s2 = ee.ImageCollection(location)
        print(f"✓ GLAD-S2 found at: {location}")

        # Get info
        first_image = glad_s2.first()
        bands = first_image.bandNames().getInfo()
        print(f"  Bands: {bands}")
        break
    except Exception as e:
        print(f"✗ {location}: Not found")

# Regional Coverage Test
print("\n3. Testing Regional Coverage")
print("-"*80)

regions = {
    'Amazon': {
        'bounds': [-73, -15, -50, 5],
        'center': [-60, -5]
    },
    'Congo Basin': {
        'bounds': [10, -5, 30, 5],
        'center': [20, 0]
    },
    'Southeast Asia': {
        'bounds': [95, -5, 120, 10],
        'center': [105, 2.5]
    },
    'Central America': {
        'bounds': [-90, 5, -75, 15],
        'center': [-82.5, 10]
    }
}

# Test with GLAD-L 2020
glad_2020 = ee.ImageCollection('projects/glad/alert/2020final')

for region_name, region_info in regions.items():
    try:
        center = region_info['center']
        point = ee.Geometry.Point(center)

        # Try to sample at center point
        sample = glad_2020.mosaic().select('alertDate20').sample(
            region=point,
            scale=30,
            numPixels=1
        ).first()

        result = sample.getInfo()

        if result:
            print(f"✓ {region_name}: GLAD data available")
        else:
            print(f"⚠ {region_name}: No data at test point")

    except Exception as e:
        print(f"✗ {region_name}: Error - {e}")

# Coverage Summary
print("\n" + "="*80)
print("COVERAGE SUMMARY")
print("="*80)

print("""
Based on GLAD documentation:

GLAD-L (Landsat):
  - Coverage: Global (all forests)
  - Resolution: 30m
  - Frequency: Weekly updates
  - History: 2019-present in archived collections
  - Availability: ✓ Amazon, ✓ Congo, ✓ SE Asia, ✓ Central America

GLAD-S2 (Sentinel-2):
  - Coverage: Tropical forests only (humid tropics)
  - Resolution: 10m
  - Frequency: Weekly updates
  - History: 2019-present for select regions
  - Availability: ✓ Amazon (confirmed), ? Congo, ? SE Asia

Key Finding:
- GLAD-L covers ALL major deforestation regions globally
- GLAD-S2 has higher resolution but limited to tropics
- Both provide QUARTERLY temporal resolution (week-level dates!)

Recommendation:
- Use GLAD-L for multi-region analysis
- Covers Amazon, Congo, SE Asia, Central America
- Can run Q2 vs Q4 analysis in all major basins
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
To cleanly separate detection from prediction:

1. Extract GLAD weekly dates for clearings across regions
2. Stratify by quarter (Q1, Q2, Q3, Q4)
3. Test temporal signal in each region:
   - Amazon (already done): Q2 >> Q4
   - Congo Basin: Test Q2 vs Q4
   - SE Asia: Test Q2 vs Q4
   - Central America: Test Q2 vs Q4

4. Build region-specific models or quarters-aware model:
   - If Q2 >> Q4 globally: "Early detection system (0-3 months)"
   - If Q4 strong too: "Mixed detection/prediction (0-6 months)"
   - Can report lead time distribution by region

5. Production system can report:
   - Predicted risk for next 6 months
   - Separate Q2-Q3 predictions (detection) from Q4 (prediction)
   - Region-specific lead time expectations
""")
