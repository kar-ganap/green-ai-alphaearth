"""
Debug Sentinel-2 feature extraction for a single sample.
"""
import pickle
import ee
from pathlib import Path
from src.utils import get_config

def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize()
        print("✓ Earth Engine initialized")
    except Exception as e:
        print(f"Need to authenticate: {e}")
        ee.Authenticate()
        ee.Initialize()
        print("✓ Earth Engine initialized after authentication")


def test_single_sample():
    """Test extraction for a single sample with verbose logging."""

    print("=" * 80)
    print("DEBUG SENTINEL-2 EXTRACTION")
    print("=" * 80)

    # Load a sample from the dataset
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'
    input_path = processed_dir / 'walk_dataset_scaled_phase1_multiscale.pkl'

    print(f"\nLoading sample from: {input_path}")
    with open(input_path, 'rb') as f:
        dataset = pickle.load(f)

    sample = dataset['data'][0]
    lat = sample['lat']
    lon = sample['lon']
    year = sample.get('year', 2021)
    date = f'{year}-06-01'

    print(f"\nTest sample:")
    print(f"  Lat: {lat}")
    print(f"  Lon: {lon}")
    print(f"  Date: {date}")
    print(f"  Label: {sample.get('label', 'unknown')}")

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    initialize_earth_engine()

    # Step 1: Get Sentinel-2 image
    print(f"\n{'='*80}")
    print("STEP 1: Get Sentinel-2 Image")
    print('='*80)

    from datetime import datetime, timedelta
    dt = datetime.strptime(date, '%Y-%m-%d')
    start_date = (dt - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (dt + timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"  Date range: {start_date} to {end_date}")

    point = ee.Geometry.Point([lon, lat])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )

    count = collection.size().getInfo()
    print(f"  Found {count} Sentinel-2 images")

    if count == 0:
        print("  ✗ No images available!")
        return

    image = collection.median()
    print(f"  ✓ Created median composite")

    # Check available bands
    band_names = image.bandNames().getInfo()
    print(f"  Available bands: {band_names}")

    # Step 2: Calculate spectral indices
    print(f"\n{'='*80}")
    print("STEP 2: Calculate Spectral Indices")
    print('='*80)

    try:
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        print("  ✓ NDVI calculated")

        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        print("  ✓ NBR calculated")

        ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
        print("  ✓ NDMI calculated")

        indices = {
            'NDVI': ndvi,
            'NBR': nbr,
            'NDMI': ndmi
        }
        print(f"  Created {len(indices)} spectral indices")

    except Exception as e:
        print(f"  ✗ Error calculating indices: {e}")
        indices = {}

    # Step 3: Calculate GLCM textures
    print(f"\n{'='*80}")
    print("STEP 3: Calculate GLCM Texture Features")
    print('='*80)

    try:
        # Select NIR band and scale to 8-bit integers for GLCM
        nir = image.select('B8').multiply(0.0255).toUint8()
        print("  ✓ Selected NIR band and scaled to 8-bit")

        glcm = nir.glcmTexture(size=3)
        print("  ✓ Calculated GLCM")

        # Check GLCM band names
        glcm_bands = glcm.bandNames().getInfo()
        print(f"  GLCM bands: {glcm_bands}")

        textures = {
            'contrast': glcm.select('B8_contrast'),
            'correlation': glcm.select('B8_corr'),
            'entropy': glcm.select('B8_ent'),
            'homogeneity': glcm.select('B8_idm'),
            'asm': glcm.select('B8_asm')
        }
        print(f"  Created {len(textures)} texture features")

    except Exception as e:
        print(f"  ✗ Error calculating textures: {e}")
        textures = {}

    # Step 4: Combine features
    print(f"\n{'='*80}")
    print("STEP 4: Combine Features")
    print('='*80)

    all_features = {**indices, **textures}
    print(f"  Total features to extract: {len(all_features)}")
    print(f"  Feature names: {list(all_features.keys())}")

    if len(all_features) == 0:
        print("  ✗ No features to extract!")
        return

    # Step 5: Extract statistics at point
    print(f"\n{'='*80}")
    print("STEP 5: Extract Statistics at Point")
    print('='*80)

    buffer_point = point.buffer(50)
    print(f"  Using 50m buffer around point")

    extracted_features = {}

    for name, img in all_features.items():
        print(f"\n  Processing {name}...")

        try:
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), '', True
                ).combine(
                    ee.Reducer.minMax(), '', True
                ),
                geometry=buffer_point,
                scale=10,
                maxPixels=1e9
            ).getInfo()

            print(f"    Stats keys: {list(stats.keys())}")
            print(f"    Stats values: {stats}")

            for stat_name, value in stats.items():
                if value is not None:
                    feature_name = f's2_{name}_{stat_name}'.replace('_mean', '_avg')
                    extracted_features[feature_name] = float(value)
                    print(f"      Added: {feature_name} = {value:.4f}")

        except Exception as e:
            print(f"    ✗ Error extracting {name}: {e}")

    # Final summary
    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print('='*80)
    print(f"  Total features extracted: {len(extracted_features)}")
    print(f"\n  Feature names:")
    for feat_name in sorted(extracted_features.keys()):
        print(f"    - {feat_name}")


if __name__ == '__main__':
    test_single_sample()
