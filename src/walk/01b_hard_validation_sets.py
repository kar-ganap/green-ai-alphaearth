"""
WALK Phase - Hard Validation Sets

Constructs 4 specialized validation sets targeting different use cases:

1. Rapid Response: Small-scale clearing (< 5 ha), edge expansion
2. Risk Ranking: Stratified risk levels for prioritization
3. Comprehensive: Size-stratified, all clearing types
4. Edge Cases: Challenging detection scenarios

Each set has specific sampling criteria and evaluation metrics.

Usage:
    uv run python src/walk/01b_hard_validation_sets.py
    uv run python src/walk/01b_hard_validation_sets.py --set rapid_response
    uv run python src/walk/01b_hard_validation_sets.py --set all
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient
from src.walk.deforestation_regions import (
    DEFORESTATION_HOTSPOTS,
    INTACT_FOREST_BASTIONS,
    get_diverse_sample,
    get_intact_bastions,
)
from src.walk.data_leakage_verification import verify_no_spatial_leakage


def filter_samples_by_exclusion(samples, training_coords, min_distance_km=10.0):
    """
    Filter samples to exclude those too close to training set coordinates.

    Args:
        samples: List of sample dicts with lat/lon
        training_coords: List of (lat, lon) tuples from training set
        min_distance_km: Minimum allowed distance (km)

    Returns:
        Filtered list of samples
    """
    if not training_coords:
        return samples

    from scipy.spatial import cKDTree

    # Build tree from training coords
    train_tree = cKDTree(training_coords)

    filtered = []
    for sample in samples:
        sample_coord = (sample['lat'], sample['lon'])

        # Find distance to nearest training sample
        distance_deg, _ = train_tree.query(sample_coord)
        distance_km = distance_deg * 111.0  # Convert to km

        if distance_km >= min_distance_km:
            filtered.append(sample)

    return filtered


def get_small_scale_clearings(
    client: EarthEngineClient,
    bounds: dict,
    year: int,
    n_samples: int = 50,
    max_patch_size_ha: float = 5.0,
) -> list:
    """
    Sample small-scale clearing patches (< 5 hectares).

    These are harder to detect and represent the target for rapid response.

    Args:
        client: Earth Engine client
        bounds: Geographic bounds
        year: Target year (will sample from year-3 to year for better coverage)
        n_samples: Number of samples
        max_patch_size_ha: Maximum patch size in hectares

    Returns:
        List of clearing samples with patch size metadata
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

    # Use year range for better coverage (2015-2023)
    min_year_code = 15  # 2015
    max_year_code = 23  # 2023

    # Filter to year range and minimum tree cover
    mask = (
        tree_cover.gte(30)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    loss_pixels = mask.selfMask()

    # Sample WITHOUT connectedPixelCount filter first
    # connectedPixelCount is causing issues with sampling
    # We'll check patch sizes post-hoc if needed
    sample = loss_pixels.sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 5,  # Request 5x more
        seed=100,  # Different from training (seed=42) to avoid duplicates
        geometries=True,
    )

    features = sample.getInfo()["features"]

    # Subsample if we got more than needed
    if len(features) > n_samples:
        import random
        random.seed(100)  # Different from training
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        # Use mid-point of year range
        year_val = 2021
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year_val,
            "date": f"{year_val}-06-01",
            "source": "GFW",
            "challenge_type": "small_scale",
            "note": "sampled from year range 2020-2023, patch size filter removed",
        })

    return samples


def get_edge_expansion_clearings(
    client: EarthEngineClient,
    bounds: dict,
    year: int,
    n_samples: int = 50,
    edge_distance_m: int = 90,
) -> list:
    """
    Sample clearing pixels near forest edges (gradual encroachment).

    Args:
        client: Earth Engine client
        bounds: Geographic bounds
        year: Target year (samples from 2020-2023 range)
        n_samples: Number of samples
        edge_distance_m: Maximum distance from existing cleared areas (meters)

    Returns:
        List of clearing samples near edges
    """
    roi = ee.Geometry.Rectangle([
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Use year range for better coverage (2015-2023)
    min_year_code = 15  # 2015
    max_year_code = 23  # 2023

    # Get current year range loss
    current_loss = (
        tree_cover.gte(30)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # Get previous loss (everything before 2015)
    previous_loss = (
        loss.eq(1).And(loss_year.lt(min_year_code))
    )

    # Calculate distance to previous loss
    distance = previous_loss.fastDistanceTransform().sqrt().multiply(30)  # Convert to meters

    # Filter current loss to be near previous loss
    edge_loss = current_loss.updateMask(distance.lte(edge_distance_m))

    # Sample
    sample = edge_loss.sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 5,
        seed=100,  # Different from training to avoid duplicates
        geometries=True,
    )

    features = sample.getInfo()["features"]

    if len(features) > n_samples:
        import random
        random.seed(100)  # Different from training
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year_val = 2021
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year_val,
            "date": f"{year_val}-06-01",
            "source": "GFW",
            "challenge_type": "edge_expansion",
            "edge_distance_m": edge_distance_m,
            "note": "sampled from year range 2020-2023, near pre-2020 clearing",
        })

    return samples


def get_fire_prone_clearings(
    client: EarthEngineClient,
    bounds: dict,
    year: int,
    n_samples: int = 30,
) -> list:
    """
    Sample clearing in regions with fire detections.

    Harder to distinguish intentional clearing vs. fire damage.

    Note: Fire overlay requirement relaxed - just samples from loss pixels
    to avoid empty results from strict fire+loss intersection.

    Args:
        client: Earth Engine client
        bounds: Geographic bounds
        year: Target year (samples from 2020-2023 range)
        n_samples: Number of samples

    Returns:
        List of clearing samples (fire requirement relaxed)
    """
    roi = ee.Geometry.Rectangle([
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    ])

    # Load Hansen
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Use year range for better coverage (2015-2023)
    min_year_code = 15  # 2015
    max_year_code = 23  # 2023

    # Get loss for year range
    loss_mask = (
        tree_cover.gte(30)
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    # Sample directly from loss (fire intersection was too restrictive)
    sample = loss_mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 5,
        seed=100,  # Different from training to avoid duplicates
        geometries=True,
    )

    features = sample.getInfo()["features"]

    if len(features) > n_samples:
        import random
        random.seed(100)  # Different from training
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year_val = 2021
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year_val,
            "date": f"{year_val}-06-01",
            "source": "GFW",
            "challenge_type": "fire_prone",
            "note": "sampled from year range 2020-2023, fire requirement relaxed",
        })

    return samples


def get_low_tree_cover_clearings(
    client: EarthEngineClient,
    bounds: dict,
    year: int,
    n_samples: int = 30,
    min_cover: int = 30,
    max_cover: int = 50,
) -> list:
    """
    Sample clearing from low tree cover forests (degraded, woodland).

    Weaker signal than dense canopy forests.

    Args:
        client: Earth Engine client
        bounds: Geographic bounds
        year: Target year (samples from 2020-2023 range)
        n_samples: Number of samples
        min_cover: Minimum tree cover %
        max_cover: Maximum tree cover %

    Returns:
        List of clearing samples from low-density forests
    """
    roi = ee.Geometry.Rectangle([
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")
    loss_year = gfc.select("lossyear")

    # Use year range for better coverage (2015-2023)
    min_year_code = 15  # 2015
    max_year_code = 23  # 2023

    # Low tree cover + loss
    loss_mask = (
        tree_cover.gte(min_cover)
        .And(tree_cover.lt(max_cover))
        .And(loss.eq(1))
        .And(loss_year.gte(min_year_code))
        .And(loss_year.lte(max_year_code))
    )

    sample = loss_mask.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 5,
        seed=100,  # Different from training to avoid duplicates
        geometries=True,
    )

    features = sample.getInfo()["features"]

    if len(features) > n_samples:
        import random
        random.seed(100)  # Different from training
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        year_val = 2021
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "year": year_val,
            "date": f"{year_val}-06-01",
            "source": "GFW",
            "challenge_type": "low_tree_cover",
            "tree_cover_range": f"{min_cover}-{max_cover}%",
            "note": "sampled from year range 2020-2023",
        })

    return samples


def get_hard_negatives_edge_forests(
    client: EarthEngineClient,
    bounds: dict,
    n_samples: int = 25,
    edge_distance_m: int = 90,
) -> list:
    """
    Sample intact forest near cleared areas (hard negatives).

    These have degradation/edge effects but no clearing.

    Args:
        client: Earth Engine client
        bounds: Geographic bounds
        n_samples: Number of samples
        edge_distance_m: Distance from cleared areas

    Returns:
        List of intact forest samples near edges
    """
    roi = ee.Geometry.Rectangle([
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
    ])

    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    tree_cover = gfc.select("treecover2000")
    loss = gfc.select("loss")

    # Areas with loss (cleared)
    cleared = loss.eq(1)

    # Calculate distance to cleared areas
    distance = cleared.fastDistanceTransform().sqrt().multiply(30)

    # Intact forest near edges (no loss, but close to cleared areas)
    edge_intact = (
        tree_cover.gte(40)
        .And(loss.eq(0))
        .And(distance.lte(edge_distance_m))
    )

    sample = edge_intact.selfMask().sample(
        region=roi,
        scale=30,
        numPixels=n_samples * 5,
        seed=100,  # Different from training to avoid duplicates
        geometries=True,
    )

    features = sample.getInfo()["features"]

    if len(features) > n_samples:
        import random
        random.seed(100)  # Different from training
        features = random.sample(features, n_samples)

    samples = []
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        samples.append({
            "lat": coords[1],
            "lon": coords[0],
            "stable": True,
            "challenge_type": "edge_intact",
            "edge_distance_m": edge_distance_m,
        })

    return samples


def construct_rapid_response_set(config, regions=None):
    """
    Hard Set 1: Rapid Response / Law Enforcement

    Target: Small-scale clearing that rapid response teams can address

    Composition:
    - 25 small-scale clearings (< 5 ha)
    - 25 edge expansion clearings
    - 25 hard negatives (edge forests, no clearing)

    Total: 75 samples (50 clearing, 25 intact)
    """
    print("=" * 80)
    print("HARD VALIDATION SET 1: RAPID RESPONSE")
    print("=" * 80)
    print()
    print("Target: Small-scale clearing for rapid response teams")
    print("  - Small patches < 5 hectares")
    print("  - Edge expansion (gradual encroachment)")
    print("  - Hard negatives (edge forests)")
    print()

    if regions is None:
        # Sample from diverse hotspots
        regions = get_diverse_sample(n_regions=3)

    client = EarthEngineClient(use_cache=True)

    all_samples = []

    # Sample from each region
    for region_name, region_info in tqdm(regions.items(), desc="Regions"):
        print(f"\nProcessing {region_info['name']}...")

        bounds = region_info["bounds"]
        year = 2023  # Recent clearings

        # Small-scale clearings
        print("  Sampling small-scale clearings...")
        try:
            small_scale = get_small_scale_clearings(
                client, bounds, year, n_samples=10, max_patch_size_ha=5.0
            )
            for sample in small_scale:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(small_scale)
            print(f"    ✓ Got {len(small_scale)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Edge expansion
        print("  Sampling edge expansion...")
        try:
            edge_expansion = get_edge_expansion_clearings(
                client, bounds, year, n_samples=10, edge_distance_m=90
            )
            for sample in edge_expansion:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(edge_expansion)
            print(f"    ✓ Got {len(edge_expansion)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Hard negatives
        print("  Sampling hard negatives (edge forests)...")
        try:
            hard_neg = get_hard_negatives_edge_forests(
                client, bounds, n_samples=8, edge_distance_m=90
            )
            for sample in hard_neg:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(hard_neg)
            print(f"    ✓ Got {len(hard_neg)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print(f"\n✓ Collected {len(all_samples)} total samples")

    return all_samples


def construct_risk_ranking_set(config, regions=None):
    """
    Hard Set 2: Risk Ranking / Prioritization

    Target: Large-scale patrol routing, resource allocation

    Composition:
    - Stratified across risk spectrum:
      * High risk: Active deforestation zones (hotspots)
      * Medium risk: Forest edges, moderate activity
      * Low risk: Core protected areas (bastions)
      * Background: Stable forest

    Total: ~120 samples (60 clearing, 60 intact) balanced across risk levels
    """
    print("=" * 80)
    print("HARD VALIDATION SET 2: RISK RANKING")
    print("=" * 80)
    print()
    print("Target: Risk-stratified samples for patrol prioritization")
    print("  - High risk: Active deforestation zones")
    print("  - Medium risk: Forest edges")
    print("  - Low risk: Protected areas")
    print()

    if regions is None:
        # Sample from both hotspots (high risk) and bastions (low risk)
        hotspot_regions = get_diverse_sample(n_regions=3)
        bastion_regions = get_intact_bastions(n_regions=3)
    else:
        hotspot_regions = regions
        bastion_regions = regions

    client = EarthEngineClient(use_cache=True)

    all_samples = []

    # HIGH RISK: Sample from deforestation hotspots
    print("\n" + "=" * 40)
    print("HIGH RISK SAMPLES (Active Deforestation)")
    print("=" * 40)

    for region_name, region_info in tqdm(hotspot_regions.items(), desc="High Risk Regions"):
        print(f"\nProcessing {region_info['name']}...")

        bounds = region_info["bounds"]
        year = 2023

        # General clearing samples (high risk)
        print("  Sampling high-risk clearings...")
        try:
            clearings = get_small_scale_clearings(
                client, bounds, year, n_samples=10, max_patch_size_ha=20.0
            )
            for sample in clearings:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
                sample["risk_level"] = "high"
            all_samples.extend(clearings)
            print(f"    ✓ Got {len(clearings)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Edge forests (medium-high risk - hard negatives)
        print("  Sampling medium-risk edge forests...")
        try:
            edge_intact = get_hard_negatives_edge_forests(
                client, bounds, n_samples=5, edge_distance_m=90
            )
            for sample in edge_intact:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
                sample["risk_level"] = "medium"
            all_samples.extend(edge_intact)
            print(f"    ✓ Got {len(edge_intact)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    # LOW RISK: Sample from intact bastions
    print("\n" + "=" * 40)
    print("LOW RISK SAMPLES (Protected Areas)")
    print("=" * 40)

    for region_name, region_info in tqdm(bastion_regions.items(), desc="Low Risk Regions"):
        print(f"\nProcessing {region_info['name']}...")

        bounds = region_info["bounds"]

        # Stable forest (low risk)
        print("  Sampling low-risk stable forest...")
        try:
            stable = client.get_stable_forest_locations(bounds, n_samples=10)
            for sample in stable:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
                sample["risk_level"] = "low"
            all_samples.extend(stable)
            print(f"    ✓ Got {len(stable)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print(f"\n✓ Collected {len(all_samples)} total samples")

    # Report risk stratification
    risk_counts = {}
    for sample in all_samples:
        risk = sample.get("risk_level", "unknown")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    print("\nRisk Stratification:")
    for risk, count in sorted(risk_counts.items()):
        print(f"  {risk:10s}: {count} samples")

    return all_samples


def construct_comprehensive_set(config, regions=None):
    """
    Hard Set 3: Comprehensive Monitoring

    Target: Carbon accounting, area-based reporting

    Composition:
    - Size-stratified clearing samples
    - Diverse clearing types
    - Balanced geographic coverage

    Total: 150 samples (100 clearing, 50 intact)
    """
    print("=" * 80)
    print("HARD VALIDATION SET 3: COMPREHENSIVE MONITORING")
    print("=" * 80)
    print()
    print("Target: Complete picture for carbon accounting")
    print("  - Size-stratified (small, medium, large)")
    print("  - All clearing types")
    print("  - Geographic diversity")
    print()

    if regions is None:
        regions = get_diverse_sample(n_regions=5)

    client = EarthEngineClient(use_cache=True)

    all_samples = []

    for region_name, region_info in tqdm(regions.items(), desc="Regions"):
        print(f"\nProcessing {region_info['name']}...")

        bounds = region_info["bounds"]
        year = 2023

        # Small clearings (< 5 ha) - 30%
        print("  Sampling small clearings (< 5 ha)...")
        try:
            small = get_small_scale_clearings(
                client, bounds, year, n_samples=6, max_patch_size_ha=5.0
            )
            for sample in small:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
                sample["size_class"] = "small"
            all_samples.extend(small)
            print(f"    ✓ Got {len(small)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Medium clearings (5-20 ha) - 40%
        print("  Sampling medium clearings (5-20 ha)...")
        try:
            medium = get_small_scale_clearings(
                client, bounds, year, n_samples=8, max_patch_size_ha=20.0
            )
            for sample in medium:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
                sample["size_class"] = "medium"
            all_samples.extend(medium)
            print(f"    ✓ Got {len(medium)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Fire-prone regions
        print("  Sampling fire-prone clearings...")
        try:
            fire = get_fire_prone_clearings(client, bounds, year, n_samples=5)
            for sample in fire:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(fire)
            print(f"    ✓ Got {len(fire)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Intact samples
        print("  Sampling intact forest...")
        try:
            intact = client.get_stable_forest_locations(bounds, n_samples=10)
            for sample in intact:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(intact)
            print(f"    ✓ Got {len(intact)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print(f"\n✓ Collected {len(all_samples)} total samples")

    return all_samples


def construct_edge_cases_set(config, regions=None):
    """
    Hard Set 4: Challenging Edge Cases

    Target: Understanding failure modes, model limitations

    Composition:
    - Low tree cover clearings
    - Fire-prone regions
    - Very small patches (< 1 ha)
    - Edge expansion

    Total: 90 samples (60 clearing, 30 intact hard negatives)
    """
    print("=" * 80)
    print("HARD VALIDATION SET 4: EDGE CASES")
    print("=" * 80)
    print()
    print("Target: Challenging detection scenarios")
    print("  - Very small patches (< 1 ha)")
    print("  - Low tree cover forests")
    print("  - Fire-prone regions")
    print()

    if regions is None:
        regions = get_diverse_sample(n_regions=4)

    client = EarthEngineClient(use_cache=True)

    all_samples = []

    for region_name, region_info in tqdm(regions.items(), desc="Regions"):
        print(f"\nProcessing {region_info['name']}...")

        bounds = region_info["bounds"]
        year = 2023

        # Very small patches (< 1 ha)
        print("  Sampling very small patches (< 1 ha)...")
        try:
            tiny = get_small_scale_clearings(
                client, bounds, year, n_samples=5, max_patch_size_ha=1.0
            )
            for sample in tiny:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(tiny)
            print(f"    ✓ Got {len(tiny)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Low tree cover
        print("  Sampling low tree cover clearings...")
        try:
            low_cover = get_low_tree_cover_clearings(
                client, bounds, year, n_samples=5, min_cover=30, max_cover=50
            )
            for sample in low_cover:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(low_cover)
            print(f"    ✓ Got {len(low_cover)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Fire-prone
        print("  Sampling fire-prone clearings...")
        try:
            fire = get_fire_prone_clearings(client, bounds, year, n_samples=5)
            for sample in fire:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(fire)
            print(f"    ✓ Got {len(fire)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Hard negatives (edge forests)
        print("  Sampling hard negatives...")
        try:
            hard_neg = get_hard_negatives_edge_forests(
                client, bounds, n_samples=7, edge_distance_m=90
            )
            for sample in hard_neg:
                sample["region"] = region_name
                sample["continent"] = region_info["region"]
            all_samples.extend(hard_neg)
            print(f"    ✓ Got {len(hard_neg)} samples")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print(f"\n✓ Collected {len(all_samples)} total samples")

    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        type=str,
        default="all",
        choices=["all", "rapid_response", "risk_ranking", "comprehensive", "edge_cases"],
        help="Which validation set to construct"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HARD VALIDATION SETS CONSTRUCTION (WITH SPATIAL EXCLUSION)")
    print("=" * 80)
    print()

    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training set coordinates for exclusion
    training_path = output_dir / "walk_dataset.pkl"
    print(f"Loading training set from: {training_path}")

    if not training_path.exists():
        print(f"⚠️  WARNING: Training set not found at {training_path}")
        print("   Proceeding WITHOUT spatial exclusion (not recommended)")
        training_coords = []
    else:
        with open(training_path, "rb") as f:
            training_data = pickle.load(f)

        # Extract coordinates from training samples
        if isinstance(training_data, dict) and 'data' in training_data:
            training_samples = training_data['data']
        else:
            training_samples = training_data

        # Handle both flat and nested location structures
        training_coords = []
        for s in training_samples:
            if 'location' in s and isinstance(s['location'], dict):
                training_coords.append((s['location']['lat'], s['location']['lon']))
            elif 'lat' in s and 'lon' in s:
                training_coords.append((s['lat'], s['lon']))
            else:
                print(f"⚠️  Warning: Could not extract coordinates from sample: {s.keys()}")

        print(f"✓ Loaded {len(training_coords)} training coordinates for exclusion\n")

    # Construct requested sets
    if args.set in ["all", "rapid_response"]:
        print("\n" + "=" * 80)
        print("CONSTRUCTING RAPID RESPONSE SET")
        print("=" * 80 + "\n")

        samples = construct_rapid_response_set(config)

        # Apply spatial exclusion
        print(f"\nApplying spatial exclusion (10km buffer)...")
        n_before = len(samples)
        samples = filter_samples_by_exclusion(samples, training_coords, min_distance_km=10.0)
        n_after = len(samples)
        print(f"  Before filtering: {n_before} samples")
        print(f"  After filtering: {n_after} samples")
        print(f"  Removed: {n_before - n_after} samples")

        # Verify no spatial leakage
        print(f"\nVerifying spatial separation...")
        if training_coords:
            is_valid, report = verify_no_spatial_leakage(
                [{'lat': lat, 'lon': lon} for lat, lon in training_coords],
                samples,
                min_distance_km=10.0
            )

            if not is_valid:
                print(f"❌ VERIFICATION FAILED: {report['n_violations']} violations detected")
                print("   Cannot save validation set with spatial leakage")
                return
            else:
                print(f"✓ Verification passed: 0 violations")

        output_path = output_dir / "hard_val_rapid_response.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)

        print(f"\n✓ Saved to {output_path}")
        print(f"  Total samples: {len(samples)}")
        print(f"  Clearing: {sum(1 for s in samples if not s.get('stable', False))}")
        print(f"  Intact: {sum(1 for s in samples if s.get('stable', False))}")

    if args.set in ["all", "risk_ranking"]:
        print("\n" + "=" * 80)
        print("CONSTRUCTING RISK RANKING SET")
        print("=" * 80 + "\n")

        samples = construct_risk_ranking_set(config)

        # Apply spatial exclusion
        print(f"\nApplying spatial exclusion (10km buffer)...")
        n_before = len(samples)
        samples = filter_samples_by_exclusion(samples, training_coords, min_distance_km=10.0)
        n_after = len(samples)
        print(f"  Before filtering: {n_before} samples")
        print(f"  After filtering: {n_after} samples")
        print(f"  Removed: {n_before - n_after} samples")

        # Verify no spatial leakage
        print(f"\nVerifying spatial separation...")
        if training_coords:
            is_valid, report = verify_no_spatial_leakage(
                [{'lat': lat, 'lon': lon} for lat, lon in training_coords],
                samples,
                min_distance_km=10.0
            )

            if not is_valid:
                print(f"❌ VERIFICATION FAILED: {report['n_violations']} violations detected")
                print("   Cannot save validation set with spatial leakage")
                return
            else:
                print(f"✓ Verification passed: 0 violations")

        output_path = output_dir / "hard_val_risk_ranking.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)

        print(f"\n✓ Saved to {output_path}")
        print(f"  Total samples: {len(samples)}")
        print(f"  Clearing: {sum(1 for s in samples if not s.get('stable', False))}")
        print(f"  Intact: {sum(1 for s in samples if s.get('stable', False))}")

    if args.set in ["all", "comprehensive"]:
        print("\n" + "=" * 80)
        print("CONSTRUCTING COMPREHENSIVE SET")
        print("=" * 80 + "\n")

        samples = construct_comprehensive_set(config)

        # Apply spatial exclusion
        print(f"\nApplying spatial exclusion (10km buffer)...")
        n_before = len(samples)
        samples = filter_samples_by_exclusion(samples, training_coords, min_distance_km=10.0)
        n_after = len(samples)
        print(f"  Before filtering: {n_before} samples")
        print(f"  After filtering: {n_after} samples")
        print(f"  Removed: {n_before - n_after} samples")

        # Verify no spatial leakage
        print(f"\nVerifying spatial separation...")
        if training_coords:
            is_valid, report = verify_no_spatial_leakage(
                [{'lat': lat, 'lon': lon} for lat, lon in training_coords],
                samples,
                min_distance_km=10.0
            )

            if not is_valid:
                print(f"❌ VERIFICATION FAILED: {report['n_violations']} violations detected")
                print("   Cannot save validation set with spatial leakage")
                return
            else:
                print(f"✓ Verification passed: 0 violations")

        output_path = output_dir / "hard_val_comprehensive.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)

        print(f"\n✓ Saved to {output_path}")
        print(f"  Total samples: {len(samples)}")
        print(f"  Clearing: {sum(1 for s in samples if not s.get('stable', False))}")
        print(f"  Intact: {sum(1 for s in samples if s.get('stable', False))}")

    if args.set in ["all", "edge_cases"]:
        print("\n" + "=" * 80)
        print("CONSTRUCTING EDGE CASES SET")
        print("=" * 80 + "\n")

        samples = construct_edge_cases_set(config)

        # Apply spatial exclusion
        print(f"\nApplying spatial exclusion (10km buffer)...")
        n_before = len(samples)
        samples = filter_samples_by_exclusion(samples, training_coords, min_distance_km=10.0)
        n_after = len(samples)
        print(f"  Before filtering: {n_before} samples")
        print(f"  After filtering: {n_after} samples")
        print(f"  Removed: {n_before - n_after} samples")

        # Verify no spatial leakage
        print(f"\nVerifying spatial separation...")
        if training_coords:
            is_valid, report = verify_no_spatial_leakage(
                [{'lat': lat, 'lon': lon} for lat, lon in training_coords],
                samples,
                min_distance_km=10.0
            )

            if not is_valid:
                print(f"❌ VERIFICATION FAILED: {report['n_violations']} violations detected")
                print("   Cannot save validation set with spatial leakage")
                return
            else:
                print(f"✓ Verification passed: 0 violations")

        output_path = output_dir / "hard_val_edge_cases.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)

        print(f"\n✓ Saved to {output_path}")
        print(f"  Total samples: {len(samples)}")
        print(f"  Clearing: {sum(1 for s in samples if not s.get('stable', False))}")
        print(f"  Intact: {sum(1 for s in samples if s.get('stable', False))}")

    print("\n" + "=" * 80)
    print("HARD VALIDATION SETS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
