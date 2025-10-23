"""
WALK Phase - Feature Extraction for Hard Validation Sets

Extracts AlphaEarth embeddings and temporal features for hard validation sets.

Usage:
    uv run python src/walk/01c_extract_features_for_hard_sets.py --set rapid_response
    uv run python src/walk/01c_extract_features_for_hard_sets.py --set all
"""

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def compute_temporal_features(embeddings):
    """
    Compute temporal features from dual-year embeddings.

    Computes features for THREE temporal views:
    1. Baseline (Y-1): Landscape susceptibility
    2. Current (Y): Recent state
    3. Delta (Y - Y-1): Recent change (KEY for causal detection!)

    Args:
        embeddings: Dict with 'baseline', 'current', 'delta' keys

    Returns:
        Dict with computed features for all temporal views
    """
    def compute_features_from_quarters(quarters_dict):
        """Helper to compute features from Q1-Q4 embeddings."""
        timepoint_names = ['Q1', 'Q2', 'Q3', 'Q4']
        baseline_emb = quarters_dict['Q1']

        # Distances from Q1 baseline
        distances = {}
        for name in timepoint_names:
            dist = np.linalg.norm(quarters_dict[name] - baseline_emb)
            distances[name] = float(dist)

        # Velocities (change between consecutive quarters)
        velocities = {}
        for i in range(len(timepoint_names) - 1):
            t1 = timepoint_names[i]
            t2 = timepoint_names[i + 1]
            velocity = np.linalg.norm(quarters_dict[t2] - quarters_dict[t1])
            velocities[f'{t1}_{t2}'] = float(velocity)

        # Accelerations (velocity change)
        accelerations = {}
        velocity_values = list(velocities.values())
        velocity_names = list(velocities.keys())
        for i in range(len(velocity_values) - 1):
            acceleration = velocity_values[i + 1] - velocity_values[i]
            # Simplify name: Q1_Q2_Q2_Q3 -> Q1_Q2_Q3
            parts_1 = velocity_names[i].split('_')
            parts_2 = velocity_names[i + 1].split('_')
            simplified_name = f'{parts_1[0]}_{parts_1[1]}_{parts_2[1]}'
            accelerations[simplified_name] = float(acceleration)

        # Trend consistency
        distance_values = [distances[name] for name in timepoint_names]
        diffs = np.diff(distance_values)
        trend_consistency = float(np.mean(diffs > 0))

        return {
            'distances': distances,
            'velocities': velocities,
            'accelerations': accelerations,
            'trend_consistency': trend_consistency,
        }

    # Compute features for baseline (Y-1)
    baseline_features = compute_features_from_quarters(embeddings['baseline'])

    # Compute features for current (Y)
    current_features = compute_features_from_quarters(embeddings['current'])

    # Compute delta-specific features (magnitude of change)
    delta_magnitudes = {
        label: float(np.linalg.norm(embeddings['delta'][label]))
        for label in ['Q1', 'Q2', 'Q3', 'Q4']
    }

    delta_features = {
        'delta_magnitudes': delta_magnitudes,
        'mean_delta_magnitude': float(np.mean(list(delta_magnitudes.values()))),
        'max_delta_magnitude': float(np.max(list(delta_magnitudes.values()))),
        'delta_trend': float(delta_magnitudes['Q4'] - delta_magnitudes['Q1']),
    }

    return {
        'baseline': baseline_features,  # Y-1 features (landscape susceptibility)
        'current': current_features,    # Y features (recent state)
        'delta': delta_features,        # Change features (KEY!)
    }


def extract_features_for_sample(client, sample):
    """
    Extract features for a single sample.

    Args:
        client: EarthEngineClient
        sample: Sample dict with lat, lon, year, date

    Returns:
        Updated sample dict with 'features' and 'embeddings' keys
    """
    lat = sample['lat']
    lon = sample['lon']
    year = sample.get('year', 2021)

    # DUAL-YEAR TEMPORAL APPROACH (baseline + current + delta)
    # Extract from BOTH Y-1 and Y for temporal control experiment
    #
    # Year Y-1 (Baseline - guaranteed clean):
    # Q1: Mar Y-1, Q2: Jun Y-1, Q3: Sep Y-1, Q4: Dec Y-1
    #
    # Year Y (Current - may include clearing):
    # Q1: Mar Y, Q2: Jun Y, Q3: Sep Y, Q4: Dec Y
    #
    # Delta (Y - Y-1): Recent change signal (key for causal detection)

    baseline_timepoints = {
        'Q1': f'{year - 1}-03-01',
        'Q2': f'{year - 1}-06-01',
        'Q3': f'{year - 1}-09-01',
        'Q4': f'{year - 1}-12-01',
    }

    current_timepoints = {
        'Q1': f'{year}-03-01',
        'Q2': f'{year}-06-01',
        'Q3': f'{year}-09-01',
        'Q4': f'{year}-12-01',
    }

    # Fetch baseline embeddings (Y-1)
    baseline = {}
    for name, date in baseline_timepoints.items():
        try:
            emb = client.get_embedding(lat, lon, date)
            baseline[name] = emb
        except Exception as e:
            print(f"    ✗ Failed to get baseline embedding for {name} ({date}): {e}")
            return None

    # Fetch current embeddings (Y)
    current = {}
    for name, date in current_timepoints.items():
        try:
            emb = client.get_embedding(lat, lon, date)
            current[name] = emb
        except Exception as e:
            print(f"    ✗ Failed to get current embedding for {name} ({date}): {e}")
            return None

    # Compute delta (recent change)
    delta = {
        label: current[label] - baseline[label]
        for label in ['Q1', 'Q2', 'Q3', 'Q4']
    }

    embeddings = {
        'baseline': baseline,
        'current': current,
        'delta': delta,
        'clearing': None,  # Not needed for validation sets
    }

    # Compute features
    features = compute_temporal_features(embeddings)

    # Add features and embeddings to sample
    sample['features'] = features
    sample['embeddings'] = embeddings

    # Add label (0 = intact, 1 = clearing)
    if 'label' not in sample:
        sample['label'] = 0 if sample.get('stable', False) else 1

    return sample


def extract_features_for_set(set_name, config):
    """
    Extract features for a hard validation set.

    Args:
        set_name: Name of validation set (e.g., 'rapid_response')
        config: Config object

    Returns:
        Enriched samples with features
    """
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    input_file = processed_dir / f"hard_val_{set_name}.pkl"
    output_file = processed_dir / f"hard_val_{set_name}_features.pkl"

    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return None

    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES FOR: {set_name}")
    print(f"{'='*80}\n")

    # Load samples
    with open(input_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    n_clearing = sum(1 for s in samples if not s.get('stable', False))
    n_intact = len(samples) - n_clearing
    print(f"  Clearing: {n_clearing}")
    print(f"  Intact: {n_intact}\n")

    # Initialize Earth Engine client
    client = EarthEngineClient(use_cache=True)

    # Extract features for each sample
    enriched_samples = []
    failed_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Extracting features")):
        try:
            enriched_sample = extract_features_for_sample(client, sample)
            if enriched_sample is not None:
                enriched_samples.append(enriched_sample)
            else:
                failed_samples.append(i)
        except Exception as e:
            print(f"\n  ✗ Failed on sample {i}: {e}")
            failed_samples.append(i)

    print(f"\n✓ Extracted features for {len(enriched_samples)}/{len(samples)} samples")
    if failed_samples:
        print(f"  ✗ Failed: {len(failed_samples)} samples")
        print(f"    Indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")

    # Save enriched dataset
    with open(output_file, 'wb') as f:
        pickle.dump(enriched_samples, f)

    print(f"\n✓ Saved to {output_file}")

    return enriched_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set',
        type=str,
        default='rapid_response',
        choices=['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases', 'all'],
        help='Which validation set to extract features for'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE EXTRACTION FOR HARD VALIDATION SETS")
    print("=" * 80)

    config = get_config()

    if args.set == 'all':
        sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']
    else:
        sets = [args.set]

    for set_name in sets:
        extract_features_for_set(set_name, config)

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext: Run evaluation with:")
    print("  uv run python src/walk/03_evaluate_all_sets.py")


if __name__ == "__main__":
    main()
