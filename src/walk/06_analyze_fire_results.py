"""
Analyze Fire Detection Results

Investigates why we got 0 fire detections across all samples.
"""

import pickle
from pathlib import Path

from src.utils import get_config


def analyze_fire_results():
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / "processed"

    print("=" * 80)
    print("FIRE DETECTION RESULTS ANALYSIS")
    print("=" * 80)

    validation_sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']

    all_samples = []

    for set_name in validation_sets:
        fire_file = processed_dir / f"hard_val_{set_name}_fire.pkl"

        if not fire_file.exists():
            print(f"\n✗ {set_name}: File not found")
            continue

        with open(fire_file, 'rb') as f:
            samples = pickle.load(f)

        print(f"\n{set_name.upper()} ({len(samples)} samples)")
        print("-" * 80)

        # Count fire detections
        fire_detected = sum(
            1 for s in samples
            if s.get('fire_features', {}).get('fire_detections_total', 0) > 0
        )

        print(f"  Samples with fire: {fire_detected}/{len(samples)}")

        # Show fire pattern distribution
        patterns = {}
        for s in samples:
            pattern = s.get('fire_features', {}).get('fire_temporal_pattern', 'none')
            patterns[pattern] = patterns.get(pattern, 0) + 1

        print(f"  Patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"    {pattern}: {count}")

        # Show a few samples with their fire features
        print(f"\n  Sample fire features (first 5):")
        for i, sample in enumerate(samples[:5]):
            fire_feats = sample.get('fire_features', {})
            label = sample.get('label', 0)
            label_str = "CLEARING" if label == 1 else "INTACT"

            print(f"\n    Sample {i} ({label_str}):")
            print(f"      Location: ({sample.get('lat', 0):.4f}, {sample.get('lon', 0):.4f})")
            print(f"      Year: {sample.get('year', 'N/A')}")
            print(f"      Fire detections: {fire_feats.get('fire_detections_total', 0)}")
            print(f"      Pattern: {fire_feats.get('fire_temporal_pattern', 'N/A')}")

        all_samples.extend(samples)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_samples = len(all_samples)
    total_fire = sum(
        1 for s in all_samples
        if s.get('fire_features', {}).get('fire_detections_total', 0) > 0
    )

    print(f"\nTotal samples: {total_samples}")
    print(f"Samples with fire: {total_fire} ({total_fire/total_samples*100:.1f}%)")

    # Check edge cases specifically (fire-prone samples)
    edge_cases_file = processed_dir / "hard_val_edge_cases_fire.pkl"
    if edge_cases_file.exists():
        with open(edge_cases_file, 'rb') as f:
            edge_samples = pickle.load(f)

        print(f"\n{'='*80}")
        print("EDGE CASES DETAILED ANALYSIS (Fire-prone regions)")
        print("=" * 80)

        # The edge cases include 5 fire-prone samples
        # Let's examine all clearing samples in edge cases
        clearing_samples = [s for s in edge_samples if s.get('label', 0) == 1]

        print(f"\nClearing samples in edge cases: {len(clearing_samples)}")
        print("\nAll clearing samples:")
        for i, sample in enumerate(clearing_samples):
            fire_feats = sample.get('fire_features', {})
            print(f"\n  Sample {i}:")
            print(f"    Location: ({sample.get('lat', 0):.4f}, {sample.get('lon', 0):.4f})")
            print(f"    Year: {sample.get('year', 'N/A')}")
            print(f"    Fire total: {fire_feats.get('fire_detections_total', 0)}")
            print(f"    Fire before: {fire_feats.get('fire_detections_before', 0)}")
            print(f"    Fire after: {fire_feats.get('fire_detections_after', 0)}")
            print(f"    Pattern: {fire_feats.get('fire_temporal_pattern', 'N/A')}")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print("=" * 80)

    if total_fire == 0:
        print("\n❌ PROBLEM: 0 fire detections across all samples")
        print("\nPossible causes:")
        print("  1. MODIS MCD64A1 resolution too coarse (1km)")
        print("  2. Fire outside 6-month detection window")
        print("  3. Wrong MODIS product (MCD64A1 is monthly burned area)")
        print("  4. 'Fire-prone' refers to regional risk, not actual fire events")
        print("\nRecommended next steps:")
        print("  → Try VIIRS active fire (375m, daily detections)")
        print("  → Extract NBR spectral features (Normalized Burn Ratio)")
        print("  → Verify fire events in FIRMS database for sample locations")
        print("  → OR: Move to next priority (multi-scale embeddings)")


if __name__ == "__main__":
    analyze_fire_results()
