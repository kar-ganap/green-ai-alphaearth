"""
Data Leakage Verification

Implements spatial and temporal leakage checks to ensure:
1. No spatial overlap between train/val/test sets (10km buffer)
2. No temporal causality violations (embeddings extracted before clearing)

Critical for scientific validity and deployment confidence.

Usage:
    from src.walk.data_leakage_verification import (
        verify_no_spatial_leakage,
        verify_temporal_causality,
        run_full_verification
    )
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any
from scipy.spatial import cKDTree
from datetime import datetime

from src.utils import haversine_distance


def verify_no_spatial_leakage(
    train_samples: List[Dict],
    val_samples: List[Dict],
    min_distance_km: float = 10.0,
    coord_keys: Tuple[str, str] = ('lat', 'lon')
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify no validation sample is within min_distance_km of any training sample.

    Args:
        train_samples: List of training samples with coordinates
        val_samples: List of validation samples with coordinates
        min_distance_km: Minimum allowed distance between train and val (km)
        coord_keys: Tuple of (lat_key, lon_key) to extract coordinates

    Returns:
        (is_valid, report_dict)
            is_valid: True if no violations found
            report_dict: Detailed verification report
    """
    lat_key, lon_key = coord_keys

    # Extract coordinates, handling nested location dicts
    def get_coords(sample):
        if 'location' in sample and isinstance(sample['location'], dict):
            return sample['location'][lat_key], sample['location'][lon_key]
        return sample[lat_key], sample[lon_key]

    train_coords = np.array([get_coords(s) for s in train_samples])
    val_coords = np.array([get_coords(s) for s in val_samples])

    # Build spatial tree for efficient nearest neighbor search
    train_tree = cKDTree(train_coords)

    violations = []
    min_actual_distance_km = float('inf')

    for i, val_coord in enumerate(val_coords):
        # Find nearest training sample
        distance_deg, train_idx = train_tree.query(val_coord)

        # Convert degrees to km (approximate: 1 degree ≈ 111 km at equator)
        distance_km = distance_deg * 111.0

        if distance_km < min_actual_distance_km:
            min_actual_distance_km = distance_km

        if distance_km < min_distance_km:
            violations.append({
                'val_idx': i,
                'train_idx': int(train_idx),
                'distance_km': round(distance_km, 2),
                'val_coord': (round(val_coord[0], 4), round(val_coord[1], 4)),
                'train_coord': (round(train_coords[train_idx][0], 4),
                               round(train_coords[train_idx][1], 4)),
                'severity': 'CRITICAL' if distance_km < 1.0 else 'HIGH' if distance_km < 5.0 else 'MEDIUM'
            })

    is_valid = len(violations) == 0

    # Sort violations by distance (most severe first)
    violations.sort(key=lambda x: x['distance_km'])

    report = {
        'is_valid': is_valid,
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_violations': len(violations),
        'violation_rate': len(violations) / len(val_samples) if val_samples else 0,
        'min_allowed_distance_km': min_distance_km,
        'min_actual_distance_km': round(min_actual_distance_km, 2) if min_actual_distance_km != float('inf') else None,
        'violations': violations[:10],  # Top 10 most severe
        'severity_breakdown': {
            'CRITICAL': sum(1 for v in violations if v['severity'] == 'CRITICAL'),
            'HIGH': sum(1 for v in violations if v['severity'] == 'HIGH'),
            'MEDIUM': sum(1 for v in violations if v['severity'] == 'MEDIUM')
        }
    }

    return is_valid, report


def verify_temporal_causality(
    samples: List[Dict],
    embedding_dates_func = None,
    allow_same_year_embeddings: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify all embeddings are extracted BEFORE clearing event.

    Args:
        samples: List of samples with year and embedding dates
        embedding_dates_func: Function that returns embedding dates for a sample
                             If None, uses default conservative approach
        allow_same_year_embeddings: If False (default), requires all embeddings
                                   from year < clearing_year for safety

    Returns:
        (is_valid, report_dict)
            is_valid: True if no violations found
            report_dict: Detailed verification report
    """
    if embedding_dates_func is None:
        embedding_dates_func = get_conservative_embedding_dates

    violations = []
    warnings = []

    for i, sample in enumerate(samples):
        # Get clearing year
        clearing_year = sample.get('year')
        if clearing_year is None:
            warnings.append({
                'sample_idx': i,
                'issue': 'No year field found in sample'
            })
            continue

        # Get embedding dates
        try:
            if 'embeddings' in sample:
                # Embeddings already extracted, check their dates
                embedding_info = sample.get('embedding_metadata', {})
                embedding_dates = embedding_info.get('dates', {})
            else:
                # Get what dates would be used for extraction
                embedding_dates = embedding_dates_func(sample)
        except Exception as e:
            warnings.append({
                'sample_idx': i,
                'issue': f'Failed to get embedding dates: {e}'
            })
            continue

        # Check each embedding quarter
        for quarter, date_str in embedding_dates.items():
            if quarter == 'Clearing':
                continue  # Clearing embedding is expected to be after

            try:
                # Parse date string (format: YYYY-MM-DD)
                emb_year = int(date_str.split('-')[0])
                emb_month = int(date_str.split('-')[1])

                # Conservative check: embedding year < clearing year
                if not allow_same_year_embeddings:
                    if emb_year >= clearing_year:
                        violations.append({
                            'sample_idx': i,
                            'clearing_year': clearing_year,
                            'quarter': quarter,
                            'embedding_date': date_str,
                            'embedding_year': emb_year,
                            'embedding_month': emb_month,
                            'issue': f'{quarter} embedding from year {emb_year} >= clearing year {clearing_year}',
                            'severity': 'CRITICAL'
                        })
                else:
                    # Less conservative: allow same year if embedding is early enough
                    # Assume clearing could happen anytime in clearing_year
                    # Embedding is unsafe if same year and month >= 6 (mid-year)
                    if emb_year > clearing_year:
                        violations.append({
                            'sample_idx': i,
                            'clearing_year': clearing_year,
                            'quarter': quarter,
                            'embedding_date': date_str,
                            'embedding_year': emb_year,
                            'embedding_month': emb_month,
                            'issue': f'{quarter} embedding from year {emb_year} > clearing year {clearing_year}',
                            'severity': 'CRITICAL'
                        })
                    elif emb_year == clearing_year and emb_month >= 6:
                        violations.append({
                            'sample_idx': i,
                            'clearing_year': clearing_year,
                            'quarter': quarter,
                            'embedding_date': date_str,
                            'embedding_year': emb_year,
                            'embedding_month': emb_month,
                            'issue': f'{quarter} embedding from {date_str} in same year as clearing, month >= 6 (unsafe)',
                            'severity': 'HIGH'
                        })

            except (ValueError, IndexError) as e:
                warnings.append({
                    'sample_idx': i,
                    'quarter': quarter,
                    'issue': f'Failed to parse date {date_str}: {e}'
                })

    is_valid = len(violations) == 0

    report = {
        'is_valid': is_valid,
        'n_samples': len(samples),
        'n_violations': len(violations),
        'n_warnings': len(warnings),
        'violation_rate': len(violations) / len(samples) if samples else 0,
        'allow_same_year': allow_same_year_embeddings,
        'violations': violations,
        'warnings': warnings,
        'severity_breakdown': {
            'CRITICAL': sum(1 for v in violations if v.get('severity') == 'CRITICAL'),
            'HIGH': sum(1 for v in violations if v.get('severity') == 'HIGH')
        }
    }

    return is_valid, report


def get_conservative_embedding_dates(sample: Dict) -> Dict[str, str]:
    """
    Return embedding dates that GUARANTEE temporal causality.

    All embeddings from year BEFORE clearing year.
    Trade-off: Longer lag times (6-18 months instead of 0-12 months).

    Args:
        sample: Sample dict with 'year' field

    Returns:
        Dict mapping quarter to date string (YYYY-MM-DD)
    """
    clearing_year = sample['year']

    return {
        'Q1': f"{clearing_year - 1}-03-01",  # Mar Y-1 (15-18 months before)
        'Q2': f"{clearing_year - 1}-06-01",  # Jun Y-1 (12-15 months before)
        'Q3': f"{clearing_year - 1}-09-01",  # Sep Y-1 (9-12 months before)
        'Q4': f"{clearing_year - 1}-12-01",  # Dec Y-1 (6-9 months before)
        'Clearing': f"{clearing_year + 1}-06-01",  # Jun Y+1 (verification)
    }


def get_current_embedding_dates(sample: Dict) -> Dict[str, str]:
    """
    Return embedding dates used in CURRENT implementation.

    WARNING: These dates may violate temporal causality!

    Args:
        sample: Sample dict with 'year' field

    Returns:
        Dict mapping quarter to date string (YYYY-MM-DD)
    """
    year = sample['year']

    return {
        'Q1': f"{year-1}-06-01",  # Jun Y-1 (9-12 months before)
        'Q2': f"{year}-03-01",    # Mar Y (6-9 months before? UNSAFE!)
        'Q3': f"{year}-06-01",    # Jun Y (3-6 months before? UNSAFE!)
        'Q4': f"{year}-09-01",    # Sep Y (0-3 months before? UNSAFE!)
        'Clearing': f"{year+1}-06-01",
    }


def verify_within_set_spatial_splits(
    dataset: Dict,
    min_distance_km: float = 10.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify spatial cross-validation splits within a dataset.

    Checks that train/val/test splits respect spatial buffer.

    Args:
        dataset: Dataset dict with 'data' and 'splits' keys
        min_distance_km: Minimum allowed distance between splits

    Returns:
        (is_valid, report_dict)
    """
    data = dataset['data']
    splits = dataset['splits']

    reports = {}
    all_valid = True

    # Check train vs val
    train_samples = [data[i] for i in splits['train']]
    val_samples = [data[i] for i in splits['val']]

    is_valid, report = verify_no_spatial_leakage(
        train_samples, val_samples, min_distance_km
    )
    reports['train_vs_val'] = report
    all_valid = all_valid and is_valid

    # Check train vs test
    test_samples = [data[i] for i in splits['test']]

    is_valid, report = verify_no_spatial_leakage(
        train_samples, test_samples, min_distance_km
    )
    reports['train_vs_test'] = report
    all_valid = all_valid and is_valid

    # Check val vs test
    is_valid, report = verify_no_spatial_leakage(
        val_samples, test_samples, min_distance_km
    )
    reports['val_vs_test'] = report
    all_valid = all_valid and is_valid

    summary = {
        'is_valid': all_valid,
        'min_distance_km': min_distance_km,
        'splits_checked': ['train_vs_val', 'train_vs_test', 'val_vs_test'],
        'total_violations': sum(r['n_violations'] for r in reports.values()),
        'reports': reports
    }

    return all_valid, summary


def run_full_verification(
    data_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete leakage verification on all datasets.

    Checks:
    1. Spatial leakage: Training vs all validation sets
    2. Within-set spatial splits
    3. Temporal causality for all datasets

    Args:
        data_dir: Path to data/processed directory
        verbose: Print detailed results

    Returns:
        Complete verification report
    """
    processed_dir = Path(data_dir) / "processed"

    results = {
        'timestamp': datetime.now().isoformat(),
        'spatial_leakage': {},
        'temporal_causality': {},
        'overall_valid': True
    }

    if verbose:
        print("=" * 80)
        print("DATA LEAKAGE VERIFICATION")
        print("=" * 80)
        print()

    # Load datasets
    datasets = {}

    # Training set
    train_file = processed_dir / "walk_dataset.pkl"
    if train_file.exists():
        with open(train_file, 'rb') as f:
            datasets['training'] = pickle.load(f)
        if verbose:
            print(f"✓ Loaded training set: {len(datasets['training']['data'])} samples")

    # Hard validation sets
    val_sets = ['rapid_response', 'risk_ranking', 'comprehensive', 'edge_cases']
    for set_name in val_sets:
        # Try both with and without _multiscale suffix
        for suffix in ['_multiscale', '_features', '']:
            val_file = processed_dir / f"hard_val_{set_name}{suffix}.pkl"
            if val_file.exists():
                with open(val_file, 'rb') as f:
                    datasets[f'val_{set_name}'] = pickle.load(f)
                if verbose:
                    print(f"✓ Loaded {set_name}: {len(datasets[f'val_{set_name}'])} samples")
                break

    if not datasets:
        if verbose:
            print("✗ No datasets found!")
        results['overall_valid'] = False
        return results

    print()

    # 1. SPATIAL LEAKAGE CHECKS
    if verbose:
        print("=" * 80)
        print("SPATIAL LEAKAGE VERIFICATION")
        print("=" * 80)
        print()

    if 'training' in datasets:
        train_data = datasets['training']['data']

        for val_name, val_samples in datasets.items():
            if not val_name.startswith('val_'):
                continue

            if verbose:
                print(f"Checking: Training vs {val_name}...")

            is_valid, report = verify_no_spatial_leakage(
                train_data,
                val_samples if isinstance(val_samples, list) else val_samples['data'],
                min_distance_km=10.0
            )

            results['spatial_leakage'][f'train_vs_{val_name}'] = report
            results['overall_valid'] = results['overall_valid'] and is_valid

            if verbose:
                status = "✓ PASS" if is_valid else "✗ FAIL"
                print(f"  {status}: {report['n_violations']} violations")
                if report['n_violations'] > 0:
                    print(f"  Minimum distance: {report['min_actual_distance_km']} km")
                    print(f"  Severity: {report['severity_breakdown']}")
                print()

        # Check within-training-set splits
        if verbose:
            print("Checking: Within training set splits...")

        is_valid, report = verify_within_set_spatial_splits(datasets['training'])
        results['spatial_leakage']['within_training'] = report
        results['overall_valid'] = results['overall_valid'] and is_valid

        if verbose:
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"  {status}: {report['total_violations']} total violations")
            print()

    # 2. TEMPORAL CAUSALITY CHECKS
    if verbose:
        print("=" * 80)
        print("TEMPORAL CAUSALITY VERIFICATION")
        print("=" * 80)
        print()

    for dataset_name, dataset in datasets.items():
        if verbose:
            print(f"Checking: {dataset_name}...")

        samples = dataset['data'] if isinstance(dataset, dict) and 'data' in dataset else dataset

        # Check with current dates (expected to fail)
        is_valid_current, report_current = verify_temporal_causality(
            samples,
            embedding_dates_func=get_current_embedding_dates,
            allow_same_year_embeddings=False
        )

        # Check with conservative dates (should pass)
        is_valid_conservative, report_conservative = verify_temporal_causality(
            samples,
            embedding_dates_func=get_conservative_embedding_dates,
            allow_same_year_embeddings=False
        )

        results['temporal_causality'][dataset_name] = {
            'current_approach': report_current,
            'conservative_approach': report_conservative
        }

        # Overall validity requires conservative approach
        results['overall_valid'] = results['overall_valid'] and is_valid_conservative

        if verbose:
            print(f"  Current approach:")
            status = "✓ PASS" if is_valid_current else "✗ FAIL"
            print(f"    {status}: {report_current['n_violations']} violations")
            if report_current['n_violations'] > 0:
                print(f"    Severity: {report_current['severity_breakdown']}")

            print(f"  Conservative approach (Y-1 only):")
            status = "✓ PASS" if is_valid_conservative else "✗ FAIL"
            print(f"    {status}: {report_conservative['n_violations']} violations")
            print()

    # SUMMARY
    if verbose:
        print("=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print()

        if results['overall_valid']:
            print("✓ ALL CHECKS PASSED")
            print()
            print("Datasets are valid for model training.")
        else:
            print("✗ VALIDATION FAILED")
            print()
            print("Issues found:")

            # Summarize spatial issues
            spatial_violations = sum(
                r['n_violations']
                for r in results['spatial_leakage'].values()
                if isinstance(r, dict) and 'n_violations' in r
            )
            if spatial_violations > 0:
                print(f"  - Spatial leakage: {spatial_violations} violations")

            # Summarize temporal issues
            temporal_violations = sum(
                r['current_approach']['n_violations']
                for r in results['temporal_causality'].values()
            )
            if temporal_violations > 0:
                print(f"  - Temporal causality: {temporal_violations} violations (current approach)")

            print()
            print("Required actions:")
            if spatial_violations > 0:
                print("  1. Re-sample violating samples from different geographic regions")
            if temporal_violations > 0:
                print("  2. Re-extract embeddings with conservative dates (Y-1 only)")

        print("=" * 80)

    return results


def main():
    """Run verification on current datasets."""
    from src.utils import get_config

    config = get_config()
    data_dir = config.get_path("paths.data_dir")

    results = run_full_verification(data_dir, verbose=True)

    # Save report
    output_file = data_dir / "processed" / "leakage_verification_report.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed report saved to: {output_file}")

    # Exit code reflects validation status
    import sys
    sys.exit(0 if results['overall_valid'] else 1)


if __name__ == "__main__":
    main()
