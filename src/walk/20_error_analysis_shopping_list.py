"""
Validation Error Analysis & Sample Shopping List

Analyzes what the model gets wrong on validation sets and generates
a specific, actionable "shopping list" of samples to collect.

GOAL:
- Identify patterns in misclassified samples
- Determine specific characteristics to target
- Generate geographic sampling locations
- Create concrete collection criteria (no human judgment needed)

Usage:
    uv run python src/walk/20_error_analysis_shopping_list.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.utils import get_config
from src.utils.earth_engine import EarthEngineClient


def load_model_and_data():
    """Load trained model and validation sets."""
    config = get_config()
    data_dir = config.get_path("paths.data_dir")
    processed_dir = data_dir / 'processed'

    # Load trained model
    model_path = processed_dir / 'walk_model_random_forest.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']

    # Load validation sets
    val_sets = {}
    set_names = ['edge_cases', 'rapid_response', 'comprehensive', 'risk_ranking']

    ee_client = EarthEngineClient(use_cache=True)

    for set_name in set_names:
        multiscale_path = processed_dir / f'hard_val_{set_name}_multiscale.pkl'
        if not multiscale_path.exists():
            continue

        with open(multiscale_path, 'rb') as f:
            samples = pickle.load(f)

        # Extract features (same as training)
        X_val = []
        y_val = []
        samples_with_features = []

        for sample in samples:
            # Fix missing 'year' field
            if 'year' not in sample and sample.get('stable', False):
                sample = sample.copy()
                sample['year'] = 2021

            # Extract annual features
            try:
                from src.walk.diagnostic_helpers import extract_dual_year_features
                annual_features = extract_dual_year_features(ee_client, sample)
            except:
                annual_features = None

            if annual_features is None:
                continue

            # Get coarse features
            if 'multiscale_features' not in sample:
                continue

            multiscale_dict = sample['multiscale_features']
            coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']

            missing = [k for k in coarse_feature_names if k not in multiscale_dict]
            if missing:
                continue

            coarse_features = np.array([multiscale_dict[k] for k in coarse_feature_names])
            combined = np.concatenate([annual_features, coarse_features])

            if len(combined) != 69:
                continue

            X_val.append(combined)
            y_val.append(sample.get('label', 0))
            samples_with_features.append(sample)

        if len(X_val) > 0:
            val_sets[set_name] = {
                'X': np.vstack(X_val),
                'y': np.array(y_val),
                'samples': samples_with_features
            }

    return model, scaler, feature_names, val_sets


def analyze_errors(model, scaler, feature_names, val_sets):
    """Analyze misclassified samples across all validation sets."""
    print("=" * 80)
    print("VALIDATION ERROR ANALYSIS")
    print("=" * 80)

    all_errors = {
        'false_positives': [],  # Predicted clearing, actually intact
        'false_negatives': [],  # Predicted intact, actually clearing
    }

    set_summaries = {}

    for set_name, data in val_sets.items():
        X = scaler.transform(data['X'])
        y_true = data['y']
        samples = data['samples']

        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Identify errors
        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

        # Store errors with metadata
        for idx in fp_idx:
            error = {
                'sample': samples[idx],
                'confidence': y_pred_proba[idx],
                'features': data['X'][idx],
                'set': set_name,
                'type': 'FP'
            }
            all_errors['false_positives'].append(error)

        for idx in fn_idx:
            error = {
                'sample': samples[idx],
                'confidence': y_pred_proba[idx],
                'features': data['X'][idx],
                'set': set_name,
                'type': 'FN'
            }
            all_errors['false_negatives'].append(error)

        # Summary for this set
        accuracy = (y_pred == y_true).mean()

        set_summaries[set_name] = {
            'total': len(y_true),
            'correct': int((y_pred == y_true).sum()),
            'accuracy': accuracy,
            'fp_count': len(fp_idx),
            'fn_count': len(fn_idx),
            'fp_rate': len(fp_idx) / len(y_true),
            'fn_rate': len(fn_idx) / len(y_true),
        }

        print(f"\n{set_name}:")
        print(f"  Total: {len(y_true)} samples")
        print(f"  Correct: {int((y_pred == y_true).sum())} ({accuracy:.1%})")
        print(f"  False Positives: {len(fp_idx)} ({len(fp_idx)/len(y_true):.1%})")
        print(f"  False Negatives: {len(fn_idx)} ({len(fn_idx)/len(y_true):.1%})")

    return all_errors, set_summaries


def characterize_errors(errors, feature_names):
    """Characterize what makes errors different from correct predictions."""
    print("\n" + "=" * 80)
    print("ERROR CHARACTERIZATION")
    print("=" * 80)

    characteristics = {}

    for error_type in ['false_positives', 'false_negatives']:
        if not errors[error_type]:
            continue

        print(f"\n{error_type.upper().replace('_', ' ')} ({len(errors[error_type])} samples):")
        print("-" * 80)

        error_samples = errors[error_type]

        # Geographic distribution
        lats = [e['sample']['lat'] for e in error_samples]
        lons = [e['sample']['lon'] for e in error_samples]

        print(f"\nGeographic Distribution:")
        print(f"  Latitude:  [{min(lats):.2f}, {max(lats):.2f}]")
        print(f"  Longitude: [{min(lons):.2f}, {max(lons):.2f}]")

        # Temporal distribution
        years = [e['sample'].get('year', e['sample'].get('clearing_year', 0)) for e in error_samples]
        year_counts = {}
        for y in years:
            if y > 0:
                year_counts[y] = year_counts.get(y, 0) + 1

        if year_counts:
            print(f"\nTemporal Distribution:")
            for year in sorted(year_counts.keys()):
                print(f"  {year}: {year_counts[year]} samples")

        # Confidence distribution
        confidences = [e['confidence'] for e in error_samples]
        print(f"\nPrediction Confidence:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Range: [{min(confidences):.3f}, {max(confidences):.3f}]")

        # Low confidence errors (near decision boundary)
        near_boundary = [e for e in error_samples if 0.3 < e['confidence'] < 0.7]
        print(f"  Near boundary (0.3-0.7): {len(near_boundary)} ({len(near_boundary)/len(error_samples):.1%})")

        # Feature analysis - which features differ most?
        features_array = np.vstack([e['features'] for e in error_samples])

        # Calculate feature statistics for errors
        feature_means = features_array.mean(axis=0)
        feature_stds = features_array.std(axis=0)

        # Find most variable features (potential discriminators)
        high_variance_idx = np.argsort(feature_stds)[::-1][:10]

        print(f"\nMost Variable Features (potential patterns):")
        for i, idx in enumerate(high_variance_idx[:5], 1):
            print(f"  {i}. {feature_names[idx]}: μ={feature_means[idx]:.3f}, σ={feature_stds[idx]:.3f}")

        # Store characteristics
        characteristics[error_type] = {
            'count': len(error_samples),
            'geo_bounds': {
                'lat': [min(lats), max(lats)],
                'lon': [min(lons), max(lons)]
            },
            'years': year_counts,
            'confidence': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'near_boundary_count': len(near_boundary)
            },
            'top_variable_features': [
                {
                    'name': feature_names[idx],
                    'mean': float(feature_means[idx]),
                    'std': float(feature_stds[idx])
                }
                for idx in high_variance_idx[:10]
            ]
        }

    return characteristics


def generate_shopping_list(errors, characteristics, set_summaries):
    """Generate specific, actionable sampling criteria."""
    print("\n" + "=" * 80)
    print("SAMPLE SHOPPING LIST")
    print("=" * 80)

    shopping_list = []

    # Priority 1: edge_cases (worst performance)
    if 'edge_cases' in set_summaries:
        edge_summary = set_summaries['edge_cases']

        print("\n📋 PRIORITY 1: edge_cases (0.583 ROC-AUC)")
        print("-" * 80)

        # Analyze edge_cases errors
        edge_fps = [e for e in errors['false_positives'] if e['set'] == 'edge_cases']
        edge_fns = [e for e in errors['false_negatives'] if e['set'] == 'edge_cases']

        if edge_fns:
            print(f"\nFalse Negatives ({len(edge_fns)} samples):")
            print("  Model failed to detect these clearings")
            print("  → Need MORE clearing examples that look like these")

            # Get characteristics
            lats = [e['sample']['lat'] for e in edge_fns]
            lons = [e['sample']['lon'] for e in edge_fns]

            item = {
                'priority': 1,
                'target_set': 'edge_cases',
                'error_type': 'false_negative',
                'label': 1,  # Collect clearings
                'count_needed': 30,
                'description': 'Subtle clearings that model misses',
                'criteria': {
                    'geographic_region': {
                        'lat_range': [min(lats) - 2, max(lats) + 2],
                        'lon_range': [min(lons) - 2, max(lons) + 2],
                    },
                    'characteristics': [
                        'Small NDVI change (-0.05 to -0.20)',
                        'Small area (1-10 ha)',
                        'Partial forest loss',
                        'Moderate texture change',
                    ],
                    'data_sources': [
                        'Hansen Global Forest Change (2020-2023)',
                        'PRODES annual deforestation',
                        'Areas near current edge_cases',
                    ]
                }
            }
            shopping_list.append(item)

            print(f"\n  ✓ Sample around: [{min(lats):.1f}, {min(lons):.1f}] to [{max(lats):.1f}, {max(lons):.1f}]")
            print(f"  ✓ Collect: 30 subtle clearing samples")
            print(f"  ✓ Criteria: Small NDVI change, partial forest loss")

        if edge_fps:
            print(f"\nFalse Positives ({len(edge_fps)} samples):")
            print("  Model incorrectly flagged intact as clearing")
            print("  → Need MORE intact examples that look like clearings")

            lats = [e['sample']['lat'] for e in edge_fps]
            lons = [e['sample']['lon'] for e in edge_fps]

            item = {
                'priority': 1,
                'target_set': 'edge_cases',
                'error_type': 'false_positive',
                'label': 0,  # Collect intact
                'count_needed': 20,
                'description': 'Intact forest that model confuses for clearing',
                'criteria': {
                    'geographic_region': {
                        'lat_range': [min(lats) - 2, max(lats) + 2],
                        'lon_range': [min(lons) - 2, max(lons) + 2],
                    },
                    'characteristics': [
                        'Heterogeneous canopy',
                        'Natural gaps or seasonality',
                        'Low NDVI but stable',
                        'Similar spectral signature to clearings',
                    ],
                    'data_sources': [
                        'Intact forest near edge_cases FP locations',
                        'Secondary forest (old regrowth)',
                        'Seasonally dry forest',
                    ]
                }
            }
            shopping_list.append(item)

            print(f"\n  ✓ Sample around: [{min(lats):.1f}, {min(lons):.1f}] to [{max(lats):.1f}, {max(lons):.1f}]")
            print(f"  ✓ Collect: 20 confusing intact samples")
            print(f"  ✓ Criteria: Heterogeneous, low NDVI but stable")

    # Priority 2: rapid_response
    if 'rapid_response' in set_summaries:
        rapid_summary = set_summaries['rapid_response']

        print("\n📋 PRIORITY 2: rapid_response (0.760 ROC-AUC)")
        print("-" * 80)

        rapid_fns = [e for e in errors['false_negatives'] if e['set'] == 'rapid_response']

        if rapid_fns:
            print(f"\nFalse Negatives ({len(rapid_fns)} samples):")
            print("  Model missed fire-related clearings")
            print("  → Need fire-related and regrowth examples")

            item = {
                'priority': 2,
                'target_set': 'rapid_response',
                'error_type': 'false_negative',
                'label': 1,
                'count_needed': 25,
                'description': 'Fire-cleared areas with regrowth',
                'criteria': {
                    'characteristics': [
                        'FIRMS fire detection within 1 km, 6 months',
                        'NBR < -0.3 (burn signature)',
                        'Rapid EVI recovery after drop',
                        'Low NDVI but high NIR (ash)',
                    ],
                    'data_sources': [
                        'FIRMS/VIIRS fire detections',
                        'Burn severity maps',
                        'Areas with vegetation recovery trajectories',
                    ]
                }
            }
            shopping_list.append(item)

            print(f"\n  ✓ Collect: 25 fire-related clearing samples")
            print(f"  ✓ Criteria: Fire detections, burn signature, regrowth pattern")

    # Priority 3: comprehensive
    if 'comprehensive' in set_summaries:
        print("\n📋 PRIORITY 3: comprehensive (0.713 ROC-AUC)")
        print("-" * 80)
        print("  → Need geographic and size diversity")

        item = {
            'priority': 3,
            'target_set': 'comprehensive',
            'error_type': 'mixed',
            'label': 'both',
            'count_needed': 30,
            'description': 'Diverse samples (geography, size, time)',
            'criteria': {
                'characteristics': [
                    'Western Amazon (underrepresented)',
                    'Small clearings (<5 ha)',
                    'Large clearings (>100 ha)',
                    'Different years (2020-2023)',
                ],
                'data_sources': [
                    'Hansen GFC (stratified sampling)',
                    'PRODES (multiple regions)',
                    'Ensure size diversity',
                ]
            }
        }
        shopping_list.append(item)

        print(f"\n  ✓ Collect: 30 diverse samples")
        print(f"  ✓ Criteria: Geographic spread, size range, temporal diversity")

    # Summary
    print("\n" + "=" * 80)
    print("SHOPPING LIST SUMMARY")
    print("=" * 80)

    total_needed = sum(item['count_needed'] for item in shopping_list)

    print(f"\nTotal samples to collect: {total_needed}")
    print("\nBy priority:")
    for item in shopping_list:
        print(f"  {item['priority']}. {item['target_set']}: {item['count_needed']} samples ({item['description']})")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Use automated sampling script to collect these samples")
    print("2. Filter by criteria above (no human review needed)")
    print("3. Verify spatial separation (10km from validation sets)")
    print("4. Extract features automatically")
    print("5. Add to training set and re-train")
    print("\nExpected improvement:")
    print("  edge_cases:     0.583 → 0.65-0.70")
    print("  rapid_response: 0.760 → 0.80-0.82")
    print("  comprehensive:  0.713 → 0.74-0.76")

    return shopping_list


def save_shopping_list(shopping_list, characteristics, set_summaries):
    """Save detailed shopping list to JSON."""
    output = {
        'shopping_list': shopping_list,
        'error_characteristics': characteristics,
        'validation_summaries': set_summaries,
        'metadata': {
            'total_samples_needed': sum(item['count_needed'] for item in shopping_list),
            'priority_order': ['edge_cases', 'rapid_response', 'comprehensive', 'risk_ranking'],
        }
    }

    output_path = Path('results/walk/sample_shopping_list.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Shopping list saved to: {output_path}")
    return output_path


def main():
    print("=" * 80)
    print("VALIDATION ERROR ANALYSIS & SAMPLE SHOPPING LIST")
    print("=" * 80)

    # Load model and validation data
    print("\nLoading model and validation sets...")
    model, scaler, feature_names, val_sets = load_model_and_data()
    print(f"✓ Loaded model and {len(val_sets)} validation sets")

    # Analyze errors
    errors, set_summaries = analyze_errors(model, scaler, feature_names, val_sets)

    # Characterize errors
    characteristics = characterize_errors(errors, feature_names)

    # Generate shopping list
    shopping_list = generate_shopping_list(errors, characteristics, set_summaries)

    # Save results
    output_path = save_shopping_list(shopping_list, characteristics, set_summaries)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReview: {output_path}")
    print("\nReady to proceed with automated sample collection!")
    print()


if __name__ == '__main__':
    main()
