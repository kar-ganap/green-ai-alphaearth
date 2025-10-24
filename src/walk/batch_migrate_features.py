#!/usr/bin/env python3
"""
Batch Migration Script for Feature Extraction Consolidation

Migrates scripts that use inline feature assembly to use the
consolidated feature_extraction module.

Usage:
    python src/walk/batch_migrate_features.py
"""

import re
from pathlib import Path

# Scripts to migrate (found via grep)
SCRIPTS_TO_MIGRATE = [
    "10b_phase1_train_alphaearth_only.py",
    "10c_phase1_train_vector_deltas.py",
    "11_train_random_forest.py",
    "13_train_xgboost_69d.py",
    "15_train_xgboost_sentinel2.py",
    "19_diagnose_perfect_cv.py",
    "20_error_analysis_shopping_list.py",
    "23_train_quickwin_rf.py",
    "26_train_edge_cases_complete_rf.py",
    "29_train_all_hard_samples_rf.py",
    "30_threshold_optimization.py",
    "31_temporal_validation.py",
    "31b_temporal_validation_from_existing.py",
    "31c_evaluate_temporal_model_on_validation_sets.py",
    "34_phase4_temporal_validation.py",
    "35_train_production_model.py",
    "36_analyze_2024_drift.py",
    "40_phase4_uniform_30pct_validation.py",
    "41_phase_a_temporal_adaptation.py",
    "42_phase_b_model_diversity.py",
    "44_comprehensive_evaluation.py",
    "45_complete_comprehensive_evaluation.py",
    "50_model_ensemble_hard_sets.py",
    "51_final_models_2020_2024.py",
]

# Pattern to find and replace the old extract_features function
OLD_PATTERN = r'''def extract_features\(samples\):
    """Extract 70D features from samples using Phase B method\."""
    X = \[\]
    y = \[\]

    for sample in samples:
        # Extract annual features \(3D\)
        annual_features = sample\.get\('annual_features'\)
        if annual_features is None:
            raise ValueError\(f"Sample missing annual_features: {sample\.get\('lat', 'unknown'\)}, {sample\.get\('lon', 'unknown'\)}"\)
        annual_features = np\.array\(annual_features\)\.flatten\(\)

        # Extract coarse features \(66D\) from multiscale_features dict
        multiscale_dict = sample\.get\('multiscale_features', \{\}\)
        if not isinstance\(multiscale_dict, dict\):
            raise ValueError\(f"multiscale_features must be a dict, got {type\(multiscale_dict\)}"\)

        # Define feature names in correct order: 64 embeddings \+ 2 stats
        coarse_feature_names = \[f'coarse_emb_{i}' for i in range\(64\)\] \+ \['coarse_heterogeneity', 'coarse_range'\]
        coarse_features = np\.array\(\[multiscale_dict\[k\] for k in coarse_feature_names\]\)

        # Extract or compute year feature \(1D\)
        year = sample\.get\('year', 2021\)
        year_feature = \(year - 2020\) / 4\.0  # Normalize to \[0,1\] for range 2020-2024

        # Combine: 3D \+ 66D \+ 1D = 70D
        combined = np\.concatenate\(\[annual_features, coarse_features, \[year_feature\]\]\)
        X\.append\(combined\)
        y\.append\(sample\.get\('label', 0\)\)

    return np\.array\(X\), np\.array\(y\)'''

NEW_CODE = '''def extract_features(samples):
    """
    Extract 70D features from samples using consolidated module.

    Uses features_to_array() from consolidated feature extraction module.
    """
    X = []
    y = []

    for sample in samples:
        # Use consolidated module to convert sample features to 70D array
        features_70d = features_to_array(sample)

        if features_70d is None:
            raise ValueError(f"Failed to extract features for sample: {sample.get('lat', 'unknown')}, {sample.get('lon', 'unknown')}")

        X.append(features_70d)
        y.append(sample.get('label', 0))

    return np.array(X), np.array(y)'''


def add_import_if_needed(content: str) -> str:
    """Add import for features_to_array if not already present."""
    if 'from src.walk.utils import' in content and 'features_to_array' in content:
        return content  # Already has the import

    # Find the import section (after existing imports from src.utils or src.walk)
    # Look for pattern: from src.utils import ...
    import_pattern = r'(from src\.utils import [^\n]+)'

    if re.search(import_pattern, content):
        # Add after src.utils import
        content = re.sub(
            import_pattern,
            r'\1\nfrom src.walk.utils import features_to_array',
            content,
            count=1
        )
    else:
        # No src.utils import, look for sys.path.append pattern
        sys_path_pattern = r'(sys\.path\.append\(.*?\)\n)'
        if re.search(sys_path_pattern, content):
            content = re.sub(
                sys_path_pattern,
                r'\1from src.utils import get_config\nfrom src.walk.utils import features_to_array\n',
                content,
                count=1
            )

    return content


def migrate_script(script_path: Path) -> bool:
    """Migrate a single script to use consolidated features."""
    print(f"\nMigrating: {script_path.name}")

    with open(script_path, 'r') as f:
        content = f.read()

    original_content = content

    # Replace the extract_features function
    content, n_replacements = re.subn(OLD_PATTERN, NEW_CODE, content, flags=re.MULTILINE)

    if n_replacements == 0:
        print(f"  ⚠️ Pattern not found - skipping")
        return False

    # Add import if needed
    content = add_import_if_needed(content)

    if content == original_content:
        print(f"  ⚠️ No changes made")
        return False

    # Write back
    with open(script_path, 'w') as f:
        f.write(content)

    print(f"  ✓ Migrated ({n_replacements} replacements)")
    return True


def main():
    """Run batch migration."""
    print("=" * 80)
    print("BATCH MIGRATION: Feature Extraction Consolidation")
    print("=" * 80)

    walk_dir = Path(__file__).parent
    migrated = 0
    skipped = 0
    errors = 0

    for script_name in SCRIPTS_TO_MIGRATE:
        script_path = walk_dir / script_name

        if not script_path.exists():
            print(f"\n⚠️ File not found: {script_name}")
            skipped += 1
            continue

        try:
            if migrate_script(script_path):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n✗ Error migrating {script_name}: {e}")
            errors += 1

    # Summary
    print("\n" + "=" * 80)
    print("MIGRATION COMPLETE")
    print("=" * 80)
    print(f"✓ Migrated: {migrated}")
    print(f"⚠ Skipped:  {skipped}")
    print(f"✗ Errors:   {errors}")
    print(f"\nTotal scripts processed: {len(SCRIPTS_TO_MIGRATE)}")


if __name__ == "__main__":
    main()
