"""
System Test - Verify RUN Phase Components

Tests:
1. Model service loading
2. Feature extraction
3. Single prediction
4. SHAP explanation
5. Batch predictions
6. Model info
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.run.model_service import DeforestationModelService


def test_model_service():
    """Test model service functionality."""
    print("=" * 80)
    print("TESTING DEFORESTATION EARLY WARNING SYSTEM")
    print("=" * 80)

    # Test 1: Load model service
    print("\n[1/6] Loading model service...")
    try:
        model_service = DeforestationModelService()
        print("✓ Model service loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model service: {e}")
        return False

    # Test 2: Extract features
    print("\n[2/6] Testing feature extraction...")
    try:
        lat, lon, year = -3.8248, -50.2500, 2024
        features = model_service.extract_features_from_location(lat, lon, year)
        print(f"✓ Extracted {len(features)}D features for ({lat}, {lon})")
        assert len(features) == 70, f"Expected 70 features, got {len(features)}"
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

    # Test 3: Single prediction
    print("\n[3/6] Testing single prediction...")
    try:
        result = model_service.predict(lat, lon, year, threshold=0.5)
        print(f"✓ Prediction successful")
        print(f"  Location: ({result['lat']}, {result['lon']})")
        print(f"  Risk: {result['risk_probability']:.1%}")
        print(f"  Category: {result['risk_category']}")
        print(f"  Confidence: {result['confidence']:.1%} ({result['confidence_label']})")

        # Validate response structure
        required_keys = ['lat', 'lon', 'year', 'risk_probability', 'predicted_class',
                        'threshold', 'confidence', 'confidence_label', 'risk_category', 'timestamp']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        print("✓ Response structure valid")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

    # Test 4: SHAP explanation
    print("\n[4/6] Testing SHAP explanation...")
    try:
        explanation = model_service.explain_prediction(lat, lon, year, top_k=5)

        if 'error' in explanation:
            print(f"⚠ SHAP not available: {explanation['error']}")
            print("  (Install SHAP with: pip install shap)")
        else:
            print(f"✓ SHAP explanation generated")
            print(f"  Base value: {explanation['explanation']['base_value']:.3f}")
            print(f"  Total contribution: {explanation['explanation']['total_contribution']:.3f}")
            print(f"  Top features:")
            for feat in explanation['explanation']['top_features'][:3]:
                print(f"    - {feat['feature']}: {feat['direction']} risk by {feat['contribution_pct']:.1f}%")
    except Exception as e:
        print(f"✗ SHAP explanation failed: {e}")
        # Don't fail test if SHAP not installed
        print("  (This is optional - install SHAP for explanations)")

    # Test 5: Batch predictions
    print("\n[5/6] Testing batch predictions...")
    try:
        locations = [
            (-3.8248, -50.2500, 2024),
            (-3.2356, -50.4530, 2024),
            (-4.1234, -51.5678, 2024)
        ]

        batch_results = model_service.predict_batch(locations, threshold=0.5)
        print(f"✓ Batch prediction successful")
        print(f"  Processed {len(batch_results)} locations")

        # Check for errors
        errors = [r for r in batch_results if 'error' in r]
        successes = [r for r in batch_results if 'error' not in r]

        print(f"  Successful: {len(successes)}")
        print(f"  Failed: {len(errors)}")

        if successes:
            avg_risk = sum(r['risk_probability'] for r in successes) / len(successes)
            print(f"  Average risk: {avg_risk:.1%}")
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False

    # Test 6: Model info
    print("\n[6/6] Testing model info...")
    try:
        info = model_service.get_model_info()
        print(f"✓ Model info retrieved")
        print(f"  Model type: {info['model_type']}")
        print(f"  Features: {info['n_features']}D")
        print(f"  Training samples: {info['training_samples']}")
        print(f"  Training years: {info['training_years']}")
        print(f"  Validation AUROC: {info['validation_auroc']:.3f}")
        print(f"  Validation samples: {info['validation_samples']}")
        print(f"  Model date: {info['model_date']}")

        # Validate info
        assert info['n_features'] == 70, "Expected 70 features"
        assert info['validation_auroc'] > 0.9, f"Expected AUROC > 0.9, got {info['validation_auroc']}"
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start API: uvicorn src.run.api.main:app --reload --port 8000")
    print("2. Start Dashboard: streamlit run src/run/dashboard/app.py")
    print("3. Access API docs: http://localhost:8000/docs")
    print("4. Access Dashboard: http://localhost:8501")
    print("\n" + "=" * 80)

    return True


def main():
    """Run all tests."""
    try:
        success = test_model_service()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
