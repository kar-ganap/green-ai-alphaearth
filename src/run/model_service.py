"""
Model Service for Deforestation Prediction

Handles model loading, feature extraction, predictions, and explanations.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import EarthEngineClient, get_config
from src.walk.diagnostic_helpers import extract_dual_year_features
import importlib.util

# Import from numbered file using importlib
_spec = importlib.util.spec_from_file_location(
    "multiscale_embeddings",
    Path(__file__).parent.parent / "walk" / "08_multiscale_embeddings.py"
)
_multiscale_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_multiscale_module)
extract_multiscale_features_for_sample = _multiscale_module.extract_multiscale_features_for_sample


class DeforestationModelService:
    """Service for making deforestation risk predictions."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model service.

        Args:
            model_path: Path to trained model file. If None, loads default production model.
        """
        self.config = get_config()
        self.data_dir = self.config.get_path("paths.data_dir")

        # Load model
        if model_path is None:
            model_path = self.data_dir / 'processed' / 'final_xgb_model_2020_2024.pkl'

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Initialize Earth Engine client
        self.ee_client = EarthEngineClient()

        # Feature names for interpretability
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        """Get human-readable feature names."""
        annual_names = ['delta_1yr', 'delta_2yr', 'acceleration']
        coarse_names = [f'coarse_emb_{i}' for i in range(64)]
        stat_names = ['coarse_heterogeneity', 'coarse_range']
        year_name = ['normalized_year']

        return annual_names + coarse_names + stat_names + year_name

    def extract_features_from_location(
        self,
        lat: float,
        lon: float,
        year: int
    ) -> np.ndarray:
        """
        Extract 70D features for a location.

        Args:
            lat: Latitude
            lon: Longitude
            year: Year for prediction

        Returns:
            70D feature vector
        """
        # Create sample dict
        sample = {'lat': lat, 'lon': lon, 'year': year}

        # Extract annual features (3D)
        annual_features = extract_dual_year_features(self.ee_client, sample)
        if annual_features is None or len(annual_features) != 3:
            raise ValueError(f"Failed to extract annual features for ({lat}, {lon})")
        annual_features = np.array(annual_features).flatten()

        # Extract multiscale features (66D)
        enriched_sample = extract_multiscale_features_for_sample(
            self.ee_client,
            sample,
            timepoint='annual'
        )
        if enriched_sample is None or 'multiscale_features' not in enriched_sample:
            raise ValueError(f"Failed to extract multiscale features for ({lat}, {lon})")

        # Extract coarse features: 64 embeddings + 2 stats
        multiscale_features = enriched_sample['multiscale_features']
        coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + \
                              ['coarse_heterogeneity', 'coarse_range']
        coarse_features = np.array([multiscale_features[k] for k in coarse_feature_names])

        # Year feature (1D) - normalized to [0,1] for 2020-2024 range
        year_feature = (year - 2020) / 4.0

        # Combine: 3D + 66D + 1D = 70D
        features = np.concatenate([annual_features, coarse_features, [year_feature]])

        return features

    def predict(
        self,
        lat: float,
        lon: float,
        year: int,
        threshold: float = 0.5
    ) -> Dict:
        """
        Make a prediction for a location.

        Args:
            lat: Latitude
            lon: Longitude
            year: Year for prediction
            threshold: Classification threshold (default 0.5)

        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_features_from_location(lat, lon, year)

        # Make prediction
        risk_probability = self.model.predict_proba([features])[0, 1]
        predicted_class = int(risk_probability >= threshold)

        # Classify confidence
        confidence = abs(risk_probability - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        if confidence >= 0.7:
            confidence_label = "high"
        elif confidence >= 0.4:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        return {
            'lat': lat,
            'lon': lon,
            'year': year,
            'risk_probability': float(risk_probability),
            'predicted_class': predicted_class,
            'threshold': threshold,
            'confidence': float(confidence),
            'confidence_label': confidence_label,
            'risk_category': self._categorize_risk(risk_probability),
            'timestamp': datetime.now().isoformat(),
        }

    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level."""
        if probability >= 0.8:
            return "very_high"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "low"
        else:
            return "very_low"

    def predict_batch(
        self,
        locations: List[Tuple[float, float, int]],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Make predictions for multiple locations.

        Args:
            locations: List of (lat, lon, year) tuples
            threshold: Classification threshold

        Returns:
            List of prediction dictionaries
        """
        results = []
        for lat, lon, year in locations:
            try:
                result = self.predict(lat, lon, year, threshold)
                results.append(result)
            except Exception as e:
                results.append({
                    'lat': lat,
                    'lon': lon,
                    'year': year,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })

        return results

    def explain_prediction(
        self,
        lat: float,
        lon: float,
        year: int,
        top_k: int = 5
    ) -> Dict:
        """
        Explain a prediction using SHAP values.

        Args:
            lat: Latitude
            lon: Longitude
            year: Year
            top_k: Number of top features to return

        Returns:
            Dictionary with SHAP explanations
        """
        try:
            import shap
        except ImportError:
            return {
                'error': 'SHAP not installed. Run: pip install shap',
                'lat': lat,
                'lon': lon,
                'year': year,
            }

        # Extract features
        features = self.extract_features_from_location(lat, lon, year)

        # Get prediction
        prediction_result = self.predict(lat, lon, year)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values([features])[0]

        # Get top contributing features
        feature_contributions = list(zip(self.feature_names, shap_values, features))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for name, shap_val, feature_val in feature_contributions[:top_k]:
            direction = "increases" if shap_val > 0 else "decreases"
            top_features.append({
                'feature': name,
                'value': float(feature_val),
                'shap_value': float(shap_val),
                'direction': direction,
                'contribution_pct': float(abs(shap_val) / (abs(shap_values).sum() + 1e-10) * 100),
            })

        return {
            **prediction_result,
            'explanation': {
                'top_features': top_features,
                'base_value': float(explainer.expected_value),
                'total_contribution': float(shap_values.sum()),
            }
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_type': type(self.model).__name__,
            'n_features': 70,
            'feature_names': self.feature_names,
            'training_samples': 847,
            'training_years': '2020-2024',
            'validation_auroc': 0.913,
            'validation_samples': 340,
            'model_date': '2025-10-23',
        }


def main():
    """Test the model service."""
    print("=" * 80)
    print("TESTING DEFORESTATION MODEL SERVICE")
    print("=" * 80)

    # Initialize service
    print("\nInitializing model service...")
    service = DeforestationModelService()

    # Test prediction
    print("\nTesting prediction for Amazon location...")
    lat, lon, year = -3.8248, -50.2500, 2024

    try:
        result = service.predict(lat, lon, year)
        print(f"\nPrediction Results:")
        print(f"  Location: ({lat}, {lon})")
        print(f"  Year: {year}")
        print(f"  Risk Probability: {result['risk_probability']:.3f}")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Risk Category: {result['risk_category']}")
        print(f"  Confidence: {result['confidence']:.3f} ({result['confidence_label']})")

        # Test explanation
        print("\nGetting SHAP explanation...")
        explanation = service.explain_prediction(lat, lon, year)

        print(f"\nTop Contributing Features:")
        for feat in explanation['explanation']['top_features']:
            print(f"  {feat['feature']}: {feat['direction']} risk by {feat['contribution_pct']:.1f}%")
            print(f"    Value: {feat['value']:.3f}, SHAP: {feat['shap_value']:.3f}")

    except Exception as e:
        print(f"  Error: {e}")

    # Model info
    print("\nModel Information:")
    info = service.get_model_info()
    for key, value in info.items():
        if key != 'feature_names':
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
