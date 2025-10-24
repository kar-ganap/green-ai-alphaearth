"""
Prediction Explorer - Interactive Map for Single Location Predictions

Features:
- Interactive map with click-to-predict
- Manual coordinate entry
- Year selector
- Risk visualization
- SHAP explanation waterfall chart
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.run.model_service import DeforestationModelService

# Page config
st.set_page_config(page_title="Prediction Explorer", page_icon="🗺️", layout="wide")

# Title
st.title("🗺️ Prediction Explorer")
st.markdown("Click on the map or enter coordinates to predict deforestation risk")

# Initialize model service in session state
@st.cache_resource
def load_model_service():
    """Load model service once and cache."""
    with st.spinner("Loading model..."):
        return DeforestationModelService()

try:
    model_service = load_model_service()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Please ensure the trained model exists at: data/processed/final_xgb_model_2020_2024.pkl")
    model_loaded = False
    st.stop()

# Initialize session state for location
if 'prediction_lat' not in st.session_state:
    st.session_state.prediction_lat = -3.8248
if 'prediction_lon' not in st.session_state:
    st.session_state.prediction_lon = -50.2500
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'explanation_result' not in st.session_state:
    st.session_state.explanation_result = None

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🌍 Interactive Map")
    st.markdown("Click anywhere in the Amazon to predict deforestation risk")

    # Create map centered on Amazon
    m = folium.Map(
        location=[st.session_state.prediction_lat, st.session_state.prediction_lon],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    # Add marker for current location
    if st.session_state.prediction_result:
        risk_prob = st.session_state.prediction_result['risk_probability']
        risk_category = st.session_state.prediction_result['risk_category']

        # Color based on risk
        if risk_prob >= 0.8:
            color = 'red'
        elif risk_prob >= 0.6:
            color = 'orange'
        elif risk_prob >= 0.4:
            color = 'yellow'
        elif risk_prob >= 0.2:
            color = 'lightgreen'
        else:
            color = 'green'

        folium.Marker(
            location=[st.session_state.prediction_lat, st.session_state.prediction_lon],
            popup=f"Risk: {risk_prob:.1%} ({risk_category})",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

    # Display map and capture clicks
    map_data = st_folium(m, width=700, height=500, key="prediction_map")

    # Update location if map was clicked
    if map_data and map_data.get('last_clicked'):
        st.session_state.prediction_lat = map_data['last_clicked']['lat']
        st.session_state.prediction_lon = map_data['last_clicked']['lng']

with col2:
    st.markdown("### 📍 Prediction Settings")

    # Manual coordinate entry
    with st.form("prediction_form"):
        lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.prediction_lat,
            step=0.01,
            format="%.4f"
        )

        lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.prediction_lon,
            step=0.01,
            format="%.4f"
        )

        year = st.selectbox(
            "Prediction Year",
            options=[2020, 2021, 2022, 2023, 2024, 2025],
            index=4
        )

        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Threshold for classifying as high risk"
        )

        submitted = st.form_submit_button("🔍 Predict Risk", type="primary", use_container_width=True)

        if submitted:
            # Update session state
            st.session_state.prediction_lat = lat
            st.session_state.prediction_lon = lon

            # Make prediction
            with st.spinner("Extracting features and predicting..."):
                try:
                    result = model_service.predict(lat, lon, year, threshold)
                    st.session_state.prediction_result = result

                    # Get explanation
                    explanation = model_service.explain_prediction(lat, lon, year, top_k=10)
                    st.session_state.explanation_result = explanation

                    st.success("Prediction complete!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.session_state.prediction_result = None
                    st.session_state.explanation_result = None

# Display results
if st.session_state.prediction_result:
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    result = st.session_state.prediction_result

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Risk Probability",
            value=f"{result['risk_probability']:.1%}",
            delta=None
        )

    with col2:
        risk_category = result['risk_category'].replace('_', ' ').title()
        st.metric(
            label="Risk Category",
            value=risk_category,
            delta=None
        )

    with col3:
        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.1%}",
            delta=None
        )

    with col4:
        confidence_label = result['confidence_label'].title()
        st.metric(
            label="Confidence Level",
            value=confidence_label,
            delta=None
        )

    # Risk gauge
    st.markdown("#### Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['risk_probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Deforestation Risk (%)", 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#388e3c'},
                {'range': [20, 40], 'color': '#689f38'},
                {'range': [40, 60], 'color': '#fbc02d'},
                {'range': [60, 80], 'color': '#f57c00'},
                {'range': [80, 100], 'color': '#d32f2f'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# Display SHAP explanation
if st.session_state.explanation_result and 'explanation' in st.session_state.explanation_result:
    st.markdown("---")
    st.markdown("### 🔍 SHAP Explanation - What Drives This Prediction?")

    explanation = st.session_state.explanation_result['explanation']
    top_features = explanation['top_features']

    # Create waterfall chart
    st.markdown("#### Top 10 Contributing Features")

    # Prepare data for waterfall
    feature_names = [f['feature'] for f in top_features]
    shap_values = [f['shap_value'] for f in top_features]
    feature_values = [f['value'] for f in top_features]

    # Create DataFrame for display
    df_features = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values,
        'SHAP Value': shap_values,
        'Direction': [f['direction'].title() for f in top_features],
        'Contribution %': [f['contribution_pct'] for f in top_features]
    })

    # Horizontal bar chart
    colors = ['red' if x > 0 else 'blue' for x in shap_values]

    fig_shap = go.Figure(go.Bar(
        x=shap_values,
        y=feature_names,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:.3f}" for val in shap_values],
        textposition='auto',
    ))

    fig_shap.update_layout(
        title="Feature Contributions to Prediction (SHAP Values)",
        xaxis_title="SHAP Value (← Decreases Risk | Increases Risk →)",
        yaxis_title="Feature",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig_shap, use_container_width=True)

    # Feature details table
    st.markdown("#### Feature Details")
    st.dataframe(
        df_features.style.format({
            'Value': '{:.3f}',
            'SHAP Value': '{:.3f}',
            'Contribution %': '{:.1f}%'
        }),
        use_container_width=True
    )

    # Interpretation guide
    with st.expander("📖 How to interpret SHAP values"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values explain individual predictions:

        - **Positive SHAP values** (red bars): Features that increase deforestation risk
        - **Negative SHAP values** (blue bars): Features that decrease deforestation risk
        - **Magnitude**: Larger absolute values = stronger influence on prediction
        - **Contribution %**: Percentage of total explanation attributed to this feature

        **Key Features**:
        - `delta_1yr`: Change in forest cover from 1 year ago
        - `delta_2yr`: Change in forest cover from 2 years ago
        - `acceleration`: Rate of forest change acceleration
        - `coarse_emb_X`: Landscape context embeddings (64 dimensions)
        - `coarse_heterogeneity`: Landscape complexity
        - `coarse_range`: Landscape variability
        - `normalized_year`: Year of prediction (temporal trend)

        **Base value**: {:.3f} - Average model prediction across all training data
        """.format(explanation['base_value']))

else:
    st.info("👆 Enter coordinates and click 'Predict Risk' to see results and SHAP explanation")
