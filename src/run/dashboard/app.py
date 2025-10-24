"""
Deforestation Early Warning System - Interactive Dashboard

Main Streamlit application with multiple pages:
- Prediction Explorer: Interactive map for single predictions
- Historical Playback: Validate model on past clearings
- ROI Calculator: Cost-benefit analysis
- Batch Analysis: Upload CSV for bulk predictions
- Model Performance: Metrics dashboard

Run with: streamlit run src/run/dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="Deforestation Early Warning",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-very-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-high {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-medium {
        color: #fbc02d;
        font-weight: bold;
    }
    .risk-low {
        color: #689f38;
        font-weight: bold;
    }
    .risk-very-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<div class="main-header">🌳 Deforestation Early Warning System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered 90-day deforestation risk prediction for the Amazon rainforest</div>',
    unsafe_allow_html=True
)

# Welcome section
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **Interactive Map**: Click to predict risk for any location
    - **SHAP Explanations**: Understand what drives predictions
    - **Historical Playback**: See past performance (2021-2024)
    - **ROI Calculator**: Estimate cost-benefit of intervention
    - **Batch Analysis**: Upload CSV for bulk predictions
    """)

with col2:
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    - **Model**: XGBoost Classifier
    - **AUROC**: 0.913 on hard validation
    - **Training**: 847 samples (2020-2024)
    - **Validation**: 340 challenging samples
    - **Features**: 70D (Annual + Multiscale + Year)
    """)

with col3:
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Navigate to **Prediction Explorer** (sidebar)
    2. Click on the map or enter coordinates
    3. Select year and view risk prediction
    4. Explore SHAP explanation
    5. Try other pages for advanced features
    """)

st.markdown("---")

# Technology section
st.markdown("### 🔧 Technology Stack")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Data**")
    st.markdown("- AlphaEarth embeddings")
    st.markdown("- Google Earth Engine")
    st.markdown("- Hansen Global Forest Change")
with col2:
    st.markdown("**ML**")
    st.markdown("- XGBoost 2.0+")
    st.markdown("- SHAP explanations")
    st.markdown("- Scikit-learn")
with col3:
    st.markdown("**Backend**")
    st.markdown("- FastAPI REST API")
    st.markdown("- Python 3.9+")
    st.markdown("- NumPy/Pandas")
with col4:
    st.markdown("**Frontend**")
    st.markdown("- Streamlit")
    st.markdown("- Plotly charts")
    st.markdown("- Folium maps")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Deforestation Early Warning System v1.0</strong></p>
    <p>Built with AlphaEarth embeddings | Powered by Google Earth Engine</p>
    <p>🌳 Protecting tropical forests with AI 🌳</p>
</div>
""", unsafe_allow_html=True)
