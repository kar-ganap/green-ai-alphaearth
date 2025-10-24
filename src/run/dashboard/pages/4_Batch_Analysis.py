"""
Batch Analysis - Upload CSV for Bulk Predictions

Features:
- Upload CSV with locations (lat, lon, year)
- Batch predictions
- Download results
- Risk heatmap
- Priority ranking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path
import io

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.run.model_service import DeforestationModelService

# Page config
st.set_page_config(page_title="Batch Analysis", page_icon="📊", layout="wide")

# Title
st.title("📊 Batch Analysis")
st.markdown("Upload a CSV file to predict deforestation risk for multiple locations at once")

# Initialize model service
@st.cache_resource
def load_model_service():
    """Load model service once and cache."""
    with st.spinner("Loading model..."):
        return DeforestationModelService()

try:
    model_service = load_model_service()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Instructions
with st.expander("📖 CSV Format Instructions", expanded=False):
    st.markdown("""
    **Required columns**:
    - `lat`: Latitude (decimal degrees, -90 to 90)
    - `lon`: Longitude (decimal degrees, -180 to 180)
    - `year`: Year for prediction (2020-2030)

    **Optional columns**:
    - `location_id`: Unique identifier for each location
    - `name`: Location name/description

    **Example**:
    ```csv
    location_id,name,lat,lon,year
    1,Site A,-3.8248,-50.2500,2024
    2,Site B,-3.2356,-50.4530,2024
    3,Site C,-4.1234,-51.5678,2025
    ```
    """)

    # Provide sample CSV
    sample_data = pd.DataFrame({
        'location_id': [1, 2, 3],
        'name': ['Amazon Site A', 'Amazon Site B', 'Amazon Site C'],
        'lat': [-3.8248, -3.2356, -4.1234],
        'lon': [-50.2500, -50.4530, -51.5678],
        'year': [2024, 2024, 2024]
    })

    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_sample,
        file_name="sample_locations.csv",
        mime="text/csv"
    )

st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=['csv'],
    help="Upload a CSV file with lat, lon, year columns"
)

# Threshold setting
threshold = st.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Threshold for classifying as high risk"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df_input = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ['lat', 'lon', 'year']
        missing_cols = [col for col in required_cols if col not in df_input.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # Display input data
        st.markdown("### 📋 Uploaded Data")
        st.dataframe(df_input.head(10), use_container_width=True)
        st.info(f"Total locations: {len(df_input)}")

        if len(df_input) > 100:
            st.warning(f"⚠️ File contains {len(df_input)} locations. Processing only first 100 (API limit).")
            df_input = df_input.head(100)

        # Run predictions button
        if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
            with st.spinner(f"Making predictions for {len(df_input)} locations..."):
                # Prepare locations
                locations = [(row['lat'], row['lon'], row['year']) for _, row in df_input.iterrows()]

                # Make predictions
                results = model_service.predict_batch(locations, threshold=threshold)

                # Combine with input data
                df_results = pd.DataFrame(results)

                # Merge with original data (preserve location_id and name if present)
                if 'location_id' in df_input.columns:
                    df_results['location_id'] = df_input['location_id'].values
                if 'name' in df_input.columns:
                    df_results['name'] = df_input['name'].values

                # Store in session state
                st.session_state.batch_results = df_results

            st.success("✅ Predictions complete!")
            st.rerun()

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

# Display results
if 'batch_results' in st.session_state:
    df_results = st.session_state.batch_results

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_risk = df_results['risk_probability'].mean()
        st.metric(
            label="Average Risk",
            value=f"{avg_risk:.1%}"
        )

    with col2:
        high_risk_count = (df_results['predicted_class'] == 1).sum()
        st.metric(
            label="High Risk Locations",
            value=high_risk_count,
            delta=f"{high_risk_count / len(df_results):.0%}"
        )

    with col3:
        max_risk = df_results['risk_probability'].max()
        st.metric(
            label="Maximum Risk",
            value=f"{max_risk:.1%}"
        )

    with col4:
        avg_confidence = df_results['confidence'].mean()
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.1%}"
        )

    # Visualizations
    st.markdown("#### Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Risk histogram
        fig_hist = px.histogram(
            df_results,
            x='risk_probability',
            nbins=20,
            title="Risk Probability Distribution",
            labels={'risk_probability': 'Risk Probability'},
            color_discrete_sequence=['#3498db']
        )

        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}",
            annotation_position="top right"
        )

        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Risk category breakdown
        risk_cat_counts = df_results['risk_category'].value_counts().reset_index()
        risk_cat_counts.columns = ['risk_category', 'count']

        # Define color mapping
        color_map = {
            'very_low': '#388e3c',
            'low': '#689f38',
            'medium': '#fbc02d',
            'high': '#f57c00',
            'very_high': '#d32f2f'
        }

        fig_cat = px.bar(
            risk_cat_counts,
            x='risk_category',
            y='count',
            title="Locations by Risk Category",
            labels={'count': 'Count', 'risk_category': 'Risk Category'},
            color='risk_category',
            color_discrete_map=color_map,
            category_orders={'risk_category': ['very_low', 'low', 'medium', 'high', 'very_high']}
        )

        fig_cat.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Geographic visualization (scatter map)
    st.markdown("#### Geographic Risk Map")

    # Create scatter map
    fig_map = px.scatter_geo(
        df_results,
        lat='lat',
        lon='lon',
        color='risk_probability',
        size='risk_probability',
        hover_name='name' if 'name' in df_results.columns else None,
        hover_data={
            'lat': ':.4f',
            'lon': ':.4f',
            'risk_probability': ':.1%',
            'risk_category': True,
            'confidence': ':.1%'
        },
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 1],
        title="Risk Map"
    )

    fig_map.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="lightgray",
        showcoastlines=True,
        coastlinecolor="gray",
        center=dict(lat=df_results['lat'].mean(), lon=df_results['lon'].mean()),
        projection_scale=5
    )

    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    # Priority ranking
    st.markdown("---")
    st.markdown("#### Priority Ranking")

    # Sort by risk (descending)
    df_priority = df_results.sort_values('risk_probability', ascending=False).reset_index(drop=True)
    df_priority['rank'] = df_priority.index + 1

    # Display top 10 high-priority locations
    st.markdown("**Top 10 Highest Risk Locations**")

    display_cols = ['rank', 'lat', 'lon', 'year', 'risk_probability', 'risk_category', 'confidence']
    if 'name' in df_priority.columns:
        display_cols.insert(1, 'name')
    if 'location_id' in df_priority.columns:
        display_cols.insert(1, 'location_id')

    st.dataframe(
        df_priority[display_cols].head(10).style.format({
            'lat': '{:.4f}',
            'lon': '{:.4f}',
            'risk_probability': '{:.1%}',
            'confidence': '{:.1%}'
        }).background_gradient(subset=['risk_probability'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )

    # Full results table
    st.markdown("---")
    st.markdown("#### Complete Results")

    st.dataframe(
        df_priority[display_cols].style.format({
            'lat': '{:.4f}',
            'lon': '{:.4f}',
            'risk_probability': '{:.1%}',
            'confidence': '{:.1%}'
        }),
        use_container_width=True,
        height=400
    )

    # Export results
    col1, col2 = st.columns(2)

    with col1:
        # Full results CSV
        csv_full = df_priority.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_full,
            file_name="batch_predictions_full.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # High-risk only CSV
        df_high_risk = df_priority[df_priority['predicted_class'] == 1]
        csv_high_risk = df_high_risk.to_csv(index=False)
        st.download_button(
            label="📥 Download High Risk Only (CSV)",
            data=csv_high_risk,
            file_name="batch_predictions_high_risk.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Statistics summary
    st.markdown("---")
    st.markdown("#### Summary Statistics")

    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Locations',
            'High Risk (predicted_class=1)',
            'Low Risk (predicted_class=0)',
            'Average Risk Probability',
            'Median Risk Probability',
            'Std Dev Risk',
            'Average Confidence',
            'Very High Risk (≥80%)',
            'High Risk (60-80%)',
            'Medium Risk (40-60%)',
            'Low Risk (20-40%)',
            'Very Low Risk (<20%)'
        ],
        'Value': [
            len(df_results),
            (df_results['predicted_class'] == 1).sum(),
            (df_results['predicted_class'] == 0).sum(),
            f"{df_results['risk_probability'].mean():.1%}",
            f"{df_results['risk_probability'].median():.1%}",
            f"{df_results['risk_probability'].std():.1%}",
            f"{df_results['confidence'].mean():.1%}",
            (df_results['risk_probability'] >= 0.8).sum(),
            ((df_results['risk_probability'] >= 0.6) & (df_results['risk_probability'] < 0.8)).sum(),
            ((df_results['risk_probability'] >= 0.4) & (df_results['risk_probability'] < 0.6)).sum(),
            ((df_results['risk_probability'] >= 0.2) & (df_results['risk_probability'] < 0.4)).sum(),
            (df_results['risk_probability'] < 0.2).sum()
        ]
    })

    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

else:
    st.info("👆 Upload a CSV file to get started")
