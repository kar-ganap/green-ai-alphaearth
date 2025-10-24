"""
Historical Playback - Validate Model on Past Clearings

Features:
- Timeline slider for years 2021-2024
- Show actual clearings from validation set
- Overlay model predictions made 90 days before
- Hit rate statistics
- Animated playback
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.run.model_service import DeforestationModelService
from src.utils import get_config

# Page config
st.set_page_config(page_title="Historical Playback", page_icon="⏮️", layout="wide")

# Title
st.title("⏮️ Historical Playback")
st.markdown("See how well the model predicted past deforestation events")

# Load configuration
config = get_config()
data_dir = config.get_path("paths.data_dir")

# Helper function to flatten sample structure
def flatten_samples(samples):
    """Flatten nested location structure in samples."""
    flattened = []
    for sample in samples:
        if isinstance(sample, dict):
            flat_sample = sample.copy()
            # If location is nested, flatten it
            if 'location' in sample and isinstance(sample['location'], dict):
                flat_sample['lat'] = sample['location']['lat']
                flat_sample['lon'] = sample['location']['lon']
            flattened.append(flat_sample)
    return flattened

# Load validation data with pre-extracted features
@st.cache_data(ttl=None, show_spinner=False, hash_funcs=None, max_entries=None)
def load_validation_data_with_features():
    """Load hard validation datasets with pre-extracted 70D features (2022-2024)."""
    import glob
    from pathlib import Path

    datasets = {}

    # Map use case names to file prefixes
    use_case_map = {
        'Risk Ranking': 'risk_ranking',
        'Comprehensive': 'comprehensive',
        'Rapid Response': 'rapid_response',
        'Edge Cases': 'edge_cases',
    }

    for display_name, file_prefix in use_case_map.items():
        # Find timestamped feature files for this use case (years 2020-2024)
        pattern = str(data_dir / 'processed' / f'hard_val_{file_prefix}_20??_*_features.pkl')
        feature_files = sorted(glob.glob(pattern))

        if not feature_files:
            # st.info(f"No pre-extracted features found for {display_name}")
            continue

        try:
            # Load and combine all years
            all_samples = []
            for feature_file in feature_files:
                with open(feature_file, 'rb') as f:
                    samples = pickle.load(f)
                all_samples.extend(samples)

            if all_samples:
                datasets[display_name] = all_samples
                # st.info(f"Loaded {len(all_samples)} samples for {display_name} (years 2022-2024)")
        except Exception as e:
            st.warning(f"Could not load features for {display_name}: {e}")

    return datasets

# Helper function to extract 70D feature vector from enriched sample
def extract_70d_features(sample):
    """Extract 70D feature vector from sample with pre-extracted features."""
    try:
        import numpy as np

        # Get components
        annual_features = sample.get('annual_features')
        multiscale_features = sample.get('multiscale_features')
        year_feature = sample.get('year_feature')

        if annual_features is None or multiscale_features is None or year_feature is None:
            return None

        # Convert annual features to array
        if isinstance(annual_features, (list, tuple)):
            annual_array = np.array(annual_features)
        else:
            annual_array = annual_features

        # Extract multiscale features (66D: 64 embeddings + 2 stats)
        if isinstance(multiscale_features, dict):
            coarse_feature_names = [f'coarse_emb_{i}' for i in range(64)] + ['coarse_heterogeneity', 'coarse_range']
            coarse_values = [multiscale_features.get(k, 0.0) for k in coarse_feature_names]
            multiscale_array = np.array(coarse_values)
        else:
            multiscale_array = np.array(multiscale_features)

        # Combine: 3D + 66D + 1D = 70D
        feature_vector = np.concatenate([annual_array, multiscale_array, [year_feature]])

        return feature_vector
    except Exception as e:
        return None

# Initialize model service
@st.cache_resource
def load_model_service():
    """Load model service once and cache."""
    with st.spinner("Loading model..."):
        return DeforestationModelService()

try:
    model_service = load_model_service()
    validation_data = load_validation_data_with_features()

    if not validation_data:
        st.error("No validation datasets found. Please run validation data collection scripts first.")
        st.stop()

except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Sidebar controls
st.sidebar.markdown("### Playback Controls")

# Dataset selector
selected_dataset = st.sidebar.selectbox(
    "Validation Dataset",
    options=list(validation_data.keys()),
    help="Select which validation set to analyze"
)

# Helper functions to safely extract fields - handle both flat and nested structures
def get_year(sample):
    """Safely extract year from sample, handling nested structures."""
    if 'year' in sample:
        return sample['year']
    elif 'location' in sample and isinstance(sample['location'], dict):
        return sample['location'].get('year', 2020)
    return 2020

def get_lat(sample):
    """Safely extract latitude from sample, handling nested structures."""
    if 'lat' in sample:
        return sample['lat']
    elif 'location' in sample and isinstance(sample['location'], dict):
        return sample['location'].get('lat', 0.0)
    return 0.0

def get_lon(sample):
    """Safely extract longitude from sample, handling nested structures."""
    if 'lon' in sample:
        return sample['lon']
    elif 'location' in sample and isinstance(sample['location'], dict):
        return sample['location'].get('lon', 0.0)
    return 0.0

available_years = sorted(set(get_year(sample) for sample in validation_data[selected_dataset]))
selected_years = st.sidebar.multiselect(
    "Filter by Year",
    options=available_years,
    default=available_years,
    help="Show only clearings from selected years"
)

# Threshold slider
threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Threshold for classifying as high risk"
)

# Get filtered samples
samples = validation_data[selected_dataset]
if selected_years:
    samples = [s for s in samples if get_year(s) in selected_years]

st.sidebar.markdown(f"**Total Samples**: {len(samples)}")

# Main content
st.markdown("---")

# Metrics section
st.markdown("### 📊 Performance Summary")

if len(samples) > 0:
    # Make predictions for all samples using PRE-EXTRACTED features (FAST!)
    with st.spinner(f"Evaluating {len(samples)} samples..."):
        results = []
        for sample in samples:
            try:
                lat = get_lat(sample)
                lon = get_lon(sample)
                year = get_year(sample)

                # Try to use pre-extracted features for instant predictions
                features = extract_70d_features(sample)

                if features is not None:
                    # FAST prediction using pre-extracted features (no Earth Engine call!)
                    risk_probability = model_service.model.predict_proba([features])[0, 1]
                    predicted_class = int(risk_probability >= threshold)
                    confidence = abs(risk_probability - 0.5) * 2
                    risk_category = model_service._categorize_risk(risk_probability)
                else:
                    # Fallback to slow method if features not available
                    pred = model_service.predict(lat, lon, year, threshold=threshold)
                    risk_probability = pred['risk_probability']
                    predicted_class = pred['predicted_class']
                    confidence = pred['confidence']
                    risk_category = pred['risk_category']

                results.append({
                    'lat': lat,
                    'lon': lon,
                    'year': year,
                    'label': sample.get('label', 1),  # Assume deforestation if label not present
                    'predicted_class': predicted_class,
                    'risk_probability': risk_probability,
                    'risk_category': risk_category,
                    'confidence': confidence
                })
            except Exception as e:
                st.warning(f"Failed to predict for ({get_lat(sample)}, {get_lon(sample)}): {e}")

    df_results = pd.DataFrame(results)

    # Calculate metrics
    if len(df_results) > 0:
        # True positives (correctly predicted deforestation)
        tp = ((df_results['label'] == 1) & (df_results['predicted_class'] == 1)).sum()
        # False negatives (missed deforestation)
        fn = ((df_results['label'] == 1) & (df_results['predicted_class'] == 0)).sum()
        # False positives (false alarms)
        fp = ((df_results['label'] == 0) & (df_results['predicted_class'] == 1)).sum()
        # True negatives (correctly predicted no deforestation)
        tn = ((df_results['label'] == 0) & (df_results['predicted_class'] == 0)).sum()

        total_deforestation = (df_results['label'] == 1).sum()
        total_no_deforestation = (df_results['label'] == 0).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / len(df_results) if len(df_results) > 0 else 0

        # Display metrics (for deforestation-only datasets)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Detection Rate",
                value=f"{recall:.1%}",
                help=f"Caught {tp} out of {total_deforestation} actual clearings"
            )

        with col2:
            miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            st.metric(
                label="Miss Rate",
                value=f"{miss_rate:.1%}",
                help=f"Missed {fn} out of {total_deforestation} clearings",
                delta=f"-{miss_rate:.1%}",
                delta_color="inverse"
            )

        with col3:
            avg_risk_all = df_results['risk_probability'].mean()
            st.metric(
                label="Avg Risk Score",
                value=f"{avg_risk_all:.1%}",
                help="Average predicted risk across all clearings"
            )

        with col4:
            median_risk = df_results['risk_probability'].median()
            st.metric(
                label="Median Risk Score",
                value=f"{median_risk:.1%}",
                help="Median predicted risk (50th percentile)"
            )

        # Detection Performance (more appropriate for deforestation-only validation sets)
        st.markdown("#### Detection Performance")

        col1, col2 = st.columns(2)

        with col1:
            # Detection breakdown - simple bar chart
            fig_detection = go.Figure(data=[
                go.Bar(
                    x=['Detected', 'Missed'],
                    y=[tp, fn],
                    marker_color=['green', 'red'],
                    text=[tp, fn],
                    textposition='auto',
                    textfont={"size": 20}
                )
            ])

            fig_detection.update_layout(
                title=f"Detection Results ({total_deforestation} Actual Clearings)",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_detection, use_container_width=True)

        with col2:
            # Risk score comparison: Detected vs Missed
            detected_df = df_results[df_results['predicted_class'] == 1]
            missed_df = df_results[df_results['predicted_class'] == 0]

            fig_risk_compare = go.Figure()

            if len(detected_df) > 0:
                fig_risk_compare.add_trace(go.Box(
                    y=detected_df['risk_probability'],
                    name='Detected',
                    marker_color='green',
                    boxmean='sd'
                ))

            if len(missed_df) > 0:
                fig_risk_compare.add_trace(go.Box(
                    y=missed_df['risk_probability'],
                    name='Missed',
                    marker_color='red',
                    boxmean='sd'
                ))

            fig_risk_compare.update_layout(
                title="Risk Scores: Detected vs Missed",
                yaxis_title="Predicted Risk Probability",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_risk_compare, use_container_width=True)

        # Summary stats
        st.markdown("**Detection Summary**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            **Detected**: {tp} clearings
            Avg Risk: {detected_df['risk_probability'].mean():.1%} ± {detected_df['risk_probability'].std():.1%}
            """)

        with col2:
            st.markdown(f"""
            **Missed**: {fn} clearings
            Avg Risk: {missed_df['risk_probability'].mean():.1%} ± {missed_df['risk_probability'].std():.1%}
            """)

        with col3:
            st.markdown(f"""
            **Threshold**: {threshold:.2f}
            Detection Rate: **{recall:.1%}**
            """)

        # Risk distribution
        st.markdown("---")
        st.markdown("#### Risk Distribution Analysis")

        # Add detection status column for visualization
        df_results['detection_status'] = df_results['predicted_class'].apply(
            lambda x: 'Detected' if x == 1 else 'Missed'
        )

        col1, col2 = st.columns(2)

        with col1:
            # Risk histogram by detection status (Detected vs Missed)
            fig_hist = px.histogram(
                df_results,
                x='risk_probability',
                color='detection_status',
                nbins=20,
                title="Risk Probability Distribution",
                labels={'detection_status': 'Detection Status', 'risk_probability': 'Predicted Risk'},
                color_discrete_map={'Detected': 'green', 'Missed': 'red'},
                barmode='overlay',
                opacity=0.7
            )

            # Add threshold line
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Threshold: {threshold}",
                annotation_position="top"
            )

            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Risk category breakdown by detection status
            risk_cat_counts = df_results.groupby(['risk_category', 'detection_status']).size().reset_index(name='count')

            fig_cat = px.bar(
                risk_cat_counts,
                x='risk_category',
                y='count',
                color='detection_status',
                title="Detection Status by Risk Category",
                labels={'detection_status': 'Detection Status', 'count': 'Count'},
                color_discrete_map={'Detected': 'green', 'Missed': 'red'},
                barmode='group',
                category_orders={'risk_category': ['very_low', 'low', 'medium', 'high', 'very_high']}
            )

            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Temporal analysis
        st.markdown("---")
        st.markdown("#### Temporal Analysis")

        # Group by year
        yearly_stats = df_results.groupby('year').agg({
            'label': 'sum',  # Total actual clearings
            'predicted_class': 'sum',  # Total predicted clearings
            'risk_probability': 'mean'  # Average risk
        }).reset_index()

        yearly_stats.columns = ['Year', 'Actual Clearings', 'Predicted Clearings', 'Avg Risk']

        # Calculate recall per year
        yearly_recall = []
        for year in yearly_stats['Year']:
            year_data = df_results[df_results['year'] == year]
            year_tp = ((year_data['label'] == 1) & (year_data['predicted_class'] == 1)).sum()
            year_total = (year_data['label'] == 1).sum()
            year_recall = year_tp / year_total if year_total > 0 else 0
            yearly_recall.append(year_recall)

        yearly_stats['Recall'] = yearly_recall

        # Plot yearly performance
        fig_yearly = go.Figure()

        fig_yearly.add_trace(go.Bar(
            x=yearly_stats['Year'],
            y=yearly_stats['Actual Clearings'],
            name='Actual Clearings',
            marker_color='red'
        ))

        fig_yearly.add_trace(go.Bar(
            x=yearly_stats['Year'],
            y=yearly_stats['Predicted Clearings'],
            name='Predicted Clearings',
            marker_color='orange'
        ))

        fig_yearly.add_trace(go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['Recall'],
            name='Recall',
            yaxis='y2',
            marker_color='green',
            line=dict(width=3)
        ))

        fig_yearly.update_layout(
            title="Performance by Year",
            xaxis_title="Year",
            yaxis_title="Count",
            yaxis2=dict(
                title="Recall",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            height=400,
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig_yearly, use_container_width=True)

        # Data table
        st.markdown("---")
        st.markdown("#### Detailed Results")

        # Add interpretation column
        df_display = df_results.copy()
        df_display['Result'] = df_display.apply(
            lambda row: 'True Positive' if row['label'] == 1 and row['predicted_class'] == 1
            else 'False Negative' if row['label'] == 1 and row['predicted_class'] == 0
            else 'False Positive' if row['label'] == 0 and row['predicted_class'] == 1
            else 'True Negative',
            axis=1
        )

        # Format and display
        st.dataframe(
            df_display[['lat', 'lon', 'year', 'label', 'risk_probability', 'predicted_class', 'risk_category', 'confidence', 'Result']]
            .sort_values('risk_probability', ascending=False)
            .style.format({
                'lat': '{:.4f}',
                'lon': '{:.4f}',
                'risk_probability': '{:.1%}',
                'confidence': '{:.1%}'
            }),
            use_container_width=True,
            height=400
        )

        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"historical_playback_{selected_dataset.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

    else:
        st.warning("No results to display")

else:
    st.info("No samples found for the selected criteria")
