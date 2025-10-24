"""
ROI Calculator - Cost-Benefit Analysis for Early Warning System

Features:
- Interactive sliders for cost/benefit parameters
- Real-time ROI calculation
- Break-even analysis
- Sensitivity analysis charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="ROI Calculator", page_icon="💰", layout="wide")

# Title
st.title("💰 ROI Calculator")
st.markdown("Estimate the cost-benefit of deploying this early warning system")

# Introduction
st.markdown("""
This calculator helps you estimate the Return on Investment (ROI) of deploying the deforestation early warning system.
Adjust the parameters below to match your specific context and budget.
""")

st.markdown("---")

# Input parameters
st.markdown("### 📝 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Costs")

    cost_per_investigation = st.number_input(
        "Cost per Alert Investigation ($)",
        min_value=0,
        max_value=10000,
        value=500,
        step=50,
        help="Cost to investigate one alert (personnel, travel, equipment)"
    )

    fixed_annual_cost = st.number_input(
        "Fixed Annual Costs ($)",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=1000,
        help="Annual fixed costs (infrastructure, maintenance, training)"
    )

    st.markdown("#### Model Performance (from validation)")

    # Default values from hard validation set
    total_clearings = st.number_input(
        "Total Clearings to Monitor",
        min_value=1,
        max_value=10000,
        value=340,
        step=10,
        help="Total number of potential clearings to monitor"
    )

    recall_rate = st.slider(
        "Model Recall (Catch Rate)",
        min_value=0.0,
        max_value=1.0,
        value=0.78,
        step=0.01,
        format="%.0f%%",
        help="Proportion of clearings caught by the model (from validation)"
    )

    false_positive_rate = st.slider(
        "False Positive Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        format="%.0f%%",
        help="Proportion of false alarms"
    )

with col2:
    st.markdown("#### Benefits")

    avg_clearing_size_ha = st.number_input(
        "Average Clearing Size (hectares)",
        min_value=0.1,
        max_value=1000.0,
        value=10.0,
        step=0.5,
        help="Average size of a deforestation event"
    )

    ecosystem_value_per_ha = st.number_input(
        "Ecosystem Value per Hectare ($)",
        min_value=0,
        max_value=100000,
        value=5000,
        step=100,
        help="Economic value of forest ecosystem services per hectare per year"
    )

    carbon_value_per_ha = st.number_input(
        "Carbon Value per Hectare ($)",
        min_value=0,
        max_value=50000,
        value=2000,
        step=100,
        help="Value of carbon credits per hectare"
    )

    enforcement_success_rate = st.slider(
        "Enforcement Success Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        format="%.0f%%",
        help="Proportion of alerts that result in successful intervention"
    )

    st.markdown("#### Additional Benefits (Optional)")

    biodiversity_value = st.number_input(
        "Biodiversity Conservation Value ($)",
        min_value=0,
        max_value=1000000,
        value=0,
        step=1000,
        help="Additional value from protecting biodiversity"
    )

st.markdown("---")

# Calculate ROI
st.markdown("### 📊 ROI Analysis")

# Calculations
true_positives = int(total_clearings * recall_rate)
false_positives = int(total_clearings * false_positive_rate)
total_alerts = true_positives + false_positives

successful_interventions = int(true_positives * enforcement_success_rate)
hectares_saved = successful_interventions * avg_clearing_size_ha

# Costs
investigation_costs = total_alerts * cost_per_investigation
total_annual_costs = investigation_costs + fixed_annual_cost

# Benefits (annual)
ecosystem_benefits = hectares_saved * ecosystem_value_per_ha
carbon_benefits = hectares_saved * carbon_value_per_ha
total_annual_benefits = ecosystem_benefits + carbon_benefits + biodiversity_value

# Net benefit and ROI
net_benefit = total_annual_benefits - total_annual_costs
roi_percent = (net_benefit / total_annual_costs * 100) if total_annual_costs > 0 else 0

# Break-even point
break_even_hectares = total_annual_costs / (ecosystem_value_per_ha + carbon_value_per_ha) if (ecosystem_value_per_ha + carbon_value_per_ha) > 0 else 0

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Annual Benefits",
        value=f"${total_annual_benefits:,.0f}",
        help="Total economic value generated per year"
    )

with col2:
    st.metric(
        label="Total Annual Costs",
        value=f"${total_annual_costs:,.0f}",
        help="Total costs per year"
    )

with col3:
    st.metric(
        label="Net Annual Benefit",
        value=f"${net_benefit:,.0f}",
        delta=f"ROI: {roi_percent:.0f}%",
        delta_color="normal" if net_benefit > 0 else "inverse"
    )

with col4:
    st.metric(
        label="Hectares Protected",
        value=f"{hectares_saved:,.1f}",
        help="Forest area protected per year"
    )

# Detailed breakdown
st.markdown("---")
st.markdown("#### Detailed Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Costs Breakdown**")

    cost_data = pd.DataFrame({
        'Category': ['Alert Investigations', 'Fixed Costs'],
        'Amount': [investigation_costs, fixed_annual_cost]
    })

    fig_costs = px.pie(
        cost_data,
        values='Amount',
        names='Category',
        title='Annual Cost Distribution',
        color_discrete_sequence=['#e74c3c', '#c0392b']
    )

    fig_costs.update_layout(height=350)
    st.plotly_chart(fig_costs, use_container_width=True)

    st.markdown(f"""
    - **Total Alerts**: {total_alerts:,}
      - True Positives: {true_positives:,}
      - False Positives: {false_positives:,}
    - **Investigation Cost per Alert**: ${cost_per_investigation:,}
    - **Total Investigation Costs**: ${investigation_costs:,}
    - **Fixed Annual Costs**: ${fixed_annual_cost:,}
    """)

with col2:
    st.markdown("**Benefits Breakdown**")

    benefit_data = pd.DataFrame({
        'Category': ['Ecosystem Services', 'Carbon Credits', 'Biodiversity'],
        'Amount': [ecosystem_benefits, carbon_benefits, biodiversity_value]
    })

    fig_benefits = px.pie(
        benefit_data,
        values='Amount',
        names='Category',
        title='Annual Benefit Distribution',
        color_discrete_sequence=['#27ae60', '#229954', '#1e8449']
    )

    fig_benefits.update_layout(height=350)
    st.plotly_chart(fig_benefits, use_container_width=True)

    st.markdown(f"""
    - **Successful Interventions**: {successful_interventions:,}
    - **Hectares Protected**: {hectares_saved:,.1f}
    - **Ecosystem Value**: ${ecosystem_benefits:,}
    - **Carbon Value**: ${carbon_benefits:,}
    - **Biodiversity Value**: ${biodiversity_value:,}
    """)

# Break-even analysis
st.markdown("---")
st.markdown("#### Break-Even Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Break-Even Point",
        value=f"{break_even_hectares:,.1f} hectares",
        help="Minimum hectares needed to protect to break even"
    )

    if hectares_saved > break_even_hectares:
        margin = hectares_saved - break_even_hectares
        st.success(f"✅ System is profitable! {margin:,.1f} hectares above break-even.")
    else:
        shortfall = break_even_hectares - hectares_saved
        st.warning(f"⚠️ System needs to protect {shortfall:,.1f} more hectares to break even.")

    # Payback period
    if net_benefit > 0:
        payback_years = fixed_annual_cost / net_benefit
        st.info(f"💡 **Payback Period**: {payback_years:.1f} years")

with col2:
    # Break-even chart
    hectares_range = np.linspace(0, hectares_saved * 2, 100)
    costs_line = np.full_like(hectares_range, total_annual_costs)
    benefits_line = hectares_range * (ecosystem_value_per_ha + carbon_value_per_ha) + biodiversity_value

    fig_breakeven = go.Figure()

    fig_breakeven.add_trace(go.Scatter(
        x=hectares_range,
        y=costs_line,
        mode='lines',
        name='Total Costs',
        line=dict(color='red', width=2)
    ))

    fig_breakeven.add_trace(go.Scatter(
        x=hectares_range,
        y=benefits_line,
        mode='lines',
        name='Total Benefits',
        line=dict(color='green', width=2)
    ))

    # Add current point
    fig_breakeven.add_trace(go.Scatter(
        x=[hectares_saved],
        y=[total_annual_benefits],
        mode='markers',
        name='Current Performance',
        marker=dict(size=12, color='blue')
    ))

    # Add break-even point
    fig_breakeven.add_trace(go.Scatter(
        x=[break_even_hectares],
        y=[total_annual_costs],
        mode='markers',
        name='Break-Even',
        marker=dict(size=12, color='orange', symbol='star')
    ))

    fig_breakeven.update_layout(
        title='Break-Even Analysis',
        xaxis_title='Hectares Protected per Year',
        yaxis_title='Dollar Value ($)',
        height=350
    )

    st.plotly_chart(fig_breakeven, use_container_width=True)

# Sensitivity analysis
st.markdown("---")
st.markdown("#### Sensitivity Analysis")

st.markdown("See how ROI changes with different parameter values:")

sensitivity_param = st.selectbox(
    "Parameter to Analyze",
    options=[
        "Enforcement Success Rate",
        "Ecosystem Value per Hectare",
        "Cost per Investigation",
        "Model Recall (Catch Rate)"
    ]
)

# Generate sensitivity data
if sensitivity_param == "Enforcement Success Rate":
    param_range = np.linspace(0, 1, 50)
    roi_values = []
    for val in param_range:
        interv = int(true_positives * val)
        ha = interv * avg_clearing_size_ha
        benefits = ha * (ecosystem_value_per_ha + carbon_value_per_ha) + biodiversity_value
        roi = ((benefits - total_annual_costs) / total_annual_costs * 100) if total_annual_costs > 0 else 0
        roi_values.append(roi)
    x_label = "Enforcement Success Rate"
    x_values = param_range

elif sensitivity_param == "Ecosystem Value per Hectare":
    param_range = np.linspace(0, ecosystem_value_per_ha * 3, 50)
    roi_values = []
    for val in param_range:
        benefits = hectares_saved * (val + carbon_value_per_ha) + biodiversity_value
        roi = ((benefits - total_annual_costs) / total_annual_costs * 100) if total_annual_costs > 0 else 0
        roi_values.append(roi)
    x_label = "Ecosystem Value per Hectare ($)"
    x_values = param_range

elif sensitivity_param == "Cost per Investigation":
    param_range = np.linspace(0, cost_per_investigation * 3, 50)
    roi_values = []
    for val in param_range:
        costs = total_alerts * val + fixed_annual_cost
        roi = ((total_annual_benefits - costs) / costs * 100) if costs > 0 else 0
        roi_values.append(roi)
    x_label = "Cost per Investigation ($)"
    x_values = param_range

else:  # Model Recall
    param_range = np.linspace(0, 1, 50)
    roi_values = []
    for val in param_range:
        tp = int(total_clearings * val)
        alerts = tp + false_positives
        interv = int(tp * enforcement_success_rate)
        ha = interv * avg_clearing_size_ha
        costs = alerts * cost_per_investigation + fixed_annual_cost
        benefits = ha * (ecosystem_value_per_ha + carbon_value_per_ha) + biodiversity_value
        roi = ((benefits - costs) / costs * 100) if costs > 0 else 0
        roi_values.append(roi)
    x_label = "Model Recall (Catch Rate)"
    x_values = param_range

# Plot sensitivity
fig_sensitivity = go.Figure()

fig_sensitivity.add_trace(go.Scatter(
    x=x_values,
    y=roi_values,
    mode='lines',
    line=dict(color='blue', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 0, 255, 0.1)'
))

# Add current value marker
if sensitivity_param == "Enforcement Success Rate":
    current_val = enforcement_success_rate
elif sensitivity_param == "Ecosystem Value per Hectare":
    current_val = ecosystem_value_per_ha
elif sensitivity_param == "Cost per Investigation":
    current_val = cost_per_investigation
else:
    current_val = recall_rate

current_roi = roi_percent
fig_sensitivity.add_trace(go.Scatter(
    x=[current_val],
    y=[current_roi],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Current Value'
))

# Add zero line
fig_sensitivity.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-Even")

fig_sensitivity.update_layout(
    title=f'ROI Sensitivity to {sensitivity_param}',
    xaxis_title=x_label,
    yaxis_title='ROI (%)',
    height=400,
    showlegend=True
)

st.plotly_chart(fig_sensitivity, use_container_width=True)

# Summary
st.markdown("---")
st.markdown("### 📋 Summary")

if roi_percent > 100:
    st.success(f"""
    **Excellent ROI!** The system generates **${roi_percent:.0f}** in value for every **$100** invested.

    **Key Highlights**:
    - **{successful_interventions:,}** successful interventions per year
    - **{hectares_saved:,.1f}** hectares of forest protected
    - **${total_annual_benefits:,.0f}** in annual benefits
    - **${net_benefit:,.0f}** net annual benefit
    """)
elif roi_percent > 0:
    st.info(f"""
    **Positive ROI.** The system generates **${100 + roi_percent:.0f}** in value for every **$100** invested.

    **Key Highlights**:
    - **{successful_interventions:,}** successful interventions per year
    - **{hectares_saved:,.1f}** hectares of forest protected
    - **${total_annual_benefits:,.0f}** in annual benefits
    - **${net_benefit:,.0f}** net annual benefit

    **Recommendation**: System is profitable but could benefit from optimization.
    """)
else:
    st.warning(f"""
    **Negative ROI.** The system costs more than it generates in value.

    **Current Situation**:
    - **{successful_interventions:,}** successful interventions per year
    - **{hectares_saved:,.1f}** hectares of forest protected
    - **${total_annual_benefits:,.0f}** in annual benefits
    - **${abs(net_benefit):,.0f}** net annual loss

    **Recommendations**:
    1. Increase enforcement success rate (currently {enforcement_success_rate:.0%})
    2. Reduce investigation costs (currently ${cost_per_investigation:,})
    3. Improve model recall (currently {recall_rate:.0%})
    4. Consider ecosystem valuation methodology
    """)

# Export results
st.markdown("---")

export_data = pd.DataFrame({
    'Parameter': [
        'Total Clearings',
        'Model Recall',
        'False Positive Rate',
        'Total Alerts',
        'True Positives',
        'False Positives',
        'Enforcement Success Rate',
        'Successful Interventions',
        'Hectares Protected',
        'Avg Clearing Size (ha)',
        'Ecosystem Value ($/ha)',
        'Carbon Value ($/ha)',
        'Cost per Investigation ($)',
        'Fixed Annual Costs ($)',
        'Total Annual Costs ($)',
        'Total Annual Benefits ($)',
        'Net Annual Benefit ($)',
        'ROI (%)',
        'Break-Even (hectares)'
    ],
    'Value': [
        total_clearings,
        f"{recall_rate:.0%}",
        f"{false_positive_rate:.0%}",
        total_alerts,
        true_positives,
        false_positives,
        f"{enforcement_success_rate:.0%}",
        successful_interventions,
        f"{hectares_saved:.1f}",
        avg_clearing_size_ha,
        ecosystem_value_per_ha,
        carbon_value_per_ha,
        cost_per_investigation,
        fixed_annual_cost,
        total_annual_costs,
        total_annual_benefits,
        net_benefit,
        f"{roi_percent:.1f}",
        f"{break_even_hectares:.1f}"
    ]
})

csv = export_data.to_csv(index=False)
st.download_button(
    label="📥 Download ROI Analysis",
    data=csv,
    file_name="deforestation_roi_analysis.csv",
    mime="text/csv"
)
