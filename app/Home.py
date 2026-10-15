"""Main Streamlit operational overview dashboard for AI-Powered Food Waste Prediction and Redistribution Intelligence."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(
    page_title="Food Waste Prediction & Redistribution Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

user = render_sidebar_auth()

st.markdown(
    """
    <div class="env-banner">
        <h2 style="margin:0;color:#ffffff;">AI-Powered Food Waste Prediction & Redistribution Intelligence</h2>
        <p style="margin:0.25rem 0 0 0;opacity:0.92;">
            Galgotias University — ORIGIN MINI 2026 (Problem Statement 04) • Operational Decision-Support Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

impact_resp = api_client.request("GET", "/analytics/impact")
forecasts_resp = api_client.request("GET", "/forecasts/history", params={"limit": 6})
plans_resp = api_client.request("GET", "/production-plans", params={"limit": 6})
listings_resp = api_client.request("GET", "/surplus-listings")
offers_resp = api_client.request("GET", "/offers")

if impact_resp["status_code"] == 200:
    impact = impact_resp["data"]
    mp = impact["model_predictions"]
    sim = impact["simulated_planning_outcomes"]
    rec = impact["actual_recorded_outcomes"]
    der = impact["derived_estimates"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Held-Out ML MAE [Predicted Metric]",
        f"{mp.get('ml_mae_portions') or 0:.1f} portions",
        delta=f"-{mp.get('mae_improvement_vs_baseline_percent', 0):.1f}% vs Baseline",
        delta_color="normal",
    )
    c2.metric(
        "Simulated Overproduction Avoided [Estimated]",
        f"{sim.get('simulated_overproduction_avoided_portions', 0):,} portions",
        f"INR {der.get('simulated_production_cost_avoided_inr', 0):,.0f} est.",
    )
    c3.metric(
        "Confirmed Available Surplus [Recorded]",
        f"{rec.get('currently_available_unallocated_portions', 0):,} portions",
        f"{rec.get('currently_reserved_pending_pickup_portions', 0)} portions reserved",
    )
    c4.metric(
        "Verified Collected Surplus [Recorded]",
        f"{rec.get('verified_collected_portions', 0):,} portions",
        f"{der.get('redistribution_recovery_rate_percent', 0):.1f}% recovery rate",
    )

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Upcoming Meal Demand Forecasts [Model Predictions]")
    if forecasts_resp["status_code"] == 200 and forecasts_resp["data"]:
        fdf = pd.DataFrame(forecasts_resp["data"])[
            [
                "forecast_date",
                "meal_type",
                "menu_category",
                "predicted_demand",
                "rounded_operational_demand",
                "planning_buffer_portions",
                "model_version",
            ]
        ]
        st.dataframe(fdf, use_container_width=True)
    else:
        st.info("No forecasts generated yet. Open Page 1 (Demand Forecasting) to generate forecasts.")

with col_right:
    st.subheader("Active Production Plans vs Recommendations [Simulated/Planned]")
    if plans_resp["status_code"] == 200 and plans_resp["data"]:
        pdf = pd.DataFrame(plans_resp["data"])[
            [
                "service_date",
                "meal_type",
                "planned_production",
                "recommended_production",
                "expected_surplus_existing_plan",
                "expected_surplus_recommended_plan",
            ]
        ]
        st.dataframe(pdf, use_container_width=True)
    else:
        st.info("No production plans saved yet. Open Page 2 (Production Planning).")

st.subheader("Active Surplus Listings & Redistribution Offers [Recorded Operational State]")
if listings_resp["status_code"] == 200 and listings_resp["data"]:
    ldf = pd.DataFrame(listings_resp["data"])[
        [
            "listing_code",
            "service_date",
            "meal_type",
            "food_description",
            "food_category",
            "initial_quantity_portions",
            "remaining_quantity_portions",
            "reserved_quantity_portions",
            "collected_quantity_portions",
            "status",
            "pickup_deadline",
        ]
    ]
    st.dataframe(ldf, use_container_width=True)
