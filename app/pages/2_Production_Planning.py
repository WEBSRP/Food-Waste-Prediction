"""Page 2: Intelligent Production Planning & Shortage Risk Decision Support."""
from __future__ import annotations

from datetime import date
import pandas as pd
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(page_title="Production Planning", layout="wide")
user = render_sidebar_auth()

st.title("2. Intelligent Production Planning")
st.caption(
    "Decision-support engine balancing shortage risk tolerance and cost asymmetry against potential surplus."
)

with st.form("production_plan_form"):
    c1, c2, c3 = st.columns(3)
    s_date = c1.date_input("Service Date", value=date(2026, 10, 16))
    meal_type = c2.selectbox("Meal Type", ["breakfast", "lunch", "dinner"], index=1)
    pred_demand = c3.number_input("Forecast Demand (Portions)", min_value=0.0, max_value=5000.0, value=518.0, step=5.0)

    c4, c5, c6 = st.columns(3)
    planned_prod = c4.number_input("Current Planned Production (Portions)", min_value=0, max_value=5000, value=610, step=10)
    capacity = c5.number_input("Kitchen Production Capacity (Portions)", min_value=50, max_value=5000, value=750, step=25)
    uncertainty_std = c6.number_input("Forecast Residual Std Dev (σ)", min_value=1.0, max_value=500.0, value=21.0, step=1.0)

    c7, c8, c9 = st.columns(3)
    risk_tol = c7.slider("Shortage Risk Tolerance (α)", min_value=0.02, max_value=0.40, value=0.15, step=0.01)
    unit_cost = c8.number_input("Meal Unit Cost (INR)", min_value=10.0, max_value=500.0, value=65.0, step=5.0)
    shortage_pen = c9.number_input("Shortage Penalty per Portion (INR)", min_value=10.0, max_value=1000.0, value=110.0, step=5.0)

    submitted = st.form_submit_button("Calculate Production Recommendation")

if submitted:
    resp = api_client.request(
        "POST",
        "/production-plans",
        json_body={
            "service_date": s_date.isoformat(),
            "meal_type": meal_type,
            "predicted_demand": pred_demand,
            "planned_production": int(planned_prod),
            "production_capacity": int(capacity),
            "uncertainty_std": uncertainty_std,
            "shortage_risk_tolerance": risk_tol,
            "unit_cost_inr": unit_cost,
            "shortage_penalty_inr": shortage_pen,
            "save_plan": True,
        },
    )
    if resp["status_code"] in (200, 201):
        rec = resp["data"]
        st.success("Production Recommendation Computed & Saved")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Production", f"{rec['recommended_production']} portions", f"Buffer: +{rec['safety_buffer_portions']}")
        m2.metric("Expected Surplus (Current Plan)", f"{rec['expected_surplus_existing_plan']} portions [Estimated]")
        m3.metric("Expected Surplus (Recommended)", f"{rec['expected_surplus_recommended_plan']} portions [Estimated]")
        m4.metric("Potential Overproduction Avoided", f"{rec['potential_waste_reduction_portions']} portions", f"INR {rec['estimated_savings_inr']:,.0f}")
        st.info(rec["explanation"])
    else:
        st.error(str(resp["data"]))

st.subheader("Saved Production Plans vs Post-Service Actuals")
plans_resp = api_client.request("GET", "/production-plans")
if plans_resp["status_code"] == 200 and plans_resp["data"]:
    st.dataframe(pd.DataFrame(plans_resp["data"]), use_container_width=True)
