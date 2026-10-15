"""Page 1: Machine-Learning Demand Forecasting & Chronological Evaluation."""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(page_title="Demand Forecasting", layout="wide")
user = render_sidebar_auth()

st.title("1. Machine-Learning Demand Forecasting")
st.caption(
    "Chronological time-series evaluation comparing Seasonal Weekday Moving Average (Baseline) "
    "against Gradient Boosted Regression Trees."
)

col_form, col_result = st.columns([1, 1])

with col_form:
    st.subheader("Generate Upcoming Meal Forecast")
    with st.form("forecast_form"):
        f_date = st.date_input("Forecast Service Date", value=date(2026, 10, 16))
        meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"], index=1)
        menu_cat = st.selectbox(
            "Menu Category",
            [
                "north_indian_standard",
                "south_indian_special",
                "continental_fusion",
                "festive_thali",
                "healthy_millet_bowl",
            ],
        )
        is_hol = st.checkbox("Scheduled Campus Holiday", value=False)
        spec_event = st.selectbox(
            "Special Event Status",
            ["none", "exam_week", "sports_meet", "campus_hackathon_fest"],
        )
        weather = st.selectbox("Expected Weather", ["clear", "rainy", "extreme_heat", "cold_fog"])
        temp_c = st.slider("Expected Temperature (°C)", -5.0, 48.0, 27.5, 0.5)
        submitted = st.form_submit_button("Predict Meal Demand")

if submitted:
    resp = api_client.request(
        "POST",
        "/forecasts/predict",
        json_body={
            "forecast_date": f_date.isoformat(),
            "meal_type": meal_type,
            "menu_category": menu_cat,
            "is_holiday": 1 if is_hol else 0,
            "special_event": spec_event,
            "weather_condition": weather,
            "temperature_c": temp_c,
            "save_forecast": True,
        },
    )
    with col_result:
        if resp["status_code"] == 200:
            data = resp["data"]
            st.success(f"Forecast Generated (Model `{data['model_version']}`)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Mean Demand", f"{data['predicted_demand']} portions")
            m2.metric("Operational Demand (Rounded)", f"{data['rounded_operational_demand']} portions")
            m3.metric("Planning Buffer (Q85)", f"+{data['planning_buffer_portions']} portions")
            st.caption(
                f"Baseline comparison: **{data['baseline_demand']} portions** | "
                f"Planning interval: **[{data['lower_bound_demand']}, {data['upper_bound_demand']}] portions**"
            )
            st.markdown("#### Contributing Forecast Factors")
            st.table(pd.DataFrame(data["contributing_factors"]))
        else:
            st.error(f"Error generating forecast: {resp['data']}")

st.divider()
st.subheader("Chronological Model Evaluation vs Baseline (Held-Out Test Split)")
if st.button("Retrain & Evaluate Models on Historical Records"):
    train_resp = api_client.request("POST", "/models/train")
    if train_resp["status_code"] == 200:
        st.success(f"Trained model `{train_resp['data']['model_version']}` successfully!")
    else:
        st.error(str(train_resp["data"]))

eval_resp = api_client.request("GET", "/models/evaluations")
if eval_resp["status_code"] == 200 and eval_resp["data"]:
    latest = eval_resp["data"][0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ML MAE (Held-Out)", f"{latest['ml_mae']:.2f} portions", f"Baseline: {latest['baseline_mae']:.2f}")
    k2.metric("ML RMSE (Held-Out)", f"{latest['ml_rmse']:.2f} portions", f"Baseline: {latest['baseline_rmse']:.2f}")
    k3.metric("ML WAPE (%)", f"{latest['ml_wape']:.2f}%", f"Baseline: {latest['baseline_wape']:.2f}%")
    k4.metric("ML Forecast Bias", f"{latest['ml_bias']:+.2f} portions", f"Baseline: {latest['baseline_bias']:+.2f}")

    comp_df = pd.DataFrame(latest.get("comparison_series", []))
    if not comp_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comp_df["service_date"], y=comp_df["actual_demand"], name="Actual Effective Demand", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=comp_df["service_date"], y=comp_df["ml_predicted"], name="ML Forecast", mode="lines"))
        fig.add_trace(go.Scatter(x=comp_df["service_date"], y=comp_df["baseline_predicted"], name="Baseline Forecast", mode="lines", line=dict(dash="dot")))
        fig.update_layout(title="Held-Out Test Split: Actual vs Predicted Meal Demand", xaxis_title="Service Date", yaxis_title="Portions")
        st.plotly_chart(fig, use_container_width=True)
