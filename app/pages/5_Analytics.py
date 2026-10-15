"""Page 5: Impact Analytics & Auditable Sustainability Metrics."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(page_title="Impact Analytics", layout="wide")
user = render_sidebar_auth()

st.title("5. Auditable Impact & Forecasting Analytics")
st.caption(
    "Strictly distinguishes (1) Model Predictions, (2) Simulated Planning Outcomes, "
    "(3) Actual User-Recorded Outcomes, and (4) Derived Financial Estimates."
)

impact_resp = api_client.request("GET", "/analytics/impact")
tx_resp = api_client.request("GET", "/analytics/transactions")

if impact_resp["status_code"] == 200:
    rep = impact_resp["data"]
    st.info(rep["data_provenance_notice"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Model Predictions (Held-Out Chronological Evaluation)")
        st.json(rep["model_predictions"])
        st.subheader("2. Simulated Planning Outcomes (Recommended vs Planned)")
        st.json(rep["simulated_planning_outcomes"])
    with col2:
        st.subheader("3. Actual Recorded Operational Outcomes")
        st.json(rep["actual_recorded_outcomes"])
        st.subheader("4. Derived Estimates & Explicit Denominators")
        st.json(rep["derived_estimates"])

    st.download_button(
        label="Download Auditable Impact Report (JSON)",
        data=json.dumps(rep, indent=2),
        file_name="auditable_impact_report.json",
        mime="application/json",
    )

st.subheader("Immutable Redistribution Transaction Ledger")
if tx_resp["status_code"] == 200 and tx_resp["data"]:
    st.dataframe(pd.DataFrame(tx_resp["data"]), use_container_width=True)
