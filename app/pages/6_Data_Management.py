"""Page 6: Data Management, CSV Upload Validation, and Synthetic Data Controls."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client
from backend.core.config import SYNTHETIC_DIR

st.set_page_config(page_title="Data Management", layout="wide")
user = render_sidebar_auth()

st.title("6. Data Management & CSV Validation")
st.caption("Safely validate and import historical cafeteria meal records, or download the sample CSV template.")

sample_path = SYNTHETIC_DIR / "sample_cafeteria_upload_template.csv"
if sample_path.exists():
    st.download_button(
        "Download Sample Cafeteria CSV Template (30 Records)",
        data=sample_path.read_bytes(),
        file_name="sample_cafeteria_upload_template.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload Cafeteria Meal Records (.csv, max 10 MB)", type=["csv"])
if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    c1, c2 = st.columns(2)
    if c1.button("Validate CSV Only (Dry Run)"):
        resp = api_client.request(
            "POST",
            "/meals/validate-csv",
            files={"file": (uploaded_file.name, raw_bytes, "text/csv")},
        )
        st.json(resp["data"])
    if c2.button("Validate & Import Records into Database"):
        resp = api_client.request(
            "POST",
            "/meals/import-csv",
            files={"file": (uploaded_file.name, raw_bytes, "text/csv")},
        )
        if resp["status_code"] == 200:
            st.success(
                f"Imported {resp['data']['inserted_records']} new and updated {resp['data']['updated_records']} existing records."
            )
            st.json(resp["data"]["validation_report"])
        else:
            st.error(str(resp["data"]))

st.subheader("Recent Historical Meal Service Records in Database")
meals_resp = api_client.request("GET", "/meals", params={"limit": 50})
if meals_resp["status_code"] == 200 and meals_resp["data"].get("items"):
    st.caption(f"Total records in database: **{meals_resp['data']['total']}**")
    st.dataframe(pd.DataFrame(meals_resp["data"]["items"]), use_container_width=True)
