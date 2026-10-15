"""Page 3: Confirmed Surplus Food Listing & Safety Verification Workflow."""
from __future__ import annotations

from datetime import date
import pandas as pd
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(page_title="Food Surplus Management", layout="wide")
user = render_sidebar_auth()

st.title("3. Food Surplus Management & Safety Gate")
st.warning(
    "Operational Policy: Only surplus food explicitly reviewed for safe storage, temperature control, "
    "and handling by authorized staff may transition from DRAFT to AVAILABLE. Deadline timestamps are operational "
    "windows, not automated microbiological safety certifications."
)

with st.expander("Create New Surplus Food Listing", expanded=True):
    with st.form("create_surplus_form"):
        c1, c2, c3 = st.columns(3)
        s_date = c1.date_input("Service Date", value=date(2026, 10, 15))
        meal_type = c2.selectbox("Meal Type", ["breakfast", "lunch", "dinner"], index=1)
        food_cat = c3.selectbox(
            "Food Category",
            [
                "cooked_grains_legumes",
                "cooked_vegetables_curries",
                "breads_bakery",
                "packaged_dairy_beverages",
                "fresh_fruits_salads",
            ],
        )
        desc = st.text_input("Food Item / Menu Description", value="Dal Makhani, Jeera Rice & Steamed Veg Pulao Trays")
        qty = st.number_input("Confirmed Surplus Quantity (Portions)", min_value=5, max_value=2000, value=85, step=5)

        t1, t2, t3, t4 = st.columns(4)
        prep_time = t1.text_input("Preparation Time (ISO UTC)", value="2026-10-15T06:30:00Z")
        avail_time = t2.text_input("Available From (ISO UTC)", value="2026-10-15T09:00:00Z")
        pickup_dead = t3.text_input("Pickup Deadline (ISO UTC)", value="2026-10-16T15:30:00Z")
        safe_dead = t4.text_input("Safe-Use Deadline (ISO UTC)", value="2026-10-16T18:00:00Z")

        storage = st.text_input("Storage & Temperature Condition", value="Hot-held at 64°C in insulated food-grade stainless gastronorm pans")
        notes = st.text_input("Handling & Packaging Notes", value="Batch inspected by Mess Supervisor; sealed for immediate pickup.")
        approve_now = st.checkbox("Confirm Human Safety & Handling Review (Publish immediately as AVAILABLE)", value=True)
        create_btn = st.form_submit_button("Register Surplus Listing")

if create_btn:
    resp = api_client.request(
        "POST",
        "/surplus-listings",
        json_body={
            "service_date": s_date.isoformat(),
            "meal_type": meal_type,
            "food_description": desc,
            "food_category": food_cat,
            "initial_quantity_portions": int(qty),
            "preparation_time": prep_time,
            "available_from_time": avail_time,
            "pickup_deadline": pickup_dead,
            "safe_use_deadline": safe_dead,
            "storage_condition": storage,
            "handling_notes": notes,
            "safety_approved": approve_now,
            "is_synthetic": 0,
        },
    )
    if resp["status_code"] == 201:
        st.success(f"Created Listing `{resp['data']['listing_code']}` in status `{resp['data']['status']}`.")
    else:
        st.error(str(resp["data"]))

st.subheader("Current Surplus Inventory & State Machine")
listings_resp = api_client.request("GET", "/surplus-listings")
if listings_resp["status_code"] == 200 and listings_resp["data"]:
    st.dataframe(pd.DataFrame(listings_resp["data"]), use_container_width=True)
