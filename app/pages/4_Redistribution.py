"""Page 4: Explainable Recipient Matching, Offer Allocation, Acceptance, and Collection Confirmation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.ui_theme import render_sidebar_auth
from app.services.api_client import api_client

st.set_page_config(page_title="Redistribution Intelligence", layout="wide")
user = render_sidebar_auth()

st.title("4. Recipient Matching & Redistribution Intelligence")
st.caption(
    "Deterministic hard-constraint filtering and normalized multi-criteria ranking "
    "(30% proximity, 25% capacity fit, 25% pickup window urgency, 20% priority suitability)."
)

listings_resp = api_client.request("GET", "/surplus-listings")
listings = listings_resp["data"] if listings_resp["status_code"] == 200 else []
avail_listings = [l for l in listings if l["status"] in ("AVAILABLE", "PARTIALLY_RESERVED")]

if avail_listings:
    selected_listing_label = st.selectbox(
        "Select Available Surplus Listing to Match Recipients",
        options=[
            f"ID {l['id']} | {l['listing_code']} — {l['food_description']} ({l['remaining_quantity_portions']} portions available)"
            for l in avail_listings
        ],
    )
    selected_id = int(selected_listing_label.split("|")[0].replace("ID", "").strip())

    match_resp = api_client.request("GET", f"/surplus-listings/{selected_id}/matches")
    if match_resp["status_code"] == 200:
        mdata = match_resp["data"]
        st.subheader("Ranked Eligible Recipient Matches")
        if mdata["eligible_matches"]:
            st.dataframe(pd.DataFrame(mdata["eligible_matches"]), use_container_width=True)

            with st.form("create_offer_form"):
                rec_options = {
                    f"Rank #{m['rank']}: {m['organization_name']} (Score {m['match_score']} | Max {m['max_capacity_portions']} portions)": m
                    for m in mdata["eligible_matches"]
                }
                chosen_rec_key = st.selectbox("Recipient Organization", list(rec_options.keys()))
                chosen_match = rec_options[chosen_rec_key]
                offer_qty = st.number_input(
                    "Portions to Offer & Reserve",
                    min_value=1,
                    max_value=int(chosen_match["recommended_allocation_portions"]),
                    value=int(chosen_match["recommended_allocation_portions"]),
                )
                offer_notes = st.text_input("Dispatch Notes", value="Insulated containers ready at Mess Gate A.")
                send_offer = st.form_submit_button("Create Offer & Reserve Surplus")

            if send_offer:
                off_resp = api_client.request(
                    "POST",
                    "/offers",
                    json_body={
                        "listing_id": selected_id,
                        "recipient_id": chosen_match["recipient_id"],
                        "offered_quantity_portions": int(offer_qty),
                        "match_score": chosen_match["match_score"],
                        "match_reasons": chosen_match["match_reasons"],
                        "notes": offer_notes,
                    },
                )
                if off_resp["status_code"] == 201:
                    st.success(f"Created Offer `{off_resp['data']['offer_code']}`!")
                else:
                    st.error(str(off_resp["data"]))
        else:
            st.warning(f"No eligible recipients matched. Reason: {mdata.get('blocking_reason')}")

        with st.expander("Inspect Excluded Recipients & Hard Constraint Reasons"):
            st.table(pd.DataFrame(mdata.get("excluded_recipients", [])))

st.divider()
st.subheader("Redistribution Offers: Recipient Acceptance & Collection Handover")
offers_resp = api_client.request("GET", "/offers")
if offers_resp["status_code"] == 200 and offers_resp["data"]:
    offers_list = offers_resp["data"]
    st.dataframe(pd.DataFrame(offers_list), use_container_width=True)

    c_act1, c_act2 = st.columns(2)
    with c_act1:
        with st.form("respond_offer_form"):
            st.markdown("#### Step A: Recipient Accept / Decline Offer")
            offer_id_in = st.number_input("Offer ID", min_value=1, value=int(offers_list[0]["id"]))
            act_choice = st.selectbox("Action", ["ACCEPT", "DECLINE", "CANCEL"])
            resp_notes = st.text_input("Response Notes", value="Pickup vehicle dispatched.")
            resp_btn = st.form_submit_button("Submit Offer Response")
        if resp_btn:
            r = api_client.request(
                "POST",
                f"/offers/{int(offer_id_in)}/respond",
                json_body={"action": act_choice, "notes": resp_notes},
            )
            if r["status_code"] == 200:
                st.success(f"Offer {offer_id_in} updated to `{r['data']['status']}`.")
            else:
                st.error(str(r["data"]))

    with c_act2:
        with st.form("collect_offer_form"):
            st.markdown("#### Step B: Confirm Physical Collection")
            coll_id_in = st.number_input("Accepted Offer ID", min_value=1, value=int(offers_list[0]["id"]))
            coll_qty = st.number_input("Verified Collected Quantity (Portions)", min_value=1, max_value=2000, value=60)
            coll_notes = st.text_input("Handover Verification Notes", value="Signed receipt by NGO coordinator.")
            coll_btn = st.form_submit_button("Confirm Collection Handover")
        if coll_btn:
            r = api_client.request(
                "POST",
                f"/offers/{int(coll_id_in)}/collect",
                json_body={"collected_quantity_portions": int(coll_qty), "verification_notes": coll_notes},
            )
            if r["status_code"] == 200:
                st.success(f"Collection confirmed for Offer {coll_id_in}!")
            else:
                st.error(str(r["data"]))
