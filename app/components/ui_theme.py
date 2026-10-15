"""Shared environmental-technology styling, authentication sidebar, and provenance banners for Streamlit."""
from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

from app.services.api_client import api_client

DEMO_PASSWORDS = {
    "cafeteria_admin": "OriginCafeteria#2026",
    "recipient_gnckt": "OriginRecipient#2026",
    "recipient_yamuna": "OriginRecipient#2026",
    "project_admin": "OriginAdmin#2026",
}


def apply_environmental_theme() -> None:
    """Inject custom environmental-technology CSS (Deep Forest Green & Restrained Teal)."""
    st.markdown(
        """
        <style>
        :root {
            --forest-primary: #14532d;
            --teal-secondary: #0f766e;
            --surface-card: #f8fafc;
            --border-subtle: #cbd5e1;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        .env-banner {
            background: linear-gradient(90deg, #14532d 0%, #0f766e 100%);
            color: #ffffff;
            padding: 1.1rem 1.4rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .synthetic-notice {
            background-color: #fef3c7;
            border-left: 4px solid #d97706;
            color: #78350f;
            padding: 0.65rem 1rem;
            border-radius: 6px;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }
        .metric-badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-predicted { background: #dbeafe; color: #1e40af; }
        .badge-estimated { background: #fef3c7; color: #92400e; }
        .badge-recorded { background: #dcfce7; color: #166534; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_auth() -> Optional[Dict[str, Any]]:
    """Render role-based demo authentication switcher and return active user profile."""
    apply_environmental_theme()
    st.sidebar.markdown("### ORIGIN MINI 2026")
    st.sidebar.caption("Galgotias University • Problem Statement 04")

    accounts_resp = api_client.request("GET", "/auth/demo-accounts")
    demo_accounts = accounts_resp["data"] if accounts_resp["status_code"] == 200 else []

    if "auth_token" not in st.session_state and demo_accounts:
        default_username = demo_accounts[0]["username"]
        login_resp = api_client.request(
            "POST",
            "/auth/login",
            json_body={
                "username": default_username,
                "password": DEMO_PASSWORDS.get(default_username, "OriginCafeteria#2026"),
            },
        )
        if login_resp["status_code"] == 200:
            st.session_state["auth_token"] = login_resp["data"]["access_token"]
            st.session_state["current_user"] = login_resp["data"]["user"]

    account_labels = {
        f"{a['username']} ({a['role']})": a["username"] for a in demo_accounts
    }
    selected_label = st.sidebar.selectbox(
        "Active Demo Role Account",
        options=list(account_labels.keys()),
        help="Switch demo account to test backend role-based access control (RBAC).",
    )
    if selected_label:
        chosen_username = account_labels[selected_label]
        cur_user = st.session_state.get("current_user", {})
        if cur_user.get("username") != chosen_username:
            login_resp = api_client.request(
                "POST",
                "/auth/login",
                json_body={
                    "username": chosen_username,
                    "password": DEMO_PASSWORDS.get(chosen_username, "OriginCafeteria#2026"),
                },
            )
            if login_resp["status_code"] == 200:
                st.session_state["auth_token"] = login_resp["data"]["access_token"]
                st.session_state["current_user"] = login_resp["data"]["user"]

    api_client.set_token(st.session_state.get("auth_token"))
    user = st.session_state.get("current_user")
    if user:
        st.sidebar.success(f"Signed in: **{user['full_name']}**\nRole: `{user['role']}`")

    st.markdown(
        '<div class="synthetic-notice">'
        "<strong>SYNTHETIC DEMONSTRATION DATA NOTICE:</strong> Initial 12-month hostel mess records and recipient "
        "organizations are deterministically generated synthetic records for ORIGIN MINI 2026 evaluation. "
        "Every metric below is explicitly labeled as <em>Predicted</em>, <em>Simulated/Estimated</em>, or <em>Recorded</em>."
        "</div>",
        unsafe_allow_html=True,
    )
    return user
