"""Explainable recipient matching and multi-criteria ranking engine.

Applies strict hard eligibility constraints before ranking eligible recipients using
normalized, documented weights:
- Proximity score (weight 0.30): Haversine approximate great-circle distance (km)
- Capacity coverage score (weight 0.25): Portion absorption capability
- Pickup window & urgency alignment (weight 0.25): Available window overlap vs deadline
- Priority & suitability score (weight 0.20): Recipient priority tier and type fit
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from backend.domain.food_safety import is_listing_expired, parse_iso_timestamp

SCORING_WEIGHTS = {
    "proximity": 0.30,
    "capacity_fit": 0.25,
    "urgency_window": 0.25,
    "priority_suitability": 0.20,
}


def approximate_haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute approximate great-circle distance in kilometers using the Haversine formula.

    Explicitly labeled as an approximate straight-line/road-adjusted estimate when external
    routing APIs are unavailable.
    """
    r_earth = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r_earth * c, 2)


def _hhmm_to_minutes(hhmm: str) -> int:
    parts = hhmm.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _check_pickup_window_overlap(
    available_from_iso: str,
    pickup_deadline_iso: str,
    window_start_hhmm: str,
    window_end_hhmm: str,
) -> Tuple[bool, int]:
    """Check if the listing's [available_from, pickup_deadline] overlaps recipient's pickup window."""
    avail_dt = parse_iso_timestamp(available_from_iso)
    deadline_dt = parse_iso_timestamp(pickup_deadline_iso)

    avail_mins = avail_dt.hour * 60 + avail_dt.minute
    deadline_mins = deadline_dt.hour * 60 + deadline_dt.minute
    if deadline_dt.date() > avail_dt.date():
        deadline_mins += 24 * 60

    rec_start = _hhmm_to_minutes(window_start_hhmm)
    rec_end = _hhmm_to_minutes(window_end_hhmm)
    if rec_end <= rec_start:
        rec_end += 24 * 60

    overlap_start = max(avail_mins, rec_start)
    overlap_end = min(deadline_mins, rec_end)
    overlap_minutes = max(0, overlap_end - overlap_start)
    return overlap_minutes >= 30, overlap_minutes


def match_and_rank_recipients(
    listing: Dict[str, Any],
    recipients: List[Dict[str, Any]],
    active_offers_for_listing: Optional[List[Dict[str, Any]]] = None,
    require_verification: bool = True,
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter and rank recipients for a surplus listing with transparent explanations."""
    remaining_qty = int(listing.get("remaining_quantity_portions", 0))
    food_cat = str(listing.get("food_category", ""))
    listing_lat = float(listing.get("latitude", 28.3670))
    listing_lon = float(listing.get("longitude", 77.5410))

    expired = is_listing_expired(
        listing["pickup_deadline"], listing["safe_use_deadline"], reference_time
    )

    already_offered_recipients = set()
    if active_offers_for_listing:
        for off in active_offers_for_listing:
            if off.get("status") in ("PENDING", "ACCEPTED"):
                already_offered_recipients.add(int(off["recipient_id"]))

    eligible_matches: List[Dict[str, Any]] = []
    excluded_recipients: List[Dict[str, Any]] = []

    if expired or remaining_qty <= 0 or int(listing.get("safety_approved", 0)) != 1:
        reason = (
            "Listing has expired"
            if expired
            else "No remaining unallocated quantity"
            if remaining_qty <= 0
            else "Listing has not been safety-approved for redistribution"
        )
        return {
            "listing_id": listing.get("id"),
            "listing_status_ok": False,
            "blocking_reason": reason,
            "scoring_weights": SCORING_WEIGHTS,
            "eligible_matches": [],
            "excluded_recipients": [
                {"recipient_id": r.get("id"), "organization_name": r.get("organization_name"), "reasons": [reason]}
                for r in recipients
            ],
        }

    for rec in recipients:
        rec_id = int(rec["id"])
        org_name = str(rec["organization_name"])
        reasons_excluded: List[str] = []

        if int(rec.get("is_active", 0)) != 1:
            reasons_excluded.append("Recipient organization is marked inactive.")
        if require_verification and int(rec.get("is_verified", 0)) != 1:
            reasons_excluded.append("Recipient organization is unverified.")
        if rec_id in already_offered_recipients:
            reasons_excluded.append("Recipient already holds an active (PENDING/ACCEPTED) offer for this listing.")

        raw_cats = rec.get("accepted_categories_json") or rec.get("accepted_categories", "[]")
        accepted_cats = json.loads(raw_cats) if isinstance(raw_cats, str) else list(raw_cats)
        if food_cat not in accepted_cats:
            reasons_excluded.append(f"Does not accept food category '{food_cat}'.")

        max_cap = int(rec.get("max_capacity_portions", 0))
        if max_cap < 10:
            reasons_excluded.append(f"Insufficient pickup capacity ({max_cap} portions < 10 minimum).")

        window_ok, overlap_mins = _check_pickup_window_overlap(
            listing["available_from_time"],
            listing["pickup_deadline"],
            str(rec.get("pickup_window_start", "08:00")),
            str(rec.get("pickup_window_end", "21:00")),
        )
        if not window_ok:
            reasons_excluded.append(
                f"Pickup window ({rec.get('pickup_window_start')}–{rec.get('pickup_window_end')}) has <30 min overlap with listing availability."
            )

        if reasons_excluded:
            excluded_recipients.append(
                {
                    "recipient_id": rec_id,
                    "organization_name": org_name,
                    "reasons": reasons_excluded,
                }
            )
            continue

        # Compute normalized component scores in [0, 1]
        dist_km = approximate_haversine_distance_km(
            listing_lat, listing_lon, float(rec["latitude"]), float(rec["longitude"])
        )
        # Approximate road distance multiplier (1.25x straight-line) and 30 km/h urban speed
        approx_travel_mins = int(round((dist_km * 1.25 / 30.0) * 60.0))

        # 1. Proximity score (1.0 at 0 km, decays to 0.1 at 25+ km)
        proximity_score = max(0.05, min(1.0, 1.0 - (dist_km / 25.0)))

        # 2. Capacity fit score (how much of the remaining surplus this recipient can absorb)
        allocatable_qty = min(remaining_qty, max_cap)
        capacity_fit_score = min(1.0, allocatable_qty / max(1.0, float(remaining_qty)))

        # 3. Urgency & pickup window score (1.0 for >=180 mins overlap relative to travel time)
        usable_buffer_mins = max(0, overlap_mins - approx_travel_mins)
        urgency_window_score = min(1.0, usable_buffer_mins / 180.0)

        # 4. Priority tier suitability score (Tier 1 -> 1.0, Tier 2 -> 0.75, Tier 3 -> 0.5)
        tier = int(rec.get("priority_tier", 2))
        priority_suitability_score = 1.0 if tier == 1 else (0.75 if tier == 2 else 0.50)

        total_score = (
            SCORING_WEIGHTS["proximity"] * proximity_score
            + SCORING_WEIGHTS["capacity_fit"] * capacity_fit_score
            + SCORING_WEIGHTS["urgency_window"] * urgency_window_score
            + SCORING_WEIGHTS["priority_suitability"] * priority_suitability_score
        )

        match_reasons = [
            f"Accepts '{food_cat}' and can absorb up to {allocatable_qty} of {remaining_qty} available portions (max capacity: {max_cap}).",
            f"Approximate distance: {dist_km:.1f} km (~{approx_travel_mins} min estimated road transit).",
            f"Pickup window ({rec.get('pickup_window_start')}–{rec.get('pickup_window_end')}) provides {overlap_mins} min overlap before deadline.",
            f"Priority Tier {tier} verified {str(rec.get('recipient_type', '')).replace('_', ' ')}.",
        ]

        eligible_matches.append(
            {
                "recipient_id": rec_id,
                "organization_name": org_name,
                "recipient_type": rec.get("recipient_type"),
                "contact_person": rec.get("contact_person"),
                "contact_phone": rec.get("contact_phone"),
                "pickup_address": rec.get("pickup_address"),
                "approximate_distance_km": dist_km,
                "estimated_travel_minutes": approx_travel_mins,
                "max_capacity_portions": max_cap,
                "recommended_allocation_portions": allocatable_qty,
                "pickup_window": f"{rec.get('pickup_window_start')}–{rec.get('pickup_window_end')}",
                "priority_tier": tier,
                "match_score": round(total_score * 100.0, 1),
                "component_scores": {
                    "proximity": round(proximity_score * 100.0, 1),
                    "capacity_fit": round(capacity_fit_score * 100.0, 1),
                    "urgency_window": round(urgency_window_score * 100.0, 1),
                    "priority_suitability": round(priority_suitability_score * 100.0, 1),
                },
                "match_reasons": match_reasons,
            }
        )

    eligible_matches.sort(key=lambda m: (-m["match_score"], m["approximate_distance_km"]))
    for rank_idx, match in enumerate(eligible_matches, start=1):
        match["rank"] = rank_idx

    return {
        "listing_id": listing.get("id"),
        "listing_status_ok": True,
        "blocking_reason": None,
        "scoring_weights": SCORING_WEIGHTS,
        "eligible_matches": eligible_matches,
        "excluded_recipients": excluded_recipients,
    }
