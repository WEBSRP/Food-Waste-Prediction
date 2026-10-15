"""Food safety operational rules, time-window validation, and human approval checks.

IMPORTANT DISCLAIMER:
Passing time-window validation does NOT constitute automated microbiological safety
certification. Actual food safety depends on temperature control, hygienic handling,
storage conditions, and manual inspection by qualified cafeteria staff.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

VALID_FOOD_CATEGORIES = {
    "cooked_grains_legumes",
    "cooked_vegetables_curries",
    "breads_bakery",
    "packaged_dairy_beverages",
    "fresh_fruits_salads",
}


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    cleaned = ts_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_surplus_time_windows(
    preparation_time: str,
    available_from_time: str,
    pickup_deadline: str,
    safe_use_deadline: str,
) -> Tuple[bool, str]:
    """Validate chronological consistency of preparation, availability, pickup, and safe-use timestamps."""
    try:
        prep_dt = parse_iso_timestamp(preparation_time)
        avail_dt = parse_iso_timestamp(available_from_time)
        pickup_dt = parse_iso_timestamp(pickup_deadline)
        safe_dt = parse_iso_timestamp(safe_use_deadline)
    except Exception as exc:
        return False, f"Invalid ISO-8601 timestamp format: {exc}"

    if avail_dt < prep_dt:
        return False, "available_from_time cannot be earlier than preparation_time."
    if pickup_dt <= avail_dt:
        return False, "pickup_deadline must be strictly after available_from_time."
    if safe_dt < pickup_dt:
        return False, "safe_use_deadline must be on or after pickup_deadline."
    return True, "Time windows are chronologically consistent."


def is_listing_expired(
    pickup_deadline: str,
    safe_use_deadline: str,
    reference_time: Optional[str] = None,
) -> bool:
    """Determine whether a listing has passed its pickup or safe-use deadline."""
    ref_dt = (
        parse_iso_timestamp(reference_time)
        if reference_time
        else datetime.now(timezone.utc)
    )
    pickup_dt = parse_iso_timestamp(pickup_deadline)
    safe_dt = parse_iso_timestamp(safe_use_deadline)
    return ref_dt > min(pickup_dt, safe_dt)
