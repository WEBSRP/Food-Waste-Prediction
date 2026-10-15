"""Surplus listing state machine and transactional allocation workflow."""
from __future__ import annotations

import json
import secrets
from typing import Any, Dict, List, Optional

from backend.core.config import settings
from backend.database.models import utc_now_iso
from backend.database.session import DatabaseSession
from backend.domain.food_safety import (
    VALID_FOOD_CATEGORIES,
    is_listing_expired,
    validate_surplus_time_windows,
)

ALLOWED_LISTING_TRANSITIONS: Dict[str, set[str]] = {
    "DRAFT": {"AVAILABLE", "CANCELLED", "EXPIRED"},
    "AVAILABLE": {"PARTIALLY_RESERVED", "RESERVED", "COLLECTED", "CANCELLED", "EXPIRED"},
    "PARTIALLY_RESERVED": {"AVAILABLE", "RESERVED", "COLLECTED", "CANCELLED", "EXPIRED"},
    "RESERVED": {"AVAILABLE", "PARTIALLY_RESERVED", "COLLECTED", "CANCELLED", "EXPIRED"},
    "COLLECTED": set(),
    "CANCELLED": set(),
    "EXPIRED": set(),
}


def create_surplus_listing(
    session: DatabaseSession,
    service_date: str,
    meal_type: str,
    food_description: str,
    food_category: str,
    initial_quantity_portions: int,
    preparation_time: str,
    available_from_time: str,
    pickup_deadline: str,
    safe_use_deadline: str,
    storage_condition: str,
    handling_notes: str,
    pickup_location: Optional[str] = None,
    safety_approved: bool = False,
    created_by_user_id: Optional[int] = None,
    actor_role: str = "cafeteria_admin",
    is_synthetic: int = 0,
) -> Dict[str, Any]:
    """Create a validated surplus listing in DRAFT or AVAILABLE state."""
    if initial_quantity_portions <= 0:
        raise ValueError("Surplus quantity must be greater than zero.")
    if food_category not in VALID_FOOD_CATEGORIES:
        raise ValueError(f"Unsupported food_category '{food_category}'. Allowed: {sorted(VALID_FOOD_CATEGORIES)}")

    ok, msg = validate_surplus_time_windows(
        preparation_time=preparation_time,
        available_from_time=available_from_time,
        pickup_deadline=pickup_deadline,
        safe_use_deadline=safe_use_deadline,
    )
    if not ok:
        raise ValueError(msg)

    now_iso = utc_now_iso()
    can_auto_available = bool(safety_approved)
    status = "AVAILABLE" if can_auto_available else "DRAFT"
    listing_code = f"SL-{service_date.replace('-', '')}-{secrets.token_hex(3).upper()}"

    with session.transaction(immediate=True):
        listing_id = session.insert(
            "surplus_listings",
            {
                "listing_code": listing_code,
                "service_date": service_date,
                "meal_type": meal_type.lower().strip(),
                "food_description": food_description.strip(),
                "food_category": food_category,
                "initial_quantity_portions": int(initial_quantity_portions),
                "remaining_quantity_portions": int(initial_quantity_portions),
                "reserved_quantity_portions": 0,
                "collected_quantity_portions": 0,
                "unit": "portions",
                "preparation_time": preparation_time,
                "safe_use_deadline": safe_use_deadline,
                "available_from_time": available_from_time,
                "pickup_deadline": pickup_deadline,
                "storage_condition": storage_condition.strip(),
                "handling_notes": handling_notes.strip(),
                "pickup_location": pickup_location or settings.CAFETERIA_LOCATION_NAME,
                "latitude": settings.CAFETERIA_LAT,
                "longitude": settings.CAFETERIA_LON,
                "safety_approved": 1 if can_auto_available else 0,
                "approved_by_user_id": created_by_user_id if can_auto_available else None,
                "approved_at": now_iso if can_auto_available else None,
                "status": status,
                "is_synthetic": int(is_synthetic),
                "created_by_user_id": created_by_user_id,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )
        session.log_audit(
            event_type="SURPLUS_LISTING_CREATED",
            entity_type="surplus_listing",
            entity_id=listing_id,
            actor_user_id=created_by_user_id,
            actor_role=actor_role,
            details={"listing_code": listing_code, "quantity": initial_quantity_portions, "status": status},
        )
        created = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (listing_id,))
    return created or {}


def approve_surplus_listing_for_redistribution(
    session: DatabaseSession,
    listing_id: int,
    approver_user_id: Optional[int],
    actor_role: str,
    handling_verification_notes: str = "",
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Explicit staff approval transitioning a DRAFT listing to AVAILABLE."""
    with session.transaction(immediate=True):
        listing = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (listing_id,))
        if not listing:
            raise ValueError(f"Surplus listing {listing_id} not found.")
        if listing["status"] not in ("DRAFT", "AVAILABLE"):
            raise ValueError(f"Cannot approve listing in status '{listing['status']}'.")
        if is_listing_expired(listing["pickup_deadline"], listing["safe_use_deadline"], reference_time):
            session.update_by_id(
                "surplus_listings",
                listing_id,
                {"status": "EXPIRED", "updated_at": utc_now_iso()},
            )
            raise ValueError("Cannot approve listing because its pickup or safe-use deadline has expired.")

        now_iso = utc_now_iso()
        notes = listing["handling_notes"]
        if handling_verification_notes:
            notes = f"{notes} | Approved: {handling_verification_notes}".strip(" |")

        session.update_by_id(
            "surplus_listings",
            listing_id,
            {
                "safety_approved": 1,
                "approved_by_user_id": approver_user_id,
                "approved_at": now_iso,
                "status": "AVAILABLE",
                "handling_notes": notes,
                "updated_at": now_iso,
            },
        )
        session.log_audit(
            event_type="SURPLUS_LISTING_APPROVED",
            entity_type="surplus_listing",
            entity_id=listing_id,
            actor_user_id=approver_user_id,
            actor_role=actor_role,
            details={"notes": handling_verification_notes},
        )
        updated = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (listing_id,))
    return updated or {}


def create_redistribution_offer(
    session: DatabaseSession,
    listing_id: int,
    recipient_id: int,
    offered_quantity_portions: int,
    match_score: float = 0.0,
    match_reasons: Optional[List[str]] = None,
    notes: str = "",
    actor_user_id: Optional[int] = None,
    actor_role: str = "cafeteria_admin",
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically reserve surplus portions and create a redistribution offer.

    Prevents:
    - Overbooking beyond `remaining_quantity_portions`
    - Exceeding recipient's `max_capacity_portions`
    - Duplicate active (`PENDING` or `ACCEPTED`) reservations for the same (listing_id, recipient_id)
    - Offering unapproved (`DRAFT`), cancelled, or expired listings
    - Offering to inactive, unverified, or category-incompatible recipients
    """
    if offered_quantity_portions <= 0:
        raise ValueError("Offered quantity must be strictly positive.")

    with session.transaction(immediate=True):
        listing = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (listing_id,))
        if not listing:
            raise ValueError(f"Surplus listing {listing_id} not found.")

        if is_listing_expired(listing["pickup_deadline"], listing["safe_use_deadline"], reference_time):
            session.update_by_id(
                "surplus_listings",
                listing_id,
                {"status": "EXPIRED", "updated_at": utc_now_iso()},
            )
            raise ValueError("Cannot create offer: surplus listing has expired.")

        if int(listing["safety_approved"]) != 1 or listing["status"] == "DRAFT":
            raise ValueError("Only safety-approved listings in AVAILABLE/PARTIALLY_RESERVED status may be offered.")

        if listing["status"] not in ("AVAILABLE", "PARTIALLY_RESERVED"):
            raise ValueError(f"Listing is currently '{listing['status']}' and cannot accept new offers.")

        remaining = int(listing["remaining_quantity_portions"])
        if offered_quantity_portions > remaining:
            raise ValueError(
                f"Requested offer ({offered_quantity_portions} portions) exceeds remaining available surplus ({remaining} portions)."
            )

        recipient = session.fetch_one("SELECT * FROM recipients WHERE id = ?", (recipient_id,))
        if not recipient:
            raise ValueError(f"Recipient {recipient_id} not found.")
        if int(recipient["is_active"]) != 1 or int(recipient["is_verified"]) != 1:
            raise ValueError("Recipient organization must be active and verified to receive offers.")

        accepted_cats = json.loads(recipient["accepted_categories_json"])
        if listing["food_category"] not in accepted_cats:
            raise ValueError(
                f"Recipient does not accept food category '{listing['food_category']}'."
            )

        # Check active offers for this recipient on this listing (prevent duplicate active reservation)
        existing_active = session.fetch_one(
            """
            SELECT id, status FROM redistribution_offers
            WHERE listing_id = ? AND recipient_id = ? AND status IN ('PENDING', 'ACCEPTED')
            """,
            (listing_id, recipient_id),
        )
        if existing_active:
            raise ValueError(
                f"Duplicate reservation blocked: Recipient already has an active ({existing_active['status']}) offer on this listing."
            )

        if offered_quantity_portions > int(recipient["max_capacity_portions"]):
            raise ValueError(
                f"Offered quantity ({offered_quantity_portions}) exceeds recipient pickup capacity ({recipient['max_capacity_portions']} portions)."
            )

        now_iso = utc_now_iso()
        offer_code = f"OF-{listing_id}-{recipient_id}-{secrets.token_hex(2).upper()}"
        offer_id = session.insert(
            "redistribution_offers",
            {
                "offer_code": offer_code,
                "listing_id": listing_id,
                "recipient_id": recipient_id,
                "offered_quantity_portions": int(offered_quantity_portions),
                "accepted_quantity_portions": 0,
                "collected_quantity_portions": 0,
                "uncollected_quantity_portions": 0,
                "match_score": round(float(match_score), 3),
                "match_reasons_json": json.dumps(match_reasons or []),
                "status": "PENDING",
                "notes": notes.strip(),
                "created_by_user_id": actor_user_id,
                "responded_by_user_id": None,
                "responded_at": None,
                "collected_at": None,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )

        new_remaining = remaining - int(offered_quantity_portions)
        new_reserved = int(listing["reserved_quantity_portions"]) + int(offered_quantity_portions)
        new_status = "RESERVED" if new_remaining == 0 else "PARTIALLY_RESERVED"

        session.update_by_id(
            "surplus_listings",
            listing_id,
            {
                "remaining_quantity_portions": new_remaining,
                "reserved_quantity_portions": new_reserved,
                "status": new_status,
                "updated_at": now_iso,
            },
        )

        session.insert(
            "redistribution_transactions",
            {
                "offer_id": offer_id,
                "listing_id": listing_id,
                "recipient_id": recipient_id,
                "action_type": "OFFER_CREATED",
                "quantity_portions": int(offered_quantity_portions),
                "actor_user_id": actor_user_id,
                "verification_notes": notes.strip(),
                "created_at": now_iso,
            },
        )
        session.log_audit(
            event_type="REDISTRIBUTION_OFFER_CREATED",
            entity_type="redistribution_offer",
            entity_id=offer_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            details={
                "offer_code": offer_code,
                "listing_id": listing_id,
                "recipient_id": recipient_id,
                "offered_quantity": offered_quantity_portions,
            },
        )
        created_offer = session.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
    return created_offer or {}


def respond_to_offer(
    session: DatabaseSession,
    offer_id: int,
    action: str,
    accepted_quantity_portions: Optional[int] = None,
    actor_user_id: Optional[int] = None,
    actor_role: str = "recipient_org",
    notes: str = "",
    reference_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Allow recipient organization or admin to ACCEPT, DECLINE, or CANCEL an offer."""
    act = action.upper().strip()
    if act not in ("ACCEPT", "DECLINE", "CANCEL"):
        raise ValueError("action must be ACCEPT, DECLINE, or CANCEL.")

    with session.transaction(immediate=True):
        offer = session.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
        if not offer:
            raise ValueError(f"Offer {offer_id} not found.")
        listing = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (offer["listing_id"],))
        if not listing:
            raise ValueError("Associated surplus listing not found.")

        now_iso = utc_now_iso()

        if act == "ACCEPT":
            if offer["status"] != "PENDING":
                raise ValueError(f"Only PENDING offers can be accepted (current: {offer['status']}).")
            if is_listing_expired(listing["pickup_deadline"], listing["safe_use_deadline"], reference_time):
                raise ValueError("Cannot accept offer: listing pickup deadline has expired.")

            offered_qty = int(offer["offered_quantity_portions"])
            acc_qty = offered_qty if accepted_quantity_portions is None else int(accepted_quantity_portions)
            if acc_qty <= 0 or acc_qty > offered_qty:
                raise ValueError(f"Accepted quantity must be between 1 and {offered_qty}.")

            released_qty = offered_qty - acc_qty
            new_remaining = int(listing["remaining_quantity_portions"]) + released_qty
            new_reserved = int(listing["reserved_quantity_portions"]) - released_qty
            new_listing_status = "RESERVED" if new_remaining == 0 else "PARTIALLY_RESERVED"

            session.update_by_id(
                "redistribution_offers",
                offer_id,
                {
                    "accepted_quantity_portions": acc_qty,
                    "status": "ACCEPTED",
                    "responded_by_user_id": actor_user_id,
                    "responded_at": now_iso,
                    "notes": f"{offer['notes']} | Accepted: {notes}".strip(" |"),
                    "updated_at": now_iso,
                },
            )
            session.update_by_id(
                "surplus_listings",
                 int(listing["id"]),
                {
                    "remaining_quantity_portions": new_remaining,
                    "reserved_quantity_portions": new_reserved,
                    "status": new_listing_status,
                    "updated_at": now_iso,
                },
            )
            session.insert(
                "redistribution_transactions",
                {
                    "offer_id": offer_id,
                    "listing_id": int(listing["id"]),
                    "recipient_id": int(offer["recipient_id"]),
                    "action_type": "OFFER_ACCEPTED",
                    "quantity_portions": acc_qty,
                    "actor_user_id": actor_user_id,
                    "verification_notes": notes,
                    "created_at": now_iso,
                },
            )
        else:
            # DECLINE or CANCEL releases reserved quantity back to listing
            if offer["status"] not in ("PENDING", "ACCEPTED"):
                raise ValueError(f"Cannot {act.lower()} offer in status '{offer['status']}'.")
            reserved_by_offer = (
                int(offer["accepted_quantity_portions"])
                if offer["status"] == "ACCEPTED"
                else int(offer["offered_quantity_portions"])
            )
            new_remaining = int(listing["remaining_quantity_portions"]) + reserved_by_offer
            new_reserved = max(0, int(listing["reserved_quantity_portions"]) - reserved_by_offer)
            if int(listing["collected_quantity_portions"]) > 0 or new_reserved > 0:
                new_listing_status = "PARTIALLY_RESERVED" if new_reserved > 0 else "AVAILABLE"
            else:
                new_listing_status = "AVAILABLE"

            new_offer_status = "DECLINED" if act == "DECLINE" else "CANCELLED"
            tx_type = "OFFER_DECLINED" if act == "DECLINE" else "OFFER_CANCELLED"

            session.update_by_id(
                "redistribution_offers",
                offer_id,
                {
                    "status": new_offer_status,
                    "uncollected_quantity_portions": reserved_by_offer,
                    "responded_by_user_id": actor_user_id,
                    "responded_at": now_iso,
                    "notes": f"{offer['notes']} | {new_offer_status}: {notes}".strip(" |"),
                    "updated_at": now_iso,
                },
            )
            session.update_by_id(
                "surplus_listings",
                int(listing["id"]),
                {
                    "remaining_quantity_portions": new_remaining,
                    "reserved_quantity_portions": new_reserved,
                    "status": new_listing_status,
                    "updated_at": now_iso,
                },
            )
            session.insert(
                "redistribution_transactions",
                {
                    "offer_id": offer_id,
                    "listing_id": int(listing["id"]),
                    "recipient_id": int(offer["recipient_id"]),
                    "action_type": tx_type,
                    "quantity_portions": reserved_by_offer,
                    "actor_user_id": actor_user_id,
                    "verification_notes": notes,
                    "created_at": now_iso,
                },
            )

        session.log_audit(
            event_type=f"OFFER_{act}",
            entity_type="redistribution_offer",
            entity_id=offer_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            details={"action": act, "notes": notes},
        )
        updated_offer = session.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
    return updated_offer or {}


def confirm_collection(
    session: DatabaseSession,
    offer_id: int,
    collected_quantity_portions: int,
    actor_user_id: Optional[int] = None,
    actor_role: str = "cafeteria_admin",
    verification_notes: str = "",
) -> Dict[str, Any]:
    """Confirm physical handover and collection of an ACCEPTED offer."""
    if collected_quantity_portions <= 0:
        raise ValueError("Collected quantity must be strictly positive.")

    with session.transaction(immediate=True):
        offer = session.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
        if not offer:
            raise ValueError(f"Offer {offer_id} not found.")
        if offer["status"] != "ACCEPTED":
            raise ValueError(f"Only ACCEPTED offers can be collected (current status: {offer['status']}).")

        accepted_qty = int(offer["accepted_quantity_portions"])
        if collected_quantity_portions > accepted_qty:
            raise ValueError(
                f"Collected quantity ({collected_quantity_portions}) cannot exceed accepted quantity ({accepted_qty})."
            )

        listing = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (offer["listing_id"],))
        if not listing:
            raise ValueError("Associated surplus listing not found.")

        uncollected_diff = accepted_qty - int(collected_quantity_portions)
        now_iso = utc_now_iso()

        session.update_by_id(
            "redistribution_offers",
            offer_id,
            {
                "collected_quantity_portions": int(collected_quantity_portions),
                "uncollected_quantity_portions": uncollected_diff,
                "status": "COLLECTED",
                "collected_at": now_iso,
                "notes": f"{offer['notes']} | Collected: {verification_notes}".strip(" |"),
                "updated_at": now_iso,
            },
        )

        new_reserved = max(0, int(listing["reserved_quantity_portions"]) - accepted_qty)
        new_collected = int(listing["collected_quantity_portions"]) + int(collected_quantity_portions)
        # Any uncollected portions from partial pickup remain uncollected on the listing
        new_remaining = int(listing["remaining_quantity_portions"]) + uncollected_diff

        if new_remaining == 0 and new_reserved == 0:
            new_listing_status = "COLLECTED"
        elif new_reserved > 0:
            new_listing_status = "PARTIALLY_RESERVED"
        else:
            new_listing_status = "AVAILABLE"

        session.update_by_id(
            "surplus_listings",
            int(listing["id"]),
            {
                "remaining_quantity_portions": new_remaining,
                "reserved_quantity_portions": new_reserved,
                "collected_quantity_portions": new_collected,
                "status": new_listing_status,
                "updated_at": now_iso,
            },
        )

        session.insert(
            "redistribution_transactions",
            {
                "offer_id": offer_id,
                "listing_id": int(listing["id"]),
                "recipient_id": int(offer["recipient_id"]),
                "action_type": "COLLECTION_CONFIRMED",
                "quantity_portions": int(collected_quantity_portions),
                "actor_user_id": actor_user_id,
                "verification_notes": verification_notes,
                "created_at": now_iso,
            },
        )
        session.log_audit(
            event_type="COLLECTION_CONFIRMED",
            entity_type="redistribution_offer",
            entity_id=offer_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            details={
                "listing_id": listing["id"],
                "collected_quantity": collected_quantity_portions,
                "uncollected_quantity": uncollected_diff,
            },
        )
        updated_offer = session.fetch_one("SELECT * FROM redistribution_offers WHERE id = ?", (offer_id,))
    return updated_offer or {}
