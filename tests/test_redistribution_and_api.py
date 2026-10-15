"""Automated tests for recipient matching, surplus state machine, concurrency, and FastAPI RBAC endpoints."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.database.session import DatabaseSession, init_db
from backend.domain.recipient_matcher import match_and_rank_recipients
from backend.domain.surplus_detector import (
    approve_surplus_listing_for_redistribution,
    confirm_collection,
    create_redistribution_offer,
    create_surplus_listing,
    respond_to_offer,
)
from backend.main import app
from scripts.seed_database import seed_all


class TestRedistributionAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.test_db_path = str(Path(cls._tmp_dir.name) / "test_food_waste.db")
        settings.DATABASE_URL = f"sqlite:///{cls.test_db_path}"
        seed_all(db_path=cls.test_db_path, reset_existing=True)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def _get_token(self, username: str, password: str) -> str:
        resp = self.client.post("/api/v1/auth/login", json={"username": username, "password": password})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["access_token"]

    def test_recipient_matching_scenarios_none_one_and_multiple(self) -> None:
        base_listing = {
            "id": 999,
            "food_category": "cooked_grains_legumes",
            "remaining_quantity_portions": 80,
            "available_from_time": "2026-10-15T09:00:00Z",
            "pickup_deadline": "2026-12-31T18:00:00Z",
            "safe_use_deadline": "2026-12-31T20:00:00Z",
            "safety_approved": 1,
            "latitude": 28.3670,
            "longitude": 77.5410,
        }
        r1 = {
            "id": 1,
            "organization_name": "Org A",
            "recipient_type": "shelter",
            "latitude": 28.3800,
            "longitude": 77.5300,
            "accepted_categories": ["cooked_grains_legumes"],
            "max_capacity_portions": 100,
            "pickup_window_start": "08:00",
            "pickup_window_end": "20:00",
            "is_verified": 1,
            "is_active": 1,
            "priority_tier": 1,
        }
        r2_mismatch = {
            **r1,
            "id": 2,
            "organization_name": "Org B Category Mismatch",
            "accepted_categories": ["fresh_fruits_salads"],
        }
        r3_eligible = {
            **r1,
            "id": 3,
            "organization_name": "Org C Further Away",
            "latitude": 28.4900,
            "longitude": 77.4800,
            "priority_tier": 2,
        }

        # 1. No eligible recipients (only category-mismatched candidate)
        res_none = match_and_rank_recipients(base_listing, [r2_mismatch])
        self.assertEqual(len(res_none["eligible_matches"]), 0)
        self.assertEqual(len(res_none["excluded_recipients"]), 1)

        # 2. One eligible recipient
        res_one = match_and_rank_recipients(base_listing, [r1, r2_mismatch])
        self.assertEqual(len(res_one["eligible_matches"]), 1)
        self.assertEqual(res_one["eligible_matches"][0]["recipient_id"], 1)

        # 3. Multiple ranked recipients (Org A closer & Tier 1 -> ranked #1 above Org C)
        res_multi = match_and_rank_recipients(base_listing, [r3_eligible, r1, r2_mismatch])
        self.assertEqual(len(res_multi["eligible_matches"]), 2)
        self.assertEqual(res_multi["eligible_matches"][0]["recipient_id"], 1)
        self.assertEqual(res_multi["eligible_matches"][1]["recipient_id"], 3)

    def test_expired_listing_and_duplicate_reservation_and_capacity_limits(self) -> None:
        session = DatabaseSession(db_path=self.test_db_path)
        try:
            # Create an approved listing with 100 portions
            listing = create_surplus_listing(
                session=session,
                service_date="2026-10-15",
                meal_type="lunch",
                food_description="Test Surplus Batch",
                food_category="cooked_grains_legumes",
                initial_quantity_portions=100,
                preparation_time="2026-10-15T06:00:00Z",
                available_from_time="2026-10-15T08:00:00Z",
                pickup_deadline="2026-12-31T18:00:00Z",
                safe_use_deadline="2026-12-31T20:00:00Z",
                storage_condition="Hot-held 65C",
                handling_notes="OK",
                safety_approved=True,
            )
            l_id = int(listing["id"])

            # Exceeding recipient capacity (Recipient 3 has max_capacity 120; try 500)
            with self.assertRaises(ValueError):
                create_redistribution_offer(
                    session=session,
                    listing_id=l_id,
                    recipient_id=3,
                    offered_quantity_portions=500,
                )

            # Create valid partial offer of 40 portions to Recipient 1
            off1 = create_redistribution_offer(
                session=session,
                listing_id=l_id,
                recipient_id=1,
                offered_quantity_portions=40,
            )
            self.assertEqual(off1["status"], "PENDING")

            # Duplicate active reservation for same recipient on same listing must fail
            with self.assertRaises(ValueError):
                create_redistribution_offer(
                    session=session,
                    listing_id=l_id,
                    recipient_id=1,
                    offered_quantity_portions=20,
                )

            # Cancel offer -> restores remaining quantity back to 100
            respond_to_offer(session=session, offer_id=int(off1["id"]), action="CANCEL")
            refreshed = session.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (l_id,))
            self.assertEqual(int(refreshed["remaining_quantity_portions"]), 100)

            # Test expired listing rejection by passing a future reference_time past deadline
            with self.assertRaises(ValueError):
                create_redistribution_offer(
                    session=session,
                    listing_id=l_id,
                    recipient_id=1,
                    offered_quantity_portions=30,
                    reference_time="2027-01-02T00:00:00Z",
                )
        finally:
            session.close()

    def test_concurrent_allocation_attempts_prevent_overbooking(self) -> None:
        session = DatabaseSession(db_path=self.test_db_path)
        try:
            listing = create_surplus_listing(
                session=session,
                service_date="2026-10-16",
                meal_type="lunch",
                food_description="Concurrent Test Listing (70 portions)",
                food_category="cooked_grains_legumes",
                initial_quantity_portions=70,
                preparation_time="2026-10-16T06:00:00Z",
                available_from_time="2026-10-16T08:00:00Z",
                pickup_deadline="2026-12-31T18:00:00Z",
                safe_use_deadline="2026-12-31T20:00:00Z",
                storage_condition="Hot-held 65C",
                handling_notes="OK",
                safety_approved=True,
            )
            l_id = int(listing["id"])
        finally:
            session.close()

        # Two concurrent workers try to reserve 50 portions each from a 70-portion listing
        def _attempt_allocation(rec_id: int) -> bool:
            s = DatabaseSession(db_path=self.test_db_path)
            try:
                create_redistribution_offer(
                    session=s,
                    listing_id=l_id,
                    recipient_id=rec_id,
                    offered_quantity_portions=50,
                )
                return True
            except ValueError:
                return False
            finally:
                s.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_attempt_allocation, [1, 2]))

        # Exactly one 50-portion allocation must succeed, and one must be rejected
        self.assertEqual(sum(1 for r in results if r), 1)
        check_s = DatabaseSession(db_path=self.test_db_path)
        try:
            final_listing = check_s.fetch_one("SELECT * FROM surplus_listings WHERE id = ?", (l_id,))
            self.assertEqual(int(final_listing["remaining_quantity_portions"]), 20)
            self.assertEqual(int(final_listing["reserved_quantity_portions"]), 50)
        finally:
            check_s.close()

    def test_api_authentication_rbac_and_end_to_end_workflow(self) -> None:
        # 1. Unauthenticated request must return 401
        unauth = self.client.get("/api/v1/meals")
        self.assertEqual(unauth.status_code, 401)

        admin_token = self._get_token("cafeteria_admin", "OriginCafeteria#2026")
        rec_token = self._get_token("recipient_gnckt", "OriginRecipient#2026")

        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        rec_headers = {"Authorization": f"Bearer {rec_token}"}

        # 2. Recipient role attempting to train ML models or create surplus listing must return 403
        forbidden_train = self.client.post("/api/v1/models/train", headers=rec_headers)
        self.assertEqual(forbidden_train.status_code, 403)

        forbidden_create = self.client.post(
            "/api/v1/surplus-listings",
            headers=rec_headers,
            json={
                "service_date": "2026-10-16",
                "meal_type": "dinner",
                "food_description": "Unauthorized listing",
                "food_category": "breads_bakery",
                "initial_quantity_portions": 30,
                "preparation_time": "2026-10-16T14:00:00Z",
                "available_from_time": "2026-10-16T16:00:00Z",
                "pickup_deadline": "2026-12-31T21:00:00Z",
                "safe_use_deadline": "2026-12-31T22:00:00Z",
                "storage_condition": "Insulated",
                "handling_notes": "Test",
            },
        )
        self.assertEqual(forbidden_create.status_code, 403)

        # 3. End-to-end forecast -> plan -> listing -> match -> offer -> accept -> collect
        f_resp = self.client.post(
            "/api/v1/forecasts/predict",
            headers=admin_headers,
            json={
                "forecast_date": "2026-10-17",
                "meal_type": "lunch",
                "menu_category": "north_indian_standard",
            },
        )
        self.assertEqual(f_resp.status_code, 200)

        l_resp = self.client.post(
            "/api/v1/surplus-listings",
            headers=admin_headers,
            json={
                "service_date": "2026-10-17",
                "meal_type": "lunch",
                "food_description": "End-to-End API Test Surplus",
                "food_category": "cooked_grains_legumes",
                "initial_quantity_portions": 55,
                "preparation_time": "2026-10-17T06:30:00Z",
                "available_from_time": "2026-10-17T08:30:00Z",
                "pickup_deadline": "2026-12-31T18:00:00Z",
                "safe_use_deadline": "2026-12-31T20:00:00Z",
                "storage_condition": "Hot-held 65C",
                "handling_notes": "Verified",
                "safety_approved": True,
            },
        )
        self.assertEqual(l_resp.status_code, 201)
        listing_id = l_resp.json()["id"]

        off_resp = self.client.post(
            "/api/v1/offers",
            headers=admin_headers,
            json={
                "listing_id": listing_id,
                "recipient_id": 1,
                "offered_quantity_portions": 55,
                "match_score": 90.0,
            },
        )
        self.assertEqual(off_resp.status_code, 201)
        offer_id = off_resp.json()["id"]

        acc_resp = self.client.post(
            f"/api/v1/offers/{offer_id}/respond",
            headers=rec_headers,
            json={"action": "ACCEPT", "accepted_quantity_portions": 50, "notes": "Taking 50 portions"},
        )
        self.assertEqual(acc_resp.status_code, 200)
        self.assertEqual(acc_resp.json()["accepted_quantity_portions"], 50)

        col_resp = self.client.post(
            f"/api/v1/offers/{offer_id}/collect",
            headers=admin_headers,
            json={"collected_quantity_portions": 50, "verification_notes": "Handed over 50 portions"},
        )
        self.assertEqual(col_resp.status_code, 200)
        self.assertEqual(col_resp.json()["status"], "COLLECTED")


if __name__ == "__main__":
    unittest.main()
