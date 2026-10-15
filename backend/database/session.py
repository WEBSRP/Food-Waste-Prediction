"""Database session management, transaction locking, and query helpers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import threading
from typing import Any, Dict, Generator, List, Optional, Sequence

from backend.core.config import settings
from backend.database.models import SCHEMA_DDL, utc_now_iso

_DB_LOCK = threading.RLock()


class DatabaseSession:
    """Parameterized relational database session with re-entrant transaction control."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        self.conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = MEMORY;")
        self.conn.execute("PRAGMA synchronous = OFF;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA busy_timeout = 10000;")
        self._tx_depth = 0

    @contextmanager
    def transaction(self, immediate: bool = False) -> Generator["DatabaseSession", None, None]:
        """Execute statements atomically inside a database transaction.

        Supports nested re-entrant calls on the same session and uses a process-wide RLock
        plus `BEGIN IMMEDIATE` when `immediate=True` to prevent concurrent overbooking.
        """
        with _DB_LOCK:
            is_outermost = (self._tx_depth == 0) and (not self.conn.in_transaction)
            if is_outermost:
                begin_stmt = "BEGIN IMMEDIATE;" if immediate else "BEGIN;"
                self.conn.execute(begin_stmt)
            self._tx_depth += 1
            try:
                yield self
                if is_outermost and self.conn.in_transaction:
                    self.conn.execute("COMMIT;")
            except Exception:
                if is_outermost and self.conn.in_transaction:
                    try:
                        self.conn.execute("ROLLBACK;")
                    except sqlite3.OperationalError:
                        pass
                raise
            finally:
                self._tx_depth = max(0, self._tx_depth - 1)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with _DB_LOCK:
            return self.conn.execute(sql, params)

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        with _DB_LOCK:
            cur = self.conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with _DB_LOCK:
            cur = self.conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        with _DB_LOCK:
            cols = list(data.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
            cur = self.conn.execute(sql, [data[c] for c in cols])
            return int(cur.lastrowid)

    def update_by_id(self, table: str, record_id: int, updates: Dict[str, Any]) -> int:
        if not updates:
            return 0
        with _DB_LOCK:
            cols = list(updates.keys())
            set_clause = ", ".join([f"{c} = ?" for c in cols])
            sql = f"UPDATE {table} SET {set_clause} WHERE id = ?"
            params = [updates[c] for c in cols] + [record_id]
            cur = self.conn.execute(sql, params)
            return int(cur.rowcount)

    def log_audit(
        self,
        event_type: str,
        entity_type: str,
        entity_id: Any,
        actor_user_id: Optional[int],
        actor_role: str,
        details: Dict[str, Any],
    ) -> int:
        """Record an auditable event in the audit_events table."""
        return self.insert(
            "audit_events",
            {
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "actor_user_id": actor_user_id,
                "actor_role": actor_role,
                "details_json": json.dumps(details, default=str),
                "created_at": utc_now_iso(),
            },
        )

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize all 12 database tables, foreign keys, and indexes."""
    session = DatabaseSession(db_path=db_path)
    try:
        with session.transaction():
            for ddl in SCHEMA_DDL:
                session.execute(ddl)
    finally:
        session.close()


def get_db() -> Generator[DatabaseSession, None, None]:
    """FastAPI dependency yielding a DatabaseSession."""
    session = DatabaseSession()
    try:
        yield session
    finally:
        session.close()
