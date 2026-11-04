"""SQLite storage for draft e-challans generated from detected traffic violations.

These records are drafts for review, not official government-issued fines.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "challans.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_challan_db() -> None:
    """Create the challans table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS challans (
                challan_id TEXT PRIMARY KEY,
                plate_number TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_time TEXT NOT NULL,
                vehicle_id INTEGER,
                frame_id INTEGER,
                confidence REAL,
                evidence_json TEXT,
                fine_amount REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'DRAFT',
                created_at TEXT NOT NULL
            )
            """
        )


def _get_value(violation: Any, key: str, default=None):
    if isinstance(violation, dict):
        return violation.get(key, default)
    return getattr(violation, key, default)



def create_draft_challan(violation: Any):
    """Create a challan draft even when the vehicle plate is unreadable."""

    plate = str(
        _get_value(violation, "plate_text", "UNKNOWN") or "UNKNOWN"
    ).strip().upper()

    violation_type = str(
        _get_value(violation, "violation_type", "UNKNOWN") or "UNKNOWN"
    ).strip().upper()

    # A violation type is required, but a readable plate is not.
    if not violation_type or violation_type in {
        "UNKNOWN", "NONE", "N/A", "NULL"
    }:
        return None

    # Missing or unreadable plates are allowed for draft challans.
    if not plate or plate in {"UNKNOWN", "NONE", "N/A", "NULL"}:
        plate = "UNKNOWN"
        status = "PENDING_IDENTIFICATION"
    else:
        status = "DRAFT"

    timestamp = str(
        _get_value(violation, "timestamp", None)
        or datetime.now().isoformat()
    )
    vehicle_id = _get_value(violation, "vehicle_id")
    frame_id = _get_value(violation, "frame_id", 0)
    confidence = _get_value(violation, "confidence", 0)

    evidence = {
        "bbox": _get_value(violation, "bbox", []),
        "detection_tier": _get_value(violation, "detection_tier"),
        "plate_valid": _get_value(violation, "plate_valid", False),
        "ocr_confidence_raw": _get_value(
            violation, "ocr_confidence_raw"
        ),
    }

    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT * FROM challans
            WHERE plate_number = ?
              AND violation_type = ?
              AND violation_time = ?
            """,
            (plate, violation_type, timestamp),
        ).fetchone()

        if existing:
            return _row_to_record(existing)

        challan_id = f"CH-{uuid.uuid4().hex[:12].upper()}"
        created_at = datetime.now().isoformat()

        conn.execute(
            """
            INSERT INTO challans (
                challan_id, plate_number, violation_type, violation_time,
                vehicle_id, frame_id, confidence, evidence_json,
                status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challan_id,
                plate,
                violation_type,
                timestamp,
                vehicle_id,
                frame_id,
                float(confidence or 0),
                json.dumps(evidence, default=str),
                status,
                created_at,
            ),
        )

    return {
        "challan_id": challan_id,
        "plate_number": plate,
        "violation_type": violation_type,
        "violation_time": timestamp,
        "fine_amount": 0,
        "status": status,
        "created_at": created_at,
    }



def _row_to_record(row: sqlite3.Row) -> dict:
    record = dict(row)
    try:
        record["evidence"] = json.loads(
            record.pop("evidence_json", None) or "{}"
        )
    except (ValueError, TypeError):
        record["evidence"] = {}

    return record



def list_challans(limit: int = 100) -> list[dict]:
    """Return the newest draft challans first."""
    limit = max(1, min(int(limit), 500))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM challans ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def get_challan(challan_id: str):
    """Return a single challan, or None if it does not exist."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM challans WHERE challan_id = ?", (challan_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


init_challan_db()
