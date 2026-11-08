"""SQLite persistence helpers for PowerLine_AI inspection history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "inspection_history.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                inspection_type TEXT NOT NULL CHECK (
                    inspection_type IN ('Image', 'Video')
                ),
                source_filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Completed',
                confidence_threshold REAL NOT NULL,
                frame_interval INTEGER,
                total_frames INTEGER,
                analyzed_frames INTEGER,
                detection_count INTEGER NOT NULL DEFAULT 0,
                processing_seconds REAL NOT NULL DEFAULT 0,
                component_counts_json TEXT NOT NULL DEFAULT '{}',
                detections_json TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_created_at
            ON inspections(created_at DESC)
        """)


def save_inspection(
    *,
    inspection_type: str,
    source_filename: str,
    confidence_threshold: float,
    detection_count: int,
    processing_seconds: float,
    component_counts: dict[str, int] | None = None,
    detections: list[dict[str, Any]] | None = None,
    frame_interval: int | None = None,
    total_frames: int | None = None,
    analyzed_frames: int | None = None,
) -> int:
    if inspection_type not in {"Image", "Video"}:
        raise ValueError("inspection_type must be 'Image' or 'Video'")

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO inspections (
                created_at, inspection_type, source_filename, status,
                confidence_threshold, frame_interval, total_frames,
                analyzed_frames, detection_count, processing_seconds,
                component_counts_json, detections_json
            ) VALUES (?, ?, ?, 'Completed', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                inspection_type,
                Path(source_filename).name,
                float(confidence_threshold),
                frame_interval,
                total_frames,
                analyzed_frames,
                int(detection_count),
                float(processing_seconds),
                json.dumps(component_counts or {}, ensure_ascii=False),
                json.dumps(detections or [], ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def list_inspections(limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_inspection(inspection_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (int(inspection_id),)
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_inspection(inspection_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (int(inspection_id),)
        )
        return cursor.rowcount > 0


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["component_counts"] = json.loads(
        result.pop("component_counts_json") or "{}"
    )
    result["detections"] = json.loads(
        result.pop("detections_json") or "[]"
    )
    return result
