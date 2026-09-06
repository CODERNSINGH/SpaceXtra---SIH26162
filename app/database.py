"""
SQLite data-access layer.

Deliberately thin: plain sqlite3 + hand-written SQL rather than an ORM, so
swapping this module's internals for PostgreSQL/PostGIS later only requires
changing the connection helper and the handful of dialect-specific bits
(marked below), not any caller.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS hotspots (
    id TEXT PRIMARY KEY,            -- stable id derived from lat/lon/acq_date/acq_time
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    brightness REAL,
    frp REAL,                       -- fire radiative power
    confidence TEXT,
    acq_date TEXT NOT NULL,
    acq_time TEXT,
    satellite TEXT,
    instrument TEXT,
    daynight TEXT,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hotspots_date ON hotspots (acq_date);
CREATE INDEX IF NOT EXISTS idx_hotspots_latlon ON hotspots (latitude, longitude);

CREATE TABLE IF NOT EXISTS thermal_events (
    id TEXT PRIMARY KEY,            -- cluster id (DBSCAN label + representative point)
    centroid_lat REAL NOT NULL,
    centroid_lon REAL NOT NULL,
    member_count INTEGER NOT NULL,
    first_seen TEXT,
    last_seen TEXT,
    member_hotspot_ids TEXT NOT NULL,  -- JSON list
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS industrial_facilities (
    id TEXT PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    facility_type TEXT,
    name TEXT,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facilities_latlon ON industrial_facilities (latitude, longitude);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    event_id TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL       -- full fused AnalysisResult, serialized
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def upsert_hotspots(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO hotspots
                (id, latitude, longitude, brightness, frp, confidence,
                 acq_date, acq_time, satellite, instrument, daynight, fetched_at)
            VALUES
                (:id, :latitude, :longitude, :brightness, :frp, :confidence,
                 :acq_date, :acq_time, :satellite, :instrument, :daynight, :fetched_at)
            ON CONFLICT(id) DO UPDATE SET
                brightness=excluded.brightness,
                frp=excluded.frp,
                confidence=excluded.confidence,
                fetched_at=excluded.fetched_at
            """,
            rows,
        )
    return len(rows)


def get_hotspots(start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    query = "SELECT * FROM hotspots WHERE 1=1"
    params: list[Any] = []
    if start_date:
        query += " AND acq_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND acq_date <= ?"
        params.append(end_date)
    query += " ORDER BY acq_date DESC"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def save_analysis(analysis_id: str, event_id: str | None, lat: float, lon: float,
                   created_at: str, result: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (id, event_id, latitude, longitude, created_at, result_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET result_json=excluded.result_json
            """,
            (analysis_id, event_id, lat, lon, created_at, json.dumps(result)),
        )


def get_analysis(analysis_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d.pop("result_json"))
        return d


def delete_synthetic_hotspots() -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM hotspots WHERE instrument = 'synthetic_demo'")
        return cur.rowcount


def save_industrial_facilities(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO industrial_facilities (id, latitude, longitude, facility_type, name, fetched_at)
            VALUES (:id, :latitude, :longitude, :facility_type, :name, :fetched_at)
            ON CONFLICT(id) DO UPDATE SET
                facility_type=excluded.facility_type, name=excluded.name
            """,
            rows,
        )
    return len(rows)


def clear_industrial_facilities() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM industrial_facilities")


def count_industrial_facilities() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM industrial_facilities").fetchone()["n"]


def get_all_industrial_facilities() -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM industrial_facilities").fetchall()]


def list_analyses() -> list[dict]:
    """Lightweight summary of every stored analysis, for the map's risk-threshold filter."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, latitude, longitude, created_at, result_json FROM analyses"
        ).fetchall()
    out = []
    for r in rows:
        result = json.loads(r["result_json"])
        fusion = result.get("fusion", {})
        out.append({
            "analysis_id": r["id"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "created_at": r["created_at"],
            "classification": fusion.get("classification"),
            "risk_score": fusion.get("risk_score"),
        })
    return out


def get_history_for_point(lat: float, lon: float, radius_deg: float = 0.05) -> list[dict]:
    """Real archived detections near a point, across all fetched dates."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM hotspots
            WHERE latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
            ORDER BY acq_date ASC
            """,
            (lat - radius_deg, lat + radius_deg, lon - radius_deg, lon + radius_deg),
        ).fetchall()
        return [dict(r) for r in rows]
