"""
Layer 1 — spatial clustering of raw hotspot detections into "thermal events".

Detections that are close in space (and, implicitly, recent — callers pass
in only the current fetch window) are grouped with DBSCAN so a single
industrial site showing up as 3 adjacent pixels doesn't get analyzed 3 times
as 3 unrelated events.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from sklearn.cluster import DBSCAN

from app.database import get_connection

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass

# ~0.01 degrees latitude ~= 1.1km — detections within this are one "event"
EPS_DEGREES = 0.01
MIN_SAMPLES = 1  # a single detection is still a valid event


def cluster_hotspots_into_events(hotspots: list[dict], log: LogFn = _noop_log) -> list[dict]:
    if not hotspots:
        return []
    coords = np.array([[h["latitude"], h["longitude"]] for h in hotspots])
    labels = DBSCAN(eps=EPS_DEGREES, min_samples=MIN_SAMPLES).fit_predict(coords)

    events = []
    for label in sorted(set(labels)):
        members = [h for h, l in zip(hotspots, labels) if l == label]
        lats = [m["latitude"] for m in members]
        lons = [m["longitude"] for m in members]
        centroid_lat, centroid_lon = float(np.mean(lats)), float(np.mean(lons))
        dates = sorted(m["acq_date"] for m in members)
        event_id = hashlib.sha1(f"{centroid_lat:.4f},{centroid_lon:.4f}".encode()).hexdigest()[:16]
        events.append({
            "id": event_id,
            "centroid_lat": round(centroid_lat, 4),
            "centroid_lon": round(centroid_lon, 4),
            "member_count": len(members),
            "first_seen": dates[0],
            "last_seen": dates[-1],
            "member_hotspot_ids": json.dumps([m["id"] for m in members]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    log(f"[LAYER1] DBSCAN grouped {len(hotspots)} detections into {len(events)} thermal events.")

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO thermal_events
                (id, centroid_lat, centroid_lon, member_count, first_seen, last_seen, member_hotspot_ids, created_at)
            VALUES (:id, :centroid_lat, :centroid_lon, :member_count, :first_seen, :last_seen, :member_hotspot_ids, :created_at)
            ON CONFLICT(id) DO UPDATE SET
                member_count=excluded.member_count,
                last_seen=excluded.last_seen,
                member_hotspot_ids=excluded.member_hotspot_ids
            """,
            events,
        )
    return events


def find_event_for_point(lat: float, lon: float, tolerance_deg: float = 0.02) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id FROM thermal_events
            WHERE centroid_lat BETWEEN ? AND ? AND centroid_lon BETWEEN ? AND ?
            ORDER BY member_count DESC LIMIT 1
            """,
            (lat - tolerance_deg, lat + tolerance_deg, lon - tolerance_deg, lon + tolerance_deg),
        ).fetchone()
        return row["id"] if row else None
