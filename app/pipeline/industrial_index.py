"""
India-wide industrial facility index — used to filter the hotspot map down to
detections that are actually near real industry, rather than showing every
FIRMS thermal anomaly in the country (most of which are wildfires and
agricultural burns, not industrial sources).

Fetched once from OpenStreetMap Overpass (a single India-wide query, can take
up to ~60-90s the first time) and cached in the local SQLite store
(`industrial_facilities` table). Subsequent hotspot refreshes reuse the cache
and do a fast local nearest-neighbor lookup (sklearn BallTree, haversine
metric) instead of hitting Overpass again.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import httpx
from sklearn.neighbors import BallTree

from app.config import settings
from app.database import (
    save_industrial_facilities, count_industrial_facilities,
    get_all_industrial_facilities, clear_industrial_facilities,
)
from app.pipeline.industrial_seed import get_seed_facilities

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
EARTH_RADIUS_M = 6371000.0

# Tags that mark a location as industrial infrastructure for this project's
# purposes. Kept intentionally simple/commented per the project's synthetic
# rule-based philosophy elsewhere (see context_model.py).
#
# Queried ONE TAG AT A TIME (not combined into a single query): a single
# India-wide Overpass query across all six tags reliably times out on the
# public instance ("runtime error: Query timed out ... after 91 seconds").
# Splitting means a slow/failing tag doesn't wipe out the others.
INDUSTRIAL_QUERY_CLAUSES = [
    ("landuse", "industrial"),
    ("man_made", "works"),
    ("power", "plant"),
    ("industrial", "refinery"),
    ("landuse", "quarry"),
    ("man_made", "mineshaft"),
]


def _facility_id(lat: float, lon: float, name: str) -> str:
    return hashlib.sha1(f"{lat:.5f},{lon:.5f},{name}".encode()).hexdigest()[:20]


def _fetch_one_tag(key: str, value: str, bbox: str, client: httpx.Client, endpoint: str, log: LogFn) -> list[dict]:
    ql = f"""
    [out:json][timeout:90];
    (
      nwr["{key}"="{value}"]({bbox});
    );
    out center 2500;
    """
    resp = client.post(endpoint, data={"data": ql})
    resp.raise_for_status()
    data = resp.json()
    if data.get("remark"):
        raise ValueError(data["remark"])
    elements = data.get("elements", [])
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for el in elements:
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        tags = el.get("tags", {})
        name = tags.get("name", "")
        rows.append({
            "id": _facility_id(lat, lon, name),
            "latitude": lat,
            "longitude": lon,
            "facility_type": value,
            "name": name,
            "fetched_at": now,
        })
    return rows


def fetch_india_industrial_facilities(log: LogFn = _noop_log) -> list[dict]:
    west, south, east, north = settings.INDIA_BBOX
    bbox = f"{south},{west},{north},{east}"
    headers = {"User-Agent": "AGNIDRISHTI/1.0 (SIH hackathon prototype)"}
    log("[INDUSTRIAL-INDEX] Fetching India-wide industrial facility list from Overpass, "
        f"one tag at a time ({len(INDUSTRIAL_QUERY_CLAUSES)} tags, one-time, then cached)...")

    all_rows: list[dict] = []
    with httpx.Client(timeout=100.0, headers=headers) as client:
        for key, value in INDUSTRIAL_QUERY_CLAUSES:
            # Try each mirror in turn — the primary public instance rate-limits
            # (429 / connection reset) under repeated load, so a mirror is a
            # real fallback, not just a cosmetic retry.
            got_it = False
            for endpoint in OVERPASS_MIRRORS:
                try:
                    log(f"[INDUSTRIAL-INDEX] Querying {key}={value} via {endpoint.split('/')[2]}...")
                    rows = _fetch_one_tag(key, value, bbox, client, endpoint, log)
                    log(f"[INDUSTRIAL-INDEX]   -> {len(rows)} facilities found for {key}={value}.")
                    all_rows.extend(rows)
                    got_it = True
                    break
                except httpx.TimeoutException:
                    log(f"[INDUSTRIAL-INDEX] ERROR: {key}={value} timed out on {endpoint.split('/')[2]}.")
                except Exception as e:  # noqa: BLE001
                    log(f"[INDUSTRIAL-INDEX] ERROR: {key}={value} failed on {endpoint.split('/')[2]}: {e}")
                time.sleep(6)
            if not got_it:
                log(f"[INDUSTRIAL-INDEX] All mirrors failed for {key}={value} — skipping this tag.")
            time.sleep(4)  # be polite to the shared public instances between tags

    log(f"[INDUSTRIAL-INDEX] Fetched {len(all_rows)} industrial facilities across India total.")
    return all_rows


_tree_cache: BallTree | None = None
_facilities_cache: list[dict] | None = None
_build_lock = threading.Lock()
_build_in_progress = False


def _build_tree(facilities: list[dict]) -> BallTree:
    coords = np.radians([[f["latitude"], f["longitude"]] for f in facilities])
    return BallTree(coords, metric="haversine")


def _background_build(log: LogFn) -> None:
    global _build_in_progress, _tree_cache, _facilities_cache
    try:
        rows = fetch_india_industrial_facilities(log=log)
        if rows:
            save_industrial_facilities(rows)
            _tree_cache, _facilities_cache = None, None  # force rebuild from the enriched DB
            log(f"[INDUSTRIAL-INDEX] Merged {len(rows)} live Overpass facilities into the index "
                f"(on top of the {len(get_seed_facilities())}-entry curated seed list). "
                "Next hotspot refresh will use the enriched index.")
        else:
            log("[INDUSTRIAL-INDEX] Live Overpass enrichment returned 0 facilities this run "
                "(likely rate-limited) — continuing with the curated seed list only.")
    finally:
        _build_in_progress = False


def start_background_build_if_needed(log: LogFn = _noop_log, force: bool = False) -> bool:
    """Immediately seeds the index with a small curated list of real major
    Indian industrial facilities (instant, no network — see
    industrial_seed.py) so the "industrial only" filter works right away,
    then kicks off a slower (~2-5 min) India-wide Overpass fetch on a daemon
    thread to enrich it with many more facilities. Never blocks the caller.
    Returns True if a build was (already, or newly) in progress or complete."""
    global _tree_cache, _facilities_cache, _build_in_progress

    if force:
        clear_industrial_facilities()
        _tree_cache, _facilities_cache = None, None

    if count_industrial_facilities() == 0:
        seed_rows = get_seed_facilities()
        save_industrial_facilities(seed_rows)
        log(f"[INDUSTRIAL-INDEX] Seeded index with {len(seed_rows)} known major Indian industrial "
            "facilities (instant) — the map filter is usable immediately.")
    elif not force:
        return True

    with _build_lock:
        if _build_in_progress:
            return True
        _build_in_progress = True
    log("[INDUSTRIAL-INDEX] Starting background Overpass enrichment "
        "(can take a few minutes; the seed list already makes the filter usable).")
    threading.Thread(target=_background_build, args=(log,), daemon=True).start()
    return True


def get_or_build_index(log: LogFn = _noop_log) -> tuple[BallTree | None, list[dict]]:
    """Returns (tree, facilities) from cache only — NEVER makes a live
    Overpass call itself (that would block the request for minutes). Callers
    must treat a None tree as "industrial filter unavailable right now" and
    fail open (show all hotspots). Triggers a background build if nothing is
    cached yet."""
    global _tree_cache, _facilities_cache

    if _tree_cache is not None:
        return _tree_cache, _facilities_cache

    start_background_build_if_needed(log=log)

    facilities = get_all_industrial_facilities()
    if not facilities:
        log("[INDUSTRIAL-INDEX] No cached facilities available yet — failing open (showing all hotspots).")
        return None, []

    log(f"[INDUSTRIAL-INDEX] Using {len(facilities)} cached industrial facilities for proximity filtering.")
    tree = _build_tree(facilities)
    _tree_cache, _facilities_cache = tree, facilities
    return tree, facilities


def nearest_facility_for_point(lat: float, lon: float, log: LogFn = _noop_log) -> dict:
    """Same-shape return as app.data_sources.overpass.nearest_industrial_facility,
    but backed by the local cached index (instant, no network) instead of a
    live per-point Overpass call — used by Branch B so context features don't
    depend on Overpass being reachable at analysis time (it frequently isn't,
    under the public instance's rate limits)."""
    tree, facilities = get_or_build_index(log=log)
    if tree is None:
        return {"status": "unavailable", "is_real": False, "found": False,
                "distance_m": None, "facility_type": None, "facility_name": None}

    coords = np.radians([[lat, lon]])
    dist_rad, idx = tree.query(coords, k=1)
    distance_m = float(dist_rad[0][0] * EARTH_RADIUS_M)
    facility = facilities[idx[0][0]]
    return {
        "status": "ok", "is_real": True, "found": True,
        "distance_m": round(distance_m, 1),
        "facility_type": facility["facility_type"],
        "facility_name": facility["name"] or facility["facility_type"],
    }


def filter_hotspots_near_industry(hotspots: list[dict], radius_km: float,
                                   log: LogFn = _noop_log) -> list[dict]:
    """Annotates each hotspot with nearest_facility_m / nearest_facility_name
    and returns only those within radius_km. Fails open (returns all
    hotspots, unannotated) if the facility index isn't available."""
    tree, facilities = get_or_build_index(log=log)
    if tree is None or not hotspots:
        return hotspots

    coords = np.radians([[h["latitude"], h["longitude"]] for h in hotspots])
    dist_rad, idx = tree.query(coords, k=1)
    dist_m = dist_rad[:, 0] * EARTH_RADIUS_M
    nearest_idx = idx[:, 0]

    kept = []
    for h, d, i in zip(hotspots, dist_m, nearest_idx):
        if d <= radius_km * 1000.0:
            h = dict(h)
            h["nearest_facility_m"] = round(float(d), 1)
            h["nearest_facility_name"] = facilities[i]["name"] or facilities[i]["facility_type"]
            h["nearest_facility_type"] = facilities[i]["facility_type"]
            kept.append(h)
    log(f"[INDUSTRIAL-INDEX] {len(kept)}/{len(hotspots)} hotspots are within {radius_km}km of a known industrial facility.")
    return kept
