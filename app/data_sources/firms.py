"""
NASA FIRMS client — Layer 1 input.

Real endpoint (used whenever FIRMS_MAP_KEY is set):
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}
Docs: https://firms.modaps.eosdis.nasa.gov/api/
Free instant key: https://firms.modaps.eosdis.nasa.gov/api/map_key/

If no key is configured, falls back to a deterministic synthetic scatter of
hotspots across India so the map is still demoable. This fallback is labeled
"source": "synthetic_demo" everywhere it surfaces (row data, log lines, and
the map legend) — never presented as real FIRMS data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import random
from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx

from app.config import settings

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Every NRT hotspot-detection sensor product FIRMS exposes for the area API.
# (Landsat 30m has no NRT active-fire hotspot product in FIRMS — 16-day
# revisit makes it unsuitable for "current" detections. It's a candidate for
# a future higher-resolution *imagery* provider, same slot as Google Earth
# Engine — see app/data_sources/imagery.py — not a hotspot source.)
SOURCES = ["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


def _row_id(lat: float, lon: float, acq_date: str, acq_time: str, satellite: str) -> str:
    raw = f"{lat:.4f}|{lon:.4f}|{acq_date}|{acq_time}|{satellite}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def fetch_firms_hotspots(day_range: int = 2, log: LogFn = _noop_log) -> list[dict]:
    """Fetch current India-wide hotspots. Real FIRMS if key present, else synthetic demo."""
    if not settings.FIRMS_MAP_KEY:
        log("[FIRMS] No FIRMS_MAP_KEY configured — using synthetic demo scatter (labeled).")
        return _synthetic_india_hotspots()

    west, south, east, north = settings.INDIA_BBOX
    area = f"{west},{south},{east},{north}"
    all_rows: list[dict] = []
    with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        for source in SOURCES:
            url = f"{FIRMS_BASE}/{settings.FIRMS_MAP_KEY}/{source}/{area}/{day_range}"
            try:
                log(f"[FIRMS] GET {source} over India bbox (day_range={day_range})...")
                resp = client.get(url)
                resp.raise_for_status()
                text = resp.text
                if text.strip().lower().startswith("invalid") or "error" in text[:200].lower():
                    log(f"[FIRMS] {source} responded with an error body — skipping.")
                    continue
                reader = csv.DictReader(io.StringIO(text))
                count = 0
                for row in reader:
                    try:
                        lat = float(row["latitude"])
                        lon = float(row["longitude"])
                    except (KeyError, ValueError):
                        continue
                    acq_date = row.get("acq_date", "")
                    acq_time = row.get("acq_time", "")
                    satellite = row.get("satellite", source)
                    all_rows.append({
                        "id": _row_id(lat, lon, acq_date, acq_time, satellite),
                        "latitude": lat,
                        "longitude": lon,
                        "brightness": _safe_float(row.get("bright_ti4") or row.get("brightness")),
                        "frp": _safe_float(row.get("frp")),
                        "confidence": str(row.get("confidence", "")),
                        "acq_date": acq_date,
                        "acq_time": acq_time,
                        "satellite": satellite,
                        "instrument": row.get("instrument", source),
                        "daynight": row.get("daynight", ""),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    })
                    count += 1
                log(f"[FIRMS] {source}: {count} detections parsed.")
            except httpx.TimeoutException:
                log(f"[FIRMS] ERROR: {source} request timed out after {settings.HTTP_TIMEOUT_SECONDS}s")
            except httpx.HTTPStatusError as e:
                log(f"[FIRMS] ERROR: {source} returned HTTP {e.response.status_code}")
            except Exception as e:  # noqa: BLE001 — must never crash the pipeline
                log(f"[FIRMS] ERROR: {source} failed: {e}")

    if not all_rows:
        log("[FIRMS] Live fetch returned no usable rows — falling back to synthetic demo scatter.")
        return _synthetic_india_hotspots()

    log(f"[FIRMS] Total live detections: {len(all_rows)}")
    return all_rows


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _synthetic_india_hotspots(n: int = 220, seed: int = 42) -> list[dict]:
    """Deterministic synthetic scatter across India, clearly flagged as demo data."""
    rng = random.Random(seed)
    west, south, east, north = settings.INDIA_BBOX
    # Rough density clusters around real industrial/agricultural belts so the
    # demo map looks plausible rather than uniformly random.
    clusters = [
        (23.0, 87.0, 18),   # West Bengal / Jharkhand industrial belt
        (21.2, 81.6, 14),   # Chhattisgarh mining belt
        (19.0, 73.0, 16),   # Maharashtra industrial coast
        (28.6, 77.2, 12),   # NCR
        (13.0, 80.0, 10),   # Tamil Nadu coast
        (26.9, 75.8, 10),   # Rajasthan
        (30.9, 75.8, 14),   # Punjab (crop-burn belt)
        (17.4, 78.5, 10),   # Telangana
        (22.3, 70.8, 10),   # Gujarat (Kutch/refineries)
        (25.6, 85.1, 8),    # Bihar
    ]
    rows = []
    today = datetime.now(timezone.utc)
    for clat, clon, count in clusters:
        for _ in range(count):
            lat = max(south, min(north, clat + rng.gauss(0, 1.1)))
            lon = max(west, min(east, clon + rng.gauss(0, 1.1)))
            acq_date = (today - timedelta(days=rng.randint(0, 2))).strftime("%Y-%m-%d")
            frp = round(abs(rng.gauss(12, 10)) + 1, 1)
            brightness = round(300 + rng.gauss(20, 15), 1)
            rows.append({
                "id": _row_id(lat, lon, acq_date, f"{rng.randint(0,23):02d}{rng.randint(0,59):02d}", "SYNTH"),
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "brightness": brightness,
                "frp": frp,
                "confidence": rng.choice(["low", "nominal", "high"]),
                "acq_date": acq_date,
                "acq_time": f"{rng.randint(0,23):02d}{rng.randint(0,59):02d}",
                "satellite": "SYNTH-DEMO",
                "instrument": "synthetic_demo",
                "daynight": rng.choice(["D", "N"]),
                "fetched_at": today.isoformat(),
                "source": "synthetic_demo",
            })
    # Sprinkle a handful of random extra points for realism
    for _ in range(n - len(rows)):
        lat = rng.uniform(south, north)
        lon = rng.uniform(west, east)
        acq_date = (today - timedelta(days=rng.randint(0, 2))).strftime("%Y-%m-%d")
        rows.append({
            "id": _row_id(lat, lon, acq_date, "0000", "SYNTH"),
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "brightness": round(300 + rng.gauss(15, 10), 1),
            "frp": round(abs(rng.gauss(6, 5)) + 0.5, 1),
            "confidence": rng.choice(["low", "nominal", "high"]),
            "acq_date": acq_date,
            "acq_time": "0000",
            "satellite": "SYNTH-DEMO",
            "instrument": "synthetic_demo",
            "daynight": "D",
            "fetched_at": today.isoformat(),
            "source": "synthetic_demo",
        })
    return rows
