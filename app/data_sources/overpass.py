"""
OpenStreetMap Overpass API client — infrastructure context for Branch B.

No key required. Public instance: https://overpass-api.de/api/interpreter
Docs: https://wiki.openstreetmap.org/wiki/Overpass_API
"""
from __future__ import annotations

import math
from typing import Callable

import httpx

from app.config import settings

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

INDUSTRIAL_TAGS = [
    'landuse=industrial',
    'man_made=works',
    'industrial=refinery',
    'power=plant',
    'power=generator',
]

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_industrial_facility(lat: float, lon: float, radius_m: int = 5000,
                                 log: LogFn = _noop_log) -> dict:
    """Returns {status, distance_m, facility_type, facility_name, is_real}."""
    query_parts = []
    for tag in INDUSTRIAL_TAGS:
        k, v = tag.split("=")
        query_parts.append(f'nwr["{k}"="{v}"](around:{radius_m},{lat},{lon});')
    ql = f"""
    [out:json][timeout:15];
    (
      {" ".join(query_parts)}
    );
    out center 20;
    """
    try:
        log(f"[OVERPASS] Querying nearest industrial facility within {radius_m}m...")
        headers = {"User-Agent": "AGNIDRISHTI/1.0 (SIH hackathon prototype)"}
        overpass_timeout = max(settings.HTTP_TIMEOUT_SECONDS, 20.0)
        with httpx.Client(timeout=overpass_timeout, headers=headers) as client:
            resp = client.post(OVERPASS_URL, data={"data": ql})
            resp.raise_for_status()
            data = resp.json()
        elements = data.get("elements", [])
        if not elements:
            log("[OVERPASS] No industrial facility found within radius.")
            return {
                "status": "ok", "is_real": True, "found": False,
                "distance_m": None, "facility_type": None, "facility_name": None,
            }
        best = None
        best_dist = None
        for el in elements:
            if "lat" in el and "lon" in el:
                elat, elon = el["lat"], el["lon"]
            elif "center" in el:
                elat, elon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            d = _haversine_m(lat, lon, elat, elon)
            if best_dist is None or d < best_dist:
                best_dist = d
                best = el
        tags = best.get("tags", {}) if best else {}
        facility_type = tags.get("landuse") or tags.get("man_made") or tags.get("power") or "industrial"
        facility_name = tags.get("name", "unnamed facility")
        log(f"[OVERPASS] Nearest facility: {facility_name} ({facility_type}) at {best_dist:.0f}m")
        return {
            "status": "ok", "is_real": True, "found": True,
            "distance_m": round(best_dist, 1),
            "facility_type": facility_type,
            "facility_name": facility_name,
        }
    except httpx.TimeoutException:
        log(f"[OVERPASS] ERROR: request timed out after {settings.HTTP_TIMEOUT_SECONDS}s")
    except Exception as e:  # noqa: BLE001
        log(f"[OVERPASS] ERROR: {e}")
    return {
        "status": "unavailable", "is_real": False, "found": False,
        "distance_m": None, "facility_type": None, "facility_name": None,
    }
