"""
geo_lookup.py
-------------
REAL module: reverse-geocodes a coordinate into a human-readable place name
using OpenStreetMap's free Nominatim API. No key needed — only a descriptive
User-Agent, which Nominatim's usage policy requires of every client.
"""
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_HEADERS = {"User-Agent": "ThermoWatchAI-SIH26162-Prototype/1.0 (student hackathon demo)"}


def reverse_geocode(lat: float, lon: float, timeout: int = 15) -> dict:
    """Returns {'display_name', 'address'}; a safe fallback dict on failure
    so a flaky network never breaks the rest of the pipeline."""
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return {
            "display_name": data.get("display_name", "Unknown location"),
            "address": data.get("address", {}),
        }
    except Exception as e:
        print(f"[geo_lookup] Nominatim reverse geocode failed: {e}")
        return {"display_name": "Unknown location (lookup failed)", "address": {}}
