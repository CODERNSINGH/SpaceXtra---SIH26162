"""
context_fetcher.py
-------------------
REAL module: pulls actual context data from free public APIs —
OpenStreetMap (nearest industrial facility) and NASA POWER (weather/soil).
Nothing simulated in this file.
"""
import datetime
import requests
from utils import PipelineConfig, haversine_m

# Public Overpass mirrors, tried in order. overpass-api.de rejects requests
# with no Accept header (406) and is sometimes slow, so we send a proper
# header set, use a generous timeout, and fail over to a second mirror.
_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_OVERPASS_HEADERS = {
    "User-Agent": "ThermoWatchAI-SIH26162-Prototype/1.0 (student hackathon demo)",
    "Accept": "*/*",
}


def get_nearest_industrial_facility(lat, lon, cfg: PipelineConfig = None):
    cfg = cfg or PipelineConfig()
    r = cfg.overpass_radius_m
    query = f"""
    [out:json][timeout:25];
    (
      node["landuse"="industrial"](around:{r},{lat},{lon});
      way["landuse"="industrial"](around:{r},{lat},{lon});
      node["man_made"~"works|petroleum_well|chimney"](around:{r},{lat},{lon});
      way["man_made"~"works|petroleum_well|chimney"](around:{r},{lat},{lon});
      node["power"="plant"](around:{r},{lat},{lon});
      way["power"="plant"](around:{r},{lat},{lon});
    );
    out center;
    """
    result = {"nearest_industrial_facility": None, "facility_distance_m": None}
    overpass_timeout = max(cfg.request_timeout, 40)
    elements = None
    for url in _OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers=_OVERPASS_HEADERS, timeout=overpass_timeout)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except Exception as e:
            print(f"[context_fetcher] OSM lookup via {url} failed: {e}")

    if elements is None:
        return result

    best_name, best_dist = None, None
    for el in elements:
        tags = el.get("tags", {})
        elat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        elon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        if elat is None or elon is None:
            continue
        dist = haversine_m(lat, lon, elat, elon)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = tags.get("name") or tags.get("man_made") or tags.get("power") or "unnamed industrial site"
    result["nearest_industrial_facility"] = best_name
    result["facility_distance_m"] = round(best_dist, 1) if best_dist is not None else None
    return result


def get_weather_soil_context(lat, lon, cfg: PipelineConfig = None, days_back: int = 7):
    cfg = cfg or PipelineConfig()
    end_date = datetime.date.today() - datetime.timedelta(days=2)
    start_date = end_date - datetime.timedelta(days=days_back)
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,GWETROOT",
        "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": start_date.strftime("%Y%m%d"), "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    out = {"avg_temp_c": None, "avg_humidity_pct": None, "avg_rainfall_mm": None, "avg_root_zone_soil_moisture": None}
    try:
        resp = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]

        def avg(series):
            vals = [v for v in series.values() if v not in (-999, None)]
            return round(sum(vals) / len(vals), 2) if vals else None

        out["avg_temp_c"] = avg(data.get("T2M", {}))
        out["avg_humidity_pct"] = avg(data.get("RH2M", {}))
        out["avg_rainfall_mm"] = avg(data.get("PRECTOTCORR", {}))
        out["avg_root_zone_soil_moisture"] = avg(data.get("GWETROOT", {}))
    except Exception as e:
        print(f"[context_fetcher] NASA POWER lookup failed: {e}")
    return out


def get_full_context(lat, lon, cfg: PipelineConfig = None):
    ctx = get_nearest_industrial_facility(lat, lon, cfg)
    ctx.update(get_weather_soil_context(lat, lon, cfg))
    return ctx
