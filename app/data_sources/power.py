"""
NASA POWER client — recent weather / soil moisture context for Branch B.

No key required. Docs: https://power.larc.nasa.gov/docs/services/api/
Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx

from app.config import settings

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "T2M,RH2M,PRECTOTCORR,GWETROOT"  # temp, humidity, precip, root-zone soil moisture

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


def get_recent_weather_soil(lat: float, lon: float, days: int = 7, log: LogFn = _noop_log) -> dict:
    """Returns {status, is_real, temp_c, humidity_pct, precip_mm, soil_moisture}
    averaged over the last `days` days of available data."""
    end = datetime.now(timezone.utc).date() - timedelta(days=3)  # POWER lags a few days
    start = end - timedelta(days=days)
    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        log(f"[POWER] Querying weather/soil for ({lat:.4f}, {lon:.4f})...")
        with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            resp = client.get(POWER_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        params_data = data["properties"]["parameter"]

        def _avg(param_key):
            vals = [v for v in params_data.get(param_key, {}).values() if v is not None and v > -900]
            return round(sum(vals) / len(vals), 2) if vals else None

        result = {
            "status": "ok",
            "is_real": True,
            "temp_c": _avg("T2M"),
            "humidity_pct": _avg("RH2M"),
            "precip_mm": _avg("PRECTOTCORR"),
            "soil_moisture": _avg("GWETROOT"),
        }
        log(f"[POWER] temp={result['temp_c']}C humidity={result['humidity_pct']}% "
            f"precip={result['precip_mm']}mm soil={result['soil_moisture']}")
        return result
    except httpx.TimeoutException:
        log(f"[POWER] ERROR: request timed out after {settings.HTTP_TIMEOUT_SECONDS}s")
    except Exception as e:  # noqa: BLE001
        log(f"[POWER] ERROR: {e}")
    return {
        "status": "unavailable", "is_real": False,
        "temp_c": None, "humidity_pct": None, "precip_mm": None, "soil_moisture": None,
    }
