"""
utils.py
--------
Shared helpers used across every module in this prototype: config, distance
math, and small formatting helpers. Nothing here is "fake" — this is real,
reusable plumbing.
"""
import math
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    image_buffer_deg: float = 0.012      # ~1.3 km box around the point
    image_size: int = 512
    overpass_radius_m: int = 2000
    request_timeout: int = 30
    random_seed: int = 42


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def coord_seed(lat: float, lon: float) -> int:
    """Turn a coordinate into a stable random seed, so 'simulated' data for the
    same location looks the same every time you run the demo (important for
    rehearsing a pitch)."""
    return abs(int((lat * 1000003 + lon * 999983))) % (2**31 - 1)


def pct(x, digits=1):
    return f"{round(x, digits)}%"


def banner(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
