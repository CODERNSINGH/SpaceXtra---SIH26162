"""
Curated seed list of major, real, well-known Indian industrial facilities
(steel plants, refineries, thermal power stations, mining/smelting hubs,
petrochemical complexes). Coordinates are approximate (facility-scale, not
survey-grade) but real, named locations — not synthetic/invented data.

Why this exists: the live OpenStreetMap Overpass query in
`industrial_index.py` covers many more facilities but depends on a shared
public API that can rate-limit or time out (observed repeatedly during
development). This seed list guarantees the "industrial facilities only" map
filter still reflects genuine Indian industrial geography even if Overpass is
completely unreachable, and is merged with (not a replacement for) whatever
Overpass successfully returns.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

# (name, facility_type, latitude, longitude)
SEED_FACILITIES: list[tuple[str, str, float, float]] = [
    ("Tata Steel Jamshedpur", "steel", 22.8046, 86.2029),
    ("Bokaro Steel Plant", "steel", 23.6693, 86.1511),
    ("Durgapur Steel Plant", "steel", 23.5204, 87.3119),
    ("Rourkela Steel Plant", "steel", 22.2604, 84.8536),
    ("Bhilai Steel Plant", "steel", 21.1938, 81.3509),
    ("Visakhapatnam Steel Plant", "steel", 17.6300, 83.2200),
    ("JSW Steel Vijayanagar", "steel", 15.1856, 76.6425),
    ("JSW Dolvi Steel Plant", "steel", 18.5333, 73.0333),
    ("Jamnagar Refinery (Reliance)", "refinery", 22.3511, 69.8500),
    ("Barauni Refinery", "refinery", 25.4667, 85.9667),
    ("Mathura Refinery", "refinery", 27.4833, 77.6833),
    ("Panipat Refinery", "refinery", 29.3909, 76.9635),
    ("Numaligarh Refinery", "refinery", 26.5333, 93.7167),
    ("Digboi Refinery", "refinery", 27.3833, 95.6167),
    ("Haldia Refinery", "refinery", 22.0667, 88.0833),
    ("Vadodara Petrochemical Complex", "refinery", 22.3072, 73.1812),
    ("Dahej Petrochemical Complex", "refinery", 21.7167, 72.5667),
    ("Kochi Refinery", "refinery", 9.9667, 76.2667),
    ("Guwahati Refinery", "refinery", 26.1833, 91.7500),
    ("Korba Thermal Power Complex", "power", 22.3595, 82.7501),
    ("Singrauli Thermal Power Complex", "power", 24.1997, 82.6767),
    ("Ramagundam Thermal Power Station", "power", 18.7833, 79.4667),
    ("Neyveli Lignite Complex", "power", 11.6167, 79.4833),
    ("Talcher Thermal Power Station", "power", 20.9500, 85.2333),
    ("Vindhyachal Thermal Power Station", "power", 24.0964, 82.6864),
    ("Chandrapur Super Thermal Power Station", "power", 19.9975, 79.3206),
    ("Angul Aluminium/Steel Complex (NALCO)", "power", 20.8400, 85.1000),
    ("Sindri Fertilizer Complex", "industrial", 23.6667, 86.7167),
    ("Tuticorin Copper Smelter", "industrial", 8.7642, 78.1348),
    ("Hazira Industrial Complex", "industrial", 21.1167, 72.6500),
    ("Vapi Industrial Estate", "industrial", 20.3714, 72.9047),
    ("Ankleshwar Industrial Estate", "industrial", 21.6266, 73.0104),
    ("SIPCOT Cuddalore", "industrial", 11.7480, 79.7714),
    ("Salem Steel Plant", "steel", 11.6643, 78.1460),
    ("Manesar Industrial Area", "industrial", 28.3670, 76.9425),
    ("Faridabad Industrial Area", "industrial", 28.4089, 77.3178),
    ("Ludhiana Industrial Belt", "industrial", 30.9010, 75.8573),
    ("Kanpur Industrial Belt", "industrial", 26.4499, 80.3319),
    ("Asansol-Raniganj Coal Belt", "mine", 23.6739, 86.9524),
    ("Dhanbad-Jharia Coalfield", "mine", 23.7957, 86.4304),
    ("Korba Coalfield", "mine", 22.3460, 82.7010),
    ("Singrauli Coalfield", "mine", 24.1994, 82.6749),
    ("Bellary-Hospet Iron Ore Belt", "mine", 15.1394, 76.9214),
    ("Keonjhar Iron Ore Belt", "mine", 21.6297, 85.5817),
    ("Goa Iron Ore Mining Belt", "mine", 15.4989, 74.0083),
    ("Jharia-Bokaro Coking Coal Belt", "mine", 23.7500, 86.4000),
    ("Rajasthan Zinc Smelter (Udaipur)", "industrial", 24.5854, 73.7125),
    ("Paradip Port Industrial Area", "industrial", 20.3167, 86.6167),
    ("Kandla Port Industrial Area", "industrial", 23.0333, 70.2167),
    ("MIHAN Nagpur", "industrial", 21.0900, 79.0500),
    ("Pune Auto Industrial Belt (Chakan)", "industrial", 18.7621, 73.8637),
    ("Sriperumbudur Auto Belt", "industrial", 12.9675, 79.9430),
]


def get_seed_facilities() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for name, ftype, lat, lon in SEED_FACILITIES:
        rows.append({
            "id": "seed_" + hashlib.sha1(name.encode()).hexdigest()[:16],
            "latitude": lat,
            "longitude": lon,
            "facility_type": ftype,
            "name": name,
            "fetched_at": now,
        })
    return rows
