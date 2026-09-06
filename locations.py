"""
locations.py
------------
REAL, named locations used for the live demo — not random points. Each one
is a well-documented site chosen because NASA FIRMS genuinely detects
thermal activity there, for a genuinely different underlying reason, which
is exactly the ambiguity ThermoWatch AI exists to resolve. Coordinates are
site-level approximations (a few hundred metres), which is the right
precision for a satellite-image bounding box and a live demo narration —
not for navigation.
"""

DEMO_LOCATIONS = {
    "jamnagar_refinery": {
        "label": "Reliance Jamnagar Refinery Complex, Gujarat",
        "lat": 22.3350,
        "lon": 69.8397,
        "expected_class": "persistent_industrial_source",
        "story": (
            "The single largest oil refining complex on Earth — roughly "
            "1.24 million barrels/day of capacity. Its flare stacks and "
            "process heaters run hot every single night of the year. This "
            "is the textbook 'persistent industrial source': FIRMS lights "
            "up here on repeat, and it is never a wildfire and never news."
        ),
    },
    "jharia_coalfield": {
        "label": "Jharia Coalfield, Jharkhand",
        "lat": 23.7401,
        "lon": 86.4141,
        "expected_class": "persistent_industrial_source",
        "story": (
            "Underground coal-seam fires here have burned continuously "
            "since 1916 — over a century straight. Entire towns have been "
            "relocated because the ground itself is on fire. One of the "
            "most extreme real-world 'persistent thermal source' cases "
            "on the planet, and a genuinely dramatic demo location."
        ),
    },
    "hazira_gas_complex": {
        "label": "Hazira Gas & Petrochemical Complex, Surat, Gujarat",
        "lat": 21.1167,
        "lon": 72.6500,
        "expected_class": "gas_flare",
        "story": (
            "A major onshore gas-processing and flaring hub on India's "
            "west coast (ONGC + private operators). A routine, nightly "
            "gas-flare thermal signature — different root cause from a "
            "refinery, same 'why is this hot every night' question."
        ),
    },
    "punjab_stubble_belt": {
        "label": "Punjab Stubble-Burning Belt, near Patiala",
        "lat": 30.3398,
        "lon": 76.3869,
        "expected_class": "agricultural_burn",
        "story": (
            "Every October-November this belt lights up with thousands of "
            "FIRMS detections within days, as farmers burn paddy stubble "
            "ahead of the wheat season — a massive, recurring, seasonal "
            "'agricultural burn' signature that swamps raw hotspot feeds "
            "every single year and is a real, cited air-quality crisis."
        ),
    },
    "bandipur_forest": {
        "label": "Bandipur National Park, Karnataka",
        "lat": 11.6717,
        "lon": 76.6344,
        "expected_class": "wildfire",
        "story": (
            "Dry deciduous forest that has suffered major wildfire seasons "
            "(notably February 2019). No industry, no facility, no flare "
            "stack nearby — the clean 'wildfire' control case that proves "
            "the system doesn't just call everything industrial."
        ),
    },
}


def location_summary_table() -> str:
    """Small text table for quick printing in a notebook cell."""
    lines = [f"{'Key':<22}{'Location':<45}{'Expected class':<28}Lat/Lon"]
    lines.append("-" * 110)
    for key, site in DEMO_LOCATIONS.items():
        lines.append(
            f"{key:<22}{site['label']:<45}{site['expected_class']:<28}"
            f"{site['lat']:.4f}, {site['lon']:.4f}"
        )
    return "\n".join(lines)
