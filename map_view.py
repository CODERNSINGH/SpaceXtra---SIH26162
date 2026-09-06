"""
map_view.py
-----------
REAL module: builds an actual interactive Leaflet map (via folium) showing
the event location and analysis-radius rings — a lightweight stand-in for
the full PostGIS + MapLibre GIS dashboard described in Layer 3 of the
architecture PDF, but a genuine, clickable, zoomable map, not a screenshot.
"""
import folium


def build_event_map(lat: float, lon: float, context: dict, label: str = "") -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")

    dist = context.get("facility_distance_m")
    facility = context.get("nearest_industrial_facility")
    popup_lines = [f"<b>{label or 'Thermal event'}</b>", f"Lat/Lon: {lat:.4f}, {lon:.4f}"]
    if facility:
        popup_lines.append(f"Nearest OSM industrial facility: {facility} ({dist} m)")

    folium.Marker(
        [lat, lon],
        tooltip=label or "Thermal event",
        popup="<br>".join(popup_lines),
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    for radius_m, color, name in [(500, "#c62828", "close zone"),
                                   (1500, "#f9a825", "medium zone"),
                                   (3000, "#2e7d32", "far zone")]:
        folium.Circle(
            [lat, lon], radius=radius_m, color=color, fill=False, weight=2,
            tooltip=f"{name}: {radius_m} m radius",
        ).add_to(m)

    return m


def build_multi_site_map(locations: dict) -> folium.Map:
    """One India-wide map with every demo location plotted and color-coded
    by its expected class — the closest thing to the full GIS dashboard's
    "map + filters" view that this notebook prototype can offer."""
    m = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="OpenStreetMap")
    color_by_class = {
        "persistent_industrial_source": "blue",
        "gas_flare": "purple",
        "agricultural_burn": "orange",
        "wildfire": "red",
        "industrial_fire": "darkred",
        "mining_activity": "cadetblue",
        "unknown": "gray",
    }
    for site in locations.values():
        folium.Marker(
            [site["lat"], site["lon"]],
            tooltip=site["label"],
            popup=f"<b>{site['label']}</b><br>{site['story']}",
            icon=folium.Icon(
                color=color_by_class.get(site["expected_class"], "gray"),
                icon="info-sign",
            ),
        ).add_to(m)
    return m
