# SOURCES.md — every external data/AI source used by AGNIDRISHTI

All sources below are real, currently operating, and free or free-tier. Nothing in this
build depends on a paid service, a service requiring a credit card, or Google Earth Engine
(pending approval — see the note at the bottom).

| Source | Provides | Key required? | URL |
|---|---|---|---|
| NASA FIRMS | Raw thermal hotspot detections: lat, lon, acquisition time, brightness, FRP (fire radiative power), confidence, satellite/sensor. This is Layer 1's primary input. | Yes — free, instant, self-service | API: https://firms.modaps.eosdis.nasa.gov/api/ · Get key: https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| OpenStreetMap Overpass API | Infrastructure context: nearest industrial facility/landuse=industrial polygon, factories, refineries, mines, and distance from a hotspot to the nearest one. Feeds Branch B. | No | https://overpass-api.de/ (public instance: https://overpass-api.de/api/interpreter) |
| NASA POWER | Recent weather/land-surface context: temperature, relative humidity, precipitation, root-zone soil moisture at a coordinate. Feeds Branch B. | No | https://power.larc.nasa.gov/ (API: https://power.larc.nasa.gov/api/temporal/daily/point) |
| Esri World Imagery | High-resolution aerial/satellite basemap tiles, used as the primary satellite image source for a clicked hotspot (Branch A input image). Tried first. | No | https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer |
| EOX Sentinel-2 Cloudless | Sentinel-2 cloudless mosaic tiles, used as the automatic fallback satellite image source if Esri is unreachable. | No | https://tiles.maps.eox.at/ (layer: s2cloudless-2020) |
| Leaflet.js + OpenStreetMap tiles | The base interactive map itself (India-wide hotspot view), OSM raster tile layer. | No | https://leafletjs.com/ · https://www.openstreetmap.org/ |
| Google Gemini API | Branch A (Vision) primary classification engine: given the fetched satellite image + a structured prompt, returns a JSON classification (industrial vs. natural-looking surroundings), confidence, and evidence points. This stands in for a not-yet-trained CNN — see `/methodology`. | Yes — free tier | https://aistudio.google.com/app/apikey |
| Groq | Alternate/ensemble vision-reasoning provider (Llama-family vision models), run alongside Gemini for the agreement/ensemble view. | Yes — free tier | https://console.groq.com/keys |
| OpenRouter | Aggregator exposing several genuinely free-tier vision/text models, used as a second alternate/ensemble provider. | Yes — free tier | https://openrouter.ai/keys |

## Pending / not a dependency

**Google Earth Engine** — the team is separately applying for access. It is *not* required
anywhere in this build. The imagery-provider interface (`app/data_sources/imagery.py`) and
the land-cover lookup are written behind a small abstract provider class specifically so a
`GoogleEarthEngineProvider` can be dropped in later, once access is granted, without
touching any other part of the system (fusion, UI, branches). Until then, imagery comes
from Esri World Imagery with automatic fallback to EOX Sentinel-2 Cloudless.

## Fallback / failure behavior

Every external call in `app/data_sources/` and `app/branches/` uses an explicit timeout and
never raises an unhandled exception up to the UI. On failure (missing key, timeout, non-200
response), the call returns a structured "unavailable" result that is surfaced as a plain
line in the on-page status/log panel (e.g. `[FIRMS] ERROR: request timed out after 10s`) and
as an explicit `"status": "unavailable"` field in the JSON API response, so the fusion engine
and the UI can visibly degrade rather than silently fail or crash.
