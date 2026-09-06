# AGNIDRISHTI

AI-based detection and classification of industrial fires and persistent thermal sources,
using NASA FIRMS, OpenStreetMap, and satellite imagery. Built for SIH 2026, Problem
Statement 26162.

See [`SOURCES.md`](SOURCES.md) for every external data/AI source used and whether it needs a
key, and the in-app [`/about`](http://127.0.0.1:8000/about) page for a plain, table-style
breakdown of what's real vs. placeholder in this prototype.

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (all optional — the app runs and degrades gracefully with none set)
cp .env.example .env
# edit .env and fill in whichever keys you have — see .env.example for where to get each one free

# 4. Run
uvicorn app.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/ — you'll see a live map of India with thermal hotspots.
Click any marker to run the full analysis pipeline for that point.

## What works with zero configuration

- The India-wide hotspot map (synthetic demo scatter, clearly labeled, until `FIRMS_MAP_KEY` is set)
- Real satellite imagery fetch (Esri World Imagery / EOX Sentinel-2 Cloudless — no key needed)
- The full image-processing gallery (grayscale, edges, false color, k-means segmentation)
- Branch B (Context) — real Overpass + NASA POWER queries, real Random Forest, real held-out evaluation
- Branch C (Temporal) — real or simulated history + real anomaly z-score math
- Fusion, risk scoring, and every chart on the analysis dashboard

## What needs an API key

- Live NASA FIRMS hotspot data (`FIRMS_MAP_KEY` — free, instant, from https://firms.modaps.eosdis.nasa.gov/api/map_key/)
- Branch A (Vision) classification — `GEMINI_API_KEY` (primary), plus optional `GROQ_API_KEY`
  and `OPENROUTER_API_KEY` for the ensemble/agreement view. Without any of these, Branch A
  visibly abstains rather than crashing, and the UI shows "NO API KEY CONFIGURED" per provider.

## Project layout

```
app/
  data_sources/   Layer 1 input: FIRMS, Overpass, NASA POWER, imagery providers
  pipeline/       DBSCAN event clustering
  branches/       Branch A (vision), B (context/RF), C (temporal), fusion
  processing/     Image-processing gallery (grayscale/edges/false-color/k-means)
  charts/         Matplotlib dashboard chart generation
  routes/         FastAPI routes (pages + JSON API)
  templates/      Jinja2 HTML (old-school operational-tool styling)
  static/         CSS + Leaflet map JS
data/             SQLite DB + cached imagery (gitignored)
```

## Notes for judges / reviewers

This is a hackathon prototype and says so, in the product itself: every placeholder
component (Gemini standing in for a trained CNN, a Random Forest trained on synthetic data,
simulated thermal history) is labeled directly in the UI and API response, and the
`/about` page lays out exactly what's real vs. placeholder vs. planned, component by
component.
