# AGNIDRISHTI

AI-based detection and classification of industrial fires and persistent thermal sources,
using NASA FIRMS, OpenStreetMap, and satellite imagery. Built for SIH 2026, Problem
Statement 26162.

See [`SOURCES.md`](SOURCES.md) for every external data/AI source used and whether it needs a
key, and the in-app [`/about`](http://127.0.0.1:8000/about) page for a plain, table-style
breakdown of what's real vs. placeholder in this prototype.

## How to run — step by step

**Prerequisites:** Python 3.10+ and `pip`. No database server, no Docker, no paid
service of any kind is required.

### 1. Get the code and go to the project folder

If you haven't already, `cd` into the project directory (the one containing this
README and `requirements.txt`).

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd.exe):
venv\Scripts\activate.bat
```

Your terminal prompt should now show `(venv)` at the start of the line.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, uvicorn, scikit-learn, matplotlib, Pillow, httpx, and the rest —
takes a minute or two on first install.

### 4. Configure API keys (optional — the app runs with zero keys set)

```bash
cp .env.example .env
```

Then open `.env` in a text editor and fill in whichever keys you have. Every key is
free; none require a credit card. Leave any of them blank and that feature degrades
gracefully (clear "NOT CONFIGURED" messaging in the UI/log) instead of crashing:

| Variable | Get a free key at | Unlocks |
|---|---|---|
| `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/map_key/ (instant) | Live NASA FIRMS hotspots (otherwise a labeled synthetic demo scatter) |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey | Branch A vision classification (primary engine) |
| `GROQ_API_KEY` | https://console.groq.com/keys | Branch A ensemble provider #2 |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | Branch A ensemble provider #3 |

**Never commit your `.env` file** — it's already listed in `.gitignore`. Only
`.env.example` (with blank/placeholder values) should ever be committed.

### 5. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal running —
that's your server. (`--reload` is for development, so edits to the code restart the
server automatically; drop it for a plain run.)

> **First run only:** on startup the app seeds a small list of major Indian industrial
> facilities instantly, then kicks off a one-time background fetch from OpenStreetMap to
> enrich that list — this can take a couple of minutes. The "industrial facilities only"
> map filter works immediately off the seed list while that finishes; watch the System
> Log panel on the page, or check `GET /api/industrial-index/status`, to see progress.

### 6. Open it in your browser

Go to **http://127.0.0.1:8000/**. You should see a live map of India with thermal
hotspots plotted. Click any marker to run the full multi-branch analysis pipeline for
that point. Visit **http://127.0.0.1:8000/about** for the methodology/honesty page.

### 7. Stop the server

Press `Ctrl+C` in the terminal where uvicorn is running.

### Troubleshooting

- **Port already in use:** run with a different port, e.g. `--port 8001`, and open that
  port in the browser instead.
- **Want a totally clean slate** (hotspots, cached analyses, industrial-facility index):
  stop the server and delete `data/thermowatch.db` — it's recreated automatically on the
  next run.
- **A vision provider (Gemini/Groq/OpenRouter) shows an error in the ensemble table:**
  check the System Log panel for the specific message — free-tier model availability on
  Groq/OpenRouter changes over time; see the notes in `.env.example` for how to find a
  current vision-capable model and update `GROQ_MODEL` / `OPENROUTER_MODEL`.
- **Industrial-facility index stuck at "building":** the public OpenStreetMap Overpass
  API can be slow or rate-limited; the seed list already makes the map usable in the
  meantime, and enrichment will pick up whenever Overpass responds.

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
