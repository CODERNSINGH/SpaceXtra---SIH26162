# ThermoWatch AI — Prototype Explainer

**Smart India Hackathon — Problem Statement 26162**
*AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources Using NASA FIRMS, OSM & Satellite Data*

This document explains exactly what this prototype does, file by file, chart by chart, and —
most importantly — draws a hard line between **what is real** and **what is a labeled
placeholder**, so the team can answer any judge's question honestly and confidently.

This prototype follows the architecture in `ThermoWatch_AI_Architecture_Guide.pdf` (Layer 1 →
Layer 2's three branches → Layer 3 fusion). It is a runnable, presentable demo of that
architecture, built ahead of the real labeled datasets described in that document.

---

## 1. The one honest sentence

> Every network call, image transform, and statistic in this prototype either **really
> happened** against a real free API / real pixels / a real scikit-learn model, or is **clearly
> labeled in code comments and on-screen** as a placeholder standing in for a component the
> architecture doc says needs a trained model or a labeled dataset we don't have yet.

Nothing here is silently faked. If a judge points at any chart and asks "is this real," the
answer is already written in that chart's title or the module's docstring.

---

## 2. Repo map

```
locations.py          Real, named demo sites (Jamnagar refinery, Jharia coalfield, etc.)
utils.py               Config + small real helper math (haversine distance, seeded RNG)
geo_lookup.py          REAL — reverse geocoding via OSM Nominatim
image_fetcher.py       REAL — fetches a live satellite image (Esri / EOX)
image_processing.py    REAL — grayscale/blur/sharpen/edges/k-means + SIMULATED thermal/NDVI views
context_fetcher.py     REAL — OSM Overpass (nearest facility) + NASA POWER (weather/soil)
gemini_setup.py         Loads a Gemini API key and returns a callable model wrapper
vision_model.py         PLACEHOLDER — Gemini stands in for a trained CNN (Branch A)
tabular_model.py        REAL MODEL / SYNTHETIC DATA — Random Forest (Branch B)
temporal_model.py       REAL MATH / SIMULATED HISTORY — z-score anomaly detection (Branch C)
evaluation.py           REAL — held-out confusion matrix, ROC, PR, decision boundary
fusion_engine.py        REAL — weighted-average fusion, multi-class heuristic, stability check
map_view.py             REAL — interactive Leaflet maps via folium
dashboard_visuals.py    Chart-drawing primitives (each is real math or clearly labeled)
dashboards.py           Composes primitives into 5 named "Dashboard" figures (~30 charts)
pipeline.py             Orchestrator — the only file the notebook needs to call
thermowatch_full_prototype.ipynb   The Colab notebook: writes every file above to disk, then runs it
```

Every `.py` file also exists standalone in the repo root — the notebook's `%%writefile` cells are
generated directly from these files, so notebook and disk can never drift apart.

---

## 3. Why these 5 locations

Instead of random coordinates, the demo runs on five **real, documented, named** Indian sites —
chosen so every one of the SIH problem statement's target classes has a genuine real-world
example, and so the system is tested on cases it wasn't hand-tuned for:

| Site | Real-world story | Expected class |
|---|---|---|
| **Reliance Jamnagar Refinery**, Gujarat | The single largest oil refining complex on Earth (~1.24M barrels/day). Flare stacks run hot every night, year-round. | `persistent_industrial_source` |
| **Jharia Coalfield**, Jharkhand | Underground coal-seam fires burning continuously **since 1916** — over a century. Towns have been relocated because the ground is on fire. | `persistent_industrial_source` |
| **Hazira Gas & Petrochemical Complex**, Gujarat | Major onshore gas-processing/flaring hub on India's west coast. | `gas_flare` |
| **Punjab Stubble-Burning Belt**, near Patiala | Every Oct–Nov, thousands of FIRMS detections appear within days as farmers burn paddy stubble — a real, cited seasonal air-quality crisis. | `agricultural_burn` |
| **Bandipur National Park**, Karnataka | Dry deciduous forest with major wildfire seasons (notably Feb 2019). No industry nearby — the clean control case. | `wildfire` |

Jharia in particular is worth leading with on stage: a coal fire that has been burning for over
100 years is a genuinely dramatic, true, and citable example of exactly the "persistent thermal
source" problem this system exists to flag.

---

## 4. Layer-by-layer walkthrough

### Layer 1 — Input & Preprocessing (partially real)
- **Coordinate → satellite image**: `image_fetcher.py` makes a real HTTP request to Esri World
  Imagery (falling back to EOX Sentinel-2 Cloudless) for a ~1.3 km box around the point. This is a
  genuine image of that real place, fetched live.
- The architecture doc's DBSCAN clustering and FIRMS-archive temporal analysis are **not run
  live** in this prototype (there's no live FIRMS hotspot feed wired up yet) — `temporal_model.py`
  simulates a plausible 30-day history *per coordinate* instead. The anomaly-detection *math* run
  on top of that history (rolling mean, standard deviation, z-score) is real and unchanged from
  what would run on a real FIRMS archive.

### Layer 2 — Three Parallel Branches

**Branch A — Vision (`vision_model.py`, PLACEHOLDER for the CNN)**
The architecture doc calls for a CNN (ResNet/EfficientNet) fine-tuned on satellite imagery
(FlareSat, Kaggle Wildfire dataset). That takes a labeled dataset and training time. For today,
this branch sends the real fetched image to **Google's Gemini** with a fixed prompt asking it to
judge "industrial-looking vs natural-looking," and parses the JSON response. If Gemini is
unavailable or the API call fails for any reason (bad key, quota, network), it falls back to a
**deterministic simulated response** seeded from the coordinate — the pipeline never crashes, and
every result honestly reports its own `source` field (`gemini_vision:<model>` or
`simulated_fallback`).

**Branch B — Context / Tabular (`tabular_model.py`, real model / synthetic data)**
This is a genuine `sklearn.ensemble.RandomForestClassifier` that really calls `.fit()` and
`.predict_proba()`. Its inputs are **real**: live facility distance from OpenStreetMap, live
temperature/humidity/rainfall/soil-moisture from NASA POWER. What's synthetic is the *training
set* — 800 synthetic rows generated by a simple rule ("closer to industry + drier soil → more
industrial-like"), because no labeled `thermal_event_dataset.csv` exists yet. Swap in a real CSV
and every line of this file keeps working unchanged.

**Branch C — Temporal / Persistence (`temporal_model.py`, real math / simulated history)**
Builds a 30-day synthetic FRP history per coordinate (seeded so it's stable across demo re-runs),
injects a plausible spike on the final day, then computes a genuine z-score of the latest reading
against the trailing baseline. This is exactly the anomaly-detection approach the architecture doc
recommends as the cheapest, strongest differentiator — the only missing piece for production is a
real FIRMS archive pull to replace the simulated history.

### Layer 3 — Decision & Output

**Fusion Engine (`fusion_engine.py`, real)**
A transparent weighted average of the three branches' industrial-probability signals
(`vision: 0.4, context: 0.4, temporal: 0.2` by default) — exactly the "Model 0" fusion the
architecture doc recommends before upgrading to a learned meta-classifier. Confidence is the
fused score's distance from a 50/50 coin flip; risk score blends confidence with the temporal
anomaly score.

**Multi-Class Breakdown (heuristic, explicitly not a trained model)**
The SIH problem statement wants 7 output classes (industrial fire, persistent industrial source,
wildfire, agricultural burn, gas flare, mining activity, unknown), but this prototype only has a
*binary* trained signal (industrial vs. natural). `derive_multiclass_distribution()` redistributes
that binary score across the 7 classes using simple, explainable rules (e.g., "close to a facility
+ high anomaly → weight industrial_fire heavily"). This is clearly not a substitute for a real
multi-class classifier trained on labeled examples of all 7 classes — say so if asked.

**Model Evaluation (`evaluation.py`, real metrics / synthetic data)**
A **second**, held-out evaluation: the synthetic dataset is split 75/25, the Random Forest is
retrained on the 75% and evaluated on the untouched 25%. The confusion matrix, ROC curve
(AUC), precision-recall curve, and accuracy/precision/recall/F1/ROC-AUC bars are all genuinely
computed by scikit-learn on data the model never saw during training. The honest caveat: since the
underlying data is still synthetic, these numbers describe how well the model learned its own
synthetic rule — not real-world accuracy. A 2D decision-boundary contour (`facility_distance_m`
vs. `avg_root_zone_soil_moisture`) is drawn from a second real 2-feature Random Forest, restricted
to 2 inputs purely so its decision surface can be visualized.

**Stability Check (`fusion_engine.run_stability_check`, real)**
A genuine Monte Carlo test: each branch's signal is perturbed by ±5% and the *real* fusion formula
is re-run 200 times, and the spread of resulting confidence/risk scores is plotted as a boxplot.
This is a real robustness check on the real fusion math — it just happens to be exercising the
fused numbers with synthetic jitter rather than genuinely independent sensor noise.

**Maps (`map_view.py`, real)**
Interactive Leaflet maps via `folium` — a real, clickable, zoomable map with the event marker and
close/medium/far analysis rings, plus an India-wide map of all five demo sites. This is a
lightweight stand-in for the full PostGIS + MapLibre GIS dashboard in the architecture doc's Layer
3, but it is a genuine interactive map, not a screenshot.

---

## 5. The five dashboards (~30 charts total)

Run for one event, in order:

1. **Dashboard 1 — Image Intelligence**: original image, simulated thermal view, edge detection,
   k-means segmentation, hotspot mask, simulated vegetation index (plus a full 16-panel gallery
   with every individual transform and the 4 zoom quadrants).
2. **Dashboard 2 — Context & Branch Analysis**: branch vote comparison, Random Forest feature
   importance, live weather/soil context (normalized), feature correlation heatmap, facility
   proximity (with close/medium/far zone lines), multi-class heuristic breakdown.
3. **Dashboard 3 — Vision Branch Deep-Dive**: the image the vision branch actually saw, two more
   simulated colormap views, the vision branch's own industrial/natural verdict donut, an
   illustrative (decorative, clearly labeled) activation bar chart, and Gemini's raw text notes.
4. **Dashboard 4 — Model Evaluation**: real confusion matrix, ROC curve, precision-recall curve,
   2D decision boundary, prediction-confidence histogram split by true label, held-out metrics
   summary.
5. **Dashboard 5 — Final Fusion & Risk**: branch votes feeding fusion, fused classification pie,
   risk gauge, all-branch radar, final multi-class verdict (top class highlighted), stability
   check boxplot.

Plus a **multi-site comparison chart**: risk score per real site, colored green where the
system's top multi-class guess matched that site's documented real-world class.

---

## 6. Known rough edges (say these before a judge finds them)

- **Free public APIs are not enterprise infrastructure.** OpenStreetMap's Overpass API and
  Nominatim occasionally rate-limit (`429`) or time out under repeated calls in a short window —
  the code retries against a second Overpass mirror and always fails gracefully (the context
  branch just falls back to sensible defaults and says so), but a live demo can hit this. Run the
  flagship location once before going on stage so results are warmed up / cached in your head.
- **Gemini model availability depends on your Google account's tier.** The Gemini API's free tier
  restricts which models are callable per-project; a brand-new or free-tier project may see
  403/404/429 on every model. The vision branch's simulated fallback means the demo **never
  breaks** because of this — it just means that run used the labeled simulated response instead of
  a real Gemini call. Check `https://aistudio.google.com/apikey` for your project's access if you
  want the real calls to succeed live.
- **The multi-class breakdown is a heuristic**, not a trained 7-way classifier — don't present it
  as one.
- **The tabular model's headline "94% accuracy"-style numbers describe a synthetic dataset**, not
  real-world performance. Always mention this in the same breath as the number.

---

## 7. API key / secrets handling

This prototype calls Gemini using **your own API key**, loaded from a local `.env` file
(`GEMINI_API_KEY=...`), never hardcoded into a tracked file. `.env` is listed in `.gitignore`, so
`git status` / `git add` will never pick it up. `.env.example` is the tracked template showing the
expected shape.

**If an API key is ever pasted into a chat, a shared doc, a Slack message, or a commit, treat it
as compromised and regenerate it** at `https://aistudio.google.com/apikey` — a leaked key can be
used by anyone who saw it, even if you later delete the message.

In Colab (a separate machine with no access to your local `.env`), the notebook's Gemini-setup
cell prompts for the key via a hidden `getpass` input instead — it lives only in that session's
memory and is never written back into the notebook file.

---

## 8. What a real, non-prototype build still needs

Straight from the architecture doc's own build order, unchanged by this prototype:

1. A live NASA FIRMS ingestion pipeline (this prototype starts from a known coordinate, not a
   live hotspot feed).
2. A real labeled dataset for Branch B (OSM + land-cover + night-lights features with real
   ground-truth labels) to replace the synthetic training set.
3. A real fine-tuned CNN (ResNet/EfficientNet on FlareSat/Kaggle imagery) to replace the Gemini
   placeholder in Branch A — the interface (`run_vision_branch`) is already shaped so this is a
   one-file swap.
4. A real FIRMS-archive pull per facility to replace Branch C's simulated 30-day history — the
   anomaly-score math already works unchanged on real data.
5. A learned meta-classifier (logistic regression) to replace the fixed-weight fusion formula,
   once labeled fusion examples exist.
6. The full PostGIS + FastAPI + MapLibre + React GIS dashboard described in Layer 3, Step 3.3 —
   this prototype's `folium` maps are a lightweight stand-in.
