from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.data_sources.firms import fetch_firms_hotspots
from app.data_sources.imagery import get_satellite_image
from app.data_sources.overpass import nearest_industrial_facility
from app.data_sources.power import get_recent_weather_soil
from app.database import upsert_hotspots, get_hotspots, save_analysis, list_analyses
from app.pipeline.clustering import cluster_hotspots_into_events, find_event_for_point
from app.processing.image_ops import build_image_gallery
from app.branches.vision import run_vision_branch
from app.branches.context_model import get_context_model, FEATURE_COLUMNS
from app.branches.temporal import get_history, classify_temporal
from app.branches.fusion import fuse
from app.charts import charts as chart_lib

router = APIRouter()


class AnalyzeRequest(BaseModel):
    latitude: float
    longitude: float
    hotspot_id: str | None = None
    frp: float | None = None


@router.get("/hotspots")
def api_hotspots(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    refresh: bool = Query(True, description="Re-fetch from FIRMS (or synthetic fallback) before returning"),
):
    log: list[str] = []

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log.append(f"{ts} UTC  {msg}")

    if refresh:
        _log("[PIPELINE] Layer 1: fetching current hotspot detections...")
        rows = fetch_firms_hotspots(log=_log)
        is_synthetic = bool(rows) and rows[0].get("source") == "synthetic_demo"
        n = upsert_hotspots([{k: v for k, v in r.items() if k != "source"} for r in rows])
        _log(f"[DB] Upserted {n} hotspot rows into local store.")
        events = cluster_hotspots_into_events(rows, log=_log)
    else:
        is_synthetic = False

    stored = get_hotspots(start_date, end_date)
    _log(f"[PIPELINE] Returning {len(stored)} hotspots for map render.")

    return {
        "hotspots": stored,
        "count": len(stored),
        "is_synthetic_demo": is_synthetic,
        "last_updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "log": log,
    }


@router.post("/analyze")
def api_analyze(req: AnalyzeRequest):
    log: list[str] = []

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log.append(f"{ts} UTC  {msg}")

    lat, lon = req.latitude, req.longitude
    _log(f"[PIPELINE] Starting analysis for ({lat:.4f}, {lon:.4f})")

    event_id = find_event_for_point(lat, lon)

    # --- imagery + image processing gallery ---
    _log("[PIPELINE] Fetching satellite image...")
    imagery_result = get_satellite_image(lat, lon, log=_log)
    _log("[PIPELINE] Building image-processing gallery (grayscale, edges, false color, k-means)...")
    gallery = build_image_gallery(imagery_result.image)

    # --- Branch A: Vision ---
    _log("[PIPELINE] Branch A (Vision): running provider(s)...")
    vision = run_vision_branch(imagery_result.image, lat, lon, log=_log)

    # --- Branch B: Context/Tabular ---
    _log("[PIPELINE] Branch B (Context): querying Overpass + NASA POWER...")
    facility = nearest_industrial_facility(lat, lon, log=_log)
    weather = get_recent_weather_soil(lat, lon, log=_log)
    facility_type = facility.get("facility_type") or ""
    features = {
        "distance_m": facility.get("distance_m") if facility.get("found") else 6000.0,
        "is_industrial": 1 if facility_type == "industrial" else 0,
        "is_refinery_power": 1 if facility_type in ("power",) else 0,
        "is_mine": 1 if facility_type == "mine" else 0,
        "frp": req.frp or 5.0,
        "brightness": 320.0,
        "soil_moisture": weather.get("soil_moisture") if weather.get("soil_moisture") is not None else 0.35,
        "temp_c": weather.get("temp_c") if weather.get("temp_c") is not None else 28.0,
        "humidity_pct": weather.get("humidity_pct") if weather.get("humidity_pct") is not None else 55.0,
        "precip_mm": weather.get("precip_mm") if weather.get("precip_mm") is not None else 1.0,
    }
    context_model = get_context_model()
    context = context_model.predict(features)
    _log(f"[BRANCH-B] Random Forest top class: {context['top_class']}")

    # --- Branch C: Temporal/Persistence ---
    _log("[PIPELINE] Branch C (Temporal): building/loading history and computing anomaly z-score...")
    history = get_history(lat, lon, latest_frp=req.frp)
    temporal = classify_temporal(history)
    _log(f"[BRANCH-C] {'Simulated' if history['simulated'] else 'Real'} history -> {temporal['top_class']} "
         f"(z={history['z_latest']})")

    # --- Layer 3: Fusion ---
    _log("[PIPELINE] Layer 3: fusing branches (weighted average)...")
    fusion_result = fuse(vision, context, temporal, req.frp)
    _log(f"[PIPELINE] Fused classification: {fusion_result['classification']} "
         f"({fusion_result['confidence']*100:.1f}%), risk={fusion_result['risk_score']}")

    # --- Charts ---
    _log("[PIPELINE] Rendering charts...")
    metrics = context_model.metrics
    curves = context_model.curves
    fpr, tpr = curves["roc"]
    precision, recall = curves["pr"]
    charts = {
        "branch_votes": chart_lib.branch_votes_chart(vision, context, temporal),
        "risk_gauge": chart_lib.risk_gauge_chart(fusion_result["risk_score"]),
        "feature_importance": chart_lib.feature_importance_chart(context_model.model, FEATURE_COLUMNS),
        "correlation_heatmap": chart_lib.correlation_heatmap_chart(context_model.training_df, FEATURE_COLUMNS),
        "history_line": chart_lib.history_line_chart(history),
        "anomaly_bar": chart_lib.anomaly_bar_chart(history),
        "confusion_matrix": chart_lib.confusion_matrix_chart(curves["confusion_matrix"], curves["labels"]),
        "roc_curve": chart_lib.roc_curve_chart(fpr, tpr, metrics["auc"]),
        "pr_curve": chart_lib.pr_curve_chart(precision, recall),
    }

    result = {
        "latitude": lat,
        "longitude": lon,
        "event_id": event_id,
        "imagery_source": imagery_result.source,
        "imagery_is_real": imagery_result.is_real,
        "branch_a": {
            "class_probabilities": vision["class_probabilities"],
            "top_class": vision["top_class"],
            "evidence": vision["evidence"],
            "used_provider": vision["used_provider"],
            "ensemble": vision["ensemble"],
            "model_label": vision["model_label"],
            "images": gallery,
        },
        "branch_b": {
            "class_probabilities": context["class_probabilities"],
            "top_class": context["top_class"],
            "evidence": context["evidence"],
            "features_used": features,
            "facility": facility,
            "weather": weather,
            "model_metrics": metrics,
        },
        "branch_c": {
            "class_probabilities": temporal["class_probabilities"],
            "top_class": temporal["top_class"],
            "evidence": temporal["evidence"],
            "simulated": history["simulated"],
            "history": history,
        },
        "fusion": fusion_result,
        "charts": charts,
    }

    analysis_id = str(uuid.uuid4())
    save_analysis(analysis_id, event_id, lat, lon, datetime.now(timezone.utc).isoformat(), result)
    _log("[PIPELINE] Analysis complete and saved.")

    return {"analysis_id": analysis_id, "result": result, "log": log}


@router.get("/analyses")
def api_analyses():
    return {"analyses": list_analyses()}


@router.get("/config")
def api_config():
    from app.config import settings
    return {
        "firms_key_configured": bool(settings.FIRMS_MAP_KEY),
        "gemini_key_configured": bool(settings.GEMINI_API_KEY),
        "groq_key_configured": bool(settings.GROQ_API_KEY),
        "openrouter_key_configured": bool(settings.OPENROUTER_API_KEY),
        "fusion_weights": settings.FUSION_WEIGHTS,
    }
