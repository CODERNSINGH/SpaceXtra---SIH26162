"""
pipeline.py
-----------
The orchestrator. Two entry points are meant to be called from the notebook:

  run_full_pipeline(...)         — one event, full 5-dashboard deep dive (~30 charts)
  run_multi_site_comparison(...) — several real locations, side-by-side verdicts

Every step prints a banner so the sequence doubles as a live-demo voiceover
script.
"""
import time

from utils import PipelineConfig, banner
from image_fetcher import fetch_satellite_image
from image_processing import build_processing_gallery
from context_fetcher import get_full_context
from geo_lookup import reverse_geocode
import vision_model
import tabular_model
import temporal_model
import fusion_engine
import evaluation
import dashboard_visuals
import dashboards
import map_view


def run_full_pipeline(lat: float, lon: float, gemini_model=None, label: str = "", show_charts: bool = True):
    cfg = PipelineConfig()

    banner(f"STEP 0 — Reverse geocoding  {('(' + label + ')') if label else ''}")
    place = reverse_geocode(lat, lon)
    print(f"  {place['display_name']}")

    banner("STEP 1 — Fetching satellite image")
    image = fetch_satellite_image(lat, lon, cfg)

    banner("STEP 2 — Running image processing pipeline (grayscale, edges, thermal views, segmentation)")
    gallery = build_processing_gallery(image)

    banner("STEP 3 — Fetching real-world context (OpenStreetMap + NASA POWER)")
    context = get_full_context(lat, lon, cfg)
    print(context)

    banner("STEP 4 — Branch A: Vision model (placeholder for trained CNN)")
    vision_result = vision_model.run_vision_branch(image, lat, lon, gemini_model=gemini_model)
    print(vision_result)
    activations = vision_model.fake_activation_chart_values(lat, lon)

    banner("STEP 5 — Branch B: Context / tabular model (Random Forest on synthetic placeholder data)")
    context_clf = tabular_model.train_prototype_model()
    context_result = tabular_model.run_context_branch(context, context_clf)
    print(context_result)

    banner("STEP 6 — Branch C: Temporal / persistence model (simulated history, real anomaly math)")
    temporal_result = temporal_model.run_temporal_branch(lat, lon)
    print({k: v for k, v in temporal_result.items() if k != "history"})

    banner("STEP 7 — Fusion engine: combining all three branches")
    fusion_result = fusion_engine.fuse_branches(vision_result, context_result, temporal_result)
    print(fusion_result)

    banner("STEP 8 — Model evaluation (real held-out metrics on the tabular model)")
    eval_result = evaluation.run_holdout_evaluation()
    boundary_result = evaluation.run_2d_decision_boundary()
    print(eval_result["metrics"])

    banner("STEP 9 — Stability check (Monte Carlo re-run of the real fusion formula)")
    stability_result = fusion_engine.run_stability_check(vision_result, context_result, temporal_result)
    print(f"  confidence: median {sorted(stability_result['confidence_samples'])[len(stability_result['confidence_samples'])//2]}, "
          f"risk: median {sorted(stability_result['risk_samples'])[len(stability_result['risk_samples'])//2]} "
          f"over {stability_result['n_runs']} re-runs")

    if show_charts:
        banner("STEP 10 — Rendering all 5 dashboards (~30 charts)")
        dashboard_visuals.plot_image_gallery(gallery)
        dashboards.dashboard_1_image_intelligence(gallery, label=label)
        dashboards.dashboard_2_context_branch(vision_result, context, context_result, fusion_result, temporal_result, label=label)
        dashboards.dashboard_3_vision_deepdive(gallery, vision_result, activations, label=label)
        dashboards.dashboard_4_model_evaluation(eval_result, boundary_result, label=label)
        dashboards.dashboard_5_fusion_risk(vision_result, context_result, temporal_result, fusion_result, context, stability_result, label=label)

    dashboard_visuals.print_final_verdict_card(fusion_result, context)

    event_map = map_view.build_event_map(lat, lon, context, label=label)

    return {
        "place": place,
        "image": image,
        "gallery": gallery,
        "context": context,
        "vision_result": vision_result,
        "context_result": context_result,
        "temporal_result": temporal_result,
        "fusion_result": fusion_result,
        "eval_result": eval_result,
        "boundary_result": boundary_result,
        "stability_result": stability_result,
        "event_map": event_map,
    }


def run_multi_site_comparison(locations: dict, gemini_model=None, show_charts: bool = True):
    """Runs the lightweight branch of the pipeline (no full dashboards) over
    every named real-world site in `locations` and shows one comparison
    chart — the "does this actually tell industrial from natural apart"
    proof, across sites the team did not hand-pick per-answer."""
    cfg = PipelineConfig()
    context_clf = tabular_model.train_prototype_model()
    results = []

    for i, (key, site) in enumerate(locations.items()):
        if i > 0:
            time.sleep(2)  # be polite to the free Overpass/Nominatim endpoints across repeated calls
        banner(f"COMPARISON SITE — {site['label']}")
        print(f"  Why this site: {site['story']}")
        lat, lon = site["lat"], site["lon"]

        image = fetch_satellite_image(lat, lon, cfg)
        context = get_full_context(lat, lon, cfg)
        vision_result = vision_model.run_vision_branch(image, lat, lon, gemini_model=gemini_model)
        context_result = tabular_model.run_context_branch(context, context_clf)
        temporal_result = temporal_model.run_temporal_branch(lat, lon)
        fusion_result = fusion_engine.fuse_branches(vision_result, context_result, temporal_result)
        multiclass_dist = fusion_engine.derive_multiclass_distribution(fusion_result, temporal_result, context)
        top_class = max(multiclass_dist, key=multiclass_dist.get)

        print(f"  Predicted: {fusion_result['classification']}  |  Top multi-class guess: {top_class}  |  "
              f"Risk: {fusion_result['risk_score']}  |  Expected: {site['expected_class']}")

        results.append({
            "key": key,
            "label": site["label"],
            "expected_class": site["expected_class"],
            "classification": fusion_result["classification"],
            "top_multiclass_guess": top_class,
            "risk_score": fusion_result["risk_score"],
            "fusion_result": fusion_result,
            "context": context,
        })

    if show_charts:
        banner("Multi-site comparison chart")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.5))
        dashboard_visuals.draw_multi_site_comparison(ax, results)
        plt.tight_layout()
        plt.show()

    multi_map = map_view.build_multi_site_map(locations)
    return {"results": results, "multi_map": multi_map}
