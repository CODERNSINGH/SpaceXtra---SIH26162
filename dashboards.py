"""
dashboards.py
-------------
Composes the "draw_*" chart primitives from dashboard_visuals.py into five
named, 6-chart dashboards — the full presentation deck for one event:

  Dashboard 1 — Image Intelligence        (the fetched image, decoded 6 ways)
  Dashboard 2 — Context & Branch Analysis (what's around it, how the model votes)
  Dashboard 3 — Vision Branch Deep-Dive   (what the CNN placeholder "saw")
  Dashboard 4 — Model Evaluation          (real held-out metrics on the tabular model)
  Dashboard 5 — Final Fusion & Risk       (the combined verdict)

Every chart here is either drawn from real upstream numbers (fetched
context, real scikit-learn metrics, real fusion math) or clearly labeled as
a heuristic/illustrative stand-in — see each module's own docstring for the
exact real/simulated boundary. This file only lays charts out; it computes
nothing new except the tiny amount of glue (e.g. the correlation matrix)
needed to feed a chart.
"""
import matplotlib.pyplot as plt

import dashboard_visuals as dv
from tabular_model import _generate_synthetic_training_data, FEATURE_NAMES
from fusion_engine import derive_multiclass_distribution


def dashboard_1_image_intelligence(gallery: dict, label: str = ""):
    keys = [
        "1. Original satellite image",
        "7. Simulated thermal (inferno)",
        "6. Edge detection (Sobel-style)",
        "11. K-means segmentation (k=4)",
        "10. Simulated hotspot mask (B/W)",
        "12. Simulated vegetation index",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"DASHBOARD 1 — Image Intelligence (6 views){'  — ' + label if label else ''}",
                 fontsize=15, weight="bold")
    for ax, key in zip(axes.flat, keys):
        img = gallery.get(key)
        if img is not None:
            dv.draw_image_panel(ax, img, key)
        else:
            ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig


def dashboard_2_context_branch(vision: dict, context: dict, context_result: dict,
                                fusion_result: dict, temporal_result: dict, label: str = ""):
    train_df = _generate_synthetic_training_data()
    corr_df = train_df[FEATURE_NAMES].corr()
    multiclass_dist = derive_multiclass_distribution(fusion_result, temporal_result, context)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"DASHBOARD 2 — Context & Branch Analysis (6 charts){'  — ' + label if label else ''}",
                 fontsize=15, weight="bold")

    dv.draw_branch_vote_bars(
        axes[0, 0],
        ["Vision\n(industrial %)", "Context\n(industrial %)"],
        [vision.get("industrial_probability_percent", 0), context_result.get("industrial_probability_percent", 0)],
        colors=["#1f4c8a", "#1f7a3d"],
        title="1. Branch Vote Comparison",
    )
    dv.draw_feature_importance(axes[0, 1], context_result.get("feature_importances", {}),
                                title="2. Feature Importance\n(Random Forest, synthetic training data)")
    dv.draw_weather_context_bars(axes[0, 2], context, title="3. Live Weather/Soil Context\n(normalized to 0-100 for comparison)")
    dv.draw_correlation_heatmap(axes[1, 0], corr_df, title="4. Feature Correlation Heatmap\n(synthetic training data)")
    dv.draw_facility_proximity(axes[1, 1], context.get("facility_distance_m"), title_prefix="5. Facility Proximity")
    dv.draw_multiclass_bars(axes[1, 2], multiclass_dist,
                             title="6. Multi-Class Breakdown\n(HEURISTIC, not a trained multi-class model)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig


def dashboard_3_vision_deepdive(gallery: dict, vision_result: dict, activations, label: str = ""):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"DASHBOARD 3 — Vision Branch Deep-Dive (6 charts){'  — ' + label if label else ''}",
                 fontsize=15, weight="bold")

    ax1 = fig.add_subplot(2, 3, 1)
    dv.draw_image_panel(ax1, gallery["1. Original satellite image"], "1. Vision Branch Input (satellite image)")

    ax2 = fig.add_subplot(2, 3, 2)
    dv.draw_image_panel(ax2, gallery["8. Simulated thermal (jet)"], "2. Simulated Thermal View (jet)")

    ax3 = fig.add_subplot(2, 3, 3)
    dv.draw_image_panel(ax3, gallery["12. Simulated vegetation index"], "3. Simulated Vegetation Index")

    ax4 = fig.add_subplot(2, 3, 4)
    dv.draw_verdict_donut(ax4, vision_result, title="4. Vision Branch Verdict\n(Gemini placeholder for trained CNN)")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.bar(range(len(activations)), activations, color="#6a1f8a")
    ax5.set_title("5. Illustrative Activation Map\n(DECORATIVE — not real network weights)", fontsize=10, weight="bold", color="#8a1f1f")
    ax5.set_xlabel("Filter index (illustrative)")
    ax5.set_ylabel("Activation (illustrative)")

    ax6 = fig.add_subplot(2, 3, 6)
    notes = vision_result.get("visual_notes", [])
    source = vision_result.get("source", "unknown")
    dv.draw_text_panel(ax6, notes + [f"[source: {source}]"], title="6. Vision Model Notes")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig


def dashboard_4_model_evaluation(eval_result: dict, boundary_result: dict, label: str = ""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("DASHBOARD 4 — Model Evaluation (REAL metrics, computed on a held-out split of "
                 f"SYNTHETIC placeholder data — 6 charts){'  — ' + label if label else ''}",
                 fontsize=13.5, weight="bold")

    dv.draw_confusion_matrix(axes[0, 0], eval_result["confusion_matrix"], title="1. Confusion Matrix (real, held-out split)")
    dv.draw_roc_curve(axes[0, 1], eval_result["fpr"], eval_result["tpr"], eval_result["roc_auc"], title="2. ROC Curve (real)")
    dv.draw_pr_curve(axes[0, 2], eval_result["pr_precision"], eval_result["pr_recall"], title="3. Precision-Recall Curve (real)")
    dv.draw_decision_boundary(axes[1, 0], boundary_result, title="4. Decision Boundary (real, 2 features)")
    dv.draw_confidence_histogram(axes[1, 1], eval_result["y_test"], eval_result["y_proba"],
                                  title="5. Prediction Confidence Split by True Label")
    dv.draw_metrics_summary(axes[1, 2], eval_result["metrics"], title="6. Held-Out Metrics Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return fig


def dashboard_5_fusion_risk(vision: dict, context_result: dict, temporal_result: dict,
                             fusion_result: dict, context: dict, stability_result: dict, label: str = ""):
    multiclass_dist = derive_multiclass_distribution(fusion_result, temporal_result, context)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"DASHBOARD 5 — Final Fusion & Risk (6 charts){'  — ' + label if label else ''}",
                 fontsize=15, weight="bold")

    ax1 = fig.add_subplot(2, 3, 1)
    votes = fusion_result["branch_votes"]
    dv.draw_branch_vote_bars(ax1, list(votes.keys()), list(votes.values()),
                              colors=["#1f4c8a", "#1f7a3d", "#8a1f6a"],
                              title="1. Branch Votes Feeding the Fusion Engine")
    ax1.tick_params(axis="x", labelsize=7.5)

    ax2 = fig.add_subplot(2, 3, 2)
    dv.draw_classification_pie(ax2, fusion_result, title="2. Fused Classification Split")

    ax3 = fig.add_subplot(2, 3, 3)
    dv.draw_risk_gauge(ax3, fusion_result["risk_score"], title="3. Final Risk Gauge")

    ax4 = fig.add_subplot(2, 3, 4, projection="polar")
    dv.draw_radar(
        ax4,
        ["Context\nind. %", "Temporal\nanomaly %", "Vision\nind. %", "Confidence %", "Risk"],
        [votes["context_industrial_pct"], votes["temporal_anomaly_pct"], votes["vision_industrial_pct"],
         fusion_result["confidence_percent"], fusion_result["risk_score"]],
        title="4. All-Branch Radar",
    )

    ax5 = fig.add_subplot(2, 3, 5)
    dv.draw_multiclass_bars(ax5, multiclass_dist, highlight_top=True,
                             title="5. Final Multi-Class Verdict\n(heuristic; top class highlighted)")

    ax6 = fig.add_subplot(2, 3, 6)
    dv.draw_stability_boxplot(ax6, stability_result)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig
