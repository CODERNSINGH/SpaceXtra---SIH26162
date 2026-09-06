"""
dashboard_visuals.py
---------------------
REAL module: every chart here is drawn from real numbers produced upstream
(fetched context, computed anomaly score, fusion output) or from clearly
labeled illustrative/decorative values (CNN activation bars). This is the
"lots of charts" file — each function below is one chart.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fusion_engine import CLASS_LABELS


def plot_image_gallery(gallery: dict, cols: int = 5, figsize_scale: float = 2.6):
    n = len(gallery)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, img) in zip(axes.flat, gallery.items()):
        ax.imshow(img, cmap="gray" if img.mode == "L" else None)
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    plt.suptitle("Image Processing Pipeline — Full Gallery", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_branch_confidence_bars(vision, context, temporal):
    labels = ["Vision Branch\n(industrial %)", "Context Branch\n(industrial %)", "Temporal Branch\n(anomaly %)"]
    values = [
        vision.get("industrial_probability_percent", 0),
        context.get("industrial_probability_percent", 0),
        temporal.get("anomaly_score_percent", 0),
    ]
    colors = ["#1f4c8a", "#1f7a3d", "#8a1f6a"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v}%", ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    ax.set_title("Branch-by-Branch Vote", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_classification_pie(fusion_result: dict):
    fig, ax = plt.subplots(figsize=(5, 5))
    values = [fusion_result["fused_industrial_probability_percent"], fusion_result["fused_natural_probability_percent"]]
    labels = [f"Industrial-like\n{values[0]}%", f"Natural-like\n{values[1]}%"]
    ax.pie(values, labels=labels, colors=["#c62828", "#2e7d32"], autopct=None,
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Fused Classification Split", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_risk_gauge(risk_score: float):
    fig, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={"aspect": "equal"})
    theta = np.linspace(np.pi, 0, 100)
    for start, end, color in [(0, 30, "#2e7d32"), (30, 60, "#f9a825"), (60, 85, "#ef6c00"), (85, 100, "#c62828")]:
        seg_theta = np.linspace(np.pi * (1 - start / 100), np.pi * (1 - end / 100), 20)
        x = np.cos(seg_theta)
        y = np.sin(seg_theta)
        ax.plot(x, y, color=color, linewidth=14, solid_capstyle="butt")

    needle_theta = np.pi * (1 - risk_score / 100)
    ax.plot([0, 0.8 * np.cos(needle_theta)], [0, 0.8 * np.sin(needle_theta)], color="black", linewidth=3)
    ax.scatter([0], [0], color="black", s=40, zorder=5)
    ax.text(0, -0.3, f"RISK: {risk_score}/100", ha="center", fontsize=14, weight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    ax.set_title("Risk Score Gauge", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_temporal_history(temporal_result: dict):
    history = temporal_result["history"]
    days = list(range(-len(history) + 1, 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days, history, marker="o", color="#1f4c8a", linewidth=1.5, markersize=4)
    ax.axhline(temporal_result["baseline_mean_frp"], color="#888888", linestyle="--", label="Baseline mean")
    ax.scatter([days[-1]], [history[-1]], color="#c62828", s=90, zorder=5, label="Latest reading")
    ax.set_xlabel("Days relative to today")
    ax.set_ylabel("Simulated FRP (fire radiative power)")
    ax.set_title(f"30-Day Thermal History (simulated) — z-score {temporal_result['z_score']}", fontsize=12, weight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(context_result: dict):
    importances = context_result.get("feature_importances", {})
    if not importances:
        return
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, values, color="#1f7a3d")
    ax.set_xlabel("Importance")
    ax.set_title("Context Branch — Feature Importance\n(Random Forest trained on synthetic placeholder data)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_fake_activation_bars(activation_values, lat, lon):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(len(activation_values)), activation_values, color="#6a1f8a")
    ax.set_title("Illustrative Vision-Branch Activation Map (DECORATIVE — not real network weights)",
                 fontsize=10.5, weight="bold", color="#8a1f1f")
    ax.set_xlabel("Filter index (illustrative)")
    ax.set_ylabel("Activation (illustrative)")
    plt.tight_layout()
    plt.show()


def plot_radar_summary(vision, context, temporal, fusion_result):
    categories = ["Vision\nindustrial %", "Context\nindustrial %", "Temporal\nanomaly %", "Fusion\nconfidence %", "Risk\nscore"]
    values = [
        vision.get("industrial_probability_percent", 0),
        context.get("industrial_probability_percent", 0),
        temporal.get("anomaly_score_percent", 0),
        fusion_result.get("confidence_percent", 0),
        fusion_result.get("risk_score", 0),
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, color="#1f4c8a", linewidth=2)
    ax.fill(angles, values, color="#1f4c8a", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("All-Branch Summary Radar", fontsize=13, weight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def normalize_weather_context(context: dict) -> dict:
    """Normalizes real, differently-scaled weather/soil readings onto a
    common 0-100 axis purely so they can share one bar chart. The caps
    (50 C, 20 mm/day, fractional 0-1 soil moisture) are reasonable, labeled
    assumptions for the Indian climate context, not measured constants."""
    temp = context.get("avg_temp_c")
    humidity = context.get("avg_humidity_pct")
    rainfall = context.get("avg_rainfall_mm")
    soil = context.get("avg_root_zone_soil_moisture")
    return {
        "Temp (C)": min(100, max(0, (temp / 50) * 100)) if temp is not None else 0,
        "Humidity (%)": min(100, max(0, humidity)) if humidity is not None else 0,
        "Rainfall (mm)": min(100, max(0, (rainfall / 20) * 100)) if rainfall is not None else 0,
        "Soil moisture": min(100, max(0, soil * 100)) if soil is not None else 0,
    }


# ---------------------------------------------------------------------------
# "draw_*" primitives — each takes a matplotlib Axes and draws ONE chart into
# it, so the composite "Dashboard N" figures in dashboards.py can lay six of
# these out on a single 2x3 grid, exactly like the reference screenshots.
# ---------------------------------------------------------------------------

def draw_branch_vote_bars(ax, labels, values, colors, ylabel="Percent", title="Branch Vote Comparison"):
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v}%", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, weight="bold")


def draw_feature_importance(ax, importances: dict, title="Feature Importance\n(Random Forest, synthetic training data)"):
    if not importances:
        ax.axis("off")
        return
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    ax.barh(names, values, color="#1f7a3d")
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_weather_context_bars(ax, context: dict, title="Live Weather/Soil Context\n(normalized to 0-100 for comparison)"):
    normed = normalize_weather_context(context)
    ax.bar(list(normed.keys()), list(normed.values()), color="#00838f")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelrotation=20)


def draw_correlation_heatmap(ax, corr_df, title="Feature Correlation Heatmap\n(synthetic training data)"):
    cols = list(corr_df.columns)
    mat = corr_df.values
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(mat[i, j]) > 0.5 else "black")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_facility_proximity(ax, distance_m, close_m=500, medium_m=1500, far_m=3000,
                             title_prefix="Facility Proximity"):
    if distance_m is None:
        ax.barh(["Distance to\nnearest facility"], [far_m * 1.1], color="#e0e0e0", height=0.5,
                 hatch="//", edgecolor="#9e9e9e")
        ax.axvline(close_m, color="#c62828", linestyle="--", linewidth=1.5)
        ax.axvline(medium_m, color="#f57f17", linestyle="--", linewidth=1.5)
        ax.axvline(far_m, color="#2e7d32", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, far_m * 1.15)
        ax.set_title(f"{title_prefix} — no facility found nearby\n(OSM query returned nothing in range)",
                     fontsize=10.5, weight="bold")
        return
    ax.barh(["Distance to\nnearest facility"], [distance_m], color="#f9a825", height=0.5)
    ax.axvline(close_m, color="#c62828", linestyle="--", linewidth=1.5)
    ax.axvline(medium_m, color="#f57f17", linestyle="--", linewidth=1.5)
    ax.axvline(far_m, color="#2e7d32", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, max(far_m * 1.15, distance_m * 1.15))
    ax.set_title(f"{title_prefix} — {round(distance_m)} m\n(dashed lines: close / medium / far zones)",
                 fontsize=10.5, weight="bold")


def draw_multiclass_bars(ax, distribution: dict, highlight_top: bool = False,
                          title="Multi-Class Breakdown\n(HEURISTIC, not a trained multi-class model)"):
    names = [c for c in CLASS_LABELS if c in distribution] or list(distribution.keys())
    values = [distribution[n] for n in names]
    colors = ["#6a1f8a"] * len(names)
    if highlight_top and values:
        top_i = int(np.argmax(values))
        colors = ["#9e9e9e"] * len(names)
        colors[top_i] = "#c62828"
    ax.bar([n.replace("_", "\n") for n in names], values, color=colors)
    ax.set_ylabel("Percent")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(title, fontsize=9.5, weight="bold")


def draw_confusion_matrix(ax, cm, class_names=("Natural", "Industrial"), title="Confusion Matrix (real, held-out split)"):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, weight="bold",
                     color="white" if cm[i, j] > vmax / 2 else "black")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_roc_curve(ax, fpr, tpr, roc_auc, title="ROC Curve (real)"):
    ax.plot(fpr, tpr, color="#1f4c8a", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_pr_curve(ax, precision, recall, title="Precision-Recall Curve (real)"):
    ax.plot(recall, precision, color="#2e7d32", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_decision_boundary(ax, boundary_result, title="Decision Boundary (real, 2 features)"):
    xx, yy, zz = boundary_result["xx"], boundary_result["yy"], boundary_result["zz"]
    X, y = boundary_result["X"], boundary_result["y"]
    ax.contourf(xx, yy, zz, levels=20, cmap="RdYlGn_r", alpha=0.75)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1b5e20", s=14, edgecolor="white", linewidth=0.3, label="Natural")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#b71c1c", s=14, edgecolor="white", linewidth=0.3, label="Industrial")
    ax.set_xlabel(boundary_result["feat_x"])
    ax.set_ylabel(boundary_result["feat_y"])
    ax.set_title(title, fontsize=11, weight="bold")


def draw_confidence_histogram(ax, y_test, y_proba, title="Prediction Confidence Split by True Label"):
    y_test = np.asarray(y_test)
    natural_proba = y_proba[y_test == 0]
    industrial_proba = y_proba[y_test == 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(natural_proba, bins=bins, alpha=0.6, color="#2e7d32", label="Actual: Natural")
    ax.hist(industrial_proba, bins=bins, alpha=0.6, color="#c62828", label="Actual: Industrial")
    ax.set_xlabel("Predicted P(industrial)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(title, fontsize=11, weight="bold")


def draw_metrics_summary(ax, metrics: dict, title="Held-Out Metrics Summary"):
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    bars = ax.bar(names, values, color="#37474f")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=15)
    ax.set_title(title, fontsize=11, weight="bold")


def draw_classification_pie(ax, fusion_result: dict, title="Fused Classification Split"):
    values = [fusion_result["fused_industrial_probability_percent"], fusion_result["fused_natural_probability_percent"]]
    labels = [f"Industrial-like\n{values[0]}%", f"Natural-like\n{values[1]}%"]
    ax.pie(values, labels=labels, colors=["#c62828", "#2e7d32"], startangle=90,
           explode=(0.05, 0), wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(title, fontsize=11, weight="bold")


def draw_risk_gauge(ax, risk_score: float, title="Final Risk Gauge"):
    theta = np.linspace(np.pi, 0, 100)
    for start, end, color in [(0, 30, "#2e7d32"), (30, 60, "#f9a825"), (60, 85, "#ef6c00"), (85, 100, "#c62828")]:
        seg_theta = np.linspace(np.pi * (1 - start / 100), np.pi * (1 - end / 100), 20)
        ax.plot(np.cos(seg_theta), np.sin(seg_theta), color=color, linewidth=14, solid_capstyle="butt")
    needle_theta = np.pi * (1 - risk_score / 100)
    ax.plot([0, 0.8 * np.cos(needle_theta)], [0, 0.8 * np.sin(needle_theta)], color="black", linewidth=3)
    ax.scatter([0], [0], color="black", s=40, zorder=5)
    ax.text(0, -0.3, f"RISK: {risk_score}/100", ha="center", fontsize=13, weight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold")


def draw_radar(ax_polar, categories, values, title="All-Branch Radar"):
    vals = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax_polar.plot(angles, vals, color="#1f4c8a", linewidth=2)
    ax_polar.fill(angles, vals, color="#1f4c8a", alpha=0.25)
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories, fontsize=8)
    ax_polar.set_ylim(0, 100)
    ax_polar.set_title(title, fontsize=11, weight="bold", pad=18)


def draw_stability_boxplot(ax, stability_result: dict,
                            title="Stability Check\n(+/-5% input noise, 200 re-runs — real)"):
    data = [stability_result["confidence_samples"], stability_result["risk_samples"]]
    try:
        bp = ax.boxplot(data, tick_labels=["Confidence", "Risk Score"], patch_artist=True)
    except TypeError:  # matplotlib < 3.9 uses the old `labels` kwarg
        bp = ax.boxplot(data, labels=["Confidence", "Risk Score"], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfd8dc")
    for median in bp["medians"]:
        median.set_color("#ef6c00")
    ax.set_title(title.replace("+/-", "±"), fontsize=10.5, weight="bold")


def draw_image_panel(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap or ("gray" if getattr(img, "mode", None) == "L" else None))
    ax.set_title(title, fontsize=10, weight="bold")
    ax.axis("off")


def draw_verdict_donut(ax, vision_result: dict, title="Vision Branch Verdict"):
    industrial = vision_result.get("industrial_probability_percent", 50)
    natural = 100 - industrial
    ax.pie([industrial, natural], labels=[f"Industrial\n{industrial}%", f"Natural\n{natural}%"],
           colors=["#c62828", "#2e7d32"], startangle=90,
           wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2})
    ax.set_title(title, fontsize=11, weight="bold")


def draw_text_panel(ax, lines, title="Notes"):
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", loc="left")
    body = "\n".join(f"• {line}" for line in lines) if lines else "(no notes returned)"
    ax.text(0.02, 0.95, body, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, wrap=True, family="monospace")


def draw_multi_site_comparison(ax, results: list, title="Multi-Site Risk Comparison\n(green = top multi-class guess matched the site's documented real-world class)"):
    labels = [r["label"] for r in results]
    risks = [r["risk_score"] for r in results]
    colors = ["#2e7d32" if r.get("top_multiclass_guess") == r.get("expected_class") else "#c62828" for r in results]
    bars = ax.bar(labels, risks, color=colors)
    for bar, v in zip(bars, risks):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}", ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("Risk score (0-100)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.set_title(title, fontsize=11, weight="bold")


def print_final_verdict_card(fusion_result: dict, context: dict):
    print("\n" + "#" * 70)
    print("#  FINAL VERDICT")
    print("#" * 70)
    print(f"  Classification : {fusion_result['classification'].replace('_', ' ').title()}")
    print(f"  Confidence     : {fusion_result['confidence_percent']}%")
    print(f"  Risk Score     : {fusion_result['risk_score']}/100")
    print(f"  Nearest facility: {context.get('nearest_industrial_facility')} "
          f"({context.get('facility_distance_m')} m away)")
    print("#" * 70 + "\n")
