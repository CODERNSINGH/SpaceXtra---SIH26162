"""
Server-rendered matplotlib charts for the analysis dashboard.

Plain, unstyled matplotlib (Agg backend) on purpose — gridlines on, no
smoothing/animation, small captions — to match the old-school
operational-science-tool aesthetic rather than a flashy JS charting library.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.branches import CATEGORIES

plt.rcParams.update({
    "font.size": 8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.color": "#bbbbbb",
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def branch_votes_chart(vision: dict, context: dict, temporal: dict) -> str:
    branches = ["Vision (A)", "Context (B)", "Temporal (C)"]
    results = [vision, context, temporal]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(CATEGORIES))
    width = 0.26
    colors = ["#2e6da4", "#5cb85c", "#d9822b"]
    for i, (name, r) in enumerate(zip(branches, results)):
        vals = [r["class_probabilities"].get(c, 0.0) for c in CATEGORIES]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[i], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CATEGORIES], rotation=0, fontsize=6)
    ax.set_ylabel("branch probability")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Per-branch class votes", fontsize=9)
    return _fig_to_base64(fig)


def risk_gauge_chart(risk_score: int) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 1.3))
    ax.barh([0], [100], color="#eeeeee", edgecolor="#444", height=0.5)
    zones = [(0, 33, "#8fbf7a"), (33, 66, "#e8c14a"), (66, 100, "#c0504d")]
    for start, end, color in zones:
        ax.barh([0], [end - start], left=start, color=color, alpha=0.55, height=0.5)
    ax.barh([0], [risk_score], color="#222222", height=0.18)
    ax.axvline(risk_score, color="black", linewidth=1.5)
    ax.text(risk_score, 0.42, f"{risk_score}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("risk score (0-100)")
    ax.grid(False)
    return _fig_to_base64(fig)


def feature_importance_chart(model, feature_columns: list[str]) -> str:
    importances = model.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.barh(np.array(feature_columns)[order], importances[order], color="#5b7fa6", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("importance")
    ax.set_title("Random Forest feature importance", fontsize=9)
    return _fig_to_base64(fig)


def correlation_heatmap_chart(training_df, feature_columns: list[str]) -> str:
    corr = training_df[feature_columns].corr().values
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_columns)))
    ax.set_yticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_columns, fontsize=6)
    for i in range(len(feature_columns)):
        for j in range(len(feature_columns)):
            ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Training feature correlation", fontsize=9)
    ax.grid(False)
    return _fig_to_base64(fig)


def history_line_chart(history: dict) -> str:
    series = history["series"]
    dates = [p["date"][5:] for p in series]  # MM-DD
    values = [p["frp"] for p in series]
    mean = history["mean_baseline"]
    std = history["std_baseline"]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(dates, values, color="#c0504d", linewidth=1.2, marker="o", markersize=2, label="daily FRP")
    ax.axhline(mean, color="#2e6da4", linewidth=1, linestyle="--", label="baseline mean")
    ax.fill_between(range(len(dates)), mean - std, mean + std, color="#2e6da4", alpha=0.15, label="±1 std baseline")
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(dates[::step], rotation=45, fontsize=6)
    ax.set_ylabel("FRP (MW)")
    tag = "SIMULATED" if history["simulated"] else "REAL ARCHIVE"
    ax.set_title(f"{history['total_days']}-day thermal history [{tag}]", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    return _fig_to_base64(fig)


def anomaly_bar_chart(history: dict) -> str:
    series = history["series"]
    dates = [p["date"][5:] for p in series]
    z = history["z_scores"]
    colors = ["#c0504d" if abs(v) >= 2 else "#8fbf7a" if abs(v) < 1 else "#e8c14a" for v in z]

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.bar(range(len(dates)), z, color=colors, edgecolor="black", linewidth=0.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(2, color="#c0504d", linewidth=0.6, linestyle="--")
    ax.axhline(-2, color="#c0504d", linewidth=0.6, linestyle="--")
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(dates[::step], rotation=45, fontsize=6)
    ax.set_ylabel("z-score")
    ax.set_title("Per-day anomaly z-score vs. site baseline", fontsize=8)
    return _fig_to_base64(fig)


def confusion_matrix_chart(cm: np.ndarray, labels: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace("_", "\n") for l in labels]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    vmax = cm.max() or 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color=color)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (held-out test set)", fontsize=8)
    ax.grid(False)
    return _fig_to_base64(fig)


def roc_curve_chart(fpr, tpr, auc: float) -> str:
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot(fpr, tpr, color="#2e6da4", linewidth=1.3, label=f"micro-avg ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (one-vs-rest, micro-avg)", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    return _fig_to_base64(fig)


def pr_curve_chart(precision, recall) -> str:
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot(recall, precision, color="#d9822b", linewidth=1.3)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve (one-vs-rest, micro-avg)", fontsize=8)
    return _fig_to_base64(fig)
