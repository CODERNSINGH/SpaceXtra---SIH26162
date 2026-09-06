"""
Branch C — Temporal / Persistence.

Builds a mini daily time-series of FRP at a location (real archived FIRMS
detections pulled from the local store if enough depth exists, otherwise a
deterministic simulated history — same coordinate always reproduces the same
simulated series). Computes a real anomaly score: rolling mean/std of the
site's own history and a z-score of the latest reading against it. This math
is identical regardless of whether the series is real or simulated — only
the data source differs, and that is labeled explicitly in the output.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

from app.branches import CATEGORIES
from app.database import get_history_for_point

HISTORY_DAYS = 30
MIN_REAL_DAYS = 5  # below this many distinct real observation-days, simulate instead


def _simulate_history(lat: float, lon: float, days: int = HISTORY_DAYS) -> list[dict]:
    """Deterministic pseudo-history seeded from the coordinate itself."""
    seed = int(hashlib.sha1(f"{lat:.4f},{lon:.4f}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    # A per-site "character" derived from the seed decides whether this looks
    # like a persistent source (steady low-moderate FRP most days) or an
    # occasional-spike site (near zero most days, rare spikes).
    persistence_bias = rng.random()
    today = datetime.now(timezone.utc).date()
    series = []
    baseline = rng.uniform(1, 6) if persistence_bias > 0.4 else 0.3
    for i in range(days, 0, -1):
        date = today - timedelta(days=i)
        if persistence_bias > 0.4:
            frp = max(0.0, baseline + rng.normal(0, baseline * 0.35))
        else:
            frp = max(0.0, rng.normal(0.2, 0.3)) if rng.random() > 0.12 else rng.uniform(8, 30)
        series.append({"date": date.isoformat(), "frp": round(float(frp), 2)})
    # Ensure "today" (latest) reflects the actual current detection strongly.
    return series


def _real_history(lat: float, lon: float) -> list[dict] | None:
    rows = get_history_for_point(lat, lon)
    by_date: dict[str, float] = {}
    for r in rows:
        d = r["acq_date"]
        frp = r.get("frp") or 0.0
        by_date[d] = max(by_date.get(d, 0.0), frp)
    if len(by_date) < MIN_REAL_DAYS:
        return None
    dates = sorted(by_date.keys())[-HISTORY_DAYS:]
    return [{"date": d, "frp": by_date[d]} for d in dates]


def get_history(lat: float, lon: float, latest_frp: float | None = None) -> dict:
    real = _real_history(lat, lon)
    if real is not None:
        series = real
        simulated = False
    else:
        series = _simulate_history(lat, lon)
        simulated = True

    if latest_frp is not None and series:
        series[-1] = {"date": series[-1]["date"], "frp": round(float(latest_frp), 2)}

    values = np.array([p["frp"] for p in series], dtype=float)
    baseline_vals = values[:-1] if len(values) > 1 else values
    mean = float(np.mean(baseline_vals)) if len(baseline_vals) else 0.0
    std = float(np.std(baseline_vals)) if len(baseline_vals) else 0.0
    latest = float(values[-1]) if len(values) else 0.0
    # A floor on the rolling std avoids explosive/meaningless z-scores when a
    # site's early history is near-constant (std ~ 0).
    STD_FLOOR = 0.5
    z_scores = []
    for i, v in enumerate(values):
        if i < 2:
            z_scores.append(0.0)
            continue
        hist = values[:i]
        m = float(np.mean(hist))
        s = max(float(np.std(hist)), STD_FLOOR)
        z_scores.append(round(float((v - m) / s), 2))

    z_latest = (latest - mean) / max(std, STD_FLOOR)
    detected_days = int(np.sum(values > 0.5))
    persistence_ratio = detected_days / len(values) if len(values) else 0.0

    return {
        "series": series,
        "simulated": simulated,
        "mean_baseline": round(mean, 2),
        "std_baseline": round(std, 2),
        "latest": round(latest, 2),
        "z_latest": round(float(z_latest), 2),
        "z_scores": z_scores,
        "persistence_ratio": round(persistence_ratio, 2),
        "detected_days": detected_days,
        "total_days": len(values),
    }


def classify_temporal(history: dict) -> dict:
    """Heuristic (transparent, non-ML) mapping from persistence/anomaly stats
    to a probability lean over the 7 output categories. Documented rules:
      - high persistence_ratio -> leans persistent_industrial_source / gas_flare
      - low persistence + high z-score spike -> leans industrial_fire / wildfire / agricultural_burn
      - otherwise -> spread with a moderate 'unknown' weight
    """
    p = history["persistence_ratio"]
    z = history["z_latest"]

    dist = {c: 0.02 for c in CATEGORIES}  # small floor so nothing is exactly zero
    if p >= 0.6:
        dist["persistent_industrial_source"] += 0.42
        dist["gas_flare"] += 0.25
        dist["industrial_fire"] += 0.12
        dist["mining_activity"] += 0.08
    elif z >= 2.5:
        dist["industrial_fire"] += 0.28
        dist["wildfire"] += 0.24
        dist["agricultural_burn"] += 0.20
        dist["unknown"] += 0.05
    elif z >= 1.0:
        dist["industrial_fire"] += 0.15
        dist["agricultural_burn"] += 0.15
        dist["wildfire"] += 0.10
        dist["unknown"] += 0.10
    else:
        dist["unknown"] += 0.20
        dist["persistent_industrial_source"] += 0.10
        dist["industrial_fire"] += 0.08

    total = sum(dist.values())
    dist = {c: v / total for c, v in dist.items()}
    top_class = max(dist, key=dist.get)

    evidence = []
    tag = "simulated" if history["simulated"] else "real archived"
    evidence.append(f"{tag} {history['total_days']}-day history: detected on "
                     f"{history['detected_days']}/{history['total_days']} days "
                     f"(persistence ratio {p:.2f})")
    if history["std_baseline"] > 0:
        evidence.append(f"latest FRP is {z:.1f} standard deviations from this site's own "
                         f"{history['total_days']}-day baseline (baseline mean {history['mean_baseline']:.1f} MW)")
    else:
        evidence.append(f"latest FRP {history['latest']:.1f} MW vs. near-zero baseline at this site")

    return {"class_probabilities": dist, "top_class": top_class, "evidence": evidence,
            "simulated": history["simulated"]}
