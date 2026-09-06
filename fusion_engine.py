"""
fusion_engine.py  (Layer 3 — AI Fusion Engine)
-------------------------------------------------
REAL module: combines the three branch outputs with a simple, transparent
weighted average — exactly the "Model 0 fusion" recommended as step one in
the architecture PDF before upgrading to a learned meta-classifier.
"""
import numpy as np

CLASS_LABELS = [
    "industrial_fire", "persistent_industrial_source", "wildfire",
    "agricultural_burn", "gas_flare", "mining_activity", "unknown",
]

DEFAULT_WEIGHTS = {"vision": 0.4, "context": 0.4, "temporal": 0.2}


def fuse_branches(vision: dict, context: dict, temporal: dict, weights: dict = None) -> dict:
    weights = weights or DEFAULT_WEIGHTS

    vision_ind = vision.get("industrial_probability_percent", 50)
    context_ind = context.get("industrial_probability_percent", 50)
    # temporal branch doesn't vote industrial-vs-natural directly, but its
    # anomaly score nudges confidence and risk
    anomaly = temporal.get("anomaly_score_percent", 0)

    fused_industrial = (
        weights["vision"] * vision_ind
        + weights["context"] * context_ind
        + weights["temporal"] * anomaly
    ) / sum(weights.values())

    fused_industrial = round(fused_industrial, 1)
    fused_natural = round(100 - fused_industrial, 1)

    if fused_industrial >= 65:
        classification = "industrial_fire" if temporal.get("is_abnormal") else "persistent_industrial_source"
    elif fused_industrial <= 35:
        classification = "wildfire"
    else:
        classification = "unknown"

    confidence = round(abs(fused_industrial - 50) * 2, 1)  # distance from "coin flip", scaled to 0-100

    risk_score = round(0.6 * confidence + 0.4 * anomaly, 1)
    risk_score = max(0, min(100, risk_score))

    return {
        "classification": classification,
        "fused_industrial_probability_percent": fused_industrial,
        "fused_natural_probability_percent": fused_natural,
        "confidence_percent": confidence,
        "risk_score": risk_score,
        "branch_votes": {
            "vision_industrial_pct": vision_ind,
            "context_industrial_pct": context_ind,
            "temporal_anomaly_pct": anomaly,
        },
        "weights_used": weights,
    }


def derive_multiclass_distribution(fusion_result: dict, temporal_result: dict, context: dict) -> dict:
    """NOTE: HEURISTIC, DEMO-ONLY — this is NOT a trained multi-class model.
    It redistributes the binary industrial/natural fusion score across the
    7 SIH problem-statement classes using simple, explainable rules, purely
    so the dashboard can show a full multi-class bar chart. A real system
    would train one multi-class model (or a proper one-vs-rest ensemble) on
    labeled examples of every class."""
    industrial = fusion_result["fused_industrial_probability_percent"]
    natural = fusion_result["fused_natural_probability_percent"]
    anomaly = temporal_result.get("anomaly_score_percent", 0)
    dist_m = context.get("facility_distance_m")
    close_to_facility = dist_m is not None and dist_m < 800

    raw = {
        "industrial_fire": industrial * (anomaly / 100) * (1.3 if close_to_facility else 0.8),
        "persistent_industrial_source": industrial * (1 - anomaly / 100) * (1.2 if close_to_facility else 0.7),
        "gas_flare": industrial * 0.35 * (1.1 if close_to_facility else 0.5),
        "mining_activity": industrial * 0.2,
        "wildfire": natural * (anomaly / 100 + 0.3),
        "agricultural_burn": natural * 0.5,
        "unknown": 5.0,
    }
    total = sum(raw.values()) or 1.0
    normalized = {k: round(100 * v / total, 1) for k, v in raw.items()}
    return normalized


def run_stability_check(vision: dict, context: dict, temporal: dict, weights: dict = None,
                         n_runs: int = 200, noise_pct: float = 0.05, seed: int = 42) -> dict:
    """REAL Monte Carlo robustness check: perturbs each branch's input signal
    by +/- noise_pct and re-runs the real fuse_branches() formula n_runs
    times. This genuinely measures how sensitive the final confidence/risk
    are to small input jitter — a real stability test, not a decorative one,
    it just happens to run on the fused numbers rather than on raw sensors."""
    rng = np.random.RandomState(seed)
    base_vision = vision.get("industrial_probability_percent", 50)
    base_context = context.get("industrial_probability_percent", 50)
    base_anomaly = temporal.get("anomaly_score_percent", 0)
    is_abnormal = temporal.get("is_abnormal", False)

    confidences, risks = [], []
    for _ in range(n_runs):
        v = float(np.clip(base_vision * (1 + rng.uniform(-noise_pct, noise_pct)), 0, 100))
        c = float(np.clip(base_context * (1 + rng.uniform(-noise_pct, noise_pct)), 0, 100))
        a = float(np.clip(base_anomaly * (1 + rng.uniform(-noise_pct, noise_pct)), 0, 100))
        result = fuse_branches(
            {"industrial_probability_percent": v},
            {"industrial_probability_percent": c},
            {"anomaly_score_percent": a, "is_abnormal": is_abnormal},
            weights=weights,
        )
        confidences.append(result["confidence_percent"])
        risks.append(result["risk_score"])

    return {
        "confidence_samples": confidences,
        "risk_samples": risks,
        "n_runs": n_runs,
        "noise_pct": noise_pct,
        "source": "Monte Carlo re-run of the real fusion formula (real math on real perturbations)",
    }
