"""
Layer 3 — Fusion.

Simple, transparent weighted-average fusion of the three branches'
per-category probability distributions. Weights live in
`app.config.settings.FUSION_WEIGHTS` (config-editable, must sum to 1.0) —
nothing here is a black box.
"""
from __future__ import annotations

from app.branches import CATEGORIES
from app.config import settings

# Category severity weights used only for the final 0-100 risk score
# (documented here, not hidden): how concerning a class is, all else equal.
SEVERITY_WEIGHTS = {
    "industrial_fire": 1.0,
    "wildfire": 0.95,
    "persistent_industrial_source": 0.7,
    "gas_flare": 0.55,
    "mining_activity": 0.5,
    "agricultural_burn": 0.4,
    "unknown": 0.3,
}


def fuse(vision: dict, context: dict, temporal: dict, frp: float | None) -> dict:
    weights = settings.FUSION_WEIGHTS
    fused = {c: 0.0 for c in CATEGORIES}
    for c in CATEGORIES:
        fused[c] = (
            weights["vision"] * vision["class_probabilities"].get(c, 0.0)
            + weights["context"] * context["class_probabilities"].get(c, 0.0)
            + weights["temporal"] * temporal["class_probabilities"].get(c, 0.0)
        )
    total = sum(fused.values()) or 1.0
    fused = {c: v / total for c, v in fused.items()}

    classification = max(fused, key=fused.get)
    confidence = fused[classification]

    frp_factor = min(1.25, 1.0 + (frp or 0.0) / 80.0)
    risk_score = round(100 * confidence * SEVERITY_WEIGHTS.get(classification, 0.5) * frp_factor)
    risk_score = max(0, min(100, risk_score))

    evidence = []
    evidence.extend(f"[VISION] {e}" for e in vision.get("evidence", []))
    evidence.extend(f"[CONTEXT] {e}" for e in context.get("evidence", []))
    evidence.extend(f"[TEMPORAL] {e}" for e in temporal.get("evidence", []))

    return {
        "classification": classification,
        "confidence": confidence,
        "risk_score": risk_score,
        "class_probabilities": fused,
        "evidence": evidence,
        "weights_used": weights,
    }
