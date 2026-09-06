"""
temporal_model.py  (Branch C — "Temporal / Persistence Model")
------------------------------------------------------------------
HONEST HYBRID MODULE: the underlying 30-day detection history plotted here
is SIMULATED (deterministically, from the coordinate, so it's stable across
demo runs) because a real facility-level FIRMS history requires the archive
pull described in the architecture PDF. The anomaly-score MATH (rolling
mean, standard deviation, z-score) is real and would work unchanged on a
real history once you plug one in.
"""
import numpy as np
from utils import coord_seed


def simulate_thermal_history(lat: float, lon: float, days: int = 30):
    """SIMULATED — deterministic per-coordinate synthetic FRP history.
    Replace with a real FIRMS archive query (see architecture PDF, Layer 1)
    for production."""
    rng = np.random.RandomState(coord_seed(lat, lon))
    baseline = rng.uniform(20, 60)          # normal background FRP for this "site"
    noise = rng.normal(0, 5, size=days)
    history = np.clip(baseline + noise, 1, None)

    # Inject a plausible "event" on the final day so the demo has something
    # to react to — a spike a few standard deviations above baseline.
    spike_multiplier = rng.uniform(2.5, 5.0)
    history[-1] = baseline * spike_multiplier

    return history, baseline


def compute_anomaly_score(history: np.ndarray) -> dict:
    """REAL math: z-score of the latest reading against the trailing
    baseline (all but the last point)."""
    baseline_window = history[:-1]
    latest = history[-1]
    mean = float(np.mean(baseline_window))
    std = float(np.std(baseline_window)) or 1e-6
    z_score = (latest - mean) / std

    anomaly_percent = float(np.clip((z_score / 6.0) * 100, 0, 100))  # squash to 0-100
    return {
        "latest_frp": round(float(latest), 1),
        "baseline_mean_frp": round(mean, 1),
        "baseline_std_frp": round(std, 1),
        "z_score": round(z_score, 2),
        "anomaly_score_percent": round(anomaly_percent, 1),
        "is_abnormal": bool(z_score > 2.0),
    }


def run_temporal_branch(lat: float, lon: float, days: int = 30) -> dict:
    history, baseline = simulate_thermal_history(lat, lon, days=days)
    anomaly = compute_anomaly_score(history)
    anomaly["history"] = history.tolist()
    anomaly["source"] = "z-score anomaly detection (on SIMULATED history — see architecture PDF for the real FIRMS-archive version)"
    return anomaly
