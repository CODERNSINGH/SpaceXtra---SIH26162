"""
tabular_model.py  (Branch B — "Context / Tabular Model")
----------------------------------------------------------
HONEST HYBRID MODULE: the MODEL ITSELF is real — an actual scikit-learn
Random Forest that really fits and really predicts. What's fake is the
TRAINING DATA: since the team doesn't have a real labeled dataset yet, this
file generates a synthetic training set with rules loosely inspired by the
research roadmap (closer to industry + drier soil -> more "industrial"-like
patterns) purely so the model has something to learn from for the demo.

Swap `_generate_synthetic_training_data()` for a real labeled CSV
(thermal_event_dataset.csv from the other notebooks) and everything else in
this file works unchanged. That's the point of building it this way.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "facility_distance_m", "avg_temp_c", "avg_humidity_pct",
    "avg_rainfall_mm", "avg_root_zone_soil_moisture",
]


def _generate_synthetic_training_data(n=800, seed=42):
    """SIMULATED training set — for demo purposes only. Replace with real
    labeled data before this leaves prototype stage."""
    rng = np.random.RandomState(seed)
    facility_distance_m = rng.exponential(scale=1500, size=n)
    avg_temp_c = rng.normal(30, 5, size=n)
    avg_humidity_pct = rng.uniform(20, 90, size=n)
    avg_rainfall_mm = rng.exponential(scale=2, size=n)
    avg_root_zone_soil_moisture = rng.uniform(0.05, 0.6, size=n)

    # Rough synthetic rule: close to industry + low rainfall -> "industrial"
    industrial_score = (
        (1 / (1 + facility_distance_m / 500))
        + (1 - avg_root_zone_soil_moisture)
        - (avg_rainfall_mm / 10)
    )
    label = (industrial_score > np.median(industrial_score)).astype(int)  # 1 = industrial-like

    df = pd.DataFrame({
        "facility_distance_m": facility_distance_m,
        "avg_temp_c": avg_temp_c,
        "avg_humidity_pct": avg_humidity_pct,
        "avg_rainfall_mm": avg_rainfall_mm,
        "avg_root_zone_soil_moisture": avg_root_zone_soil_moisture,
        "label": label,
    })
    return df


def train_prototype_model(seed=42) -> RandomForestClassifier:
    df = _generate_synthetic_training_data(seed=seed)
    X = df[FEATURE_NAMES]
    y = df["label"]
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    clf.fit(X, y)
    return clf


def run_context_branch(context: dict, clf: RandomForestClassifier = None) -> dict:
    """Predicts on REAL fetched context data (facility distance, weather,
    soil moisture) using the model above. The prediction call itself is a
    genuine model.predict_proba — only the training data behind it is
    synthetic."""
    clf = clf or train_prototype_model()

    row = {
        "facility_distance_m": context.get("facility_distance_m") if context.get("facility_distance_m") is not None else 5000,
        "avg_temp_c": context.get("avg_temp_c") if context.get("avg_temp_c") is not None else 28.0,
        "avg_humidity_pct": context.get("avg_humidity_pct") if context.get("avg_humidity_pct") is not None else 50.0,
        "avg_rainfall_mm": context.get("avg_rainfall_mm") if context.get("avg_rainfall_mm") is not None else 1.0,
        "avg_root_zone_soil_moisture": context.get("avg_root_zone_soil_moisture") if context.get("avg_root_zone_soil_moisture") is not None else 0.3,
    }
    X = pd.DataFrame([row])[FEATURE_NAMES]
    proba = clf.predict_proba(X)[0]  # [P(natural), P(industrial)]
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_))

    return {
        "industrial_probability_percent": round(float(proba[1]) * 100, 1),
        "natural_probability_percent": round(float(proba[0]) * 100, 1),
        "feature_importances": importances,
        "source": "random_forest (trained on SYNTHETIC placeholder data)",
    }
