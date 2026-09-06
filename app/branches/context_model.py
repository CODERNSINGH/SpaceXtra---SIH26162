"""
Branch B — Context/Tabular.

Real, actually-fitted scikit-learn RandomForestClassifier over numeric
context features (distance to nearest industrial facility, soil moisture,
temperature, humidity, precipitation, FRP, brightness).

*** TRAINING DATA IS A SYNTHETIC PLACEHOLDER. ***
No real labeled dataset of industrial-vs-natural thermal events exists yet
for this project. `generate_synthetic_training_data()` below encodes simple,
clearly-commented heuristic rules (closer to industry + drier soil trends
"industrial-like", etc.) so the model is genuinely trained and genuinely
predicting on those rules — it is not real ground truth.

Swap-in point for a real dataset: replace the body of
`generate_synthetic_training_data()` with a loader that reads a real labeled
CSV (same column names as FEATURE_COLUMNS + a "label" column) — nothing else
in this module, or in fusion.py / the API routes, needs to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from app.branches import CATEGORIES

FEATURE_COLUMNS = [
    "distance_m", "is_industrial", "is_refinery_power", "is_mine",
    "frp", "brightness", "soil_moisture", "temp_c", "humidity_pct", "precip_mm",
]


def generate_synthetic_training_data(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """
    SYNTHETIC PLACEHOLDER training data generator.

    Encodes simple domain heuristics as generation rules, not real
    observations:
      - very close to a mine landuse tag                -> mining_activity
      - close to refinery/power infra, low/steady FRP    -> gas_flare
      - close to refinery/power infra, high FRP          -> industrial_fire
      - close to generic industrial landuse, high FRP    -> industrial_fire
      - close to generic industrial landuse, lower FRP   -> persistent_industrial_source
      - far from any facility, dry soil, hot, high FRP   -> wildfire
      - moderate distance, dry, low humidity, moderate FRP -> agricultural_burn
      - everything else                                  -> unknown

    Replace this function with a real labeled-CSV loader (same
    FEATURE_COLUMNS + "label") to move off the placeholder — nothing else
    in the branch or the fusion engine needs to change.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        facility_roll = rng.random()
        if facility_roll < 0.15:
            facility = "mine"
            distance_m = abs(rng.normal(400, 300))
        elif facility_roll < 0.35:
            facility = "refinery_power"
            distance_m = abs(rng.normal(600, 400))
        elif facility_roll < 0.60:
            facility = "industrial"
            distance_m = abs(rng.normal(800, 500))
        else:
            facility = "none"
            distance_m = abs(rng.normal(4000, 1500))
        distance_m = float(np.clip(distance_m, 20, 8000))

        soil_moisture = float(np.clip(rng.normal(0.35, 0.18), 0.02, 0.95))
        temp_c = float(np.clip(rng.normal(29, 6), 8, 46))
        humidity_pct = float(np.clip(rng.normal(55, 20), 5, 100))
        precip_mm = float(max(0, rng.exponential(2.0)))
        frp = float(max(0.1, rng.gamma(2.0, 6.0)))
        brightness = float(np.clip(rng.normal(320, 25), 270, 400))

        is_industrial = 1 if facility == "industrial" else 0
        is_refinery_power = 1 if facility == "refinery_power" else 0
        is_mine = 1 if facility == "mine" else 0

        # --- rule-based synthetic label ---
        if is_mine and distance_m < 1200:
            label = "mining_activity"
        elif is_refinery_power and distance_m < 1500:
            label = "industrial_fire" if frp > 18 else "gas_flare"
        elif is_industrial and distance_m < 2000:
            label = "industrial_fire" if frp > 20 else "persistent_industrial_source"
        elif distance_m > 3000 and soil_moisture < 0.25 and temp_c > 32 and frp > 10:
            label = "wildfire"
        elif 1200 <= distance_m <= 4500 and soil_moisture < 0.35 and humidity_pct < 50 and 3 <= frp <= 20:
            label = "agricultural_burn"
        else:
            label = "unknown"

        rows.append({
            "distance_m": distance_m, "is_industrial": is_industrial,
            "is_refinery_power": is_refinery_power, "is_mine": is_mine,
            "frp": frp, "brightness": brightness, "soil_moisture": soil_moisture,
            "temp_c": temp_c, "humidity_pct": humidity_pct, "precip_mm": precip_mm,
            "label": label,
        })
    return pd.DataFrame(rows)


class ContextBranchModel:
    """Trains once at process start; holds the fitted model + held-out metrics."""

    def __init__(self):
        df = generate_synthetic_training_data()
        X = df[FEATURE_COLUMNS]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=7, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=7, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)

        self.training_df = df
        self.X_test, self.y_test = X_test, y_test
        self.metrics, self.curves = self._evaluate(X_test, y_test)

    def _evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        classes = self.classes_

        y_test_bin = label_binarize(y_test, classes=classes)

        # macro-average one-vs-rest AUC (multiclass)
        try:
            auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            auc = float("nan")

        metrics = {
            "n_test": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "auc": float(auc),
        }

        cm = confusion_matrix(y_test, y_pred, labels=classes)

        # micro-average ROC / PR across all classes (one-vs-rest) for a single
        # representative curve in the dashboard.
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())

        curves = {
            "confusion_matrix": cm,
            "labels": classes,
            "roc": (fpr, tpr),
            "pr": (precision, recall),
        }
        return metrics, curves

    def predict(self, features: dict) -> dict:
        row = pd.DataFrame([{c: features.get(c, 0.0) for c in FEATURE_COLUMNS}])
        proba = self.model.predict_proba(row)[0]
        dist = {cls: float(p) for cls, p in zip(self.classes_, proba)}
        # ensure every category present even if the model never predicts it
        full = {c: dist.get(c, 0.0) for c in CATEGORIES}
        total = sum(full.values()) or 1.0
        full = {c: v / total for c, v in full.items()}

        top_class = max(full, key=full.get)
        evidence = []
        if features.get("distance_m") is not None:
            evidence.append(f"{features['distance_m']:.0f}m from nearest known industrial facility "
                             f"({'found' if features.get('is_industrial') or features.get('is_refinery_power') or features.get('is_mine') else 'none nearby'})")
        if features.get("soil_moisture") is not None:
            evidence.append(f"root-zone soil moisture index {features['soil_moisture']:.2f} "
                             f"({'dry' if features['soil_moisture'] < 0.3 else 'moist'})")
        if features.get("frp") is not None:
            evidence.append(f"fire radiative power {features['frp']:.1f} MW (context-branch input)")
        evidence.append(f"Random Forest (synthetic-trained) top class: {top_class} "
                         f"({full[top_class]*100:.1f}%)")

        return {"class_probabilities": full, "top_class": top_class, "evidence": evidence}


_context_model: ContextBranchModel | None = None


def get_context_model() -> ContextBranchModel:
    global _context_model
    if _context_model is None:
        _context_model = ContextBranchModel()
    return _context_model
