"""
evaluation.py  (feeds Dashboard 4 — Model Evaluation)
-------------------------------------------------------
REAL module: every number in this file is a genuinely computed evaluation —
a real train/test split, a real confusion matrix, a real ROC curve, a real
precision-recall curve, and a real 2D decision-boundary contour, all from
scikit-learn running on a held-out split that the model never saw during
training.

The one honest caveat, inherited from tabular_model.py: the DATA underneath
is still the synthetic placeholder dataset (see that file's docstring), so
these metrics describe how well the model learned its own synthetic rule,
not real-world accuracy on real labeled events. Swap in a real labeled CSV
and every function below keeps working unchanged — that is the entire point
of separating "the model" from "the data" this way.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from tabular_model import _generate_synthetic_training_data, FEATURE_NAMES


def run_holdout_evaluation(seed: int = 42, n: int = 1600, test_size: float = 0.25) -> dict:
    """Trains on 75% of the synthetic dataset, evaluates on the untouched
    25% — a genuine held-out evaluation, not a self-report on training data."""
    df = _generate_synthetic_training_data(n=n, seed=seed)
    X = df[FEATURE_NAMES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_value = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred)), 3),
        "recall": round(float(recall_score(y_test, y_pred)), 3),
        "f1": round(float(f1_score(y_test, y_pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 3),
    }

    return {
        "confusion_matrix": cm,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_value,
        "pr_precision": precision, "pr_recall": recall,
        "metrics": metrics,
        "clf": clf,
        "y_test": y_test.to_numpy(),
        "y_proba": y_proba,
    }


def run_2d_decision_boundary(seed: int = 42, n: int = 1600,
                              feat_x: str = "facility_distance_m",
                              feat_y: str = "avg_root_zone_soil_moisture") -> dict:
    """A second, genuinely-fit 2-feature Random Forest purely so its decision
    surface can be drawn as a 2D contour — a real model, deliberately
    restricted to 2 inputs so it can be visualized at all."""
    df = _generate_synthetic_training_data(n=n, seed=seed)
    X = df[[feat_x, feat_y]].values
    y = df["label"].values

    clf2 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    clf2.fit(X, y)

    x_min, x_max = X[:, 0].min() - 200, X[:, 0].max() + 200
    y_min, y_max = X[:, 1].min() - 0.02, X[:, 1].max() + 0.02
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    return {"xx": xx, "yy": yy, "zz": zz, "X": X, "y": y, "feat_x": feat_x, "feat_y": feat_y}
