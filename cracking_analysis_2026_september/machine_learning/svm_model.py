"""Support vector machine on the per-vat instrument readings.

An SVM is supervised — it is told which electrodes cracked and looks for a
boundary between them — so it answers a different question from the
clustering in clustering.py. Features are standardised inside the
pipeline, which an SVM needs and which must happen per fold, not once up
front, or the test data leaks into the scaling.
"""
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import modeling

HERE = Path(__file__).resolve().parent
DATASET = HERE / "vat_signals_dataset.csv"

GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
    "svc__kernel": ["rbf", "linear"],
}


def make_svm():
    return Pipeline([("scale", StandardScaler()),
                     ("svc", SVC(class_weight="balanced", random_state=0))])


def main():
    frame = pd.read_csv(DATASET)
    y = frame["cracked"]
    features = [c for c in frame.columns
                if c.startswith(("strike_", "plating_"))
                and c != "plating_vat"]
    X = frame[features]

    modeling.evaluate("SVM on vat signals", X, y,
                      estimator=make_svm(), grid=GRID)
    modeling.evaluate("Random forest on the same features, for comparison",
                      X, y)
    modeling.positive_control(frame, y)


if __name__ == "__main__":
    main()
