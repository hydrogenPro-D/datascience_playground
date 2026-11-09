"""Random forest on process step times, looking for what drives cracking.

Features are absolute actual seconds per step. Set SUBSET to a dict to
restrict the model to named steps, or leave it None to use every step in
the file. Performance is cross-validated and tested against shuffled
labels, because with ~58 cracked electrodes a forest will fit noise.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     permutation_test_score, train_test_split)

FOLDER = Path(__file__).resolve().parent
WIDE_FILE = FOLDER / "all_step_times_wide.csv"
SEED = 0

SUSPECTED = {
    "step_19_p1_actual": "rinse 19",
    "step_20_p1_actual": "rinse 20",
    "step_26_p1_actual": "drip 26",
    "step_31_p2_actual": "strike 31 (2nd pass)",
    "step_25_p1_actual": "rinse 25",
}
SUBSET = None          # None = every step; or SUSPECTED
MIN_COVERAGE = 20      # skip steps recorded for fewer electrodes than this


def load():
    df = pd.read_csv(WIDE_FILE).dropna(subset=["verdict"])
    y = (df["verdict"] == "cracked").astype(int)

    if SUBSET:
        X = df[list(SUBSET)].rename(columns=SUBSET).dropna()
        return X, y.loc[X.index]

    columns = [c for c in df.columns if c.endswith("_actual")
               and df[c].notna().sum() >= MIN_COVERAGE]
    X = df[columns].rename(
        columns=lambda c: c.replace("step_", "").replace("_actual", ""))
    return X, y


def make_forest():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                  class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def single_feature_auc(X, y):
    scores = {}
    for name in X.columns:
        values = X[name].dropna()
        labels = y.loc[values.index]
        if labels.nunique() == 2 and values.nunique() > 1:
            auc = roc_auc_score(labels, values)
            scores[name] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def held_out_importance(X, y, rounds=15):
    scores = []
    for i in range(rounds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=i)
        model = make_forest().fit(X_train, y_train)
        result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                        scoring="roc_auc", random_state=i)
        scores.append(result.importances_mean)
    return pd.DataFrame(scores, columns=X.columns).mean().sort_values(
        ascending=False)


def main():
    X, y = load()
    print(f"{len(X)} electrodes, {y.sum()} cracked ({y.mean():.0%}), "
          f"{X.shape[1]} features\n")

    auc_by_feature = single_feature_auc(X, y)
    medians = X.assign(cracked=y.values).groupby("cracked").median().T
    medians.columns = ["not cracked", "cracked"]

    print("best separating steps on their own (0.5 = no separation):")
    print(f"  {'step':10}{'AUC':>7}{'median good':>14}{'median cracked':>16}")
    for name, auc in auc_by_feature.head(10).items():
        good, cracked = medians.loc[name]
        print(f"  {name:10}{auc:>7.3f}{good:>14.0f}{cracked:>16.0f}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    auc = cross_val_score(make_forest(), X, y, cv=cv, scoring="roc_auc")
    print(f"\nforest, 5-fold x10 CV AUC: {auc.mean():.3f} "
          f"(sd {auc.std():.3f})")

    score, _null, p_value = permutation_test_score(
        make_forest(), X, y, cv=5, scoring="roc_auc", n_permutations=200,
        random_state=SEED, n_jobs=-1)
    print(f"against shuffled labels:    {score:.3f}, p = {p_value:.3f}")

    print("\npermutation importance on held-out data "
          "(AUC lost when a feature is shuffled):")
    for name, value in held_out_importance(X, y).head(10).items():
        print(f"  {name:22} {value:+.4f}")


if __name__ == "__main__":
    main()
