"""Shared evaluation harness for the cracking models.

Three things every run reports, in this order:

  tuned params   what the grid search picked
  nested AUC     honest score - tuning happens inside each outer fold
  permutation p  how often shuffled labels score as well, same model config

Pass `groups` whenever rows share a source value (bath chemistry is measured
every few days, so a dozen electrodes carry identical numbers). Grouped
folds keep those rows together; without that the model scores itself on its
own neighbours.

Any estimator works — pass `estimator` and `grid` to swap the forest out.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score,
                                     permutation_test_score, train_test_split)

SEED = 0
FOREST_GRID = {
    "max_depth": [2, 3, None],
    "min_samples_leaf": [1, 5, 15],
    "max_features": ["sqrt", 0.5, None],
}


def make_forest(**params):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1, **params)


def splitter(groups, splits=5):
    if groups is None:
        return StratifiedKFold(splits, shuffle=True, random_state=SEED)
    return StratifiedGroupKFold(splits, shuffle=True, random_state=SEED)


def tune(estimator, grid, X, y, groups):
    search = GridSearchCV(estimator, grid, scoring="roc_auc",
                          cv=splitter(groups, 3), n_jobs=-1)
    search.fit(X, y, groups=groups) if groups is not None else search.fit(X, y)
    return search.best_params_


def nested_auc(estimator, grid, X, y, groups):
    inner = GridSearchCV(estimator, grid, scoring="roc_auc",
                         cv=splitter(groups, 3), n_jobs=-1)
    return cross_val_score(inner, X, y, groups=groups, scoring="roc_auc",
                           cv=splitter(groups),
                           params={"groups": groups} if groups is not None
                           else None)


def permutation_p(estimator, X, y, groups, params, rounds=200):
    model = clone(estimator).set_params(**params)
    score, _null, p_value = permutation_test_score(
        model, X, y, groups=groups, scoring="roc_auc", cv=splitter(groups),
        n_permutations=rounds, random_state=SEED, n_jobs=-1)
    return score, p_value


def held_out_importance(estimator, X, y, params, rounds=15):
    scores = []
    for i in range(rounds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=i)
        model = clone(estimator).set_params(**params).fit(X_train, y_train)
        result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                        scoring="roc_auc", random_state=i)
        scores.append(result.importances_mean)
    return pd.DataFrame(scores, columns=X.columns).mean().sort_values(
        ascending=False)


def evaluate(name, X, y, groups=None, estimator=None, grid=None, show=8):
    estimator = make_forest() if estimator is None else estimator
    grid = FOREST_GRID if grid is None else grid

    print(f"\n=== {name} ===")
    print(f"{len(X)} electrodes, {int(y.sum())} cracked ({y.mean():.0%}), "
          f"{X.shape[1]} features", end="")
    print(f", {pd.Series(groups).nunique()} groups" if groups is not None
          else "")

    params = tune(estimator, grid, X, y, groups)
    print("tuned:", params)

    scores = nested_auc(estimator, grid, X, y, groups)
    print(f"nested CV AUC: {scores.mean():.3f} (sd {scores.std():.3f})")

    score, p_value = permutation_p(estimator, X, y, groups, params)
    print(f"real labels:   {score:.3f}, shuffled beat it {p_value:.1%} "
          f"of the time")

    print("permutation importance (AUC lost when shuffled):")
    importance = held_out_importance(estimator, X, y, params)
    for feature, value in importance.head(show).items():
        print(f"  {feature:26} {value:+.4f}")
    return scores.mean(), p_value


def positive_control(frame, y, **kwargs):
    """A known-signal check: cracking really does differ by plating vat and
    drift over the window. If the harness cannot find this, it is broken."""
    X = pd.DataFrame({
        "plating_vat": frame["plating_vat"].astype("category").cat.codes,
        "day": pd.to_datetime(frame["production_date"]).map(
            lambda d: d.toordinal()),
    })
    return evaluate("POSITIVE CONTROL (vat + production date)", X, y,
                    show=2, **kwargs)
