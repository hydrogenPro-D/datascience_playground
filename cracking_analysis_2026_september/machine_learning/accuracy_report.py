"""What the forest gets right, in plain counts rather than AUC.

Predictions come from cross-validation, so every electrode is judged by a
model that never saw it. Accuracy alone is misleading when 70% of
electrodes are fine — always guessing "not cracked" already scores 70% —
so the baseline and the confusion matrix are printed alongside it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import modeling

HERE = Path(__file__).resolve().parent
DATASET = HERE / "vat_signals_dataset.csv"
BEST = {"max_depth": None, "max_features": None, "min_samples_leaf": 1}


def report(name, y, predicted):
    matrix = confusion_matrix(y, predicted)
    (true_good, false_cracked), (missed, caught) = matrix
    print(f"\n{name}")
    print(f"  accuracy           {accuracy_score(y, predicted):.1%}")
    print(f"  balanced accuracy  {balanced_accuracy_score(y, predicted):.1%}")
    print(f"  of those it calls cracked, {precision_score(y, predicted, zero_division=0):.0%} "
          f"really are   (precision)")
    print(f"  it catches {recall_score(y, predicted):.0%} of the cracked "
          f"electrodes            (recall)")
    print(f"    correctly passed {true_good:3}      wrongly flagged {false_cracked:3}")
    print(f"    missed cracked   {missed:3}      caught cracked  {caught:3}")


def main():
    frame = pd.read_csv(DATASET)
    y = frame["cracked"]
    features = [c for c in frame.columns
                if c.startswith(("strike_", "plating_"))
                and c != "plating_vat"]
    X = frame[features]

    print(f"{len(X)} electrodes, {int(y.sum())} cracked, "
          f"{len(X) - int(y.sum())} not cracked")
    print(f"always guessing 'not cracked' scores "
          f"{1 - y.mean():.1%} accuracy — that is the bar to beat")

    model = modeling.make_forest(**BEST)
    cv = StratifiedKFold(5, shuffle=True, random_state=modeling.SEED)
    probability = cross_val_predict(model, X, y, cv=cv,
                                    method="predict_proba")[:, 1]
    print(f"\nAUC for reference: {roc_auc_score(y, probability):.3f}")

    report("at the model's own threshold (0.5)", y, (probability >= 0.5).astype(int))

    print("\nthe trade-off at other thresholds:")
    print(f"  {'flag if risk >':>15}{'flagged':>9}{'accuracy':>10}"
          f"{'precision':>11}{'recall':>8}")
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        predicted = (probability >= threshold).astype(int)
        print(f"  {threshold:>15.1f}{predicted.sum():>9}"
              f"{accuracy_score(y, predicted):>10.1%}"
              f"{precision_score(y, predicted, zero_division=0):>11.0%}"
              f"{recall_score(y, predicted):>8.0%}")


if __name__ == "__main__":
    main()
