"""Fit the four models once and save the results, so the GUI only reads.

    .venv/Scripts/python.exe fit_models.py

Writes model_results.json beside this file: the scores, the shuffled-label
null, the permutation importances and the confusion matrices for

    all data      x  random forest
    all data      x  neural network
    filtered data x  random forest
    filtered data x  neural network

Takes about eight minutes, nearly all of it the two forests and their
nulls. That is exactly why it is a script and not something the tabs do
when you open them.

Re-run it after the pipeline: the file records the row and column counts it
was fitted on, and the tabs say so when the data has moved on since.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "model_results.json"


def jsonable(value):
    """numpy and pandas do not survive json.dump on their own."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main():
    # The table is assembled by the GUI's own widget, so the models see
    # exactly the columns the data tab shows - no second definition to
    # drift out of step.
    app = QApplication(sys.argv)                      # noqa: F841
    import data_filters
    import gui_all_data as gad
    import gui_app
    import ml_models

    if not ml_models.available():
        raise SystemExit("scikit-learn is not installed for this "
                         "interpreter - use .venv/Scripts/python.exe")

    data = gui_app.load_data()
    data["vats"] = data["vats"].astype(str)
    everything = gad.AllDataWidget(
        data, gui_app.load_step_times("all_step_times_long.csv", "run times"),
        gad.load_plate(), gad.load_ph(), gad.load_chemical_additions(),
        gad.load_conductivity(), gad.load_cleaner(), gad.load_activator(),
        gad.load_strike(), gad.load_water(), gad.load_signals_grid(),
        gad.load_sem_elements())
    clean, kept, notes = data_filters.apply_filters(everything._frame,
                                                    everything.COLUMNS)

    datasets = {"All data": (everything._frame, everything.COLUMNS),
                "Filtered data": (clean, kept)}
    saved = {"fitted_at": datetime.now().isoformat(timespec="seconds"),
             "filter_notes": notes, "datasets": {}}

    for where, (frame, columns) in datasets.items():
        saved["datasets"][where] = {
            "rows": int(len(frame)),
            "columns": int(len(columns)),
            "models": {},
        }
        for kind in ("forest", "network"):
            started = time.time()
            print(f"fitting {where} / {ml_models.TITLES[kind]} ...",
                  flush=True)
            results = ml_models.evaluate(frame, columns, kind)
            took = time.time() - started
            for name, r in results.items():
                auc = r["auc"]
                print(f"   {name:24s} AUC {auc:.3f}  null "
                      f"{r['null_mean']:.3f}  {r['beaten']}/{r['rounds']} "
                      f"shuffles reached it")
                matched = r["confusion"]["prevalence"]
                print(f"   {'':24s} at the prevalence threshold: catches "
                      f"{matched['true_pos']} of "
                      f"{matched['true_pos'] + matched['false_neg']} cracked,"
                      f" {matched['false_pos']} false alarms")
            print(f"   done in {took:.0f}s", flush=True)
            saved["datasets"][where]["models"][kind] = {
                "title": ml_models.TITLES[kind],
                "seconds": round(took, 1),
                "results": jsonable(results),
            }

    OUTPUT.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"\n✓ {OUTPUT.name} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
