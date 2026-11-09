"""(a) Cracking vs the signals logged during each electrode's own vat visits.

Temperature, current and voltage are measured while the electrode is in the
bath, so unlike the chemistry these are genuinely per-electrode values with
no shared rows — ordinary cross-validation is sound here.

Writes signals_dataset.csv, then models it.
"""
from pathlib import Path

import pandas as pd

import modeling

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
DATASET = HERE / "signals_dataset.csv"

SIGNALS = {"Temperature": "temp", "Current": "current", "Voltage": "voltage"}
FEATURES = [f"{where}_{short}" for where in ("strike", "plating")
            for short in SIGNALS.values()]


def build():
    sem = pd.read_csv(SOURCE / "sem_measurements_filtered.csv")
    sem = sem.dropna(subset=["sem_result", "vats"])
    sem["cracked"] = (sem["sem_result"] == "bad").astype(int)
    sem["plating_vat"] = sem["vats"].astype(str).str.split("|").str[0]

    signals = pd.read_csv(SOURCE / "signals.csv")
    signals["Marking"] = signals["Marking"].astype(str).str.strip()
    signals["Vat"] = signals["Vat"].astype(str)

    rows = sem[["electrode", "cracked", "production_date", "plating_vat"]].copy()
    for signal, short in SIGNALS.items():
        part = signals[signals["signal"] == signal]
        lookup = part.set_index(["Marking", "Vat"])["value_mean"]
        rows[f"strike_{short}"] = [
            lookup.get((e, "31")) for e in rows["electrode"]]
        rows[f"plating_{short}"] = [
            lookup.get((e, v)) for e, v in zip(rows["electrode"],
                                               rows["plating_vat"])]

    rows = rows.dropna(subset=FEATURES)
    rows.to_csv(DATASET, index=False)
    print(f"✓ {len(rows)} electrodes x {rows.shape[1]} columns -> "
          f"{DATASET.name}")
    return rows


def main():
    frame = build()
    y = frame["cracked"]
    modeling.evaluate("(a) temperature + current + voltage",
                      frame[FEATURES], y)
    modeling.positive_control(frame, y)


if __name__ == "__main__":
    main()
