"""Cathode diameter statistics, Gen 2 against the rest, from
cathode_diameters.csv (Sandwich Quality Control, boxes 99-124).

d1 and d2 are kept apart: they are the two readings taken at right angles,
so averaging them first would hide any difference between the axes.

Shrinkage is NOMINAL - the measured diameter: positive means the cathode came
out of the bath smaller than it went in."""
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent / "cathode_diameters.csv"
GROUPS = [("yes", "Gen 2        (up to C16161)"),
          ("no", "after Gen 2  (C16163 on)")]
COLUMNS = [("cathode_d1", "d1"), ("cathode_d2", "d2")]
NOMINAL = 1899.0        # the diameter the cathodes are supposed to start at
SCRAP = 1889.0          # under this the electrode is scrapped for shrinking


def table(title, rows):
    header = f"{title:<32}{'n':>6}{'mean':>11}{'std':>9}{'min':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for label, name, values in rows:
        print(f"{label:<28}{name:<4}{len(values):>6}{values.mean():>11.2f}"
              f"{values.std(ddof=1):>9.2f}"
              f"{values.min():>10g}{values.max():>10g}")
    print()


def main():
    data = pd.read_csv(CSV)

    rows = []
    for flag, label in GROUPS:
        part = data[data["is_gen2"] == flag]
        for column, name in COLUMNS:
            rows.append((label if name == "d1" else "", name,
                         part[column].dropna()))

    table("diameter  (mm)", rows)
    table(f"shrinkage from {NOMINAL:g} mm",
          [(label, name, NOMINAL - values) for label, name, values in rows])

    for flag, label in GROUPS:
        part = data[data["is_gen2"] == flag]
        pooled = pd.concat([part[column].dropna() for column, _n in COLUMNS])
        shrink = NOMINAL - pooled
        print(f"{label}: {shrink.mean():.2f} mm on average, "
              f"std {shrink.std(ddof=1):.2f}, over {len(pooled):,} readings  "
              f"({shrink.mean() / NOMINAL * 100:.2f}% of nominal)")
    print()

    # An electrode counts once however many of its two readings are under the
    # scrap limit, so this is a row count, not a reading count.
    for flag, label in GROUPS:
        part = data[data["is_gen2"] == flag]
        under = part[(part["cathode_d1"] < SCRAP) | (part["cathode_d2"] < SCRAP)]
        both = (under["cathode_d1"] < SCRAP) & (under["cathode_d2"] < SCRAP)
        print(f"{label}: {len(under)} of {len(part):,} electrodes below "
              f"{SCRAP:g} mm  ({len(under) / len(part) * 100:.1f}%)"
              f"   - {int(both.sum())} on both readings, "
              f"{int((~both).sum())} on one")


if __name__ == "__main__":
    main()
