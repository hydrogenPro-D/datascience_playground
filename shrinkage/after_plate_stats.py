"""Cathode diameter statistics after plating, from
cathode_diameters_after_plate.csv.

d1 and d2 are the two readings taken at right angles and are kept apart, so
a difference between the axes stays visible.

Shrinkage is NOMINAL - the measured diameter: positive means the cathode came
out of the bath smaller than it went in. A negative value is a cathode that
measured larger than nominal, which happens often enough that min is reported
rather than clipped."""
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent / "cathode_diameters_after_plate.csv"
COLUMNS = [("cathode_d1", "d1"), ("cathode_d2", "d2")]
NOMINAL = 1899.0        # the diameter the cathodes are supposed to start at
SCRAP = 1889.0          # under this the electrode is scrapped for shrinking


def table(title, series):
    header = f"{title:<34}{'n':>6}{'mean':>11}{'std':>9}{'min':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for name, values in series:
        print(f"{name:<34}{len(values):>6}{values.mean():>11.2f}"
              f"{values.std(ddof=1):>9.2f}"
              f"{values.min():>10g}{values.max():>10g}")
    print()


def main():
    data = pd.read_csv(CSV)
    readings = [(name, data[column].dropna()) for column, name in COLUMNS]
    pooled = pd.concat([values for _name, values in readings])

    table("diameter  (mm)", readings)
    table(f"shrinkage from {NOMINAL:g} mm",
          [(name, NOMINAL - values) for name, values in readings]
          + [("both readings", NOMINAL - pooled)])

    grew = (pooled > NOMINAL).mean() * 100
    print(f"{(NOMINAL - pooled).mean():.2f} mm shrinkage on average over all "
          f"{len(pooled):,} readings  ({(NOMINAL - pooled).mean() / NOMINAL * 100:.2f}% "
          f"of the nominal diameter)")
    print(f"{grew:.1f}% of the readings came out at or above {NOMINAL:g} mm")

    # An electrode counts once however many of its two readings are under the
    # scrap limit, so this is a row count, not a reading count.
    under = data[(data["cathode_d1"] < SCRAP) | (data["cathode_d2"] < SCRAP)]
    both = (under["cathode_d1"] < SCRAP) & (under["cathode_d2"] < SCRAP)
    print(f"{len(under)} of {len(data):,} electrodes below {SCRAP:g} mm  "
          f"({len(under) / len(data) * 100:.1f}%)"
          f"   - {int(both.sum())} on both readings, "
          f"{int((~both).sum())} on one")


if __name__ == "__main__":
    main()
