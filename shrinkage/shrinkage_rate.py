"""Share of electrodes scrapped for shrinking below 1889 mm.

Reads production_figures_shrinkage.csv (the Aug / Sept / Oct 2026 tabs of
"Line 1 - Production figures.xlsx", one row per shift) and divides the
shrinkage scrap by the electrodes that could actually have been measured:
everything sent into PL1 minus the ones sacrificed for quality control,
since a sacrificed electrode is cut up and never gets a verdict."""
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent / "production_figures_shrinkage.csv"


def rate(frame):
    """(sent to PL1, sacrificed for QC, shrinkage scrap, eligible, share)."""
    sent = int(frame["sent_to_pl1"].sum())
    qc = int(frame["sacrificed_for_qc"].sum())
    shrunk = int(frame["scrapped_shrinking"].sum())
    eligible = sent - qc
    share = shrunk / eligible * 100 if eligible else float("nan")
    return sent, qc, shrunk, eligible, share


def main():
    data = pd.read_csv(CSV)

    header = (f"{'':<14}{'sent PL1':>10}{'to QC':>8}{'shrunk':>8}"
              f"{'eligible':>10}{'shrink %':>10}")
    print(header)
    print("-" * len(header))
    for name, part in data.groupby("sheet", sort=False):
        sent, qc, shrunk, eligible, share = rate(part)
        print(f"{name:<14}{sent:>10}{qc:>8}{shrunk:>8}{eligible:>10}"
              f"{share:>9.2f}%")
    print("-" * len(header))
    sent, qc, shrunk, eligible, share = rate(data)
    print(f"{'total':<14}{sent:>10}{qc:>8}{shrunk:>8}{eligible:>10}"
          f"{share:>9.2f}%")

    print(f"\n{shrunk} / ({sent} - {qc}) = {shrunk} / {eligible} "
          f"= {share:.2f}% of the electrodes that reached PL1 and were not "
          f"sacrificed were scrapped for shrinking.")


if __name__ == "__main__":
    main()
