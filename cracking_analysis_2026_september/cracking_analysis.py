"""Cracking analysis on the preprocessed SEM data.

Reads sem_measurements_filtered.csv (produced by data_preprocessing.py).
Creates a grouped bar chart of cracked vs not cracked electrodes per strike
vat, with each vat's own distribution (summing to 100%) on the bars.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path(__file__).resolve().parent / "sem_measurements_filtered.csv"
CHART_FILE = Path(__file__).resolve().parent / "cracking_by_vat.png"

VATS = [28, 29, 30]

# Colorblind-safe categorical pair (validated palette, adjacent-pair CVD
# ΔE ≥ 8); counts and percentages are printed on the bars so color never
# carries the meaning alone.
COLOR_NOT_CRACKED = "#2a78d6"  # blue
COLOR_CRACKED = "#eb6834"      # orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} electrodes from {INPUT_FILE.name}")

    # Single-vat electrodes with a verdict only; report what falls outside.
    df["vats"] = df["vats"].astype(str)
    usable = df[df["vats"].isin(str(v) for v in VATS) & df["sem_result"].notna()].copy()
    excluded = len(df) - len(usable)
    if excluded:
        print(f"Excluded {excluded} electrodes (no verdict, unknown vat, or multiple vats)")

    usable["verdict"] = usable["sem_result"].map({"good": "not cracked", "bad": "cracked"})

    counts = (usable.groupby(["vats", "verdict"]).size()
              .unstack(fill_value=0)
              .reindex([str(v) for v in VATS], fill_value=0)
              .reindex(columns=["not cracked", "cracked"], fill_value=0))
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    print("\nElectrodes per vat:")
    print(counts.to_string())
    print("\nDistribution per vat (each row sums to 100%):")
    print(percents.round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(VATS))
    width = 0.32
    gap = 0.02  # ~2px surface gap between the paired bars
    bars_not = ax.bar(x - width / 2 - gap / 2, counts["not cracked"], width,
                      color=COLOR_NOT_CRACKED, label="not cracked", zorder=3)
    bars_cracked = ax.bar(x + width / 2 + gap / 2, counts["cracked"], width,
                          color=COLOR_CRACKED, label="cracked", zorder=3)

    # Direct labels: count plus this vat's own share (per-vat rows sum to 100%)
    for bars, column in ((bars_not, "not cracked"), (bars_cracked, "cracked")):
        labels = [f"{count}\n({pct:.0f}%)"
                  for count, pct in zip(counts[column], percents[column])]
        ax.bar_label(bars, labels=labels, padding=3,
                     color=INK_SECONDARY, fontsize=9)

    ax.set_title("SEM cracking verdict by strike vat (Gen4.1 cathodes)",
                 color=INK, fontsize=12, pad=14, loc="left")
    ax.set_xlabel("Strike vat", color=INK_SECONDARY)
    ax.set_ylabel("Electrodes", color=INK_SECONDARY)
    ax.set_xticks(x, [f"Vat {v}" for v in VATS])
    ax.margins(y=0.15)  # headroom for the two-line bar labels

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper right")

    fig.tight_layout()
    fig.savefig(CHART_FILE, facecolor=SURFACE)
    print(f"\n✓ Chart saved to {CHART_FILE}")


if __name__ == "__main__":
    main()
