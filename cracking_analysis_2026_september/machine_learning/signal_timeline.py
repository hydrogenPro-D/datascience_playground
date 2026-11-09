"""Median temperature, current and voltage per electrode, over time.

One dot per electrode, joined in production order so the line traces what
the process actually did. Cracked electrodes carry both a different colour
and a different marker, so the verdict never rests on colour alone.

Writes signal_timeline.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = HERE / "vat_signals_dataset.csv"
OUTPUT = HERE / "signal_timeline.png"

GOOD = "#0EA5A5"
CRACKED = "#8B5CF6"
TRACE = "#C7CDD4"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"

SIGNALS = [("temp", "Temperature", "°C"),
           ("current", "Current", "A"),
           ("voltage", "Voltage", "V")]
PLACES = [("strike", "Strike (vat 31)"), ("plating", "Plating vat")]


def chrome(ax, title, ylabel):
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold",
                 loc="left", pad=8)
    ax.set_ylabel(ylabel, color=LABEL, fontsize=9.5, labelpad=6)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def draw(ax, frame, column, title, unit):
    ax.plot(frame["production_date"], frame[column], color=TRACE,
            linewidth=1.1, zorder=2)
    for cracked, colour, marker, size in ((0, GOOD, "o", 26),
                                          (1, CRACKED, "D", 30)):
        part = frame[frame["cracked"] == cracked]
        ax.scatter(part["production_date"], part[column], s=size,
                   color=colour, marker=marker, zorder=3,
                   edgecolors=AXES_BG, linewidths=0.8)
    ax.margins(y=0.09)
    chrome(ax, title, unit)


def main():
    frame = pd.read_csv(DATASET)
    frame["production_date"] = pd.to_datetime(frame["production_date"])
    frame = frame.sort_values("production_date")
    print(f"{len(frame)} electrodes, {int(frame['cracked'].sum())} cracked, "
          f"{frame['production_date'].min():%Y-%m-%d} to "
          f"{frame['production_date'].max():%Y-%m-%d}")

    figure, axes = plt.subplots(3, 2, figsize=(15, 10), dpi=120, sharex=True)
    figure.patch.set_facecolor(FIG_BG)
    for row, (signal, name, unit) in enumerate(SIGNALS):
        for col, (place, where) in enumerate(PLACES):
            draw(axes[row][col], frame, f"{place}_{signal}",
                 f"{name}: {where}", unit)

    for ax in axes[2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

    handles = [plt.Line2D([], [], marker="o", color=GOOD, linestyle="none",
                          markersize=6, label="not cracked"),
               plt.Line2D([], [], marker="D", color=CRACKED, linestyle="none",
                          markersize=6, label="cracked")]
    figure.legend(handles=handles, frameon=False, fontsize=10,
                  labelcolor=SUBTEXT, ncol=2, loc="upper left",
                  bbox_to_anchor=(0.05, 0.985))
    figure.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.07,
                           hspace=0.32, wspace=0.15)
    figure.savefig(OUTPUT, facecolor=FIG_BG)
    print(f"✓ {OUTPUT.name}")


if __name__ == "__main__":
    main()
