"""Presentation figure: cracking against strike bath temperature.

Bins hold an equal number of electrodes rather than an equal width, so no
point rests on one or two measurements and every interval is comparable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import temp_gui

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "temperature_figure.png"
CUT = 41.5
BINS = 8

COOL = "#0EA5A5"
HOT = "#8B5CF6"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"
MARK = "#111827"


def wilson(successes, total, z=1.96):
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total
                         + z ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def chrome(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=12.5, fontweight="bold",
                 loc="left", pad=12)
    ax.set_xlabel(xlabel, color=LABEL, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def trend_panel(ax, df):
    """Bins are spaced evenly rather than by temperature: the cold tail is
    thinly populated and on a true axis it squeezes every populated bin
    into the right-hand third."""
    df = df.assign(bin=pd.qcut(df["strike_temp"], BINS, duplicates="drop"))
    grouped = df.groupby("bin", observed=True).agg(
        n=("cracked", "size"), cracked=("cracked", "sum"),
        low=("strike_temp", "min"), high=("strike_temp", "max"),
        centre=("strike_temp", "median"))
    rate = (grouped["cracked"] / grouped["n"]).to_numpy()
    bounds = [wilson(c, n) for c, n in zip(grouped["cracked"], grouped["n"])]
    x = np.arange(len(grouped))
    hot = grouped["centre"].to_numpy() > CUT
    divider = float(np.argmax(hot)) - 0.5

    ax.axvspan(-0.6, divider, color=COOL, alpha=0.07, zorder=1)
    ax.axvline(divider, color=MARK, linewidth=1.4, linestyle=(0, (5, 3)),
               zorder=4)
    ax.plot(x, rate, color=MARK, linewidth=1.2, alpha=0.3, zorder=2)

    for i, (value, (low, high), n) in enumerate(zip(rate, bounds,
                                                    grouped["n"])):
        colour = HOT if hot[i] else COOL
        ax.vlines(i, low, high, color=colour, linewidth=1.8, alpha=0.85,
                  zorder=3)
        ax.plot(i, value, "o", markersize=10, color=colour,
                markeredgecolor="#FFFFFF", markeredgewidth=1.4, zorder=5)
        ax.annotate(f"{value:.0%}", (i, high), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10,
                    fontweight="bold", color=TEXT)

    ax.axhline(df["cracked"].mean(), color=MARK, linewidth=1.2,
               linestyle=(0, (2, 3)), alpha=0.55, zorder=2)
    ax.annotate(f"overall {df['cracked'].mean():.0%}",
                (len(grouped) - 0.6, df["cracked"].mean()),
                textcoords="offset points", xytext=(0, 5), ha="right",
                fontsize=8.5, color=SUBTEXT)
    ax.annotate(f"≈ {CUT} °C", (divider, 0.74), textcoords="offset points",
                xytext=(7, 0), fontsize=9.5, color=MARK, fontweight="bold")

    ax.set_xticks(x, [f"{lo:.1f}–{hi:.1f}\nn={n}" for lo, hi, n
                      in zip(grouped["low"], grouped["high"], grouped["n"])])
    ax.set_xlim(-0.6, len(grouped) - 0.4)
    ax.set_ylim(0, 0.82)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    chrome(ax, "Cracking rises with strike bath temperature",
           "strike bath temperature (°C), equal-sized groups",
           "share of electrodes cracked")


def split_panel(ax, df):
    cool = df[df["strike_temp"] <= CUT]
    hot = df[df["strike_temp"] > CUT]
    groups = [(f"at or below {CUT} °C", cool, COOL),
              (f"above {CUT} °C", hot, HOT)]

    for i, (name, part, colour) in enumerate(groups):
        rate = part["cracked"].mean()
        low, high = wilson(int(part["cracked"].sum()), len(part))
        ax.bar(i, rate, width=0.5, color=colour, alpha=0.75, zorder=3)
        ax.vlines(i, low, high, color=MARK, linewidth=1.6, zorder=5)
        ax.annotate(f"{rate:.0%}", (i, high), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=13,
                    fontweight="bold", color=TEXT)

    ax.set_xticks(range(2), [f"{name}\n{len(part)} electrodes\n"
                             f"{int(part['cracked'].sum())} cracked"
                             for name, part, _c in groups])
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    chrome(ax, "Split at the threshold", ylabel="share cracked")


def main():
    df = temp_gui.load()
    figure = plt.figure(figsize=(13, 5.6), dpi=120)
    figure.patch.set_facecolor(FIG_BG)
    grid = figure.add_gridspec(1, 2, width_ratios=[2.1, 1])
    trend_panel(figure.add_subplot(grid[0, 0]), df)
    split_panel(figure.add_subplot(grid[0, 1]), df)

    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18,
                           wspace=0.22)
    figure.savefig(OUTPUT, facecolor=FIG_BG)
    print(f"✓ {OUTPUT.name}")


if __name__ == "__main__":
    main()
