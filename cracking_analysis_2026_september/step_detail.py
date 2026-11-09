"""Every electrode behind one step's bar pair.

The time tabs show a mean per group; when that mean sits far from the
median, a handful of long holds are driving it. This dumps the underlying
per-electrode rows to CSV and draws the distribution so the reason is
visible rather than inferred.

Edit STEP_FILE / STEP_VALUE to look at a different step.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_measurements_filtered.csv"

STEP_FILE = FOLDER / "rinse_times.csv"      # or drip_times.csv, ...
STEP_VALUE = "20"                            # vat/step to inspect
STEP_NAME = f"Rinse {STEP_VALUE}"

# House style (see PLOT_STYLE_GUIDE.md)
COLOR_GOOD = "#0EA5A5"
COLOR_CRACKED = "#8B5CF6"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"
TARGET_LINE = "#111827"

ZOOM_TOP = 300      # % of set time shown in the right-hand panel


def load():
    sem = pd.read_csv(SEM_FILE)
    sem["verdict"] = sem["sem_result"].map({"good": 1, "bad": 0})
    sem = sem.dropna(subset=["verdict"])

    steps = pd.read_csv(STEP_FILE)
    steps = steps[steps["vat"].astype(str) == STEP_VALUE]

    df = sem.merge(steps[["electrode", "set_time", "actual_time", "overrun"]],
                   on="electrode", how="inner")
    df["pct_of_set"] = 100 * df["actual_time"] / df["set_time"]
    df["pct_over_set"] = df["pct_of_set"] - 100
    df["verdict_label"] = np.where(df["verdict"] == 1, "not cracked", "cracked")
    return df.sort_values("pct_of_set", ascending=False)


def summarise(df):
    print(f"{STEP_NAME}: {len(df)} electrodes\n")
    header = (f"{'group':12}{'n':>5}{'mean':>9}{'median':>9}{'sd':>9}"
              f"{'min':>8}{'max':>10}")
    print(header)
    for name, part in df.groupby("verdict_label", sort=False):
        a = part["pct_of_set"]
        print(f"{name:12}{len(part):>5}{a.mean():>9.1f}{a.median():>9.1f}"
              f"{a.std():>9.1f}{a.min():>8.1f}{a.max():>10.1f}")

    on_target = (df["pct_of_set"].round(1) == 100.0).sum()
    print(f"\nexactly on recipe: {on_target} of {len(df)} "
          f"({100 * on_target / len(df):.0f}%)")

    worst = df.iloc[0]
    without = df.iloc[1:]
    same = without[without["verdict_label"] == worst["verdict_label"]]
    print(f"longest hold: {worst['electrode']} "
          f"({worst['verdict_label']}) {worst['actual_time']:.0f}s "
          f"= {worst['pct_of_set']:.0f}% of recipe")
    print(f"  drop it and '{worst['verdict_label']}' mean falls "
          f"{worst['pct_of_set'] - 0:.0f}% -> {same['pct_of_set'].mean():.1f}%")


def draw(df, path):
    fig, (ax_log, ax_zoom) = plt.subplots(
        1, 2, figsize=(12, 5.5), dpi=110)
    fig.patch.set_facecolor(FIG_BG)

    groups = [("not cracked", COLOR_GOOD), ("cracked", COLOR_CRACKED)]
    rng = np.random.default_rng(0)      # jitter only, fixed for reproducibility

    for ax, log in ((ax_log, True), (ax_zoom, False)):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        for i, (name, color) in enumerate(groups):
            part = df[df["verdict_label"] == name]
            values = part["pct_of_set"]
            jitter = rng.uniform(-0.17, 0.17, len(part))
            ax.scatter(i + jitter, values, s=26, color=color, alpha=0.55,
                       edgecolors="none", zorder=3)
            # mean and median markers, so the gap between them is explicit
            ax.hlines(values.mean(), i - 0.32, i + 0.32, color=TARGET_LINE,
                      linewidth=1.8, linestyle=(0, (4, 2)), zorder=5)
            ax.hlines(values.median(), i - 0.32, i + 0.32, color=TARGET_LINE,
                      linewidth=2.2, zorder=5)
            ax.annotate(f"mean {values.mean():.0f}%", (i + 0.34, values.mean()),
                        fontsize=8.5, color=SUBTEXT, va="center")
            ax.annotate(f"median {values.median():.0f}%",
                        (i + 0.34, values.median()), fontsize=8.5,
                        color=SUBTEXT, va="top")

        ax.axhline(100, color=SPINE, linewidth=1.2, zorder=1)
        ax.set_xticks(range(len(groups)),
                      [f"{name}\nn = {(df['verdict_label'] == name).sum()}"
                       for name, _c in groups])
        ax.set_xlim(-0.6, 1.9)
        ax.set_ylabel("% of recipe set time", color=LABEL, fontsize=10,
                      labelpad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

        if log:
            ax.set_yscale("log")
            ax.set_title(f"{STEP_NAME} — every electrode (log scale)",
                         color=TEXT, fontsize=12, fontweight="bold",
                         loc="left", pad=10)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        else:
            hidden = (df["pct_of_set"] > ZOOM_TOP).sum()
            ax.set_ylim(0, ZOOM_TOP)
            ax.set_title(f"{STEP_NAME} — zoomed to {ZOOM_TOP}% "
                         f"({hidden} point{'s' if hidden != 1 else ''} above)",
                         color=TEXT, fontsize=12, fontweight="bold",
                         loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(path, facecolor=FIG_BG)
    print(f"\n✓ Figure saved to {path.name}")


def main():
    df = load()
    summarise(df)

    columns = ["electrode", "verdict_label", "production_date", "vats",
               "note_corrected", "set_time", "actual_time", "overrun",
               "pct_of_set", "pct_over_set"]
    out = FOLDER / f"step_detail_{STEP_NAME.lower().replace(' ', '_')}.csv"
    df[columns].round(2).to_csv(out, index=False)
    print(f"\n✓ {len(df)} rows saved to {out.name}")

    draw(df, FOLDER / f"step_detail_{STEP_NAME.lower().replace(' ', '_')}.png")


if __name__ == "__main__":
    main()
