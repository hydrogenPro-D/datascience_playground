"""Strike temperature vs cracking — a focused look at the one signal that
showed anything.

The temperature barely moves during a visit (median range 0.05 C), so there
is no curve to analyse: every electrode is one number, and the question is
how those numbers differ between cracked and not cracked, and whether that
difference holds up once time is taken into account.

Escape closes the window.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_measurements_filtered.csv"
SIGNALS_FILE = FOLDER / "signals.csv"

COLOR_GOOD = "#0EA5A5"
COLOR_CRACKED = "#8B5CF6"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"
MARK = "#111827"

PLOT_HEIGHT = 6.5


def load():
    sem = pd.read_csv(SEM_FILE).dropna(subset=["sem_result", "production_date"])
    sem["cracked"] = (sem["sem_result"] == "bad").astype(int)
    sem["production_date"] = pd.to_datetime(sem["production_date"])
    sem["plating_vat"] = sem["vats"].astype(str).str.split("|").str[0]

    signals = pd.read_csv(SIGNALS_FILE)
    signals["Marking"] = signals["Marking"].astype(str).str.strip()
    signals["Vat"] = signals["Vat"].astype(str)
    temp = signals[signals["signal"] == "Temperature"]

    strike = temp[temp["Vat"] == "31"].set_index("Marking")
    frame = sem[["electrode", "cracked", "production_date", "plating_vat"]].copy()
    frame["strike_temp"] = frame["electrode"].map(strike["value_mean"])
    frame["strike_min"] = frame["electrode"].map(strike["value_min"])
    frame["strike_max"] = frame["electrode"].map(strike["value_max"])
    frame["strike_readings"] = frame["electrode"].map(strike["readings"])
    frame["strike_range"] = frame["strike_max"] - frame["strike_min"]

    plating = temp.set_index(["Marking", "Vat"])["value_mean"]
    frame["plating_temp"] = [plating.get((e, v)) for e, v
                             in zip(frame["electrode"], frame["plating_vat"])]

    frame = frame.dropna(subset=["strike_temp"])
    frame["month"] = frame["production_date"].dt.to_period("M").astype(str)
    return frame


def auc(cracked, values):
    good, bad = values[cracked == 0], values[cracked == 1]
    if not len(good) or not len(bad):
        return np.nan
    u = stats.mannwhitneyu(bad, good).statistic
    return u / (len(good) * len(bad))


def wilson(successes, total, z=1.96):
    if not total:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total
                         + z ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def chrome(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=LABEL, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def canvas_tab(draw):
    """Build a tab whose only content is one figure drawn by `draw(fig)`."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(24, 14, 24, 14)
    figure = Figure(figsize=(11.5, PLOT_HEIGHT), dpi=100)
    figure.patch.set_facecolor(FIG_BG)
    draw(figure)
    layout.addWidget(FigureCanvas(figure))
    return page


def legend(fig, extra=()):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               color=COLOR_GOOD, label="not cracked"),
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               color=COLOR_CRACKED, label="cracked"),
        *extra]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles),
               frameon=False, fontsize=9, labelcolor=SUBTEXT,
               bbox_to_anchor=(0.99, 0.995))


def distribution_tab(df):
    def draw(fig):
        ax_strip, ax_hist = fig.subplots(1, 2)
        rng = np.random.default_rng(0)

        for i, (name, colour) in enumerate((("not cracked", COLOR_GOOD),
                                            ("cracked", COLOR_CRACKED))):
            part = df[df["cracked"] == i]["strike_temp"]
            ax_strip.scatter(i + rng.uniform(-0.16, 0.16, len(part)), part,
                             s=26, color=colour, alpha=0.55,
                             edgecolors="none", zorder=3)
            ax_strip.hlines(part.mean(), i - 0.3, i + 0.3, color=MARK,
                            linewidth=1.6, linestyle=(0, (4, 2)), zorder=5)
            ax_strip.hlines(part.median(), i - 0.3, i + 0.3, color=MARK,
                            linewidth=2.2, zorder=5)
            ax_strip.annotate(f"mean {part.mean():.2f}  median "
                              f"{part.median():.2f}", (i, part.max()),
                              textcoords="offset points", xytext=(0, 10),
                              ha="center", fontsize=8.5, color=SUBTEXT)
            ax_hist.hist(part, bins=np.arange(33, 48, 0.5), color=colour,
                         alpha=0.55, label=name, zorder=3, density=True)

        ax_strip.set_xticks([0, 1], [
            f"not cracked\nn = {(df['cracked'] == 0).sum()}",
            f"cracked\nn = {(df['cracked'] == 1).sum()}"])
        ax_strip.set_xlim(-0.6, 1.6)
        chrome(ax_strip, "Strike temperature per electrode", ylabel="°C")
        chrome(ax_hist, "Same data as distributions", xlabel="°C",
               ylabel="share of group")

        good = df[df["cracked"] == 0]["strike_temp"]
        bad = df[df["cracked"] == 1]["strike_temp"]
        p = stats.mannwhitneyu(bad, good).pvalue
        fig.text(0.02, 0.02,
                 f"difference of means {bad.mean() - good.mean():+.2f} °C   ·   "
                 f"Mann-Whitney p = {p:.4f}   ·   "
                 f"AUC = {auc(df['cracked'], df['strike_temp']):.3f}   ·   "
                 f"within-visit range median "
                 f"{df['strike_range'].median():.2f} °C (flat, nothing to model)",
                 fontsize=9, color=SUBTEXT)
        legend(fig)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16,
                            wspace=0.22)
    return canvas_tab(draw)


def timeline_tab(df):
    def draw(fig):
        ax_temp, ax_rate = fig.subplots(2, 1, sharex=True)

        for value, colour in ((0, COLOR_GOOD), (1, COLOR_CRACKED)):
            part = df[df["cracked"] == value]
            ax_temp.scatter(part["production_date"], part["strike_temp"],
                            s=26, color=colour, alpha=0.6, edgecolors="none",
                            zorder=3)
        rolling = (df.sort_values("production_date")
                   .set_index("production_date")["strike_temp"]
                   .rolling("7D").mean())
        ax_temp.plot(rolling.index, rolling.values, color=MARK, linewidth=1.8,
                     zorder=5)
        chrome(ax_temp, "Strike temperature over the production window",
               ylabel="°C")

        weekly = (df.set_index("production_date")
                  .resample("W")["cracked"].agg(["size", "sum"]))
        weekly = weekly[weekly["size"] > 0]
        rate = weekly["sum"] / weekly["size"]
        ax_rate.bar(weekly.index, rate, width=5, color=COLOR_CRACKED,
                    alpha=0.55, zorder=3)
        for when, value, n in zip(weekly.index, rate, weekly["size"]):
            ax_rate.annotate(f"{n}", (when, value), textcoords="offset points",
                             xytext=(0, 3), ha="center", fontsize=7.5,
                             color=SUBTEXT)
        chrome(ax_rate, "Cracked share per week (n above each bar)",
               ylabel="cracked")
        ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

        legend(fig, [Line2D([], [], color=MARK, linewidth=1.8,
                            label="7-day mean")])
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08,
                            hspace=0.35)
    return canvas_tab(draw)


def risk_tab(df):
    def draw(fig):
        ax = fig.add_subplot()
        edges = np.arange(np.floor(df["strike_temp"].min()),
                          np.ceil(df["strike_temp"].max()) + 1, 1.0)
        binned = pd.cut(df["strike_temp"], edges)
        grouped = df.groupby(binned, observed=True)["cracked"].agg(
            ["size", "sum"])
        grouped = grouped[grouped["size"] >= 3]

        centres = [interval.mid for interval in grouped.index]
        rate = grouped["sum"] / grouped["size"]
        bounds = [wilson(s, n) for s, n in zip(grouped["sum"], grouped["size"])]
        lower = [rate.iloc[i] - b[0] for i, b in enumerate(bounds)]
        upper = [b[1] - rate.iloc[i] for i, b in enumerate(bounds)]

        ax.bar(centres, rate, width=0.85, color=COLOR_CRACKED, alpha=0.5,
               zorder=3)
        ax.errorbar(centres, rate, yerr=[lower, upper], fmt="none",
                    ecolor=MARK, elinewidth=1.4, capsize=4, zorder=5)
        for centre, value, n in zip(centres, rate, grouped["size"]):
            ax.annotate(f"n={n}", (centre, 0), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color=SUBTEXT)
        ax.axhline(df["cracked"].mean(), color=MARK, linewidth=1.4,
                   linestyle=(0, (4, 2)), zorder=4)

        chrome(ax, "Cracked share by strike temperature (95% intervals)",
               xlabel="°C", ylabel="share cracked")
        fig.text(0.02, 0.02,
                 "dashed line = overall cracked share. Bins with fewer than "
                 "3 electrodes are omitted; wide intervals mean the bin is "
                 "too small to read anything into.",
                 fontsize=9, color=SUBTEXT)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.16)
    return canvas_tab(draw)


def stability_tab(df):
    def draw(fig):
        ax_auc, ax_scatter = fig.subplots(1, 2)
        months = sorted(df["month"].unique())

        values, labels = [], []
        for month in months:
            part = df[df["month"] == month]
            values.append(auc(part["cracked"], part["strike_temp"]))
            labels.append(f"{month}\nn = {len(part)} / "
                          f"{int(part['cracked'].sum())} cracked")
        colours = [COLOR_CRACKED if v > 0.5 else COLOR_GOOD for v in values]
        ax_auc.bar(range(len(months)), values, width=0.5, color=colours,
                   alpha=0.7, zorder=3)
        for i, value in enumerate(values):
            ax_auc.annotate(f"{value:.2f}", (i, value),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=9, color=SUBTEXT)
        ax_auc.axhline(0.5, color=MARK, linewidth=1.4, zorder=4)
        ax_auc.set_xticks(range(len(months)), labels)
        ax_auc.set_ylim(0, 1)
        chrome(ax_auc, "Does hotter mean cracked? (AUC per month)",
               ylabel="AUC — 0.5 is no separation")

        for value, colour in ((0, COLOR_GOOD), (1, COLOR_CRACKED)):
            part = df[df["cracked"] == value]
            ax_scatter.scatter(part["month"], part["strike_temp"], s=26,
                               color=colour, alpha=0.55, edgecolors="none",
                               zorder=3)
        for i, month in enumerate(months):
            for value, offset in ((0, -0.18), (1, 0.18)):
                part = df[(df["month"] == month) & (df["cracked"] == value)]
                if len(part):
                    ax_scatter.hlines(part["strike_temp"].median(),
                                      i + offset - 0.16, i + offset + 0.16,
                                      color=MARK, linewidth=2.0, zorder=5)
        chrome(ax_scatter, "Strike temperature by month (median marked)",
               ylabel="°C")

        legend(fig)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14,
                            wspace=0.22)
    return canvas_tab(draw)


def pairing_tab(df):
    def draw(fig):
        ax_pair, ax_plating = fig.subplots(1, 2)
        usable = df.dropna(subset=["plating_temp"])

        for value, colour in ((0, COLOR_GOOD), (1, COLOR_CRACKED)):
            part = usable[usable["cracked"] == value]
            ax_pair.scatter(part["strike_temp"], part["plating_temp"], s=30,
                            color=colour, alpha=0.6, edgecolors="none",
                            zorder=3)
        chrome(ax_pair, "Strike vs plating temperature",
               xlabel="strike °C", ylabel="plating °C")

        rng = np.random.default_rng(1)
        for i, colour in ((0, COLOR_GOOD), (1, COLOR_CRACKED)):
            part = usable[usable["cracked"] == i]["plating_temp"]
            ax_plating.scatter(i + rng.uniform(-0.16, 0.16, len(part)), part,
                               s=26, color=colour, alpha=0.55,
                               edgecolors="none", zorder=3)
            ax_plating.hlines(part.median(), i - 0.3, i + 0.3, color=MARK,
                              linewidth=2.2, zorder=5)
        ax_plating.set_xticks([0, 1], ["not cracked", "cracked"])
        ax_plating.set_xlim(-0.6, 1.6)
        chrome(ax_plating, "Plating temperature, for comparison", ylabel="°C")

        fig.text(0.02, 0.02,
                 f"strike AUC {auc(usable['cracked'], usable['strike_temp']):.3f}"
                 f"   ·   plating AUC "
                 f"{auc(usable['cracked'], usable['plating_temp']):.3f}"
                 f"   ·   {len(usable)} electrodes with both",
                 fontsize=9, color=SUBTEXT)
        legend(fig)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16,
                            wspace=0.22)
    return canvas_tab(draw)


class MainWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Strike temperature analysis")
        tabs = QTabWidget()
        tabs.addTab(distribution_tab(df), "Distribution")
        tabs.addTab(timeline_tab(df), "Over time")
        tabs.addTab(risk_tab(df), "Risk vs temperature")
        tabs.addTab(stability_tab(df), "Stability by month")
        tabs.addTab(pairing_tab(df), "Strike vs plating")
        self.setCentralWidget(tabs)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    df = load()
    print(f"{len(df)} electrodes with a strike temperature, "
          f"{int(df['cracked'].sum())} cracked")

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background: {FIG_BG}; }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{ background: transparent; padding: 10px 24px;
                        font-size: 14px; color: {SUBTEXT};
                        border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold;
                                 border-bottom: 2px solid {COLOR_GOOD}; }}
    """)
    window = MainWindow(df)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
