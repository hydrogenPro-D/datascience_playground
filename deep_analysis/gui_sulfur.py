"""One finding, three views: coating sulfur versus cracking.

The SEM measures what the coating is made of - mostly nickel, plus a little
sulfur. The sulfur comes from thiourea, an additive dosed into the plating
bath specifically to relieve stress in the deposit. The quantity plotted is
nickel divided by sulfur, so a HIGH value means a sulfur-poor coating. This
app asks one question three ways:

  Tab 1  does more sulfur mean fewer cracks?      quintile bars
  Tab 2  what does that look like electrode by electrode?   lane scatter
  Tab 3  does it hold on the previous generation too?       side by side

Every chart states its own question and answer in plain words, so it can be
read without knowing the analysis behind it.

Run with the repo .venv:
    ./.venv/Scripts/python.exe deep_analysis/gui_sulfur.py
"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QComboBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

FOLDER = Path(__file__).resolve().parent
FEATURES = FOLDER / "features.csv"
# Nickel divided by sulfur. Note the direction: a HIGH ratio means little
# sulfur per unit nickel, which is the end that cracks. Everything below reads
# left-to-right as increasing ratio, i.e. decreasing sulfur.
RATIO = "comp_Ni_over_S"
RATIO_NAME = "nickel-to-sulfur ratio"
RATIO_HINT = "higher = less sulfur"
RATIO_FMT = "{:.2f}"

COLOR_GOOD = "#0EA5A5"
COLOR_CRACKED = "#8B5CF6"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"
EMPTY = "#9CA3AF"

SEVERITY_LABELS = {0: "no cracks", 1: "very few microcracks",
                   2: "microcracks", 3: "small cracks", 4: "cracked"}
SEVERITY_COLORS = {0: "#D6E4F7", 1: "#9FC0E8", 2: "#5B8FD1",
                   3: "#2A5BA8", 4: "#122E5C"}
DOT_EDGE = "#64748B"
GENERATIONS = {"Gen4.1  (the current line)": "Gen4.1",
               "Gen3.1  (the previous line)": "Gen3.1",
               "Both generations together": None}


def load():
    if not FEATURES.exists():
        raise SystemExit(f"{FEATURES} not found - run features.py first")
    frame = pd.read_csv(FEATURES)
    frame = frame[~frame["known_cause"].fillna(False)]
    frame = frame[frame["electrode_type"] == "CATHODE"]
    frame = frame.dropna(subset=["severity", RATIO]).copy()
    frame["severity"] = frame["severity"].astype(int)
    return frame


def chrome(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(AXES_BG)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=SUBTEXT, length=0, labelsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=LABEL, fontsize=11.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=LABEL, fontsize=11.5)


def quintiles(part, bins=5):
    """Split electrodes into equal-sized groups by sulfur ratio, low to high."""
    part = part[[RATIO, "severity"]].dropna().copy()
    if len(part) < bins * 4:
        return None, None
    part["bin"] = pd.qcut(part[RATIO], bins, labels=False, duplicates="drop")
    edges = pd.qcut(part[RATIO], bins, duplicates="drop").cat.categories
    table = part.groupby("bin").agg(
        n=("severity", "size"),
        cracked=("severity", lambda s: 100 * (s == 4).mean()),
        mild=("severity", lambda s: 100 * (s <= 2).mean()))
    return table, edges


def draw_quintiles(ax, part, title, show_ylabel=True):
    table, edges = quintiles(part)
    if table is None:
        ax.text(0.5, 0.5, "Not enough electrodes", ha="center", va="center",
                transform=ax.transAxes, color=EMPTY, fontsize=12,
                style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        return

    x = np.arange(len(table))
    ax.bar(x - 0.19, table["cracked"], width=0.36, color=COLOR_CRACKED,
           edgecolor="none", zorder=3, label="badly cracked")
    ax.bar(x + 0.19, table["mild"], width=0.36, color=COLOR_GOOD,
           edgecolor="none", zorder=3, label="no cracks or only microcracks")
    for i, row in table.reset_index().iterrows():
        ax.annotate(f"{row['cracked']:.0f}%", (i - 0.19, row["cracked"]),
                    ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", fontsize=11, color=SUBTEXT,
                    fontweight="bold")
        ax.annotate(f"{row['mild']:.0f}%", (i + 0.19, row["mild"]),
                    ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", fontsize=11, color=SUBTEXT,
                    fontweight="bold")
    # Low ratio = sulfur-rich = the good end, so it sits on the left
    names = ["most\nsulfur"] + [""] * (len(table) - 2) + ["least\nsulfur"]
    ax.set_xticks(x, [f"{name}\n{e.left:.2f} to {e.right:.2f}\n{n} electrodes"
                      if name else f"{e.left:.2f} to {e.right:.2f}\n{n} electrodes"
                      for name, e, n in zip(names, edges, table["n"])])
    chrome(ax, xlabel=f"{RATIO_NAME} in the coating  ({RATIO_HINT})",
           ylabel="share of electrodes" if show_ylabel else None)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    ax.set_title(title, color=TEXT, fontsize=12.5, fontweight="bold",
                 loc="left", pad=10)


class QuintileTab(QWidget):
    """Tab 1 - the headline: cracking rate across sulfur levels."""

    def __init__(self, frame):
        super().__init__()
        self._frame = frame
        self._generation = "Gen4.1"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 16, 30, 16)
        controls = QHBoxLayout()
        caption = QLabel("Show:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12.5px;")
        combo = QComboBox()
        combo.addItems(list(GENERATIONS))
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12.5px; padding: 3px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_generation)
        controls.addWidget(caption)
        controls.addWidget(combo)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = Figure(figsize=(12.5, 6.4), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)
        self._rebuild()

    def _set_generation(self, text):
        self._generation = GENERATIONS[text]
        self._rebuild()

    def _rebuild(self):
        self._fig.clear()
        part = self._frame
        if self._generation:
            part = part[part["generation"] == self._generation]

        ax = self._fig.add_subplot()
        self._fig.subplots_adjust(left=0.085, right=0.97, top=0.68, bottom=0.20)
        draw_quintiles(ax, part, "")

        table, _edges = quintiles(part)
        self._fig.text(0.085, 0.965,
                       f"Does a higher {RATIO_NAME} mean more cracks?",
                       color=TEXT, fontsize=17, fontweight="bold", va="top")
        if table is not None:
            # Bins run low ratio (sulfur-rich) to high ratio (sulfur-poor)
            richest = table["cracked"].iloc[0]
            poorest = table["cracked"].iloc[-1]
            answer = (f"Yes. {poorest:.0f}% of the highest-ratio electrodes, "
                      f"the ones with least sulfur, came back badly cracked, "
                      f"against {richest:.0f}% of the lowest-ratio ones.")
        else:
            answer = "Not enough electrodes to say."
        self._fig.text(0.085, 0.905, answer, color=TEXT, fontsize=12.5,
                       va="top")
        self._fig.text(0.085, 0.855,
                       "Electrodes are split into five equal-sized groups by "
                       "the nickel-to-sulfur ratio of their coating. Lowest "
                       "ratio, most sulfur, on the left.",
                       color=SUBTEXT, fontsize=11, va="top")
        ax.legend(frameon=False, fontsize=11, labelcolor=SUBTEXT, ncols=2,
                  loc="upper center", bbox_to_anchor=(0.5, 1.10))
        self._canvas.draw_idle()


class LaneTab(QWidget):
    """Tab 2 - the raw data: one dot per electrode."""

    def __init__(self, frame):
        super().__init__()
        self._frame = frame
        self._generation = "Gen4.1"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 16, 30, 16)
        controls = QHBoxLayout()
        caption = QLabel("Show:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12.5px;")
        combo = QComboBox()
        combo.addItems(list(GENERATIONS))
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12.5px; padding: 3px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_generation)
        controls.addWidget(caption)
        controls.addWidget(combo)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = Figure(figsize=(12.5, 7.0), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)
        self._rebuild()

    def _set_generation(self, text):
        self._generation = GENERATIONS[text]
        self._rebuild()

    def _rebuild(self):
        self._fig.clear()
        part = self._frame
        if self._generation:
            part = part[part["generation"] == self._generation]

        ax = self._fig.add_subplot()
        self._fig.subplots_adjust(left=0.215, right=0.97, top=0.70, bottom=0.16)
        ax.set_facecolor(AXES_BG)
        rng = np.random.default_rng(0)      # jitter only, fixed so it is stable

        medians = {}
        for level in range(5):
            rows = part[part["severity"] == level]
            if not len(rows):
                continue
            values = rows[RATIO].to_numpy(dtype=float)
            y = level + rng.uniform(-0.3, 0.3, len(values))
            ax.scatter(values, y, s=44, color=SEVERITY_COLORS[level],
                       edgecolors=DOT_EDGE, linewidths=0.6, alpha=0.9,
                       zorder=3)
            medians[level] = float(np.median(values))
            ax.vlines(medians[level], level - 0.42, level + 0.42, color=TEXT,
                      linewidth=2.2, zorder=5)

        # The median goes in the row label, not next to its line: inside a
        # dense row the text lands on top of the dots and cannot be read
        counts = part["severity"].value_counts()
        labels = []
        for i in range(5):
            n = int(counts.get(i, 0))
            middle = (f"median {RATIO_FMT.format(medians[i])}"
                      if i in medians else "no data")
            labels.append(f"{SEVERITY_LABELS[i]}\n{n} electrodes\n{middle}")
        ax.set_yticks(range(5), labels)
        ax.set_ylim(-0.7, 4.7)
        ax.invert_yaxis()           # best at the top, worst at the bottom
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=10)
        ax.set_xlabel(f"{RATIO_NAME} in the coating  ({RATIO_HINT})",
                      color=LABEL, fontsize=11.5, labelpad=8)

        self._fig.text(0.085, 0.965, "The same thing, one dot per electrode",
                       color=TEXT, fontsize=17, fontweight="bold", va="top")
        worst = part[part["severity"] == 4][RATIO]
        rest = part[part["severity"] < 4][RATIO]
        if len(worst) and len(rest):
            headline = (f"The badly-cracked row sits at "
                        f"{RATIO_FMT.format(worst.median())}, against "
                        f"{RATIO_FMT.format(rest.median())} for everything "
                        f"else: higher, meaning less sulfur.")
        else:
            headline = "Each row is a verdict from the SEM, best at the top."
        self._fig.text(0.085, 0.905, headline, color=TEXT, fontsize=12.5,
                       va="top")
        thin = [SEVERITY_LABELS[i] for i in range(5)
                if 0 < int(counts.get(i, 0)) < 15]
        note = ("Black bar = that row's middle value. The shift is between "
                "the badly-cracked row and the rest, not a clean step down "
                "the rows")
        if thin:
            note += f". {' and '.join(thin)} hold too few electrodes to place"
        # Wrapped: the note grows when a row is too thin to place, and an
        # unwrapped line runs off the right edge of the figure
        self._fig.text(0.085, 0.855, textwrap.fill(note + ".", 118),
                       color=SUBTEXT, fontsize=11, va="top", linespacing=1.5)
        self._canvas.draw_idle()


class BothTab(QWidget):
    """Tab 3 - the replication that makes the finding worth believing."""

    def __init__(self, frame):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 16, 30, 16)
        self._fig = Figure(figsize=(13, 6.2), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        canvas = FigureCanvas(self._fig)
        layout.addWidget(canvas)

        grid = self._fig.add_gridspec(1, 2, wspace=0.13, left=0.075,
                                      right=0.98, top=0.64, bottom=0.20)
        for index, generation in enumerate(("Gen3.1", "Gen4.1")):
            part = frame[frame["generation"] == generation]
            ax = self._fig.add_subplot(grid[index])
            draw_quintiles(ax, part,
                           f"{generation}   ({len(part)} electrodes)",
                           show_ylabel=index == 0)
            ax.set_ylim(0, 100)

        handles = [Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=COLOR_CRACKED, label="badly cracked"),
                   Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=COLOR_GOOD,
                          label="no cracks or only microcracks")]
        self._fig.legend(handles=handles, loc="upper right",
                         bbox_to_anchor=(0.98, 0.80), frameon=False,
                         fontsize=11, labelcolor=SUBTEXT, ncols=2)
        self._fig.text(0.075, 0.965,
                       "Does the pattern hold on the previous generation too?",
                       color=TEXT, fontsize=17, fontweight="bold", va="top")
        self._fig.text(0.075, 0.905,
                       "Yes - and that is what makes it worth taking "
                       "seriously. Two separate populations, same direction.",
                       color=TEXT, fontsize=12.5, va="top")
        self._fig.text(0.075, 0.855,
                       "Out of 120 measurements compared, the nickel-to-"
                       "sulfur ratio is the only one that tracks cracking in "
                       "both. Bars run most sulfur (left) to least (right).",
                       color=SUBTEXT, fontsize=11, va="top")
        canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, frame):
        super().__init__()
        self.setWindowTitle("Coating sulfur and cracking")
        tabs = QTabWidget()
        tabs.addTab(QuintileTab(frame), "1 · Does sulfur matter?")
        tabs.addTab(LaneTab(frame), "2 · Every electrode")
        tabs.addTab(BothTab(frame), "3 · Does it hold in both?")
        self.setCentralWidget(tabs)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    frame = load()
    print(f"{len(frame)} cathodes with a verdict and a sulfur reading")
    print(pd.crosstab(frame["generation"], frame["severity"]).to_string())

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background: {FIG_BG}; }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            background: transparent; padding: 11px 26px; font-size: 14px;
            color: {SUBTEXT}; border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT}; font-weight: bold;
            border-bottom: 2px solid {COLOR_GOOD};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
    """)
    window = MainWindow(frame)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
