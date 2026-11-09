"""Electrodes against how stale the activator bath was when they were made.

The same two lanes as the vat timeline - good above, cracked below, one
hollow circle per electrode - but the x axis is no longer the calendar. It
is `chem_v18_age`: for each electrode, the days between the last Slotoclean
Decasel 5 dose into vat 18 and the moment that electrode was produced. An
electrode made right after a dose sits at 0, one made a fortnight later
sits at 14.

That makes the question readable straight off the axis: are the cracked
circles further right than the sound ones?

Electrodes within 0.1 days of each other stack inside their lane so a busy
shift reads as a column of circles rather than one dark blob.
"""
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from gui_app import (AXES_BG, COLOR_CRACKED, COLOR_GOOD, EMPTY, FIG_BG, GRID,
                     LABEL, SPINE, SUBTEXT, TEXT)

COLUMN = "chem_v18_age"
HEADING = "Electrodes by the age of the last Decasel 5 dose into vat 18"
AXIS = "days since the last Decasel 5 dose into vat 18"
BIN = 0.1     # days; electrodes this close together stack in their lane


class DoseAgeTab(QWidget):

    def __init__(self, frame, column=COLUMN, heading=HEADING, axis=AXIS):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(10)

        title = QLabel(heading)
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        title.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Minimum)
        layout.addWidget(title)

        note = QLabel("Left is a freshly dosed bath, right is a stale one. "
                      "Each circle is one electrode; same-age electrodes "
                      "stack inside their lane.")
        note.setStyleSheet(f"color: {SUBTEXT}; font-size: 12px;")
        note.setWordWrap(True)
        note.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)
        layout.addWidget(note)

        canvas = self._figure(self._points(frame, column), axis)
        canvas.setMinimumHeight(250)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
        layout.addWidget(canvas)
        layout.addStretch(1)

    @staticmethod
    def _points(frame, column):
        out = pd.DataFrame({
            "age": pd.to_numeric(frame.get(column), errors="coerce"),
            "cracked": pd.to_numeric(frame.get("coating_label"),
                                     errors="coerce"),
        }).dropna()
        # 1 = good, 0 = cracked, the lanes used everywhere else
        out["verdict"] = 1 - out["cracked"]
        return out.sort_values("age")

    def _figure(self, points, axis):
        figure = Figure(figsize=(13, 2.5), dpi=100)
        figure.patch.set_facecolor(FIG_BG)
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

        if not len(points):
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no electrodes", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=11, style="italic")
            figure.subplots_adjust(left=0.10, right=0.98, top=0.86,
                                   bottom=0.22)
            return canvas

        good = int((points["verdict"] == 1).sum())
        cracked = int((points["verdict"] == 0).sum())
        step = 0.06
        points = points.assign(slot=(points["age"] / BIN).round())
        for value, colour in ((1, COLOR_GOOD), (0, COLOR_CRACKED)):
            part = points[points["verdict"] == value]
            if not len(part):
                continue
            offsets = part.groupby("slot").cumcount()
            spread = part.groupby("slot")["verdict"].transform("size") - 1
            y = value + (offsets - spread / 2) * step
            ax.plot(part["age"], y, linestyle="none", marker="o",
                    markersize=5.0, markerfacecolor="#FFFFFF",
                    markeredgecolor=colour, markeredgewidth=1.0, zorder=4)

        ax.set_yticks([0, 1], [f"cracked (0)\nn = {cracked}",
                               f"good (1)\nn = {good}"])
        # A stack runs six deep at most, so ±0.18 covers every lane
        ax.set_ylim(-0.30, 1.30)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(-0.6, float(points["age"].max()) + 0.6)
        ax.set_xlabel(axis, color=LABEL, fontsize=10, labelpad=8)
        ax.set_title(f"{len(points)} electrodes, {good} good, "
                     f"{cracked} cracked "
                     f"({100 * cracked / len(points):.0f}%)",
                     color=TEXT, fontsize=11, fontweight="bold", loc="left",
                     pad=6)
        figure.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.22)
        canvas.draw_idle()
        return canvas
