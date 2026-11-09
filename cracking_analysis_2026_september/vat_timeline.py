"""One vat on one time axis: what was dosed into it, what it measured, and
which electrodes came out cracked.

Built for the two features the forest kept picking up - `chem_v18_age` and
`chem_s_nicl2_age`, both "days since this vat was last dosed". A number like
that says nothing about whether the dose actually moved the bath, or whether
cracking followed. Five panels on a shared date axis do.

    additions      one stem per dose, labelled with the kilos
    one bath       whichever of the three measurements the dropdown
                   selects. One at a time rather than all at once: pH runs
                   2-8 and Decasel 5 to 65 g/L, so on a shared axis the
                   flatter series is a straight line
    electrodes     two lanes of hollow circles, good above cracked, the
                   same shape as the verdict panels in gui_app so the two
                   read alike; same-day electrodes stack inside their lane

The dashed vertical through every panel is a dose. That is the whole point
of the layout: it lets you follow one dose downward and see whether the
bath responded and whether the electrodes after it cracked.

Nothing is smoothed and nothing is interpolated. The bath is sampled every
few days, so its line is drawn between the samples that exist - a straight
segment there is an absence of measurement, not a steady bath.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QScrollArea, QFrame, QSizePolicy)
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui_app import (AXES_BG, COLOR_CRACKED, COLOR_GOOD, EMPTY, FIG_BG, GRID,
                     LABEL, SPINE, SUBTEXT, TEXT)

FOLDER = Path(__file__).resolve().parent
CHEM_FILE = FOLDER / "chemical_additions_long.csv"
DOSE_LINE = "#B9A7E8"

# (title, vat, chemical dosed, bath log, [(column, axis label)])
TIMELINES = {
    "Vat 18 activation": {
        "vat": "18",
        "chemical": "Slotoclean Decasel 5",
        "log": "activator_bath_long.csv",
        "series": [("pH Mean", "pH"),
                   ("Decasel 5 [g/L] Mean", "Decasel 5 [g/L]"),
                   ("Ni2+ [g/L] Mean", "Ni2+ [g/L]")],
    },
    "Strike vat 31": {
        "vat": "31",
        "chemical": "NiCl2*6H2O",
        "log": "strike_bath_long.csv",
        # Nickel and HCl both, as asked: NiCl2 supplies the nickel and the
        # chloride, so a dose should show in one and the acid in the other
        "series": [("pH Mean", "pH"),
                   ("Ni2+ [g/L] Mean", "Ni2+ [g/L]"),
                   ("avg. ml 36% HCl/L", "HCl 36% [ml/L]")],
    },
}


def load_additions(vat, chemical):
    if not CHEM_FILE.exists():
        return pd.DataFrame(columns=["added_at", "amount", "unit"])
    chems = pd.read_csv(CHEM_FILE, parse_dates=["added_at"])
    chems["vat"] = chems["vat"].astype(str)
    part = chems[(chems["vat"] == str(vat))
                 & (chems["chemical"].astype(str).str.strip() == chemical)]
    return part.sort_values("added_at")


def load_bath(name):
    path = FOLDER / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["measured_at"]).sort_values(
        "measured_at")


class VatTimelineTab(QWidget):
    """Dosing, one chosen bath measurement, and the electrodes."""

    def __init__(self, frame, which):
        super().__init__()
        spec = TIMELINES[which]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {FIG_BG}; "
                             f"border: none; }}")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        title = QLabel(f"{which}: dosing, bath and electrodes on one axis")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        title.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Minimum)
        layout.addWidget(title)

        additions = load_additions(spec["vat"], spec["chemical"])
        self._bath = load_bath(spec["log"])
        electrodes = self._electrodes(frame)
        choices = list(spec["series"])

        note = QLabel(
            f"Dashed verticals are {spec['chemical']} doses into vat "
            f"{spec['vat']}. The bath is sampled every few days, so a "
            f"straight run in a measurement panel is a gap between samples, "
            f"not a steady bath.")
        note.setStyleSheet(f"color: {SUBTEXT}; font-size: 12px;")
        note.setWordWrap(True)
        note.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)
        layout.addWidget(note)

        picker = QHBoxLayout()
        picker.setSpacing(8)
        caption = QLabel("Bath measurement:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        picker.addWidget(caption)
        self._combo = QComboBox()
        for column, text in choices:
            self._combo.addItem(text, column)
        self._combo.setMinimumWidth(240)
        self._combo.currentIndexChanged.connect(self._choose)
        picker.addWidget(self._combo)
        picker.addStretch(1)
        holder = QWidget()
        holder.setLayout(picker)
        holder.setSizePolicy(QSizePolicy.Policy.Preferred,
                             QSizePolicy.Policy.Minimum)
        layout.addWidget(holder)

        canvas = self._figure(spec, additions, self._bath, electrodes,
                              choices[0] if choices else ("", ""))
        canvas.setMinimumHeight(560)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas, 1)

    def _choose(self, index):
        """Redraw the middle panel only; the other two do not change."""
        column = self._combo.itemData(index)
        if column is None:
            return
        ax = self._series_ax
        ax.clear()
        self._draw_series(ax, self._bath, column, self._combo.itemText(index),
                          self._first, self._last)
        ax.set_xlim(self._first, self._last)
        for when in self._doses:
            ax.axvline(when, color=DOSE_LINE, linewidth=1.1,
                       linestyle=(0, (4, 3)), zorder=1)
        ax.tick_params(labelbottom=False)
        # clear() on a shared axis wipes the date locator off the whole
        # group, so the bottom panel needs its ticks put back
        self._style_dates(self._bottom_ax)
        self._canvas.draw_idle()

    @staticmethod
    def _electrodes(frame):
        out = pd.DataFrame({
            "date": pd.to_datetime(frame.get("date"), errors="coerce"),
            "cracked": pd.to_numeric(frame.get("coating_label"),
                                     errors="coerce"),
        }).dropna()
        # 1 = good, 0 = cracked, matching the verdict lanes elsewhere
        out["verdict"] = 1 - out["cracked"]
        return out.sort_values("date")

    def _figure(self, spec, additions, bath, electrodes, first_choice):
        figure = Figure(figsize=(13, 6.6), dpi=100)
        figure.patch.set_facecolor(FIG_BG)
        canvas = FigureCanvas(figure)
        grid = figure.add_gridspec(3, 1, hspace=0.30,
                                   height_ratios=[1.0, 2.05, 1.0])

        # The window the electrodes live in, with a little air either side
        if len(electrodes):
            first = electrodes["date"].min() - pd.Timedelta(days=5)
            last = electrodes["date"].max() + pd.Timedelta(days=3)
        elif len(bath):
            first, last = bath["measured_at"].min(), bath["measured_at"].max()
        else:
            first, last = pd.Timestamp("2026-07-01"), pd.Timestamp.today()

        doses = additions[(additions["added_at"] >= first)
                          & (additions["added_at"] <= last)]
        axes = []

        ax = figure.add_subplot(grid[0])
        self._draw_additions(ax, doses, spec)
        axes.append(ax)
        ax = figure.add_subplot(grid[1], sharex=axes[0])
        self._draw_series(ax, bath, first_choice[0], first_choice[1],
                          first, last)
        self._series_ax = ax
        axes.append(ax)
        ax = figure.add_subplot(grid[2], sharex=axes[0])
        self._draw_electrodes(ax, electrodes)
        axes.append(ax)

        for ax in axes:
            ax.set_xlim(first, last)
            for when in doses["added_at"]:
                ax.axvline(when, color=DOSE_LINE, linewidth=1.1,
                           linestyle=(0, (4, 3)), zorder=1)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        self._style_dates(axes[-1])
        self._bottom_ax = axes[-1]
        figure.subplots_adjust(left=0.10, right=0.98, top=0.955,
                               bottom=0.10)
        self._canvas = canvas
        self._doses = list(doses["added_at"])
        self._first, self._last = first, last
        canvas.draw_idle()
        return canvas

    @staticmethod
    def _style_dates(ax):
        """A tick every few days rather than every fortnight, so a dose can
        be placed on the calendar without counting gridlines."""
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        for item in ax.get_xticklabels():
            item.set_rotation(45)
            item.set_horizontalalignment("right")
            item.set_fontsize(8)

    @staticmethod
    def _chrome(ax, ylabel):
        ax.set_facecolor(AXES_BG)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _draw_additions(self, ax, doses, spec):
        self._chrome(ax, f"dosed [{doses['unit'].iloc[0]}]"
                     if len(doses) else "dosed")
        if not len(doses):
            ax.text(0.5, 0.5, f"no {spec['chemical']} doses in this window",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=11, style="italic")
            return
        ax.vlines(doses["added_at"], 0, doses["amount"],
                  color=COLOR_CRACKED, linewidth=2.4, zorder=3)
        ax.plot(doses["added_at"], doses["amount"], "o", markersize=6,
                color=COLOR_CRACKED, zorder=4)
        for when, amount in zip(doses["added_at"], doses["amount"]):
            ax.annotate(f"{amount:g}", (when, amount),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8.5, color=SUBTEXT)
        ax.set_ylim(0, float(doses["amount"].max()) * 1.35)
        ax.set_title(f"{spec['chemical']} into vat {spec['vat']}",
                     color=TEXT, fontsize=11, fontweight="bold", loc="left",
                     pad=6)

    def _draw_series(self, ax, bath, column, label_text, first, last):
        self._chrome(ax, label_text)
        if not len(bath) or column not in bath.columns:
            ax.text(0.5, 0.5, f"{column} not in the log",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=11, style="italic")
            return
        part = bath[["measured_at", column]].copy()
        part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna()
        window = part[(part["measured_at"] >= first)
                      & (part["measured_at"] <= last)]
        if not len(window):
            ax.text(0.5, 0.5, "no readings in this window",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=11, style="italic")
            return
        ax.plot(window["measured_at"], window[column], "-o", markersize=4.5,
                linewidth=1.5, color=COLOR_GOOD, zorder=3)
        ax.set_title(label_text, color=TEXT, fontsize=11, fontweight="bold",
                     loc="left", pad=6)
        ax.annotate(f"{len(window)} readings",
                    (0.995, 0.92), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color=SUBTEXT)

    def _draw_electrodes(self, ax, electrodes):
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
        if not len(electrodes):
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no electrodes", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=11, style="italic")
            return

        good = int((electrodes["verdict"] == 1).sum())
        cracked = int((electrodes["verdict"] == 0).sum())
        # Same-day electrodes stack within their lane, and the markers are
        # hollow so a dense run reads as a column of circles rather than
        # filling in as a solid bar
        step = 0.06
        for value, colour in ((1, COLOR_GOOD), (0, COLOR_CRACKED)):
            part = electrodes[electrodes["verdict"] == value].sort_values(
                "date")
            if not len(part):
                continue
            offsets = part.groupby("date").cumcount()
            spread = part.groupby("date")["verdict"].transform("size") - 1
            y = value + (offsets - spread / 2) * step
            ax.plot(part["date"], y, linestyle="none", marker="o",
                    markersize=5.0, markerfacecolor="#FFFFFF",
                    markeredgecolor=colour, markeredgewidth=1.0, zorder=4)

        ax.set_yticks([0, 1], [f"cracked (0)\nn = {cracked}",
                               f"good (1)\nn = {good}"])
        ax.set_ylim(-0.42, 1.42)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(f"{len(electrodes)} electrodes, {good} good, "
                     f"{cracked} cracked "
                     f"({100 * cracked / len(electrodes):.0f}%)",
                     color=TEXT, fontsize=11, fontweight="bold", loc="left",
                     pad=6)
