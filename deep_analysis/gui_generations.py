"""Gen3.1 versus Gen4.1 across every recorded parameter.

Tab 1  Good vs cracked   the binary verdict the old pipeline used
Tab 2  Crack severity     the full 0-4 scale from the SEM note, light to dark
Tab 3  Parameter ranking  every parameter scored on both halves of a causal
                          story, sortable, double-click to plot it

Tabs 1 and 2 are two scatter panels sharing one y axis, Gen3.1 on the left and
Gen4.1 on the right, with the chosen parameter up the y axis and production
date along the x. The generations ran in different periods so their x ranges
cannot be shared; the y axis is shared on purpose, because comparing the two
bands is the point.

Reading tab 3: Gen4.1 cracks far more than Gen3.1, so anything that CAUSED
that has to differ between the generations - but hundreds of things differ,
because the two ran consecutively and the whole recipe changed. A parameter is
only a real candidate if it also tracks severity WITHIN a generation, where
nothing else is changing alongside it. The table gives both numbers; neither
one alone is evidence.

Run with the repo .venv:
    ./.venv/Scripts/python.exe deep_analysis/gui_generations.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QCheckBox, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
import matplotlib.dates as mdates

FOLDER = Path(__file__).resolve().parent
FEATURES = FOLDER / "features.csv"

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

SEVERITY_ORDER = [0, 1, 2, 3, 4]
SEVERITY_LABELS = {0: "sound", 1: "very few microcracks", 2: "microcracks",
                   3: "small cracks", 4: "cracked"}
# Severity is a magnitude, so one hue light to dark rather than five competing
# colours; dot size carries the same order so it never rides on colour alone.
SEVERITY_COLORS = {0: "#D6E4F7", 1: "#9FC0E8", 2: "#5B8FD1",
                   3: "#2A5BA8", 4: "#122E5C"}
SEVERITY_SIZES = {0: 26, 1: 32, 2: 38, 3: 46, 4: 56}
DOT_EDGE = "#64748B"

# Step numbers to what the step is, from the recipe groupings the earlier
# pipeline used
RINSE_STEPS = {12, 13, 14, 19, 20, 21, 22, 23, 24, 25}
DRIP_STEPS = {15, 16, 26, 27}
PLATE_STEPS = {28, 29, 30}
STRIKE_STEP = 31
SUFFIX_NAMES = {"actual": ("time in step", "s"), "over": ("overrun", "s"),
                "amp": ("charge", "Ah"), "temp": ("temperature", "°C")}
META_NAMES = {
    "meta_alarm_s": ("Alarm time on this electrode", "s"),
    "meta_manstop_s": ("Manual stop time", "s"),
    "meta_alarm_plus_stop": ("Alarm + manual stop", "s"),
    "meta_start_tid": ("Start_tid counter", ""),
    "meta_vb_total": ("Cumulative electrode number", ""),
    "meta_vb_day": ("Position in the day's run", ""),
    "meta_flight_bar": ("Flight bar number", ""),
    "meta_recipe": ("Recipe number", ""),
    "meta_hour": ("Hour of day started", "h"),
    "meta_weekday": ("Weekday started", "0=Mon"),
    "route_steps": ("Steps in the route", ""),
    "route_set": ("Whole route, recipe time", "s"),
    "route_actual": ("Whole route, actual time", "s"),
    "route_overrun": ("Whole route, total overrun", "s"),
    "route_overrun_max": ("Worst single-step overrun", "s"),
    "route_overrun_pct": ("Whole route, overrun", "%"),
    "route_amp_total": ("Whole route, total charge", "Ah"),
    "route_temp_max": ("Hottest step on the route", "°C"),
    "plate_amp_total": ("Plating charge", "Ah"),
    "plate_seconds": ("Plating time", "s"),
    "plate_temp_mean": ("Plating temperature", "°C"),
    "plate_amp_rate": ("Plating charge per second", "Ah/s"),
    "sem_points": ("SEM points measured", ""),
    "sem_positions": ("SEM positions measured", ""),
}
ELEMENT_NAMES = {"Ni": "nickel", "S": "sulfur", "C": "carbon", "N": "nitrogen",
                 "Co": "cobalt", "O": "oxygen"}


def pretty(column):
    """Human-readable name and unit for a feature column."""
    if column in META_NAMES:
        return META_NAMES[column]
    if column.startswith("comp_"):
        rest = column[len("comp_"):]
        if rest in ("Ni_over_S", "S_over_Ni"):
            a, b = rest.split("_over_")
            return f"Coating {ELEMENT_NAMES[a]}/{ELEMENT_NAMES[b]} ratio", ""
        if rest == "Ni_plus_S":
            return "Coating nickel + sulfur", "%"
        if rest.endswith("_sd"):
            element = rest[:-3]
            return (f"Coating {ELEMENT_NAMES.get(element, element)}, "
                    f"spread across points"), "%"
        return f"Coating {ELEMENT_NAMES.get(rest, rest)}", "%"
    if column.startswith("step_"):
        parts = column.split("_")
        step, pass_no, suffix = int(parts[1]), int(parts[2][1:]), parts[3]
        if step in RINSE_STEPS:
            what = f"Rinse {step}"
        elif step in DRIP_STEPS:
            what = f"Drip {step}"
        elif step in PLATE_STEPS:
            what = f"Plate vat {step}"
        elif step == STRIKE_STEP:
            what = "Strike (31)"
        else:
            what = f"Step {step}"
        name, unit = SUFFIX_NAMES.get(suffix, (suffix, ""))
        tail = "" if pass_no == 1 else f" pass {pass_no}"
        return f"{what}{tail} — {name}", unit
    return column, ""


def load():
    if not FEATURES.exists():
        raise SystemExit(f"{FEATURES} not found - run features.py first")
    frame = pd.read_csv(FEATURES)
    frame = frame[~frame["known_cause"].fillna(False)]
    frame = frame.dropna(subset=["severity", "generation"]).copy()
    frame["severity"] = frame["severity"].astype(int)
    # The binary label the earlier pipeline used: only a bare "Cracked" or
    # "Large cracks" note counted as bad, every milder wording as good.
    frame["binary"] = np.where(frame["severity"] == 4, "cracked", "good")
    frame["run_date"] = pd.to_datetime(frame["meta_started_at"], utc=True,
                                       errors="coerce")
    return frame.dropna(subset=["run_date"])


NEVER_PLOT = {"severity", "binary", "known_cause", "run_date", "electrode",
              "generation", "electrode_type", "note", "operator",
              "measured_on", "meta_started_at", "meta_date", "severity_name",
              "sem_result"}


def plottable(frame, minimum=15):
    """Numeric columns with enough data in BOTH generations to compare."""
    g41 = frame[frame["generation"] == "Gen4.1"]
    g31 = frame[frame["generation"] == "Gen3.1"]
    out = []
    for column in frame.columns:
        if column in NEVER_PLOT or not pd.api.types.is_numeric_dtype(
                frame[column]):
            continue
        if (g41[column].notna().sum() < minimum
                or g31[column].notna().sum() < minimum):
            continue
        if frame[column].nunique(dropna=True) < 2:
            continue
        out.append(column)
    return sorted(out, key=lambda c: pretty(c)[0])


def rank_parameters(frame, kind="CATHODE"):
    """Score every parameter on both halves of a causal story.

    between  rank-biserial effect, Gen4.1 vs Gen3.1: +1 means Gen4.1 is always
             higher, -1 always lower. Large means the recipe change moved it.
    rho41    Spearman correlation with severity inside Gen4.1
    rho31    the same inside Gen3.1, an independent replication
    """
    part = frame if kind is None else frame[frame["electrode_type"] == kind]
    g41 = part[part["generation"] == "Gen4.1"]
    g31 = part[part["generation"] == "Gen3.1"]
    rows = []
    for column in plottable(part):
        a, b = g41[column].dropna(), g31[column].dropna()
        if len(a) < 15 or len(b) < 15:
            continue
        try:
            u, p_between = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = 2 * u / (len(a) * len(b)) - 1
        except ValueError:
            effect, p_between = np.nan, np.nan
        row = {"column": column, "name": pretty(column)[0],
               "unit": pretty(column)[1], "n41": len(a), "n31": len(b),
               "med41": a.median(), "med31": b.median(),
               "between": effect, "p_between": p_between}
        for tag, group in (("41", g41), ("31", g31)):
            sub = group[[column, "severity"]].dropna()
            if len(sub) >= 20 and sub[column].nunique() >= 3:
                rho, p = stats.spearmanr(sub[column], sub["severity"])
            else:
                rho, p = np.nan, np.nan
            row[f"rho{tag}"], row[f"p{tag}"] = rho, p
        rows.append(row)
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    # Candidate = moved between generations AND tracks severity within one
    table["candidate"] = (table["between"].abs().fillna(0)
                          * table[["rho41", "rho31"]].abs().max(axis=1).fillna(0))
    return table.sort_values("candidate", ascending=False).reset_index(drop=True)


class Hover:
    """Nearest-point tooltip across every axes on a canvas."""

    RADIUS = 18

    def __init__(self, canvas):
        self._canvas = canvas
        self._series = []
        self._shown = None
        self._ann = Annotation(
            "", xy=(0, 0), xycoords="figure pixels", xytext=(14, 14),
            textcoords="offset pixels", fontsize=9, color="#FFFFFF", zorder=30,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=TEXT,
                      edgecolor="none", alpha=0.95))
        self._ann.set_visible(False)
        canvas.figure.add_artist(self._ann)
        canvas.mpl_connect("motion_notify_event", self._move)
        canvas.mpl_connect("figure_leave_event", lambda _e: self._hide())

    def reset(self):
        self._series = []
        self._canvas.figure.add_artist(self._ann)
        self._ann.set_visible(False)
        self._shown = None

    def add(self, ax, x, y, labels, color):
        self._series.append((ax, np.asarray(x, dtype=float),
                             np.asarray(y, dtype=float), list(labels), color))

    def _hide(self):
        if self._shown is not None:
            self._shown = None
            self._ann.set_visible(False)
            self._canvas.draw_idle()

    def _move(self, event):
        if event.inaxes is None or event.x is None:
            self._hide()
            return
        best = None
        for ax, xs, ys, labels, color in self._series:
            if ax is not event.inaxes or not len(xs):
                continue
            points = ax.transData.transform(np.column_stack([xs, ys]))
            distance = np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
            i = int(np.argmin(distance))
            if distance[i] <= self.RADIUS and (best is None
                                               or distance[i] < best[0]):
                best = (distance[i], points[i], labels[i], color)
        if best is None:
            self._hide()
            return
        _d, point, label, color = best
        key = (label, tuple(point))
        if key == self._shown:
            return
        self._shown = key
        width, height = (self._canvas.figure.bbox.width,
                         self._canvas.figure.bbox.height)
        dx = 14 if point[0] < width - 210 else -14
        dy = 14 if point[1] < height - 70 else -14
        self._ann.set_position((dx, dy))
        self._ann.set_ha("left" if dx > 0 else "right")
        self._ann.set_va("bottom" if dy > 0 else "top")
        self._ann.xy = (point[0], point[1])
        self._ann.set_text(label)
        self._ann.get_bbox_patch().set_facecolor(color)
        self._ann.set_visible(True)
        self._canvas.draw_idle()


def style_combo(combo):
    combo.setStyleSheet(
        f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
        f"border: 1px solid {SPINE}; border-radius: 4px; "
        f"background: {AXES_BG}; }}")
    return combo


class GenerationScatter(QWidget):
    """Two scatter panels, one per generation, coloured by outcome."""

    def __init__(self, frame, mode, columns, default=None):
        super().__init__()
        self._frame = frame
        self._mode = mode
        self._columns = columns
        self._column = default if default in columns else columns[0]
        self._kind = "CATHODE"
        self._medians = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 14, 26, 14)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Parameter:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        self._combo = style_combo(QComboBox())
        self._combo.setMinimumWidth(430)
        for column in columns:
            name, unit = pretty(column)
            self._combo.addItem(f"{name}{f'  ({unit})' if unit else ''}", column)
        self._combo.setCurrentIndex(columns.index(self._column))
        self._combo.currentIndexChanged.connect(self._set_column)
        controls.addWidget(caption)
        controls.addWidget(self._combo)

        controls.addSpacing(18)
        kind_caption = QLabel("Type:")
        kind_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        kind_combo = style_combo(QComboBox())
        kind_combo.addItems(["Cathode", "Anode", "Both"])
        kind_combo.currentTextChanged.connect(self._set_kind)
        controls.addWidget(kind_caption)
        controls.addWidget(kind_combo)

        controls.addSpacing(18)
        median_box = QCheckBox("Median line per group")
        median_box.setChecked(True)
        median_box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
        median_box.toggled.connect(self._set_medians)
        controls.addWidget(median_box)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = Figure(figsize=(13, 6.2), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._hover = Hover(self._canvas)
        layout.addWidget(self._canvas)
        self._rebuild()

    def show_column(self, column):
        if column in self._columns:
            self._combo.setCurrentIndex(self._columns.index(column))

    def _set_column(self, index):
        self._column = self._combo.itemData(index)
        self._rebuild()

    def _set_kind(self, text):
        self._kind = {"Cathode": "CATHODE", "Anode": "ANODE",
                      "Both": None}[text]
        self._rebuild()

    def _set_medians(self, on):
        self._medians = bool(on)
        self._rebuild()

    def _subset(self, generation):
        part = self._frame[self._frame["generation"] == generation]
        if self._kind:
            part = part[part["electrode_type"] == self._kind]
        return part.dropna(subset=[self._column])

    def _groups(self):
        if self._mode == "binary":
            return [("good", "good", COLOR_GOOD, 40),
                    ("cracked", "cracked", COLOR_CRACKED, 40)]
        return [(s, SEVERITY_LABELS[s], SEVERITY_COLORS[s], SEVERITY_SIZES[s])
                for s in SEVERITY_ORDER]

    def _label_column(self):
        return "binary" if self._mode == "binary" else "severity"

    def _rebuild(self):
        self._fig.clear()
        self._hover.reset()
        grid = self._fig.add_gridspec(1, 2, wspace=0.08, left=0.075,
                                      right=0.985, top=0.76, bottom=0.13)
        panels = []
        for index, generation in enumerate(("Gen3.1", "Gen4.1")):
            ax = self._fig.add_subplot(grid[index],
                                       sharey=panels[0] if panels else None)
            panels.append(ax)
            self._draw(ax, generation, first=index == 0)

        values = pd.concat([self._subset("Gen3.1")[self._column],
                            self._subset("Gen4.1")[self._column]])
        if len(values) and values.notna().any():
            low, high = float(values.min()), float(values.max())
            pad = max(1e-9, 0.06 * (high - low)) if high > low else max(
                0.5, abs(high) * 0.05)
            panels[0].set_ylim(low - pad, high + pad)

        handles = [Line2D([], [], marker="o", linestyle="", markersize=7.5,
                          markerfacecolor=color, markeredgecolor=DOT_EDGE,
                          markeredgewidth=0.6, label=label)
                   for _k, label, color, _s in self._groups()]
        # Below the header, not beside it: parameter names run long and would
        # otherwise collide with the first legend entry
        self._fig.legend(handles=handles, loc="upper right",
                         bbox_to_anchor=(0.985, 0.90), frameon=False,
                         fontsize=10, labelcolor=SUBTEXT, ncols=len(handles),
                         handletextpad=0.4, columnspacing=1.5)

        name, unit = pretty(self._column)
        kind = {"CATHODE": "cathodes", "ANODE": "anodes",
                None: "cathodes + anodes"}[self._kind]
        heading = ("Good vs cracked" if self._mode == "binary"
                   else "Crack severity")
        self._fig.text(0.075, 0.975, f"{name} — {heading}", color=TEXT,
                       fontsize=15, fontweight="bold", va="top")
        self._fig.text(0.075, 0.928,
                       f"{self._column}   ·   {kind}   ·   shared y "
                       f"axis   ·   hover a point for its electrode",
                       color=SUBTEXT, fontsize=10.5, va="top")
        self._canvas.draw_idle()

    def _draw(self, ax, generation, first):
        ax.set_facecolor(AXES_BG)
        part = self._subset(generation)
        label_column = self._label_column()

        if not len(part):
            ax.text(0.5, 0.5, f"No {generation} data for this parameter",
                    ha="center", va="center", transform=ax.transAxes,
                    color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.tick_params(labelleft=first)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{generation}   n = 0", color=TEXT, fontsize=12.5,
                         fontweight="bold", loc="left", pad=16)
            return

        for key, label, color, size in self._groups():
            rows = part[part[label_column] == key]
            if not len(rows):
                continue
            x = mdates.date2num(rows["run_date"].dt.tz_localize(None))
            y = rows[self._column].to_numpy(dtype=float)
            ax.scatter(x, y, s=size, color=color, edgecolors=DOT_EDGE,
                       linewidths=0.6, alpha=0.9, zorder=3)
            unit = pretty(self._column)[1]
            tips = [f"{e}  ·  {label}\n{v:g} {unit}  ·  {d:%Y-%m-%d}"
                    f"\nnote: {str(n)[:46]}"
                    for e, v, d, n in zip(rows["electrode"], y,
                                          rows["run_date"], rows["note"])]
            self._hover.add(ax, x, y, tips, color)
            if self._medians:
                ax.axhline(np.median(y), color=color, linewidth=1.3,
                           linestyle=(0, (5, 3)), zorder=2, alpha=0.85)

        counts = part[label_column].value_counts()
        summary = ", ".join(f"{label} {int(counts.get(key, 0))}"
                            for key, label, _c, _s in self._groups()
                            if counts.get(key, 0))
        span = (f"{part['run_date'].min():%b %Y} – "
                f"{part['run_date'].max():%b %Y}")
        ax.set_title(f"{generation}   n = {len(part)}   ({span})", color=TEXT,
                     fontsize=12.5, fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.017, summary, transform=ax.transAxes, color=SUBTEXT,
                fontsize=9.5, va="bottom")

        # Gen4.1 spans two months and Gen3.1 seven, so a fixed "%b %Y" either
        # repeats itself or drops the year; Concise picks the right unit
        locator = mdates.AutoDateLocator(maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        if first:
            name, unit = pretty(self._column)
            ax.set_ylabel(f"{name}{f' ({unit})' if unit else ''}",
                          color=LABEL, fontsize=11)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5,
                       labelleft=first)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)


class RankingTab(QWidget):
    """Every parameter scored on both halves of a causal story."""

    plot_requested = pyqtSignal(str)

    COLUMNS = [
        ("Parameter", "name"), ("Gen3.1 median", "med31"),
        ("Gen4.1 median", "med41"), ("Between-gen effect", "between"),
        ("rho vs severity, Gen4.1", "rho41"),
        ("rho vs severity, Gen3.1", "rho31"),
        ("n Gen4.1", "n41"), ("n Gen3.1", "n31"), ("column", "column"),
    ]

    def __init__(self, frame):
        super().__init__()
        self._frame = frame
        self._kind = "CATHODE"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 14, 26, 14)

        heading = QLabel("What could be causing the cracks?")
        heading.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        layout.addWidget(heading)
        blurb = QLabel(
            "Gen4.1 cracks far more than Gen3.1, so a cause must differ "
            "between the generations — but so do hundreds of things, "
            "because the two ran consecutively and the recipe changed "
            "wholesale. A parameter is only a real candidate if it ALSO "
            "tracks severity within a generation, where nothing else is "
            "moving with it. Sorted by both together; click any header to "
            "re-sort, double-click a row to plot it.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {SUBTEXT}; font-size: 11.5px;")
        layout.addWidget(blurb)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 6, 0, 4)
        caption = QLabel("Electrode type:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = style_combo(QComboBox())
        combo.addItems(["Cathode", "Anode", "Both"])
        combo.currentTextChanged.connect(self._set_kind)
        controls.addWidget(caption)
        controls.addWidget(combo)
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {SUBTEXT}; font-size: 11.5px;")
        controls.addSpacing(16)
        controls.addWidget(self._status)
        controls.addStretch()
        layout.addLayout(controls)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {AXES_BG}; gridline-color: {GRID}; "
            f"color: {TEXT}; font-size: 12px; }}"
            f"QTableWidget::item {{ padding: 3px 6px; }}"
            f"QHeaderView::section {{ background: {FIG_BG}; color: {LABEL}; "
            f"font-weight: bold; font-size: 11.5px; padding: 6px; "
            f"border: none; border-bottom: 1px solid {SPINE}; }}")
        self._table.doubleClicked.connect(self._emit_plot)
        layout.addWidget(self._table)
        self._fill()

    def _set_kind(self, text):
        self._kind = {"Cathode": "CATHODE", "Anode": "ANODE",
                      "Both": None}[text]
        self._fill()

    def _emit_plot(self, index):
        item = self._table.item(index.row(), len(self.COLUMNS) - 1)
        if item is not None:
            self.plot_requested.emit(item.text())

    def _fill(self):
        table = rank_parameters(self._frame, self._kind)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(table))
        if not len(table):
            self._status.setText("not enough data for this electrode type")
            self._table.setSortingEnabled(True)
            return
        replicated = int(((table["p41"] < 0.05) & (table["p31"] < 0.05)
                          & (np.sign(table["rho41"])
                             == np.sign(table["rho31"]))).sum())
        self._status.setText(
            f"{len(table)} parameters · {replicated} track severity "
            f"significantly in BOTH generations with the same sign")

        for row, record in table.iterrows():
            for col, (_title, key) in enumerate(self.COLUMNS):
                value = record[key]
                item = QTableWidgetItem()
                if key in ("name", "column"):
                    unit = record["unit"]
                    text = (f"{value}  ({unit})"
                            if key == "name" and unit else str(value))
                    item.setData(Qt.ItemDataRole.DisplayRole, text)
                elif pd.isna(value):
                    item.setData(Qt.ItemDataRole.DisplayRole, "")
                else:
                    item.setData(Qt.ItemDataRole.DisplayRole, float(value))
                    item.setText(f"{value:,.0f}" if key.startswith("n")
                                 else f"{value:+.3f}" if key.startswith("rho")
                                 or key == "between" else f"{value:,.4g}")
                    item.setData(Qt.ItemDataRole.UserRole, float(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                # Tint the two severity columns by strength, so a strong
                # within-generation signal is findable at a glance
                if key in ("rho41", "rho31") and not pd.isna(value):
                    weight = min(1.0, abs(float(value)) / 0.4)
                    item.setBackground(QColor(139, 92, 246,
                                              int(28 + 90 * weight)))
                self._table.setItem(row, col, item)

        self._table.setSortingEnabled(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(col,
                                        QHeaderView.ResizeMode.ResizeToContents)
        self._table.setColumnHidden(len(self.COLUMNS) - 1, True)


class MainWindow(QMainWindow):
    def __init__(self, frame, columns):
        super().__init__()
        self.setWindowTitle("Gen3.1 vs Gen4.1 : every parameter")
        self._tabs = QTabWidget()
        default = ("step_31_p1_temp" if "step_31_p1_temp" in columns
                   else columns[0])
        self._binary = GenerationScatter(frame, "binary", columns, default)
        self._severity = GenerationScatter(frame, "severity", columns, default)
        ranking = RankingTab(frame)
        ranking.plot_requested.connect(self._plot)
        self._tabs.addTab(self._binary, "Good vs cracked")
        self._tabs.addTab(self._severity, "Crack severity")
        self._tabs.addTab(ranking, "Parameter ranking")
        self.setCentralWidget(self._tabs)

    def _plot(self, column):
        self._binary.show_column(column)
        self._severity.show_column(column)
        self._tabs.setCurrentIndex(1)     # severity reads better than binary

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    frame = load()
    columns = plottable(frame)
    print(f"{len(frame)} labelled electrodes, {len(columns)} comparable "
          f"parameters")

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background: {FIG_BG}; }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            background: transparent; padding: 10px 24px; font-size: 14px;
            color: {SUBTEXT}; border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT}; font-weight: bold;
            border-bottom: 2px solid {COLOR_GOOD};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
    """)
    window = MainWindow(frame, columns)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
