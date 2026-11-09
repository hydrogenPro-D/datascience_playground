"""The SEM measurements, as tables and as a chart.

Tab 1  the measurements themselves, one table per generation
Tab 2  how the quality rating distributes across each element ratio
Tab 3  the same, binned by a raw element level instead of a ratio
Tab 4  plate bath chemistry over time, one sub-tab per vat, each split
       again into nickel + thiourea and the ratio. The ratio view carries a
       second panel underneath: the coating Ni:S of every Gen4.1 electrode
       plated in that same vat, dated from dbo.MetaData and coloured by its
       quality rating, so bath chemistry and the coating it produced share one
       date axis. A running average cuts through the scatter, its window set
       by the slider from 1 to 10 days, and the rating checkboxes show or hide
       each group - the average and the median follow whatever is left
       visible. Hover any point for its exact date and value. A third view
       pairs the two directly: bath ratio against coating ratio, one point per
       bath sampling, with a permutation band showing what no relationship
       would look like.

On tab 2 the five ratings start as five separate bars, and the row of group
boxes merges any of them together: put two ratings in the same group and they
become one bar. The grouping is a view, not a change to the data, and is not
saved - it starts fresh every run.


qc_data.vw_sem_measurement stores one row per electrode / position / point /
element. This reduces it to one row per electrode, with each element as its own
column (the mean of that electrode's readings).

Which elements exist depends entirely on the generation: Gen3.1 was quantified
with Ni and S alone, Gen4.1 adds C, N and Co for everything and then O, Cl, Al,
Si, Br, K for some electrodes but not others. So each table is given only the
columns its own generation actually has, and every element header carries the
share of that generation's electrodes that hold a value. A cell an electrode
never had is left BLANK, never zero: a missing measurement and a measured zero
are different facts and must not look alike.

The SEM note column holds the categorised verdict from note_dictionary.py
rather than the operator's raw free text, and Rating is its quality level: 5
fine, 4 very few microcracks, 3 microcracks, 2 small cracks, 1 cracked. A note
the dictionary does not recognise, or an electrode with no note, leaves both
blank.

The trailing ratio columns are worked out per row from the element means. A
ratio needs both of its elements, so one is only offered to a generation that
measured both: Gen3.1 quantified Ni and S alone and therefore gets Ni:S only,
while Gen4.1 gets all five. Within a table, a row missing either element (or
holding a zero denominator) is left blank.

Anodes and cathodes sit in the same table, separated by the Type column.

Gen4.1 production starts at B13900. The 17 markings below that are trial runs
and are left out. Gen3.1 tops out at B13853, so the cutoff never touches it.

Run with the repo .venv:
    ./.venv/Scripts/python.exe sem_analysis/gui_sem.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QAbstractItemView, QHeaderView,
                             QSplitter, QComboBox, QTabBar, QCheckBox,
                             QSlider)
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from scipy import stats

import note_dictionary as nd

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_raw.csv"

FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"
ACCENT = "#0EA5A5"

# Gen4.1 production begins here; anything below is a trial run
GEN4_FIRST = 13900

# Ratios derived from the element means, as (numerator, denominator)
RATIOS = [("Ni", "S"), ("Ni", "Co"), ("Ni", "C"), ("Ni", "N"), ("S", "N")]

# Rating 5 is the best coating, so it gets the palest fill and 1 the darkest:
# one hue light to dark, because the rating is a magnitude, not five unrelated
# categories.
RATING_COLORS = {5: "#D6E4F7", 4: "#9FC0E8", 3: "#5B8FD1",
                 2: "#2A5BA8", 1: "#122E5C"}
BINS = 5                     # x-axis groups the charts open with
# Offered in the dropdown. Two is the coarsest split worth drawing and ten
# leaves about 22 electrodes a group on Gen4.1, below which a single
# electrode swings a bar by five points.
BIN_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# Past this many groups a two-line "3.07 to 3.57 / 36 electrodes" label is
# wider than the group it sits under, so the labels go to one tilted line.
CROWDED_BINS = 6
# Bars per panel past which the value on top of a bar is wider than the bar.
# Ten groups of five ratings is fifty bars, where "36" and "32" run together
# into "3632"; standing the numbers up keeps them legible without dropping
# them.
CROWDED_BARS = 30
EMPTY = "#9CA3AF"
DOT_EDGE = "#64748B"
# Letters rather than numbers, so a group is never confused with a rating
GROUP_KEYS = ["A", "B", "C", "D", "E"]
# A sixth choice that is not a group: the rating leaves the chart entirely.
# Not the same as merging it into a neighbour - an excluded electrode drops
# out of the denominator too, so the bars that remain are shares of the
# electrodes still in, and the count under each x-axis group falls to match.
EXCLUDE_KEY = "out"

# Elements offered on the by-element chart. Si, Br and K are measured on too
# few electrodes (17, 15 and 2) to split into five bins at all, so they are
# left out rather than listed and always refused.
ELEMENT_CHOICES = ["Ni", "S", "Co", "C", "N", "O", "Cl", "Al"]

# --- plate bath chemistry -------------------------------------------------
BATH_FILE = FOLDER / "plate_bath.csv"
CONTEXT_FILE = FOLDER / "electrode_context.csv"
# Centred on each point, so the line tracks the middle of the cloud rather
# than lagging it. Time-based rather than a point count, because electrodes
# are produced in bursts and a fixed count would span days in one place and
# hours in another.
ROLLING_DEFAULT_DAYS = 7
ROLLING_MIN_DAYS, ROLLING_MAX_DAYS = 1, 10

# A bath sampling holds until the next one, so every electrode plated in that
# gap carries the same bath value. Pair them that way, never forward.
BATH_TOLERANCE_DAYS = 14
PERMUTATIONS = 400        # for the "no relationship" envelope
NULL_BAND = "#CBD5E1"
BATH_START = "2026-07-01"
BATH_VATS = (28, 29, 30)
# Nickel now comes from a 445nm photometric reading, three per sampling.
# The Ni2+ titration column it replaced has been empty since July 2026.
NI_ABSORBANCE_COLUMNS = ["Ni Absorbance 445nm 1st", "Ni Absorbance 445nm 2nd",
                         "Ni Absorbance 445nm 3rd"]
NI_BLANK = 0.003          # absorbance of the blank
NI_SLOPE = 0.216466       # absorbance per mg/L
NI_MOLAR_MASS = 58.693    # g/mol

TU_COLUMNS = ["ThioUrea Titration 1st (ml)", "ThioUrea Titration 2nd (ml)"]
# Thiourea is still titrated; only nickel changed method
TU_G_PER_L_PER_ML = 2.2836
# One accent per quantity, and the ratio takes a third rather than reusing
# either of the two it is built from
BATH_COLORS = {"nickel": "#0EA5A5", "thiourea": "#8B5CF6",
               "ratio": "#2A5BA8"}

# Fixed columns in front of the per-element ones
CONTEXT = [("electrode", "Electrode"), ("electrode_type", "Type"),
           ("measured_on", "Measured on"), ("positions", "Positions"),
           ("points", "Readings"), ("category", "SEM note"),
           ("quality", "Rating")]


class Hover:
    """Nearest-point tooltip across every axes on a canvas."""

    RADIUS = 20   # px

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
        """Re-register after a Figure.clear() stripped the annotation."""
        self._series = []
        self._canvas.figure.add_artist(self._ann)
        self._ann.set_visible(False)
        self._shown = None

    def add(self, artist, labels, colour):
        self._series.append((artist, list(labels), colour))

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
        for artist, labels, colour in self._series:
            if artist.axes is not event.inaxes or not labels:
                continue
            xs = np.asarray(artist.get_xdata(orig=False), dtype=float)
            ys = np.asarray(artist.get_ydata(orig=False), dtype=float)
            if not len(xs):
                continue
            points = artist.axes.transData.transform(np.column_stack([xs, ys]))
            distance = np.hypot(points[:, 0] - event.x, points[:, 1] - event.y)
            i = int(np.argmin(distance))
            if distance[i] <= self.RADIUS and (best is None
                                               or distance[i] < best[0]):
                best = (distance[i], points[i], labels[i], colour)
        if best is None:
            self._hide()
            return
        _d, point, label, colour = best
        key = (label, tuple(point))
        if key == self._shown:
            return
        self._shown = key
        width, height = (self._canvas.figure.bbox.width,
                         self._canvas.figure.bbox.height)
        dx = 14 if point[0] < width - 180 else -14
        dy = 14 if point[1] < height - 60 else -14
        self._ann.set_position((dx, dy))
        self._ann.set_ha("left" if dx > 0 else "right")
        self._ann.set_va("bottom" if dy > 0 else "top")
        self._ann.xy = (point[0], point[1])
        self._ann.set_text(label)
        self._ann.get_bbox_patch().set_facecolor(colour)
        self._ann.set_visible(True)
        self._canvas.draw_idle()


def load():
    if not SEM_FILE.exists():
        raise SystemExit(f"{SEM_FILE.name} not found - run fetch_sem.py first")
    sem = pd.read_csv(SEM_FILE)
    sem["electrode"] = sem["electrode"].astype(str).str.strip()
    return sem


def nickel_g_per_l(bath):
    """Nickel concentration from the three 445nm absorbance readings.

    Per reading: mg/L = (absorbance - blank) / slope, then to mol/L; the three
    are averaged as mol/L and converted back to g/L. The two molar-mass terms
    differ by a factor of ten (/1000 out, x10000 back), which is where the
    dilution enters - net, the result is ten times the mean mg/L.
    """
    missing = [c for c in NI_ABSORBANCE_COLUMNS if c not in bath.columns]
    if missing:
        print(f"⚠ plate_bath.csv has no {missing} - nickel will be blank. "
              f"Re-run fetch_sem.py against the current view.")
        return pd.Series(np.nan, index=bath.index)
    readings = bath[NI_ABSORBANCE_COLUMNS].apply(pd.to_numeric,
                                                 errors="coerce")
    mg_per_l = (readings - NI_BLANK) / NI_SLOPE
    mol_per_l = mg_per_l / (NI_MOLAR_MASS * 1000)
    return mol_per_l.mean(axis=1) * NI_MOLAR_MASS * 10000


def load_bath():
    """Plate bath log from BATH_START onward, one row per sampling.

    Nickel comes from three 445nm absorbance readings, thiourea from two
    titrations; each set is averaged before being converted to g/L."""
    if not BATH_FILE.exists():
        return None
    bath = pd.read_csv(BATH_FILE)
    bath["Date"] = pd.to_datetime(bath["Date"], errors="coerce")
    bath = bath[bath["Date"] >= BATH_START].copy()
    bath["nickel"] = nickel_g_per_l(bath)
    bath["thiourea"] = bath[TU_COLUMNS].mean(axis=1) * TU_G_PER_L_PER_ML
    # A ratio needs both, so a sampling that skipped one is left blank
    bath["ratio"] = bath["nickel"] / bath["thiourea"].where(
        bath["thiourea"] != 0)
    return bath.sort_values(["vat", "Date"])


def load_electrodes(sem):
    """Gen4.1 electrodes with a production date, plating vat and coating Ni:S.

    The bath panels are per vat, so an electrode is only shown under the vat it
    actually plated in; one without a resolvable vat or date cannot be placed
    and is left out."""
    if not CONTEXT_FILE.exists():
        return None
    table, _elements, ratios = build_table(sem, "Gen4.1", GEN4_FIRST)
    if not len(table) or "Ni:S" not in ratios:
        return None
    context = pd.read_csv(CONTEXT_FILE)
    context["electrode"] = context["electrode"].astype(str).str.strip()
    context["produced"] = pd.to_datetime(context["produced"], errors="coerce")

    frame = table[["electrode", "Ni:S", "quality", "category"]].merge(
        context[["electrode", "produced", "plating_vat"]], on="electrode",
        how="inner")
    return frame.dropna(subset=["produced", "plating_vat", "Ni:S"])


def marking_number(electrode):
    """The numeric part of a marking, so B13900 sorts and compares as 13900."""
    return pd.to_numeric(
        electrode.str.extract(r"^[A-Za-z]*(\d+)")[0], errors="coerce")


def build_table(sem, generation, cutoff=None):
    """One row per electrode of this generation, elements as columns."""
    part = sem[sem["generation"] == generation]
    if cutoff is not None:
        part = part[marking_number(part["electrode"]) >= cutoff]
    if not len(part):
        return pd.DataFrame(), [], []

    context = part.groupby("electrode").agg(
        electrode_type=("electrode_type", "first"),
        measured_on=("measured_on", "first"),
        note=("note", "first"),
        positions=("position", "nunique"),
        points=("point", "nunique"))

    readings = part.dropna(subset=["symbol", "value"])
    elements = readings.pivot_table(index="electrode", columns="symbol",
                                    values="value", aggfunc="mean")
    # Commonest element first, so the columns every electrode has lead and the
    # ragged ones trail
    order = (readings.groupby("symbol")["electrode"].nunique()
             .sort_values(ascending=False).index.tolist())
    elements = elements.reindex(columns=order)

    # The raw note is kept only long enough to derive these two; the table
    # shows the categorised verdict, not the operator's wording
    context["category"] = context["note"].map(nd.category)
    context["quality"] = context["note"].map(nd.quality)

    table = context.join(elements, how="left").reset_index()

    # A ratio is only offered when the generation measured both elements at
    # all; within that, a row missing either one stays blank rather than
    # inventing a number
    ratios = []
    for top, bottom in RATIOS:
        if top not in table.columns or bottom not in table.columns:
            continue
        name = f"{top}:{bottom}"
        denominator = table[bottom].where(table[bottom] != 0)
        table[name] = table[top] / denominator
        ratios.append(name)

    table = table.sort_values(["electrode_type", "electrode"])
    return table, order, ratios


class GenerationTable(QWidget):
    """Heading plus a sortable table for one generation."""

    def __init__(self, sem, generation, cutoff=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(4)

        heading = QLabel(generation)
        heading.setStyleSheet(
            f"color: {TEXT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(heading)

        caption = QLabel()
        caption.setWordWrap(True)
        caption.setStyleSheet(f"color: {SUBTEXT}; font-size: 11.5px;")
        layout.addWidget(caption)

        table, elements, ratios = build_table(sem, generation, cutoff)
        if not len(table):
            caption.setText("No electrodes for this generation.")
            return

        counts = table["electrode_type"].value_counts()
        kinds = ", ".join(f"{int(n)} {kind.lower()}"
                          for kind, n in counts.items())
        filled = {e: table[e].notna().sum() for e in elements}
        caption.setText(f"{len(table)} electrodes ({kinds})")

        columns = (CONTEXT
                   + [(e, f"{e}\n{100 * filled[e] / len(table):.0f}%")
                      for e in elements]
                   + [(r, f"{r}\n{100 * table[r].notna().mean():.0f}%")
                      for r in ratios])
        widget = QTableWidget()
        widget.setColumnCount(len(columns))
        widget.setHorizontalHeaderLabels([title for _key, title in columns])
        widget.setRowCount(len(table))
        widget.setSortingEnabled(False)
        widget.setAlternatingRowColors(True)
        widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        widget.verticalHeader().setVisible(False)
        widget.setStyleSheet(
            f"QTableWidget {{ background: {AXES_BG}; gridline-color: {GRID}; "
            f"color: {TEXT}; font-size: 12px; }}"
            f"QTableWidget::item {{ padding: 3px 6px; }}"
            f"QHeaderView::section {{ background: {FIG_BG}; color: {LABEL}; "
            f"font-weight: bold; font-size: 11px; padding: 5px; "
            f"border: none; border-bottom: 1px solid {SPINE}; }}")

        numeric = set(elements) | set(ratios) | {"positions", "points",
                                                  "quality"}
        for row, (_index, record) in enumerate(table.iterrows()):
            for column, (key, _title) in enumerate(columns):
                value = record[key]
                item = QTableWidgetItem()
                if pd.isna(value):
                    # Left blank on purpose: this element was never quantified
                    # for this electrode, which is not the same as measuring 0
                    item.setData(Qt.ItemDataRole.DisplayRole, "")
                elif key in numeric:
                    item.setData(Qt.ItemDataRole.DisplayRole, float(value))
                    item.setText(f"{value:,.0f}"
                                 if key in ("positions", "points", "quality")
                                 else f"{value:.2f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setData(Qt.ItemDataRole.DisplayRole, str(value))
                widget.setItem(row, column, item)

        widget.setSortingEnabled(True)
        header = widget.horizontalHeader()
        for column in range(len(columns)):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        # Nothing needs a wide column now the long free text is gone; let the
        # last element column take up the slack instead
        header.setStretchLastSection(True)
        layout.addWidget(widget)


class SemTab(QWidget):
    def __init__(self, sem):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 16, 26, 16)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {GRID}; height: 3px; }}")
        # Only Gen4.1 has a production cutoff; Gen3.1 ends at B13853
        splitter.addWidget(GenerationTable(sem, "Gen3.1"))
        splitter.addWidget(GenerationTable(sem, "Gen4.1", cutoff=GEN4_FIRST))
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)


class QualityTab(QWidget):
    """Quality rating across bins of a measured value, one panel per
    generation.

    `mode` picks what the x axis bins by: "ratio" for the derived element
    ratios, "element" for a raw element level. Everything else is the same -
    equal-sized groups along the x axis, how many being a control, one bar
    per rating within each, bars summing to 100%, anodes and cathodes
    pooled. An electrode with no rating, or missing
    the value being binned, cannot be placed and is dropped; the count under
    each group says how many actually went in."""

    def __init__(self, sem, mode="ratio"):
        super().__init__()
        self._sem = sem
        self._mode = mode
        self._choices = ([f"{top}:{bottom}" for top, bottom in RATIOS]
                         if mode == "ratio" else list(ELEMENT_CHOICES))
        self._value = self._choices[0]
        self._bins = BINS
        # Every rating in its own group to start with, so the chart opens with
        # all five shown separately
        self._group_of = {rating: GROUP_KEYS[index] for index, rating
                          in enumerate((5, 4, 3, 2, 1))}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 14, 26, 14)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Ratio:" if mode == "ratio" else "Element:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12.5px;")
        combo = QComboBox()
        combo.addItems(self._choices)
        combo.setStyleSheet(self._combo_style())
        combo.currentTextChanged.connect(self._set_value)
        controls.addWidget(caption)
        controls.addWidget(combo)

        controls.addSpacing(26)
        # How finely the x axis is cut. Separate from "Group ratings" below,
        # which merges the bars inside each of these groups.
        bin_caption = QLabel("X-axis groups:")
        bin_caption.setStyleSheet(f"color: {LABEL}; font-size: 12.5px;")
        bin_picker = QComboBox()
        bin_picker.addItems([str(count) for count in BIN_CHOICES])
        bin_picker.setCurrentText(str(self._bins))
        bin_picker.setStyleSheet(self._combo_style())
        bin_picker.currentTextChanged.connect(self._set_bins)
        controls.addWidget(bin_caption)
        controls.addWidget(bin_picker)

        controls.addSpacing(26)
        group_caption = QLabel("Group ratings:")
        group_caption.setStyleSheet(f"color: {LABEL}; font-size: 12.5px;")
        controls.addWidget(group_caption)
        for rating in (5, 4, 3, 2, 1):
            tag = QLabel(f"{rating}")
            tag.setStyleSheet(f"color: {SUBTEXT}; font-size: 12.5px;")
            picker = QComboBox()
            picker.addItems(GROUP_KEYS + [EXCLUDE_KEY])
            picker.setCurrentText(self._group_of[rating])
            picker.setStyleSheet(self._combo_style())
            picker.currentTextChanged.connect(
                lambda key, r=rating: self._set_group(r, key))
            controls.addWidget(tag)
            controls.addWidget(picker)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = Figure(figsize=(13.5, 6.4), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)
        self._rebuild()

    @staticmethod
    def _combo_style():
        return (f"QComboBox {{ color: {TEXT}; font-size: 12.5px; "
                f"padding: 3px 8px; border: 1px solid {SPINE}; "
                f"border-radius: 4px; background: {AXES_BG}; }}")

    def _set_value(self, text):
        self._value = text
        self._rebuild()

    def _set_bins(self, text):
        self._bins = int(text)
        self._rebuild()

    def _headline(self):
        if self._mode == "ratio":
            return f"Coating quality against the {self._value} ratio"
        return f"Coating quality against {self._value} content"

    def _axis_label(self):
        if self._mode == "ratio":
            bottom = self._value.split(":")[1]
            return (f"{self._value} ratio in the coating  "
                    f"(higher = less {bottom})")
        return f"{self._value} in the coating  (higher = more {self._value})"

    def _set_group(self, rating, key):
        self._group_of[rating] = key
        self._rebuild()

    def _kept(self):
        """Ratings still in the chart, in rating order."""
        return [rating for rating in (5, 4, 3, 2, 1)
                if self._group_of[rating] != EXCLUDE_KEY]

    def _groups(self):
        """Active groups, best first: (key, ratings in it, label, colour)."""
        members = {}
        for rating in self._kept():
            members.setdefault(self._group_of[rating], []).append(rating)
        out = []
        for key, ratings in members.items():
            ratings = sorted(ratings, reverse=True)
            if len(ratings) == 1:
                label = f"{ratings[0]}  {nd.QUALITY_LEVELS[ratings[0]]}"
            else:
                names = " / ".join(nd.QUALITY_LEVELS[r] for r in ratings)
                label = f"{'+'.join(str(r) for r in ratings)}  {names}"
            # Shade a merged group by the middle of what it contains
            shade = RATING_COLORS[int(round(sum(ratings) / len(ratings)))]
            out.append((key, ratings, label, shade))
        return sorted(out, key=lambda item: -max(item[1]))

    def _rebuild(self):
        self._fig.clear()
        # Tilted labels need more room underneath than flat ones
        bottom = 0.28 if self._bins > CROWDED_BINS else 0.20
        grid = self._fig.add_gridspec(1, 2, wspace=0.13, left=0.055,
                                      right=0.985, top=0.76, bottom=bottom)
        panels = []
        for index, (generation, cutoff) in enumerate(
                (("Gen3.1", None), ("Gen4.1", GEN4_FIRST))):
            ax = self._fig.add_subplot(grid[index])
            self._draw(ax, generation, cutoff, first=index == 0)
            panels.append(ax)
        # A shared y range, so a 67% bar in one generation is not drawn the
        # same height as a 56% bar in the other
        top = max(ax.get_ylim()[1] for ax in panels)
        for ax in panels:
            ax.set_ylim(0, top)

        groups = self._groups()
        handles = [Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=shade, label=label)
                   for _key, _ratings, label, shade in groups]
        self._fig.legend(handles=handles, loc="upper right",
                         bbox_to_anchor=(0.985, 0.915), frameon=False,
                         fontsize=9.5, labelcolor=SUBTEXT,
                         ncols=min(len(handles), 3 if len(handles) < 4 else 5),
                         handletextpad=0.4, columnspacing=1.4)

        self._fig.text(0.055, 0.965, self._headline(), color=TEXT,
                       fontsize=15.5, fontweight="bold", va="top")
        self._canvas.draw_idle()

    def _draw(self, ax, generation, cutoff, first):
        ax.set_facecolor(AXES_BG)
        table, elements, ratios = build_table(self._sem, generation, cutoff)
        available = ratios if self._mode == "ratio" else elements

        if self._value not in available:
            if self._mode == "ratio":
                missing = [e for e in self._value.split(":")
                           if e not in elements]
                message = (f"{generation} never measured "
                           f"{' or '.join(missing)},\nso {self._value} cannot "
                           f"be calculated")
            else:
                message = f"{generation} never measured {self._value}"
            ax.text(0.5, 0.5, message,
                    ha="center", va="center", transform=ax.transAxes,
                    color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            # Every spine, not just three: a lone bottom rule under an empty
            # panel reads as an axis that failed to draw
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{generation}", color=TEXT, fontsize=12.5,
                         fontweight="bold", loc="left", pad=10)
            return

        part = table[table["quality"].notna() & table[self._value].notna()]
        part = part[[self._value, "quality"]].copy()
        # Excluded ratings go before the x axis is cut, so the quantile
        # edges and the counts underneath describe the electrodes actually
        # drawn rather than a population half of which is not on the chart
        kept = self._kept()
        part = part[part["quality"].isin(kept)]
        if not kept:
            ax.text(0.5, 0.5, "Every rating is set to ‘out’",
                    ha="center", va="center", transform=ax.transAxes,
                    color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{generation}", color=TEXT, fontsize=12.5,
                         fontweight="bold", loc="left", pad=10)
            return
        if len(part) < self._bins * 4:
            ax.text(0.5, 0.5, "Not enough rated electrodes", ha="center",
                    va="center", transform=ax.transAxes, color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        part["bin"] = pd.qcut(part[self._value], self._bins, labels=False,
                              duplicates="drop")
        edges = pd.qcut(part[self._value], self._bins,
                        duplicates="drop").cat.categories
        counts = part.groupby("bin").size()
        share = (part.groupby(["bin", "quality"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=kept, fill_value=0))
        share = 100 * share.div(share.sum(axis=1), axis=0)

        groups = self._groups()
        x = np.arange(len(share))
        width = 0.8 / len(groups)
        middle = (len(groups) - 1) / 2
        tallest = 0.0
        packed = len(share) * len(groups) > CROWDED_BARS
        for offset, (_key, ratings, _label, shade) in enumerate(groups):
            position = x + (offset - middle) * width
            # A merged group is just the sum of its ratings' shares
            values = share[ratings].sum(axis=1).to_numpy(dtype=float)
            tallest = max(tallest, float(values.max()))
            ax.bar(position, values, width=width * 0.92, color=shade,
                   edgecolor=DOT_EDGE, linewidth=0.4, zorder=3)
            for pos, value in zip(position, values):
                if value >= 4:      # below this the label will not fit
                    ax.annotate(f"{value:.0f}", (pos, value),
                                ha="center", va="bottom", xytext=(0, 2),
                                textcoords="offset points",
                                rotation=90 if packed else 0,
                                fontsize=7 if packed
                                else (7.5 if len(groups) > 3 else 9),
                                color=SUBTEXT)

        # Flat two-line labels collide once the groups get narrow, so
        # past CROWDED_BINS they collapse to one line and tilt. "n=" keeps
        # the count readable without the word.
        crowded = len(edges) > CROWDED_BINS
        ax.set_xticks(x, [
            (f"{e.left:.2f}-{e.right:.2f}  n={n}" if crowded
             else f"{e.left:.2f} to {e.right:.2f}\n{n} electrodes")
            for e, n in zip(edges, counts)])
        if crowded:
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_rotation_mode("anchor")
                tick.set_horizontalalignment("right")
                tick.set_fontsize(8.5)
        ax.set_xlabel(self._axis_label(), color=SUBTEXT, fontsize=10.5,
                      labelpad=6)
        if first:
            ax.set_ylabel("share of the group", color=LABEL, fontsize=11)
        # Measured on the bars actually drawn: a merged group is the sum of
        # its ratings and can reach 100%, well above any single rating's
        # share. Upright labels stand above their bar and need the headroom.
        headroom = 1.28 if packed else 1.15
        ax.set_ylim(0, min(112, max(45, tallest * headroom)))
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5)
        ax.set_title(f"{generation}   ({len(part)} rated electrodes)",
                     color=TEXT, fontsize=12.5, fontweight="bold",
                     loc="left", pad=10)


class BathTab(QWidget):
    """Plate bath chemistry over time, one sub-tab per vat.

    Three stacked panels on a shared date axis: nickel, thiourea, and nickel
    divided by thiourea. They sit at roughly 11.5 ml, 21 ml and 0.56, so a
    single axes would either need a second scale or squash the ratio flat
    against the floor; stacking keeps each on its own scale and still lines
    the dates up.

    Values are titration volumes in millilitres, not concentrations."""

    # column -> (panel title, y-axis unit, tooltip format). The title carries
    # the name, so the y label only needs the unit or it would say it twice.
    PANELS = {"nickel": ("Nickel", "g/L", "{:.2f} g/L"),
              "thiourea": ("Thiourea", "g/L", "{:.2f} g/L"),
              "ratio": ("Nickel : thiourea", "ratio of g/L", "{:.3f}")}
    # "electrodes" is not a bath column: it is the coating Ni:S panel, drawn
    # from the SEM side and sharing the bath panel's date axis
    VIEWS = [("Nickel & thiourea", ["nickel", "thiourea"]),
             ("Ratio", ["ratio", "electrodes"]),
             ("Bath vs coating", ["scatter"])]

    def __init__(self, bath, electrodes=None):
        super().__init__()
        self._bath = bath
        self._electrodes = electrodes
        self._ratings_on = {rating: True for rating in (5, 4, 3, 2, 1)}
        self._window_days = ROLLING_DEFAULT_DAYS
        self._vat = BATH_VATS[0]
        self._view = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 14, 26, 14)

        layout.addLayout(self._bar_row(
            [f"Plate {vat}" for vat in BATH_VATS], self._set_vat,
            BATH_COLORS["ratio"], 12))
        # A second bar, indented and smaller, so the nesting is visible
        layout.addLayout(self._bar_row(
            [name for name, _columns in self.VIEWS], self._set_view,
            BATH_COLORS["nickel"], 11, indent=18))

        # Only meaningful on the ratio view, so it is hidden on the other
        self._ratings_row = QWidget()
        row = QHBoxLayout(self._ratings_row)
        row.setContentsMargins(18, 0, 0, 0)
        caption = QLabel("Show ratings:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 11.5px;")
        row.addWidget(caption)
        for rating in (5, 4, 3, 2, 1):
            box = QCheckBox(f"{rating} {nd.QUALITY_LEVELS[rating]}")
            box.setChecked(True)
            box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; "
                              f"font-size: 11.5px; }}")
            box.toggled.connect(
                lambda on, r=rating: self._set_rating(r, on))
            row.addWidget(box)

        self._window_box = QWidget()
        window_row = QHBoxLayout(self._window_box)
        window_row.setContentsMargins(22, 0, 0, 0)
        row.addWidget(self._window_box)
        window_caption = QLabel("Average window:")
        window_caption.setStyleSheet(f"color: {LABEL}; font-size: 11.5px;")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(ROLLING_MIN_DAYS)
        slider.setMaximum(ROLLING_MAX_DAYS)
        slider.setValue(ROLLING_DEFAULT_DAYS)
        slider.setFixedWidth(150)
        slider.setPageStep(1)
        slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ height: 3px; background: {SPINE}; "
            f"border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; "
            f"background: {BATH_COLORS['ratio']}; border-radius: 6px; }}"
            f"QSlider::sub-page:horizontal {{ background: "
            f"{BATH_COLORS['ratio']}; height: 3px; border-radius: 2px; }}")
        self._window_label = QLabel(f"{ROLLING_DEFAULT_DAYS} days")
        self._window_label.setStyleSheet(
            f"color: {SUBTEXT}; font-size: 11.5px;")
        self._window_label.setFixedWidth(52)
        slider.valueChanged.connect(self._set_window)
        window_row.addWidget(window_caption)
        window_row.addWidget(slider)
        window_row.addWidget(self._window_label)
        row.addStretch()
        layout.addWidget(self._ratings_row)

        self._fig = Figure(figsize=(13.0, 7.0), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._hover = Hover(self._canvas)
        layout.addWidget(self._canvas)
        self._sync_controls()
        self._rebuild()

    @staticmethod
    def _bar_row(names, handler, accent, size, indent=0):
        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for name in names:
            bar.addTab(name)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 5px 16px; "
            f"font-size: {size}px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {accent}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(handler)
        row = QHBoxLayout()
        row.setContentsMargins(indent, 0, 0, 0)
        row.addWidget(bar)
        row.addStretch()
        return row

    def _set_vat(self, index):
        self._vat = BATH_VATS[int(index)]
        self._rebuild()

    def _set_view(self, index):
        self._view = int(index)
        self._sync_controls()
        self._rebuild()

    def _sync_controls(self):
        columns = self.VIEWS[self._view][1]
        uses_electrodes = bool({"electrodes", "scatter"} & set(columns))
        self._ratings_row.setVisible(uses_electrodes)
        # The rolling window only exists on the time-series view
        self._window_box.setVisible("electrodes" in columns)

    def _set_rating(self, rating, on):
        self._ratings_on[rating] = bool(on)
        self._rebuild()

    def _set_window(self, days):
        self._window_days = int(days)
        self._window_label.setText(
            f"{days} day" + ("" if days == 1 else "s"))
        self._rebuild()

    def _rebuild(self):
        self._fig.clear()
        self._hover.reset()
        if self._bath is None:
            self._message("plate_bath.csv not found - run fetch_sem.py first")
            return
        part = self._bath[self._bath["vat"] == self._vat]
        if not len(part):
            self._message(f"No plate {self._vat} measurements since "
                          f"{BATH_START}")
            return

        columns = self.VIEWS[self._view][1]
        grid = self._fig.add_gridspec(len(columns), 1, hspace=0.26,
                                      left=0.075, right=0.98, top=0.90,
                                      bottom=0.11)
        first = None
        for index, column in enumerate(columns):
            ax = self._fig.add_subplot(grid[index], sharex=first)
            first = first or ax
            last = index == len(columns) - 1
            if column == "scatter":
                self._draw_scatter(ax)
            elif column == "electrodes":
                self._draw_electrodes(ax, last)
            else:
                label, unit, fmt = self.PANELS[column]
                self._draw(ax, part, column, label, unit, fmt, last)

        self._fig.text(0.075, 0.975, f"Plate bath {self._vat} chemistry",
                       color=TEXT, fontsize=15.5, fontweight="bold", va="top")
        self._canvas.draw_idle()

    def _draw_electrodes(self, ax, last):
        """Coating Ni:S of the electrodes plated in this vat, over time."""
        ax.set_facecolor(AXES_BG)
        frame = self._electrodes
        if frame is None or not len(frame):
            self._blank(ax, "No electrode data - run fetch_sem.py")
            self._date_axis(ax, last)
            return
        part = frame[frame["plating_vat"] == self._vat]
        if not len(part):
            self._blank(ax, f"No Gen4.1 electrodes plated in vat {self._vat}")
            self._date_axis(ax, last)
            return

        # An unchecked rating leaves the panel entirely: its dots go, and the
        # average and median are taken over what is left, so the line always
        # describes what you can actually see
        hidden = [r for r, on in self._ratings_on.items() if not on]
        part = part[~part["quality"].isin(hidden)]
        if not len(part):
            self._blank(ax, "No electrodes in the selected ratings")
            self._date_axis(ax, last)
            return

        rated = part["quality"].notna()
        groups = [(int(r), nd.QUALITY_LEVELS[int(r)], RATING_COLORS[int(r)])
                  for r in (5, 4, 3, 2, 1) if self._ratings_on[r]]
        for rating, name, colour in groups:
            rows = part[part["quality"] == rating]
            if not len(rows):
                continue
            line, = ax.plot(rows["produced"], rows["Ni:S"], linestyle="none",
                            marker="o", markersize=5.5, color=colour,
                            markeredgecolor=DOT_EDGE, markeredgewidth=0.5,
                            zorder=3)
            self._hover.add(line, [
                f"{e}\n{d:%A %d %B %Y}\nNi:S {v:.3f}  ·  {rating} {name}"
                for e, d, v in zip(rows["electrode"], rows["produced"],
                                   rows["Ni:S"])], colour)
        unrated = part[~rated]
        if len(unrated):
            line, = ax.plot(unrated["produced"], unrated["Ni:S"],
                            linestyle="none", marker="o", markersize=5.5,
                            color=EMPTY, markeredgecolor=DOT_EDGE,
                            markeredgewidth=0.5, zorder=3)
            self._hover.add(line, [
                f"{e}\n{d:%A %d %B %Y}\nNi:S {v:.3f}  ·  no rating"
                for e, d, v in zip(unrated["electrode"], unrated["produced"],
                                   unrated["Ni:S"])], EMPTY)

        if len(part) >= 3:
            median = float(part["Ni:S"].median())
            ax.axhline(median, color=DOT_EDGE, linewidth=1.1, alpha=0.55,
                       linestyle=(0, (5, 3)), zorder=2)
            ax.annotate(f"median {median:,.3f}", (1.0, median),
                        xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=9, color=SUBTEXT)

        # min_periods=1 and dropna between them keep the line unbroken: a NaN
        # anywhere in the series splits the plotted line in two, which made
        # sparse stretches look like the average had stopped existing
        smooth = (part.sort_values("produced").set_index("produced")["Ni:S"]
                  .rolling(f"{self._window_days}D", center=True,
                           min_periods=1).mean().dropna())
        if len(smooth):
            ax.plot(smooth.index, smooth.to_numpy(), color=TEXT, linewidth=2.0,
                    alpha=0.85, zorder=5)

        handles = [Line2D([], [], marker="o", linestyle="", markersize=6.5,
                          color=colour, markeredgecolor=DOT_EDGE,
                          markeredgewidth=0.5, label=f"{rating} {name}")
                   for rating, name, colour in groups
                   if (part["quality"] == rating).any()]
        handles.append(Line2D([], [], color=TEXT, linewidth=2.0,
                              label=f"{self._window_days}-day average"))
        if handles:
            # Figure level, up beside the title: between the two panels it
            # reads as though it belonged to the bath chart above
            self._fig.legend(handles=handles, loc="upper right",
                             bbox_to_anchor=(0.98, 0.995), frameon=False,
                             fontsize=9, labelcolor=SUBTEXT,
                             ncols=len(handles), handletextpad=0.3,
                             columnspacing=1.1)

        ax.set_ylabel("Ni:S in the coating", color=LABEL, fontsize=10.5)
        ax.set_title(f"Electrodes plated in vat {self._vat}   "
                     f"({len(part)} shown)", color=TEXT, fontsize=11,
                     fontweight="bold", loc="left", pad=6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5,
                       labelbottom=last)
        self._date_axis(ax, last)

    def _paired(self):
        """Electrodes joined to the bath sampling they were plated under.

        A sampling holds until the next one, so this is a backward as-of join:
        every electrode carries the most recent bath reading at or before it.
        """
        if self._bath is None or self._electrodes is None:
            return None, None
        bath = (self._bath[self._bath["vat"] == self._vat]
                .dropna(subset=["ratio"]).sort_values("Date")
                [["Date", "ratio"]])
        hidden = [r for r, on in self._ratings_on.items() if not on]
        rows = self._electrodes[
            (self._electrodes["plating_vat"] == self._vat)
            & ~self._electrodes["quality"].isin(hidden)].sort_values("produced")
        if not len(bath) or not len(rows):
            return None, None
        paired = pd.merge_asof(
            rows, bath, left_on="produced", right_on="Date",
            direction="backward",
            tolerance=pd.Timedelta(days=BATH_TOLERANCE_DAYS))
        paired = paired.dropna(subset=["ratio", "Ni:S"])
        if not len(paired):
            return None, None
        windows = paired.groupby("Date").agg(
            bath=("ratio", "first"), coating=("Ni:S", "median"),
            electrodes=("Ni:S", "size")).reset_index()
        return paired, windows

    def _draw_scatter(self, ax):
        """Bath ratio against coating ratio, one point per bath sampling."""
        ax.set_facecolor(AXES_BG)
        paired, windows = self._paired()
        if windows is None or len(windows) < 3:
            self._blank(ax, f"Not enough paired samplings for vat {self._vat}")
            ax.set_xticks([])
            return

        # Every electrode in a window shares its bath value, so these line up
        # in vertical stripes. That is the point: it shows how few independent
        # bath readings sit behind all those electrodes.
        ax.scatter(paired["ratio"], paired["Ni:S"], s=16, color=DOT_EDGE,
                   alpha=0.22, edgecolors="none", zorder=2)

        x = windows["bath"].to_numpy(dtype=float)
        y = windows["coating"].to_numpy(dtype=float)
        sizes = 28 + 12 * windows["electrodes"].to_numpy(dtype=float)
        ax.scatter(x, y, s=sizes, color=BATH_COLORS["ratio"],
                   edgecolors=AXES_BG, linewidths=1.0, zorder=4)

        grid = np.linspace(x.min(), x.max(), 60)
        # What no relationship looks like: refit on shuffled pairings and keep
        # the middle 95% of the fits. A real line inside this band is noise.
        rng = np.random.default_rng(0)
        null = np.vstack([
            np.polyval(np.polyfit(x, rng.permutation(y), 1), grid)
            for _ in range(PERMUTATIONS)])
        low, high = np.percentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(grid, low, high, color=NULL_BAND, alpha=0.55,
                        zorder=1, label="no-relationship band (95%)")
        ax.plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color=TEXT,
                linewidth=2.0, zorder=5, label="fit")

        rho, p = stats.spearmanr(x, y)
        per_rho, per_p = stats.spearmanr(paired["ratio"], paired["Ni:S"])
        inside = ((np.polyval(np.polyfit(x, y, 1), grid) >= low)
                  & (np.polyval(np.polyfit(x, y, 1), grid) <= high)).mean()
        verdict = ("no relationship" if p >= 0.05
                   else "a relationship, but read the caveat")
        ax.text(0.015, 0.975,
                f"{len(windows)} bath samplings, {len(paired)} electrodes\n"
                f"spearman {rho:+.3f}   p = {p:.3f}   -> {verdict}\n"
                f"fit sits inside the null band {100 * inside:.0f}% of its "
                f"length\n"
                f"per electrode it would read {per_rho:+.3f} (p = {per_p:.3f}) "
                f"- inflated, they share {len(windows)} bath values",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                color=SUBTEXT, linespacing=1.5)

        ax.set_xlabel("bath nickel : thiourea  (g/L basis)", color=LABEL,
                      fontsize=11, labelpad=6)
        ax.set_ylabel("coating Ni:S   (median per sampling)", color=LABEL,
                      fontsize=11)
        ax.set_title(f"Does the bath ratio set the coating ratio?   vat "
                     f"{self._vat}", color=TEXT, fontsize=11.5,
                     fontweight="bold", loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5)
        ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                  labelcolor=SUBTEXT)

        labels = [f"{d:%d %b %Y}\nbath {b:.3f}  ->  coating {c:.3f}\n"
                  f"{int(n)} electrodes"
                  for d, b, c, n in zip(windows["Date"], windows["bath"],
                                        windows["coating"],
                                        windows["electrodes"])]
        marks, = ax.plot(x, y, linestyle="none", marker="o", markersize=0.1,
                         alpha=0)
        self._hover.add(marks, labels, BATH_COLORS["ratio"])

    @staticmethod
    def _blank(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, color=EMPTY, fontsize=11,
                style="italic")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    @staticmethod
    def _date_axis(ax, last):
        ax.tick_params(labelbottom=last)
        if not last:
            return
        # Dense ticks: the samplings are irregular, so a reader needs enough
        # dated gridlines to place a point without counting across
        locator = mdates.AutoDateLocator(minticks=9, maxticks=18)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

    def _message(self, text):
        ax = self._fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                va="center", color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        self._canvas.draw_idle()

    def _draw(self, ax, part, column, label, unit, fmt, last):
        ax.set_facecolor(AXES_BG)
        values = part.dropna(subset=[column])
        colour = BATH_COLORS[column]

        if not len(values):
            ax.text(0.5, 0.5, f"No {label.lower()} recorded", ha="center",
                    va="center", transform=ax.transAxes, color=EMPTY,
                    fontsize=11, style="italic")
        else:
            line, = ax.plot(values["Date"], values[column], color=colour,
                            linewidth=1.4, marker="o", markersize=5.5,
                            markeredgecolor=AXES_BG, markeredgewidth=0.8,
                            zorder=3)
            self._hover.add(line, [
                f"{date:%A %d %B %Y}\n{label}: {fmt.format(value)}"
                for date, value in zip(values["Date"], values[column])],
                colour)
            median = float(values[column].median())
            ax.axhline(median, color=colour, linewidth=1.1, alpha=0.55,
                       linestyle=(0, (5, 3)), zorder=2)
            ax.annotate(f"median {median:,.2f}", (1.0, median),
                        xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=9, color=SUBTEXT)

        ax.set_ylabel(unit, color=LABEL, fontsize=10.5)
        ax.set_title(f"{label}   ({len(values)} samplings)", color=TEXT,
                     fontsize=11, fontweight="bold", loc="left", pad=6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(y=0.25)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5,
                       labelbottom=last)
        self._date_axis(ax, last)


class MainWindow(QMainWindow):
    def __init__(self, sem):
        super().__init__()
        self.setWindowTitle("SEM measurements")
        tabs = QTabWidget()
        tabs.addTab(SemTab(sem), "SEM data")
        tabs.addTab(QualityTab(sem, "ratio"), "Quality vs ratio")
        tabs.addTab(QualityTab(sem, "element"), "Quality vs element")
        tabs.addTab(BathTab(load_bath(), load_electrodes(sem)),
                    "Plate bath chemistry")
        self.setCentralWidget(tabs)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    sem = load()
    for generation, cutoff in (("Gen3.1", None), ("Gen4.1", GEN4_FIRST)):
        table, elements, ratios = build_table(sem, generation, cutoff)
        print(f"{generation}: {len(table)} electrodes, elements {elements}, "
              f"ratios {ratios}")

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
            border-bottom: 2px solid {ACCENT};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
    """)
    window = MainWindow(sem)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
