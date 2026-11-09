"""PyQt6 app (opens maximized): SEM cracking verdict over production date.

Tabs: Strike (all electrodes) and Plate 28/29/30 (per strike vat).
Each tab plots production date (x) vs verdict (y): 1 = good, 0 = cracked.
Escape also closes the window.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget,
                             QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QSlider, QComboBox, QStackedWidget, QLabel,
                             QTabBar, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QHeaderView, QPushButton,
                             QFileDialog, QMessageBox, QStyle,
                             QStyleOptionHeader, QSizePolicy)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from matplotlib.lines import Line2D
from matplotlib.colors import (LinearSegmentedColormap, Normalize, to_rgb,
                               to_hex)
from matplotlib.ticker import FuncFormatter, NullLocator
import matplotlib.dates as mdates

CSV_FILE = Path(__file__).resolve().parent / "sem_measurements_filtered.csv"

# The 1-5 coat rating lives in sem_analysis/note_dictionary.py. Imported
# rather than copied: one dictionary for both apps, or they drift apart and
# the same note ends up scored two ways.
_SEM_ANALYSIS = Path(__file__).resolve().parent.parent / "sem_analysis"
if str(_SEM_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_SEM_ANALYSIS))
try:
    import note_dictionary as _notes
except ImportError:                                  # pragma: no cover
    _notes = None
    print("⚠ sem_analysis/note_dictionary.py not found : the coat rating "
          "panels will be empty")
BATH_FILE = Path(__file__).resolve().parent / "strike_bath_filtered.csv"

# Chart heights in inches (dpi=100, so x100 = pixels). Qt splits the tab's
# vertical space roughly in this ratio.
VERDICT_PLOT_HEIGHT = 3
BATH_PLOT_HEIGHT = 3.0
CHEMICALS_PLOT_HEIGHT = 3.5
PLATING_PLOT_HEIGHT = 5.0

# Style per PLOT_STYLE_GUIDE.md (trend-plot house style): grey chrome, one
# accent per quantity. Teal/violet pair stays colorblind-safe, and the y
# position (lane 1 vs 0) also separates the groups, so color never carries
# the meaning alone.
COLOR_GOOD = "#0EA5A5"     # teal,   y = 1
COLOR_CRACKED = "#8B5CF6"  # violet, y = 0
FIG_BG = "#FAFBFC"         # figure background (outside the axes)
AXES_BG = "#FFFFFF"        # plotting area
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"           # title
LABEL = "#374151"          # y label
SUBTEXT = "#6B7280"        # tick labels, stats line
EMPTY = "#9CA3AF"
TARGET_LINE = "#111827"    # recipe set-time reference on the plating tab


class HoverReadout:
    """Nearest-point tooltip for a canvas.

    Register one marker artist per series with the per-point label texts;
    on mouse move the closest visible point within HIT_RADIUS px wins and
    its label is shown in a box tinted with the series colour."""

    HIT_RADIUS = 20  # px

    def __init__(self, canvas):
        self._canvas = canvas
        self._entries = []
        self._columns = None
        self._shown = None
        # Figure-level artist, so it draws above every (twin) axes
        self._ann = Annotation(
            "", xy=(0, 0), xycoords="figure pixels",
            xytext=(14, 14), textcoords="offset pixels",
            fontsize=9, color="#FFFFFF", zorder=30,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=TEXT,
                      edgecolor="none", alpha=0.95))
        self._ann.set_visible(False)
        self._ann.set_clip_on(False)
        canvas.figure.add_artist(self._ann)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("figure_leave_event", lambda _e: self._hide())

    def add(self, artist, labels, color):
        self._entries.append((artist, list(labels), color))

    def reattach(self):
        """Re-add the annotation after a Figure.clear() stripped it, for
        canvases that redraw by clearing the whole figure. The registered
        artists belong to the discarded figure, so they are dropped too and
        the caller re-registers whatever it draws next."""
        self._entries = []
        self._columns = None
        self._canvas.figure.add_artist(self._ann)
        self._ann.set_visible(False)
        self._shown = None

    def add_columns(self, ax, centers, half_width, labels, colors):
        """Full-height day columns: hovering anywhere inside one hits it."""
        self._columns = (ax, np.asarray(centers, dtype=float), half_width,
                         list(labels), list(colors))

    def _hide(self):
        if self._shown is not None:
            self._shown = None
            self._ann.set_visible(False)
            self._canvas.draw_idle()

    def _on_move(self, event):
        if event.inaxes is None or event.x is None:
            self._hide()
            return

        best = None
        for artist, labels, color in self._entries:
            if not artist.get_visible() or not labels:
                continue
            xs = artist.get_xdata(orig=False)
            ys = artist.get_ydata(orig=False)
            if len(xs) == 0:
                continue
            pts = artist.axes.transData.transform(
                np.column_stack([np.asarray(xs, dtype=float),
                                 np.asarray(ys, dtype=float)]))
            dist = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
            i = int(np.argmin(dist))
            if dist[i] <= self.HIT_RADIUS and (best is None or dist[i] < best[0]):
                best = (dist[i], pts[i], labels[i], color)

        if best is None and self._columns is not None:
            ax, centers, half_width, labels, colors = self._columns
            if event.inaxes is ax and len(centers):
                x_data = ax.transData.inverted().transform(
                    (event.x, event.y))[0]
                i = int(np.argmin(np.abs(centers - x_data)))
                if abs(centers[i] - x_data) <= half_width:
                    # Anchor at the column's centre, high in the plot, so the
                    # tooltip holds still while the cursor moves inside it
                    bbox = ax.get_window_extent()
                    px = ax.transData.transform((centers[i], 0))[0]
                    py = bbox.y0 + 0.5 * bbox.height
                    best = (0.0, (px, py), labels[i], colors[i])

        if best is None:
            self._hide()
            return

        _d, point, label, color = best
        key = (label, tuple(point))
        if key == self._shown:
            return
        self._shown = key
        # Flip the box to the left/below near the canvas edges so it stays
        # on screen
        width, height = self._canvas.figure.bbox.width, self._canvas.figure.bbox.height
        dx = 14 if point[0] < width - 170 else -14
        dy = 14 if point[1] < height - 60 else -14
        self._ann.set_position((dx, dy))
        self._ann.set_ha("left" if dx > 0 else "right")
        self._ann.set_va("bottom" if dy > 0 else "top")
        self._ann.xy = (point[0], point[1])
        self._ann.set_text(label)
        self._ann.get_bbox_patch().set_facecolor(color)
        self._ann.set_visible(True)
        self._canvas.draw_idle()


# Production begins at B13900. Everything below it is a trial run - new vat
# commissioning, dummy electrodes, pH experiments - and does not belong in any
# of the charts. Dropped once here so every tab agrees.
FIRST_PRODUCTION = 13900


def marking_number(electrode):
    """The numeric part of a marking, so B13900 compares as 13900."""
    return pd.to_numeric(
        electrode.astype(str).str.extract(r"^[A-Za-z]*(\d+)")[0],
        errors="coerce")


def load_data():
    df = pd.read_csv(CSV_FILE)
    # 5 fine .. 1 cracked, straight off the raw SEM note
    df["quality"] = (pd.to_numeric(df["note"].map(_notes.quality),
                                   errors="coerce")
                     if _notes is not None and "note" in df.columns
                     else float("nan"))
    number = marking_number(df["electrode"])
    trials = number < FIRST_PRODUCTION
    if trials.any():
        print(f"Dropped {int(trials.sum())} trial electrodes below "
              f"B{FIRST_PRODUCTION}")
        df = df[~trials]
    df["production_date"] = pd.to_datetime(df["production_date"])
    df["verdict"] = df["sem_result"].map({"good": 1, "bad": 0})
    usable = df.dropna(subset=["verdict", "production_date"]).copy()
    dropped = len(df) - len(usable)
    if dropped:
        print(f"Dropped {dropped} electrodes without verdict or production date")
    return usable


# Day-background tint on the lane view. Red/green is exactly the pair
# colour-blind readers cannot separate, so severity rides on lightness:
# in-spec is much lighter than any breach, and the ramp darkens steadily.
# The swatch legend above the chart names every shade with its score.
BAND_CLEAN = "#A7F3D0"
BREACH_CMAP = LinearSegmentedColormap.from_list(
    "breach", ["#F87171", "#7F1D1D"])   # starts well darker than BAND_CLEAN
BAND_ALPHA = 0.55


def band_color(score, norm):
    """RGBA of a score's band, and the hex of how it looks on the white
    plot background (what the legend swatch has to match)."""
    rgba = to_rgb(BAND_CLEAN) if score == 0 else BREACH_CMAP(norm(score))[:3]
    onwhite = tuple(c * BAND_ALPHA + (1 - BAND_ALPHA) for c in rgba)
    return rgba, to_hex(onwhite)


# A bound may be None, meaning the limit is one-sided: the activator bath's
# nickel is a contaminant with upper limits only. A large finite stand-in
# keeps "which pair is wider" answerable without infinities cancelling out.
LIMIT_UNBOUNDED = 1e9


def limit_bounds(pair):
    _kind, low, high, _banded = pair
    return (low if low is not None else -LIMIT_UNBOUNDED,
            high if high is not None else LIMIT_UNBOUNDED)


def limit_width(pair):
    low, high = limit_bounds(pair)
    return high - low


def outside_limit(value, pair):
    low, high = limit_bounds(pair)
    return value < low or value > high


def breach_severity(value, pairs):
    """0 = inside the inner pair, 1 = between the pairs (kinda bad),
    2 = outside both (really bad). None when there is no reading."""
    if pairs is None or not len(pairs) or pd.isna(value):
        return None
    if outside_limit(value, max(pairs, key=limit_width)):
        return 2
    if outside_limit(value, min(pairs, key=limit_width)):
        return 1
    return 0


def short_quantity(column):
    return column.replace(" [g/L] Mean", "").replace(" Mean", "")


class LaneView:
    """The per-electrode lane chart, plus the day-background tint showing
    how far the bath chemistry was out of spec on each production day.

    Which quantities count is driven by the bath plot's checkboxes, so the
    tint follows whatever is currently plotted below."""

    def __init__(self, canvas, hover, ax, point_days, bath_df, limits):
        self.canvas = canvas
        self.hover = hover
        self._ax = ax
        self._point_days = point_days      # artist -> day number per point
        self._limits = limits or {}
        self._bands = []
        self._columns = []
        self._enabled = True
        self._scale = []                   # [(score, swatch hex), ...]
        self._scale_cb = None

        self._days = sorted({d for days in point_days.values() for d in days})
        # A measurement describes the bath from the day it was taken until
        # the next one, so bands are drawn per measurement period rather
        # than per production day — otherwise a day when nothing happened to
        # be produced would punch a white hole in a known bath state.
        self._bath_days, self._bath_rows = [], []
        if bath_df is not None and len(bath_df):
            bath = bath_df.sort_values("Date")
            self._bath_days = list(mdates.date2num(
                bath["Date"].dt.normalize().to_numpy()))
            self._bath_rows = [row for _, row in bath.iterrows()]

    def bind_scale_callback(self, callback):
        """Report the shades in use, so a legend can be drawn for them."""
        self._scale_cb = callback
        callback(self._scale)

    def set_enabled(self, on):
        """Master switch for the whole shading feature."""
        self._enabled = bool(on)
        self._render()

    def set_columns(self, columns):
        """Re-tint for the given bath quantities (the ones currently shown)."""
        self._columns = list(columns or [])
        self._render()

    def _render(self):
        for band in self._bands:
            band.remove()
        self._bands = []

        cols = ([c for c in self._columns if c in self._limits]
                if self._enabled else [])

        # Each measurement holds until the next one; the last holds to the
        # end of production. Spans are half-open in day columns, so the band
        # covers exactly the days that measurement judges.
        spans = []
        if cols and self._bath_days and self._days:
            window_end = self._days[-1] + 0.5
            for i, (day, row) in enumerate(zip(self._bath_days,
                                               self._bath_rows)):
                start = day - 0.5
                end = (self._bath_days[i + 1] - 0.5
                       if i + 1 < len(self._bath_days) else window_end)
                end = min(end, window_end)
                if end <= start:
                    continue          # superseded same-day measurement
                score = 0
                for column in cols:
                    severity = breach_severity(row.get(column),
                                               self._limits[column])
                    if severity is not None:
                        score += severity
                spans.append((start, end, score))

        scale = []
        if spans:
            highest = max(score for _s, _e, score in spans)
            norm = Normalize(vmin=1, vmax=max(2, highest))
            for start, end, score in spans:
                rgba, _hex = band_color(score, norm)
                self._bands.append(self._ax.axvspan(
                    start, end, facecolor=rgba, alpha=BAND_ALPHA,
                    edgecolor="none", zorder=0.5))
            scale = [(s, band_color(s, norm)[1]) for s in range(highest + 1)]

        self._scale = scale
        if self._scale_cb:
            self._scale_cb(scale)
        self.canvas.draw_idle()


def make_canvas(df, title, margins=(0.07, 0.97), xlim=None,
                bath_df=None, limits=None):
    fig = Figure(figsize=(11.5, VERDICT_PLOT_HEIGHT), dpi=100)
    fig.patch.set_facecolor(FIG_BG)
    ax = fig.add_subplot()
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)

    if not len(df):
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return LaneView(FigureCanvas(fig), None, ax, {}, None, None)

    n_good = int((df["verdict"] == 1).sum())
    n_cracked = int((df["verdict"] == 0).sum())
    pct_cracked = 100 * n_cracked / len(df)

    # Per-day tallies, so a tooltip can report that day as a whole
    day = df["production_date"].dt.normalize()
    day_tally = (df.assign(_day=day, _cracked=(df["verdict"] == 0).astype(int))
                 .groupby("_day")
                 .agg(cracked=("_cracked", "sum"), total=("_cracked", "size")))

    # Same-day electrodes stack vertically inside their lane; markers are
    # hollow (white face, accent edge) so dense runs don't become solid bars
    step = 0.06
    hover_specs = []
    point_days = {}
    for value, color in ((1, COLOR_GOOD), (0, COLOR_CRACKED)):
        part = df[df["verdict"] == value].sort_values("production_date")
        offsets = part.groupby("production_date").cumcount()
        y = value + (offsets - (part.groupby("production_date")["verdict"]
                                .transform("size") - 1) / 2) * step
        artist, = ax.plot(part["production_date"], y, linestyle="none",
                          marker="o", markersize=5.5,
                          markerfacecolor="#FFFFFF", markeredgecolor=color,
                          markeredgewidth=1.0, zorder=4)

        verdict_word = "good" if value == 1 else "cracked"
        labels = []
        for row in part.itertuples():
            tally = day_tally.loc[row.production_date.normalize()]
            good, cracked = int(tally.total - tally.cracked), int(tally.cracked)
            labels.append(
                f"{row.electrode} : {verdict_word}\n"
                f"{row.production_date:%d.%m.%Y}\n"
                f"that day: {good} good, {cracked} cracked")
        hover_specs.append((artist, labels, color))
        point_days[artist] = mdates.date2num(
            part["production_date"].dt.normalize().to_numpy())

    # The latest electrode gets a filled accent marker with white edge
    latest = df.loc[df["production_date"].idxmax()]
    latest_color = COLOR_GOOD if latest["verdict"] == 1 else COLOR_CRACKED
    ax.plot([latest["production_date"]], [latest["verdict"]], marker="o",
            markersize=10, markerfacecolor=latest_color,
            markeredgecolor="#FFFFFF", markeredgewidth=2.0, zorder=5)

    ax.set_title(f"{title} : {len(df)} electrodes, {n_good} good, "
                 f"{n_cracked} cracked ({pct_cracked:.0f}%)",
                 color=TEXT, fontsize=13, fontweight="bold",
                 loc="left", pad=14)
    # No axis labels: the lane tick labels name the y categories, and the
    # ticks are dates : "Date" underneath would say nothing

    ax.set_yticks([0, 1], [f"cracked (0)\nn = {n_cracked}",
                           f"good (1)\nn = {n_good}"])
    ax.set_ylim(-0.55, 1.55)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)

    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.0)

    if xlim is not None:
        ax.set_xlim(xlim)
    fig.subplots_adjust(left=margins[0], right=margins[1],
                        top=0.80, bottom=0.17)

    canvas = FigureCanvas(fig)
    hover = HoverReadout(canvas)
    for artist, labels, color in hover_specs:
        hover.add(artist, labels, color)
    return LaneView(canvas, hover, ax, point_days, bath_df, limits)


# Per-day view: zero cracked is green, then a light -> dark red ramp.
COLOR_DAY_CLEAN = "#A7F3D0"
CRACKED_CMAP = LinearSegmentedColormap.from_list(
    "cracked", ["#FCA5A5", "#7F1D1D"])


def make_day_canvas(df, title, margins=(0.07, 0.97), xlim=None):
    """One full-height column per production day, coloured by how many
    electrodes cracked that day. Days with no electrodes stay blank."""
    fig = Figure(figsize=(11.5, VERDICT_PLOT_HEIGHT), dpi=100)
    fig.patch.set_facecolor(FIG_BG)
    ax = fig.add_subplot()
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)

    if not len(df):
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return FigureCanvas(fig), None

    per_day = (df.assign(cracked=(df["verdict"] == 0).astype(int))
               .groupby(df["production_date"].dt.normalize())
               .agg(cracked=("cracked", "sum"), total=("cracked", "size")))
    worst = int(per_day["cracked"].max())
    # vmax >= 2 so a single cracked electrode isn't already the deepest red
    norm = Normalize(vmin=1, vmax=max(2, worst))

    centers, labels, tip_colors = [], [], []
    for day, row in per_day.iterrows():
        cracked, total = int(row["cracked"]), int(row["total"])
        day_num = mdates.date2num(day)
        if cracked:
            fill = CRACKED_CMAP(norm(cracked))
            tip_colors.append(CRACKED_CMAP(max(0.55, norm(cracked))))
        else:
            fill = to_rgb(COLOR_DAY_CLEAN) + (1.0,)
            tip_colors.append("#047857")
        ax.axvspan(day_num - 0.5, day_num + 0.5, facecolor=fill,
                   edgecolor=AXES_BG, linewidth=0.6, zorder=2)

        if cracked:
            # Count printed on the column, so the reading never depends on
            # colour alone (and survives red/green colour blindness)
            luminance = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
            ax.text(day_num, 0.94, str(cracked),
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8, zorder=3,
                    color="#FFFFFF" if luminance < 0.6 else "#7F1D1D")

        centers.append(day_num)
        marks = list(df.loc[(df["production_date"].dt.normalize() == day)
                            & (df["verdict"] == 0), "electrode"])
        detail = (", ".join(marks[:6]) + (f" +{len(marks) - 6} more"
                                          if len(marks) > 6 else "")
                  if marks else "none cracked")
        labels.append(f"{day:%d.%m.%Y}\n{cracked} cracked of {total}\n{detail}")

    n_cracked = int(per_day["cracked"].sum())
    n_days = len(per_day)
    clean_days = int((per_day["cracked"] == 0).sum())
    ax.set_title(f"{title} : {n_cracked} cracked over {n_days} production days "
                 f", {clean_days} clean, worst day {worst}",
                 color=TEXT, fontsize=13, fontweight="bold", loc="left", pad=14)

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)

    if xlim is not None:
        ax.set_xlim(xlim)
    fig.subplots_adjust(left=margins[0], right=margins[1],
                        top=0.80, bottom=0.17)

    canvas = FigureCanvas(fig)
    hover = HoverReadout(canvas)
    hover.add_columns(ax, centers, 0.5, labels, tip_colors)
    return canvas, hover


class VerdictPlotWidget(QWidget):
    """The tab's top chart, with a dropdown switching between the
    per-electrode lanes and the per-day cracked heat columns."""

    VIEWS = ["Electrodes: good / cracked", "Cracked per day"]

    def __init__(self, df, title, margins, xlim, bath_df=None, limits=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        picker_row = QHBoxLayout()
        picker_row.setContentsMargins(8, 0, 0, 0)
        caption = QLabel("View:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = QComboBox()
        combo.addItems(self.VIEWS)
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        picker_row.addWidget(caption)
        picker_row.addWidget(combo)

        self._shade_box = QCheckBox("Bath shading")
        self._shade_box.setChecked(True)
        self._shade_box.setToolTip(
            "Tint each production day by how far the bath chemistry shown "
            "below was out of spec")
        self._shade_box.setStyleSheet(
            f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
        self._shade_box.toggled.connect(self._on_shading_toggled)
        picker_row.addSpacing(18)
        picker_row.addWidget(self._shade_box)

        # Swatch legend: one square per score in use, its number underneath
        self._legend_row = QHBoxLayout()
        self._legend_row.setContentsMargins(0, 0, 0, 0)
        self._legend_row.setSpacing(3)
        self._legend_widgets = []
        picker_row.addSpacing(10)
        picker_row.addLayout(self._legend_row)
        picker_row.addStretch()
        layout.addLayout(picker_row)

        self._lane = make_canvas(df, title, margins, xlim, bath_df, limits)
        self._lane_canvas = self._lane.canvas
        self._lane_hover = self._lane.hover
        self._day_canvas, self._day_hover = make_day_canvas(
            df, title, margins, xlim)
        self._lane.bind_scale_callback(self._rebuild_legend)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._lane_canvas)
        self._stack.addWidget(self._day_canvas)
        combo.currentIndexChanged.connect(self._stack.setCurrentIndex)
        combo.currentIndexChanged.connect(self._sync_shading_controls)
        layout.addWidget(with_right_gutter(self._stack))

    @property
    def canvases(self):
        return (self._lane_canvas, self._day_canvas)

    def set_breach_columns(self, columns):
        """Tint the lane view's day backgrounds for these bath quantities."""
        self._lane.set_columns(columns)

    def _on_shading_toggled(self, on):
        self._lane.set_enabled(on)

    def _sync_shading_controls(self, index):
        # The shading belongs to the lane view only
        lane = index == 0
        self._shade_box.setVisible(lane)
        for widget in self._legend_widgets:
            widget.setVisible(lane)

    def _rebuild_legend(self, scale):
        """scale: [(score, swatch hex), ...] for the shades currently drawn."""
        for widget in self._legend_widgets:
            self._legend_row.removeWidget(widget)
            widget.deleteLater()
        self._legend_widgets = []

        if not scale:
            return

        caption = QLabel("out of spec:")
        caption.setStyleSheet(f"color: {SUBTEXT}; font-size: 11px;")
        caption.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._legend_row.addWidget(caption)
        self._legend_widgets.append(caption)

        for score, color in scale:
            entry = QWidget()
            column = QVBoxLayout(entry)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(1)

            swatch = QLabel()
            swatch.setFixedSize(26, 11)
            swatch.setStyleSheet(
                f"background: {color}; border: 1px solid {SPINE};")
            number = QLabel(str(score))
            number.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            number.setStyleSheet(f"color: {SUBTEXT}; font-size: 10px;")

            column.addWidget(swatch)
            column.addWidget(number)
            self._legend_row.addWidget(entry)
            self._legend_widgets.append(entry)


# (column in the filtered bath csv, label, accent, on by default)
# Accents follow the quantity mapping: teal pH, violet Ni2+, indigo Cl-/HCl,
# green ThioUrea, purple Boric Acid.
STRIKE_SERIES = [
    ("pH Mean", "pH Mean", "#0EA5A5", True),
    ("Ni2+ [g/L] Mean", "Ni²⁺ [g/L] Mean", "#8B5CF6", False),
    ("avg. ml 30% HCl/L", "avg. ml 30% HCl/L", "#6366F1", False),
]
# Decasel 5 starts checked rather than pH: the activator's pH has no
# configured limits, so opening on pH would leave the limit lines and the
# day shading with nothing to say.
ACTIVATOR_SERIES = [
    ("pH Mean", "pH Mean", "#0EA5A5", False),
    ("Decasel 5 [g/L] Mean", "Decasel 5 [g/L] Mean", "#10B981", True),
    ("Ni2+ [g/L] Mean", "Ni²⁺ [g/L] Mean", "#8B5CF6", False),
]
PLATE_SERIES = [
    ("pH Mean", "pH Mean", "#0EA5A5", True),
    ("Ni2+ [g/L] Mean", "Ni²⁺ [g/L] Mean", "#8B5CF6", False),
    ("Cl- [g/L] Mean", "Cl⁻ [g/L] Mean", "#6366F1", False),
    ("ThioUrea [g/L] Mean", "ThioUrea [g/L] Mean", "#10B981", False),
    ("Boric Acid [g/L] Mean", "Boric Acid [g/L] Mean", "#A855F7", False),
]


# --- Bath limits (see PLOT_LIMITS.md) ------------------------------------
# Colour and dash per limit kind, matching the trend app's convention.
LIMIT_STYLE = {
    "control":   ("#10B981", (0, (3, 3))),   # emerald, short dashes
    "reaction":  ("#F59E0B", (0, (6, 4))),   # amber, long dashes
    "tolerance": ("#EF4444", (0, (3, 3))),   # red, short dashes
}

# column -> [(kind, low, high, banded), ...]
# CAREFUL: "reaction" is the OUTER pair on strike but the INNER pair on plate,
# and the shaded band is the good zone on strike pH/Ni2+ but the outer
# tolerance range on strike HCl and every plate quantity.
STRIKE_LIMITS = {
    "pH Mean": [("control", -0.1, 0.1, True),
                ("reaction", -0.25, 0.25, False)],
    # Ni2+ target 38.52 g/L: control +-10 %, reaction +-25 %
    "Ni2+ [g/L] Mean": [("control", 34.668, 42.372, True),
                        ("reaction", 28.89, 48.15, False)],
    "avg. ml 30% HCl/L": [("reaction", 82.0, 134.0, False),
                          ("tolerance", 72.0, 144.0, True)],
}
# Activator bath. Decasel 5 is the ingredient (nominal 60 g/L); nickel is a
# contaminant, so it has upper limits only — hence the None lower bounds.
# pH has no configured limits here, so it is simply absent from the dict.
ACTIVATOR_LIMITS = {
    "Decasel 5 [g/L] Mean": [("reaction", 53.0, 67.0, False),
                             ("tolerance", 50.0, 70.0, True)],
    "Ni2+ [g/L] Mean": [("reaction", None, 7.0, False),
                        ("tolerance", None, 8.0, True)],
}
# One config section feeds plate baths 28, 29 and 30 alike
PLATE_LIMITS = {
    "pH Mean": [("reaction", 3.7, 4.3, False),
                ("tolerance", 3.5, 4.5, True)],
    "Ni2+ [g/L] Mean": [("reaction", 30.0, 38.0, False),
                        ("tolerance", 29.0, 40.0, True)],
    "Cl- [g/L] Mean": [("reaction", 12.0, 18.0, False),
                       ("tolerance", 10.0, 20.0, True)],
    "ThioUrea [g/L] Mean": [("reaction", 45.0, 55.0, False),
                            ("tolerance", 43.0, 57.0, True)],
    "Boric Acid [g/L] Mean": [("reaction", 34.0, 42.0, False),
                              ("tolerance", 33.0, 43.0, True)],
}


def threshold_status(value, pairs):
    """Classify a reading: a breach reports the widest bound crossed, a
    reading inside everything reports the narrowest band containing it."""
    if pd.isna(value):
        return None
    breached = [p for p in pairs if outside_limit(value, p)]
    if breached:
        pair = max(breached, key=limit_width)
        _low, high = limit_bounds(pair)
        return f"{'Above' if value > high else 'Below'} {pair[0]} limits"
    return f"Within {min(pairs, key=limit_width)[0]} limits"


class BathPlotWidget(QWidget):
    """Bath chemistry over time; one y-axis per quantity, toggled by
    checkboxes. Only pH is on by default : the quantities live on very
    different scales, so each enabled series brings its own axis.

    Limit lines are drawn per quantity but only shown when exactly one
    series is selected; with two or more the plot would be unreadable."""

    def __init__(self, bath_df, title, series, margins=(0.07, 0.92), xlim=None,
                 limits=None):
        super().__init__()
        self.SERIES = series
        self._title = title
        self._margins = margins
        self._xlim = xlim
        self._margin_cb = None
        self._limits = limits or {}
        self._limit_artists = {}
        self._ylims = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        checks_row = QHBoxLayout()
        checks_row.setContentsMargins(8, 0, 0, 0)
        self._checks = {}
        for column, label, accent, default in self.SERIES:
            box = QCheckBox(label)
            box.setChecked(default)
            box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}"
                              f"QCheckBox::indicator:checked {{ background: {accent}; "
                              f"border: 1px solid {accent}; }}")
            box.toggled.connect(lambda on, c=column: self._toggle(c, on))
            checks_row.addWidget(box)
            self._checks[column] = box
        checks_row.addStretch()
        layout.addLayout(checks_row)

        fig = Figure(figsize=(11.5, BATH_PLOT_HEIGHT), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(fig)
        layout.addWidget(with_right_gutter(self._canvas))

        ax = fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)

        if bath_df is None or not len(bath_df):
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            self._parts = {}
            return

        ax.set_title(self._title, color=TEXT, fontsize=13,
                     fontweight="bold", loc="left", pad=14)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
        ax.tick_params(axis="x", colors=SUBTEXT, labelsize=9,
                       length=4, width=0.8)
        for side in ("top",):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)

        # One axis per series: main ax carries the first series; the others
        # are twins, stacked outward on the right, so every quantity keeps
        # its own scale
        self._parts = {}
        self._hover = HoverReadout(self._canvas)
        for i, (column, label, accent, default) in enumerate(self.SERIES):
            if i == 0:
                sax = ax
                side = "left"
            else:
                sax = ax.twinx()
                side = "right"
                sax.spines["right"].set_position(("outward", 56 * (i - 1)))

            part = bath_df.dropna(subset=[column])
            artists = []
            # soft glow stroke under the line, then the line, then hollow points
            artists += sax.plot(part["Date"], part[column], color=accent,
                                linewidth=6, alpha=0.18, solid_capstyle="round",
                                zorder=2)
            artists += sax.plot(part["Date"], part[column], color=accent,
                                linewidth=2.4, solid_capstyle="round",
                                solid_joinstyle="round", zorder=3)
            artists += sax.plot(part["Date"], part[column], linestyle="none",
                                marker="o", markersize=4.5,
                                markerfacecolor="#FFFFFF",
                                markeredgecolor=accent, markeredgewidth=1.0,
                                zorder=4)

            pairs = self._limits.get(column, [])
            self._hover.add(
                artists[-1],
                [f"{label}\n{d:%d.%m.%Y}   {v:g}"
                 + (f"\n{threshold_status(v, pairs)}" if pairs else "")
                 for d, v in zip(part["Date"], part[column])],
                accent)
            self._draw_limits(sax, column, pairs)

            sax.set_ylabel(label, color=accent, fontsize=10, labelpad=8)
            sax.tick_params(axis="y", colors=SUBTEXT, labelsize=9,
                            length=4, width=0.8)
            spine = "left" if side == "left" else "right"
            sax.spines[spine].set_color(SPINE)
            for other in ("top", "right" if side == "left" else "left"):
                sax.spines[other].set_visible(False)

            self._parts[column] = {"artists": artists, "ax": sax,
                                   "spine": spine}
            self._set_visible(column, default)

        if self._xlim is not None:
            ax.set_xlim(self._xlim)
        fig.subplots_adjust(left=self._margins[0], right=self._margins[1],
                            top=0.85, bottom=0.14)
        self._refresh_limits()

    def _draw_limits(self, sax, column, pairs):
        # Capture the data-only y range first: axhline would otherwise drag
        # the autoscale out to the limit values.
        sax.autoscale_view()
        self._ylims[column] = {"data": sax.get_ylim()}
        if not pairs:
            return

        artists, handles = [], []
        for pair in pairs:
            kind, low, high, banded = pair
            color, dashes = LIMIT_STYLE[kind]
            for value in (low, high):
                if value is None:
                    continue          # one-sided limit, nothing to draw
                artists.append(sax.axhline(value, color=color, linewidth=4,
                                           alpha=0.12, zorder=1.9))
                artists.append(sax.axhline(value, color=color, linewidth=1.4,
                                           alpha=0.85, linestyle=dashes,
                                           zorder=2))
            if banded:
                # An open side is shaded to well past the axes; the y range
                # is set explicitly afterwards, so it never shows
                span_low, span_high = limit_bounds(pair)
                artists.append(sax.axhspan(span_low, span_high, color=color,
                                           alpha=0.05, zorder=0))
            # One legend entry per pair, not per line
            handles.append(Line2D([], [], color=color, linewidth=1.4,
                                  linestyle=dashes,
                                  label=f"{kind.capitalize()} limits"))

        data_lo, data_hi = self._ylims[column]["data"]
        lo = min([data_lo] + [p[1] for p in pairs if p[1] is not None])
        hi = max([data_hi] + [p[2] for p in pairs if p[2] is not None])
        pad = 0.06 * (hi - lo)
        self._ylims[column]["limits"] = (lo - pad, hi + pad)

        legend = sax.legend(handles=handles, loc="upper right", fontsize=8,
                            frameon=True, facecolor=AXES_BG, edgecolor=GRID,
                            framealpha=0.92, labelcolor=SUBTEXT)
        artists.append(legend)
        for artist in artists:
            artist.set_visible(False)
        self._limit_artists[column] = artists

    def _refresh_limits(self):
        """Limits are only readable for a single quantity: show them when
        exactly one series is selected, hide them otherwise."""
        checked = [c for c, box in self._checks.items() if box.isChecked()]
        solo = checked[0] if len(checked) == 1 else None
        for column, artists in self._limit_artists.items():
            for artist in artists:
                artist.set_visible(column == solo)
        for column, bounds in self._ylims.items():
            key = "limits" if (column == solo and "limits" in bounds) else "data"
            self._parts[column]["ax"].set_ylim(bounds[key])

    def _set_visible(self, column, on):
        part = self._parts[column]
        for artist in part["artists"]:
            artist.set_visible(on)
        part["ax"].yaxis.set_visible(on)
        part["ax"].spines[part["spine"]].set_visible(on)

    def bind_selection_callback(self, callback):
        """Report which quantities are checked, so the lane view above can
        tint its day backgrounds for the same set."""
        self._selection_cb = callback
        self._emit_selection()

    def _emit_selection(self):
        callback = getattr(self, "_selection_cb", None)
        if callback:
            callback([c for c, box in self._checks.items() if box.isChecked()])

    def bind_margin_callback(self, callback):
        """MainWindow hooks all of the tab's figures in here so the plots
        stretch to the right edge and only give up space for right-side
        axes that are actually enabled."""
        self._margin_cb = callback
        self._update_layout()

    def _update_layout(self):
        # Compact the VISIBLE right axes onto offsets 0, 56, 112... and
        # reserve right margin only for those.
        k = 0
        for i, (column, *_rest) in enumerate(self.SERIES):
            if i == 0 or column not in self._parts:
                continue
            if self._checks[column].isChecked():
                self._parts[column]["ax"].spines["right"].set_position(
                    ("outward", 56 * k))
                k += 1
        self._refresh_limits()
        self._emit_selection()
        if self._margin_cb:
            self._margin_cb(0.97 - 0.05 * k)

    def _toggle(self, column, on):
        if column in self._parts:
            self._set_visible(column, on)
            self._update_layout()
            self._canvas.draw_idle()


GUTTER_WIDTH = 24  # px; the chemicals plot's y-slider, mirrored as empty
                   # space on the other charts so all three stay date-aligned


def with_right_gutter(widget, gutter=None):
    """Wrap a canvas with a fixed-width right-side column (a control, or
    empty space matching the control on sibling charts)."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    row.addWidget(widget)
    if gutter is None:
        gutter = QWidget()
    gutter.setFixedWidth(GUTTER_WIDTH)
    row.addWidget(gutter)
    return container


# Stable color per chemical, aligned with the bath accents where the
# quantity matches (TU green, BA purple, Ni violet, Cl indigo, HCl teal).
CHEM_COLORS = {
    "TU": "#10B981",
    "BA": "#A855F7",
    "CoSO4*7H2O": "#EC4899",
    "NiSO4*6H2O": "#8B5CF6",
    "NiCl2*6H2O": "#6366F1",
    "HCl 30%": "#0EA5A5",
}
CHEM_FALLBACK_COLORS = ["#F59E0B", "#3B82F6", "#EF4444", "#64748B"]


class ChemicalsPlotWidget(QWidget):
    """Chemical additions to one vat over time. One dot per addition,
    linear kg axis, a checkbox per chemical : all on by default."""

    def __init__(self, chem_df, title, margins, xlim):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        fig = Figure(figsize=(11.5, CHEMICALS_PLOT_HEIGHT), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)

        # Vertical slider controlling the y-axis max: 100 = the autoscaled
        # default; lower values zoom into the small daily doses.
        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setRange(2, 100)
        self._slider.setValue(100)
        self._slider.setToolTip("Y-axis max (100 % = autoscale)")
        self._slider.valueChanged.connect(self._on_slider)

        has_data = chem_df is not None and len(chem_df.dropna(subset=["amount"]))
        if not has_data:
            self._slider.setEnabled(False)
            layout.addWidget(with_right_gutter(self._canvas, self._slider))
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            self._parts = {}
            return

        chem_df = chem_df.dropna(subset=["amount"])
        totals = (chem_df.groupby("chemical")["amount"].sum()
                  .sort_values(ascending=False))

        checks_row = QHBoxLayout()
        checks_row.setContentsMargins(8, 0, 0, 0)
        self._checks = {}
        self._parts = {}
        self._hover = HoverReadout(self._canvas)
        fallback = iter(CHEM_FALLBACK_COLORS * 4)
        for chemical in totals.index:
            color = CHEM_COLORS.get(chemical) or next(fallback)
            box = QCheckBox(chemical)
            box.setChecked(True)
            box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}"
                              f"QCheckBox::indicator:checked {{ background: {color}; "
                              f"border: 1px solid {color}; }}")
            box.toggled.connect(lambda on, c=chemical: self._toggle(c, on))
            checks_row.addWidget(box)
            self._checks[chemical] = box

            part = chem_df[chem_df["chemical"] == chemical]
            artists = ax.plot(part["Date"], part["amount"], linestyle="none",
                              marker="o", markersize=6, markerfacecolor=color,
                              alpha=0.85, markeredgecolor="#FFFFFF",
                              markeredgewidth=1.0, zorder=3)
            self._parts[chemical] = artists
            self._hover.add(
                artists[0],
                [f"{chemical}\n{row.Date:%d.%m.%Y}   {row.amount:g} "
                 f"{row.unit or ''}".rstrip()
                 + (f"\nadded by {row.Initials}"
                    if pd.notna(row.Initials) else "")
                 for row in part.itertuples()],
                color)
        checks_row.addStretch()
        layout.addLayout(checks_row)
        layout.addWidget(with_right_gutter(self._canvas, self._slider))

        ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold",
                     loc="left", pad=14)
        ax.set_ylabel("kg added", color=LABEL, fontsize=10, labelpad=8)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8,
                       which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
        if xlim is not None:
            ax.set_xlim(xlim)
        fig.subplots_adjust(left=margins[0], right=margins[1],
                            top=0.89, bottom=0.11)

        # Autoscaled top = the slider's 100 % reference; the bottom hugs 0
        # (amounts are never negative) and scales with the zoom
        self._ax = ax
        self._y_top = ax.get_ylim()[1]
        self._apply_top(self._y_top)

    def _apply_top(self, top):
        self._ax.set_ylim(-0.03 * top, top)
        self._canvas.draw_idle()

    def _on_slider(self, value):
        self._apply_top(self._y_top * value / 100)

    def _toggle(self, chemical, on):
        for artist in self._parts.get(chemical, []):
            artist.set_visible(on)
        self._canvas.draw_idle()


def grouped_scatter(ax, groups, column, stat, fmt, label_size=8, seed=0):
    """One jittered dot per electrode, clustered where its bar would be,
    with the mean (dashed) and median (solid) drawn across each cluster.

    A bar shows one number; this shows every electrode behind it, which is
    what makes a mean sitting far from its median obvious at a glance."""
    x = np.arange(len(groups))
    offsets = (-0.19, 0.19)
    colors = (COLOR_GOOD, COLOR_CRACKED)
    rng = np.random.default_rng(seed)    # jitter only; fixed so it is stable

    values_seen = []
    for i, offset in enumerate(offsets):
        for pos, (_name, good, cracked) in zip(x + offset, groups):
            part = good if i == 0 else cracked
            values = part[column].dropna() if len(part) else part
            if not len(values):
                continue
            values_seen.append(values)
            ax.scatter(pos + rng.uniform(-0.11, 0.11, len(values)), values,
                       s=20, color=colors[i], alpha=0.5, edgecolors="none",
                       zorder=3)
            middle = values.median() if stat == "median" else values.mean()
            ax.hlines(values.mean(), pos - 0.15, pos + 0.15,
                      color=TARGET_LINE, linewidth=1.5,
                      linestyle=(0, (4, 2)), zorder=5)
            ax.hlines(values.median(), pos - 0.15, pos + 0.15,
                      color=TARGET_LINE, linewidth=2.0, zorder=5)
            # The two labels of a pair sit at similar heights, so lift the
            # cracked one a line to keep them apart when groups are tight
            ax.annotate(fmt.format(middle), (pos, middle),
                        textcoords="offset points",
                        xytext=(0, 7 if i == 0 else 17),
                        ha="center", fontsize=label_size, color=SUBTEXT)

    # A 50x spread flattens everything onto the floor on a linear axis
    if values_seen:
        allv = pd.concat(values_seen)
        positive = allv[allv > 0]
        if len(positive) and positive.max() / positive.min() > 8:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _p: f"{v:g}"))


class StepTimeWidget(QWidget):
    """Mean time an electrode spent in a process step, cracked vs not
    cracked. Serves the plating vats (grouped 28/29/30/All) and the strike
    step (a single group, since there is only one strike).

    Everything is expressed as a percentage of each electrode's own recipe,
    so differing set times never distort the comparison."""

    PLATING_GROUPS = [("Vat 28", "28"), ("Vat 29", "29"), ("Vat 30", "30"),
                      ("All", None)]
    STRIKE_GROUPS = [("Strike", None)]
    # Rinse steps in process order, then the running total of whichever
    # of them are switched on
    RINSE_STEPS = ["14", "13", "12", "19", "20", "25", "24", "23", "22", "21"]
    RINSE_GROUPS = ([(f"Rinse {s}", s) for s in RINSE_STEPS]
                    + [("All checked", "All")])
    DRIP_STEPS = ["16", "26", "27", "15"]   # execution order
    DRIP_GROUPS = ([(f"Drip {s}", s) for s in DRIP_STEPS]
                   + [("All checked", "All")])

    def __init__(self, data, plating, groups=None, heading="Plating time",
                 what="plating vat", group_column="vats",
                 step_checkboxes=False, isolate=None):
        super().__init__()
        self.GROUPS = groups or self.PLATING_GROUPS
        # Steps whose set time is so unlike the rest that sharing an axis
        # flattens everything else (drip 26 is set to 1 second)
        self._isolate = set(isolate or [])
        self._heading = heading
        self._what = what
        # "vats" groups by the electrode's plating vat (from the SEM table);
        # "vat" groups by the step the time row itself belongs to
        self._group_column = group_column
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        self._merged = None
        self._stat = "mean"
        self._checks = {}

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        stat_caption = QLabel("Statistic:")
        stat_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        stat_combo = QComboBox()
        stat_combo.addItems(["Mean", "Median"])
        stat_combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        stat_combo.currentTextChanged.connect(self._set_stat)
        controls.addWidget(stat_caption)
        controls.addWidget(stat_combo)

        if step_checkboxes:
            controls.addSpacing(20)
            caption = QLabel("Steps:")
            caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
            controls.addWidget(caption)
            for label, value in self.GROUPS:
                if value in (None, "All"):
                    continue
                box = QCheckBox(value)
                box.setChecked(True)
                box.setStyleSheet(
                    f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
                box.toggled.connect(lambda _on: self._rebuild())
                controls.addWidget(box)
                self._checks[value] = box
        controls.addStretch()
        layout.addLayout(controls)

        # Many groups will not fit side by side, so stack the two charts
        self._stacked = len(self.GROUPS) > 5
        if self._isolate:
            height = 7.5        # two rows: shared scale, plus its own column
        elif self._stacked:
            height = 8.0
        else:
            height = PLATING_PLOT_HEIGHT
        self._fig = Figure(figsize=(11.5, height), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)

        if plating is None or not len(plating):
            self._merged = None
            ax = self._fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return

        wanted = ["electrode", "set_time", "actual_time", "overrun"]
        if "vat" in plating.columns:
            wanted.insert(1, "vat")     # rinse groups on the step itself
        merged = data.merge(plating[wanted], on="electrode", how="inner")

        # Percentages of each electrode's own recipe, so differing set times
        # never distort the comparison
        set_time = merged["set_time"].replace(0, np.nan)
        merged["actual_pct"] = 100 * merged["actual_time"] / set_time
        merged["overrun_pct"] = merged["actual_pct"] - 100
        self._merged = merged
        self._rebuild()

    def _set_stat(self, text):
        self._stat = text.lower()
        self._rebuild()

    def _aggregate(self, series):
        return series.median() if self._stat == "median" else series.mean()

    def _enabled_steps(self):
        return [value for value, box in self._checks.items()
                if box.isChecked()]

    def _totals_over(self, values):
        """Per-electrode sum across the given steps, then percentages — so
        the total reflects exactly the steps that are switched on."""
        merged = self._merged
        part = merged[merged[self._group_column].astype(str).isin(values)]
        if not len(part):
            return part
        totals = part.groupby("electrode").agg(
            set_time=("set_time", "sum"),
            actual_time=("actual_time", "sum"),
            verdict=("verdict", "first")).reset_index()
        totals["overrun"] = totals["actual_time"] - totals["set_time"]
        totals["actual_pct"] = (100 * totals["actual_time"]
                                / totals["set_time"].replace(0, np.nan))
        totals["overrun_pct"] = totals["actual_pct"] - 100
        return totals

    def _rebuild(self):
        if self._merged is None:
            return
        merged = self._merged
        self._fig.clear()

        enabled = self._enabled_steps() if self._checks else None
        groups, isolated = [], []
        for label, value in self.GROUPS:
            if value == "All":
                part = self._totals_over(enabled if enabled is not None
                                         else [])
            elif value is None:
                part = merged
            else:
                if enabled is not None and value not in enabled:
                    continue          # step switched off
                part = merged[merged[self._group_column].astype(str) == value]
            entry = (label,
                     part[part["verdict"] == 1] if len(part) else part,
                     part[part["verdict"] == 0] if len(part) else part)
            (isolated if value in self._isolate else groups).append(entry)

        if isolated:
            # Its own column, narrower since it holds a single pair
            grid = self._fig.add_gridspec(2, 2, width_ratios=[3, 1])
            ax_time = self._fig.add_subplot(grid[0, 0])
            ax_over = self._fig.add_subplot(grid[1, 0])
            ax_time_iso = self._fig.add_subplot(grid[0, 1])
            ax_over_iso = self._fig.add_subplot(grid[1, 1])
        elif self._stacked:
            ax_time, ax_over = self._fig.subplots(2, 1)
            ax_time_iso = ax_over_iso = None
        else:
            ax_time, ax_over = self._fig.subplots(1, 2)
            ax_time_iso = ax_over_iso = None

        # Same numbers twice: the left/top view says "everyone is near
        # target", the other drops the 100 % and zooms into the differences.
        stat = self._stat.capitalize()
        self._draw(ax_time, groups, "actual_pct",
                   f"Time in {self._what} : every electrode "
                   f"({stat.lower()} labelled)", "% of recipe set time",
                   target=100, fmt="{:.0f}%", style="scatter")
        self._draw(ax_over, groups, "overrun_pct",
                   f"{self._what.capitalize()} : {self._stat} ± 1 sd "
                   f"(zoomed)", "% over set time",
                   fmt="{:+.1f}%", second_column="overrun",
                   second_fmt="{:+,.0f} s", style="point")

        if isolated:
            names = ", ".join(label for label, _g, _c in isolated)
            self._draw(ax_time_iso, isolated, "actual_pct",
                       f"{names} (own scale)", "% of recipe set time",
                       target=100, fmt="{:.0f}%", style="scatter")
            self._draw(ax_over_iso, isolated, "overrun_pct",
                       f"{names} : {self._stat} ± 1 sd", "% over set time",
                       fmt="{:+.0f}%", second_column="overrun",
                       second_fmt="{:+,.0f} s", style="point")

        fig = self._fig
        # One figure-level legend, so it cannot collide with the bar labels
        fig.legend(handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_CRACKED, label="cracked"),
            Line2D([], [], color=TARGET_LINE, linewidth=1.5,
                   linestyle=(0, (4, 2)), label="mean"),
            Line2D([], [], color=TARGET_LINE, linewidth=2.0,
                   label="median")],
            loc="upper right", ncol=4, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.99, 0.995))
        if self._isolate:
            fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.09,
                                hspace=0.5, wspace=0.2)
        elif self._stacked:
            fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08,
                                hspace=0.45)
        else:
            fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.13,
                                wspace=0.22)
        self._canvas.draw_idle()

    def _draw(self, ax, groups, column, title, ylabel, target=None,
              fmt="{:,.0f}", second_column=None, second_fmt="{:,.0f}",
              style="bar"):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)

        x = np.arange(len(groups))
        # The value labels are nearly as wide as a bar, so the pair needs a
        # real gap between them or the two labels collide
        width, gap = 0.32, 0.06
        series = ((1, COLOR_GOOD, "not cracked"), (0, COLOR_CRACKED, "cracked"))
        label_size = 7 if len(groups) > 6 else 8

        for i, (verdict, color, label) in enumerate(series if style != "scatter"
                                                    else ()):
            # Points sit where the bars would, so their labels get the same
            # room — they collide otherwise once there are many groups
            offset = (-1 if i == 0 else 1) * (width / 2 + gap / 2)
            values, errors, labels = [], [], []
            for _name, good, cracked in groups:
                part = good if verdict == 1 else cracked
                value = self._aggregate(part[column]) if len(part) else np.nan
                spread = part[column].std() if len(part) else np.nan
                values.append(value)
                errors.append(0.0 if pd.isna(spread) else spread)
                if pd.isna(value):
                    labels.append("-")
                elif second_column:
                    # Secondary figure sits on the upper line, so the primary
                    # one stays closest to its bar
                    second = self._aggregate(part[second_column])
                    labels.append(f"{second_fmt.format(second)}\n"
                                  f"{fmt.format(value)}")
                else:
                    labels.append(fmt.format(value))

            if style == "point":
                # Centre point plus +-1 sd: a truncated bar would read as a
                # magnitude, and the spread is the point of this view
                ax.errorbar(x + offset, values, yerr=errors, fmt="o",
                            markersize=8, color=color,
                            markeredgecolor="#FFFFFF", markeredgewidth=1.2,
                            elinewidth=1.6, capsize=5, capthick=1.6,
                            label=label, zorder=3)
                for pos, value, err, text in zip(x + offset, values,
                                                 errors, labels):
                    if not pd.isna(value):
                        ax.annotate(text, (pos, value + err),
                                    textcoords="offset points",
                                    xytext=(0, 7), ha="center",
                                    fontsize=label_size, color=SUBTEXT,
                                    linespacing=1.4)
            else:
                bars = ax.bar(x + offset,
                              [0 if pd.isna(v) else v for v in values],
                              width, color=color, label=label, zorder=3)
                # A "101.0%" label is wider than its bar, so the pair needs
                # the gap above; labels shrink as groups multiply
                ax.bar_label(bars, labels=labels, padding=3, color=SUBTEXT,
                             fontsize=label_size, linespacing=1.4)

        if style == "scatter":
            grouped_scatter(ax, groups, column, self._stat, fmt, label_size)

        if target is not None:
            ax.axhline(target, color=SPINE, linewidth=1.4, zorder=1)

        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                     loc="left", pad=10)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=8)
        ax.set_xticks(x, [f"{name}\n{len(good)} / {len(cracked)}"
                          for name, good, cracked in groups])
        if len(groups) == 1:
            # A lone pair would otherwise stretch across the whole axes
            ax.set_xlim(-1.0, 1.0)
        # headroom for the labels (two lines, or a whisker plus a label)
        if style in ("point", "scatter"):
            ax.margins(y=0.3, x=0.15)
        else:
            ax.margins(y=0.26 if second_column else 0.16)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)


class SignalWidget(QWidget):
    """Mean value of one logged signal (temperature, current or voltage)
    an electrode saw, cracked vs not cracked.

    ProductionData writes a row only when a channel CHANGES, so averaging its
    raw rows weights a value that held for an hour the same as one that held
    ten seconds. These numbers come from signals_grid.csv, where each
    electrode has been resampled onto a 10-second grid across its plating
    plateau (build_signal_grid.py), so every point carries equal real time and
    each electrode still counts once."""

    # Strike runs well below plating on every signal, so it gets its own
    # column rather than squashing the plating differences on a shared axis.
    GROUPS = [("Vat 28", ["28"]), ("Vat 29", ["29"]), ("Vat 30", ["30"]),
              ("All plating", ["28", "29", "30"])]
    ISOLATED = [("Strike\n(vat 31)", ["31"])]
    # The cleaner runs 1001 A against plating's 709, and 47 C against 53, so
    # it cannot share an axis with either without flattening both.
    CLEANER = [("Cleaner\n(vat 17)", ["17"])]
    # The rating panel is the cleaner again, split five ways instead of two,
    # so the pair sits side by side on one scale: same electrodes, same
    # station, the only difference being how finely the coat is judged.
    RATING_VATS = ["17"]
    # Same scale and palette as gui_sem.py: 5 is the best coat and gets the
    # palest fill, 1 the darkest. One hue light to dark, because a rating is
    # a magnitude, not five unrelated categories.
    RATING_COLORS = {5: "#D6E4F7", 4: "#9FC0E8", 3: "#5B8FD1",
                     2: "#2A5BA8", 1: "#122E5C"}
    RATING_LABELS = {5: "5\nfine", 4: "4\nvery few", 3: "3\nmicro",
                     2: "2\nsmall", 1: "1\ncracked"}

    def __init__(self, data, signals, signal="Temperature", unit="°C"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        self._merged = None
        self._stat = "mean"
        self._signal = signal
        self._unit = unit
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Statistic:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = QComboBox()
        combo.addItems(["Mean", "Median"])
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_stat)
        controls.addWidget(caption)
        controls.addWidget(combo)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = fig = Figure(figsize=(11.5, 7.5), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        if signals is not None and len(signals) and "signal" in signals:
            signals = signals[signals["signal"] == signal]
        if signals is None or not len(signals):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return

        signals = signals.copy()
        signals["Marking"] = signals["Marking"].astype(str).str.strip()
        merged = data.merge(signals, left_on="electrode", right_on="Marking",
                            how="inner")
        merged["Vat"] = merged["Vat"].astype(str)
        self._merged = merged
        self._rebuild()

    def _set_stat(self, text):
        self._stat = text.lower()
        self._rebuild()

    def _aggregate(self, series):
        return series.median() if self._stat == "median" else series.mean()

    def _rebuild(self):
        if self._merged is None:
            return
        merged = self._merged
        fig = self._fig
        fig.clear()

        def build(spec):
            out = []
            for label, vats in spec:
                part = merged[merged["Vat"].isin(vats)]
                out.append((label,
                            part[part["verdict"] == 1],
                            part[part["verdict"] == 0]))
            return out

        groups = build(self.GROUPS)
        isolated, cleaner = build(self.ISOLATED), build(self.CLEANER)
        ratings = self._rating_groups(merged)

        columns = [3, 1, 1] + ([2] if ratings else [])
        grid = fig.add_gridspec(2, len(columns), width_ratios=columns)
        stat = self._stat.capitalize()
        low = self._signal.lower()
        self._draw_scatter(fig.add_subplot(grid[0, 0]), groups,
                           f"{low.capitalize()} : every electrode "
                           f"({stat.lower()} labelled)")
        self._draw_zoom(fig.add_subplot(grid[1, 0]), groups,
                        f"Plating : {self._stat} ± 1 sd (zoomed)")
        self._draw_scatter(fig.add_subplot(grid[0, 1]), isolated,
                           "Strike (own scale)")
        self._draw_zoom(fig.add_subplot(grid[1, 1]), isolated,
                        f"Strike : {self._stat} ± 1 sd")
        cleaner_top = fig.add_subplot(grid[0, 2])
        cleaner_low = fig.add_subplot(grid[1, 2])
        self._draw_scatter(cleaner_top, cleaner, "Cleaner (own scale)")
        self._draw_zoom(cleaner_low, cleaner,
                        f"Cleaner : {self._stat} ± 1 sd")
        if ratings:
            rating_top = fig.add_subplot(grid[0, 3])
            rating_low = fig.add_subplot(grid[1, 3])
            self._draw_rating_scatter(rating_top, ratings,
                                      "Cleaner by coat rating")
            self._draw_rating_zoom(
                rating_low, ratings,
                f"Cleaner by rating : {self._stat} ± 1 sd")
            # Two views of the same electrodes are only comparable on one
            # scale, so each row takes the union of its pair's range.
            for left, right in ((cleaner_top, rating_top),
                                (cleaner_low, rating_low)):
                lo = min(left.get_ylim()[0], right.get_ylim()[0])
                hi = max(left.get_ylim()[1], right.get_ylim()[1])
                left.set_ylim(lo, hi)
                right.set_ylim(lo, hi)

        fig.legend(handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_CRACKED, label="cracked"),
            Line2D([], [], color=TARGET_LINE, linewidth=1.5,
                   linestyle=(0, (4, 2)), label="mean"),
            Line2D([], [], color=TARGET_LINE, linewidth=2.0,
                   label="median")],
            loc="upper right", ncol=4, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.99, 0.995))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09,
                            hspace=0.5, wspace=0.2)
        self._canvas.draw_idle()

    @staticmethod
    def _ticks(ax, x, groups):
        ax.set_xticks(x, [f"{name}\n{len(good)} / {len(cracked)}"
                          for name, good, cracked in groups])

    def _chrome(self, ax, title):
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                     loc="left", pad=10)
        ax.set_ylabel(self._unit, color=LABEL, fontsize=10, labelpad=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _rating_groups(self, merged):
        """(label, rows, colour) per coat rating, best first, or [] when the
        rating is not available at all."""
        if "quality" not in merged.columns:
            return []
        part = merged[merged["Vat"].isin(self.RATING_VATS)]
        rating = pd.to_numeric(part["quality"], errors="coerce")
        out = []
        for level in (5, 4, 3, 2, 1):
            out.append((self.RATING_LABELS[level], part[rating == level],
                        self.RATING_COLORS[level]))
        return out if any(len(rows) for _l, rows, _c in out) else []

    @staticmethod
    def _rating_ticks(ax, x, groups):
        ax.set_xticks(x, [f"{label}\n{len(rows)}"
                          for label, rows, _color in groups])

    def _draw_rating_scatter(self, ax, groups, title):
        """The binary panels answer "did it crack"; this one keeps the
        severity, so a drift that only moves microcracks to small cracks is
        still visible."""
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        rng = np.random.default_rng(11)
        x = np.arange(len(groups))
        for index, (_label, rows, color) in enumerate(groups):
            values = rows["value_mean"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            jitter = index + rng.uniform(-0.17, 0.17, len(values))
            ax.scatter(jitter, values, s=24, color=color, alpha=0.9,
                       linewidths=0.4, edgecolors="#FFFFFF", zorder=3)
            ax.plot([index - 0.24, index + 0.24], [np.median(values)] * 2,
                    color=TARGET_LINE, linewidth=2.0, zorder=5)
            ax.plot([index - 0.24, index + 0.24], [values.mean()] * 2,
                    color=TARGET_LINE, linewidth=1.3, linestyle=(0, (4, 2)),
                    zorder=5)
            shown = (np.median(values) if self._stat == "median"
                     else values.mean())
            ax.annotate(f"{shown:.1f}", (index, values.max()),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=SUBTEXT)
        self._chrome(ax, title)
        ax.margins(y=0.3, x=0.12)
        self._rating_ticks(ax, x, groups)

    def _draw_rating_zoom(self, ax, groups, title):
        """Same drop-the-baseline view as the vat panels, one point a rating.

        A rating with a single electrode gets a point and no whisker: one
        value has no spread, and drawing a zero-length bar would read as a
        tightly controlled group rather than as n = 1."""
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        x = np.arange(len(groups))
        for index, (_label, rows, color) in enumerate(groups):
            values = rows["value_mean"].dropna()
            if not len(values):
                continue
            middle = self._aggregate(values)
            spread = values.std() if len(values) > 1 else np.nan
            ax.errorbar([index], [middle],
                        yerr=[[0 if pd.isna(spread) else spread]] * 2,
                        fmt="o", markersize=8, color=color,
                        markeredgecolor="#FFFFFF", markeredgewidth=1.2,
                        elinewidth=1.6, capsize=5, capthick=1.6, zorder=3)
            ax.annotate(f"{middle:.1f}",
                        (index, middle + (0 if pd.isna(spread) else spread)),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=SUBTEXT)
        self._chrome(ax, title)
        ax.margins(y=0.25, x=0.12)
        self._rating_ticks(ax, x, groups)

    def _draw_scatter(self, ax, groups, title):
        """Every electrode as a dot, with mean and median across each
        cluster — the distribution is the point, not a single bar."""
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        x = np.arange(len(groups))

        grouped_scatter(ax, groups, "value_mean", self._stat, "{:.1f}")

        self._chrome(ax, title)
        ax.margins(y=0.3, x=0.15)
        if len(groups) == 1:
            ax.set_xlim(-1.0, 1.0)
        self._ticks(ax, x, groups)

    def _draw_zoom(self, ax, groups, title):
        """Zero-based bars hide a 1 °C difference at 50 °C, so this view
        drops the baseline — points with ±1 sd rather than bars, since a
        truncated bar would misread as a magnitude."""
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        x = np.arange(len(groups))

        for i, (verdict, color) in enumerate(((1, COLOR_GOOD),
                                              (0, COLOR_CRACKED))):
            offset = -0.12 if i == 0 else 0.12
            values, errors = [], []
            for _name, good, cracked in groups:
                part = good if verdict == 1 else cracked
                values.append(self._aggregate(part["value_mean"])
                              if len(part) else np.nan)
                errors.append(part["value_mean"].std() if len(part) else np.nan)
            ax.errorbar(x + offset, values, yerr=errors, fmt="o",
                        markersize=8, color=color, markeredgecolor="#FFFFFF",
                        markeredgewidth=1.2, elinewidth=1.6, capsize=5,
                        capthick=1.6, zorder=3)
            for pos, value, err in zip(x + offset, values, errors):
                if not pd.isna(value):
                    ax.annotate(f"{value:.1f}", (pos, value + (err or 0)),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=8, color=SUBTEXT)

        self._chrome(ax, title)
        ax.margins(y=0.25, x=0.15)
        if len(groups) == 1:
            ax.set_xlim(-1.0, 1.0)
        self._ticks(ax, x, groups)


def rank_sum_p(a, b):
    """Two-sided Mann-Whitney U p-value, normal approximation with a tie
    correction. scipy is not a dependency of this repo, and at these sample
    sizes (tens per group) the approximation is accurate well past the
    precision anyone reads off a chart."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None
    both = np.concatenate([a, b])
    ranks = pd.Series(both).rank().to_numpy()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    _values, counts = np.unique(both, return_counts=True)
    n = n1 + n2
    ties = float((counts ** 3 - counts).sum())
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return None
    # Continuity correction, floored at 0 so a dead heat returns p = 1
    # rather than overshooting past it
    z = max(0.0, abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


# The coworker's candidate combination. Pass matters: step 31 runs twice and
# only its short second dip is in the hypothesis, so a component is keyed
# (step, pass) rather than by step alone.
CUMULATIVE_STEPS = [(19, 1, "19"), (20, 1, "20"), (25, 1, "25"),
                    (26, 1, "26"), (31, 2, "31 2nd")]
# One hue, light to dark, in process order: the stack is a single quantity
# split into contributions, not five unrelated categories. A ramp also keeps
# the segments clear of the teal/violet the verdict wears on every other tab,
# so the two encodings never compete for the same reading.
CUMULATIVE_COLORS = ("#BFDBFE", "#93C5FD", "#60A5FA", "#2563EB", "#1E3A8A")
CUMULATIVE_BINS = 8          # equal-count bins for the rate panel

# Bath chemistry joined per electrode: the most recent sampling of its own
# plating vat at or before it was produced, a reading holding until the next
# one. Feeds the charge report and the date sort.
CHEM_TRACKS = [("Ni", "Ni2+ [g/L] Mean", "g/L"),
               ("TU", "ThioUrea [g/L] Mean", "g/L"),
               ("Cl", "Cl- [g/L] Mean", "g/L"),
               ("pH", "pH Mean", "")]
CHEM_TOLERANCE_DAYS = 14
CUMULATIVE_CLIP_PCT = 97     # hard ceiling on the clip, whatever the fence says


def build_chemistry(data, baths):
    """One row per electrode: its plating vat and the bath reading it was made
    under. Returns None when the bath files are not available."""
    if baths is None or "production_date" not in data.columns:
        return None
    frames = []
    for vat in ("28", "29", "30"):
        bath = baths.get(f"Plate {vat}")
        if bath is None or not len(bath):
            continue
        keep = ["Date"] + [c for _n, c, _u in CHEM_TRACKS
                           if c in bath.columns]
        part = bath[keep].copy()
        part["vats"] = pd.Series([vat] * len(part), index=part.index,
                                 dtype="string")
        frames.append(part)
    if not frames:
        return None
    bath = pd.concat(frames, ignore_index=True).sort_values("Date")

    left = data[["electrode", "production_date", "vats"]].dropna(
        subset=["production_date"]).copy()
    left["vats"] = (left["vats"].astype("string")
                    .str.split("|").str[0].astype("string"))
    left = left.sort_values("production_date")
    joined = pd.merge_asof(
        left, bath, left_on="production_date", right_on="Date",
        by="vats", direction="backward",
        tolerance=pd.Timedelta(days=CHEM_TOLERANCE_DAYS))
    return joined.set_index("electrode")


class NumericItem(QTableWidgetItem):
    """A cell that shows formatted text but sorts on its number.

    QTableWidgetItem compares the DisplayRole, and setText() overwrites that
    role - so a cell displaying "1,197" sorted as the string it is, landing
    before "182". Keeping the value in UserRole fixes the ordering and gives
    the spreadsheet export a real number to write.
    """

    def __init__(self, value, text):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, float(value))
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight
                              | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        mine = self.data(Qt.ItemDataRole.UserRole)
        theirs = other.data(Qt.ItemDataRole.UserRole)
        if mine is None or theirs is None:
            return super().__lt__(other)
        return mine < theirs


# Charge report: the coworker's sheet, worst coat first inside each vat.
REPORT_STEPS = [(19, 1, "Pos. 19"), (20, 1, "Pos. 20"), (25, 1, "Pos. 25"),
                (26, 1, "Pos. 26"), (31, 2, "Pos. 31")]
# Best first; anything unrecognised sorts to the bottom of its vat
COAT_ORDER = ["Fine", "Microcracks", "Microcracks/small cracks",
              "Small cracks", "Less cracked", "Cracked", "Large cracks"]


class MultiLineHeader(QHeaderView):
    """A header that honours newlines in its labels.

    QHeaderView hands the whole label to the style as a single string and the
    style draws it on one line, so a "\n" neither wraps nor takes any height:
    the label runs on until it is elided. This paints the label itself, one
    line per "\n", and measures a section against its widest line rather than
    against the whole run-on string, which is what keeps the columns narrow."""

    PAD_X, PAD_Y = 12, 10

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self.setSortIndicatorShown(True)
        # Set by the caller when a stylesheet colour has to be matched
        self.label_color = None

    def _lines(self, index):
        model = self.model()
        if model is None:
            return [""]
        text = model.headerData(index, self.orientation(),
                                Qt.ItemDataRole.DisplayRole)
        return str("" if text is None else text).split("\n")

    def _rows(self):
        return max([len(self._lines(i)) for i in range(self.count())] or [1])

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(self.fontMetrics().height() * self._rows() + self.PAD_Y)
        return size

    def sectionSizeFromContents(self, index):
        metrics = self.fontMetrics()
        width = max(metrics.horizontalAdvance(line)
                    for line in self._lines(index))
        return QSize(width + self.PAD_X,
                     metrics.height() * self._rows() + self.PAD_Y)

    def paintSection(self, painter, rect, index):
        option = QStyleOptionHeader()
        self.initStyleOptionForIndex(option, index)
        option.rect = rect
        # Blank the label so the style paints only the background and the
        # sort indicator; placing the text is ours to do.
        option.text = ""
        painter.save()
        self.style().drawControl(QStyle.ControlElement.CE_Header, option,
                                 painter, self)
        painter.restore()

        painter.save()
        painter.setPen(self.label_color or self.palette().color(
            QPalette.ColorRole.ButtonText))
        metrics = painter.fontMetrics()
        lines = self._lines(index)
        step = metrics.height()
        top = rect.top() + (rect.height() - step * len(lines)) // 2
        for number, line in enumerate(lines):
            painter.drawText(
                QRect(rect.left() + 4, top + number * step,
                      rect.width() - 8, step),
                int(Qt.AlignmentFlag.AlignCenter), line)
        painter.restore()


class ChargeReportWidget(QWidget):
    """The charge report as a table: one row per electrode, grouped by vat and
    ordered worst coat first.

    Three blocks, as on the paper sheet: what was plated, the overrun at each
    position, and the bath composition it was plated in. Bath figures come
    from that vat's most recent sampling at or before the electrode, so
    electrodes made between two samplings share a value. A blank cell is a
    measurement that does not exist, never a zero."""

    COLUMNS = ([("electrode", "Batch #"),
                ("date", "Date"),
                ("vat", "Vat")]
               # "overrun", spelled out: these are actual minus recipe, not
               # how long the step took. Step 26 in particular is set to 1 s,
               # so its whole duration reads as overrun.
               + [(f"pos{step}", f"{label}\noverrun [sec.]")
                  for step, _p, label in REPORT_STEPS]
               + [("Ni", "[Ni2+]\ng/L (age)"),
                  ("TU", "[TU]\ng/L (age)"),
                  ("Cl", "[Cl-]\ng/L (age)"),
                  ("coat", "Coat integrity")])

    def __init__(self, data, long_steps, baths=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        self._widget = None
        self._ages = {}
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Charge report")
        heading.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        header.addWidget(heading)
        header.addStretch()
        save = QPushButton("Export (PDF, Excel, CSV)")
        save.setStyleSheet(
            f"QPushButton {{ color: {TEXT}; font-size: 12px; "
            f"padding: 5px 14px; border: 1px solid {SPINE}; "
            f"border-radius: 4px; background: {AXES_BG}; }}"
            f"QPushButton:hover {{ border-color: {COLOR_GOOD}; }}")
        save.clicked.connect(self._save_pdf)
        header.addWidget(save)
        layout.addLayout(header)

        table = self._build(data, long_steps, baths)
        if table is None or not len(table):
            layout.addWidget(QLabel("No electrodes to report."))
            return
        # Keyed by marking, not row, so a re-sort cannot shuffle it
        self._ages = dict(zip(table["electrode"], table["bath_age"]))
        self._widget = self._table(table)
        layout.addWidget(self._widget)

    def _save_pdf(self):
        """Export exactly what is on screen, in the order it is shown.

        Read back out of the table rather than re-derived, so a re-sort by the
        user is what lands in the file."""
        if self._widget is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save charge report", "charge_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        widget = self._widget
        titles = [widget.horizontalHeaderItem(c).text().replace("\n", " ")
                  for c in range(widget.columnCount())]
        # The PDF columns are too narrow to carry the word as well, so it goes
        # there and stays in the spreadsheets, where the width costs nothing.
        pdf_titles = [title.replace(" overrun", "") for title in titles]
        # Two readings of the same cells: the PDF wants the text as shown, the
        # spreadsheet wants real numbers so it can be sorted and totalled
        shown, typed = [], []
        for r in range(widget.rowCount()):
            cells = [widget.item(r, c) for c in range(widget.columnCount())]
            shown.append([cell.text() if cell else "" for cell in cells])
            typed.append([self._cell_value(cell) for cell in cells])

        try:
            self._write_pdf(path, pdf_titles, shown)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(self, "Could not save",
                                f"{type(exc).__name__}: {exc}")
            return

        book = str(Path(path).with_suffix(".xlsx"))
        sheet = str(Path(path).with_suffix(".csv"))
        try:
            frame = self._export_frame(titles, typed)
            self._write_excel(book, frame)
            frame.to_csv(sheet, index=False)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(
                self, "PDF saved, the rest did not",
                f"The PDF is at\n{path}\n\n{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(
            self, "Saved",
            f"{len(shown)} rows written to\n{path}\n{book}\n{sheet}")

    @staticmethod
    def _cell_value(cell):
        """The number behind a cell where there is one, else its text."""
        if cell is None:
            return None
        number = cell.data(Qt.ItemDataRole.UserRole)
        if number is not None:
            return number
        text = cell.text()
        return text if text else None

    def _export_frame(self, titles, rows):
        """The table as data: real numbers, and the bath age as its own column.

        On screen the age rides along in the cell text ("31.5  (2)"), which is
        readable but useless to a spreadsheet. Here the values stay numeric and
        the age gets a column of its own."""
        # An empty string in a numeric column makes the whole column text in
        # Excel, which kills sorting and totals; None becomes a real blank
        clean = [[None if value == "" else value for value in row]
                 for row in rows]
        frame = pd.DataFrame(clean, columns=titles)

        keys = [key for key, _title in self.COLUMNS]
        markings = frame[titles[keys.index("electrode")]]
        ages = [self._ages.get(mark) for mark in markings]
        ages = [None if age is None or pd.isna(age) else int(age)
                for age in ages]
        frame.insert(keys.index("Cl") + 1, "Bath reading age [days]", ages)
        return frame

    @staticmethod
    def _write_excel(path, frame):
        """One sheet, header frozen and columns sized to fit."""
        from openpyxl.utils import get_column_letter

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Charge report")
            sheet = writer.sheets["Charge report"]
            sheet.freeze_panes = "A2"
            for number, title in enumerate(frame.columns, start=1):
                values = frame[title].dropna()
                longest = max([len(str(title))]
                              + [len(f"{value}") for value in values] or [0])
                sheet.column_dimensions[get_column_letter(number)].width = (
                    min(30, longest + 3))

    @staticmethod
    def _write_pdf(path, titles, rows, per_page=30):
        """A4 landscape, paginated, with the header repeated on every page."""
        pages = max(1, -(-len(rows) // per_page))
        with PdfPages(path) as pdf:
            for page in range(pages):
                chunk = rows[page * per_page:(page + 1) * per_page]
                fig = Figure(figsize=(11.69, 8.27), dpi=200)
                fig.patch.set_facecolor("#FFFFFF")
                ax = fig.add_subplot()
                ax.axis("off")
                table = ax.table(cellText=chunk, colLabels=titles,
                                 cellLoc="center", loc="upper center")
                table.auto_set_font_size(False)
                table.set_fontsize(6.5)
                table.scale(1.0, 1.35)
                for (row, _col), cell in table.get_celld().items():
                    cell.set_edgecolor("#B8BEC7")
                    cell.set_linewidth(0.4)
                    if row == 0:
                        cell.set_facecolor("#EEF1F5")
                        cell.set_text_props(fontweight="bold")
                    elif row % 2 == 0:
                        cell.set_facecolor("#FAFBFC")
                fig.suptitle(f"Charge report   ({len(rows)} electrodes)   "
                             f"page {page + 1} of {pages}",
                             fontsize=10, fontweight="bold", x=0.08,
                             ha="left", y=0.965)
                fig.subplots_adjust(left=0.02, right=0.98, top=0.92,
                                    bottom=0.03)
                pdf.savefig(fig)

    @staticmethod
    def _build(data, long_steps, baths):
        if long_steps is None or not len(long_steps):
            return None
        steps = long_steps.copy()
        steps["electrode"] = steps["electrode"].astype(str).str.strip()
        steps["key"] = list(zip(steps["step"], steps["pass"]))
        wanted = {(step, p): f"pos{step}" for step, p, _l in REPORT_STEPS}
        steps = steps[steps["key"].isin(wanted)]
        overruns = steps.pivot_table(index="electrode", columns="key",
                                     values="overrun", aggfunc="first")
        overruns.columns = [wanted[k] for k in overruns.columns]

        frame = data[["electrode", "production_date", "vats"]].copy()
        frame["electrode"] = frame["electrode"].astype(str).str.strip()
        coat = ("note_corrected" if "note_corrected" in data.columns
                else "sem_result")
        frame["coat"] = data[coat].values
        frame["vat"] = frame["vats"].astype(str).str.split("|").str[0]
        frame["date"] = pd.to_datetime(frame["production_date"],
                                       errors="coerce")
        frame = frame.set_index("electrode").join(overruns)

        chem = build_chemistry(data, baths)
        frame["bath_age"] = float("nan")
        if chem is not None:
            renames = {c: n for n, c, _u in CHEM_TRACKS}
            frame = frame.join(chem[[c for c in renames if c in chem.columns]]
                               .rename(columns=renames))
            # How stale the reading was when this electrode was plated. The
            # bath is sampled every few days, so a value can be a week old.
            sampled = chem["Date"].reindex(frame.index)
            frame["bath_age"] = (frame["date"] - sampled).dt.days
        for name in ("Ni", "TU", "Cl"):
            if name not in frame.columns:
                frame[name] = float("nan")

        rank = {name: i for i, name in enumerate(COAT_ORDER)}
        frame["_rank"] = frame["coat"].map(rank).fillna(len(COAT_ORDER))
        frame = frame.sort_values(["vat", "_rank", "date"])
        return frame.reset_index()

    def _table(self, frame):
        widget = QTableWidget()
        widget.setColumnCount(len(self.COLUMNS))
        widget.setHorizontalHeaderLabels(
            [title for _key, title in self.COLUMNS])
        widget.setRowCount(len(frame))
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
            f"QHeaderView::section {{ background: {FIG_BG}; "
            f"border: none; border-bottom: 1px solid {SPINE}; }}")

        seconds = {f"pos{step}" for step, _p, _l in REPORT_STEPS}
        grams = {"Ni", "TU", "Cl"}
        for row, (_i, record) in enumerate(frame.iterrows()):
            for column, (key, _title) in enumerate(self.COLUMNS):
                value = record.get(key)
                item = QTableWidgetItem()
                if value is None or (not isinstance(value, str)
                                     and pd.isna(value)):
                    # Blank, never 0: a missing reading is not a measured zero
                    item.setData(Qt.ItemDataRole.DisplayRole, "")
                elif key == "date":
                    item.setData(Qt.ItemDataRole.DisplayRole,
                                 f"{value:%Y-%m-%d}")
                elif key in seconds:
                    item = NumericItem(value, f"{value:,.0f}")
                elif key in grams:
                    age = record.get("bath_age")
                    stale = ("" if age is None or pd.isna(age)
                             else f"  ({int(age)})")
                    item = NumericItem(value, f"{value:,.1f}{stale}")
                else:
                    item.setData(Qt.ItemDataRole.DisplayRole, str(value))
                if key == "coat" and str(value) in ("Cracked", "Large cracks"):
                    item.setForeground(QColor(COLOR_CRACKED))
                widget.setItem(row, column, item)

        header = MultiLineHeader(widget)
        # A section is measured with the widget font, so the label font has to
        # live there and not in the stylesheet, or the two disagree.
        font = header.font()
        font.setPixelSize(11)
        font.setBold(True)
        header.setFont(font)
        header.setMinimumSectionSize(44)
        header.label_color = QColor(LABEL)
        widget.setHorizontalHeader(header)
        # Switching sorting on makes the view sort by whatever the sort
        # indicator says, and a fresh header says "column 0, descending" -
        # which silently replaced the vat/coat order with batch number.
        # Section -1 means unsorted, so the rows keep the order they were
        # inserted in until somebody actually clicks a heading.
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        widget.setSortingEnabled(True)
        for column, (key, _t) in enumerate(self.COLUMNS):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch if key == "coat"
                else QHeaderView.ResizeMode.ResizeToContents)
        return widget


class CumulativeOverrunWidget(QWidget):
    """Every electrode's summed overrun across a chosen set of steps, stacked
    by the step that contributed it, sorted low to high, with the SEM verdict
    laid directly underneath and the cracked rate per equal-count bin below.

    This is the "no single vat runs late enough to matter, but the sum does"
    hypothesis drawn out. If it holds, cracked ticks crowd the right-hand end
    of the rug and the rate panel climbs left to right. A flat rate panel means
    the sum carries no signal, however tall the stack gets.

    Two readings of the same stack: overrun in seconds, and overrun as a share
    of the combined recipe time of exactly the steps that are switched on."""

    def __init__(self, data, long_steps, baths=None,
                 heading="Cumulative overrun"):
        super().__init__()
        self._chem = self._build_chemistry(data, baths)
        self._heading = heading
        self._percent = False
        self._clip = True
        self._by_date = False
        self._vat = None                 # None means every vat
        self._checks = {}
        # Taken from the SEM table rather than the bath join, so the filter
        # works even when no bath files are loaded
        self._vats = (data.drop_duplicates("electrode")
                      .set_index("electrode")["vats"]
                      .astype(str).str.split("|").str[0])
        self._overrun = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        unit_caption = QLabel("Unit:")
        unit_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        unit_combo = QComboBox()
        unit_combo.addItems(["Seconds", "% of set time"])
        unit_combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        unit_combo.currentTextChanged.connect(self._set_unit)
        controls.addWidget(unit_caption)
        controls.addWidget(unit_combo)

        controls.addSpacing(20)
        vat_caption = QLabel("Vat:")
        vat_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        vat_combo = QComboBox()
        vat_combo.addItems(["All vats", "Vat 28", "Vat 29", "Vat 30"])
        vat_combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        vat_combo.currentTextChanged.connect(self._set_vat)
        controls.addWidget(vat_caption)
        controls.addWidget(vat_combo)

        controls.addSpacing(20)
        steps_caption = QLabel("Steps:")
        steps_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        controls.addWidget(steps_caption)
        for step, pass_no, label in CUMULATIVE_STEPS:
            box = QCheckBox(label)
            box.setChecked(True)
            box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
            box.toggled.connect(lambda _on: self._rebuild())
            controls.addWidget(box)
            self._checks[(step, pass_no)] = box

        controls.addSpacing(20)
        clip_box = QCheckBox("Clip tail")
        clip_box.setChecked(True)
        clip_box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
        clip_box.toggled.connect(self._set_clip)
        controls.addWidget(clip_box)

        controls.addSpacing(16)
        date_box = QCheckBox("Sort by date")
        date_box.setChecked(False)
        date_box.setStyleSheet(f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}")
        date_box.toggled.connect(self._set_sort)
        controls.addWidget(date_box)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = Figure(figsize=(12.5, 9.0), dpi=100)
        self._fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._hover = HoverReadout(self._canvas)
        layout.addWidget(self._canvas)

        if long_steps is None or not len(long_steps):
            self._empty("No step-time data available")
            return

        keys = [(step, pass_no) for step, pass_no, _label in CUMULATIVE_STEPS]
        steps = long_steps[["electrode", "step", "pass", "set_time",
                            "overrun"]].copy()
        steps["key"] = list(zip(steps["step"], steps["pass"]))
        steps = steps[steps["key"].isin(keys)]
        if not len(steps):
            self._empty("None of these steps are in the step-time file")
            return

        self._overrun = steps.pivot_table(index="electrode", columns="key",
                                          values="overrun", aggfunc="first")
        self._set_time = steps.pivot_table(index="electrode", columns="key",
                                           values="set_time", aggfunc="first")
        self._verdict = data.drop_duplicates(
            "electrode").set_index("electrode")["verdict"]
        self._rebuild()

    @staticmethod
    def _build_chemistry(data, baths):
        return build_chemistry(data, baths)

    @staticmethod
    def _unused_build_chemistry(data, baths):
        """One row per electrode: its plating vat and the bath reading it was
        made under. Returns None when the bath files are not available."""
        if baths is None or "production_date" not in data.columns:
            return None
        frames = []
        for vat in ("28", "29", "30"):
            bath = baths.get(f"Plate {vat}")
            if bath is None or not len(bath):
                continue
            keep = ["Date"] + [c for _n, c, _u in CHEM_TRACKS
                               if c in bath.columns]
            part = bath[keep].copy()
            part["vats"] = pd.Series([vat] * len(part),
                                     index=part.index, dtype="string")
            frames.append(part)
        if not frames:
            return None
        bath = pd.concat(frames, ignore_index=True).sort_values("Date")

        left = data[["electrode", "production_date", "vats"]].dropna(
            subset=["production_date"]).copy()
        left["vats"] = (left["vats"].astype("string")
                        .str.split("|").str[0].astype("string"))
        left = left.sort_values("production_date")
        joined = pd.merge_asof(
            left, bath, left_on="production_date", right_on="Date",
            by="vats", direction="backward",
            tolerance=pd.Timedelta(days=CHEM_TOLERANCE_DAYS))
        return joined.set_index("electrode")

    def _empty(self, message):
        ax = self._fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        self._canvas.draw_idle()

    def _set_unit(self, text):
        self._percent = text.startswith("%")
        self._rebuild()

    def _set_clip(self, on):
        self._clip = bool(on)
        self._rebuild()

    def _set_sort(self, on):
        self._by_date = bool(on)
        self._rebuild()

    def _set_vat(self, text):
        self._vat = None if text.startswith("All") else text.split()[-1]
        self._rebuild()

    def _label_of(self, key):
        for step, pass_no, label in CUMULATIVE_STEPS:
            if (step, pass_no) == key:
                return label
        return str(key)

    def _color_of(self, key):
        keys = [(s, p) for s, p, _l in CUMULATIVE_STEPS]
        return CUMULATIVE_COLORS[keys.index(key) % len(CUMULATIVE_COLORS)]

    def _enabled(self):
        """Checked steps, back in process order so the stack layers the same
        way no matter which boxes were clicked."""
        checked = {key for key, box in self._checks.items() if box.isChecked()}
        return [key for key in [(s, p) for s, p, _l in CUMULATIVE_STEPS]
                if key in checked]

    def _assemble(self, enabled):
        """Per-electrode segments and set totals for exactly the steps that
        are on.

        An electrode missing any enabled step is dropped rather than counted
        as zero: a sum is only comparable when every term is present."""
        overrun = self._overrun.reindex(columns=enabled)
        set_time = self._set_time.reindex(columns=enabled)
        complete = overrun.notna().all(axis=1) & set_time.notna().all(axis=1)
        overrun, set_time = overrun[complete], set_time[complete]

        recipe = set_time.sum(axis=1)
        if self._percent:
            # Share of the combined recipe of the switched-on steps, so the
            # denominator always matches the selection. Per-step relative
            # overrun would be useless here: step 26 is set to 1 second, so
            # its own ratio runs to ~2800% and swamps every other segment.
            segments = 100 * overrun.div(recipe.replace(0, np.nan), axis=0)
        else:
            segments = overrun

        frame = segments.dropna()
        verdict = self._verdict.reindex(frame.index)
        keep = verdict.notna()
        frame = frame[keep]
        return frame, verdict[keep].astype(int), recipe.reindex(frame.index)

    def _fmt(self, value):
        return f"{value:,.0f}%" if self._percent else f"{value:,.0f} s"

    def _rebuild(self):
        if self._overrun is None:
            return
        self._fig.clear()
        self._hover.reattach()
        enabled = self._enabled()
        if not enabled:
            self._empty("No steps selected")
            return

        frame, verdict, recipe = self._assemble(enabled)
        if self._vat is not None:
            keep = self._vats.reindex(frame.index) == self._vat
            frame = frame[keep.fillna(False)]
            verdict, recipe = verdict.loc[frame.index], recipe.loc[frame.index]
        if not len(frame):
            self._empty("No electrodes have all of the selected steps"
                        + ("" if self._vat is None
                           else f" in vat {self._vat}"))
            return

        total = frame.sum(axis=1)
        # The rate panel bins by overrun magnitude and needs its own ascending
        # order; the stack and the tracks follow whichever sort is selected
        by_overrun = total.sort_values()
        order_dates = None
        if self._by_date and self._chem is not None:
            dates = self._chem["production_date"].reindex(total.index)
            order = dates.sort_values(na_position="last").index
            order_dates = dates.loc[order]
        else:
            order = by_overrun.index
        frame, verdict = frame.loc[order], verdict.loc[order]
        total, recipe = total.loc[order], recipe.loc[order]
        rate_total = by_overrun
        rate_verdict = verdict.reindex(by_overrun.index)
        x = np.arange(len(order))

        grid = self._fig.add_gridspec(
            3, 1, height_ratios=[5.0, 0.42, 2.3], hspace=0.16,
            left=0.075, right=0.985, top=0.925, bottom=0.075)
        ax = self._fig.add_subplot(grid[0])
        rug = self._fig.add_subplot(grid[1], sharex=ax)
        rate = self._fig.add_subplot(grid[2])

        self._draw_stack(ax, x, frame, enabled, total)
        self._draw_rug(rug, x, verdict, order_dates)
        self._draw_rate(rate, rate_total, rate_verdict)
        self._draw_title(frame, verdict, total, recipe, enabled)
        self._bind_hover(ax, x, frame, verdict, total, enabled)
        self._canvas.draw_idle()

    def _draw_stack(self, ax, x, frame, enabled, total):
        ax.set_facecolor(AXES_BG)
        self._clipped = None
        # Positive and negative runs keep their own baseline, so the one
        # electrode that finished a step early still stacks correctly
        up, down = np.zeros(len(x)), np.zeros(len(x))
        for key in enabled:
            values = frame[key].to_numpy(dtype=float)
            ax.bar(x, values, width=1.0,
                   bottom=np.where(values >= 0, up, down),
                   color=self._color_of(key), edgecolor="none", zorder=3,
                   label=f"step {self._label_of(key)}")
            up = up + np.clip(values, 0, None)
            down = down + np.clip(values, None, 0)

        values = total.to_numpy(dtype=float)
        cap = None
        if self._clip and len(values) > 10:
            # The Tukey upper fence, not a percentile: total overrun is skewed
            # hard enough that even the 97th percentile leaves the bulk of the
            # electrodes squashed into the bottom few pixels. The percentile
            # still caps the cap, for data with no real tail to clip.
            q1, q3 = np.percentile(values, [25, 75])
            candidate = min(q3 + 1.5 * (q3 - q1),
                            float(np.percentile(values, CUMULATIVE_CLIP_PCT)))
            # Only worth clipping when the tail really does flatten the rest
            if candidate > 0 and values.max() > 1.6 * candidate:
                cap = candidate
        if cap is not None:
            over = values > cap
            ax.set_ylim(min(0.0, float(down.min()) * 1.05), cap * 1.08)
            ax.plot(x[over], np.full(int(over.sum()), cap * 1.02), linestyle="",
                    marker="^", markersize=5, color=CUMULATIVE_COLORS[-1],
                    clip_on=False, zorder=6)
            self._clipped = (int(over.sum()), cap, float(values.max()))
        else:
            ax.margins(y=0.12)

        ax.set_ylabel("Overrun, % of combined set time" if self._percent
                      else "Overrun (s)", color=LABEL, fontsize=11)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v:,.0f}%" if self._percent else f"{v:,.0f}"))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelbottom=False)
        ax.set_xlim(-0.5, len(x) - 0.5)
        legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5,
                           labelcolor=SUBTEXT, ncols=len(enabled),
                           handlelength=1.1, columnspacing=1.3,
                           borderpad=0.0, handletextpad=0.5)
        legend.set_zorder(7)

    def _sort_caption(self):
        return ("one bar per electrode, oldest to newest"
                if self._by_date and self._chem is not None
                else "one bar per electrode, sorted by total overrun "
                     "(low to high)")

    def _draw_rug(self, ax, x, verdict, dates=None):
        """One tick per electrode in the same order as the stack above it: the
        quality result laid directly under the overrun that produced it."""
        ax.set_facecolor(AXES_BG)
        colors = np.where(verdict.to_numpy() == 1, COLOR_GOOD, COLOR_CRACKED)
        ax.bar(x, np.ones(len(x)), width=1.0, color=colors, edgecolor="none",
               zorder=3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        # The axis is position, not time, so the ticks are evenly spaced along
        # the bars and each one is labelled with the date of the electrode
        # standing there. Repeats are dropped rather than printed twice.
        if dates is not None and dates.notna().any():
            spots = np.unique(np.linspace(0, len(x) - 1, 9).astype(int))
            seen, ticks, labels = None, [], []
            for spot in spots:
                when = dates.iloc[int(spot)]
                if pd.isna(when):
                    continue
                text = f"{when:%d %b}"
                if text == seen:
                    continue
                seen = text
                ticks.append(int(spot))
                labels.append(text)
            ax.set_xticks(ticks, labels)
            ax.tick_params(colors=SUBTEXT, length=0, labelsize=9,
                           pad=2)
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_horizontalalignment("right")
        else:
            ax.set_xticks([])
        ax.set_ylabel("SEM", color=LABEL, fontsize=10, rotation=0,
                      ha="right", va="center", labelpad=12)
        handles = [Line2D([], [], marker="s", linestyle="", markersize=7,
                          color=COLOR_GOOD, label="not cracked"),
                   Line2D([], [], marker="s", linestyle="", markersize=7,
                          color=COLOR_CRACKED, label="cracked")]
        if dates is None:
            ax.set_xlabel(self._sort_caption(), color=SUBTEXT, fontsize=10,
                          labelpad=6)
            ax.legend(handles=handles, loc="upper right",
                      bbox_to_anchor=(1.0, -0.15), frameon=False,
                      fontsize=9.5, labelcolor=SUBTEXT, ncols=2,
                      handlelength=1.0, borderpad=0.0, handletextpad=0.5)
        else:
            # The dates occupy the space under the strip, so the legend goes
            # up to the title line and the caption is dropped: the tick labels
            # already say the bars run oldest to newest
            self._fig.legend(handles=handles, loc="upper right",
                             bbox_to_anchor=(0.985, 0.995), frameon=False,
                             fontsize=9.5, labelcolor=SUBTEXT, ncols=2,
                             handlelength=1.0, handletextpad=0.5)

    @staticmethod
    def _bins(values):
        """Roughly equal-count bins that never cut through a run of tied
        totals.

        Plain equal-count splitting is wrong here: switch off enough steps and
        a hundred-odd electrodes sit at exactly 0 s, so a boundary drawn mid-tie
        puts the same total on both sides and manufactures a rate difference
        out of nothing but the sort order."""
        target = max(1, len(values) // CUMULATIVE_BINS)
        chunks, start = [], 0
        while start < len(values):
            end = min(start + target, len(values))
            while end < len(values) and values[end] == values[end - 1]:
                end += 1
            chunks.append(np.arange(start, end))
            start = end
        if len(chunks) > 1 and len(chunks[-1]) < target / 2:
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()
        return chunks

    def _draw_rate(self, ax, total, verdict):
        """Cracked share per bin of total overrun: the panel the threshold
        claim lives or dies on. Equal counts, not equal widths, because the
        long tail would otherwise leave most bins empty; every bar carries its
        n so a bin of three is never mistaken for evidence."""
        ax.set_facecolor(AXES_BG)
        chunks = self._bins(total.to_numpy(dtype=float))
        values, labels, counts = [], [], []
        for chunk in chunks:
            part = verdict.iloc[chunk]
            values.append(100 * float((part == 0).sum()) / len(part))
            counts.append(len(part))
            low, high = total.iloc[chunk[0]], total.iloc[chunk[-1]]
            labels.append(self._fmt(low) if low == high
                          else f"{self._fmt(low)}\nto {self._fmt(high)}")

        pos = np.arange(len(chunks))
        bars = ax.bar(pos, values, width=0.72, color=COLOR_CRACKED,
                      edgecolor="none", zorder=3)
        drawn = ax.bar_label(bars, labels=[f"{v:.0f}%\nn={n}"
                                           for v, n in zip(values, counts)],
                             padding=3, fontsize=9, color=SUBTEXT)
        for text in drawn:      # so a label never reads through the mean line
            text.set_bbox(dict(boxstyle="round,pad=0.12", facecolor=AXES_BG,
                               edgecolor="none"))
            text.set_zorder(6)

        overall = 100 * float((verdict == 0).sum()) / len(verdict)
        ax.axhline(overall, color=TARGET_LINE, linewidth=1.3,
                   linestyle=(0, (4, 2)), zorder=5)
        # In the headroom above the bars, where it cannot land on a bar label
        ax.text(0.004, 0.98, f"dashed line : overall {overall:.0f}% cracked",
                transform=ax.transAxes, va="top", fontsize=9, color=SUBTEXT)

        ax.set_xticks(pos, labels)
        ax.set_ylabel("Cracked (%)", color=LABEL, fontsize=11)
        ax.set_xlabel("total overrun, equal-count bins", color=SUBTEXT,
                      fontsize=10, labelpad=4)
        ax.set_ylim(0, max(values + [overall]) * 1.35 + 4)
        ax.set_xlim(-0.6, len(chunks) - 0.4)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)
        ax.tick_params(colors=SUBTEXT, length=0, labelsize=9)

    def _draw_title(self, frame, verdict, total, recipe, enabled):
        """Title only, in figure coordinates. Anchored to the stack axes
        instead, a title pad (points) and text offsets (fractions of a panel
        that resizes with the selection) end up on a collision course."""
        names = ", ".join(self._label_of(key) for key in enabled)
        # Which vat is showing belongs in the title, or a screenshot of a
        # filtered chart is indistinguishable from the whole line
        where = "" if self._vat is None else f"   ·   vat {self._vat}"
        self._fig.text(0.075, 0.975,
                       f"{self._heading} : steps {names}{where}",
                       color=TEXT, fontsize=15, fontweight="bold", va="top")

    def _bind_hover(self, ax, x, frame, verdict, total, enabled):
        labels, colors = [], []
        for electrode in frame.index:
            row = frame.loc[electrode]
            cracked = verdict[electrode] == 0
            lines = [f"{electrode} - {'cracked' if cracked else 'not cracked'}",
                     f"total {self._fmt(total[electrode])}"]
            lines += [f"  step {self._label_of(key):<7}{self._fmt(row[key])}"
                      for key in enabled]
            labels.append("\n".join(lines))
            colors.append(COLOR_CRACKED if cracked else COLOR_GOOD)
        self._hover.add_columns(ax, x, 0.5, labels, colors)


class SignalTimelineWidget(QWidget):
    """One signal, one dot per electrode in production order, joined by a
    trace.

    The matching Temperature/Current/Voltage tabs pool electrodes by vat,
    which averages away the slow drift of the process itself; here the weeks
    stay on the x axis, so a run of cracked electrodes can be read against
    what the vat was doing at the time. Verdict rides on marker shape as
    well as colour."""

    # (panel title, vat logged, plating vat an electrode must have used).
    # Strike and the cleaner take every electrode; each plating panel takes
    # only the electrodes that ran in that vat, so the three are directly
    # comparable. A place with no requirement is also left off the shared
    # plating y range, which is what gives it its own scale.
    #
    # Cleaner sits last to match the panel order on the matching
    # Temperature/Current/Voltage tabs, and so the tab that opens first does
    # not change.
    PLACES = [("Strike (vat 31)", "31", None),
              ("Plate 28", "28", "28"),
              ("Plate 29", "29", "29"),
              ("Plate 30", "30", "30"),
              ("Cleaner (vat 17)", "17", None)]
    TRACE = "#C7CDD4"

    def __init__(self, data, signals, signal="Temperature", unit="°C"):
        super().__init__()
        self._signal = signal
        self._unit = unit
        self._place = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        # One vat on screen at a time: four panels at once left each too small
        # to follow a trace through. Violet underline and smaller type so the
        # bar reads as a level below the window's own teal tabs.
        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for title, _vat, _requires in self.PLACES:
            bar.addTab(title)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_place)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row)

        self._stat = "median"
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Statistic:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = QComboBox()
        combo.addItems(["Median", "Mean"])
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_stat)
        controls.addWidget(caption)
        controls.addWidget(combo)
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = fig = Figure(figsize=(13.0, 5.9), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self._hover = HoverReadout(canvas)

        self._frame = self._build(data, signals)
        if self._frame is None or not len(self._frame):
            self._empty("No data available")
            return
        self._rebuild()

    def _build(self, data, signals):
        """One row per electrode, one column per vat and statistic."""
        if signals is None or not len(signals):
            return None
        if "value_median" not in signals.columns:
            print("⚠ signals.csv has no value_median — re-run fetch_sem_data.py")
            return None

        signals = signals.copy()
        signals["Marking"] = signals["Marking"].astype(str).str.strip()
        signals["Vat"] = signals["Vat"].astype(str)
        signals = signals[signals["signal"] == self._signal]
        signals = signals.drop_duplicates(["Marking", "Vat"])
        if not len(signals):
            return None
        lookup = signals.set_index(["Marking", "Vat"])

        frame = data.dropna(subset=["production_date", "verdict"]).copy()
        frame["plating_vat"] = frame["vats"].astype(str).str.split("|").str[0]
        for _title, vat, requires in self.PLACES:
            belongs = (frame["plating_vat"] == requires if requires
                       else pd.Series(True, index=frame.index))
            for stat in ("mean", "median"):
                column = lookup[f"value_{stat}"]
                frame[f"vat{vat}_{stat}"] = [
                    column.get((e, vat)) if ok else None
                    for e, ok in zip(frame["electrode"], belongs)]
        return frame.sort_values("production_date")

    def _set_stat(self, text):
        self._stat = text.lower()
        self._rebuild()

    def _set_place(self, index):
        self._place = int(index)
        self._rebuild()

    def _plating_limits(self):
        """One y range covering all three plating vats.

        With the panels side by side a shared axis was automatic; one at a
        time it has to be forced, or switching sub-tabs silently rescales and
        two vats that differ by a degree look identical."""
        columns = [f"vat{vat}_{self._stat}"
                   for _t, vat, requires in self.PLACES if requires]
        values = pd.concat([self._frame[c] for c in columns
                            if c in self._frame.columns]).dropna()
        if not len(values):
            return None
        low, high = float(values.min()), float(values.max())
        pad = (0.12 * (high - low) if high > low
               else max(0.5, abs(high) * 0.05))
        return low - pad, high + pad

    def _empty(self, message):
        ax = self._fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()

        where, vat, requires = self.PLACES[self._place]
        ax = fig.add_subplot()
        self._draw(ax, f"vat{vat}_{self._stat}", f"{self._signal} : {where}")

        # Strike runs on its own scale; the plating vats only compare on a
        # shared one, so pin them even though only one is visible
        if requires:
            limits = self._plating_limits()
            if limits:
                ax.set_ylim(*limits)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,
                                                          maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

        fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=7,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="D", linestyle="none", markersize=7,
                   color=COLOR_CRACKED, label="cracked")],
            loc="upper right", ncol=2, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.99, 0.995))
        fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.15)
        self._canvas.draw_idle()

    def _draw(self, ax, column, title):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        unit = self._unit

        present = self._frame.dropna(subset=[column])
        cracked = int((present["verdict"] == 0).sum())
        title = f"{title}\n{len(present)} electrodes, {cracked} cracked"
        ax.plot(present["production_date"], present[column],
                color=self.TRACE, linewidth=1.1, zorder=2)
        for verdict, color, marker in ((1, COLOR_GOOD, "o"),
                                       (0, COLOR_CRACKED, "D")):
            part = present[present["verdict"] == verdict]
            line, = ax.plot(part["production_date"], part[column],
                            marker=marker, linestyle="none", markersize=5,
                            color=color, markeredgecolor=AXES_BG,
                            markeredgewidth=0.7, zorder=3)
            self._hover.add(line, [
                f"{e}   {d:%d %b}   {v:.2f} {unit}"
                for e, d, v in zip(part["electrode"],
                                   part["production_date"], part[column])],
                color)

        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                     loc="left", pad=10)
        ax.set_ylabel(unit, color=LABEL, fontsize=10.5, labelpad=6)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        ax.margins(y=0.12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)


# Electrode timestamps come out of dbo.MetaData in UTC, while the lab writes
# the cleaner bath log in Oslo wall-clock. Anything that puts the two on one
# x axis has to convert first, or an electrode made at 23:30 in July lands
# two hours early and, at the edges, on the wrong day.
PLANT_TZ = "Europe/Oslo"

# Amber: the one accent not already spoken for by a bath quantity or a
# chemical, so the Slotoclean trace cannot be confused with any of them.
SLOTOCLEAN_ACCENT = "#D97706"
SLOTOCLEAN_UNIT = "g/L"


def local_production_time(data):
    """Electrode production time as Oslo wall-clock, to match the lab logs."""
    stamp = pd.to_datetime(data.get("production_datetime_utc"),
                           errors="coerce", utc=True)
    return stamp.dt.tz_convert(PLANT_TZ).dt.tz_localize(None)


class CleanerTabBase(QWidget):
    """Shared parts of the cleaner (vat 17) tabs.

    Both tabs read the same two things - every electrode's logged value at
    the cleaner, and the Slotoclean EL DCG the tank was carrying - and offer
    the same three controls: which signal, mean or median, and how far back.
    What differs is only what they draw with it, so subclasses supply
    _rebuild() and nothing else.

    The concentration is a lab titration run every few days, never a reading
    at the minute an electrode went through. Everything here that pairs the
    two does it with a backward as-of join: what the bath was last known to
    be when that electrode ran.

    Subclasses override the pixel bands below when their axes need different
    room - a scatter pays for an x label the timeline does not have."""

    SIGNALS = [("Voltage", "V"), ("Current", "A"), ("Temperature", "°C")]
    VAT = "17"
    TRACE = "#C7CDD4"
    # (label, days of history); None spans everything
    WINDOWS = [("All", None), ("Last 30 days", 30), ("Last 14 days", 14),
               ("Last 7 days", 7)]

    def __init__(self, data, signals, cleaner):
        super().__init__()
        self._signal, self._unit = self.SIGNALS[0]
        self._stat = "median"
        self._window = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        # Violet underline and smaller type, so this bar reads as a level
        # below the window's own teal tabs - the same rule as the
        # "<signal> over time" tabs, where it carries the vat instead.
        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for signal, _unit in self.SIGNALS:
            bar.addTab(signal)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_signal)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, 0)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        controls.addWidget(self._caption("Statistic:"))
        controls.addWidget(self._combo(["Median", "Mean"], self._set_stat))
        controls.addSpacing(18)
        controls.addWidget(self._caption("Window:"))
        controls.addWidget(self._combo([label for label, _d in self.WINDOWS],
                                       self._set_window))
        controls.addStretch()
        layout.addLayout(controls)

        self._fig = fig = Figure(figsize=(13.0, 7.4), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(240)
        # stretch 1 against the rows' 0: the figure is the only thing here
        # that should grow when the window does
        layout.addWidget(canvas, 1)
        # Figure margins are fractions, so a fixed fraction turns into more
        # and more blank as the window grows - which is exactly how a title
        # ended up floating a hundred pixels above its own axes. Recompute
        # them from pixels on every resize instead.
        canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())
        self._hover = HoverReadout(canvas)

        self._frame = self._build(data, signals)
        self._bath = self._build_bath(cleaner)
        if self._frame is None or not len(self._frame):
            self._empty("No cleaner data available")
            self._canvas.draw_idle()
            return
        self._rebuild()

    def _rebuild(self):
        raise NotImplementedError

    # ------------------------------------------------------------- controls
    @staticmethod
    def _caption(text):
        label = QLabel(text)
        label.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        return label

    @staticmethod
    def _combo(items, on_change):
        combo = QComboBox()
        combo.addItems(items)
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(on_change)
        return combo

    def _set_signal(self, index):
        self._signal, self._unit = self.SIGNALS[int(index)]
        self._rebuild()

    def _set_stat(self, text):
        self._stat = text.lower()
        self._rebuild()

    def _set_window(self, text):
        self._window = dict(self.WINDOWS).get(text)
        self._rebuild()

    # ------------------------------------------------------------------ data
    def _build(self, data, signals):
        """One row per electrode: when it was made, and its cleaner value
        for each signal and statistic."""
        if signals is None or not len(signals):
            return None
        if "value_median" not in signals.columns:
            print("⚠ signals.csv has no value_median — re-run fetch_sem_data.py")
            return None
        if "production_datetime_utc" not in data.columns:
            print("⚠ sem_measurements_filtered.csv has no "
                  "production_datetime_utc — re-run data_preprocessing.py; "
                  "the cleaner timeline needs the time of day")
            return None

        signals = signals.copy()
        signals["Marking"] = signals["Marking"].astype(str).str.strip()
        signals["Vat"] = signals["Vat"].astype(str)
        signals = signals[signals["Vat"] == self.VAT]
        if not len(signals):
            return None

        frame = data.dropna(subset=["verdict"]).copy()
        frame["produced_at"] = local_production_time(frame)
        frame = frame.dropna(subset=["produced_at"])
        for signal, _unit in self.SIGNALS:
            part = (signals[signals["signal"] == signal]
                    .drop_duplicates("Marking").set_index("Marking"))
            for stat in ("mean", "median"):
                frame[f"{signal}_{stat}"] = frame["electrode"].map(
                    part[f"value_{stat}"])
        return frame.sort_values("produced_at")

    @staticmethod
    def _build_bath(cleaner):
        """The titrated rounds only: the Schloetter titration is skipped on
        some walk-rounds, and a round without it is not a zero."""
        if cleaner is None or "slotoclean" not in cleaner.columns:
            return None
        bath = cleaner.dropna(subset=["measured_at", "slotoclean"])
        return bath.sort_values("measured_at") if len(bath) else None

    def _xlim(self):
        """The electrode span, padded - the bath log starts a year earlier,
        and letting it set the axis would squeeze every electrode into the
        right-hand sliver."""
        when = self._frame["produced_at"]
        high = when.max()
        low = (when.min() if self._window is None
               else max(when.min(), high - pd.Timedelta(days=self._window)))
        pad = max((high - low) * 0.02, pd.Timedelta(hours=6))
        return low - pad, high + pad

    def _shown(self, column):
        """The electrodes inside the selected window that have a value."""
        low, high = self._xlim()
        present = self._frame.dropna(subset=[column])
        return present[(present["produced_at"] >= low)
                       & (present["produced_at"] <= high)]

    def _with_bath(self, shown, interpolate=False):
        """Each electrode carrying the Slotoclean of its bath, by one of two
        readings of the same log.

        interpolate=False is a backward as-of join - what the bath was last
        MEASURED to be - the rule gui_all_data.py uses. One titration serves
        every electrode made before the next one, so the column takes only as
        many distinct values as there were rounds, and a scatter of it comes
        out in vertical stripes rather than a cloud. Nothing is jittered to
        hide that; the stripes are the data.

        interpolate=True reads the straight line the timeline's bath panel
        draws between consecutive rounds, which is what the eye does when it
        lines the two panels up. It spreads the stripes into a cloud and lets
        an electrode sit between two measurements rather than on the older
        one. The cost is that the value is no longer measured: the bath does
        not really drift linearly - it steps when it is dosed and sags as it
        is dragged out - so a point far from a round is this chart's guess,
        not the lab's. Both tabs exist so the two can be compared.

        Outside the rounds there is no line to read, so an electrode made
        before the first titration or after the last is dropped rather than
        held flat at the nearest one."""
        if self._bath is None or shown is None or not len(shown):
            return None
        rows = (self._bath[["measured_at", "slotoclean"]]
                .drop_duplicates("measured_at", keep="last")
                .sort_values("measured_at"))
        if not len(rows):
            return None
        if interpolate:
            out = shown.copy()
            out["slotoclean"] = np.interp(
                out["produced_at"].to_numpy("datetime64[ns]").astype("int64"),
                rows["measured_at"].to_numpy("datetime64[ns]").astype("int64"),
                rows["slotoclean"].to_numpy(dtype=float),
                left=np.nan, right=np.nan)
        else:
            out = pd.merge_asof(shown.sort_values("produced_at"), rows,
                                left_on="produced_at",
                                right_on="measured_at", direction="backward")
        out = out.dropna(subset=["slotoclean"])
        return out if len(out) else None

    def _bath_window(self, low, high):
        """In-window readings plus the one either side, so the trace enters
        and leaves at the panel edge instead of starting mid-air."""
        if self._bath is None:
            return None
        when = self._bath["measured_at"]
        keep = (when >= low) & (when <= high)
        before, after = when[when < low], when[when > high]
        if len(before):
            keep |= when == before.max()
        if len(after):
            keep |= when == after.min()
        part = self._bath[keep]
        return part if len(part) else None

    # -------------------------------------------------------------- drawing
    def _empty(self, message, ax=None):
        ax = ax if ax is not None else self._fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ------------------------------------------------------------- chrome
    def _chrome(self, ax, title, ylabel, pad=10):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                     loc="left", pad=pad)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10.5, labelpad=6)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        # Barely more than the data's own range. The daily tabs can afford
        # 12% headroom because they share a y range across vats; here each
        # panel stands alone, and the whole point is the shape of the drift,
        # so the padding is just enough to keep a marker off the spine.
        ax.margins(y=0.04)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _verdict_legend(self):
        self._fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=7,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="D", linestyle="none", markersize=7,
                   color=COLOR_CRACKED, label="cracked")],
            loc="upper right", ncol=2, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.99, 0.995))

    # Fixed pixel budgets, converted to figure fractions against whatever
    # size the canvas currently has: a title band, a row of tick labels and
    # a y label cost the same number of pixels whether the window is 700 px
    # tall or 1400, while a fixed fraction would turn into more and more
    # blank as the window grows.
    TOP_BAND = 34      # title line + the legend sitting beside it
    BOTTOM_BAND = 44   # tick labels + the formatter's offset text
    PANEL_GAP = 34     # a lower panel's title, and nothing else
    LEFT_BAND = 82     # y label + the widest tick label
    RIGHT_BAND = 14    # just enough that the last marker is not clipped

    def _apply_margins(self, draw=True):
        width, height = self._fig.get_size_inches() * self._fig.dpi
        if height <= 0 or width <= 0:
            return
        top = 1 - self.TOP_BAND / height
        bottom = self.BOTTOM_BAND / height
        left = self.LEFT_BAND / width
        right = 1 - self.RIGHT_BAND / width
        if top - bottom < 0.25 or right - left < 0.3:
            # a window too small to lay out in pixels; fall back to fractions
            left, right, top, bottom = 0.07, 0.98, 0.92, 0.08
        # hspace is a fraction of the AVERAGE axes height, not of the figure,
        # so the gap has to be solved for rather than set
        free = height * (top - bottom) - self.PANEL_GAP
        hspace = 2 * self.PANEL_GAP / free if free > 0 else 0.1
        self._fig.subplots_adjust(left=left, right=right, top=top,
                                  bottom=bottom, hspace=hspace)
        if draw:
            self._canvas.draw_idle()


class CleanerTimelineWidget(CleanerTabBase):
    """The cleaner (vat 17) over time, above the Slotoclean the bath carried.

    The "<signal> over time" tabs narrowed to the one station every electrode
    passes through, with two changes.

    The x axis is each electrode's own production timestamp, not its
    production day: four or five electrodes come off a shift and on a daily
    axis they stack on a single tick, which hides both the order they ran in
    and how far apart they were. Hover gives the clock time, and the window
    selector trades span for resolution - inside a week the ticks are hours.

    Underneath, the Slotoclean EL DCG concentration of the same tank on the
    same x axis. Both panels are subplots of one figure sharing that axis, so
    they cannot drift apart: a run of cracked electrodes reads directly
    against a bath that was running rich or lean at the time."""

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()

        low, high = self._xlim()
        # One figure, shared x: nothing to keep in step by hand, and the two
        # panels stay aligned to the pixel whatever either y axis needs.
        # The gap only has to clear the lower panel's title: the top panel
        # prints no tick labels, so the band between them carries nothing
        # else and every pixel of it is height the traces lose. _apply_margins
        # sets it, in pixels.
        grid = fig.add_gridspec(2, 1, height_ratios=[1.4, 1])
        top = fig.add_subplot(grid[0])
        bottom = fig.add_subplot(grid[1], sharex=top)
        self._draw_signal(top, low, high)
        self._draw_bath(bottom, low, high)

        top.set_xlim(low, high)
        # Ticks only under the lower panel; repeating them between the two
        # just pushes the panels apart
        top.tick_params(axis="x", labelbottom=False)
        # Concise rather than the fixed "%d %b" the daily tabs use: the
        # window selector changes the span by two orders of magnitude, and
        # this picks the unit to match - months, then days, then hours -
        # with the year carried once in the corner. Labels stay upright,
        # which is what makes the short forms readable.
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        bottom.xaxis.set_major_locator(locator)
        bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        self._verdict_legend()
        self._apply_margins(draw=False)
        self._canvas.draw_idle()

    def _draw_signal(self, ax, low, high):
        column = f"{self._signal}_{self._stat}"
        unit = self._unit
        shown = self._shown(column)
        if not len(shown):
            self._empty(f"No {self._signal.lower()} logged at the cleaner",
                        ax)
            return

        self._chrome(ax, f"{self._signal} : Cleaner (vat {self.VAT})", unit)
        ax.plot(shown["produced_at"], shown[column], color=self.TRACE,
                linewidth=1.1, zorder=2)
        for verdict, color, marker in ((1, COLOR_GOOD, "o"),
                                       (0, COLOR_CRACKED, "D")):
            part = shown[shown["verdict"] == verdict]
            line, = ax.plot(part["produced_at"], part[column], marker=marker,
                            linestyle="none", markersize=5, color=color,
                            markeredgecolor=AXES_BG, markeredgewidth=0.7,
                            zorder=3)
            self._hover.add(line, [
                f"{e}   {d:%d %b %H:%M}   {v:.2f} {unit}"
                for e, d, v in zip(part["electrode"], part["produced_at"],
                                   part[column])],
                color)

    def _draw_bath(self, ax, low, high):
        part = self._bath_window(low, high)
        if part is None:
            self._empty("No Slotoclean titrations in this window", ax)
            return

        self._chrome(ax, f"Slotoclean EL DCG : cleaner bath (vat {self.VAT})",
                     f"Slotoclean [{SLOTOCLEAN_UNIT}]", pad=5)
        when, value = part["measured_at"], part["slotoclean"]
        # Glow stroke, line, hollow points - the bath-plot idiom, so the
        # trace reads as bath chemistry rather than another electrode series
        ax.plot(when, value, color=SLOTOCLEAN_ACCENT, linewidth=6,
                alpha=0.18, solid_capstyle="round", zorder=2)
        ax.plot(when, value, color=SLOTOCLEAN_ACCENT, linewidth=2.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
        points, = ax.plot(when, value, linestyle="none", marker="o",
                          markersize=4.5, markerfacecolor=AXES_BG,
                          markeredgecolor=SLOTOCLEAN_ACCENT,
                          markeredgewidth=1.2, zorder=4)
        # Gap to the round before: the lab titrates every few days, so this
        # says how stale the concentration an electrode ran against was
        gaps = when.diff().dt.total_seconds() / 86400
        self._hover.add(points, [
            f"{d:%d %b %Y}   {v:.1f} {SLOTOCLEAN_UNIT}"
            + ("" if pd.isna(g) else f"\n{g:.1f} days since the round before")
            for d, v, g in zip(when, value, gaps)],
            SLOTOCLEAN_ACCENT)


class CleanerScatterWidget(CleanerTabBase):
    """The same two quantities as the timeline, plotted against each other.

    One dot per electrode: the Slotoclean EL DCG its bath was last known to
    carry on x, what the cleaner logged for it on y. The timeline shows the
    two drifting side by side and leaves the reader to line them up by eye;
    this collapses the time axis away and asks the question directly - does
    the signal move with the concentration.

    Expect vertical stripes, not a cloud. Every electrode made between two
    titrations inherits the same concentration, so x takes one value per lab
    round and each stripe is the run of electrodes that shared it. The height
    of a stripe is how much the cleaner varied at a fixed bath strength,
    which is worth as much as the left-to-right trend: if the stripes are
    taller than the spread between them, the bath is not what is moving the
    signal. Nothing is jittered - the stripes are the data."""

    # A scatter pays for an x label under its tick labels, which the
    # timeline's shared date axis does not have
    BOTTOM_BAND = 56
    LEFT_BAND = 74
    PANEL_GAP = 0      # one panel, so there is no gap to leave

    # Which reading of the bath log the x axis carries. The interpolated
    # tab is this same chart with the other one; see _with_bath.
    INTERPOLATE = False
    METHOD = "last titration before each electrode"

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()
        self._draw(fig.add_subplot())
        self._verdict_legend()
        self._apply_margins(draw=False)
        self._canvas.draw_idle()

    def _draw(self, ax):
        column = f"{self._signal}_{self._stat}"
        unit = self._unit
        joined = self._with_bath(self._shown(column),
                                 interpolate=self.INTERPOLATE)
        if joined is None:
            self._empty("No electrode in this window falls inside the "
                        "titration log", ax)
            return

        self._chrome(
            ax,
            f"{self._signal} against Slotoclean EL DCG : cleaner "
            f"(vat {self.VAT}) \u2014 {self.METHOD}", unit)
        ax.set_xlabel(f"Slotoclean EL DCG [{SLOTOCLEAN_UNIT}]",
                      color=LABEL, fontsize=10.5, labelpad=8)
        # x gets its own margin: _chrome only sets y, and without this the
        # first and last stripe sit on the spines
        ax.margins(x=0.03)
        for verdict, color, marker in ((1, COLOR_GOOD, "o"),
                                       (0, COLOR_CRACKED, "D")):
            part = joined[joined["verdict"] == verdict]
            # Semi-transparent: a stripe is twenty-odd electrodes deep and
            # opaque markers would hide how many are stacked at one value
            line, = ax.plot(part["slotoclean"], part[column], marker=marker,
                            linestyle="none", markersize=5.5, color=color,
                            markeredgecolor=AXES_BG, markeredgewidth=0.6,
                            alpha=0.8, zorder=3)
            self._hover.add(line, [
                f"{e}   {d:%d %b %H:%M}\n"
                f"{c:.1f} {SLOTOCLEAN_UNIT}   {v:.2f} {unit}"
                for e, d, c, v in zip(part["electrode"], part["produced_at"],
                                      part["slotoclean"], part[column])],
                color)


class CleanerScatterInterpolatedWidget(CleanerScatterWidget):
    """The scatter again, with the concentration read off the line instead
    of off the last round.

    Same electrodes, same y axis; x moves from the last measured value to
    the interpolated one, so the vertical stripes open out into a cloud and
    each electrode sits where the timeline's bath trace was when it ran.

    Worth holding both in mind: this tab's x is smoother and easier to read
    a trend out of, and that is exactly its risk - the smoothness is the
    straight line this chart drew, not something the lab measured."""

    INTERPOLATE = True
    METHOD = "interpolated between titrations"


# --------------------------------------------------------------- pH dosing
# qc.vw_ph_control, reshaped by data_preprocessing.py into one row per vat and
# reading: the mean of the operator's two pH measurements, and whatever was
# dosed in response. Vat 31 is the strike and is left out - it is dosed in kg
# of a different chemical, months apart, and does not belong on these axes.
PH_VATS = ["28", "29", "30"]
PH_ACID = "H2SO4"
PH_ACID_ACCENT = "#0284C7"   # blue; no bath quantity or chemical uses it


class PhDosingWidget(QWidget):
    """Does the acid dose match the pH that called for it, and does dosing
    harder track cracking?

    Two questions, one panel each, off the same dose log.

    Top - every acid dose against the pH it answered. This is the premise
    under test: if small excursions were being over-corrected the cloud
    would sit high at the left.

    Bottom - cracking against the last acid dose before each electrode, on
    both of its axes at once: how many litres went in, and what the pH was
    when they did. A dose is not described by its size alone - 2 L into a
    bath at 4.1 is not the same act as 2 L into one at 4.6 - so collapsing
    it to litres would average the two together. A grid rather than a
    surface: litres take a handful of values, so the data is already a grid,
    and a cell can carry its own count, which is what makes a rate readable
    at this sample size.

    Every electrode is charged with exactly one dose, the most recent into
    its own vat. Nothing is summed over a window and nothing is compared to
    an expected dose; both of those were scaffolding, and both are gone."""

    PLACES = [("All vats", None), ("Vat 28", "28"), ("Vat 29", "29"),
              ("Vat 30", "30")]
    # Litres as logged: 2.0 alone accounts for more than half of all doses
    # and 1.0 for most of the rest, so fixed cuts keep the rows meaningful
    # where quantiles would collapse onto the two modes.
    DOSE_BANDS = [("≤ 1 L", 0.0, 1.0), ("1-2 L", 1.0, 2.0),
                  ("> 2 L", 2.0, float("inf"))]
    # Fixed pH bands, not quantiles. Quantile edges move with whatever is
    # on screen, so a cell meant one thing on "All vats" and another on vat
    # 30, and no two readings of the chart could be compared. These are the
    # same three columns every time, on round numbers.
    #
    # The edges are 4.1 and 4.3 rather than anything narrower because they
    # keep every cell but one populated: nobody doses more than 2 L into a
    # bath sitting below 4.1, so that corner is empty in the process, not
    # just in the sample.
    PH_BANDS = [("< 4.1", float("-inf"), 4.1), ("4.1-4.3", 4.1, 4.3),
                ("> 4.3", 4.3, float("inf"))]
    MIN_PER_CELL = 5        # below this a cell shows its count and no rate
    DOSE_CLOUD = "#C7CDD4"  # the dose layer sits behind the verdict marks

    TOP_BAND = 34
    BOTTOM_BAND = 54
    LEFT_BAND = 78
    RIGHT_BAND = 16
    # The upper panel ends in an x label and the lower one opens with a
    # title, so the gap between them pays for both.
    ROW_GAP = 102

    def __init__(self, data, ph):
        super().__init__()
        self._vat = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for title, _vat in self.PLACES:
            bar.addTab(title)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_vat)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, 0)

        self._fig = fig = Figure(figsize=(13.0, 7.8), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(300)
        layout.addWidget(canvas, 1)
        canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())
        self._hover = HoverReadout(canvas)

        self._events = self._build_events(ph)
        self._electrodes = self._build_electrodes(data)
        if self._events is None:
            self._empty("No H2SO4 doses in the pH log")
            self._canvas.draw_idle()
            return
        self._rebuild()

    # ------------------------------------------------------------- controls
    @staticmethod
    def _caption(text):
        label = QLabel(text)
        label.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        return label

    @staticmethod
    def _combo(items, on_change, index=0):
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentIndex(index)
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(on_change)
        return combo

    def _set_vat(self, index):
        self._vat = self.PLACES[int(index)][1]
        self._rebuild()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _build_events(ph):
        """One row per acid dose: which vat, when, the litres, and the pH
        that called for it.

        A "0 L" row is a check with nothing added, not a dose of nothing, so
        it is not kept as one."""
        if ph is None or not len(ph):
            return None
        log = ph.copy()
        log["vat"] = log["vat"].astype(str)
        log = log[log["vat"].isin(PH_VATS)]
        events = log[(log["chemical"] == PH_ACID) & log["dose"].notna()
                     & (log["dose"] > 0) & log["pH"].notna()].copy()
        if not len(events):
            return None
        return events.sort_values(["vat", "measured_at"])

    @staticmethod
    def _build_electrodes(data):
        """One row per sampled electrode: its plating vat, when it was made
        in Oslo time, and whether it cracked."""
        if "production_datetime_utc" not in data.columns:
            return None
        frame = data.dropna(subset=["verdict"]).copy()
        frame["made_at"] = local_production_time(frame)
        frame["vat"] = frame["vats"].astype(str).str.split("|").str[0]
        frame = frame.dropna(subset=["made_at"])
        frame = frame[frame["vat"].isin(PH_VATS)]
        return frame.sort_values("made_at") if len(frame) else None

    def _selected(self, frame):
        if frame is None or self._vat is None:
            return frame
        return frame[frame["vat"] == self._vat]

    # -------------------------------------------------------------- drawing
    def _empty(self, message, ax=None):
        ax = ax if ax is not None else self._fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                color=EMPTY, fontsize=12, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _chrome(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        ax.set_title(title, color=TEXT, fontsize=11.5, fontweight="bold",
                     loc="left", pad=9)
        ax.set_xlabel(xlabel, color=LABEL, fontsize=10, labelpad=7)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=6)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        ax.margins(x=0.03, y=0.06)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _apply_margins(self, draw=True):
        width, height = self._fig.get_size_inches() * self._fig.dpi
        if width <= 0 or height <= 0:
            return
        top = 1 - self.TOP_BAND / height
        bottom = self.BOTTOM_BAND / height
        left = self.LEFT_BAND / width
        right = 1 - self.RIGHT_BAND / width
        if top - bottom < 0.3 or right - left < 0.3:
            left, right, top, bottom = 0.07, 0.98, 0.92, 0.09
        free_h = height * (top - bottom) - self.ROW_GAP
        self._fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom,
            hspace=2 * self.ROW_GAP / free_h if free_h > 0 else 0.3)
        if draw:
            self._canvas.draw_idle()

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        self._draw_response(fig.add_subplot(grid[0]))
        self._draw_cracking(fig.add_subplot(grid[1]))
        self._apply_margins(draw=False)
        self._canvas.draw_idle()

    def _draw_response(self, ax):
        """Dose against the pH that called for it, with the electrodes that
        followed each dose coloured by verdict.

        Two layers, because a dose and an electrode are different things. In
        grey, every acid dose in the log - the dosing habit itself, and the
        only layer that shows the doses nobody was sampled after. On top,
        one mark per electrode, placed at the dose that was standing in its
        bath when it ran and coloured by how it came out.

        So a grey dot with nothing on it is a dose no sampled electrode
        followed, and a stack of marks on one spot is the run of electrodes
        that shared a single dose - which is most of them, since the litres
        land on round numbers and a dose serves everything made before the
        next one."""
        events = self._selected(self._events)
        if events is None or not len(events):
            self._empty("No acid doses for this vat", ax)
            return
        self._chrome(ax, "How much acid, for how high a pH",
                     "pH when the dose was logged", "H₂SO₄ added [L]")

        # Doses land on round numbers - 2 L some four hundred times - so a
        # plain scatter draws one marker over four hundred. Size the marker
        # by how many events share the point instead.
        pairs = events.groupby(["pH", "dose"]).size()
        sizes = 14 + 9 * np.sqrt(pairs.to_numpy())
        xs = [point[0] for point in pairs.index]
        ys = [point[1] for point in pairs.index]
        dots = ax.scatter(xs, ys, s=sizes, color=self.DOSE_CLOUD, alpha=0.55,
                          edgecolors=AXES_BG, linewidths=0.5, zorder=2)
        self._hover.add(dots, [f"pH {x:.2f}   {y:g} L   ×{n} doses"
                               for (x, y), n in zip(pairs.index, pairs)],
                        self.DOSE_CLOUD)

        electrodes = self._selected(self._last_dose())
        if electrodes is None or not len(electrodes):
            return
        for verdict, color, marker in ((1, COLOR_GOOD, "o"),
                                       (0, COLOR_CRACKED, "D")):
            part = electrodes[electrodes["verdict"] == verdict]
            if not len(part):
                continue
            # Semi-transparent: dozens of electrodes share one (pH, litres)
            # spot, and opaque marks would hide how deep the stack is
            marks = ax.scatter(part["pH"], part["dose"], s=30, color=color,
                               marker=marker, alpha=0.7,
                               edgecolors=AXES_BG, linewidths=0.5, zorder=3)
            self._hover.add(marks, [
                f"{e}   {d:%d %b %H:%M}\npH {x:.2f}   {y:g} L   "
                f"{h:.1f} h later"
                for e, d, x, y, h in zip(part["electrode"], part["made_at"],
                                         part["pH"], part["dose"],
                                         part["hours_since"])],
                color)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=6,
                   color=self.DOSE_CLOUD, label="dose (no sampled electrode)"),
            Line2D([], [], marker="o", linestyle="none", markersize=6,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="D", linestyle="none", markersize=6,
                   color=COLOR_CRACKED, label="cracked")],
            loc="upper left", ncol=3, fontsize=8.5, frameon=True,
            facecolor=AXES_BG, edgecolor=GRID, framealpha=0.92,
            labelcolor=SUBTEXT)

    def _last_dose(self):
        """Each electrode paired with the last acid dose into its own vat.

        A backward as-of join per vat, so an electrode carries the dose that
        was actually standing in its bath when it ran - its litres, the pH
        that called for it, and how long before it went in. The hours are
        kept but not yet drawn: they are the third axis, and a dose eight
        hours old describes the bath far less well than one from an hour
        ago."""
        if self._electrodes is None or self._events is None:
            return None
        pieces = []
        for vat, mine in self._electrodes.groupby("vat"):
            rows = self._events[self._events["vat"] == vat]
            rows = rows[["measured_at", "pH", "dose"]].sort_values(
                "measured_at")
            if not len(rows):
                continue
            joined = pd.merge_asof(mine.sort_values("made_at"), rows,
                                   left_on="made_at", right_on="measured_at",
                                   direction="backward")
            pieces.append(joined)
        if not pieces:
            return None
        frame = pd.concat(pieces, ignore_index=True).dropna(subset=["dose"])
        if not len(frame):
            return None
        frame["hours_since"] = ((frame["made_at"] - frame["measured_at"])
                                .dt.total_seconds() / 3600)
        return frame

    def _draw_cracking(self, ax):
        """Crack rate over last-dose litres against last-dose pH."""
        frame = self._selected(self._last_dose())
        if frame is None or len(frame) < 12:
            self._empty("Not enough electrodes follow a logged dose", ax)
            return

        rows, columns = self.DOSE_BANDS, self.PH_BANDS
        rates = np.full((len(rows), len(columns)), np.nan)
        counts = np.zeros_like(rates)
        cracked = np.zeros_like(rates)
        for r, (_row_label, low, high) in enumerate(rows):
            band = frame[(frame["dose"] > low) & (frame["dose"] <= high)]
            for c, (_col_label, left, right) in enumerate(columns):
                cell = band[(band["pH"] > left) & (band["pH"] <= right)]
                counts[r, c] = len(cell)
                cracked[r, c] = int((cell["verdict"] == 0).sum())
                if len(cell) >= self.MIN_PER_CELL:
                    rates[r, c] = cracked[r, c] / len(cell)

        self._chrome(ax, "Cracking by the last dose : how much, and at what pH",
                     "pH when that dose was logged",
                     "H\u2082SO\u2084 in that dose")
        ax.grid(False)
        # White to the app's cracked violet: one hue, so the eye reads
        # intensity as rate rather than hunting for a key
        ramp = LinearSegmentedColormap.from_list(
            "crackrate", ["#FFFFFF", COLOR_CRACKED])
        top = float(np.nanmax(rates)) if np.isfinite(rates).any() else 1.0
        mesh = ax.pcolormesh(np.arange(len(columns) + 1),
                             np.arange(len(rows) + 1),
                             rates, cmap=ramp, vmin=0.0,
                             vmax=max(top, 0.1), edgecolors=AXES_BG,
                             linewidth=2.5, zorder=2)
        bar = self._fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.035)
        bar.outline.set_edgecolor(SPINE)
        bar.ax.tick_params(colors=SUBTEXT, labelsize=8.5, length=3, width=0.7)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v * 100:.0f}%"))

        for r in range(len(rows)):
            for c in range(len(columns)):
                n = int(counts[r, c])
                if not n:
                    continue
                if n >= self.MIN_PER_CELL:
                    # Dark ink on the pale end, white on the deep end
                    deep = rates[r, c] > 0.55 * max(top, 0.1)
                    text = f"{rates[r, c] * 100:.0f}%\n{int(cracked[r, c])}/{n}"
                    color = "#FFFFFF" if deep else TEXT
                else:
                    text = f"{int(cracked[r, c])}/{n}"
                    color = EMPTY
                ax.text(c + 0.5, r + 0.5, text, ha="center", va="center",
                        fontsize=9, color=color, zorder=3)

        ax.set_xticks(np.arange(len(columns)) + 0.5)
        ax.set_xticklabels([label for label, _lo, _hi in columns])
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([label for label, _lo, _hi in rows])
        ax.set_xlim(0, len(columns))
        ax.set_ylim(0, len(rows))
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        # The baseline every cell is read against
        overall = (frame["verdict"] == 0).mean()
        ax.set_title(
            f"Cracking by the last dose : how much, and at what pH"
            f"          all {len(frame)} electrodes : "
            f"{overall * 100:.0f}% cracked",
            color=TEXT, fontsize=11.5, fontweight="bold", loc="left", pad=9)


class PhSpaceWidget(QWidget):
    """The three numbers describing an electrode's bath, on three axes.

    x is the last pH reading before it was made, y how many hours had passed
    since that reading, and z the H2SO4 logged with it. Colour is the
    verdict. Drag to rotate.

    Why the three belong together: the pH is only as good as it is fresh. A
    reading an hour old describes the bath the electrode ran in; one from
    eight hours ago describes a bath that has since had a shift of
    production through it. So the y axis is how much to trust the x axis,
    and flattening it away - which every 2D cut of this does - throws that
    away.

    Unlike the pH dosing tab this takes the last READING, not the last dose.
    30% of electrodes follow a reading where nothing was added, and those are
    the z = 0 floor: the group that tells you what cracking looks like with
    no acid in the recent picture at all.

    Rotation is the whole point, so there is no hover readout here - a
    tooltip would have to guess which of two overlapping points the cursor
    is nearer in a projection the reader keeps changing."""

    PLACES = [("All vats", None), ("Vat 28", "28"), ("Vat 29", "29"),
              ("Vat 30", "30")]
    # Hours since the reading, as an axis limit. Three quarters of the
    # electrodes sit under four hours and the tail runs to sixty-six, so on
    # a full axis the bulk collapses into a slab at the origin.
    HOUR_LIMITS = [("all", None), ("≤ 12 h", 12.0), ("≤ 6 h", 6.0),
                   ("≤ 3 h", 3.0)]

    def __init__(self, data, ph):
        super().__init__()
        self._vat = None
        self._hours_limit = 12.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for title, _vat in self.PLACES:
            bar.addTab(title)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_vat)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, 0)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Hours axis:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = QComboBox()
        combo.addItems([label for label, _h in self.HOUR_LIMITS])
        combo.setCurrentIndex(1)
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_hours)
        controls.addWidget(caption)
        controls.addWidget(combo)
        controls.addSpacing(18)
        hint = QLabel("drag inside the box to rotate")
        hint.setStyleSheet(f"color: {EMPTY}; font-size: 11px; "
                           f"font-style: italic;")
        controls.addWidget(hint)
        controls.addStretch()
        layout.addLayout(controls, 0)

        self._fig = fig = Figure(figsize=(11.0, 8.0), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(320)
        layout.addWidget(canvas, 1)

        self._frame = self._build(data, ph)
        if self._frame is None or not len(self._frame):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No electrode follows a pH reading",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        self._rebuild()

    # ------------------------------------------------------------- controls
    def _set_vat(self, index):
        self._vat = self.PLACES[int(index)][1]
        self._rebuild()

    def _set_hours(self, text):
        self._hours_limit = dict(self.HOUR_LIMITS).get(text)
        self._rebuild()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _build(data, ph):
        """Each electrode against the last pH reading of its own vat."""
        if ph is None or not len(ph) or "production_datetime_utc" not in data:
            return None
        reads = ph.copy()
        reads["vat"] = reads["vat"].astype(str)
        reads = reads[reads["vat"].isin(PH_VATS) & reads["pH"].notna()]
        if not len(reads):
            return None
        # Acid logged on that same row. A reading with nothing added is a
        # real zero here, not a gap: it is a bath nobody corrected.
        reads["acid_l"] = np.where(
            reads["chemical"].to_numpy() == PH_ACID,
            reads["dose"].fillna(0.0).to_numpy(dtype=float), 0.0)
        reads = reads[["vat", "measured_at", "pH", "acid_l"]].sort_values(
            "measured_at")

        made = data.dropna(subset=["verdict"]).copy()
        made["made_at"] = local_production_time(made)
        made["vat"] = made["vats"].astype(str).str.split("|").str[0]
        made = made.dropna(subset=["made_at"])
        made = made[made["vat"].isin(PH_VATS)]
        if not len(made):
            return None

        pieces = []
        for vat, mine in made.groupby("vat"):
            rows = reads[reads["vat"] == vat].drop(columns="vat")
            if not len(rows):
                continue
            pieces.append(pd.merge_asof(
                mine.sort_values("made_at"), rows, left_on="made_at",
                right_on="measured_at", direction="backward"))
        if not pieces:
            return None
        frame = pd.concat(pieces, ignore_index=True).dropna(subset=["pH"])
        if not len(frame):
            return None
        frame["hours_since"] = ((frame["made_at"] - frame["measured_at"])
                                .dt.total_seconds() / 3600)
        return frame

    def _selected(self):
        frame = self._frame
        if self._vat is not None:
            frame = frame[frame["vat"] == self._vat]
        if self._hours_limit is not None:
            frame = frame[frame["hours_since"] <= self._hours_limit]
        return frame

    # -------------------------------------------------------------- drawing
    def _rebuild(self):
        fig = self._fig
        fig.clear()
        frame = self._selected()
        ax = fig.add_subplot(projection="3d")
        self._chrome(ax)
        if not len(frame):
            ax.text2D(0.5, 0.5, "Nothing inside this window",
                      transform=ax.transAxes, ha="center", color=EMPTY,
                      fontsize=12, style="italic")
            self._canvas.draw_idle()
            return

        for verdict, color, marker in ((1, COLOR_GOOD, "o"),
                                       (0, COLOR_CRACKED, "D")):
            part = frame[frame["verdict"] == verdict]
            if not len(part):
                continue
            ax.scatter(part["pH"], part["hours_since"], part["acid_l"],
                       s=30, c=color, marker=marker, alpha=0.8,
                       edgecolors=AXES_BG, linewidths=0.4, depthshade=True)

        cracked = int((frame["verdict"] == 0).sum())
        ax.set_title(f"Last pH reading, its age, and the acid with it"
                     f"          {len(frame)} electrodes, {cracked} cracked",
                     color=TEXT, fontsize=12, fontweight="bold", loc="left",
                     pad=14)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=7,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="D", linestyle="none", markersize=7,
                   color=COLOR_CRACKED, label="cracked")],
            loc="upper right", ncol=2, frameon=False, fontsize=9,
            labelcolor=SUBTEXT)
        # Room along the bottom and left for the two tilted axis
        # labels, which a 3D axes draws outside its own box
        self._fig.subplots_adjust(left=0.04, right=0.95, top=0.93,
                                  bottom=0.08)
        self._canvas.draw_idle()

    def _chrome(self, ax):
        ax.set_facecolor(FIG_BG)
        # Panes in the figure background rather than matplotlib's default
        # blue-grey, so the box reads as chrome and not as a surface
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(to_rgb(AXES_BG) + (1.0,))
            axis.pane.set_edgecolor(GRID)
            axis.line.set_color(SPINE)
        ax.set_xlabel("last pH reading", color=LABEL, fontsize=10.5,
                      labelpad=10)
        ax.set_ylabel("hours since that reading", color=LABEL, fontsize=10.5,
                      labelpad=10)
        ax.set_zlabel("H₂SO₄ added [L]", color=LABEL, fontsize=10.5,
                      labelpad=8)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = GRID
            axis._axinfo["grid"]["linewidth"] = 0.7
        # A 3D axes draws itself inside a box that leaves a wide blank
        # border on every side; zoom scales the box up within the same axes
        # rectangle, which is the only way to get the cloud to fill the tab.
        ax.set_box_aspect((1.25, 1.25, 0.95), zoom=1.12)
        # A viewpoint that opens the cloud up rather than looking down a wall
        ax.view_init(elev=22, azim=-58)


# ------------------------------------------------------- crack style by pH
# The severity scale lives in sem_analysis/note_dictionary.py: 5 fine,
# 4 very few microcracks, 3 microcracks, 2 small cracks, 1 cracked. load_data
# already maps every note through it into data["quality"].
#
# Ordered categories, so the colours are a ramp and not a palette: deep
# violet at cracked, through a neutral grey in the middle, to the app's teal
# at fine. Reading left to right along a bar is reading down the severity
# scale, which is the whole point of the chart.
# Two hues for two families, so a glance says which cracks belong together:
# violet for the ones you can see (cracked, small cracks), orange for the
# ones you need a microscope for (microcracks, very few), and the app's teal
# on its own for fine. Within a family the darker shade is the worse one, so
# severity still reads as depth of colour.
CRACK_LEVELS = [(1, "cracked", "#6D28D9"),
                (2, "small cracks", "#A78BFA"),
                (3, "microcracks", "#F97316"),
                (4, "very few microcracks", "#FED7AA"),
                (5, "fine", COLOR_GOOD)]
# "fine, the middle three, cracked"
CRACK_GROUPS = [((1,), "cracked", "#6D28D9"),
                ((2, 3, 4), "small cracks / microcracks", "#CBD5E1"),
                ((5,), "fine", COLOR_GOOD)]
# Fixed, round, and chosen to leave every band with 50+ electrodes
CRACK_PH_BANDS = [("< 4.0", float("-inf"), 4.0), ("4.0-4.2", 4.0, 4.2),
                  ("4.2-4.4", 4.2, 4.4), ("> 4.4", 4.4, float("inf"))]
# Production stacks, by marking number. The boundaries are given, not
# derived - a stack is a build, and nothing in the data marks where one
# ends. Stack 20 is open-ended because it is still being built; it holds no
# sampled electrode yet and its box will start filling on its own.
CRACK_STACKS = [("Stack 17", 13900, 14739), ("Stack 18", 14740, 15408),
                ("Stack 19", 15409, 15972),
                ("Stack 20", 15973, float("inf"))]


class CrackStyleWidget(QWidget):
    """How the kind of cracking changes with the bath pH.

    One bar per pH band, each bar the full share of its electrodes split by
    severity. Shares rather than counts, because the bands hold different
    numbers of electrodes and the question is about mix, not volume; the
    count sits at the end of every bar so no share can be read without it.

    The pH is the last measurement before the electrode, carried forward.
    That is a choice, not a shortcut - see the long comment on
    PH_STALENESS_NOTE. It is stale by a median of two hours and nothing in
    this app can do better, so the chart says so in its own subtitle rather
    than implying a precision it does not have."""

    PLACES = [("All vats", None), ("Vat 28", "28"), ("Vat 29", "29"),
              ("Vat 30", "30")]
    GROUPINGS = [("5 levels", "levels"), ("3 groups", "groups"),
                 ("without microcracks", "nomicro")]
    # The two levels the third grouping drops. They are not merged into a
    # neighbour, they leave: the shares are then of the electrodes that
    # remain, and the n beside each bar falls to match.
    CRACK_DROPPED = (3, 4)

    TOP_BAND = 20      # nothing above the axes now
    BOTTOM_BAND = 92   # x label plus the legend beneath it
    LEFT_BAND = 92     # the band labels
    RIGHT_BAND = 70    # the "n = 91" at the end of each bar

    def __init__(self, data, ph):
        super().__init__()
        self._vat = None
        self._grouping = "levels"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for title, _vat in self.PLACES:
            bar.addTab(title)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_vat)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, 0)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 0, 0)
        caption = QLabel("Grouping:")
        caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        combo = QComboBox()
        combo.addItems([label for label, _key in self.GROUPINGS])
        combo.setStyleSheet(
            f"QComboBox {{ color: {TEXT}; font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {SPINE}; border-radius: 4px; "
            f"background: {AXES_BG}; }}")
        combo.currentTextChanged.connect(self._set_grouping)
        controls.addWidget(caption)
        controls.addWidget(combo)

        controls.addSpacing(26)
        stack_caption = QLabel("Stacks:")
        stack_caption.setStyleSheet(f"color: {LABEL}; font-size: 12px;")
        controls.addWidget(stack_caption)
        self._stacks = dict.fromkeys(
            (name for name, _lo, _hi in CRACK_STACKS), True)
        self._stack_boxes = {}
        for name, _low, _high in CRACK_STACKS:
            box = QCheckBox(name)
            box.setChecked(True)
            box.setStyleSheet(
                f"QCheckBox {{ color: {LABEL}; font-size: 12px; }}"
                f"QCheckBox::indicator:checked {{ background: "
                f"{COLOR_CRACKED}; border: 1px solid {COLOR_CRACKED}; }}")
            box.toggled.connect(
                lambda on, n=name: self._set_stack(n, on))
            controls.addWidget(box)
            self._stack_boxes[name] = box
        controls.addStretch()
        layout.addLayout(controls, 0)

        self._fig = fig = Figure(figsize=(12.0, 6.4), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(280)
        layout.addWidget(canvas, 1)
        canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())

        self._frame = self._build(data, ph)
        if self._frame is None or not len(self._frame):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No electrode has both a pH and a crack note",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        self._rebuild()

    # ------------------------------------------------------------- controls
    def _set_vat(self, index):
        self._vat = self.PLACES[int(index)][1]
        self._rebuild()

    def _set_grouping(self, text):
        self._grouping = dict(self.GROUPINGS).get(text, "levels")
        self._rebuild()

    def _set_stack(self, name, on):
        self._stacks[name] = bool(on)
        self._rebuild()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _build(data, ph):
        """Each electrode with a severity level and the last pH reading of
        its own vat before it was made."""
        if ph is None or not len(ph) or "quality" not in data.columns:
            return None
        reads = ph.copy()
        reads["vat"] = reads["vat"].astype(str)
        reads = reads[reads["vat"].isin(PH_VATS) & reads["pH"].notna()]
        reads = reads[["vat", "measured_at", "pH"]].sort_values("measured_at")
        if not len(reads):
            return None

        made = data.dropna(subset=["quality"]).copy()
        made["made_at"] = local_production_time(made)
        made["vat"] = made["vats"].astype(str).str.split("|").str[0]
        made = made.dropna(subset=["made_at"])
        made = made[made["vat"].isin(PH_VATS)]
        if not len(made):
            return None

        pieces = []
        for vat, mine in made.groupby("vat"):
            rows = reads[reads["vat"] == vat].drop(columns="vat")
            if not len(rows):
                continue
            pieces.append(pd.merge_asof(
                mine.sort_values("made_at"), rows, left_on="made_at",
                right_on="measured_at", direction="backward"))
        if not pieces:
            return None
        frame = pd.concat(pieces, ignore_index=True).dropna(subset=["pH"])
        if not len(frame):
            return None
        frame["hours_since"] = ((frame["made_at"] - frame["measured_at"])
                                .dt.total_seconds() / 3600)
        frame["level"] = frame["quality"].astype(int)
        number = marking_number(frame["electrode"])
        frame["stack"] = ""
        for name, low, high in CRACK_STACKS:
            frame.loc[(number >= low) & (number <= high), "stack"] = name
        return frame

    @staticmethod
    def _ink(color):
        """Label colour that stays readable on a given segment.

        Measured rather than listed: the palette has changed twice already
        and a hardcoded list of dark swatches goes stale silently, leaving
        dark text on a dark bar."""
        red, green, blue = to_rgb(color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "#FFFFFF" if luminance < 0.55 else TEXT

    def _selected(self):
        frame = self._frame
        if self._vat is not None:
            frame = frame[frame["vat"] == self._vat]
        wanted = [name for name, _lo, _hi in CRACK_STACKS
                  if self._stacks.get(name, True)]
        if len(wanted) < len(CRACK_STACKS):
            frame = frame[frame["stack"].isin(wanted)]
        return frame

    def _series(self):
        """(label, colour, levels) for whichever grouping is on."""
        if self._grouping == "groups":
            return [(label, color, set(levels))
                    for levels, label, color in CRACK_GROUPS]
        kept = self._kept()
        return [(label, color, {level})
                for level, label, color in CRACK_LEVELS if level in kept]

    def _kept(self):
        """Which severity levels this grouping counts at all.

        Only the third grouping drops any, and dropping is not the same as
        merging: an electrode with microcracks is removed from the
        denominator too, so the bars answer "of the electrodes that were
        either cracked or clean, how many were which" rather than burying
        the microcracks inside another class."""
        if self._grouping == "nomicro":
            return {level for level, _label, _color in CRACK_LEVELS
                    if level not in self.CRACK_DROPPED}
        return {level for level, _label, _color in CRACK_LEVELS}

    # -------------------------------------------------------------- drawing
    def _apply_margins(self, draw=True):
        width, height = self._fig.get_size_inches() * self._fig.dpi
        if width <= 0 or height <= 0:
            return
        top = 1 - self.TOP_BAND / height
        bottom = self.BOTTOM_BAND / height
        left = self.LEFT_BAND / width
        right = 1 - self.RIGHT_BAND / width
        if top - bottom < 0.3 or right - left < 0.3:
            left, right, top, bottom = 0.1, 0.95, 0.88, 0.14
        self._fig.subplots_adjust(left=left, right=right, top=top,
                                  bottom=bottom)
        if draw:
            self._canvas.draw_idle()

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        frame = self._selected()
        ax = fig.add_subplot()
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        if not len(frame):
            ax.text(0.5, 0.5, "No electrodes for this vat and stack",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvas.draw_idle()
            return

        series = self._series()
        frame = frame[frame["level"].isin(self._kept())]
        rows = list(enumerate(CRACK_PH_BANDS))
        totals = []
        for index, (_label, low, high) in rows:
            band = frame[(frame["pH"] > low) & (frame["pH"] <= high)]
            totals.append(len(band))
            if not len(band):
                continue
            left = 0.0
            for label, color, levels in series:
                share = band["level"].isin(levels).mean()
                if share <= 0:
                    continue
                ax.barh(index, share, left=left, height=0.62, color=color,
                        edgecolor=AXES_BG, linewidth=1.6, zorder=3)
                # Only label a segment wide enough to hold the text; the
                # legend carries the rest
                if share >= 0.09:
                    ink = self._ink(color)
                    ax.text(left + share / 2, index, f"{share * 100:.0f}%",
                            ha="center", va="center", fontsize=9.5,
                            color=ink, zorder=4)
                left += share

        for index, total in enumerate(totals):
            ax.text(1.012, index, f"n = {total}", transform=
                    ax.get_yaxis_transform(), ha="left", va="center",
                    color=SUBTEXT, fontsize=9.5)

        ax.set_yticks([index for index, _band in rows])
        ax.set_yticklabels([band[0] for _index, band in rows])
        ax.invert_yaxis()
        ax.set_ylabel("bath pH", color=LABEL, fontsize=10.5, labelpad=10)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v * 100:.0f}%"))
        ax.set_xlabel("share of the electrodes in that band", color=LABEL,
                      fontsize=10.5, labelpad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.yaxis.grid(False)
        ax.tick_params(colors=SUBTEXT, labelsize=9.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE)

        fig.legend(handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=10,
                   color=color, label=label)
            for label, color, _levels in series],
            loc="lower center", ncol=len(series), frameon=False, fontsize=9.5,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.5, 0.005))
        self._apply_margins(draw=False)
        self._canvas.draw_idle()


# ------------------------------------------------------------- idle drift
# What a plating bath does when it is left alone: no electrode through it and
# nothing dosed into it, between two pH readings. 115 such stretches in the
# log, and they are the only clean look at the bath's own chemistry - every
# other interval mixes it with acid or with production.
DRIFT_ACCENT = "#0E7490"      # deep cyan; not used by any other quantity
# Starting-pH bands for the second panel. A sequential ramp, because the
# bands are ordered and the order is the finding.
DRIFT_START_BANDS = [("starts < 4.0", float("-inf"), 4.0, "#67E8F9"),
                     ("4.0-4.2", 4.0, 4.2, "#22D3EE"),
                     ("4.2-4.4", 4.2, 4.4, "#0E7490"),
                     ("starts > 4.4", 4.4, float("inf"), "#164E63")]
# Rate is change divided by elapsed time, so a twenty-minute gap turns a
# rounding difference into a huge slope. The rate panel starts at an hour.
DRIFT_MIN_HOURS = 1.0


class IdleDriftWidget(QWidget):
    """What the bath does on its own, with nothing done to it.

    An idle stretch is two consecutive pH readings of one vat with no
    electrode plated and no chemical added in between. There are 115, and
    the pH rose in 77% of them.

    The rise is not a constant drift. How fast the pH climbs depends on
    where it starts: a bath left at 3.8 walks up by about 0.04 an hour, one
    left at 4.4 barely moves. That is a bath relaxing toward a resting
    point, not a bath drifting, and the difference matters - a constant
    drift would keep climbing forever, while this one stops.

    Left: the climb rate against the pH it started from. The line is the
    fitted relaxation and where it crosses zero is the resting pH - the
    value the bath walks back to if nobody touches it.

    Right: the same stretches as the raw change against how long the bath
    sat, coloured by where each one started, so the fanning out is visible
    rather than inferred."""

    PLACES = [("All vats", None), ("Vat 28", "28"), ("Vat 29", "29"),
              ("Vat 30", "30")]

    # Two lines of per-band rates, then the left panel's own title
    TOP_BAND = 108
    BOTTOM_BAND = 56
    LEFT_BAND = 76
    RIGHT_BAND = 18
    COL_GAP = 96

    def __init__(self, data, ph, throughput=None):
        super().__init__()
        self._vat = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        bar = QTabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        for title, _vat in self.PLACES:
            bar.addTab(title)
        bar.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; padding: 6px 18px; "
            f"font-size: 12px; color: {SUBTEXT}; "
            f"border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {TEXT}; font-weight: bold; "
            f"border-bottom: 2px solid {COLOR_CRACKED}; }}"
            f"QTabBar::tab:hover {{ color: {TEXT}; }}")
        bar.currentChanged.connect(self._set_vat)
        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.addWidget(bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, 0)

        self._fig = fig = Figure(figsize=(13.0, 6.2), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(280)
        layout.addWidget(canvas, 1)
        canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())
        self._hover = HoverReadout(canvas)

        self._frame = self._build(ph, throughput)
        if self._frame is None or not len(self._frame):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No idle stretch between two pH readings",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        self._rebuild()

    def _set_vat(self, index):
        self._vat = self.PLACES[int(index)][1]
        self._rebuild()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _build(ph, throughput):
        """Consecutive readings of one vat with nothing done in between."""
        if ph is None or not len(ph) or throughput is None:
            return None
        log = ph.copy()
        log["vat"] = log["vat"].astype(str)
        log = log[log["vat"].isin(PH_VATS) & log["pH"].notna()]
        log = log.sort_values(["vat", "measured_at"])
        if not len(log):
            return None
        dosed = log["dose"].fillna(0.0).to_numpy(dtype=float)
        log = log.assign(any_dose=dosed > 0)

        pieces = []
        for vat, part in log.groupby("vat"):
            part = part.reset_index(drop=True)
            if len(part) < 2:
                continue
            when = part["measured_at"].to_numpy("datetime64[ns]")
            made = np.sort(throughput.loc[throughput["vat"] == vat, "made_at"]
                           .to_numpy("datetime64[ns]"))
            rows = pd.DataFrame({
                "vat": vat,
                "t0": part["measured_at"].to_numpy()[:-1],
                "t1": part["measured_at"].to_numpy()[1:],
                "ph0": part["pH"].to_numpy(dtype=float)[:-1],
                "ph1": part["pH"].to_numpy(dtype=float)[1:],
                # The dose on a row goes in just after that reading, so it
                # belongs to the interval that STARTS there
                "dosed": part["any_dose"].to_numpy()[:-1],
            })
            start = np.searchsorted(made, when[:-1], side="right")
            end = np.searchsorted(made, when[1:], side="right")
            rows["plated"] = (end - start).astype(int)
            pieces.append(rows)
        if not pieces:
            return None

        frame = pd.concat(pieces, ignore_index=True)
        frame["hours"] = (frame["t1"] - frame["t0"]).dt.total_seconds() / 3600
        frame["moved"] = frame["ph1"] - frame["ph0"]
        # Idle means BOTH: nothing plated and nothing dosed
        frame = frame[(frame["hours"] > 0) & (frame["plated"] == 0)
                      & (~frame["dosed"])]
        return frame if len(frame) else None

    def _selected(self):
        frame = self._frame
        return frame if self._vat is None else frame[frame["vat"] == self._vat]

    @staticmethod
    def _relaxation(frame):
        """Fit moved = (resting - start) * (1 - exp(-k * hours)).

        Linear in the resting pH once k is fixed, so a scan over k and a
        one-line least squares inside it finds both. Beats a straight-line
        drift by a wide margin because a straight line cannot stop climbing,
        and the bath plainly does."""
        if frame is None or len(frame) < 12:
            return None
        hours = frame["hours"].to_numpy(dtype=float)
        start = frame["ph0"].to_numpy(dtype=float)
        moved = frame["moved"].to_numpy(dtype=float)
        best = None
        for k in np.linspace(0.002, 0.5, 600):
            reach = 1.0 - np.exp(-k * hours)
            denominator = float((reach * reach).sum())
            if denominator <= 0:
                continue
            resting = float(((moved + start * reach) * reach).sum()
                            / denominator)
            residual = moved - (resting - start) * reach
            score = float((residual ** 2).sum())
            if best is None or score < best[0]:
                best = (score, k, resting)
        if best is None:
            return None
        score, k, resting = best
        total = float(((moved - moved.mean()) ** 2).sum())
        return {"k": k, "resting": resting,
                "r2": 1 - score / total if total else float("nan"),
                "half_life": float(np.log(2) / k)}

    # -------------------------------------------------------------- drawing
    def _chrome(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        ax.set_title(title, color=TEXT, fontsize=11.5, fontweight="bold",
                     loc="left", pad=9)
        ax.set_xlabel(xlabel, color=LABEL, fontsize=10, labelpad=7)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=6)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        ax.margins(x=0.04, y=0.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _apply_margins(self, draw=True):
        width, height = self._fig.get_size_inches() * self._fig.dpi
        if width <= 0 or height <= 0:
            return
        top = 1 - self.TOP_BAND / height
        bottom = self.BOTTOM_BAND / height
        left = self.LEFT_BAND / width
        right = 1 - self.RIGHT_BAND / width
        if top - bottom < 0.3 or right - left < 0.3:
            left, right, top, bottom = 0.07, 0.98, 0.88, 0.12
        free = width * (right - left) - self.COL_GAP
        self._fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom,
            wspace=2 * self.COL_GAP / free if free > 0 else 0.2)
        if draw:
            self._canvas.draw_idle()

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()
        frame = self._selected()
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        # The fit still drives both panels' curves and the resting-pH rule;
        # only its printed summary is gone.
        model = self._relaxation(frame)
        self._draw_rates_row(frame)
        self._draw_rate(fig.add_subplot(grid[0]), frame, model)
        self._draw_climb(fig.add_subplot(grid[1]), frame, model)
        self._apply_margins(draw=False)
        self._canvas.draw_idle()

    def _draw_rates_row(self, frame):
        """One number per starting-pH band: the climb rate, in pH per hour.

        The median of each stretch's own rate, not the slope of a line
        through all of them - one 200-hour stretch would otherwise set the
        number for a whole band. Stretches under an hour are left out: they
        divide a measurement wobble by a fraction and produce rates of a
        quarter pH an hour that mean nothing.

        A single figure number would hide the finding, because the rate is
        not one number - it depends on where the bath starts, and these four
        are how that shows."""
        usable = frame[frame["hours"] >= DRIFT_MIN_HOURS]
        tiles = len(DRIFT_START_BANDS)
        for position, (label, low, high, color) in enumerate(
                DRIFT_START_BANDS):
            part = usable[(usable["ph0"] > low) & (usable["ph0"] <= high)]
            x = 0.0075 + position * (0.985 / tiles)
            if len(part) < 3:
                headline, note = "–", f"{label}   too few"
            else:
                rate = float((part["moved"] / part["hours"]).median())
                headline = f"{rate:+.3f} pH / h"
                note = f"{label}   ·   {len(part)} stretches"
            self._fig.text(x, 0.992, headline, color=color, fontsize=14,
                           fontweight="bold", va="top", ha="left")
            self._fig.text(x, 0.947, note, color=SUBTEXT, fontsize=9.5,
                           va="top", ha="left")

    def _draw_rate(self, ax, frame, model):
        """Climb rate against the pH the stretch started from."""
        part = frame[frame["hours"] >= DRIFT_MIN_HOURS].copy()
        if not len(part):
            ax.set_axis_off()
            return
        part["rate"] = part["moved"] / part["hours"]
        self._chrome(ax, "How fast it climbs, by where it starts",
                     "pH at the start of the idle stretch",
                     "pH change per hour")
        ax.axhline(0, color=SPINE, linewidth=1.1, zorder=1)

        # Area by hours: a two-hour stretch divides a measurement wobble by
        # two and a twenty-hour one by twenty, so the long ones are the
        # trustworthy points and should look it
        sizes = 12 + 4.5 * np.sqrt(part["hours"].to_numpy(dtype=float))
        dots = ax.scatter(part["ph0"], part["rate"], s=sizes,
                          color=DRIFT_ACCENT, alpha=0.5,
                          edgecolors=AXES_BG, linewidths=0.5, zorder=3)
        self._hover.add(dots, [
            f"vat {v}   {d:%d %b %H:%M}\npH {a:.2f} → {b:.2f}"
            f"   over {h:.1f} h   ({r:+.3f}/h)"
            for v, d, a, b, h, r in zip(part["vat"], part["t0"], part["ph0"],
                                        part["ph1"], part["hours"],
                                        part["rate"])],
            DRIFT_ACCENT)

        if model:
            xs = np.linspace(part["ph0"].min() - 0.05,
                             max(part["ph0"].max(), model["resting"]) + 0.05,
                             100)
            ax.plot(xs, model["k"] * (model["resting"] - xs),
                    color=TARGET_LINE, linewidth=2.0, zorder=4,
                    label="trendline")
            # Where the line crosses zero is the pH the bath settles at
            if xs[0] <= model["resting"] <= xs[-1]:
                ax.axvline(model["resting"], color=TARGET_LINE, linewidth=1.2,
                           linestyle=(0, (5, 3)), alpha=0.7, zorder=2)
                ax.annotate(f"settles near pH {model['resting']:.2f}",
                            xy=(model["resting"], 0), xytext=(-8, 16),
                            textcoords="offset points", ha="right",
                            color=SUBTEXT, fontsize=9)
            ax.legend(loc="upper right", fontsize=8.5, frameon=True,
                      facecolor=AXES_BG, edgecolor=GRID, framealpha=0.92,
                      labelcolor=SUBTEXT)

    def _draw_climb(self, ax, frame, model):
        """Raw change against how long the bath sat."""
        # No title: the figure's own headline already says what this tab
        # is, and the axis labels say what this panel is
        self._chrome(ax, "", "hours the bath sat idle", "pH change")
        ax.axhline(0, color=SPINE, linewidth=1.1, zorder=1)
        for label, low, high, color in DRIFT_START_BANDS:
            part = frame[(frame["ph0"] > low) & (frame["ph0"] <= high)]
            if not len(part):
                continue
            dots = ax.scatter(part["hours"], part["moved"], s=28,
                              color=color, alpha=0.75, edgecolors=AXES_BG,
                              linewidths=0.5, zorder=3, label=label)
            self._hover.add(dots, [
                f"vat {v}   {d:%d %b %H:%M}\npH {a:.2f} → {b:.2f}"
                f"   over {h:.1f} h"
                for v, d, a, b, h in zip(part["vat"], part["t0"],
                                         part["ph0"], part["ph1"],
                                         part["hours"])],
                color)
        if model:
            hours = np.linspace(0, float(frame["hours"].max()), 120)
            for label, low, high, color in DRIFT_START_BANDS:
                part = frame[(frame["ph0"] > low) & (frame["ph0"] <= high)]
                if len(part) < 5:
                    continue
                start = float(part["ph0"].median())
                ax.plot(hours,
                        (model["resting"] - start)
                        * (1 - np.exp(-model["k"] * hours)),
                        color=color, linewidth=1.8, alpha=0.9, zorder=4)
        # Hours span four orders of magnitude and 0 has to stay on the
        # axis, so symlog; explicit ticks because the default powers of ten
        # are not hours anybody thinks in.
        ax.set_xscale("symlog", linthresh=4)
        ticks = [0, 1, 2, 4, 8, 24, 72, 168]
        top = float(frame["hours"].max())
        ticks = [t for t in ticks if t <= top * 1.4]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                  facecolor=AXES_BG, edgecolor=GRID, framealpha=0.92,
                  labelcolor=SUBTEXT, title="starting pH", title_fontsize=8.5)
        ax.get_legend().get_title().set_color(SUBTEXT)


# --------------------------------------------------- coating Ni:S over time
# The ratio is read straight from sem_analysis/sem_raw.csv, the per-point EDS
# rows, and derived exactly the way gui_sem.py's build_table does it: every
# reading of an element on an electrode is averaged, anode and cathode points
# pooled, and the ratio is mean(Ni) / mean(S). sem_measurements_filtered.csv
# in this folder carries Ni_p1..p3 and S_p1..p3, which is only the first
# position - deriving from those would silently disagree with the "Quality vs
# ratio" tab in the other app for any electrode measured in five places.
SEM_RAW_FILE = Path(__file__).resolve().parent.parent / "sem_analysis" \
    / "sem_raw.csv"
# Where the first panel ends and the second begins, by marking number.
RATIO_SPLIT = 16124
RATIO_FLOOR = FIRST_PRODUCTION      # nothing below B13900 is production
RATIO_ACCENT = "#64748B"            # slate, for an electrode with no verdict
# Width of the moving average. Chosen so both panels average a comparable
# number of electrodes despite covering very different stretches of time:
# about 22 in the long panel and 16 in the short one. A window measured in
# days rather than in electrodes, because the x axis is time and production
# rate is not constant.
RATIO_WINDOW_DAYS = 5
RATIO_MIN_IN_WINDOW = 4


def load_coating_ratio():
    """One row per electrode: its Ni:S in the coating and when it was made.

    Production time comes from dbo.MetaData rather than the SEM table, so an
    electrode that was measured but never given a coat verdict still has a
    place on the x axis; it is drawn in grey."""
    if not SEM_RAW_FILE.exists():
        print(f"⚠ {SEM_RAW_FILE.name} not found : run run_pipeline.py "
              f"(step 'sem'); the Ni:S tab will be empty")
        return None
    raw = pd.read_csv(SEM_RAW_FILE,
                      usecols=["electrode", "symbol", "value"])
    raw["electrode"] = raw["electrode"].astype(str).str.strip()
    readings = raw.dropna(subset=["symbol", "value"])
    if not len(readings):
        return None
    levels = readings.pivot_table(index="electrode", columns="symbol",
                                  values="value", aggfunc="mean")
    if "Ni" not in levels.columns or "S" not in levels.columns:
        print("⚠ sem_raw.csv has no Ni or no S : the Ni:S tab is empty")
        return None
    # A sulphur reading of zero is not a very high ratio, it is a reading
    # that cannot form one
    sulphur = levels["S"].where(levels["S"] != 0)
    frame = pd.DataFrame({"ratio": levels["Ni"] / sulphur})
    frame = frame.dropna().reset_index()

    folder = Path(__file__).resolve().parent
    meta_file = folder / "electrode_metadata.csv"
    if not meta_file.exists():
        return None
    meta = pd.read_csv(meta_file, usecols=["Marking", "Production_datetime"])
    meta["electrode"] = meta["Marking"].astype(str).str.strip()
    stamp = pd.to_datetime(meta["Production_datetime"], errors="coerce",
                           utc=True)
    meta["made_at"] = stamp.dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    frame = frame.merge(meta[["electrode", "made_at"]], on="electrode",
                        how="inner").dropna(subset=["made_at"])
    number = marking_number(frame["electrode"])
    frame = frame[number >= RATIO_FLOOR]
    return frame.sort_values("made_at") if len(frame) else None


class CoatingRatioWidget(QWidget):
    """The coating's Ni:S against production time, before and after B16124.

    Two panels, one per side of the split, on a shared y axis so a ratio in
    one reads at the same height as in the other. The x axes are separate
    because the panels cover different stretches of time - putting them on
    one axis would leave each panel's points crushed into its own half.

    Higher means less sulphur in the coating. Where the electrode also has a
    coat verdict it is coloured by it; one that was measured but never rated
    is grey rather than dropped, since the ratio is what this tab is about
    and the verdict is only context."""

    PANELS = [("B13900 – B16123", RATIO_FLOOR, RATIO_SPLIT - 1),
              (f"B{RATIO_SPLIT} and up", RATIO_SPLIT, None)]

    TOP_BAND = 42
    BOTTOM_BAND = 58
    LEFT_BAND = 66
    RIGHT_BAND = 18
    # The upper panel ends in its own date ticks - the two cover different
    # stretches of time and cannot share an axis - and the lower one opens
    # with a title, so the gap pays for both.
    ROW_GAP = 76

    def __init__(self, data, ratios):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 10, 28, 8)
        layout.setSpacing(4)

        self._fig = fig = Figure(figsize=(13.0, 6.0), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(280)
        layout.addWidget(canvas, 1)
        canvas.mpl_connect("resize_event", lambda _e: self._apply_margins())
        self._hover = HoverReadout(canvas)

        self._frame = self._build(data, ratios)
        if self._frame is None or not len(self._frame):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No electrode has both a Ni:S and a "
                              "production time", transform=ax.transAxes,
                    ha="center", color=EMPTY, fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        self._rebuild()

    @staticmethod
    def _build(data, ratios):
        """The ratios, with this app's coat verdict attached where it has
        one."""
        if ratios is None or not len(ratios):
            return None
        frame = ratios.copy()
        verdicts = data[["electrode", "verdict"]].copy()
        verdicts["electrode"] = verdicts["electrode"].astype(str).str.strip()
        frame = frame.merge(verdicts.drop_duplicates("electrode"),
                            on="electrode", how="left")
        frame["number"] = marking_number(frame["electrode"])
        return frame

    # -------------------------------------------------------------- drawing
    def _apply_margins(self, draw=True):
        width, height = self._fig.get_size_inches() * self._fig.dpi
        if width <= 0 or height <= 0:
            return
        top = 1 - self.TOP_BAND / height
        bottom = self.BOTTOM_BAND / height
        left = self.LEFT_BAND / width
        right = 1 - self.RIGHT_BAND / width
        if top - bottom < 0.3 or right - left < 0.3:
            left, right, top, bottom = 0.06, 0.98, 0.9, 0.12
        free = height * (top - bottom) - self.ROW_GAP
        self._fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom,
            hspace=2 * self.ROW_GAP / free if free > 0 else 0.3)
        if draw:
            self._canvas.draw_idle()

    def _rebuild(self):
        fig = self._fig
        fig.clear()
        self._hover.reattach()
        grid = fig.add_gridspec(2, 1)
        axes = []
        for index, (title, low, high) in enumerate(self.PANELS):
            ax = fig.add_subplot(grid[index])
            self._draw(ax, title, low, high, last=index == len(self.PANELS) - 1)
            axes.append(ax)

        # One y range across both, or the same ratio sits at two heights and
        # the comparison the split exists to make cannot be read
        spans = [ax.get_ylim() for ax in axes if ax.has_data()]
        if spans:
            low = min(span[0] for span in spans)
            high = max(span[1] for span in spans)
            for ax in axes:
                ax.set_ylim(low, high)
        fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=7,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="D", linestyle="none", markersize=7,
                   color=COLOR_CRACKED, label="cracked"),
            Line2D([], [], marker="o", linestyle="none", markersize=7,
                   color=RATIO_ACCENT, label="not rated"),
            Line2D([], [], color=TARGET_LINE, linewidth=2.2,
                   label=f"{RATIO_WINDOW_DAYS}-day average")],
            loc="upper right", ncol=4, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.995, 0.995))
        self._apply_margins(draw=False)
        self._canvas.draw_idle()

    def _draw(self, ax, title, low, high, last):
        ax.set_facecolor(AXES_BG)
        ax.set_axisbelow(True)
        part = self._frame[self._frame["number"] >= low]
        if high is not None:
            part = part[part["number"] <= high]

        if not len(part):
            ax.text(0.5, 0.5, "No electrodes in this range yet",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                         loc="left", pad=10)
            return

        ax.set_title(f"{title}   ({len(part)} electrodes)", color=TEXT,
                     fontsize=12, fontweight="bold", loc="left", pad=10)
        # Verdict last, so a rated electrode is never hidden under a grey one
        series = ((part["verdict"].isna(), RATIO_ACCENT, "o", "not rated"),
                  (part["verdict"] == 1, COLOR_GOOD, "o", "not cracked"),
                  (part["verdict"] == 0, COLOR_CRACKED, "D", "cracked"))
        for mask, color, marker, label in series:
            chunk = part[mask]
            if not len(chunk):
                continue
            dots = ax.plot(chunk["made_at"], chunk["ratio"], marker=marker,
                           linestyle="none", markersize=5, color=color,
                           markeredgecolor=AXES_BG, markeredgewidth=0.6,
                           alpha=0.85, zorder=3)[0]
            self._hover.add(dots, [
                f"{e}   {d:%d %b %Y %H:%M}\nNi:S {r:.2f}   {label}"
                for e, d, r in zip(chunk["electrode"], chunk["made_at"],
                                   chunk["ratio"])],
                color)

        # The moving average last, so it sits on top of every dot. Centred,
        # so a rise shows at the time it happened rather than a window later.
        rolling = (part.set_index("made_at")["ratio"].sort_index()
                   .rolling(f"{RATIO_WINDOW_DAYS}D", center=True,
                            min_periods=RATIO_MIN_IN_WINDOW).mean().dropna())
        if len(rolling) > 1:
            ax.plot(rolling.index, rolling.to_numpy(dtype=float),
                    color=TARGET_LINE, linewidth=2.2, zorder=5,
                    solid_capstyle="round")

        if last:
            ax.set_xlabel("production date", color=LABEL, fontsize=10,
                          labelpad=7)
        ax.set_ylabel("Ni:S   (higher = less S)", color=LABEL,
                      fontsize=10.5, labelpad=6)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        ax.margins(x=0.04, y=0.08)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)


# Drying runs at one of four stations and an electrode only ever sees one of
# them, so the station is a category rather than a dose. All four are set to
# the same 30 minutes, which is why the times are a spike with a thin tail.
DRYING_STEPS = [8, 9, 10, 11]
DRYING_SET_TIME = 1800


def wilson(successes, trials, z=1.96):
    """A 95% interval for a proportion that still behaves at small n.

    The textbook p +- 1.96*sqrt(p(1-p)/n) runs off the end of the scale and
    collapses to zero width when nobody cracked. Wilson's does neither, which
    matters here: two of the four stations have fewer than twenty electrodes
    and a bar without honest uncertainty invites reading noise as a finding."""
    if not trials:
        return 0.0, 0.0
    hit = successes / trials
    span = 1 + z * z / trials
    centre = (hit + z * z / (2 * trials)) / span
    half = z * math.sqrt(hit * (1 - hit) / trials
                         + z * z / (4 * trials * trials)) / span
    return max(0.0, centre - half), min(1.0, centre + half)


class DryingWidget(QWidget):
    """Drying station against cracking, and time in the dryer against it.

    Two questions, one per panel. Top: does any station crack more than the
    others? Bottom: within a station, did the electrodes that cracked spend
    longer in there? Both are asked of the same electrodes, each counted once.

    The time axis covers the full range, stragglers included: every station
    is set to 1800 s and almost every electrode leaves within seconds of it,
    so the picture is a dense line on the setpoint with a thin tail above."""

    def __init__(self, data, long_steps):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)
        self._fig = fig = Figure(figsize=(11.5, 7.5), dpi=100)
        fig.patch.set_facecolor(FIG_BG)
        self._canvas = FigureCanvas(fig)
        layout.addWidget(self._canvas)

        frame = self._build(data, long_steps)
        if frame is None or not len(frame):
            ax = fig.add_subplot()
            ax.set_facecolor(AXES_BG)
            ax.text(0.5, 0.5, "No drying steps in the data",
                    transform=ax.transAxes, ha="center", color=EMPTY,
                    fontsize=12, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return
        self._draw(frame)

    @staticmethod
    def _build(data, long_steps):
        if long_steps is None or not len(long_steps):
            return None
        steps = long_steps.copy()
        steps["electrode"] = steps["electrode"].astype(str).str.strip()
        steps = steps[steps["step"].isin(DRYING_STEPS)]
        if not len(steps):
            return None
        # One electrode was logged at two stations; keep the first it saw so
        # no electrode is counted twice or claimed by two bars.
        steps = (steps.sort_values(["electrode", "step", "pass"])
                 .drop_duplicates("electrode", keep="first"))
        verdicts = data[["electrode", "verdict"]].copy()
        verdicts["electrode"] = verdicts["electrode"].astype(str).str.strip()
        frame = verdicts.merge(steps[["electrode", "step", "actual_time"]],
                               on="electrode", how="inner")
        frame = frame.dropna(subset=["actual_time", "verdict"])
        frame["cracked"] = (frame["verdict"] == 0).astype(int)
        return frame

    def _draw(self, frame):
        fig = self._fig
        stations = [step for step in DRYING_STEPS
                    if (frame["step"] == step).any()]
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.25])
        self._rates(fig.add_subplot(grid[0]), frame, stations)
        self._times(fig.add_subplot(grid[1]), frame, stations)
        fig.legend(handles=[
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_GOOD, label="not cracked"),
            Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=COLOR_CRACKED, label="cracked")],
            loc="upper right", ncol=2, frameon=False, fontsize=9,
            labelcolor=SUBTEXT, bbox_to_anchor=(0.99, 0.995))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08,
                            hspace=0.45)
        self._canvas.draw_idle()

    def _chrome(self, ax, title, ylabel):
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                     loc="left", pad=10)
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10, labelpad=8)
        ax.set_facecolor(AXES_BG)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _rates(self, ax, frame, stations):
        """Share cracked at each station, with an interval wide enough to
        show how little the thin stations actually pin down."""
        totals = [int((frame["step"] == step).sum()) for step in stations]
        cracked = [int(frame.loc[frame["step"] == step, "cracked"].sum())
                   for step in stations]
        rates = [100 * k / n for k, n in zip(cracked, totals)]
        bounds = [wilson(k, n) for k, n in zip(cracked, totals)]
        lower = [rate - 100 * low for rate, (low, _h) in zip(rates, bounds)]
        upper = [100 * high - rate for rate, (_l, high) in zip(rates, bounds)]

        x = np.arange(len(stations))
        ax.bar(x, rates, width=0.55, color=COLOR_CRACKED, alpha=0.85,
               zorder=2)
        ax.errorbar(x, rates, yerr=[lower, upper], fmt="none", ecolor=SUBTEXT,
                    elinewidth=1.1, capsize=5, zorder=3)
        overall = 100 * sum(cracked) / sum(totals)
        ax.axhline(overall, color=TARGET_LINE, linewidth=1.3,
                   linestyle=(0, (4, 2)), zorder=1)
        ax.text(0.995, overall, f"all stations {overall:.0f}%  ",
                transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                color=SUBTEXT, fontsize=9)
        for pos, rate, k, n in zip(x, rates, cracked, totals):
            ax.text(pos, rate + 1.5, f"{rate:.0f}%", ha="center",
                    va="bottom", color=TEXT, fontsize=10, fontweight="bold")
        ax.set_xticks(x, [f"Step {step}\n{k} cracked / {n}"
                          for step, k, n in zip(stations, cracked, totals)])
        ax.set_xlim(-0.6, len(stations) - 0.4)
        ax.set_ylim(0, max(100 * high for _l, high in bounds) * 1.12)
        self._chrome(ax, "Cracked share by drying station", "% cracked")

    def _times(self, ax, frame, stations):
        """Time in the dryer, each electrode a dot, cracked beside not."""
        every = frame["actual_time"].to_numpy(dtype=float)
        ceiling, floor = float(every.max()), float(every.min())
        margin = max((ceiling - floor) * 0.04, 20)
        rng = np.random.default_rng(7)

        for index, step in enumerate(stations):
            part = frame[frame["step"] == step]
            sides = ((part.loc[part["cracked"] == 0, "actual_time"],
                      -0.17, COLOR_GOOD),
                     (part.loc[part["cracked"] == 1, "actual_time"],
                      0.17, COLOR_CRACKED))
            for values, offset, color in sides:
                y = values.to_numpy(dtype=float)
                if not len(y):
                    continue
                centre = index + offset
                jitter = centre + rng.uniform(-0.06, 0.06, len(y))
                ax.scatter(jitter, y, s=26, color=color, alpha=0.75,
                           linewidths=0, zorder=3)
                ax.plot([centre - 0.13, centre + 0.13],
                        [np.median(y)] * 2, color=TARGET_LINE, linewidth=2,
                        zorder=5)

        ax.axhline(DRYING_SET_TIME, color=SPINE, linewidth=1.1,
                   linestyle=(0, (2, 3)), zorder=1)
        # Below the line, not above it: above runs into the step 11 median
        ax.text(0.995, DRYING_SET_TIME, f"set {DRYING_SET_TIME} s  ",
                transform=ax.get_yaxis_transform(), va="top", ha="right",
                color=SUBTEXT, fontsize=9)
        counts = [(int(((frame["step"] == step)
                        & (frame["cracked"] == 0)).sum()),
                   int(((frame["step"] == step)
                        & (frame["cracked"] == 1)).sum()))
                  for step in stations]
        ax.set_xticks(np.arange(len(stations)),
                      [f"Step {step}\n{good} / {bad}"
                       for step, (good, bad) in zip(stations, counts)])
        ax.set_xlim(-0.6, len(stations) - 0.4)
        ax.set_ylim(floor - margin, ceiling + margin)
        self._chrome(ax, "Time in the dryer, cracked vs not", "seconds")


class MainWindow(QMainWindow):
    def __init__(self, data, baths, chemicals=None, plating=None,
                 strike_times=None, rinse_times=None, drip_times=None,
                 temperatures=None, cumulative=None, cleaner=None,
                 ph_control=None, throughput=None, ratios=None):
        super().__init__()
        self.setWindowTitle("Cracking analysis")

        tabs = QTabWidget()
        tab_data = [
            ("Activation", data, "Activator bath chemistry", ACTIVATOR_SERIES,
             ACTIVATOR_LIMITS, 18),
            ("Strike", data, "Strike bath chemistry", STRIKE_SERIES,
             STRIKE_LIMITS, 31),
            ("Plate 28", data[data["vats"] == "28"],
             "Plate bath 28 chemistry", PLATE_SERIES, PLATE_LIMITS, 28),
            ("Plate 29", data[data["vats"] == "29"],
             "Plate bath 29 chemistry", PLATE_SERIES, PLATE_LIMITS, 29),
            ("Plate 30", data[data["vats"] == "30"],
             "Plate bath 30 chemistry", PLATE_SERIES, PLATE_LIMITS, 30),
        ]
        for name, subset, bath_title, series, limits, chem_vat in tab_data:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(28, 16, 28, 16)

            # All plots share margins and x-limits so the dates line up
            # column-perfect. The right margin starts at the screen edge;
            # the bath widget shrinks it dynamically (via bind_margin_callback)
            # only when extra right-side axes are actually enabled.
            bath_df = baths.get(name)
            margins = (0.09, 0.97)
            dates = list(subset["production_date"].dropna())
            if bath_df is not None and len(bath_df):
                dates += list(bath_df["Date"])
            xlim = None
            if dates:
                pad = pd.Timedelta(days=1)
                xlim = (min(dates) - pad, max(dates) + pad)

            verdict_widget = VerdictPlotWidget(subset, name, margins, xlim,
                                               bath_df, limits)
            bath_widget = BathPlotWidget(bath_df, bath_title, series,
                                         margins, xlim, limits)
            chem_subset = None
            if chemicals is not None and len(chemicals):
                chem_subset = chemicals[chemicals["vat"] == chem_vat]
            chem_widget = ChemicalsPlotWidget(
                chem_subset, f"Chemicals added : vat {chem_vat}",
                margins, xlim)
            layout.addWidget(verdict_widget)
            layout.addWidget(bath_widget)
            layout.addWidget(chem_widget)
            tabs.addTab(page, name)

            # Keep the charts' right edges moving together when bath
            # axes are toggled on/off
            canvases = (*verdict_widget.canvases, bath_widget._canvas,
                        chem_widget._canvas)

            def apply_right(right, cvs=canvases):
                for cv in cvs:
                    cv.figure.subplots_adjust(right=right)
                    cv.draw_idle()

            bath_widget.bind_margin_callback(apply_right)
            bath_widget.bind_selection_callback(
                verdict_widget.set_breach_columns)

        tabs.addTab(StepTimeWidget(data, rinse_times,
                                   StepTimeWidget.RINSE_GROUPS,
                                   "Rinse time", "rinse step",
                                   group_column="vat", step_checkboxes=True),
                    "Rinse time")
        tabs.addTab(StepTimeWidget(data, drip_times,
                                   StepTimeWidget.DRIP_GROUPS,
                                   "Drip time", "drip step",
                                   group_column="vat", step_checkboxes=True,
                                   isolate=["26"]),
                    "Drip time")
        tabs.addTab(StepTimeWidget(data, strike_times,
                                   StepTimeWidget.STRIKE_GROUPS,
                                   "Strike time", "strike step"),
                    "Strike time")
        tabs.addTab(StepTimeWidget(data, plating,
                                   StepTimeWidget.PLATING_GROUPS,
                                   "Plating time", "plating vat"),
                    "Plating time")
        tabs.addTab(CumulativeOverrunWidget(data, cumulative, baths),
                    "Cumulative overrun")
        tabs.addTab(ChargeReportWidget(data, cumulative, baths),
                    "Charge report")
        tabs.addTab(DryingWidget(data, cumulative), "Drying")
        signal_units = (("Temperature", "°C"), ("Current", "A"),
                        ("Voltage", "V"))
        for signal, unit in signal_units:
            tabs.addTab(SignalWidget(data, temperatures, signal, unit),
                        signal)
        for signal, unit in signal_units:
            tabs.addTab(SignalTimelineWidget(data, temperatures, signal, unit),
                        f"{signal} over time")
        tabs.addTab(CleanerTimelineWidget(data, temperatures, cleaner),
                    "Cleaner over time")
        tabs.addTab(CleanerScatterWidget(data, temperatures, cleaner),
                    "Cleaner vs Slotoclean")
        tabs.addTab(CleanerScatterInterpolatedWidget(data, temperatures,
                                                    cleaner),
                    "Cleaner vs Slotoclean (interpolated)")
        tabs.addTab(PhDosingWidget(data, ph_control), "pH dosing")
        tabs.addTab(PhSpaceWidget(data, ph_control), "pH space (3D)")
        tabs.addTab(CrackStyleWidget(data, ph_control), "Crack style vs pH")
        tabs.addTab(IdleDriftWidget(data, ph_control, throughput),
                    "Idle pH drift")
        tabs.addTab(CoatingRatioWidget(data, ratios), "Coating Ni:S")

        self.setCentralWidget(tabs)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def load_bath_data(path):
    if not path.exists():
        print(f"⚠ {path.name} not found : bath plot will be empty")
        return None
    bath = pd.read_csv(path)
    bath["Date"] = pd.to_datetime(bath["Date"])
    return bath.sort_values("Date")


def load_all_baths():
    folder = Path(__file__).resolve().parent
    return {
        "Activation": load_bath_data(folder / "activator_bath_filtered.csv"),
        "Strike": load_bath_data(BATH_FILE),
        "Plate 28": load_bath_data(folder / "plate_bath_28_filtered.csv"),
        "Plate 29": load_bath_data(folder / "plate_bath_29_filtered.csv"),
        "Plate 30": load_bath_data(folder / "plate_bath_30_filtered.csv"),
    }


def load_step_times(filename, what):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        print(f"⚠ {path.name} not found — {what} tab will be empty")
        return None
    return pd.read_csv(path)


def load_signals():
    """Prefer the dense 10-second grid over the raw change-logged rows.

    ProductionData only stores a row when a value moves, so a strike that ran
    an hour at a constant 780 A is two rows - 780 and 0 - and their median is
    390 A, a current the process never ran at. build_signal_grid.py rebuilds
    each electrode on a 10-second grid the way the electrode report does, and
    the statistics come out time-weighted."""
    folder = Path(__file__).resolve().parent
    grid = folder / "signals_grid.csv"
    if grid.exists():
        return pd.read_csv(grid)
    print("⚠ signals_grid.csv not found : falling back to the raw rows in "
          "signals.csv, whose current values are averaging artefacts. "
          "Run build_signal_grid.py.")
    return load_step_times("signals.csv", "temperature / current / voltage")


def load_cleaner():
    """The cathodic cleaner bath log : pH per round and, on the rounds that
    were titrated, the Slotoclean EL DCG concentration."""
    path = Path(__file__).resolve().parent / "cleaner_bath_long.csv"
    if not path.exists():
        print(f"⚠ {path.name} not found : the cleaner timeline's "
              f"Slotoclean panel will be empty")
        return None
    return pd.read_csv(path, parse_dates=["measured_at"])


def load_vat_throughput():
    """When every electrode passed through each plating vat, in Oslo time.

    strike_vats.csv is one row per plating STEP, not per electrode - B15000
    has three rows in vat 30 - so it is deduplicated to one visit per vat.
    Counting the raw rows triples the electrodes in every interval, which
    would turn idle stretches into busy ones."""
    folder = Path(__file__).resolve().parent
    vats, meta = folder / "strike_vats.csv", folder / "electrode_metadata.csv"
    if not vats.exists() or not meta.exists():
        print("\u26a0 strike_vats.csv / electrode_metadata.csv not found : the "
              "idle drift tab cannot tell an idle bath from a busy one")
        return None
    rows = pd.read_csv(vats, usecols=["Marking", "Vat"])
    rows["electrode"] = rows["Marking"].astype(str).str.strip()
    rows["vat"] = rows["Vat"].astype(str)
    rows = rows.drop_duplicates(["electrode", "vat"])
    made = pd.read_csv(meta, usecols=["Marking", "Production_datetime"])
    made["electrode"] = made["Marking"].astype(str).str.strip()
    stamp = pd.to_datetime(made["Production_datetime"], errors="coerce",
                           utc=True)
    made["made_at"] = stamp.dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    joined = rows.merge(made[["electrode", "made_at"]], on="electrode",
                        how="inner").dropna(subset=["made_at"])
    joined = joined[joined["vat"].isin(PH_VATS)]
    return joined.sort_values("made_at") if len(joined) else None


def load_ph_control():
    """The pH check log, long: one row per vat and reading, with whatever was
    dosed in response. Oslo local time, like the other lab logs."""
    path = Path(__file__).resolve().parent / "ph_control_filtered.csv"
    if not path.exists():
        print(f"⚠ {path.name} not found : the pH dosing tab will be "
              f"empty")
        return None
    log = pd.read_csv(path, parse_dates=["measured_at"])
    log["vat"] = log["vat"].astype(str)
    return log.sort_values(["vat", "measured_at"])


def load_chemicals():
    path = Path(__file__).resolve().parent / "adding_chemicals_filtered.csv"
    if not path.exists():
        print(f"⚠ {path.name} not found : chemicals plot will be empty")
        return None
    chems = pd.read_csv(path)
    chems["Date"] = pd.to_datetime(chems["Date"])
    return chems.sort_values("Date")


def main():
    data = load_data()
    data["vats"] = data["vats"].astype(str)
    baths = load_all_baths()
    chemicals = load_chemicals()

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background: {FIG_BG}; }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            background: transparent;
            padding: 10px 24px;
            font-size: 14px;
            color: {SUBTEXT};
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT};
            font-weight: bold;
            border-bottom: 2px solid {COLOR_GOOD};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
    """)
    window = MainWindow(
        data, baths, chemicals,
        load_step_times("plating_times.csv", "plating time"),
        load_step_times("strike_step_times.csv", "strike time"),
        load_step_times("rinse_times.csv", "rinse time"),
        load_step_times("drip_times.csv", "drip time"),
        load_signals(),
        load_step_times("all_step_times_long.csv", "cumulative overrun"),
        load_cleaner(),
        load_ph_control(),
        load_vat_throughput(),
        load_coating_ratio())
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
