"""How well does each column of the all-data table track cracking?

One signed number per column: Spearman's rho against coating_label, which is
1 cracked and 0 not. Rank-based, so an outlier cannot carry it and a
monotone-but-bent relationship still shows; and it is the measure the rest of
this investigation used, so the numbers stay comparable with the S/Ni finding
(rho = -0.344).

Columns with no natural order - the vat, the operator - get Cramer's V
instead, in its own column so it is never read as a rho, plus the cracked
rate per category underneath, which is the part you would act on.

Nothing here is a cause. Two things in particular will light up together and
mean one thing: the fifteen "age" columns are all proxies for WHEN an
electrode was made, and so is the date, so any drift in cracking over the
autumn marks all of them at once.

The honest caveats, all visible in the table rather than buried here:

  * 101 columns against 64 cracked electrodes. At p < 0.05 about five
    columns are expected to look real by chance, which is why a
    Benjamini-Hochberg q sits beside every p. On past evidence almost
    nothing survives it; that is the finding, not a fault in the tab.
  * a column an electrode only sometimes visits - the plating vats, the
    dryers - is a subgroup comparison wearing a correlation's clothes, and
    says so in its note.
  * the bath columns share readings between electrodes: 223 electrodes carry
    28 plate-bath samples between them. The effective sample size is the
    number of readings, not the number of electrodes, so those p values are
    optimistic. Marked as shared.
"""
import math

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView)

from gui_app import (AXES_BG, COLOR_CRACKED, COLOR_GOOD, FIG_BG, GRID, LABEL,
                     SPINE, SUBTEXT, TEXT, MultiLineHeader, NumericItem)

TARGET = "coating_label"
# Below this a rho is noise: a three-electrode column can read -0.62 and
# would otherwise top the sort.
MIN_N = 30
# One value covering this much of a column leaves nothing to correlate, even
# where a second value technically exists (the plating current is 709 A on
# all but one electrode).
NEAR_CONSTANT_SHARE = 0.95
PERMUTATIONS = 2000
SEED = 20260924

# coat IS the verdict coating_label is derived from, so its correlation is 1
# by construction and tells you only that the derivation works.
TARGET_COLUMNS = {"coat", "coating_label"}
IDENTIFIERS = {"electrode"}
# No natural order, so a rank correlation is meaningless
CATEGORICAL = {"vat", "chem_by"}
# An electrode visits exactly one of each set, so the column is only filled
# for that subgroup and the rho describes the subgroup, not the line.
EXCLUSIVE = {
    "pos28": "plating vat", "pos29": "plating vat", "pos30": "plating vat",
    "pos8": "dryer", "pos9": "dryer", "pos10": "dryer", "pos11": "dryer",
}
# Readings shared between electrodes: the effective sample size is the number
# of readings, well below the number of electrodes.
SHARED_PREFIXES = ("plate_", "act_", "stk_", "cleaner_", "slotoclean",
                   "water_", "cond", "ph", "chem_")


def spearman(values, labels):
    """Signed rank correlation, and a two-sided p. Returns (rho, p, n).

    The p comes from the Fisher z transform of the rank correlation rather
    than from a t table: scipy is not a dependency of this repo, and at these
    sample sizes the approximation is accurate well past the precision
    anybody reads off a screen."""
    pair = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(values).reset_index(drop=True),
                           errors="coerce"),
        "y": pd.to_numeric(pd.Series(labels).reset_index(drop=True),
                           errors="coerce"),
    }).dropna()
    n = len(pair)
    if n < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None, None, n
    rho = pair["x"].rank().corr(pair["y"].rank())
    if rho is None or not np.isfinite(rho):
        return None, None, n
    rho = float(rho)
    if n < 6 or abs(rho) >= 1.0:
        return rho, None, n
    z = (0.5 * math.log((1 + rho) / (1 - rho))
         * math.sqrt((n - 3) / 1.06))
    return rho, float(math.erfc(abs(z) / math.sqrt(2))), n


def _chi_square(table):
    """Pearson statistic for a contingency table given as an array."""
    total = table.sum()
    if total <= 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0,
                         (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def cramers_v(values, labels):
    """Unsigned 0-1 association for a column with no order.

    The p is a permutation test rather than a chi-square table: the operator
    column has twenty categories and several hold a handful of electrodes,
    which is exactly where the tabulated distribution is least trustworthy."""
    pair = pd.DataFrame({
        "g": pd.Series(values).reset_index(drop=True).astype("object"),
        "y": pd.to_numeric(pd.Series(labels).reset_index(drop=True),
                           errors="coerce"),
    }).dropna()
    n = len(pair)
    counts = pd.crosstab(pair["g"], pair["y"])
    if n < MIN_N or min(counts.shape) < 2:
        return None, None, n, counts

    observed = _chi_square(counts.to_numpy(dtype=float))
    smaller = min(counts.shape) - 1
    v = math.sqrt(observed / (n * smaller)) if smaller > 0 else None

    rng = np.random.default_rng(SEED)
    groups = pair["g"].to_numpy()
    outcome = pair["y"].to_numpy()
    beaten = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(outcome)
        table = pd.crosstab(groups, shuffled).to_numpy(dtype=float)
        if _chi_square(table) >= observed - 1e-12:
            beaten += 1
    # +1 on both sides: a permutation test cannot honestly report p = 0
    p = (beaten + 1) / (PERMUTATIONS + 1)
    return v, p, n, counts


def benjamini_hochberg(pvalues):
    """Expected false-discovery share at each p, aligned to the input."""
    # pd.isna, not "is not None": once these p values have been through a
    # DataFrame the missing ones are NaN, and NaN passes an "is not None"
    # test, then poisons the running minimum and lets q fall below p.
    present = [(index, float(p)) for index, p in enumerate(pvalues)
               if p is not None and not pd.isna(p)]
    out = [None] * len(pvalues)
    if not present:
        return out
    present.sort(key=lambda pair: pair[1])
    total = len(present)
    running = 1.0
    # Walk from the largest p down, keeping the running minimum, so the
    # q values never decrease as p increases
    for rank in range(total, 0, -1):
        index, p = present[rank - 1]
        running = min(running, p * total / rank)
        out[index] = running
    return out


def _note_for(key, series, n):
    """Why this column cannot carry a correlation, or a caveat about it."""
    if key in TARGET_COLUMNS:
        return "this is the target", True
    if key in IDENTIFIERS:
        return "row identifier, not a measurement", True
    filled = series.dropna()
    if not len(filled):
        return "never filled", True
    counts = filled.value_counts()
    share = float(counts.iloc[0]) / len(filled)
    if counts.size < 2:
        return f"constant ({counts.index[0]})", True
    if share >= NEAR_CONSTANT_SHARE:
        return (f"{share:.0%} of electrodes share one value "
                f"({counts.index[0]})", True)
    if n < MIN_N:
        return f"only {n} electrodes", True
    notes = []
    if key in EXCLUSIVE:
        notes.append(f"subgroup: one {EXCLUSIVE[key]} only")
    if key == "date":
        notes.append("time trend, not a cause")
    if key.endswith("_age"):
        notes.append("proxy for when it was made")
    if any(key.startswith(prefix) for prefix in SHARED_PREFIXES) \
            and not key.endswith("_age"):
        notes.append("reading shared between electrodes")
    return "; ".join(notes), False


def analyse(frame, columns):
    """One row per column of the all-data table, plus category breakdowns.

    `columns` is the (key, title, kind) list the table was built from, so the
    correlation tab always covers exactly what the data tab shows."""
    if TARGET not in frame.columns:
        return pd.DataFrame()
    labels = pd.to_numeric(frame[TARGET], errors="coerce")

    rows = []
    for key, title, kind in columns:
        if key not in frame.columns:
            continue
        series = frame[key]
        if kind == "date":
            values = (pd.to_datetime(series, errors="coerce")
                      - pd.to_datetime(series, errors="coerce").min()).dt.days
        elif kind in ("int", "float", "float2"):
            values = pd.to_numeric(series, errors="coerce")
        else:
            values = series
        usable = pd.DataFrame({"v": values.reset_index(drop=True),
                               "y": labels.reset_index(drop=True)}).dropna()
        note, blocking = _note_for(key, values, len(usable))

        row = {"key": key, "title": title.replace("\n", " "),
               "n": len(usable), "rho": None, "v": None, "p": None,
               "note": note, "level": 0}
        if blocking:
            rows.append(row)
            continue
        if key in CATEGORICAL:
            strength, p, n, counts = cramers_v(values, labels)
            row.update({"v": strength, "p": p, "n": n})
            rows.append(row)
            # The per-category rates, which is what you would act on
            for category in counts.index:
                seen = int(counts.loc[category].sum())
                cracked = int(counts.loc[category].get(1, 0))
                rows.append({
                    "key": f"{key}::{category}", "level": 1,
                    "title": f"{category}", "n": seen,
                    "rho": None, "v": None, "p": None,
                    "note": f"{100 * cracked / seen:.0f}% cracked "
                            f"({cracked} of {seen})" if seen else "",
                })
            continue
        rho, p, n = spearman(values, labels)
        row.update({"rho": rho, "p": p, "n": n})
        rows.append(row)

    out = pd.DataFrame(rows)
    if not len(out):
        return out
    # q over the columns that produced a p, not over the category rows
    tested = out["level"] == 0
    out.loc[tested, "q"] = benjamini_hochberg(list(out.loc[tested, "p"]))
    if "q" not in out.columns:
        out["q"] = None

    # Strongest first, then the columns that could not be tested. A category
    # row always travels directly under its own column.
    # pd.notna again: rho is NaN rather than None by now, and abs(NaN) is
    # NaN, which would sort every categorical column to the bottom however
    # strong its V.
    def strength(row):
        if pd.notna(row["rho"]):
            return abs(float(row["rho"]))
        if pd.notna(row["v"]):
            return float(row["v"])
        return -1.0

    out["strength"] = out.apply(strength, axis=1)
    parents = out[out["level"] == 0].sort_values(
        "strength", ascending=False, kind="stable")
    ordered = []
    children = {k: [] for k in parents["key"]}
    for _index, child in out[out["level"] == 1].iterrows():
        children.setdefault(child["key"].split("::")[0], []).append(child)
    for _index, parent in parents.iterrows():
        ordered.append(parent)
        ordered.extend(children.get(parent["key"], []))
    return pd.DataFrame(ordered).reset_index(drop=True)


class CorrelationTab(QWidget):
    """The analysis as a sortable table, one row per column of the data tab."""

    COLUMNS = [("title", "Column", "text"),
               ("rho", "Spearman\nrho", "float3"),
               ("v", "Cramer's\nV", "float3"),
               ("n", "n", "int"),
               ("p", "p", "sci"),
               ("q", "q\n(BH)", "sci"),
               ("note", "Note", "text")]

    def __init__(self, frame, columns, heading="Coating correlation"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        title = QLabel(heading)
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch()
        layout.addLayout(bar)

        if frame is None or not len(frame) or TARGET not in frame.columns:
            layout.addWidget(QLabel("No data to correlate."))
            return

        found = analyse(frame, columns)
        labels = pd.to_numeric(frame[TARGET], errors="coerce")
        cracked = int((labels == 1).sum())
        tested = int(((found["level"] == 0)
                      & found["p"].notna()).sum()) if len(found) else 0
        survived = int(((found["level"] == 0)
                        & (found["q"] <= 0.05)).sum()) if len(found) else 0
        caption = QLabel(
            f"{len(frame)} electrodes, {cracked} cracked.  "
            f"{tested} of {len(columns)} columns could be tested; "
            f"{survived} clear a 5% false-discovery rate.  "
            f"rho is signed: positive means more of it goes with more "
            f"cracking.")
        caption.setStyleSheet(f"color: {SUBTEXT}; font-size: 12px;")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        layout.addWidget(self._table(found))

    def _table(self, found):
        widget = QTableWidget()
        widget.setColumnCount(len(self.COLUMNS))
        widget.setHorizontalHeaderLabels([t for _k, t, _kind in self.COLUMNS])
        widget.setRowCount(len(found))
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

        for row, record in found.iterrows():
            for column, (key, _title, kind) in enumerate(self.COLUMNS):
                widget.setItem(row, column, self._cell(record, key, kind))

        header = MultiLineHeader(widget)
        font = header.font()
        font.setPixelSize(11)
        font.setBold(True)
        header.setFont(font)
        header.setMinimumSectionSize(44)
        header.label_color = QColor(LABEL)
        widget.setHorizontalHeader(header)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        widget.setSortingEnabled(True)
        for column, (key, _t, _kind) in enumerate(self.COLUMNS):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch if key == "note"
                else QHeaderView.ResizeMode.ResizeToContents)
        return widget

    @staticmethod
    def _cell(record, key, kind):
        value = record.get(key)
        blank = value is None or (not isinstance(value, str)
                                  and pd.isna(value))
        if key == "title":
            # A category row is indented under the column it belongs to
            text = str(value)
            item = QTableWidgetItem(("      " if record.get("level") else "")
                                    + text)
            if record.get("level"):
                item.setForeground(QColor(SUBTEXT))
            return item
        if blank:
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, "")
            return item
        if kind == "int":
            return NumericItem(value, f"{value:,.0f}")
        if kind == "float3":
            item = NumericItem(value, f"{value:+.3f}")
            # A strong association is worth seeing without reading the number
            if abs(value) >= 0.2:
                item.setForeground(QColor(COLOR_CRACKED if value > 0
                                          else COLOR_GOOD))
            return item
        if kind == "sci":
            text = f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"
            return NumericItem(value, text)
        item = QTableWidgetItem(str(value))
        item.setForeground(QColor(SUBTEXT))
        return item
