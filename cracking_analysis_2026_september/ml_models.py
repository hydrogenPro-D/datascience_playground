"""Can a model predict cracking from this table? Random forest and a small
fully-connected network, each scored honestly.

Four decisions do most of the work here, and all four are there to stop the
numbers flattering themselves:

  GROUPED cross-validation. Electrodes made on the same day share a bath
  reading, a conductivity round and often an operator. Splitting at random
  puts near-copies of the same row on both sides of the fold and the score
  climbs for no good reason, so folds are cut by PRODUCTION DAY: a day is
  wholly in training or wholly in test.

  AUC, not accuracy. 71 of 277 electrodes cracked, so a model that answers
  "not cracked" every time is 74% accurate and useless. Area under the ROC
  curve asks whether a cracked electrode is ranked above a sound one, and
  0.5 is the score of a coin.

  A SHUFFLED-LABEL null. The same model, the same folds, the same number of
  features, but the verdicts shuffled. That distribution is what "no signal"
  looks like at this sample size, and the real score is only interesting
  where it sits outside it. With ~110 features and 277 rows a flexible model
  will fit noise, and this is what exposes it.

  With and WITHOUT the SEM columns. Ni:S, Ni, O and the rest are measured on
  the same SEM scan that produced the verdict. A model using them is not
  predicting cracking from the process, it is describing what a cracked
  coating looks like - useful, but a different question, and it cannot be
  used to steer the line. The process-only row answers that second question.

Permutation importance rather than the forest's own impurity importance,
which inflates whatever has many distinct values - here the age columns,
which have a different value for nearly every electrode. And measured on
each held-out fold rather than on a model fitted to everything: a forest
that has memorised its training rows barely notices a feature being
shuffled, so importances taken that way come out near zero and rank
nothing.
"""
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except ImportError:                                     # pragma: no cover
    SKLEARN = False

TARGET = "coating_label"
# Never features: the target, its own source, and the row's name
EXCLUDED = {"coat", "coating_label", "electrode", "date"}
# Measured on the same SEM scan as the verdict
SEM_PREFIX = "sem_"
FOLDS = 5
# Enough to place the real score against the null without making the
# tab take minutes; the floor on p is 1/(rounds+1)
NULL_ROUNDS = 20
SEED = 20260929


def available():
    return SKLEARN


def build_model(kind, seed=SEED):
    """A forest, or a small fully connected network.

    The network is deliberately small - one hidden layer of 32 - because
    277 rows cannot support anything deeper without memorising them."""
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("net", MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0,
                              max_iter=2000, early_stopping=True,
                              n_iter_no_change=20, random_state=seed)),
    ])


def prepare(frame, columns, drop_sem=False):
    """(X, y, groups, names). Numeric columns plus the vat, one-hot."""
    label = pd.to_numeric(frame[TARGET], errors="coerce")
    usable = label.notna()
    frame = frame[usable]
    label = label[usable].astype(int)

    names, values = [], []
    for key, _title, kind in columns:
        if key in EXCLUDED or kind not in ("int", "float", "float2"):
            continue
        if drop_sem and key.startswith(SEM_PREFIX):
            continue
        column = pd.to_numeric(frame[key], errors="coerce")
        if column.notna().sum() < 20 or column.nunique(dropna=True) < 2:
            continue
        names.append(key)
        values.append(column.to_numpy(dtype=float))

    # The vat is a real candidate and has no order, so one column each
    if "vat" in frame.columns:
        for vat in sorted(v for v in frame["vat"].dropna().unique()):
            names.append(f"vat_{vat}")
            values.append((frame["vat"] == vat).to_numpy(dtype=float))

    features = np.column_stack(values) if values else np.empty((len(frame), 0))
    # Same-day electrodes share their bath readings, so a day is one group
    day = pd.to_datetime(frame.get("date"), errors="coerce").dt.date
    groups = day.fillna(pd.Timestamp("1970-01-01").date()).astype(str)
    return features, label.to_numpy(), groups.to_numpy(), names


def cross_validate(model_kind, features, label, groups, names=None,
                   seed=SEED):
    """Out-of-fold AUC, folds cut by production day, and - when `names` is
    given - permutation importance measured on each held-out fold."""
    if len(np.unique(label)) < 2:
        return None, None, pd.DataFrame()
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
                                    random_state=seed)
    predicted = np.full(len(label), np.nan)
    gains = []
    for train, test in splitter.split(features, label, groups):
        if len(np.unique(label[train])) < 2:
            continue
        model = build_model(model_kind, seed)
        if model_kind == "forest":
            filler = SimpleImputer(strategy="median")
            fitted_on = filler.fit_transform(features[train])
            scored_on = filler.transform(features[test])
        else:
            fitted_on, scored_on = features[train], features[test]
        model.fit(fitted_on, label[train])
        predicted[test] = model.predict_proba(scored_on)[:, 1]

        if names and len(np.unique(label[test])) > 1:
            # How much test AUC this fold loses when a column is shuffled
            gains.append(permutation_importance(
                model, scored_on, label[test], n_repeats=5,
                random_state=seed, scoring="roc_auc",
                n_jobs=-1).importances_mean)

    done = ~np.isnan(predicted)
    if done.sum() < 10 or len(np.unique(label[done])) < 2:
        return None, None, pd.DataFrame()
    auc = float(roc_auc_score(label[done], predicted[done]))

    ranked = pd.DataFrame()
    if gains:
        mean = np.mean(gains, axis=0)
        ranked = pd.DataFrame({"feature": names, "importance": mean,
                               "spread": np.std(gains, axis=0)})
        ranked = ranked.sort_values("importance",
                                    ascending=False).reset_index(drop=True)
    return auc, predicted, ranked


def null_distribution(model_kind, features, label, groups, rounds=NULL_ROUNDS):
    """The same pipeline on shuffled verdicts: what no signal scores here."""
    rng = np.random.default_rng(SEED)
    scores = []
    for index in range(rounds):
        shuffled = rng.permutation(label)
        auc, _p, _i = cross_validate(model_kind, features, shuffled, groups,
                                     seed=SEED + index)
        if auc is not None:
            scores.append(auc)
    return np.array(scores)


def confusion(label, predicted, threshold):
    """Counts at one threshold, plus what people actually ask about.

    recall is the share of cracked electrodes the model catches; precision
    the share of its warnings that were right."""
    flagged = predicted >= threshold
    true_pos = int((flagged & (label == 1)).sum())
    false_pos = int((flagged & (label == 0)).sum())
    false_neg = int((~flagged & (label == 1)).sum())
    true_neg = int((~flagged & (label == 0)).sum())
    caught = true_pos + false_neg
    warned = true_pos + false_pos
    return {
        "threshold": float(threshold),
        "true_pos": true_pos, "false_pos": false_pos,
        "false_neg": false_neg, "true_neg": true_neg,
        "recall": true_pos / caught if caught else None,
        "precision": true_pos / warned if warned else None,
        "accuracy": (true_pos + true_neg) / len(label) if len(label) else None,
    }


def confusions(label, predicted):
    """At 0.5, and at the threshold that flags as many as really cracked.

    0.5 on its own is misleading here: only a quarter of electrodes cracked,
    so a model can score well on accuracy by warning about almost nothing.
    The prevalence-matched threshold asks the fairer question - if it had to
    name the same NUMBER of suspects as there are cracked electrodes, how
    many would it get right?"""
    done = ~np.isnan(predicted)
    label, predicted = label[done], predicted[done]
    share = float((label == 1).mean())
    matched = float(np.quantile(predicted, 1 - share)) if len(predicted) \
        else 0.5
    return {"half": confusion(label, predicted, 0.5),
            "prevalence": confusion(label, predicted, matched)}


def evaluate(frame, columns, model_kind):
    """Everything a tab needs, for all features and for process-only."""
    results = {}
    for label_text, drop_sem in (("all features", False),
                                 ("process only (no SEM)", True)):
        features, label, groups, names = prepare(frame, columns, drop_sem)
        if not names:
            continue
        auc, predicted, ranked = cross_validate(model_kind, features, label,
                                                groups, names)
        null = null_distribution(model_kind, features, label, groups)
        beaten = int((null >= auc).sum()) if auc is not None else 0
        results[label_text] = {
            "auc": auc,
            "features": len(names),
            "rows": len(label),
            "cracked": int(label.sum()),
            "days": len(np.unique(groups)),
            "null_mean": float(null.mean()) if len(null) else None,
            "null_high": float(np.percentile(null, 95)) if len(null) else None,
            "null": null,
            # +1 both sides: a permutation test cannot honestly report zero
            "p": (beaten + 1) / (len(null) + 1) if len(null) else None,
            "importances": ranked,
            "beaten": beaten,
            "rounds": len(null),
            "confusion": (confusions(label, predicted)
                          if predicted is not None else None),
        }
    return results


# --------------------------------------------------------------- the tab
# Reads model_results.json, written by fit_models.py. Nothing here fits a
# model: a forest with its shuffled-label null is minutes of work, and a tab
# that did that on open would look like the app had frozen.
import json                                                    # noqa: E402
from datetime import datetime                                  # noqa: E402
from pathlib import Path                                       # noqa: E402

from PyQt6.QtCore import Qt                                    # noqa: E402
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,     # noqa: E402
                             QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView,
                             QScrollArea, QFrame, QSizePolicy)
from matplotlib.backends.backend_qtagg import (                # noqa: E402
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure                           # noqa: E402

from gui_app import (AXES_BG, COLOR_CRACKED, COLOR_GOOD, FIG_BG,  # noqa: E402
                     GRID, LABEL, SPINE, SUBTEXT, TEXT)

TITLES = {"forest": "Random forest", "network": "Neural network (FCNN)"}
RESULTS_FILE = Path(__file__).resolve().parent / "model_results.json"


def load_results():
    """Everything fit_models.py saved, or None when it has not been run."""
    if not RESULTS_FILE.exists():
        return None
    try:
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError) as problem:            # pragma: no cover
        print(f"⚠ {RESULTS_FILE.name} could not be read: {problem}")
        return None


class ModelTab(QWidget):
    """One model on one dataset, loaded from disk and explained.

    `rows` and `columns` are the shape of the table as it stands NOW, so the
    tab can say when the saved fit is older than the data it describes."""

    def __init__(self, where, model_kind, heading, rows=None, columns=None):
        super().__init__()
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
        layout.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        title = QLabel(heading)
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        title.setSizePolicy(QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Minimum)
        layout.addWidget(title)

        saved = load_results()
        block = (saved or {}).get("datasets", {}).get(where, {})
        model = block.get("models", {}).get(model_kind)
        if model is None:
            note = QLabel(
                f"No saved fit for this tab. Run it once with:\n\n"
                f"    .venv/Scripts/python.exe fit_models.py\n\n"
                f"It takes about eight minutes and writes "
                f"{RESULTS_FILE.name}, which every model tab then reads "
                f"instantly.")
            note.setStyleSheet(f"color: {SUBTEXT}; font-size: 12.5px;")
            note.setWordWrap(True)
            layout.addWidget(note)
            layout.addStretch(1)
            return

        results = model["results"]
        layout.addWidget(self._provenance(saved, block, model, rows, columns))
        layout.addWidget(self._summary(results))
        layout.addWidget(self._table(results))
        canvas = self._figure(results)
        canvas.setMinimumHeight(210 * len(results) + 250)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
        layout.addWidget(canvas)
        layout.addStretch(1)

    @staticmethod
    def _label(text, colour=None):
        widget = QLabel(text)
        widget.setStyleSheet(f"color: {colour or SUBTEXT}; font-size: 12px;")
        widget.setWordWrap(True)
        widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                             QSizePolicy.Policy.Minimum)
        return widget

    def _provenance(self, saved, block, model, rows, columns):
        when = saved.get("fitted_at", "?")
        try:
            when = datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
        text = (f"Loaded from {RESULTS_FILE.name}, fitted {when} in "
                f"{model.get('seconds', 0):.0f}s on "
                f"{block.get('rows', '?')} electrodes and "
                f"{block.get('columns', '?')} columns.")
        stale = (rows is not None and block.get("rows") not in (None, rows))
        if stale:
            text += (f"  The table now has {rows} electrodes, so this fit is "
                     f"out of date - re-run fit_models.py.")
        return self._label(text, COLOR_CRACKED if stale else SUBTEXT)

    def _summary(self, results):
        """Plain words: does this beat shuffling the answers, and by how
        much does the SEM shortcut flatter it."""
        whole = results.get("all features")
        process = results.get("process only (no SEM)")
        if not whole or whole.get("auc") is None:
            return self._label("This model could not be scored.")
        lines = [
            f"{whole['rows']} electrodes, {whole['cracked']} cracked, over "
            f"{whole['days']} production days. Folds are cut by day, so "
            f"electrodes that share a bath reading never sit on both sides "
            f"of a split.",
            f"Using every column it scores {whole['auc']:.3f} where "
            f"shuffled verdicts score {whole['null_mean']:.3f}, and "
            f"{whole['beaten']} of {whole['rounds']} shuffles reached it - "
            f"so the table really does know something about cracking.",
        ]
        if process and process.get("auc") is not None:
            lines.append(
                f"Without the SEM columns, which are measured on the very "
                f"scan that decided the verdict, it scores "
                f"{process['auc']:.3f}. That lower number is the one that "
                f"matters for changing anything on the line.")
        matched = whole.get("confusion", {}).get("prevalence")
        if matched:
            caught = matched["true_pos"] + matched["false_neg"]
            lines.append(
                f"Asked to name as many suspects as there are cracked "
                f"electrodes, it gets {matched['true_pos']} of {caught} "
                f"right and raises {matched['false_pos']} false alarms.")
        return self._label("  ".join(lines))

    def _table(self, results):
        rows = list(results.items())
        headers = ["features used", "columns", "AUC", "null mean",
                   "null 95th", "shuffles reaching it", "electrodes",
                   "cracked", "days"]
        widget = QTableWidget()
        widget.setColumnCount(len(headers))
        widget.setHorizontalHeaderLabels(headers)
        widget.setRowCount(len(rows))
        widget.setAlternatingRowColors(True)
        widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.verticalHeader().setVisible(False)
        widget.setMaximumHeight(112)
        widget.setStyleSheet(
            f"QTableWidget {{ background: {AXES_BG}; gridline-color: {GRID}; "
            f"color: {TEXT}; font-size: 12px; }}"
            f"QHeaderView::section {{ background: {FIG_BG}; color: {LABEL}; "
            f"font-weight: bold; font-size: 11px; padding: 4px; "
            f"border: none; border-bottom: 1px solid {SPINE}; }}")
        for row, (name, r) in enumerate(rows):
            cells = [name, r["features"],
                     f"{r['auc']:.3f}" if r.get("auc") is not None else "",
                     f"{r['null_mean']:.3f}" if r.get("null_mean") else "",
                     f"{r['null_high']:.3f}" if r.get("null_high") else "",
                     f"{r.get('beaten')} of {r.get('rounds')}",
                     r["rows"], r["cracked"], r["days"]]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(str(value))
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                widget.setItem(row, column, item)
        header = widget.horizontalHeader()
        for column in range(len(headers)):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch if column == 0
                else QHeaderView.ResizeMode.ResizeToContents)
        return widget

    def _figure(self, results):
        figure = Figure(figsize=(13, 4.4 * len(results)), dpi=100)
        figure.patch.set_facecolor(FIG_BG)
        canvas = FigureCanvas(figure)
        grid = figure.add_gridspec(len(results), 3,
                                   width_ratios=[1.1, 2.0, 1.2],
                                   hspace=0.6, wspace=0.3)
        for row, (name, r) in enumerate(results.items()):
            self._draw_null(figure.add_subplot(grid[row, 0]), r, name)
            self._draw_importance(figure.add_subplot(grid[row, 1]), r, name)
            self._draw_confusion(figure.add_subplot(grid[row, 2]), r, name)
        figure.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10)
        canvas.draw_idle()
        return canvas

    @staticmethod
    def _chrome(ax, title):
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold",
                     loc="left", pad=8)
        ax.set_facecolor(AXES_BG)
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)

    def _draw_null(self, ax, r, name):
        null = r.get("null") or []
        if not null or r.get("auc") is None:
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax.hist(null, bins=10, color=SUBTEXT, alpha=0.45,
                label="shuffled verdicts")
        ax.axvline(r["auc"], color=COLOR_CRACKED, linewidth=2.2,
                   label=f"real, {r['auc']:.3f}")
        ax.axvline(0.5, color=SPINE, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_xlabel("AUC", color=LABEL, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=SUBTEXT,
                  loc="upper left")
        self._chrome(ax, f"{name}: score against the null")

    def _draw_importance(self, ax, r, name):
        ranked = r.get("importances") or {}
        features = ranked.get("feature") or []
        if not features:
            ax.set_xticks([])
            ax.set_yticks([])
            return
        top = list(zip(features, ranked["importance"],
                       ranked.get("spread", [0] * len(features))))[:12]
        top = top[::-1]
        y = np.arange(len(top))
        values = [v for _f, v, _s in top]
        ax.barh(y, values, xerr=[s for _f, _v, s in top],
                color=[COLOR_CRACKED if v > 0 else COLOR_GOOD
                       for v in values],
                alpha=0.85,
                error_kw={"ecolor": SUBTEXT, "elinewidth": 0.9})
        ax.set_yticks(y, [f for f, _v, _s in top])
        ax.axvline(0, color=SPINE, linewidth=1.0)
        ax.set_xlabel("AUC lost when the column is shuffled", color=LABEL,
                      fontsize=9.5)
        self._chrome(ax, f"{name}: what it leans on")

    def _draw_confusion(self, ax, r, name):
        matched = (r.get("confusion") or {}).get("prevalence")
        if not matched:
            ax.set_xticks([])
            ax.set_yticks([])
            return
        counts = np.array([[matched["true_neg"], matched["false_pos"]],
                           [matched["false_neg"], matched["true_pos"]]],
                          dtype=float)
        # Shaded by share of the ROW, so the two very unequal groups can be
        # compared: 206 sound electrodes against 71 cracked
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Purples", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                value = int(counts[row, column])
                ax.text(column, row,
                        f"{value}\n{100 * shares[row, column]:.0f}%",
                        ha="center", va="center", fontsize=11,
                        color="#FFFFFF" if shares[row, column] > 0.55
                        else TEXT)
        ax.set_xticks([0, 1], ["said sound", "said cracked"])
        ax.set_yticks([0, 1], ["was sound", "was cracked"])
        ax.tick_params(colors=SUBTEXT, labelsize=9, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        recall = matched.get("recall")
        precision = matched.get("precision")
        ax.set_title(f"{name}: at the prevalence threshold "
                     f"({matched['threshold']:.2f})\n"
                     f"catches {100 * recall:.0f}% of cracked, "
                     f"{100 * precision:.0f}% of warnings right"
                     if recall is not None and precision is not None
                     else name,
                     color=TEXT, fontsize=10, fontweight="bold",
                     loc="left", pad=8)
