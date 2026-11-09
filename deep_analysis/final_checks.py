"""Consolidating checks behind the sulfur story, plus the summary figure.

1  additive per unit of plating charge, Gen3.1 vs Gen4.1 - the quantity the
   recipe change actually moved
2  how well severity can be predicted out-of-fold from composition alone,
   from process alone, and from both, against a permutation null
3  the dose-response figure
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

FOLDER = Path(__file__).resolve().parent
pd.set_option("display.width", 220)

COLOR_GOOD = "#0EA5A5"
COLOR_CRACKED = "#8B5CF6"
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load():
    frame = pd.read_csv(FOLDER / "features_with_bath.csv")
    cath = frame[(frame["electrode_type"] == "CATHODE")
                 & frame["severity"].notna()]
    return cath, cath[cath["generation"] == "Gen4.1"], \
        cath[cath["generation"] == "Gen3.1"]


def additive_per_charge(g41, g31):
    rule("1  THIOUREA PER UNIT OF PLATING CHARGE")
    print(" Dose is recorded per electrode, but what a coating sees is dose")
    print(" relative to how much metal was deposited onto it.")
    rows = []
    for name, part in (("Gen3.1", g31), ("Gen4.1", g41)):
        tu = part["tu_dose"].median()
        amp = part["plate_amp_total"].median()
        rows.append({"generation": name, "n": len(part),
                     "TU g/electrode": tu, "plating Ah": amp,
                     "TU g per Ah": tu / amp if amp else np.nan,
                     "coating S": part["comp_S"].median(),
                     "coating Ni": part["comp_Ni"].median(),
                     "S/Ni": part["comp_S_over_Ni"].median(),
                     "% cracked": 100 * (part["severity"] == 4).mean(),
                     "% any cracking": 100 * (part["severity"] > 0).mean()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    a, b = table.iloc[1], table.iloc[0]
    print(f"\n Gen4.1 plates {a['plating Ah'] / b['plating Ah']:.1f}x the charge "
          f"on {a['TU g/electrode'] / b['TU g/electrode']:.1f}x the thiourea,")
    print(f" so additive per amp-hour fell "
          f"{100 * (1 - a['TU g per Ah'] / b['TU g per Ah']):.0f}%.")


def predictive_ceiling(g41):
    rule("2  WHAT CAN PREDICT SEVERITY OUT-OF-FOLD (Gen4.1 cathodes)")
    usable = g41.copy()
    usable["day"] = pd.to_datetime(usable["meta_started_at"], utc=True,
                                   errors="coerce").dt.date.astype(str)
    usable = usable[usable["day"] != "NaT"]
    y = usable["severity"].to_numpy(dtype=float)
    groups = usable["day"]

    comp = [c for c in usable.columns if c.startswith("comp_")]
    proc = [c for c in usable.columns
            if c.startswith(("step_", "route_", "plate_", "meta_", "bath_"))]

    def usable_only(columns):
        return [c for c in columns
                if pd.api.types.is_numeric_dtype(usable[c])
                and usable[c].notna().sum() > 0.5 * len(usable)
                and usable[c].nunique(dropna=True) >= 3]

    splitter = GroupKFold(n_splits=5)
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                          learning_rate=0.06, random_state=0)
    rng = np.random.default_rng(0)
    for name, columns in (("composition only", usable_only(comp)),
                          ("process only", usable_only(proc)),
                          ("both", usable_only(comp + proc))):
        x = usable[columns].to_numpy(dtype=float)
        pred = cross_val_predict(model, x, y, cv=splitter, groups=groups)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rho = stats.spearmanr(pred, y).statistic
        null = []
        for _i in range(150):
            shuffled = rng.permutation(y)
            guess = cross_val_predict(model, x, shuffled, cv=splitter,
                                      groups=groups)
            null.append(1 - np.sum((shuffled - guess) ** 2)
                        / np.sum((shuffled - shuffled.mean()) ** 2))
        null = np.array(null)
        p = (1 + (null >= r2).sum()) / (1 + len(null))
        print(f" {name:18s} {len(columns):3d} features   R2={r2:+.3f}  "
              f"rho={rho:+.3f}  permutation p={p:.3f}")


def figure(g41, g31):
    part = g41[["comp_S_over_Ni", "severity"]].dropna().copy()
    part["bin"] = pd.qcut(part["comp_S_over_Ni"], 5, labels=False)
    edges = pd.qcut(part["comp_S_over_Ni"], 5).cat.categories
    grouped = part.groupby("bin").agg(
        n=("severity", "size"),
        cracked=("severity", lambda s: 100 * (s == 4).mean()),
        mild=("severity", lambda s: 100 * (s <= 2).mean()))

    fig = Figure(figsize=(12.5, 5.6), dpi=110)
    fig.patch.set_facecolor(FIG_BG)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.24,
                            left=0.07, right=0.98, top=0.72, bottom=0.16)

    ax = fig.add_subplot(grid[0])
    ax.set_facecolor(AXES_BG)
    x = np.arange(len(grouped))
    ax.bar(x - 0.19, grouped["cracked"], width=0.36, color=COLOR_CRACKED,
           edgecolor="none", zorder=3, label="fully cracked")
    ax.bar(x + 0.19, grouped["mild"], width=0.36, color=COLOR_GOOD,
           edgecolor="none", zorder=3, label="sound or microcracks only")
    for i, row in grouped.reset_index().iterrows():
        ax.annotate(f"{row['cracked']:.0f}%", (i - 0.19, row["cracked"]),
                    ha="center", va="bottom", fontsize=9.5, color=SUBTEXT,
                    xytext=(0, 3), textcoords="offset points")
        ax.annotate(f"{row['mild']:.0f}%", (i + 0.19, row["mild"]),
                    ha="center", va="bottom", fontsize=9.5, color=SUBTEXT,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x, [f"{e.left:.2f}-{e.right:.2f}\nn={n}"
                      for e, n in zip(edges, grouped["n"])])
    ax.set_xlabel("coating S/Ni ratio, quintiles (low to high)", color=SUBTEXT,
                  fontsize=10.5, labelpad=6)
    ax.set_ylabel("share of electrodes (%)", color=LABEL, fontsize=11)
    ax.set_ylim(0, 88)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=SUBTEXT, length=0, labelsize=9.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=SUBTEXT,
              loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.13))

    bx = fig.add_subplot(grid[1])
    bx.set_facecolor(AXES_BG)
    names = {0: "sound", 1: "very few\nmicro", 2: "micro\ncracks",
             3: "small\ncracks", 4: "cracked"}
    width = 0.38
    for offset, (gen, part2, color) in enumerate(
            (("Gen3.1", g31, COLOR_GOOD), ("Gen4.1", g41, COLOR_CRACKED))):
        share = (part2["severity"].value_counts(normalize=True)
                 .reindex(range(5), fill_value=0) * 100)
        bx.bar(np.arange(5) + (offset - 0.5) * width, share, width=width,
               color=color, edgecolor="none", zorder=3,
               label=f"{gen} (n={len(part2)})")
    bx.set_xticks(range(5), [names[i] for i in range(5)])
    bx.set_ylabel("share of cathodes (%)", color=LABEL, fontsize=11)
    bx.set_xlabel("crack severity from the SEM note", color=SUBTEXT,
                  fontsize=10.5, labelpad=6)
    bx.yaxis.grid(True, color=GRID, linewidth=0.8)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set_color(SPINE)
    bx.tick_params(colors=SUBTEXT, length=0, labelsize=9.5)
    bx.legend(frameon=False, fontsize=10, labelcolor=SUBTEXT,
              loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.13))

    fig.text(0.07, 0.955, "Coating sulfur tracks cracking, and Gen4.1 lost its "
             "sound population", color=TEXT, fontsize=15.5, fontweight="bold",
             va="top")
    fig.text(0.07, 0.895, "Left: Gen4.1 cathodes binned by coating S/Ni ratio."
             "   Right: severity distribution, Gen3.1 vs Gen4.1 cathodes.",
             color=SUBTEXT, fontsize=10.5, va="top")
    fig.savefig(FOLDER / "sulfur_finding.png", facecolor=FIG_BG)
    print(f"\nwrote {(FOLDER / 'sulfur_finding.png').name}")


def main():
    cath, g41, g31 = load()
    additive_per_charge(g41, g31)
    predictive_ceiling(g41)
    figure(g41, g31)


if __name__ == "__main__":
    main()
