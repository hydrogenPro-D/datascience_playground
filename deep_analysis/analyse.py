"""The analysis proper.

Structured so a null result is as readable as a positive one:

  1  what the label actually is, per generation
  2  Gen3.1 vs Gen4.1 - what changed between a mostly-sound population and a
     universally-cracked one
  3  within Gen4.1 cathodes, what tracks severity (FDR-controlled)
  4  the same test inside Gen3.1 cathodes, as an independent replication
  5  features that survive in both - the only ones worth believing
  6  a model with day-grouped CV and a permutation null, to put a ceiling on
     how much of severity the process data can explain at all
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

FOLDER = Path(__file__).resolve().parent
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

LABEL_COLUMNS = {"severity", "severity_name", "note", "known_cause",
                 "sem_result", "electrode", "generation", "electrode_type",
                 "operator", "measured_on", "meta_started_at", "meta_date"}


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def numeric_features(frame):
    """Model-eligible columns: numeric, not the label, varying, and present
    for at least half the rows."""
    out = []
    for column in frame.columns:
        if column in LABEL_COLUMNS or not pd.api.types.is_numeric_dtype(
                frame[column]):
            continue
        values = frame[column]
        if values.notna().sum() < max(20, 0.5 * len(frame)):
            continue
        if values.nunique(dropna=True) < 3:
            continue
        out.append(column)
    return out


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    keep = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    if keep.sum():
        out[keep] = stats.false_discovery_control(p[keep], method="bh")
    return out


def spearman_sweep(frame, columns, target="severity"):
    rows = []
    for column in columns:
        part = frame[[column, target]].dropna()
        if len(part) < 25 or part[column].nunique() < 3:
            continue
        rho, p = stats.spearmanr(part[column], part[target])
        rows.append({"feature": column, "n": len(part), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = bh_fdr(out["p"])
        out = out.sort_values("p")
    return out


def main():
    frame = pd.read_csv(FOLDER / "features.csv")
    frame = frame[~frame["known_cause"].fillna(False)]      # known-cause batch

    rule("1  WHAT THE LABEL IS")
    print(pd.crosstab([frame["generation"], frame["electrode_type"]],
                      frame["severity_name"]).to_string())
    print("\nShare of scored electrodes showing ANY cracking:")
    scored = frame[frame["severity"].notna()]
    share = scored.groupby(["generation", "electrode_type"]).agg(
        n=("severity", "size"),
        any_cracking=("severity", lambda s: 100 * (s > 0).mean()),
        median_severity=("severity", "median"),
        cracked_pct=("severity", lambda s: 100 * (s == 4).mean()))
    print(share.to_string(float_format=lambda v: f"{v:.1f}"))

    cath = scored[scored["electrode_type"] == "CATHODE"]
    g41 = cath[cath["generation"] == "Gen4.1"]
    g31 = cath[cath["generation"] == "Gen3.1"]

    rule("2  GEN3.1 vs GEN4.1 CATHODES - what changed")
    columns = numeric_features(cath)
    rows = []
    for column in columns:
        a = g41[column].dropna()
        b = g31[column].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial: +1 means Gen4.1 always higher, -1 always lower
        effect = 2 * u / (len(a) * len(b)) - 1
        rows.append({"feature": column, "n41": len(a), "n31": len(b),
                     "med41": a.median(), "med31": b.median(),
                     "effect": effect, "p": p})
    between = pd.DataFrame(rows)
    between["p_fdr"] = bh_fdr(between["p"])
    between = between.reindex(between["effect"].abs().sort_values(
        ascending=False).index)
    print("Top 25 by effect size (rank-biserial; |effect| > 0.5 is a big shift)")
    print(between.head(25).to_string(index=False,
                                     float_format=lambda v: f"{v:.4g}"))

    rule("3  WITHIN GEN4.1 CATHODES - what tracks severity")
    sweep41 = spearman_sweep(g41, numeric_features(g41))
    print(f"{len(sweep41)} features tested, "
          f"{int((sweep41['p_fdr'] < 0.05).sum())} significant after FDR")
    print(sweep41.head(20).to_string(index=False,
                                     float_format=lambda v: f"{v:.4g}"))

    rule("4  WITHIN GEN3.1 CATHODES - independent replication")
    sweep31 = spearman_sweep(g31, numeric_features(g31))
    print(f"{len(sweep31)} features tested, "
          f"{int((sweep31['p_fdr'] < 0.05).sum())} significant after FDR")
    print(sweep31.head(20).to_string(index=False,
                                     float_format=lambda v: f"{v:.4g}"))

    rule("5  FEATURES THAT REPLICATE IN BOTH GENERATIONS")
    both = sweep41.merge(sweep31, on="feature", suffixes=("_41", "_31"))
    agree = both[(both["p_41"] < 0.05) & (both["p_31"] < 0.05)
                 & (np.sign(both["rho_41"]) == np.sign(both["rho_31"]))]
    if len(agree):
        print(agree[["feature", "n_41", "rho_41", "p_41", "n_31", "rho_31",
                     "p_31"]].to_string(index=False,
                                        float_format=lambda v: f"{v:.4g}"))
    else:
        print("none - nothing that tracks severity in Gen4.1 does so in Gen3.1")

    rule("6  HOW MUCH OF SEVERITY IS PREDICTABLE AT ALL (Gen4.1 cathodes)")
    model_cv(g41)


def model_cv(frame, target="severity", rounds=200, seed=0):
    """Day-grouped CV plus a permutation null.

    Grouping by production day matters: electrodes made on one day share a
    bath, a shift and a crane, so a random split lets the model memorise the
    day and score well without learning anything transferable."""
    columns = numeric_features(frame)
    usable = frame[frame[target].notna()].copy()
    groups = pd.to_datetime(usable["meta_started_at"], utc=True,
                            errors="coerce").dt.date.astype(str)
    keep = groups.notna() & (groups != "NaT")
    usable, groups = usable[keep], groups[keep]

    x = usable[columns].to_numpy(dtype=float)
    y = usable[target].to_numpy(dtype=float)
    print(f"{len(usable)} electrodes, {len(columns)} features, "
          f"{groups.nunique()} production days")

    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                          learning_rate=0.06,
                                          random_state=seed)
    predicted = cross_val_predict(model, x, y, cv=splitter, groups=groups)
    r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = stats.spearmanr(predicted, y).statistic
    print(f"out-of-fold R2   = {r2:+.3f}   (0 = no better than the mean)")
    print(f"out-of-fold rho  = {rho:+.3f}")

    rng = np.random.default_rng(seed)
    null = []
    for _i in range(rounds):
        shuffled = rng.permutation(y)
        guess = cross_val_predict(model, x, shuffled, cv=splitter, groups=groups)
        null.append(1 - np.sum((shuffled - guess) ** 2)
                    / np.sum((shuffled - shuffled.mean()) ** 2))
    null = np.array(null)
    p = (1 + (null >= r2).sum()) / (1 + len(null))
    print(f"permutation null R2: median {np.median(null):+.3f}, "
          f"95th pct {np.percentile(null, 95):+.3f}")
    print(f"permutation p = {p:.3f}")
    if p > 0.05:
        print("-> the process data does not predict severity better than chance")


if __name__ == "__main__":
    main()
