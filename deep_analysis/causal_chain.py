"""Is the sulfur finding real chemistry, or an artefact of measuring a cracked
surface?

The worry: EDS on a cracked coating can see through to the substrate, and if
the substrate is nickel that alone would raise apparent Ni and lower apparent
S - producing exactly the correlation we found, with no causal content.

The test: sulfur in the coating comes from thiourea (TU, CH4N2S) dosed into
the plating bath, and the bath logs record both the dose (g/electrode) and a
measured ThioUrea titration per vat per date. If what was put in the bath
predicts what ends up in the coating, the SEM is reading real deposition
chemistry. CoSO4 dosing versus cobalt in the coating is the positive control:
cobalt has nothing to do with cracking, so if dose->coating works for Co, the
measurement chain is sound.

Chain under test:   bath thiourea -> coating S -> crack severity
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FOLDER = Path(__file__).resolve().parent
pd.set_option("display.width", 220)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def corr(frame, a, b, label=None):
    """Spearman with n, printed as one line."""
    name = label or f"{a} vs {b}"
    if a not in frame.columns or b not in frame.columns:
        print(f"  {name:52s} -- column missing")
        return None
    part = frame[[a, b]].dropna()
    if len(part) < 15 or part[a].nunique() < 3 or part[b].nunique() < 3:
        print(f"  {name:52s} n={len(part):4d}  too few / no spread")
        return None
    rho, p = stats.spearmanr(part[a], part[b])
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:52s} n={len(part):4d}  rho={rho:+.3f}  "
          f"p={p:<10.3g}{stars}")
    return rho, p, len(part)


def plating_vat(steps):
    """Which of vats 28/29/30 an electrode was actually plated in."""
    plate = steps[steps["step"].isin([28, 29, 30])].copy()
    plate = plate[plate["actual_time_raw"] > 0]
    busiest = plate.sort_values("actual_time_raw").groupby("electrode").tail(1)
    return busiest.set_index("electrode")["step"].rename("plate_vat")


def asof_join(left, right, on_left, on_right, value_columns, by=None):
    """Most recent bath measurement on or before each electrode's run."""
    left = left.dropna(subset=[on_left]).sort_values(on_left)
    right = right.dropna(subset=[on_right]).sort_values(on_right)
    merged = pd.merge_asof(left, right[[on_right, *value_columns] +
                                       ([by] if by else [])],
                           left_on=on_left, right_on=on_right,
                           by=by, direction="backward",
                           tolerance=pd.Timedelta(days=14))
    return merged


def main():
    frame = pd.read_csv(FOLDER / "features.csv")
    frame = frame[~frame["known_cause"].fillna(False)]
    steps = pd.read_csv(FOLDER / "process_steps.csv")
    steps["electrode"] = steps["electrode"].astype(str).str.strip()
    frame = frame.merge(plating_vat(steps), left_on="electrode",
                        right_index=True, how="left")
    frame["plate_vat"] = frame["plate_vat"].astype(float)
    frame = frame.copy()          # de-fragment after the wide joins
    frame["run_date"] = pd.to_datetime(frame["meta_started_at"], utc=True,
                                       errors="coerce").dt.tz_localize(None)

    # --- dosing, by date ------------------------------------------------
    dose = pd.read_csv(FOLDER / "production_tracking.csv")
    dose["Date"] = pd.to_datetime(dose["Date"], errors="coerce")
    dose = dose.groupby("Date", as_index=False).agg(
        tu_dose=("TU (g/electrode)", "mean"),
        ba_dose=("BA (g/electrode)", "mean"),
        co_dose=("CoSO4 (g/electrode)", "mean"))
    frame = asof_join(frame, dose, "run_date", "Date",
                      ["tu_dose", "ba_dose", "co_dose"])

    # --- measured bath chemistry, by vat and date ------------------------
    baths = []
    for vat in (28, 29, 30):
        bath = pd.read_csv(FOLDER / f"plate_bath_{vat}.csv")
        bath["Date"] = pd.to_datetime(bath["Date"], errors="coerce")
        bath["plate_vat"] = float(vat)
        keep = pd.DataFrame({
            "Date": bath["Date"], "plate_vat": float(vat),
            "bath_tu": bath[["ThioUrea Titration 1st (ml)",
                             "ThioUrea Titration 2nd (ml)"]].mean(axis=1),
            "bath_ni": bath[["Ni2+ Titration 1st (ml)",
                             "Ni2+ Titration 2nd (ml)"]].mean(axis=1),
            "bath_cl": bath[["Cl- Titration 1st (ml)",
                             "Cl- Titration 2nd (ml)"]].mean(axis=1),
            "bath_boric": bath[["Boric Acid Titration 1st (ml)",
                                "Boric Acid Titration 2nd (ml)"]].mean(axis=1),
            "bath_ph": bath[["pH 1st", "pH 2nd"]].mean(axis=1)})
        baths.append(keep)
    bath = pd.concat(baths, ignore_index=True)
    frame = asof_join(frame, bath, "run_date", "Date",
                      ["bath_tu", "bath_ni", "bath_cl", "bath_boric", "bath_ph"],
                      by="plate_vat")

    frame.to_csv(FOLDER / "features_with_bath.csv", index=False)
    cath = frame[(frame["electrode_type"] == "CATHODE")
                 & frame["severity"].notna()]
    g41 = cath[cath["generation"] == "Gen4.1"]
    g31 = cath[cath["generation"] == "Gen3.1"]

    rule("COVERAGE")
    for name, part in (("Gen4.1 cathodes", g41), ("Gen3.1 cathodes", g31)):
        print(f"  {name}: n={len(part)}, "
              f"bath TU {part['bath_tu'].notna().sum()}, "
              f"TU dose {part['tu_dose'].notna().sum()}, "
              f"Co dose {part['co_dose'].notna().sum()}, "
              f"vat known {part['plate_vat'].notna().sum()}")

    rule("POSITIVE CONTROL - does dosing show up in the coating at all?")
    print(" Cobalt has nothing to do with cracking. If CoSO4 dosing predicts")
    print(" cobalt in the coating, the SEM is reading real deposition chemistry.")
    for name, part in (("Gen4.1", g41), ("Gen3.1", g31), ("both", cath)):
        print(f" [{name}]")
        corr(part, "co_dose", "comp_Co", "CoSO4 dose (g/electrode) -> coating Co")

    rule("LINK 1 - bath thiourea -> sulfur in the coating")
    for name, part in (("Gen4.1", g41), ("Gen3.1", g31), ("both", cath)):
        print(f" [{name}]")
        corr(part, "bath_tu", "comp_S", "bath ThioUrea titration -> coating S")
        corr(part, "tu_dose", "comp_S", "TU dose (g/electrode)   -> coating S")
        corr(part, "bath_tu", "comp_S_over_Ni", "bath ThioUrea -> coating S/Ni")

    rule("LINK 2 - sulfur in the coating -> crack severity")
    for name, part in (("Gen4.1", g41), ("Gen3.1", g31)):
        print(f" [{name}]")
        corr(part, "comp_S_over_Ni", "severity", "coating S/Ni -> severity")
        corr(part, "comp_S", "severity", "coating S    -> severity")
        corr(part, "comp_Ni", "severity", "coating Ni   -> severity")

    rule("LINK 3 - bath thiourea -> crack severity (the actionable one)")
    for name, part in (("Gen4.1", g41), ("Gen3.1", g31)):
        print(f" [{name}]")
        corr(part, "bath_tu", "severity", "bath ThioUrea -> severity")
        corr(part, "tu_dose", "severity", "TU dose       -> severity")
        corr(part, "bath_ph", "severity", "bath pH       -> severity")

    rule("ARTEFACT CHECK - would cracks alone explain the composition shift?")
    print(" If cracks expose substrate, the reading should get MORE variable")
    print(" across points as severity rises, and surface C/O should fall.")
    for name, part in (("Gen4.1", g41), ("Gen3.1", g31)):
        print(f" [{name}]")
        for column, label in (("comp_Ni_sd", "spread of Ni across points"),
                              ("comp_S_sd", "spread of S across points"),
                              ("comp_C", "carbon (surface layer)"),
                              ("comp_O", "oxygen (surface layer)"),
                              ("comp_N", "nitrogen (from thiourea CH4N2S)")):
            corr(part, column, "severity", f"{label} -> severity")

    rule("DOSE-RESPONSE - severity across S/Ni quintiles (Gen4.1 cathodes)")
    part = g41[["comp_S_over_Ni", "severity"]].dropna()
    part["bin"] = pd.qcut(part["comp_S_over_Ni"], 5,
                          labels=["lowest S/Ni", "2", "3", "4", "highest S/Ni"])
    table = part.groupby("bin", observed=True).agg(
        n=("severity", "size"), mean_severity=("severity", "mean"),
        pct_cracked=("severity", lambda s: 100 * (s == 4).mean()),
        pct_sound_or_micro=("severity", lambda s: 100 * (s <= 2).mean()))
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))

    rule("CONTROLLING FOR COATING THICKNESS")
    print(" Gen4.1 plates ~4.4x the charge of Gen3.1, and a thicker deposit")
    print(" carries more internal stress. Does S/Ni still matter within it?")
    part = g41[["comp_S_over_Ni", "severity", "plate_amp_total"]].dropna()
    print(f" n = {len(part)}")
    corr(part, "plate_amp_total", "severity", "plating charge -> severity")
    corr(part, "plate_amp_total", "comp_S_over_Ni", "plating charge -> S/Ni")
    resid_s = part["comp_S_over_Ni"] - np.poly1d(np.polyfit(
        part["plate_amp_total"], part["comp_S_over_Ni"], 1))(
            part["plate_amp_total"])
    resid_y = part["severity"] - np.poly1d(np.polyfit(
        part["plate_amp_total"], part["severity"], 1))(part["plate_amp_total"])
    rho, p = stats.spearmanr(resid_s, resid_y)
    print(f"  {'S/Ni -> severity, charge partialled out':52s} "
          f"n={len(part):4d}  rho={rho:+.3f}  p={p:.4g}")


if __name__ == "__main__":
    main()
