"""Join every source into one per-electrode feature matrix.

Three families, kept separate because they mean different things:

  comp_*    coating composition from the SEM (Ni, S, C, N, Co, O and the S/Ni
            ratio). Measured at the same session that produced the crack note,
            so it is a co-measurement, not a predictor: useful for mechanism,
            useless for process control on its own.
  step_*    what the line did - set/actual/overrun seconds, amp-hours and
            temperature, per process step and pass.
  meta_*    alarm time, manual stop time, flight bar, recipe, position in the
            day's run.

Writes features.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import labels

FOLDER = Path(__file__).resolve().parent
ELEMENTS = ["Ni", "S", "C", "N", "Co", "O"]


def sem_features():
    """One row per electrode: the note-derived label plus mean and spread of
    each element across the SEM points."""
    sem = pd.read_csv(FOLDER / "sem_raw.csv")
    sem["electrode"] = sem["electrode"].astype(str).str.strip()

    context = sem.groupby("electrode").agg(
        generation=("generation", "first"),
        electrode_type=("electrode_type", "first"),
        note=("note", "first"),
        operator=("operator", "first"),
        measured_on=("measured_on", "first"),
        sem_points=("point", "nunique"),
        sem_positions=("position", "nunique")).reset_index()
    context = labels.add_severity(context)

    readings = sem.dropna(subset=["symbol", "value"])
    readings = readings[readings["symbol"].isin(ELEMENTS)]
    wide = readings.pivot_table(index="electrode", columns="symbol",
                                values="value", aggfunc="mean")
    spread = readings.pivot_table(index="electrode", columns="symbol",
                                  values="value", aggfunc="std")
    wide.columns = [f"comp_{c}" for c in wide.columns]
    spread.columns = [f"comp_{c}_sd" for c in spread.columns]
    comp = wide.join(spread).reset_index()

    # The operators write "high ratio" / "ratio low" in their notes, so the
    # Ni-to-S balance is already a quantity they watch; make it explicit.
    if {"comp_Ni", "comp_S"}.issubset(comp.columns):
        comp["comp_Ni_over_S"] = comp["comp_Ni"] / comp["comp_S"].replace(0, np.nan)
        comp["comp_S_over_Ni"] = comp["comp_S"] / comp["comp_Ni"].replace(0, np.nan)
        comp["comp_Ni_plus_S"] = comp["comp_Ni"] + comp["comp_S"]
    return context.merge(comp, on="electrode", how="left")


def correct_cumulative(steps):
    """Aktuel_tid holds a running total when a step repeats, so a row's real
    duration is its value minus the previous occurrence of the same step."""
    steps = steps.sort_values(["electrode", "sequence_id"]).copy()
    steps["pass"] = steps.groupby(["electrode", "step"]).cumcount() + 1
    previous = steps.groupby(["electrode", "step"])["actual_time_raw"].shift()
    cumulative = previous.notna() & (steps["actual_time_raw"] >= previous)
    steps["actual_time"] = steps["actual_time_raw"].where(
        ~cumulative, steps["actual_time_raw"] - previous)
    steps["overrun"] = steps["actual_time"] - steps["set_time"]
    # Ampere_timer is a running total across passes too, so summing the raw
    # column double-counts every repeat: keep each pass its own increment.
    amp_previous = steps.groupby(["electrode", "step"])["amp_hours"].shift()
    amp_cumulative = (amp_previous.notna() & (amp_previous > 0)
                      & (steps["amp_hours"] >= amp_previous))
    steps["amp_hours"] = steps["amp_hours"].where(
        ~amp_cumulative, steps["amp_hours"] - amp_previous)
    steps["temperature"] = steps["temperature_raw"] / 100.0   # stored x100
    return steps


def step_features():
    steps = correct_cumulative(pd.read_csv(FOLDER / "process_steps.csv"))
    steps["electrode"] = steps["electrode"].astype(str).str.strip()
    # Steps 1 and 2 are load/unload bookkeeping with a 9999 s set time
    steps = steps[~steps["step"].isin([1, 2])]

    frames = []
    for value, prefix in (("actual_time", "actual"), ("overrun", "over"),
                          ("amp_hours", "amp"), ("temperature", "temp")):
        wide = steps.pivot_table(
            index="electrode", columns=["step", "pass"], values=value,
            aggfunc="first")
        wide.columns = [f"step_{s}_p{p}_{prefix}" for s, p in wide.columns]
        frames.append(wide)
    per_step = pd.concat(frames, axis=1)

    # Whole-route roll-ups: the totals a cumulative-overrun story would use,
    # plus the charge and heat the electrode actually received.
    plating = steps[steps["step"].isin([28, 29, 30])]
    rollup = steps.groupby("electrode").agg(
        route_steps=("step", "size"),
        route_set=("set_time", "sum"),
        route_actual=("actual_time", "sum"),
        route_overrun=("overrun", "sum"),
        route_overrun_max=("overrun", "max"),
        route_amp_total=("amp_hours", "sum"),
        route_temp_max=("temperature", "max"))
    rollup["route_overrun_pct"] = (100 * rollup["route_overrun"]
                                   / rollup["route_set"].replace(0, np.nan))
    rollup = rollup.join(plating.groupby("electrode").agg(
        plate_amp_total=("amp_hours", "sum"),
        plate_seconds=("actual_time", "sum"),
        plate_temp_mean=("temperature", "mean")))
    # Charge per second of plating: the deposition rate, which sets internal
    # stress far more directly than either term on its own.
    rollup["plate_amp_rate"] = (rollup["plate_amp_total"]
                                / rollup["plate_seconds"].replace(0, np.nan))
    return per_step.join(rollup).reset_index()


def meta_features():
    meta = pd.read_csv(FOLDER / "metadata.csv")
    meta["electrode"] = meta["electrode"].astype(str).str.strip()
    started = pd.to_datetime(meta["started_at"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "electrode": meta["electrode"],
        "meta_started_at": started,
        "meta_date": started.dt.date,
        "meta_hour": started.dt.hour + started.dt.minute / 60,
        "meta_weekday": started.dt.weekday,
        "meta_alarm_s": meta["Alarm_tid"],
        "meta_manstop_s": meta["Man_stop_tid"],
        "meta_start_tid": meta["Start_tid"],
        "meta_vb_total": meta["Vb_total"],      # cumulative electrode counter
        "meta_vb_day": meta["Vb_dag"],          # position within the day's run
        "meta_flight_bar": meta["Flight_bar_nr"],
        "meta_recipe": meta["Rcp_nr"],
    })
    out["meta_alarm_plus_stop"] = out["meta_alarm_s"] + out["meta_manstop_s"]
    return out


def build():
    frame = sem_features()
    frame = frame.merge(step_features(), on="electrode", how="left")
    frame = frame.merge(meta_features(), on="electrode", how="left")

    signals = FOLDER / "signals_raw.csv"
    if signals.exists():
        frame = frame.merge(signal_features(signals), on="electrode",
                            how="left")
    else:
        print("! signals_raw.csv not present - skipping vat signal features")
    return frame


SIGNAL_NAMES = {0: "temp", 1: "current", 2: "volt"}


def signal_features(path):
    """Per-vat logged temperature/current/voltage, plus the air gap between
    consecutive vats reconstructed from the reading timestamps."""
    sig = pd.read_csv(path)
    sig["electrode"] = sig["electrode"].astype(str).str.strip()
    sig["first_reading"] = pd.to_datetime(sig["first_reading"], utc=True,
                                          errors="coerce")
    sig["last_reading"] = pd.to_datetime(sig["last_reading"], utc=True,
                                         errors="coerce")

    sig["name"] = sig["Type"].map(SIGNAL_NAMES)
    values = sig.dropna(subset=["name"]).pivot_table(
        index="electrode", columns=["vat", "name"],
        values=["value_mean", "value_sd"], aggfunc="first")
    values.columns = [f"sig_v{vat}_{name}_{stat.replace('value_', '')}"
                      for stat, vat, name in values.columns]

    # One interval per vat, then the gaps between them: time the electrode was
    # out of every bath, which is what a crane delay actually costs.
    span = sig.groupby(["electrode", "vat"]).agg(
        start=("first_reading", "min"), end=("last_reading", "max")).reset_index()
    span = span.sort_values(["electrode", "start"])
    span["prev_end"] = span.groupby("electrode")["end"].shift()
    span["gap_s"] = (span["start"] - span["prev_end"]).dt.total_seconds()
    gaps = span.groupby("electrode").agg(
        air_gap_total=("gap_s", "sum"),
        air_gap_max=("gap_s", "max"),
        air_gap_mean=("gap_s", "mean"),
        vats_logged=("vat", "nunique"),
        route_seconds=("start", lambda s: np.nan))
    gaps = gaps.drop(columns=["route_seconds"])
    total = span.groupby("electrode").agg(first=("start", "min"),
                                          last=("end", "max"))
    gaps["route_wall_s"] = (total["last"] - total["first"]).dt.total_seconds()
    return values.join(gaps).reset_index()


if __name__ == "__main__":
    frame = build()
    frame.to_csv(FOLDER / "features.csv", index=False)
    print(f"features.csv : {len(frame)} electrodes x {len(frame.columns)} columns")
    print("\nseverity by generation / type:")
    print(pd.crosstab([frame["generation"], frame["electrode_type"]],
                      frame["severity_name"], dropna=False).to_string())
