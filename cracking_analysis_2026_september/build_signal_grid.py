"""Per-electrode signal statistics from a dense 10-second grid, not raw rows.

ProductionData logs a channel only when its value CHANGES, so a step that ran
an hour at a constant 780 A is stored as two rows: 780, then 0. Averaging those
raw rows treats a value that held for an hour the same as one that held for ten
seconds, and the strike Current "median" comes out at 390 A for essentially
every electrode - the midpoint of on and off, a number the process never saw.

This rebuilds each electrode/vat the way the electrode report does
(report_data_generation_spec.md, Output C):

  1  scale the value      Type 0 and 2 are stored x100, Type 1 is not
  2  union timeline       all three channels' timestamps, forward-filled;
                          the leading gap is 0 for Voltage/Current and the
                          first observed value for Temperature
  3  longest current      the run where |I| > 10 A holds steady within 10 A,
     plateau              extended by one sample. A station that logs no
                          current at all - the dryers - has no plateau, so
                          its window is the whole time the electrode was in
                          there instead.
  4  10-second grid       every channel resampled onto it, ffill then 0

The statistics are then taken over that grid, so every point carries ten
seconds of real time and the average is time-weighted.

Covers the whole B production line: every station it logs and every
electrode, not just the ones under SEM analysis. The raw points go to
parquet because there are about four million of them and this repo syncs
through OneDrive; the medians go to signals_grid.csv, which has the same
columns as signals.csv so the GUI can read either one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from fetch_sem_data import load_config, connect_database, SIGNAL_TYPES

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_measurements_filtered.csv"
OUTPUT_FILE = FOLDER / "signals_grid.csv"
RAW_FILE = FOLDER / "signals_raw_points.parquet"
# The B line. The A pilot line reuses the same Kar numbers for other
# equipment, so the prefix is what separates them. A LIKE on a fixed prefix
# still seeks the index; wrapping the column in a function would not.
LINE_PREFIX = "B"

# Every station the B line still logs. 27, 36, 37 and 39 also appear but
# stopped on 2025-06-24, between them covering 2 to 76 electrodes, all long
# before anything under analysis - so they are left out.
#   17          cathodic cleaner        T / I / V
#   28, 29, 30  plating vats            T / I / V
#   31          strike                  T / I / V
#   40, 41, 44  drying ovens            temperature only
VATS = (17, 28, 29, 30, 31, 40, 41, 44)
# Stations with no current channel: no plateau to find, so the window is the
# electrode's stay. Checked against the logged drying durations and it
# matches to within a few seconds.
NO_CURRENT = (40, 41, 44)
# ...but "first sample to last sample" is not the stay: a handful of markings
# reappear in a dryer days or months later, and one spanned 238 days, which
# is 2 million grid points and a meaningless median. So the samples are cut
# into sessions wherever they stop for longer than this and the longest one
# wins. Dryer samples are 10 s apart at the median and 164 s at the 99th
# percentile, while a drying cycle is 1800 s, so 600 s separates a pause in
# the run from a return trip without splitting a single cycle: it cuts only
# 138 of 10,807 electrode/dryer pairs into more than one session.
SESSION_GAP = pd.Timedelta(seconds=600)
ZERO_TOL = 10.0        # |I| above this counts as "current on"
PLATEAU_TOL = 10.0     # a jump bigger than this starts a new plateau
GRID = "10s"
# Type -> the lead rule for the gap before a channel's first real sample
LEAD_ZERO = (1, 2)     # Current, Voltage start from nothing
LEAD_FIRST = (0,)      # Temperature was already at its first observed value


def fetch_raw(engine):
    """Every B-line row, one station at a time.

    No function on Unik_nummer in the WHERE clause - wrapping it in
    LTRIM/RTRIM defeats the index and the query never returns. Pulled per
    station rather than in one go so a four million row transfer reports
    progress and never holds two copies in memory."""
    frames = []
    for vat in VATS:
        query = """SELECT Unik_nummer AS Marking, Kar AS Vat, Type, Value,
                          Aktuel_tid AS Datetime
                   FROM dbo.ProductionData
                   WHERE Type IN (0, 1, 2)
                     AND Kar = :vat
                     AND Unik_nummer LIKE :prefix
                   ORDER BY Unik_nummer, Aktuel_tid"""
        with engine.connect() as conn:
            part = pd.read_sql(text(query), conn,
                               params={"vat": int(vat),
                                       "prefix": f"{LINE_PREFIX}%"})
        part["Marking"] = part["Marking"].astype(str).str.strip()
        part["Datetime"] = pd.to_datetime(part["Datetime"], utc=True,
                                          errors="coerce")
        part["Value"] = pd.to_numeric(part["Value"], errors="coerce")
        # Type 0 temperature and Type 2 voltage are stored x100; Type 1 is not
        scale = part["Type"].isin([0, 2])
        part.loc[scale, "Value"] = part.loc[scale, "Value"] / 100.0
        part = part.dropna(subset=["Datetime", "Value"])
        print(f"  Kar {vat:>2}: {len(part):>9,} rows, "
              f"{part['Marking'].nunique():>6,} electrodes")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f"  total : {len(df):>9,} rows, {df['Marking'].nunique():>6,} "
          f"electrodes")
    return df


def channel_series(part, signal_type):
    series = (part.loc[part["Type"] == signal_type, ["Datetime", "Value"]]
              .dropna(subset=["Datetime"])
              .drop_duplicates(subset=["Datetime"], keep="last")
              .set_index("Datetime")["Value"])
    return series.astype(float) if len(series) else pd.Series(dtype=float)


def build_union_ffill(part):
    """One timeline carrying all three channels, each held at its last value."""
    channels = {t: channel_series(part, t) for t in (0, 1, 2)}
    index = pd.DatetimeIndex([])
    for series in channels.values():
        index = index.union(series.index)
    times = pd.DatetimeIndex(index.unique().sort_values())
    if not len(times):
        return times, {}

    out = {}
    for signal_type, series in channels.items():
        if series.empty:
            out[signal_type] = pd.Series(0.0, index=times)
            continue
        filled = series.reindex(times).ffill()
        before_first = times < series.index.min()
        if before_first.any():
            lead = (float(series.iloc[0]) if signal_type in LEAD_FIRST
                    else 0.0)
            filled.iloc[np.where(before_first)[0]] = lead
        out[signal_type] = filled.astype(float)
    return times, out


def longest_session(times):
    """The longest unbroken run of samples, ignoring return trips."""
    if len(times) < 2:
        return (times[0], times[0]) if len(times) else None
    gaps = times.to_series().diff()
    starts = [0] + list(np.flatnonzero(gaps.to_numpy() > SESSION_GAP))
    ends = [index - 1 for index in starts[1:]] + [len(times) - 1]
    spans = [(times[end] - times[start], start, end)
             for start, end in zip(starts, ends)]
    _span, start, end = max(spans)
    return times[start], times[end]


def longest_current_plateau(times, current):
    """Start and end of the longest steady run with the current on."""
    values = current.astype(float).to_numpy()
    on = np.isfinite(values) & (np.abs(values) > ZERO_TOL)
    if not on.any():
        return None
    step = np.concatenate(([0.0], np.abs(np.diff(values))))
    previous_on = np.concatenate(([False], on[:-1]))
    starts_here = on & (~previous_on | (step > PLATEAU_TOL))
    group = np.cumsum(starts_here.astype(int))
    group = np.where(on, group, 0)
    mask = group > 0
    if not mask.any():
        return None
    starts = np.flatnonzero(
        mask & np.concatenate(([True], group[1:] != group[:-1])))
    ends = np.flatnonzero(
        mask & np.concatenate((group[1:] != group[:-1], [True])))
    longest = int(np.argmax(ends - starts + 1))
    start, end = int(starts[longest]), int(ends[longest])
    end = min(end + 1, len(times) - 1)       # include the next sample
    return times[start], times[end]


def grid_for(part):
    """The dense 10-second sample set for one electrode in one vat."""
    times, series = build_union_ffill(part)
    if not len(times):
        return None
    vat = part["Vat"].iloc[0] if len(part) else None
    if int(vat) in NO_CURRENT or int((part["Type"] == 1).sum()) == 0:
        # No current channel, so no plateau to find: the window is the
        # electrode's longest unbroken stay in the station. For the dryers
        # that lands within a few seconds of the logged drying duration.
        # Keyed on the station as well as on the data, so the four stray
        # current rows in Kar 40 cannot send a dryer down the plateau path.
        found = longest_session(times)
        if found is None:
            return None
        start, end = found
    else:
        found = longest_current_plateau(times, series[1])
        if found is None:
            return None
        start, end = found
    grid = pd.date_range(start, end, freq=GRID)
    if not len(grid):
        return None
    return {t: series[t].reindex(grid).ffill().fillna(0.0).astype(float)
            for t in (0, 1, 2)}


def summarise(raw):
    rows, skipped, empty = [], [], []
    groups = raw.groupby(["Marking", "Vat"], sort=False)
    for (marking, vat), part in groups:
        sampled = grid_for(part)
        if sampled is None:
            skipped.append((marking, vat))
            continue
        for signal_type, values in sampled.items():
            raw_count = int((part["Type"] == signal_type).sum())
            # A channel the logger never wrote for this step was filled by the
            # lead rule, which is 0 for Current and Voltage. That is fine for
            # drawing a trace but it is not a measurement: summarising it would
            # report a real 0 degrees. Emit nothing instead.
            if raw_count == 0:
                empty.append((marking, int(vat),
                              SIGNAL_TYPES[signal_type][0]))
                continue
            rows.append({
                "Marking": marking, "Vat": int(vat), "Type": int(signal_type),
                "readings": int(len(values)),
                "raw_readings": raw_count,
                "value_mean": float(values.mean()),
                "value_min": float(values.min()),
                "value_max": float(values.max()),
                "value_median": float(values.median()),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["signal"] = out["Type"].map({t: name for t, (name, _u)
                                         in SIGNAL_TYPES.items()})
    return out, skipped, empty


def main():
    engine = connect_database(load_config())
    print(f"Fetching raw ProductionData for the {LINE_PREFIX} line...")
    raw = fetch_raw(engine)
    raw.to_parquet(RAW_FILE, index=False)
    size = RAW_FILE.stat().st_size / (1024 * 1024)
    print(f"  raw points -> {RAW_FILE.name} ({size:.0f} MB)")

    print("\nBuilding the 10-second grid per electrode/vat...")
    grid, skipped, empty = summarise(raw)
    grid.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ {len(grid)} electrode/vat/channel rows "
          f"({grid['Marking'].nunique()} electrodes) -> {OUTPUT_FILE.name}")
    if skipped:
        print(f"⚠ {len(skipped)} electrode/vat pairs had no current plateau "
              f"and were skipped: {skipped[:6]}")
    if empty:
        counts = pd.Series([f"{signal} in vat {vat}"
                            for _m, vat, signal in empty]).value_counts()
        print(f"⚠ {len(empty)} electrode/vat channels were never logged at "
              f"all and carry no row (rather than a fabricated zero):")
        print("   " + "; ".join(f"{k} x{v}" for k, v in counts.items()))

    print("\nGrid points per channel (was: raw rows):")
    print(grid.groupby(["signal", "Vat"]).agg(
        n=("readings", "size"), grid_points=("readings", "median"),
        raw_points=("raw_readings", "median")).to_string())

    old = FOLDER / "signals.csv"
    if old.exists():
        before = pd.read_csv(old)
        before["Marking"] = before["Marking"].astype(str).str.strip()
        merged = grid.merge(before, on=["Marking", "Vat", "Type"],
                            how="inner", suffixes=("_grid", "_raw"))
        print("\nWhat changes, median across electrodes:")
        print(merged.groupby(["signal_grid", "Vat"]).agg(
            median_grid=("value_median_grid", "median"),
            median_raw=("value_median_raw", "median"),
            mean_grid=("value_mean_grid", "median"),
            mean_raw=("value_mean_raw", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
