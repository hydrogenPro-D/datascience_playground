"""Every process step of every analysed electrode, set time and actual time.

Steps 1 and 2 are excluded. Writes two views of the same data:

  all_step_times_long.csv  one row per step occurrence
  all_step_times_wide.csv  one row per electrode, a column per step/pass

ProcessData stores Actual_time cumulatively when a step repeats (strike 31
runs twice, plating 28/29/30 three times), so a row's real duration is its
value minus the previous occurrence of the same step. The long file keeps
both the raw value and the corrected one; the wide file uses the corrected
one, since that is the actual time in the step.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from fetch_sem_data import load_config, connect_database

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_measurements_filtered.csv"
LONG_FILE = FOLDER / "all_step_times_long.csv"
WIDE_FILE = FOLDER / "all_step_times_wide.csv"
EXCLUDED_STEPS = (1, 2)


def fetch(electrodes, engine):
    markings = ", ".join(f"'{m}'" for m in electrodes)
    excluded = ", ".join(str(s) for s in EXCLUDED_STEPS)
    query = f"""SELECT Id AS sequence_id, Marking AS electrode,
                Proces AS step, Rcp_tid AS set_time,
                Aktuel_tid AS actual_time_raw
                FROM dbo.ProcessData
                WHERE Proces NOT IN ({excluded})
                  AND Marking IN ({markings})
                ORDER BY Marking, Id"""
    with engine.connect() as conn:
        print(f"Fetching all process steps (excluding {excluded})...")
        df = pd.read_sql(text(query), conn)
    df["electrode"] = df["electrode"].astype(str).str.strip()
    print(f"✓ {len(df)} rows for {df['electrode'].nunique()} electrodes")
    return df


def correct_cumulative(df):
    """Undo the running total, per electrode and step."""
    df = df.sort_values(["electrode", "sequence_id"]).copy()
    df["pass"] = df.groupby(["electrode", "step"]).cumcount() + 1
    previous = df.groupby(["electrode", "step"])["actual_time_raw"].shift()
    cumulative = previous.notna() & (df["actual_time_raw"] >= previous)
    df["actual_time"] = df["actual_time_raw"].where(
        ~cumulative, df["actual_time_raw"] - previous)
    df["overrun"] = df["actual_time"] - df["set_time"]
    print(f"  corrected {int(cumulative.sum())} rows that held a running total")
    return df


def main():
    sem = pd.read_csv(SEM_FILE)
    sem["verdict"] = sem["sem_result"].map({"good": "not cracked",
                                            "bad": "cracked"})
    context = sem[["electrode", "verdict", "sem_result", "note_corrected",
                   "production_date", "vats"]].rename(
        columns={"vats": "plating_vat"})

    engine = connect_database(load_config())
    steps = correct_cumulative(fetch(list(sem["electrode"]), engine))

    long = context.merge(steps, on="electrode", how="inner")
    long = long.sort_values(["electrode", "sequence_id"])
    long.to_csv(LONG_FILE, index=False)
    print(f"\n✓ {len(long)} rows -> {LONG_FILE.name}")

    # One column pair per step/pass; most electrodes touch ~22 of them
    wide = long.copy()
    wide["key"] = ("step_" + wide["step"].astype(str)
                   + "_p" + wide["pass"].astype(str))
    pivot = wide.pivot_table(index="electrode", columns="key",
                             values=["set_time", "actual_time"],
                             aggfunc="first")
    pivot.columns = [f"{key}_{value.replace('_time', '')}"
                     for value, key in pivot.columns]
    ordered = sorted(pivot.columns,
                     key=lambda c: (int(c.split("_")[1]),
                                    int(c.split("_p")[1].split("_")[0]),
                                    c.endswith("actual")))
    flat = context.merge(pivot[ordered].reset_index(), on="electrode",
                         how="inner")
    flat.to_csv(WIDE_FILE, index=False)
    print(f"✓ {len(flat)} electrodes x {len(flat.columns)} columns "
          f"-> {WIDE_FILE.name}")

    print("\nsteps covered:", sorted(long['step'].unique()))
    missing = set(sem["electrode"]) - set(long["electrode"])
    if missing:
        print("electrodes with no ProcessData rows:", sorted(missing))


if __name__ == "__main__":
    main()
