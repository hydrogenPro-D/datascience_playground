"""Pull the SEM measurements and the plate bath chemistry from the database.

qc_data.vw_sem_measurement is one row per electrode / position / point /
element, so an electrode with three positions and three points appears many
times over. qc.vw_plate_bath_28/29/30 are one row per bath sampling, each
titration measured twice. dbo.MetaData gives each electrode its production
timestamp and dbo.ProcessData which plating vat it actually ran in. Everything
is written raw here; gui_sem.py does the reducing.
"""
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "sem_raw.csv"
BATH_OUTPUT = FOLDER / "plate_bath.csv"
CONTEXT_OUTPUT = FOLDER / "electrode_context.csv"
PLATE_VATS = (28, 29, 30)
sys.path.insert(0, str(FOLDER.parents[0] / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database


def fetch_baths(conn):
    """The three plating baths stacked into one frame with a vat column."""
    frames = []
    for vat in PLATE_VATS:
        df = pd.read_sql(text(f"SELECT * FROM qc.vw_plate_bath_{vat}"), conn)
        df.insert(0, "vat", vat)
        frames.append(df)
    bath = pd.concat(frames, ignore_index=True)
    bath["Date"] = pd.to_datetime(bath["Date"], errors="coerce")
    return bath.sort_values(["vat", "Date"])


def fetch_context(conn, electrodes):
    """Production timestamp and plating vat, one row per electrode.

    No function on Marking in the WHERE clause - wrapping it in LTRIM/RTRIM
    defeats the index. SQL Server ignores trailing spaces in an IN comparison,
    so the bare column matches padded values anyway.
    """
    markings = ", ".join(f"'{e}'" for e in electrodes)
    meta = pd.read_sql(text(f"""
        SELECT LTRIM(RTRIM(Marking)) AS electrode, Date AS produced
        FROM dbo.MetaData WHERE Marking IN ({markings})"""), conn)
    vats = pd.read_sql(text(f"""
        SELECT LTRIM(RTRIM(Marking)) AS electrode, Proces AS plating_vat,
               SUM(CAST(Aktuel_tid AS bigint)) AS seconds
        FROM dbo.ProcessData
        WHERE Proces IN (28, 29, 30) AND Marking IN ({markings})
        GROUP BY LTRIM(RTRIM(Marking)), Proces"""), conn)

    meta["produced"] = pd.to_datetime(meta["produced"], utc=True,
                                      errors="coerce").dt.tz_localize(None)
    # An electrode can touch more than one plating vat; the one it actually
    # plated in is where it spent the time
    busiest = vats.sort_values("seconds").groupby("electrode").tail(1)
    return meta.merge(busiest[["electrode", "plating_vat"]], on="electrode",
                      how="outer").drop_duplicates("electrode")


def main():
    engine = connect_database(load_config())
    query = """SELECT generation, electrode, electrode_type, folder, position,
                      zoom, point, symbol, value, operator, note, measured_on
               FROM qc_data.vw_sem_measurement
               ORDER BY generation, electrode_type, electrode, position, point"""
    with engine.connect() as conn:
        print("Fetching qc_data.vw_sem_measurement ...")
        df = pd.read_sql(text(query), conn)
    df["electrode"] = df["electrode"].astype(str).str.strip()
    df.to_csv(OUTPUT, index=False)
    print(f"✓ {len(df):,} rows, {df['electrode'].nunique()} electrodes "
          f"-> {OUTPUT.name}")
    print()
    print(df.groupby(["generation", "electrode_type"])["electrode"]
          .nunique().to_string())

    with engine.connect() as conn:
        print()
        print("Fetching qc.vw_plate_bath_28/29/30 ...")
        bath = fetch_baths(conn)
    bath.to_csv(BATH_OUTPUT, index=False)
    print(f"✓ {len(bath)} bath log rows -> {BATH_OUTPUT.name}")
    print(bath.groupby("vat")["Date"].agg(["size", "min", "max"]).to_string())

    electrodes = sorted(df["electrode"].dropna().unique())
    with engine.connect() as conn:
        print()
        print("Fetching production date and plating vat ...")
        context = fetch_context(conn, electrodes)
    context.to_csv(CONTEXT_OUTPUT, index=False)
    print(f"✓ {len(context)} electrodes -> {CONTEXT_OUTPUT.name}")
    print(f"  with a production date: "
          f"{int(context['produced'].notna().sum())}")
    print(f"  with a plating vat    : "
          f"{int(context['plating_vat'].notna().sum())}")
    print(context["plating_vat"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
