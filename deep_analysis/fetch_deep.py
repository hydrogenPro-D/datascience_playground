"""Fresh, wide pull: every per-electrode signal this database holds.

Earlier work used four hand-picked slices of ProcessData. This pulls the
columns nobody has touched as well:

  MetaData           Alarm_tid, Man_stop_tid, Flight_bar_nr, Rcp_nr, Vb_dag
  ProcessData        Ampere_timer (plating charge) and Temperatur, per step
  ProductionData     real timestamps, so dwell and the air gap between vats
                     can be reconstructed rather than inferred from set times
  vw_sem_measurement the per-point elemental composition (Ni, S, C, N, Co),
                     which the old pipeline collapsed into a text note

Writes one CSV per source into this folder; features.py joins them.
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parents[0] / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database

# Both generations: Gen4.1 is the line under investigation, Gen3.1 is an
# independent set to replicate any finding against.
GENERATIONS = ("Gen4.1", "Gen3.1")


def run(conn, label, sql, path, **params):
    print(f"\n{label} ...")
    df = pd.read_sql(text(sql), conn, params=params or None)
    df.to_csv(path, index=False)
    print(f"  {len(df):,} rows x {len(df.columns)} cols -> {path.name}")
    return df


def main():
    engine = connect_database(load_config())
    with engine.connect() as conn:
        # --- SEM: one row per electrode/position/point/element -------------
        sem = run(conn, "SEM measurements (both generations)", """
            SELECT generation, electrode, electrode_type, position, zoom,
                   point, symbol, value, operator, note, measured_on
            FROM qc_data.vw_sem_measurement
            WHERE generation IN ('Gen4.1', 'Gen3.1')
            """, FOLDER / "sem_raw.csv")

        electrodes = sorted(sem["electrode"].dropna().astype(str)
                            .str.strip().unique())
        print(f"\n{len(electrodes)} distinct SEM electrodes")
        # Parameterising an IN list of this size is slower than a temp join,
        # but it keeps the pull honest: exactly the electrodes we analyse.
        in_list = ", ".join(f"'{e}'" for e in electrodes)

        run(conn, "MetaData (alarm, manual stop, flight bar, recipe)", f"""
            SELECT LTRIM(RTRIM(Marking)) AS electrode, Date AS started_at,
                   Vb_total, Vb_dag, Start_tid, Man_stop_tid, Alarm_tid,
                   Flight_bar_nr, Rcp_nr, Full_name
            FROM dbo.MetaData
            WHERE LTRIM(RTRIM(Marking)) IN ({in_list})
            """, FOLDER / "metadata.csv")

        run(conn, "ProcessData (all steps, with amp-hours and temperature)", f"""
            SELECT Id AS sequence_id, LTRIM(RTRIM(Marking)) AS electrode,
                   Proces AS step, Rcp_tid AS set_time,
                   Aktuel_tid AS actual_time_raw,
                   Ampere_timer AS amp_hours, Temperatur AS temperature_raw
            FROM dbo.ProcessData
            WHERE LTRIM(RTRIM(Marking)) IN ({in_list})
            ORDER BY LTRIM(RTRIM(Marking)), Id
            """, FOLDER / "process_steps.csv")

        # --- ProductionData: aggregate in SQL, 4.6M rows is too much raw ---
        run(conn, "ProductionData signals per electrode/vat/type", f"""
            SELECT LTRIM(RTRIM(Unik_nummer)) AS electrode, Kar AS vat, Type,
                   COUNT(*) AS readings,
                   MIN(Aktuel_tid) AS first_reading,
                   MAX(Aktuel_tid) AS last_reading,
                   AVG(CAST(Value AS float))
                       / CASE WHEN Type IN (0, 2) THEN 100.0 ELSE 1.0 END
                       AS value_mean,
                   MIN(CAST(Value AS float))
                       / CASE WHEN Type IN (0, 2) THEN 100.0 ELSE 1.0 END
                       AS value_min,
                   MAX(CAST(Value AS float))
                       / CASE WHEN Type IN (0, 2) THEN 100.0 ELSE 1.0 END
                       AS value_max,
                   STDEV(CAST(Value AS float))
                       / CASE WHEN Type IN (0, 2) THEN 100.0 ELSE 1.0 END
                       AS value_sd
            FROM dbo.ProductionData
            WHERE LTRIM(RTRIM(Unik_nummer)) IN ({in_list})
            GROUP BY LTRIM(RTRIM(Unik_nummer)), Kar, Type
            """, FOLDER / "signals_raw.csv")

        run(conn, "electrode status (released / scrapped)", f"""
            SELECT electrode, status, source, updated_at_utc
            FROM qc_data.electrode_status
            WHERE electrode IN ({in_list})
            """, FOLDER / "electrode_status.csv")

        # --- bath chemistry, keyed by date not electrode ------------------
        for name, view in (("strike_bath", "qc.vw_strike_bath"),
                           ("activator_bath", "qc.vw_activator_bath"),
                           ("plate_bath_28", "qc.vw_plate_bath_28"),
                           ("plate_bath_29", "qc.vw_plate_bath_29"),
                           ("plate_bath_30", "qc.vw_plate_bath_30"),
                           ("conductivity", "qc.vw_conductivity"),
                           ("contamination", "qc.vw_contamination"),
                           ("water_quality", "qc.vw_water_quality"),
                           ("adding_chemicals", "qc.vw_adding_chemicals"),
                           ("production_tracking", "qc.vw_production_tracking")):
            try:
                run(conn, f"{name}", f"SELECT * FROM {view}",
                    FOLDER / f"{name}.csv")
            except Exception as exc:
                print(f"  ! {name}: {str(exc)[:120]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
