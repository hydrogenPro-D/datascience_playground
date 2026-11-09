"""Follow-ups: is there a real batch grouping, and how do SEM notes map to a verdict."""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 90)
engine = connect_database(load_config())


def show(title, sql, conn):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    try:
        print(pd.read_sql(text(sql), conn).to_string(index=False))
    except Exception as exc:
        print("!", str(exc)[:300])


with engine.connect() as conn:
    show("MetaData sample (newest)", """
        SELECT TOP 6 Id, Marking, Date, Vb_total, Vb_dag, Start_tid,
               Man_stop_tid, Alarm_tid, Flight_bar_nr, Rcp_nr, Full_name
        FROM dbo.MetaData ORDER BY Id DESC""", conn)

    show("MetaData coverage", """
        SELECT COUNT(*) AS rows_total, COUNT(DISTINCT Marking) AS markings,
               COUNT(DISTINCT Flight_bar_nr) AS flight_bars,
               COUNT(DISTINCT Rcp_nr) AS recipes,
               MIN(Date) AS first_date, MAX(Date) AS last_date,
               SUM(CASE WHEN Alarm_tid > 0 THEN 1 ELSE 0 END) AS with_alarm,
               SUM(CASE WHEN Man_stop_tid > 0 THEN 1 ELSE 0 END) AS with_manstop
        FROM dbo.MetaData""", conn)

    show("Electrodes per flight bar per CALENDAR DAY (the real batch?)", """
        SELECT per_bar, COUNT(*) AS bar_days FROM (
            SELECT Flight_bar_nr, CAST(Date AS date) AS d,
                   COUNT(*) AS per_bar
            FROM dbo.MetaData GROUP BY Flight_bar_nr, CAST(Date AS date)) t
        GROUP BY per_bar ORDER BY per_bar""", conn)

    show("Recipe numbers in use since July 2026", """
        SELECT Rcp_nr, COUNT(*) AS n, MIN(CAST(Date AS date)) AS first_day,
               MAX(CAST(Date AS date)) AS last_day
        FROM dbo.MetaData WHERE Date >= '2026-07-01'
        GROUP BY Rcp_nr ORDER BY n DESC""", conn)

    show("SEM notes for Gen4.1 cathodes (the label source)", """
        SELECT note, COUNT(DISTINCT electrode) AS electrodes
        FROM qc_data.vw_sem_measurement
        WHERE generation = 'Gen4.1' AND electrode_type = 'CATHODE'
        GROUP BY note ORDER BY electrodes DESC""", conn)

    show("Gen4.1 cathode elemental coverage per electrode", """
        SELECT symbol, COUNT(DISTINCT electrode) AS electrodes,
               COUNT(*) AS readings, MIN(value) AS lo, AVG(value) AS mean,
               MAX(value) AS hi
        FROM qc_data.vw_sem_measurement
        WHERE generation = 'Gen4.1' AND electrode_type = 'CATHODE'
          AND symbol IS NOT NULL
        GROUP BY symbol ORDER BY electrodes DESC""", conn)

    show("SEM positions and zooms used on Gen4.1 cathodes", """
        SELECT position, zoom, COUNT(DISTINCT electrode) AS electrodes,
               COUNT(*) AS readings
        FROM qc_data.vw_sem_measurement
        WHERE generation = 'Gen4.1' AND electrode_type = 'CATHODE'
        GROUP BY position, zoom ORDER BY electrodes DESC""", conn)

    show("Do SEM electrodes appear in MetaData / ProcessData?", """
        SELECT
          (SELECT COUNT(DISTINCT electrode) FROM qc_data.vw_sem_measurement
             WHERE generation='Gen4.1' AND electrode_type='CATHODE') AS sem_gen41,
          (SELECT COUNT(DISTINCT s.electrode) FROM qc_data.vw_sem_measurement s
             JOIN dbo.MetaData m ON LTRIM(RTRIM(m.Marking)) = s.electrode
             WHERE s.generation='Gen4.1' AND s.electrode_type='CATHODE')
             AS in_metadata,
          (SELECT COUNT(DISTINCT s.electrode) FROM qc_data.vw_sem_measurement s
             JOIN dbo.ProcessData p ON LTRIM(RTRIM(p.Marking)) = s.electrode
             WHERE s.generation='Gen4.1' AND s.electrode_type='CATHODE')
             AS in_processdata,
          (SELECT COUNT(DISTINCT s.electrode) FROM qc_data.vw_sem_measurement s
             JOIN qc_data.electrode_status e ON e.electrode = s.electrode
             WHERE s.generation='Gen4.1' AND s.electrode_type='CATHODE')
             AS in_status""", conn)
