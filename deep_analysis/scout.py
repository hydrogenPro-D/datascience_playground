"""Shape-check the tables nobody has used, before building a fetch around them."""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
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
    show("MetaData sample", """
        SELECT TOP 8 * FROM dbo.MetaData ORDER BY Id DESC""", conn)

    show("MetaData coverage and ranges", """
        SELECT COUNT(*) AS rows_total,
               COUNT(DISTINCT Marking) AS markings,
               COUNT(DISTINCT Flight_bar_nr) AS flight_bars,
               COUNT(DISTINCT Rcp_nr) AS recipes,
               MIN(Date) AS first_date, MAX(Date) AS last_date,
               SUM(CASE WHEN Alarm_tid > 0 THEN 1 ELSE 0 END) AS with_alarm,
               SUM(CASE WHEN Man_stop_tid > 0 THEN 1 ELSE 0 END) AS with_manstop
        FROM dbo.MetaData""", conn)

    show("MetaData: how many electrodes share a flight bar", """
        SELECT per_bar, COUNT(*) AS bars FROM (
            SELECT Flight_bar_nr, Date, COUNT(DISTINCT Marking) AS per_bar
            FROM dbo.MetaData GROUP BY Flight_bar_nr, Date) t
        GROUP BY per_bar ORDER BY per_bar""", conn)

    show("MetaData: rows per marking (is it one row per electrode?)", """
        SELECT rows_per, COUNT(*) AS markings FROM (
            SELECT Marking, COUNT(*) AS rows_per
            FROM dbo.MetaData GROUP BY Marking) t
        GROUP BY rows_per ORDER BY rows_per""", conn)

    show("SEM view: what the per-point rows look like", """
        SELECT TOP 12 electrode, position, zoom, point, symbol, value,
               operator, note, measured_on
        FROM qc_data.vw_sem_measurement
        ORDER BY electrode DESC, position, point""", conn)

    show("SEM: which elements are measured, and how often", """
        SELECT symbol, COUNT(*) AS readings,
               COUNT(DISTINCT electrode) AS electrodes,
               AVG(value) AS mean_value
        FROM qc_data.vw_sem_measurement
        WHERE symbol IS NOT NULL
        GROUP BY symbol ORDER BY readings DESC""", conn)

    show("SEM: generations and electrode types", """
        SELECT generation, electrode_type, COUNT(DISTINCT electrode) AS n,
               MIN(measured_on) AS first_seen, MAX(measured_on) AS last_seen
        FROM qc_data.vw_sem_measurement
        GROUP BY generation, electrode_type
        ORDER BY n DESC""", conn)

    show("ProcessData: are Ampere_timer / Temperatur populated?", """
        SELECT Proces,
               COUNT(*) AS rows_total,
               SUM(CASE WHEN Ampere_timer <> 0 THEN 1 ELSE 0 END) AS amp_nonzero,
               SUM(CASE WHEN Temperatur <> 0 THEN 1 ELSE 0 END) AS temp_nonzero,
               AVG(CAST(Ampere_timer AS float)) AS amp_mean,
               AVG(CAST(Temperatur AS float)) AS temp_mean
        FROM dbo.ProcessData GROUP BY Proces ORDER BY Proces""", conn)

    show("ProductionData: timestamp span and per-electrode coverage", """
        SELECT MIN(Aktuel_tid) AS first_ts, MAX(Aktuel_tid) AS last_ts,
               COUNT(DISTINCT Unik_nummer) AS electrodes,
               COUNT(DISTINCT Kar) AS vats
        FROM dbo.ProductionData""", conn)

    show("electrode_status: what statuses exist", """
        SELECT status, source, COUNT(*) AS n
        FROM qc_data.electrode_status
        GROUP BY status, source ORDER BY n DESC""", conn)
