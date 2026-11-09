"""Fetch qc_data.vw_sem_measurement from the DB (dbReadOnly user) into sem_measurements.csv."""
from pathlib import Path
import json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool
import pandas as pd

# Resolve config relative to the repo root so the script works
# regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "config" / "config.json"

OUTPUT_FILE = Path(__file__).resolve().parent / "sem_measurements.csv"
VATS_FILE = Path(__file__).resolve().parent / "strike_vats.csv"
STRIKE_STEP_FILE = Path(__file__).resolve().parent / "strike_step.csv"
RINSE_STEPS_FILE = Path(__file__).resolve().parent / "rinse_steps.csv"
RINSE_STEPS = [14, 13, 12, 19, 20, 25, 24, 23, 22, 21]   # process order
DRIP_STEPS_FILE = Path(__file__).resolve().parent / "drip_steps.csv"
DRIP_STEPS = [16, 26, 27, 15]   # execution order
SIGNALS_FILE = Path(__file__).resolve().parent / "signals.csv"
# ProductionData Type -> what it measures. Temperature and voltage are
# stored x100; current is not scaled.
SIGNAL_TYPES = {0: ("Temperature", "°C"), 1: ("Current", "A"),
                2: ("Voltage", "V")}
METADATA_FILE = Path(__file__).resolve().parent / "electrode_metadata.csv"
STRIKE_BATH_FILE = Path(__file__).resolve().parent / "strike_bath.csv"
SEM_VIEW = "qc_data.vw_sem_measurement"
STRIKE_BATH_VIEW = "qc.vw_strike_bath"
ACTIVATOR_BATH_VIEW = "qc.vw_activator_bath"
ACTIVATOR_BATH_FILE = Path(__file__).resolve().parent / "activator_bath.csv"
CHEMICALS_VIEW = "qc.vw_adding_chemicals"
# The line's own pH log: one row per check, several a day, wide by vat, with
# whatever acid or base was dosed in response. Its timestamps are OSLO LOCAL
# while dbo.MetaData is UTC - data_preprocessing.py reconciles the two.
PH_CONTROL_VIEW = "qc.vw_ph_control"
PH_CONTROL_FILE = Path(__file__).resolve().parent / "ph_control.csv"
# Rinse-water conductivity, one row per walk-round and a column per rinse
# position. Local timestamps, like the pH log.
CONDUCTIVITY_VIEW = "qc.vw_conductivity"
CONDUCTIVITY_FILE = Path(__file__).resolve().parent / "conductivity.csv"
# The cathodic cleaner (vat 17): pH twice a round, plus the Schloetter
# protocol titration for the Slotoclean concentration.
CLEANER_BATH_VIEW = "qc.vw_cleaner_bath"
CLEANER_BATH_FILE = Path(__file__).resolve().parent / "cleaner_bath.csv"
# The feed water itself: conductivity and pH of what fills the rinse tanks.
WATER_VIEW = "qc.vw_water_quality"
WATER_FILE = Path(__file__).resolve().parent / "water_quality.csv"
CHEMICALS_FILE = Path(__file__).resolve().parent / "adding_chemicals.csv"
PLATE_VATS = [28, 29, 30]


def plate_bath_file(vat):
    return Path(__file__).resolve().parent / f"plate_bath_{vat}.csv"


def load_config():
    print(f"Loading configuration from {CONFIG_FILE}...")
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        json_config = json.load(f)

    print("✓ Configuration loaded successfully")
    return {
        "driver": json_config.get("DB_DRIVER"),
        "server": json_config.get("DB_SERVER"),
        "database": json_config.get("DB_DATABASE"),
        "username": json_config.get("DB_USERNAME"),
        "password": json_config.get("DB_PASSWORD"),
    }


def connect_database(config):
    """Establish the SQL connection."""
    print(f"Connecting to database {config['database']} on server {config['server']} "
          f"as {config['username']}...")
    connection_url = URL.create(
        "mssql+pyodbc",
        query={
            "odbc_connect": (
                f"DRIVER={config['driver']};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']}"
            )
        }
    )
    engine = create_engine(
        connection_url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=None"},
    )
    print("✓ Database connection established")
    return engine


def fetch_sem_measurements(engine):
    query = f"SELECT * FROM {SEM_VIEW}"
    with engine.connect() as conn:
        print(f"\nFetching {SEM_VIEW}...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ SEM measurements fetched: {len(df)} rows, {len(df.columns)} columns")
    return df


def fetch_strike_vats(engine):
    """Vat usage: ProcessData rows where Proces is a strike vat (28/29/30)."""
    query = """SELECT Id, Marking, Proces AS Vat,
               Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
               FROM dbo.ProcessData
               WHERE Proces IN (28, 29, 30)
               ORDER BY Id"""
    with engine.connect() as conn:
        print("\nFetching strike vat rows from ProcessData...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ Vat rows fetched: {len(df)} rows, {df['Marking'].nunique()} markings")
    return df


def fetch_strike_step(engine):
    """ProcessData rows for the strike step (Proces 31).

    Kept in its own file rather than added to strike_vats.csv: that file
    drives the per-electrode vat assignment, and folding step 31 into it
    would append "31" to every electrode's vat string.
    """
    query = """SELECT Id, Marking, Proces AS Vat,
               Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
               FROM dbo.ProcessData
               WHERE Proces = 31
               ORDER BY Id"""
    with engine.connect() as conn:
        print("\nFetching strike step rows from ProcessData...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ Strike step rows fetched: {len(df)} rows, "
              f"{df['Marking'].nunique()} markings")
    return df


def fetch_rinse_steps(engine):
    """ProcessData rows for the rinse steps (Proces 19, 20 and 25)."""
    steps = ", ".join(str(s) for s in RINSE_STEPS)
    query = f"""SELECT Id, Marking, Proces AS Vat,
                Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
                FROM dbo.ProcessData
                WHERE Proces IN ({steps})
                ORDER BY Id"""
    with engine.connect() as conn:
        print("\nFetching rinse step rows from ProcessData...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ Rinse step rows fetched: {len(df)} rows, "
              f"{df['Marking'].nunique()} markings")
    return df


def fetch_drip_steps(engine):
    """ProcessData rows for the drip steps (Proces 15, 16, 26 and 27)."""
    steps = ", ".join(str(s) for s in DRIP_STEPS)
    query = f"""SELECT Id, Marking, Proces AS Vat,
                Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
                FROM dbo.ProcessData
                WHERE Proces IN ({steps})
                ORDER BY Id"""
    with engine.connect() as conn:
        print("\nFetching drip step rows from ProcessData...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ Drip step rows fetched: {len(df)} rows, "
              f"{df['Marking'].nunique()} markings")
    return df


def fetch_signals(engine):
    """Per-electrode temperature, current and voltage in the strike and
    plating vats.

    Type 0 temperature and Type 2 voltage are stored x100 (5000 = 50.00);
    Type 1 current is not scaled. Kar is the vat: 28/29/30 plating, 31
    strike — but only on the B line; on the A pilot line those numbers mean
    something else, hence the Unik_nummer filter. Summarised in SQL: the raw
    log is ~10 s per reading.

    The median comes from PERCENTILE_CONT, which is a window function rather
    than an aggregate, so the whole query is windowed over the same partition
    and de-duplicated instead of grouped.
    """
    query = """WITH scaled AS (
                   SELECT Unik_nummer AS Marking, Kar AS Vat, Type,
                          CAST(Value AS float)
                              / CASE WHEN Type IN (0, 2) THEN 100.0 ELSE 1.0 END
                              AS value
                   FROM dbo.ProductionData
                   WHERE Type IN (0, 1, 2)
                     AND Kar IN (28, 29, 30, 31)
                     AND Unik_nummer LIKE 'B%'
               )
               SELECT DISTINCT Marking, Vat, Type,
                      COUNT(*) OVER (PARTITION BY Marking, Vat, Type)
                          AS readings,
                      AVG(value) OVER (PARTITION BY Marking, Vat, Type)
                          AS value_mean,
                      MIN(value) OVER (PARTITION BY Marking, Vat, Type)
                          AS value_min,
                      MAX(value) OVER (PARTITION BY Marking, Vat, Type)
                          AS value_max,
                      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value)
                          OVER (PARTITION BY Marking, Vat, Type)
                          AS value_median
               FROM scaled
               ORDER BY Marking, Vat, Type"""
    with engine.connect() as conn:
        print("\nFetching temperature/current/voltage from ProductionData...")
        df = pd.read_sql(text(query), conn)
        df["signal"] = df["Type"].map({t: name for t, (name, _u)
                                       in SIGNAL_TYPES.items()})
        print(f"✓ Signal rows fetched: {len(df)} electrode/vat/type rows, "
              f"{df['Marking'].nunique()} markings")
    return df


def fetch_electrode_metadata(engine):
    """Production metadata per electrode (incl. the production timestamp)."""
    query = """SELECT Marking, Rcp_nr AS Recipe, Full_name AS Description,
               CONVERT(VARCHAR(19), Date, 120) AS Production_datetime
               FROM dbo.MetaData"""
    with engine.connect() as conn:
        print("\nFetching electrode metadata from MetaData...")
        df = pd.read_sql(text(query), conn)
        print(f"✓ Metadata fetched: {len(df)} rows")
    return df


def main():
    config = load_config()
    print()
    engine = connect_database(config)

    df = fetch_sem_measurements(engine)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved to {OUTPUT_FILE}")

    vats = fetch_strike_vats(engine)
    vats.to_csv(VATS_FILE, index=False)
    print(f"✓ Saved to {VATS_FILE}")

    strike_step = fetch_strike_step(engine)
    strike_step.to_csv(STRIKE_STEP_FILE, index=False)
    print(f"✓ Saved to {STRIKE_STEP_FILE}")

    rinse = fetch_rinse_steps(engine)
    rinse.to_csv(RINSE_STEPS_FILE, index=False)
    print(f"✓ Saved to {RINSE_STEPS_FILE}")

    drip = fetch_drip_steps(engine)
    drip.to_csv(DRIP_STEPS_FILE, index=False)
    print(f"✓ Saved to {DRIP_STEPS_FILE}")

    signals = fetch_signals(engine)
    signals.to_csv(SIGNALS_FILE, index=False)
    print(f"✓ Saved to {SIGNALS_FILE}")

    meta = fetch_electrode_metadata(engine)
    meta.to_csv(METADATA_FILE, index=False)
    print(f"✓ Saved to {METADATA_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {STRIKE_BATH_VIEW}...")
        bath = pd.read_sql(text(f"SELECT * FROM {STRIKE_BATH_VIEW}"), conn)
        print(f"✓ Strike bath measurements fetched: {len(bath)} rows")
    bath.to_csv(STRIKE_BATH_FILE, index=False)
    print(f"✓ Saved to {STRIKE_BATH_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {ACTIVATOR_BATH_VIEW}...")
        activator = pd.read_sql(
            text(f"SELECT * FROM {ACTIVATOR_BATH_VIEW}"), conn)
        print(f"✓ Activator bath measurements fetched: {len(activator)} rows")
    activator.to_csv(ACTIVATOR_BATH_FILE, index=False)
    print(f"✓ Saved to {ACTIVATOR_BATH_FILE}")

    for vat in PLATE_VATS:
        view = f"qc.vw_plate_bath_{vat}"
        with engine.connect() as conn:
            print(f"\nFetching {view}...")
            plate = pd.read_sql(text(f"SELECT * FROM {view}"), conn)
            print(f"✓ Plate bath {vat} measurements fetched: {len(plate)} rows")
        plate.to_csv(plate_bath_file(vat), index=False)
        print(f"✓ Saved to {plate_bath_file(vat)}")

    # Chemical additions for the strike vat (31) and the three plating vats
    with engine.connect() as conn:
        print(f"\nFetching {CHEMICALS_VIEW}...")
        chems = pd.read_sql(text(f"""
            SELECT [Date], [Initials], [Time when added],
                   [Vat. 17 (Cathodic) (kg)],
                   [Vat. 18 (Activation ) (kg)],
                   [Vat. 28 (Plating 1) (kg)],
                   [Vat. 29 (Plating 2) (kg)],
                   [Vat. 30 (Plating 3) (kg)],
                   [Vat. 31 (Strike) (kg)],
                   [Notes]
            FROM {CHEMICALS_VIEW}
            ORDER BY [Date]"""), conn)
        print(f"✓ Chemical additions fetched: {len(chems)} rows")
    chems.to_csv(CHEMICALS_FILE, index=False)
    print(f"✓ Saved to {CHEMICALS_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {PH_CONTROL_VIEW}...")
        ph = pd.read_sql(text(f"SELECT * FROM {PH_CONTROL_VIEW}"), conn)
        print(f"✓ pH control rows fetched: {len(ph)} rows")
    ph.to_csv(PH_CONTROL_FILE, index=False)
    print(f"✓ Saved to {PH_CONTROL_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {CONDUCTIVITY_VIEW}...")
        cond = pd.read_sql(text(f"SELECT * FROM {CONDUCTIVITY_VIEW}"), conn)
        print(f"✓ Conductivity rounds fetched: {len(cond)} rows")
    cond.to_csv(CONDUCTIVITY_FILE, index=False)
    print(f"✓ Saved to {CONDUCTIVITY_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {CLEANER_BATH_VIEW}...")
        cleaner = pd.read_sql(text(f"SELECT * FROM {CLEANER_BATH_VIEW}"), conn)
        print(f"✓ Cleaner bath rounds fetched: {len(cleaner)} rows")
    cleaner.to_csv(CLEANER_BATH_FILE, index=False)
    print(f"✓ Saved to {CLEANER_BATH_FILE}")

    with engine.connect() as conn:
        print(f"\nFetching {WATER_VIEW}...")
        water = pd.read_sql(text(f"SELECT * FROM {WATER_VIEW}"), conn)
        print(f"✓ Water quality rows fetched: {len(water)} rows")
    water.to_csv(WATER_FILE, index=False)
    print(f"✓ Saved to {WATER_FILE}")


if __name__ == "__main__":
    main()
