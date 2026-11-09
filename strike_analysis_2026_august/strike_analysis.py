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
    print(f"Connecting to database {config['database']} on server {config['server']}...")
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


def fetch_data(engine):
    """Pull MetaData and ProcessData tables into dataframes."""
    query_metadata = """SELECT Marking,
                       Rcp_nr AS Recipe,
                       Full_name AS Description,
                       CONVERT(VARCHAR(19), Date, 120) AS Datetime
                       FROM MetaData"""

    query_processdata = """SELECT Id AS Sequence_nr, Marking, Proces AS Step_nr,
                       Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
                       FROM ProcessData
                       ORDER BY Id"""

    with engine.connect() as conn:
        print("\nFetching metadata...")
        metadata_df = pd.read_sql(text(query_metadata), conn)
        print(f"✓ Metadata fetched: {len(metadata_df)} rows")

        print("\nFetching process data...")
        processdata_df = pd.read_sql(text(query_processdata), conn)
        print(f"✓ Process data fetched: {len(processdata_df)} rows")

    return metadata_df, processdata_df


STRIKE_STEP_NR = 31
MIN_B_NUMBER = 13900


def select_electrodes(metadata_df):
    """Production cathodes: Marking B<number> with number >= MIN_B_NUMBER and plain CATHODE recipe."""
    meta = metadata_df.copy()
    meta["Marking"] = meta["Marking"].astype(str).str.strip()
    meta["B_number"] = meta["Marking"].str.extract(r"^B(\d+)$")[0].astype("Int64")
    is_cathode = meta["Description"].astype(str).str.strip().eq("CATHODE")
    selected = meta[is_cathode & (meta["B_number"] >= MIN_B_NUMBER)]
    return set(selected["Marking"])


def run_strike_analysis(metadata_df, processdata_df):
    """Per electrode: planned (Set_time) vs actual (Actual_time) in the two strike passes (Step_nr 31)."""
    electrodes = select_electrodes(metadata_df)
    print(f"\nSelected electrodes (B >= {MIN_B_NUMBER}, CATHODE recipe): {len(electrodes)}")

    strikes = processdata_df[
        (processdata_df["Step_nr"] == STRIKE_STEP_NR)
        & (processdata_df["Marking"].astype(str).str.strip().isin(electrodes))
    ].copy()
    strikes = strikes.sort_values("Sequence_nr")
    strikes["Strike_pass"] = strikes.groupby("Marking").cumcount() + 1

    # DB bug: Actual_time of strike pass >= 2 is cumulative — it carries the sum of
    # the previous pass (e.g. recorded 643 = 420 from pass 1 + real 223 of pass 2).
    # Undo it by subtracting the previous pass's recorded value per electrode.
    # A recorded value smaller than the previous pass cannot be cumulative — keep it raw.
    prev_recorded = strikes.groupby("Marking")["Actual_time"].shift()
    cumulative = ((strikes["Strike_pass"] >= 2)
                  & (strikes["Actual_time"] > 0) & (prev_recorded > 0)
                  & (strikes["Actual_time"] >= prev_recorded))
    strikes.loc[cumulative, "Actual_time"] = (
        strikes.loc[cumulative, "Actual_time"] - prev_recorded[cumulative])
    not_cumulative = ((strikes["Strike_pass"] >= 2)
                      & (strikes["Actual_time"] > 0) & (prev_recorded > 0)
                      & ~cumulative)
    if not_cumulative.any():
        print(f"\n⚠ {int(not_cumulative.sum())} strike rows were recorded smaller than their "
              f"previous pass (not cumulative, kept as recorded) — markings: "
              f"{sorted(strikes.loc[not_cumulative, 'Marking'].unique())}")

    if strikes.empty:
        print(f"No Step_nr {STRIKE_STEP_NR} rows found for the selected electrodes — nothing to report.")
        return

    counts = strikes.groupby("Marking").size()
    print("\n" + "=" * 50)
    print(f"Strike step (Step_nr = {STRIKE_STEP_NR}) overview")
    print("=" * 50)
    print(f"Electrodes with at least one strike pass: {counts.size}")
    print("Strike passes per electrode:")
    print(counts.value_counts().sort_index().to_string())
    extra = counts[counts > 2]
    if not extra.empty:
        print(f"\nNote: {extra.size} electrodes have more than 2 strike passes; "
              "only the first two are shown below.")

    # Keep the first two passes chronologically, then label them by Set_time
    # (higher set time -> strike 1, i.e. the main 420s strike; the short 60s
    # strike -> strike 2), so an electrode that ran them in reverse order
    # (e.g. B14103) still lands in the right columns.
    kept = strikes[strikes["Strike_pass"] <= 2].copy()
    kept = kept.sort_values(["Set_time", "Sequence_nr"], ascending=[False, True])
    kept["Strike_pass"] = kept.groupby("Marking").cumcount() + 1

    # One row per electrode, columns for pass 1 and pass 2
    pivot = kept.pivot(
        index="Marking", columns="Strike_pass", values=["Set_time", "Actual_time"]
    )
    pivot.columns = [f"{value}_strike{p}" for value, p in pivot.columns]

    # Keep electrodes in order of first appearance in the process data
    first_seen = strikes.groupby("Marking")["Sequence_nr"].min()
    result = pivot.loc[first_seen.sort_values().index].reset_index()

    # reindex (not []) so a missing pass column becomes NaN instead of a KeyError,
    # e.g. when no selected electrode has a second strike pass
    ordered_cols = ["Marking",
                    "Set_time_strike1", "Actual_time_strike1",
                    "Set_time_strike2", "Actual_time_strike2"]
    result = result.reindex(columns=ordered_cols)
    result["Set_time_total"] = result[
        ["Set_time_strike1", "Set_time_strike2"]].sum(axis=1, min_count=1)
    result["Actual_time_total"] = result[
        ["Actual_time_strike1", "Actual_time_strike2"]].sum(axis=1, min_count=1)
    # Actual_time = 0 means the step has not completed yet
    result["Completed"] = (
        (result["Actual_time_strike1"] > 0) & (result["Actual_time_strike2"] > 0))

    print("\n" + "=" * 50)
    print("Planned vs actual time per electrode (seconds)")
    print("=" * 50)
    with pd.option_context("display.max_rows", None, "display.width", None):
        print(result.to_string(index=False))

    output_file = Path(__file__).resolve().parent / "strike_times.csv"
    result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")


def main():
    config = load_config()
    print()
    engine = connect_database(config)

    metadata_df, processdata_df = fetch_data(engine)
    run_strike_analysis(metadata_df, processdata_df)


if __name__ == "__main__":
    main()
