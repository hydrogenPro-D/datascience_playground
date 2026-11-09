from pathlib import Path
import json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool
import pandas as pd

def load_config():
    print("Loading configuration from config/config.json...")
    config_file = Path("config/config.json")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")
    with open(config_file, "r") as f:
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
    try:
        engine = create_engine(
            connection_url,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args={"options": "-c timezone=None"},
        )
        print("✓ Database connection established")
        return engine
    except Exception as e:
        print(f"✗ Database Connection Failed: {e}")
        return None
    
config = load_config()
print()
engine = connect_database(config)

metadata_table = "MetaData"
processdata_table = "ProcessData"
productiondata_table = "ProductionData"

query_metadata = f"""SELECT Marking,
                   Rcp_nr AS Recipe,
                   Full_name AS Description,
                   CONVERT(VARCHAR(19), Date, 120) AS Datetime
                   FROM {metadata_table}
                   WHERE 1=1"""

query_processdata = f"""SELECT Id AS Sequence_nr, Marking, Proces AS Step_nr,
                   Rcp_tid AS Set_time, Aktuel_tid AS Actual_time
                    FROM ProcessData
                    ORDER BY Id"""

query_productiondata = f"""SELECT Unik_nummer AS Marking, Kar AS Step_nr, Type, Value,
                    CONVERT(VARCHAR(19), Aktuel_tid, 120) AS Datetime
                    FROM ProductionData
                    ORDER BY Unik_nummer, Aktuel_tid"""


with engine.connect() as conn:
    print("\nFetching metadata...")
    metadata_df = pd.read_sql(text(query_metadata), conn)
    print(f"✓ Metadata fetched: {len(metadata_df)} rows")

    print("\nFetching process data...")
    processdata_df = pd.read_sql(text(query_processdata), conn)
    print(f"✓ Process data fetched: {len(processdata_df)} rows")

    print("\nFetching production data...")
    productiondata_df = pd.read_sql(text(query_productiondata), conn)
    print(f"✓ Production data fetched: {len(productiondata_df)} rows")

print("\n" + "="*50)
print("Saving data to CSV files...")
print("="*50)

# Ensure data folder exists
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Save files
metadata_file_path = data_folder / "metadata.csv"
metadata_df.to_csv(metadata_file_path, index=False)
print(f"✓ Metadata saved to {metadata_file_path}")

processdata_df_file_path = data_folder / "processdata.csv"
processdata_df.to_csv(processdata_df_file_path, index=False)
print(f"✓ Process data saved to {processdata_df_file_path}")

productiondata_df_file_path = data_folder / "productiondata.csv"
productiondata_df.to_csv(productiondata_df_file_path, index=False)
print(f"✓ Production data saved to {productiondata_df_file_path}")

print("\n" + "="*50)
print("✓ All data successfully fetched and saved!")
print("="*50)

