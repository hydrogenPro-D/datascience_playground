"""What is actually in this database?

Every analysis so far has drawn on four tables that someone picked by hand.
Before modelling anything, list every table and column, so a feature that was
never looked at (a timestamp, a rack id, an operator) gets a chance to exist.
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 60)

engine = connect_database(load_config())

with engine.connect() as conn:
    print("\n" + "=" * 78)
    print("TABLES AND VIEWS")
    print("=" * 78)
    objects = pd.read_sql(text("""
        SELECT TABLE_SCHEMA AS [schema], TABLE_NAME AS [name],
               TABLE_TYPE AS [type]
        FROM INFORMATION_SCHEMA.TABLES
        ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME"""), conn)
    print(objects.to_string(index=False))

    print("\n" + "=" * 78)
    print("COLUMNS OF EVERY TABLE / VIEW")
    print("=" * 78)
    columns = pd.read_sql(text("""
        SELECT TABLE_SCHEMA AS [schema], TABLE_NAME AS [name],
               COLUMN_NAME AS [column], DATA_TYPE AS [type]
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION"""), conn)
    for (schema, name), part in columns.groupby(["schema", "name"]):
        cols = ", ".join(f"{c}:{t}" for c, t in
                         zip(part["column"], part["type"]))
        print(f"\n[{schema}.{name}]\n  {cols}")

    print("\n" + "=" * 78)
    print("ROW COUNTS (tables only)")
    print("=" * 78)
    for _i, row in objects[objects["type"] == "BASE TABLE"].iterrows():
        full = f"[{row['schema']}].[{row['name']}]"
        try:
            n = pd.read_sql(text(f"SELECT COUNT(*) AS n FROM {full}"),
                            conn)["n"].iloc[0]
            print(f"  {full:50s} {n:>12,}")
        except Exception as exc:
            print(f"  {full:50s}  ! {str(exc)[:60]}")
