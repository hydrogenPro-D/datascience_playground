"""One row per electrode: what the instruments read in each vat it visited.

All three signals are summarised by their median, which ignores the spikes
in current and voltage. Raw readings are pulled rather than the
pre-aggregated signals.csv, which only holds means.
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
DATASET = HERE / "vat_signals_dataset.csv"
sys.path.insert(0, str(SOURCE))

from fetch_sem_data import load_config, connect_database  # noqa: E402

SCALED = (0, 2)          # temperature and voltage are stored x100
NAMES = {0: "temp", 1: "current", 2: "voltage"}
HOW = {0: "median", 1: "median", 2: "median"}


def fetch(electrodes, engine):
    markings = ", ".join(f"'{m}'" for m in electrodes)
    query = f"""SELECT Unik_nummer AS electrode, Kar AS vat, Type AS type,
                CAST(Value AS float) AS value
                FROM dbo.ProductionData
                WHERE Type IN (0, 1, 2) AND Kar IN (28, 29, 30, 31)
                  AND Unik_nummer IN ({markings})"""
    with engine.connect() as conn:
        print("Fetching raw temperature / current / voltage readings...")
        df = pd.read_sql(text(query), conn)
    df["electrode"] = df["electrode"].astype(str).str.strip()
    df["vat"] = df["vat"].astype(str)
    df.loc[df["type"].isin(SCALED), "value"] /= 100.0
    print(f"✓ {len(df)} readings for {df['electrode'].nunique()} electrodes")
    return df


def summarise(readings):
    stats = []
    for (electrode, vat, type_), part in readings.groupby(
            ["electrode", "vat", "type"]):
        how = HOW[type_]
        stats.append({"electrode": electrode, "vat": vat,
                      "signal": NAMES[type_],
                      "value": getattr(part["value"], how)(),
                      "readings": len(part)})
    return pd.DataFrame(stats)


def main():
    sem = pd.read_csv(SOURCE / "sem_measurements_filtered.csv")
    sem = sem.dropna(subset=["sem_result", "vats"])
    sem["cracked"] = (sem["sem_result"] == "bad").astype(int)
    sem["plating_vat"] = sem["vats"].astype(str).str.split("|").str[0]

    engine = connect_database(load_config())
    stats = summarise(fetch(list(sem["electrode"]), engine))
    lookup = stats.set_index(["electrode", "vat", "signal"])["value"]

    rows = sem[["electrode", "cracked", "production_date", "plating_vat"]].copy()
    for where in ("strike", "plating"):
        for signal in NAMES.values():
            rows[f"{where}_{signal}"] = [
                lookup.get((e, "31" if where == "strike" else v, signal))
                for e, v in zip(rows["electrode"], rows["plating_vat"])]

    features = [c for c in rows.columns
                if c.startswith(("strike_", "plating_"))
                and c != "plating_vat"]
    complete = rows.dropna(subset=features)
    complete.to_csv(DATASET, index=False)
    print(f"✓ {len(complete)} electrodes x {complete.shape[1]} columns -> "
          f"{DATASET.name}")
    print(f"  dropped {len(rows) - len(complete)} electrodes missing a signal")
    print("\n" + complete[features].describe().round(2).to_string())


if __name__ == "__main__":
    main()
