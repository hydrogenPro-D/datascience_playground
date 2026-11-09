"""Bath chemistry and chemical dosing, keyed by date.

Needed to tell the two readings of the sulfur finding apart. If the S content
of a coating tracks what was dosed into the bath, sulfur co-deposition is a
real process effect. If it tracks nothing, the composition difference is more
likely an artefact of measuring a cracked surface.
"""
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parents[0] / "cracking_analysis_2026_september"))
from fetch_sem_data import load_config, connect_database

VIEWS = {
    "strike_bath": "qc.vw_strike_bath",
    "activator_bath": "qc.vw_activator_bath",
    "plate_bath_28": "qc.vw_plate_bath_28",
    "plate_bath_29": "qc.vw_plate_bath_29",
    "plate_bath_30": "qc.vw_plate_bath_30",
    "adding_chemicals": "qc.vw_adding_chemicals",
    "production_tracking": "qc.vw_production_tracking",
    "conductivity": "qc.vw_conductivity",
    "contamination": "qc.vw_contamination",
    "water_quality": "qc.vw_water_quality",
}


def main():
    engine = connect_database(load_config())
    with engine.connect() as conn:
        chemicals = pd.read_sql(text("SELECT * FROM qc.chemical"), conn)
        chemicals.to_csv(FOLDER / "chemical.csv", index=False)
        print("chemical list:")
        print(chemicals.to_string(index=False))

        for name, view in VIEWS.items():
            try:
                df = pd.read_sql(text(f"SELECT * FROM {view}"), conn)
                df.to_csv(FOLDER / f"{name}.csv", index=False)
                print(f"  {name:22s} {len(df):>6,} rows x {len(df.columns)} cols")
            except Exception as exc:
                print(f"  {name:22s} ! {str(exc)[:100]}")


if __name__ == "__main__":
    main()
