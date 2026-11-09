"""(b) Cracking vs the bath chemistry each electrode was made in.

Chemistry is measured every few days, so an electrode carries the last
measurement on or before its production date and a dozen electrodes share
identical numbers. Two consequences drive the method here:

  grouped folds  electrodes sharing a measurement stay in the same fold,
                 or the model scores itself on its own neighbours
  date baseline  bath state drifts with time and so does cracking, so a
                 chemistry result only counts if it beats date alone

Writes chemistry_dataset.csv, then models it.
"""
from pathlib import Path

import pandas as pd

import modeling

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
DATASET = HERE / "chemistry_dataset.csv"

ACTIVATOR = {"pH Mean": "act_ph",
             "Decasel 5 [g/L] Mean": "act_decasel",
             "Ni2+ [g/L] Mean": "act_ni"}
STRIKE = {"pH Mean": "strike_ph",
          "Ni2+ [g/L] Mean": "strike_ni",
          "avg. ml 30% HCl/L": "strike_hcl30"}
PLATE = {"pH Mean": "plate_ph",
         "Ni2+ [g/L] Mean": "plate_ni",
         "Cl- [g/L] Mean": "plate_cl",
         "ThioUrea [g/L] Mean": "plate_tu",
         "Boric Acid [g/L] Mean": "plate_ba"}


def carry_forward(electrodes, bath, columns, tag):
    """Last measurement on or before each electrode's production date."""
    bath = bath.rename(columns=columns)[["Date", *columns.values()]]
    bath = bath.assign(Date=pd.to_datetime(bath["Date"])).sort_values("Date")
    bath[f"{tag}_measured"] = bath["Date"]
    merged = pd.merge_asof(electrodes.sort_values("production_date"), bath,
                           left_on="production_date", right_on="Date",
                           direction="backward")
    return merged.drop(columns="Date")


def build():
    sem = pd.read_csv(SOURCE / "sem_measurements_filtered.csv")
    sem = sem.dropna(subset=["sem_result", "vats", "production_date"])
    sem["cracked"] = (sem["sem_result"] == "bad").astype(int)
    sem["plating_vat"] = sem["vats"].astype(str).str.split("|").str[0]
    sem["production_date"] = pd.to_datetime(sem["production_date"])
    rows = sem[["electrode", "cracked", "production_date", "plating_vat"]]

    rows = carry_forward(rows, pd.read_csv(SOURCE / "activator_bath_filtered.csv"),
                         ACTIVATOR, "act")
    rows = carry_forward(rows, pd.read_csv(SOURCE / "strike_bath_filtered.csv"),
                         STRIKE, "strike")

    plated = []
    for vat, part in rows.groupby("plating_vat"):
        path = SOURCE / f"plate_bath_{vat}_filtered.csv"
        if not path.exists():
            continue
        plated.append(carry_forward(part, pd.read_csv(path), PLATE, "plate"))
    rows = pd.concat(plated, ignore_index=True)

    features = [*ACTIVATOR.values(), *STRIKE.values(), *PLATE.values()]
    rows = rows.dropna(subset=features)
    rows.to_csv(DATASET, index=False)
    print(f"✓ {len(rows)} electrodes x {rows.shape[1]} columns -> "
          f"{DATASET.name}")
    return rows, features


def main():
    frame, features = build()
    y = frame["cracked"]
    # electrodes sharing a plating measurement share every chemistry number
    groups = frame["plate_measured"].astype(str)
    print(f"chemistry states: {groups.nunique()} distinct plating "
          f"measurements behind {len(frame)} electrodes")

    modeling.evaluate("(b) bath chemistry, grouped folds",
                      frame[features], y, groups=groups)

    day = pd.DataFrame({"day": frame["production_date"].map(
        lambda d: d.toordinal())})
    modeling.evaluate("BASELINE: production date only", day, y, groups=groups,
                      show=1)

    modeling.positive_control(frame, y)


if __name__ == "__main__":
    main()
