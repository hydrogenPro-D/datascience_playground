"""PyQt6 app (opens maximized): the charge report as plain, sortable data.

The charge report is a sheet to read; this is the same electrodes laid out to
be worked with. The differences are deliberate:

  * run times, not overruns - the actual seconds the step took, one column
    per process position rather than the five the charge report picks out
  * the age of the bath reading is its own column instead of riding along in
    brackets behind the value
  * the coat verdict keeps its words on screen but carries a number behind it,
    so clicking the heading sorts best-to-worst instead of alphabetically
  * coating_label: the binary the analysis actually uses, 1 cracked, 0 not
  * every heading sorts, ascending then descending

The heavy lifting - loading, the bath as-of join, the two-line header - is
imported from gui_app rather than copied, so the two stay in step.
"""
import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTableWidget, QTableWidgetItem, QFileDialog,
                             QMessageBox, QAbstractItemView, QHeaderView)

import coating_correlation as correlation
import data_filters
import dose_age_tab
import ml_models
import vat_timeline
import initials_dictionary as initials
from gui_app import (AXES_BG, COAT_ORDER, COLOR_CRACKED,
                     FIG_BG, GRID, LABEL, SPINE, SUBTEXT, TEXT,
                     MultiLineHeader, NumericItem, load_data,
                     load_step_times)

# Positions 1 and 2 are dropped by export_step_times.py before the data ever
# gets here; this is belt and braces in case that changes.
EXCLUDED_STEPS = {1, 2}


def step_numbers(long_steps):
    """Every process position in the data, low to high, 1 and 2 aside.

    Read off the data rather than hard-coded: a recipe change that adds a
    position should add a column, not go unnoticed."""
    if long_steps is None or not len(long_steps):
        return []
    found = pd.to_numeric(long_steps["step"], errors="coerce").dropna()
    return sorted({int(value) for value in found} - EXCLUDED_STEPS)

# Higher is better, the same direction as the SEM quality score: the best
# coat in COAT_ORDER gets the highest number, anything unrecognised none.
COAT_RATING = {name: len(COAT_ORDER) - i for i, name in enumerate(COAT_ORDER)}


PH_FILE = "ph_control_filtered.csv"
# The line is in Norway and the pH log is written in local wall-clock time,
# while dbo.MetaData stores UTC. Converting through the zone rather than
# adding two hours keeps both sides of the March and October changeovers
# right, and the log is only a few hours apart so two hours is the
# difference between the correct reading and the one before it.
PLANT_TZ = "Europe/Oslo"
# Past this, the last dose says nothing about the bath the electrode saw
DOSE_WINDOW_HOURS = 7 * 24


def load_ph():
    """The tidied pH log, or None when the pipeline has not produced it."""
    path = Path(__file__).resolve().parent / PH_FILE
    if not path.exists():
        print(f"WARNING {PH_FILE} not found - run data_preprocessing.py; "
              f"the pH columns will be empty")
        return None
    ph = pd.read_csv(path, parse_dates=["measured_at"])
    ph["vat"] = ph["vat"].astype(str)
    return ph


def local_production_time(data):
    """Electrode production time as Oslo wall-clock, to match the pH log."""
    stamp = pd.to_datetime(data.get("production_datetime_utc"),
                           errors="coerce", utc=True)
    return stamp.dt.tz_convert(PLANT_TZ).dt.tz_localize(None)


def _asof(made, rows, columns, when="measured_at"):
    """Each electrode's most recent row of `rows` at or before it was made."""
    rows = rows.dropna(subset=[when]).sort_values(when)
    # One log entry can list several chemicals at the same instant; keeping
    # the last leaves merge_asof a single row per timestamp to land on.
    rows = rows.drop_duplicates(subset=[when], keep="last")
    if not len(rows) or not len(made):
        return None
    joined = pd.merge_asof(made.sort_values("made_at"), rows[[when] + columns],
                           left_on="made_at", right_on=when,
                           direction="backward")
    joined["age_h"] = ((joined["made_at"] - joined[when])
                       .dt.total_seconds() / 3600)
    return joined.set_index("electrode")


def _per_vat(made, readings, columns, when="measured_at"):
    """Same, but each electrode is matched only against its own vat."""
    pieces = []
    for vat, part in made.groupby("vat"):
        rows = readings[readings["vat"] == vat]
        joined = _asof(part, rows, columns, when)
        if joined is not None:
            pieces.append(joined)
    return pd.concat(pieces) if pieces else None


def _made_frame(data):
    """One row per electrode: which plating vat, and when it was made."""
    return pd.DataFrame({
        "electrode": data["electrode"].astype(str).str.strip().values,
        "vat": data["vats"].astype(str).str.split("|").str[0].values,
        "made_at": local_production_time(data).values,
    }).dropna(subset=["made_at"])


def build_ph(data, ph):
    """Per electrode: the last pH of its own vat before it was plated, and
    the last acid or base dosed into that plating vat.

    The dose is the plating vat's alone. qc.vw_ph_control carries a column for
    the strike vat as well, but it has never been filled in - 0 rows of 998 -
    so there is nothing to show; the strike make-up that does get logged lives
    in qc.vw_adding_chemicals and is a different kind of event.

    Its strike pH is not read either: that log stopped on 2026-07-20, so it
    could only ever show a reading a month or two stale. The strike bath log
    (strike_bath_long.csv) is current and carries a pH mean instead.

    Unlike the lab bath readings this is joined on the exact production
    timestamp, not the day: pH is logged every few hours and an electrode
    made at 23:51 saw a different bath than one made at 06:00."""
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for name in ("ph", "ph_age", "dose", "chemical", "dose_age"):
        out[name] = None if name == "chemical" else float("nan")
    if ph is None or "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)

    measured = ph[ph["pH"].notna()]
    plating = _per_vat(made, measured, ["pH"])
    if plating is not None:
        out["ph"] = plating["pH"]
        out["ph_age"] = plating["age_h"]

    dosed = _per_vat(made, ph[ph["dose"].notna()], ["dose", "chemical"])
    if dosed is not None:
        # Anything older than the window is not what this electrode was
        # plated in, so it is left blank rather than shown as stale
        fresh = dosed["age_h"] <= DOSE_WINDOW_HOURS
        out["dose"] = dosed["dose"].where(fresh)
        out["chemical"] = dosed["chemical"].where(fresh)
        out["dose_age"] = dosed["age_h"].where(fresh)
    return out


CHEM_LONG_FILE = "chemical_additions_long.csv"
# (key, vat, chemical, label). A vat of None means the electrode's own
# plating vat, so those columns always describe the bath it was actually in.
# The chemicals are exactly the ones the log contains; a new one would need
# a line here, and would otherwise go unreported.
CHEM_SPECS = [
    ("v17", "17", "Slotoclean EL DCG", "V17 Slotoclean"),
    ("v18", "18", "Slotoclean Decasel 5", "V18 Decasel 5"),
    ("tu", None, "TU", "Plating TU"),
    ("ba", None, "BA", "Plating BA"),
    ("co", None, "CoSO4*7H2O", "Plating CoSO4"),
    ("niso4", None, "NiSO4*6H2O", "Plating NiSO4"),
    ("nicl2", None, "NiCl2*6H2O", "Plating NiCl2"),
    ("hcl", "31", "HCl 30%", "Strike HCl 30%"),
    ("s_nicl2", "31", "NiCl2*6H2O", "Strike NiCl2"),
]


def load_chemical_additions():
    """The full parsed addition log, or None if the pipeline has not run."""
    path = Path(__file__).resolve().parent / CHEM_LONG_FILE
    if not path.exists():
        print(f"WARNING {CHEM_LONG_FILE} not found - run "
              f"data_preprocessing.py; the chemical columns will be empty")
        return None
    chems = pd.read_csv(path, parse_dates=["added_at"])
    chems["vat"] = chems["vat"].astype(str)
    chems["chemical"] = chems["chemical"].astype(str).str.strip()
    return chems


def build_chemicals(data, chems):
    """Per electrode and chemical: the last dose into the relevant vat.

    Vats 17, 18 and 31 are shared by every electrode; the plating chemicals
    come from whichever of 28/29/30 the electrode was actually in. Each
    chemical carries its own age because they are dosed on wildly different
    rhythms: TU goes in daily, NiSO4 a few times a year."""
    keys = [f"chem_{key}" for key, *_rest in CHEM_SPECS]
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key in keys:
        out[key] = float("nan")
        out[f"{key}_age"] = float("nan")
    out["chem_by"] = None
    if chems is None or "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)
    for key, vat, chemical, _label in CHEM_SPECS:
        rows = chems[chems["chemical"] == chemical]
        if vat is None:
            joined = _per_vat(made, rows, ["amount"], when="added_at")
        else:
            joined = _asof(made, rows[rows["vat"] == vat], ["amount"],
                           when="added_at")
        if joined is None:
            continue
        out[f"chem_{key}"] = joined["amount"]
        # Days, not hours: these span daily top-ups to months-old make-up
        out[f"chem_{key}_age"] = joined["age_h"] / 24

    # Who logged the most recent addition to this electrode's plating vat,
    # whatever was in it. Blank where the operator left Initials empty.
    signed = _per_vat(made, chems, ["Initials"], when="added_at")
    if signed is not None:
        # Through the dictionary, so "dc" and "DC", or "AT MA" and "MA AT",
        # are one operator rather than two rows that never add up
        out["chem_by"] = signed["Initials"].map(initials.canonical)
        if initials.UNKNOWN:
            print(f"NOTE {len(initials.UNKNOWN)} operator spellings are not "
                  f"in initials_dictionary.py and were normalised by rule: "
                  f"{sorted(initials.UNKNOWN)}")
    return out


CONDUCTIVITY_FILE = "conductivity_long.csv"
# In rinse-cascade order, as the view lists them: an electrode drips through
# 14 -> 13 -> 12 after the cleaner, 19 -> 20 after activation, 25 after the
# strike, and 24 -> 23 -> 22 -> 21 after plating.
#
# All ten are read on one walk-round, so they share a single age column. The
# values therefore come from ONE round - the last before the electrode - and
# a tank the operator skipped that round is blank rather than filled from an
# older round the age would then misdescribe. Position 25 is the one skipped
# most (missing from 30 of 165 rounds).
CONDUCTIVITY_POSITIONS = [14, 13, 12, 19, 20, 25, 24, 23, 22, 21]

CLEANER_FILE = "cleaner_bath_long.csv"


def load_conductivity():
    """The rinse conductivity log, or None if the pipeline has not run."""
    path = Path(__file__).resolve().parent / CONDUCTIVITY_FILE
    if not path.exists():
        print(f"WARNING {CONDUCTIVITY_FILE} not found - run "
              f"data_preprocessing.py; conductivity columns will be empty")
        return None
    return pd.read_csv(path, parse_dates=["measured_at"])


def load_cleaner():
    """The cathodic cleaner bath log, or None if the pipeline has not run."""
    path = Path(__file__).resolve().parent / CLEANER_FILE
    if not path.exists():
        print(f"WARNING {CLEANER_FILE} not found - run "
              f"data_preprocessing.py; the cleaner pH column will be empty")
        return None
    return pd.read_csv(path, parse_dates=["measured_at"])


def build_conductivity(data, cond, cleaner):
    """Per electrode: the last conductivity reading of each rinse tank, and
    the last cleaner-bath pH, both as of the moment it was made.

    None of this is vat-specific - every electrode runs the same rinse
    cascade and the same cleaner - so each position is matched against the
    whole log rather than per vat."""
    keys = [f"cond{position}" for position in CONDUCTIVITY_POSITIONS]
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key in keys + ["cond_age"]:
        out[key] = float("nan")
    for key in ("cleaner_ph", "slotoclean"):
        out[key] = float("nan")
        out[f"{key}_age"] = float("nan")
    if "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)
    if cond is not None:
        # Back to one row per walk-round, so the whole round is matched at
        # once and every value in it shares the one age
        rounds = cond.pivot_table(index="measured_at", columns="position",
                                  values="conductivity", aggfunc="last")
        rounds.columns = [f"cond{int(position)}"
                          for position in rounds.columns]
        rounds = rounds.reset_index()
        joined = _asof(made, rounds,
                       [c for c in rounds.columns if c != "measured_at"])
        if joined is not None:
            for key in keys:
                if key in joined.columns:
                    out[key] = joined[key]
            out["cond_age"] = joined["age_h"] / 24
    if cleaner is not None:
        joined = _asof(made, cleaner[cleaner["pH"].notna()], ["pH"])
        if joined is not None:
            out["cleaner_ph"] = joined["pH"]
            out["cleaner_ph_age"] = joined["age_h"] / 24
        # Its own join and age: the Schloetter titration is run on 67 of the
        # 85 rounds, so the newest concentration can be older than the
        # newest pH from the same bath.
        if "slotoclean" in cleaner.columns:
            titrated = cleaner[cleaner["slotoclean"].notna()]
            joined = _asof(made, titrated, ["slotoclean"])
            if joined is not None:
                out["slotoclean"] = joined["slotoclean"]
                out["slotoclean_age"] = joined["age_h"] / 24
    return out


ACTIVATOR_FILE = "activator_bath_long.csv"
STRIKE_FILE = "strike_bath_long.csv"
# (key, column in the derived log, heading). The conversions themselves are
# done once in data_preprocessing.py - titration ml x 0.05 for [H+], x 120.06
# for Decasel 5, x 2.935 for Ni2+, x 8.58 for HCl at 36%, x 1.2 for 30% - so
# there is one place where a factor can be wrong, not two.
# Means only: the 1st and 2nd readings behind each one stay in
# activator_bath_long.csv and strike_bath_long.csv for anyone who wants them.
ACTIVATOR_COLUMNS = [
    ("act_h", "Decasel 5 [H+] Mean", "Act. [H+]\nmean"),
    ("act_decasel", "Decasel 5 [g/L] Mean", "Act. Decasel 5\n[g/L]"),
    ("act_ni", "Ni2+ [g/L] Mean", "Act. Ni2+\nmean [g/L]"),
]
STRIKE_COLUMNS = [
    ("stk_ph", "pH Mean", "Strike bath\npH mean"),
    ("stk_ni", "Ni2+ [g/L] Mean", "Strike Ni2+\nmean [g/L]"),
    ("stk_hcl36", "avg. ml 36% HCl/L", "Strike HCl\navg 36% [ml/L]"),
    ("stk_hcl30", "avg. ml 30% HCl/L", "Strike HCl\navg 30% [ml/L]"),
]


def _load_log(name, what, when="measured_at"):
    path = Path(__file__).resolve().parent / name
    if not path.exists():
        print(f"WARNING {name} not found - run the pipeline; "
              f"the {what} columns will be empty")
        return None
    parse = [when] if when else None
    return pd.read_csv(path, parse_dates=parse)


def load_activator():
    return _load_log(ACTIVATOR_FILE, "activator bath")


def load_strike():
    return _load_log(STRIKE_FILE, "strike bath")


def build_baths(data, activator, strike):
    """Per electrode: the activator and strike bath readings it was made under.

    Both tanks are shared by every electrode, so neither is matched per vat.
    Each bath is sampled as one round, so its values share a single age."""
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key, _source, _title in ACTIVATOR_COLUMNS + STRIKE_COLUMNS:
        out[key] = float("nan")
    out["act_age"] = float("nan")
    out["stk_age"] = float("nan")
    if "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)
    for log, spec, age_key in ((activator, ACTIVATOR_COLUMNS, "act_age"),
                               (strike, STRIKE_COLUMNS, "stk_age")):
        if log is None:
            continue
        wanted = [source for _key, source, _title in spec
                  if source in log.columns]
        joined = _asof(made, log, wanted)
        if joined is None:
            continue
        for key, source, _title in spec:
            if source in joined.columns:
                out[key] = joined[source]
        out[age_key] = joined["age_h"] / 24
    return out


PLATE_FILE = "plate_bath_long.csv"
# Final values only, as asked. The conversions live in data_preprocessing.py:
# Cl- x 10.638, ThioUrea x 2.2836, Boric Acid x 6.18, nickel photometric at
# 445 nm, and cobalt as what the Ni+Co titration counts that the nickel does
# not. The 1st/2nd readings behind each mean stay in plate_bath_long.csv.
PLATE_COLUMNS = [
    ("plate_ph", "pH Mean", "Plate bath\npH mean"),
    ("plate_ni", "Ni2+ [g/L] Mean", "Plate Ni2+\n[g/L]"),
    ("plate_co", "Co2+ [g/L] Mean", "Plate Co2+\n[g/L]"),
    ("plate_cl", "Cl- [g/L] Mean", "Plate Cl-\n[g/L]"),
    ("plate_tu", "ThioUrea [g/L] Mean", "Plate ThioUrea\n[g/L]"),
    ("plate_ba", "Boric Acid [g/L] Mean", "Plate Boric Acid\n[g/L]"),
]


def load_plate():
    plate = _load_log(PLATE_FILE, "plate bath")
    if plate is not None:
        # read_csv makes "28" an integer; the electrode's vat is a string,
        # and an int never equals a str, so the join would silently find
        # nothing at all
        plate["vat"] = plate["vat"].astype(str)
    return plate


def build_plate(data, plate):
    """Per electrode: the last reading of its OWN plating vat before it ran.

    This replaces the day-level join the table used to make through
    gui_app.build_chemistry: same three numbers, plus pH, cobalt and boric
    acid, matched on the production timestamp instead of the calendar day
    and reaching back past the analysed window when a vat went unsampled."""
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key, _source, _title in PLATE_COLUMNS:
        out[key] = float("nan")
    out["plate_age"] = float("nan")
    if plate is None or "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)
    wanted = [source for _key, source, _title in PLATE_COLUMNS
              if source in plate.columns]
    joined = _per_vat(made, plate, wanted)
    if joined is None:
        return out
    for key, source, _title in PLATE_COLUMNS:
        if source in joined.columns:
            out[key] = joined[source]
    out["plate_age"] = joined["age_h"] / 24
    return out


WATER_FILE = "water_quality_long.csv"
# The feed water itself, not a tank: one reading serves every electrode, so
# it is never matched per vat. Logging only started on 2026-08-11, so
# electrodes made before that have no reading to reach back to at all.
WATER_COLUMNS = [
    ("water_cond", "Conductivity (\u00b5S/cm)",
     "Water quality\ncond. [\u00b5S/cm]"),
    ("water_ph", "pH", "Water quality\npH"),
]


def load_water():
    return _load_log(WATER_FILE, "water quality")


def build_water(data, water):
    """Per electrode: the last feed-water reading before it was made."""
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key, _source, _title in WATER_COLUMNS:
        out[key] = float("nan")
    out["water_age"] = float("nan")
    if water is None or "production_datetime_utc" not in data.columns:
        return out

    made = _made_frame(data)
    wanted = [source for _key, source, _title in WATER_COLUMNS
              if source in water.columns]
    joined = _asof(made, water, wanted)
    if joined is None:
        return out
    for key, source, _title in WATER_COLUMNS:
        if source in joined.columns:
            out[key] = joined[source]
    out["water_age"] = joined["age_h"] / 24
    return out


SIGNALS_FILE = "signals_grid.csv"
# (role, the Kar it means, heading prefix). "own" is the electrode's own
# station, picked from whichever of the listed ones it actually ran in.
SIGNAL_ROLES = [
    ("cleaner", [17], "Cleaner"),
    ("plating", "own_vat", "Plating"),
    ("strike", [31], "Strike"),
    ("dryer", "own_dryer", "Dryer"),
]
DRYER_KARS = [40, 41, 44]
# Type -> (key suffix, heading suffix). The dryers log temperature only, so
# their current and voltage columns would be blank for every electrode and
# are left out.
SIGNAL_CHANNELS = {0: ("t", "T median [\u00b0C]"),
                   1: ("i", "I median [A]"),
                   2: ("v", "V median [V]")}


def load_signals_grid():
    """The per-electrode medians off the 10 s grid, built by
    build_signal_grid.py. Not the raw points - those are four million rows
    in signals_raw_points.parquet and nothing here needs them."""
    return _load_log(SIGNALS_FILE, "signal median", when=None)


def build_signals(data, signals):
    """Per electrode: the median of each channel over its 10 s grid.

    The grid is the zero-order hold rebuild - ProductionData only logs a
    change, so a step that held 780 A for an hour is two rows, and averaging
    raw rows would weight that hour the same as ten seconds. Every point on
    the grid carries ten seconds of real time instead.

    Plating and drying are matched to the electrode's own station; the
    cleaner and the strike are shared by everyone."""
    keys = [f"sig_{role}_{suffix}"
            for role, _kars, _label in SIGNAL_ROLES
            for _t, (suffix, _title) in SIGNAL_CHANNELS.items()
            if not (role == "dryer" and suffix in ("i", "v"))]
    out = pd.DataFrame(index=pd.Index(data["electrode"].astype(str).str.strip(),
                                      name="electrode"))
    for key in keys:
        out[key] = float("nan")
    if signals is None or not len(signals):
        return out

    grid = signals.copy()
    grid["Marking"] = grid["Marking"].astype(str).str.strip()
    grid["Vat"] = pd.to_numeric(grid["Vat"], errors="coerce")
    own = pd.Series(data["vats"].astype(str).str.split("|").str[0].values,
                    index=pd.Index(data["electrode"].astype(str).str.strip()))

    for role, kars, _label in SIGNAL_ROLES:
        if kars == "own_vat":
            # The electrode's own plating vat, from the SEM data
            wanted = grid[grid["Marking"].map(own).astype(str)
                          == grid["Vat"].astype("Int64").astype(str)]
        elif kars == "own_dryer":
            # An electrode runs one dryer; where the logger wrote more than
            # one, the longest grid is the real stay
            dryers = grid[grid["Vat"].isin(DRYER_KARS)]
            best = (dryers.sort_values("readings")
                    .drop_duplicates(subset=["Marking", "Type"], keep="last"))
            wanted = best
        else:
            wanted = grid[grid["Vat"].isin(kars)]
        for signal_type, (suffix, _title) in SIGNAL_CHANNELS.items():
            key = f"sig_{role}_{suffix}"
            if key not in keys:
                continue
            part = wanted[wanted["Type"] == signal_type]
            if not len(part):
                continue
            values = (part.drop_duplicates(subset=["Marking"], keep="last")
                      .set_index("Marking")["value_median"])
            out[key] = values
    return out


SEM_RAW_FILE = "sem_raw.csv"
# Where sem_analysis keeps its own pull. Read from there rather than from the
# _p1/_p2/_p3 columns in sem_measurements_filtered.csv: those carry one SEM
# position, while 69 of these electrodes were measured at five, and averaging
# only the first makes C, N, Cl and Al disagree with gui_sem by up to 0.14
# percentage points. One source, one set of numbers.
SEM_FOLDER = Path(__file__).resolve().parent.parent / "sem_analysis"
# Every element the EDS ever reported, commonest first. The thin ones are
# kept because this is the all-data table and a blank is honest: Si, Br and K
# are measured on 17, 15 and 2 electrodes.
#
# Deliberately unitless, as in gui_sem.py. sem_raw.csv records a symbol and a
# value and nothing else, so the unit is not known: the points that carry a
# full element set do sum to about 100, which says the figures are normalised
# percentages, but only 333 of 2833 points do - most are missing O (~27) or C
# (~25) and sum to about 45. Calling the column "%" would imply a row adds up
# when it usually does not, and would still not say whether the lab exported
# weight or atomic percent, which are different numbers.
SEM_ELEMENTS = ["Ni", "S", "Co", "C", "N", "O", "Cl", "Al", "Si", "Br", "K"]
# The same five gui_sem offers, in the same order
SEM_RATIOS = [("Ni", "S"), ("Ni", "Co"), ("Ni", "C"), ("Ni", "N"), ("S", "N")]


def load_sem_elements():
    """One row per electrode, one column per element, or None if absent."""
    path = SEM_FOLDER / SEM_RAW_FILE
    if not path.exists():
        print(f"WARNING {path.name} not found - run the pipeline's 'sem' "
              f"step; the element and ratio columns will be empty")
        return None
    raw = pd.read_csv(path)
    raw["electrode"] = raw["electrode"].astype(str).str.strip()
    readings = raw.dropna(subset=["symbol", "value"])
    # Mean over every reading of that element - each position and each point
    # - which is exactly what gui_sem's build_table does.
    return readings.pivot_table(index="electrode", columns="symbol",
                                values="value", aggfunc="mean")


def build_sem(data, sem):
    """Per electrode: each element's mean percentage and the five ratios."""
    index = pd.Index(data["electrode"].astype(str).str.strip(),
                     name="electrode")
    out = pd.DataFrame(index=index)
    for element in SEM_ELEMENTS:
        out[f"sem_{element}"] = float("nan")
    for top, bottom in SEM_RATIOS:
        out[f"sem_{top}_{bottom}"] = float("nan")
    if sem is None:
        return out

    for element in SEM_ELEMENTS:
        if element in sem.columns:
            out[f"sem_{element}"] = sem[element].reindex(index)
    for top, bottom in SEM_RATIOS:
        if top not in sem.columns or bottom not in sem.columns:
            continue
        above = sem[top].reindex(index)
        below = sem[bottom].reindex(index)
        # A zero denominator is not a ratio of infinity, it is unknown
        out[f"sem_{top}_{bottom}"] = above / below.where(below != 0)
    return out


class LabelItem(NumericItem):
    """A cell that reads as words but sorts as a number.

    NumericItem already keeps the value behind the text; this only takes the
    right-alignment back off, because the text here is a word, not a figure."""

    def __init__(self, value, text):
        super().__init__(value, text)
        self.setTextAlignment(Qt.AlignmentFlag.AlignLeft
                              | Qt.AlignmentFlag.AlignVCenter)


class AllDataWidget(QWidget):
    """One row per electrode, every field in its own column, all sortable."""

    # (key, heading, kind). "kind" decides both the formatting and the sort:
    # anything but "text" and "date" gets the number kept behind the label.
    # The position columns sit between HEAD and TAIL and are built per
    # instance, because which positions exist is a property of the data.
    HEAD = [("electrode", "Batch #", "text"),
            ("date", "Date", "date"),
            ("vat", "Vat", "text")]
    TAIL = ([(key, title, "float2")
             for key, _source, title in PLATE_COLUMNS]
            + [("plate_age", "Plate bath\nage [days]", "float"),
            ("ph", "pH\n(own vat)", "float2"),
            ("ph_age", "pH age\n[hours]", "float"),
            ("cleaner_ph", "Cleaner bath\npH", "float2"),
            ("cleaner_ph_age", "Cleaner pH\nage [days]", "float"),
            ("slotoclean", "Slotoclean\nEL DCG [g/L]", "float2"),
            ("slotoclean_age", "Slotoclean\nage [days]", "float"),
            ]
            + [(key, title, "float2")
               for key, _source, title in ACTIVATOR_COLUMNS]
            + [("act_age", "Activator bath\nage [days]", "float")]
            + [(key, title, "float2")
               for key, _source, title in STRIKE_COLUMNS]
            + [("stk_age", "Strike bath\nage [days]", "float")]
            + [(key, title, "float2")
               for key, _source, title in WATER_COLUMNS]
            + [("water_age", "Water quality\nage [days]", "float")]
            + [(f"sig_{role}_{suffix}", f"{label}\n{title}", "float2")
               for role, _kars, label in SIGNAL_ROLES
               for _t, (suffix, title) in SIGNAL_CHANNELS.items()
               if not (role == "dryer" and suffix in ("i", "v"))]
            + [(f"sem_{element}", f"SEM {element}", "float2")
               for element in SEM_ELEMENTS]
            + [(f"sem_{top}_{bottom}", f"SEM {top}:{bottom}\nratio",
                "float2")
               for top, bottom in SEM_RATIOS]
            + [
            # "plating" spelled out: this is the acid or base dosed into the
            # electrode's own plating vat in answer to a pH reading, not
            # anything added to the strike vat or any bulk make-up.
            ("dose", "Last plating\naddition [L]", "float2"),
            ("chemical", "Last plating\nchemical", "text"),
            ("dose_age", "Plating addition\nage [hours]", "float"),
            ("chem_by", "Added by\n(plating vat)", "text"),
            ("coat", "Coat integrity", "coat"),
            ("coating_label", "coating_label", "int")])

    # A reading per rinse tank, then the one age they all share
    CONDUCTIVITY = ([(f"cond{position}",
                      f"Cond. pos.{position}\n[\u00b5S/cm]", "float2")
                     for position in CONDUCTIVITY_POSITIONS]
                    + [("cond_age", "Conductivity\nage [days]", "float")])

    # An amount and its own age for every chemical the log records
    CHEMICALS = [pair
                 for key, _vat, _chem, label in CHEM_SPECS
                 for pair in ((f"chem_{key}", f"{label}\n[kg]", "float2"),
                              (f"chem_{key}_age", f"{label}\nage [days]",
                               "float"))]

    @classmethod
    def columns_for(cls, steps):
        """One run-time column per position, between the head and the tail."""
        return (cls.HEAD
                + [(f"pos{step}", f"Pos. {step}\nrun time [sec.]", "int")
                   for step in steps]
                + cls.CONDUCTIVITY
                + cls.CHEMICALS
                + cls.TAIL)

    def __init__(self, data, long_steps, plate=None, ph=None, chems=None,
                 cond=None, cleaner=None, activator=None, strike=None,
                 water=None, signals=None, sem=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("All data")
        heading.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        bar.addWidget(heading)
        bar.addStretch()
        save = QPushButton("Export (PDF, Excel, CSV)")
        save.setStyleSheet(
            f"QPushButton {{ color: {TEXT}; font-size: 12px; "
            f"padding: 5px 14px; border: 1px solid {SPINE}; "
            f"border-radius: 4px; background: {AXES_BG}; }}"
            f"QPushButton:hover {{ border-color: {COLOR_CRACKED}; }}")
        save.clicked.connect(self._export)
        bar.addWidget(save)
        layout.addLayout(bar)

        self._widget = None
        # Shadows the class attribute: the positions come from the data
        self.COLUMNS = self.columns_for(step_numbers(long_steps))
        self._frame = self._build(data, long_steps, plate, ph, chems,
                                  cond, cleaner, activator, strike, water,
                                  signals, sem)
        if self._frame is None or not len(self._frame):
            layout.addWidget(QLabel("No electrodes to report."))
            return
        self._widget = self._table(self._frame)
        layout.addWidget(self._widget)

    # ------------------------------------------------------------------ data

    @staticmethod
    def _build(data, long_steps, plate=None, ph=None, chems=None, cond=None,
               cleaner=None, activator=None, strike=None, water=None,
               signals=None, sem=None):
        """One row per electrode: run times, bath reading, coat verdict."""
        if long_steps is None or not len(long_steps):
            return None
        steps = long_steps.copy()
        steps["electrode"] = steps["electrode"].astype(str).str.strip()
        steps["step"] = pd.to_numeric(steps["step"], errors="coerce")
        steps = steps[steps["step"].notna()
                      & ~steps["step"].isin(EXCLUDED_STEPS)]
        # actual_time, not overrun: the seconds the step took, already
        # de-accumulated upstream for the passes that log a running total.
        # Summed across passes, so a position an electrode visits more than
        # once (28/29/30 run three, 31 runs two) reads as total time there.
        times = steps.pivot_table(index="electrode", columns="step",
                                  values="actual_time", aggfunc="sum")
        times.columns = [f"pos{int(step)}" for step in times.columns]

        frame = data[["electrode", "production_date", "vats"]].copy()
        frame["electrode"] = frame["electrode"].astype(str).str.strip()
        coat = ("note_corrected" if "note_corrected" in data.columns
                else "sem_result")
        frame["coat"] = data[coat].values
        # The binary the rest of the analysis runs on: verdict is 1 good,
        # 0 cracked, so the label is its mirror image.
        frame["coating_label"] = 1 - pd.to_numeric(data["verdict"].values,
                                                   errors="coerce")
        frame["vat"] = frame["vats"].astype(str).str.split("|").str[0]
        frame["date"] = pd.to_datetime(frame["production_date"],
                                       errors="coerce")
        frame = frame.set_index("electrode").join(times)

        frame = frame.join(build_plate(data, plate))
        frame = frame.join(build_ph(data, ph))
        frame = frame.join(build_chemicals(data, chems))
        frame = frame.join(build_conductivity(data, cond, cleaner))
        frame = frame.join(build_baths(data, activator, strike))
        frame = frame.join(build_water(data, water))
        frame = frame.join(build_signals(data, signals))
        frame = frame.join(build_sem(data, sem))

        frame["coat_rating"] = frame["coat"].map(COAT_RATING)
        # Vat first, then best coat first, the same order as the charge report
        frame["_rank"] = (frame["coat"].map({n: i for i, n
                                             in enumerate(COAT_ORDER)})
                          .fillna(len(COAT_ORDER)))
        frame = frame.sort_values(["vat", "_rank", "date"])
        return frame.reset_index()

    # ------------------------------------------------------------------ view

    def _table(self, frame):
        widget = QTableWidget()
        widget.setColumnCount(len(self.COLUMNS))
        widget.setHorizontalHeaderLabels(
            [title for _k, title, _kind in self.COLUMNS])
        widget.setRowCount(len(frame))
        widget.setSortingEnabled(False)
        widget.setAlternatingRowColors(True)
        widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        widget.verticalHeader().setVisible(False)
        widget.setStyleSheet(
            f"QTableWidget {{ background: {AXES_BG}; gridline-color: {GRID}; "
            f"color: {TEXT}; font-size: 12px; }}"
            f"QTableWidget::item {{ padding: 3px 6px; }}"
            f"QHeaderView::section {{ background: {FIG_BG}; "
            f"border: none; border-bottom: 1px solid {SPINE}; }}")

        for row, (_i, record) in enumerate(frame.iterrows()):
            for column, (key, _title, kind) in enumerate(self.COLUMNS):
                widget.setItem(row, column, self._cell(record, key, kind))

        header = MultiLineHeader(widget)
        font = header.font()
        font.setPixelSize(11)
        font.setBold(True)
        header.setFont(font)
        header.setMinimumSectionSize(44)
        header.label_color = QColor(LABEL)
        widget.setHorizontalHeader(header)
        # Section -1 is "unsorted": without it, switching sorting on would
        # immediately sort by column 0 and lose the vat/coat order above.
        # Clicking a heading still sorts, ascending then descending.
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        widget.setSortingEnabled(True)
        for column, (key, _t, _kind) in enumerate(self.COLUMNS):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch if key == "coat"
                else QHeaderView.ResizeMode.ResizeToContents)
        return widget

    @staticmethod
    def _cell(record, key, kind):
        value = record.get(key)
        blank = value is None or (not isinstance(value, str) and pd.isna(value))
        if blank:
            # A missing measurement is blank, never a zero
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, "")
            return item
        if kind == "date":
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, f"{value:%Y-%m-%d}")
            return item
        if kind == "int":
            return NumericItem(value, f"{value:,.0f}")
        if kind == "float":
            return NumericItem(value, f"{value:,.1f}")
        if kind == "float2":
            return NumericItem(value, f"{value:,.2f}")
        if kind == "coat":
            # The words stay on screen; the rating rides behind them so the
            # column sorts by quality rather than by first letter.
            rating = record.get("coat_rating")
            item = LabelItem(0 if pd.isna(rating) else rating, str(value))
            if str(value) in ("Cracked", "Large cracks"):
                item.setForeground(QColor(COLOR_CRACKED))
            return item
        if kind != "text":
            # A kind with no branch above used to fall through to str() and
            # print 3.243636363636364 where 3.24 was meant. Better to fail
            # on the spot than to ship raw floats into the table again.
            raise ValueError(f"no formatting defined for kind {kind!r}")
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, str(value))
        return item

    # ---------------------------------------------------------------- export

    def _export(self):
        """Write what is on screen, in the order it is shown, three ways."""
        if self._widget is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save all data", "all_data.pdf", "PDF (*.pdf)")
        if not path:
            return
        widget = self._widget
        titles = [widget.horizontalHeaderItem(c).text().replace("\n", " ")
                  for c in range(widget.columnCount())]
        # The PDF columns are too narrow to carry the words as well
        pdf_titles = [title.replace(" run time", "") for title in titles]
        shown = [[widget.item(r, c).text() if widget.item(r, c) else ""
                  for c in range(widget.columnCount())]
                 for r in range(widget.rowCount())]

        from gui_app import ChargeReportWidget
        try:
            ChargeReportWidget._write_pdf(path, pdf_titles, shown)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(self, "Could not save",
                                f"{type(exc).__name__}: {exc}")
            return

        from pathlib import Path
        book = str(Path(path).with_suffix(".xlsx"))
        sheet = str(Path(path).with_suffix(".csv"))
        try:
            frame = self._export_frame(titles, shown)
            ChargeReportWidget._write_excel(book, frame)
            frame.to_csv(sheet, index=False)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(
                self, "PDF saved, the rest did not",
                f"The PDF is at\n{path}\n\n{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(
            self, "Saved",
            f"{len(shown)} rows written to\n{path}\n{book}\n{sheet}")

    def _export_frame(self, titles, shown):
        """The source rows in display order, with real numbers.

        Read from the frame rather than off the cells, so a spreadsheet gets
        values it can sort and total. The coat rating cannot hide behind the
        text in a file, so it becomes a column of its own."""
        order = [row[0] for row in shown]
        source = self._frame.set_index("electrode").reindex(order)
        out = pd.DataFrame({"Batch #": order})
        for key, title, _kind in self.COLUMNS[1:]:
            column = source[key]
            if key == "date":
                column = column.dt.strftime("%Y-%m-%d")
            out[title.replace("\n", " ")] = column.values
        out.insert(out.columns.get_loc("Coat integrity") + 1,
                   "Coat rating", source["coat_rating"].values)
        return out


class FilteredTab(AllDataWidget):
    """The same table under data_filters, with what it did written on it.

    Subclasses AllDataWidget rather than rebuilding it: the cell formatting,
    the sortable header, and the PDF/Excel/CSV export are all wanted here
    unchanged. Only the construction differs - the frame arrives already
    filtered instead of being assembled from the loaders.
    """

    def __init__(self, frame, columns, notes, heading="Filtered data"):
        # Deliberately not AllDataWidget.__init__: that one builds the frame
        QWidget.__init__(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 16)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        title = QLabel(heading)
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch()
        save = QPushButton("Export (PDF, Excel, CSV)")
        save.setStyleSheet(
            f"QPushButton {{ color: {TEXT}; font-size: 12px; "
            f"padding: 5px 14px; border: 1px solid {SPINE}; "
            f"border-radius: 4px; background: {AXES_BG}; }}"
            f"QPushButton:hover {{ border-color: {COLOR_CRACKED}; }}")
        save.clicked.connect(self._export)
        bar.addWidget(save)
        layout.addLayout(bar)

        note = QLabel("Same electrodes as All data, nothing dropped by row. "
                      + "; ".join(notes) + ".")
        note.setStyleSheet(f"color: {SUBTEXT}; font-size: 12px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.COLUMNS = columns
        self._frame = frame
        self._widget = self._table(frame)
        layout.addWidget(self._widget)


class MainWindow(QMainWindow):
    def __init__(self, data, long_steps, plate=None, ph=None, chems=None,
                 cond=None, cleaner=None, activator=None, strike=None,
                 water=None, signals=None, sem=None):
        super().__init__()
        self.setWindowTitle("Electrode data")
        tabs = QTabWidget()

        everything = AllDataWidget(data, long_steps, plate, ph, chems, cond,
                                   cleaner, activator, strike, water,
                                   signals, sem)
        tabs.addTab(everything, "All data")
        # Fed the same COLUMNS the table was built from, so the correlation
        # always covers exactly what the data tab shows - a column added
        # there turns up here without a second edit.
        tabs.addTab(correlation.CorrelationTab(
            everything._frame, everything.COLUMNS,
            "All data : coating correlation"),
            "All data coating correlation")

        # The same rows, with duplicate columns gone, stale readings
        # blanked and a measured/not indicator on every patchy column
        clean, kept, notes = data_filters.apply_filters(
            everything._frame, everything.COLUMNS)
        for line in notes:
            print(f"  filter: {line}")
        tabs.addTab(FilteredTab(clean, kept, notes), "Filtered data")
        tabs.addTab(correlation.CorrelationTab(
            clean, kept, "Filtered data : coating correlation"),
            "Filtered data coating correlation")

        # These only READ model_results.json, written by fit_models.py.
        # Nothing is fitted while the window is open; each tab is told the
        # table's current shape so it can flag a fit that predates it.
        for frame, columns, where in ((everything._frame,
                                       everything.COLUMNS, "All data"),
                                      (clean, kept, "Filtered data")):
            for kind in ("forest", "network"):
                tabs.addTab(
                    ml_models.ModelTab(where, kind,
                                       f"{where} : "
                                       f"{ml_models.TITLES[kind]}",
                                       rows=len(frame),
                                       columns=len(columns)),
                    f"{where} {ml_models.TITLES[kind]}")

        tabs.addTab(dose_age_tab.DoseAgeTab(everything._frame),
                    "Decasel dose age")

        # The two vats behind the forest's "days since dosing" features,
        # each with its dosing, its bath and its electrodes on one axis
        for which in vat_timeline.TIMELINES:
            tabs.addTab(
                vat_timeline.VatTimelineTab(everything._frame, which),
                which)
        self.setCentralWidget(tabs)


def main():
    data = load_data()
    data["vats"] = data["vats"].astype(str)

    app = QApplication(sys.argv)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{ background: {FIG_BG}; }}
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            background: transparent;
            padding: 10px 24px;
            font-size: 14px;
            color: {SUBTEXT};
            border-bottom: 2px solid transparent;
        }}
        QTabBar::tab:selected {{
            color: {TEXT};
            font-weight: bold;
            border-bottom: 2px solid {COLOR_CRACKED};
        }}
        QTabBar::tab:hover {{ color: {TEXT}; }}
    """)
    window = MainWindow(
        data, load_step_times("all_step_times_long.csv", "run times"),
        load_plate(), load_ph(), load_chemical_additions(),
        load_conductivity(), load_cleaner(), load_activator(),
        load_strike(), load_water(), load_signals_grid(),
        load_sem_elements())
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
