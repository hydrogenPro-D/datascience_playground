"""Preprocessing for the cracking analysis on SEM measurements.

Reads sem_measurements.csv (produced by fetch_sem_data.py) and never modifies it;
all output goes to a separate filtered CSV. Analysis logic to be defined.
"""
import re
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(__file__).resolve().parent / "sem_measurements.csv"
VATS_FILE = Path(__file__).resolve().parent / "strike_vats.csv"
METADATA_FILE = Path(__file__).resolve().parent / "electrode_metadata.csv"
STRIKE_BATH_FILE = Path(__file__).resolve().parent / "strike_bath.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "sem_measurements_filtered.csv"
BATH_OUTPUT_FILE = Path(__file__).resolve().parent / "strike_bath_filtered.csv"
PH_CONTROL_FILE = Path(__file__).resolve().parent / "ph_control.csv"
PH_OUTPUT_FILE = Path(__file__).resolve().parent / "ph_control_filtered.csv"
CHEM_LONG_FILE = "chemical_additions_long.csv"
CONDUCTIVITY_FILE = Path(__file__).resolve().parent / "conductivity.csv"
CONDUCTIVITY_LONG_FILE = "conductivity_long.csv"
CLEANER_BATH_FILE = Path(__file__).resolve().parent / "cleaner_bath.csv"
CLEANER_BATH_LONG_FILE = "cleaner_bath_long.csv"
ACTIVATOR_LONG_FILE = "activator_bath_long.csv"
PLATE_LONG_FILE = "plate_bath_long.csv"
WATER_FILE = Path(__file__).resolve().parent / "water_quality.csv"
WATER_LONG_FILE = "water_quality_long.csv"
STRIKE_LONG_FILE = "strike_bath_long.csv"


GENERATION = "Gen4.1"

# Titration factors for the strike bath (mirrors the weekly result scheme's
# weekly_result_scheme.strike config; the Plate baths use different Cl- etc.
# factors). Applied as volume [ml] x factor, nothing is calculated in SQL.
STRIKE_NI_FACTOR = 2.935     # Ni2+ [g/L] per ml titrated
STRIKE_CL_FACTOR = 11.51     # Cl-  [g/L] per ml titrated
STRIKE_HCL_FACTOR = 8.58     # HCl  [ml/L] per ml titrated (36 %)
HCL_30_RATIO = 1.2           # 36 % average -> 30 % average

# Plate bath factors (weekly_result_scheme.plate); note Cl- differs from the
# strike bath (10.638 vs 11.51) — factors never carry across baths. One
# generic pattern serves vats 28/29/30, like _process_plate_titration_table.
#
# Ni2+ is NOT in this list any more. The plate baths stopped titrating nickel
# in July 2026 and moved to a 445nm photometric reading; the old
# "Ni2+ Titration" columns are still in the view but have been empty ever
# since, so deriving from them would silently blank the series. The strike and
# activator baths still titrate nickel and are untouched.
PLATE_FACTORS = [
    ("Cl-", 10.638),
    ("ThioUrea", 2.2836),
    ("Boric Acid", 6.18),
]
# Three readings per sampling, converted per reading and averaged as mol/L.
# The two molar-mass terms differ by a factor of ten (/1000 out, x10000 back),
# which is where the dilution enters.
PLATE_NI_ABSORBANCE = ["Ni Absorbance 445nm 1st", "Ni Absorbance 445nm 2nd",
                       "Ni Absorbance 445nm 3rd"]
PLATE_NI_BLANK = 0.003        # absorbance of the blank
PLATE_NI_SLOPE = 0.216466     # absorbance per mg/L
PLATE_NI_MOLAR_MASS = 58.693  # g/mol
# The photometric nickel is read on a 1:10000 dilution, so every mol/L from
# it is that much below the bath's real concentration.
PLATE_DILUTION = 10000
# Cobalt is never measured on its own. The EDTA titration counts nickel AND
# cobalt together, so cobalt is what is left once the photometric nickel is
# taken off it. 0.05 mol/L per ml is the titrant strength.
PLATE_NICO_COLUMNS = ["Ni2+ + Co2+ Titration 1st (ml)",
                      "Ni2+ + Co2+ Titration 2nd (ml)"]
PLATE_NICO_FACTOR = 0.05      # mol/L of Ni+Co per ml titrated
PLATE_CO_MOLAR_MASS = 58.93   # g/mol
PLATE_VATS = [28, 29, 30]

# Chemical additions: one free-text cell holds several lines like
# "4.05 kg TU" / "1 kg CoSO4*7H2O" — parsed into one row per chemical.
CHEM_VAT_COLUMNS = {
    17: "Vat. 17 (Cathodic) (kg)",
    18: "Vat. 18 (Activation ) (kg)",   # the stray space is in the view
    28: "Vat. 28 (Plating 1) (kg)",
    29: "Vat. 29 (Plating 2) (kg)",
    30: "Vat. 30 (Plating 3) (kg)",
    31: "Vat. 31 (Strike) (kg)",
}
CHEM_LINE_RE = re.compile(r"^([\d]+(?:[.,]\d+)?)\s*(kg|g|L|ml)\s+(.+)$",
                          re.IGNORECASE)


# Activator bath factors. Decasel 5 goes through [H+] first: titration ml
# x 0.05 gives [H+], and [H+] x 120.06 gives g/L. Nickel shares the plate
# bath's 2.935 and is a contaminant here, not an ingredient.
ACTIVATOR_H_FACTOR = 0.05
ACTIVATOR_DECASEL_FACTOR = 120.06
ACTIVATOR_NI_FACTOR = 2.935


def derive_activator_bath(bath):
    """Add the calculated columns to the activator bath frame.

    Same rounding discipline as the other baths: volumes rounded on read,
    every derived step rounded to 2 decimals, and means taken over the
    already-rounded values. Blank titrations stay blank rather than
    becoming a fake zero concentration.
    """
    bath["pH Mean"] = _pair_mean(_vol(bath, "pH 1st"), _vol(bath, "pH 2nd"))

    h1 = (_vol(bath, "Decasel5 Titration 1st (ml)") * ACTIVATOR_H_FACTOR).round(2)
    h2 = (_vol(bath, "Decasel5 Titration 2nd (ml)") * ACTIVATOR_H_FACTOR).round(2)
    bath["Decasel 5 [H+] 1st"] = h1
    bath["Decasel 5 [H+] 2nd"] = h2
    h_mean = _pair_mean(h1, h2)
    bath["Decasel 5 [H+] Mean"] = h_mean
    bath["Decasel 5 [g/L] Mean"] = (h_mean * ACTIVATOR_DECASEL_FACTOR).round(2)

    ni1 = (_vol(bath, "Nickel Titration 1st (ml)") * ACTIVATOR_NI_FACTOR).round(2)
    ni2 = (_vol(bath, "Nickel Titration 2nd (ml)") * ACTIVATOR_NI_FACTOR).round(2)
    bath["Ni2+ [g/L] 1st"] = ni1
    bath["Ni2+ [g/L] 2nd"] = ni2
    bath["Ni2+ [g/L] Mean"] = _pair_mean(ni1, ni2)
    return bath


def derive_step_times(vat_rows, per_vat=False):
    """Total time each electrode spent in a process step (plating vat 28/29/
    30, or the strike step 31 — the same shape either way).

    ProcessData stores Actual_time cumulatively when a step repeats: a
    modern electrode's three rows in one vat read e.g. 30 -> 59 -> 3664,
    where 59 is 30+29 and 3664 is 59+3605. A row's real duration is its
    value minus the previous row's, applied only when the value is not
    smaller than the previous one (a smaller value cannot be a running
    total).

    The running total resets per step, so the chain is per (Marking, Vat):
    older electrodes have one 180s row in each of vats 28/29/30, and those
    are independent steps, not a running total. The strike step's two
    passes (420s then 60s) do chain, and are corrected the same way.
    """
    rows = vat_rows.sort_values("Id").copy()
    rows["Marking"] = rows["Marking"].astype(str).str.strip()

    previous = rows.groupby(["Marking", "Vat"])["Actual_time"].shift()
    cumulative = previous.notna() & (rows["Actual_time"] >= previous)
    rows["step_time"] = rows["Actual_time"].where(
        ~cumulative, rows["Actual_time"] - previous)

    # per_vat keeps one row per step (rinsing visits 19, 20 and 25);
    # otherwise everything an electrode did is totalled into one row
    if per_vat:
        totals = rows.groupby(["Marking", "Vat"]).agg(
            steps=("Vat", "size"),
            set_time=("Set_time", "sum"),
            actual_time=("step_time", "sum")).reset_index()
    else:
        totals = rows.groupby("Marking").agg(
            vat=("Vat", lambda v: "|".join(str(x) for x in sorted(v.unique()))),
            steps=("Vat", "size"),
            set_time=("Set_time", "sum"),
            actual_time=("step_time", "sum")).reset_index()

    totals["overrun"] = totals["actual_time"] - totals["set_time"]
    totals = totals.rename(columns={"Marking": "electrode", "Vat": "vat"})
    totals["vat"] = totals["vat"].astype(str)
    return totals


def parse_chemicals(chems):
    """Melt the per-vat free-text columns into long rows:
    Date, Initials, Time, vat, amount, unit, chemical.
    Lines that don't parse keep their raw text with a blank amount."""
    rows = []
    for _, entry in chems.iterrows():
        for vat, column in CHEM_VAT_COLUMNS.items():
            cell = entry[column]
            if pd.isna(cell):
                continue
            for line in re.split(r"[\r\n]+", str(cell)):
                line = line.strip()
                if not line:
                    continue
                m = CHEM_LINE_RE.match(line)
                if m:
                    amount = float(m.group(1).replace(",", "."))
                    unit, chemical = m.group(2), m.group(3).strip()
                else:
                    amount, unit, chemical = None, None, line
                rows.append({"Date": entry["Date"],
                             "Initials": entry["Initials"],
                             "Time when added": entry["Time when added"],
                             "vat": vat, "amount": amount, "unit": unit,
                             "chemical": chemical})
    return pd.DataFrame(rows)


def chemical_timestamps(chem_long):
    """When each addition happened, as Oslo wall-clock.

    Only 122 of the 536 log entries record a time of day. A date-only entry
    is stamped at the start of its day, so an electrode made that day counts
    the addition rather than reaching back to the previous one.

    Local time, like the pH log and unlike dbo.MetaData, which is UTC."""
    day = pd.to_datetime(chem_long["Date"], errors="coerce")
    clock = pd.to_timedelta(chem_long["Time when added"], errors="coerce")
    return day + clock.fillna(pd.Timedelta(0))


# ------------------------------------------------------------ conductivity
# qc.vw_conductivity is one row per walk-round with a column per rinse
# position. Reshaped here to one row per position and round, which is what an
# as-of join against an electrode needs. Position numbers match the process
# positions the step times use, so "pos.14" is the same tank either way.
CONDUCTIVITY_COLUMNS = {
    14: "Cathodic Rinse 1 - pos.14 (\u00b5S/cm)",
    13: "Cathodic Rinse 2 - pos.13 (\u00b5S/cm)",
    12: "Cathodic Rinse 3 - pos.12 (\u00b5S/cm)",
    19: "Activation Rinse 1 - pos.19 (\u00b5S/cm)",
    20: "Activation Rinse 2 - pos.20 (\u00b5S/cm)",
    25: "Strike Rinse 1 - pos.25 (\u00b5S/cm)",
    24: "Plating Rinse 1 - pos.24 (\u00b5S/cm)",
    23: "Plating Rinse 2 - pos.23 (\u00b5S/cm)",
    22: "Plating Rinse 3 - pos.22 (\u00b5S/cm)",
    21: "Plating Rinse 4 - pos.21 (\u00b5S/cm)",
}


def log_timestamps(raw):
    """When a hand-written log entry happened, as Oslo wall-clock.

    These logs carry a date and, sometimes, the moment the row was typed up.
    The write time is trusted only when it lands on the day being reported -
    entries are often typed up a day or more later, and their clock then says
    nothing about when the sample was taken. Everything else is stamped at
    the start of its day.

    Local time, like the rest of the QC logs and unlike dbo.MetaData, which
    is UTC."""
    day = pd.to_datetime(raw["Date"], errors="coerce")
    # When the view records the measurement time itself, that beats both:
    # it is when the sample was taken, not when somebody typed it up.
    if "Time of measurement" in raw.columns:
        clock = pd.to_timedelta(raw["Time of measurement"], errors="coerce")
        measured = day + clock
    else:
        measured = pd.Series(pd.NaT, index=raw.index)
    written = pd.to_datetime(raw.get("Row added"), errors="coerce")
    if written is None or written.isna().all():
        return measured.fillna(day)
    same_day = written.dt.normalize() == day.dt.normalize()
    return measured.fillna(written.where(same_day.fillna(False), day))


def derive_conductivity(raw):
    """Wide conductivity log -> long: position, measured_at, value, Name.

    There is no measurement-time column, only the date and when the row was
    written. 17 of the 128 written entries were typed up a day or more after
    the date they report, and their clock says nothing about when the probe
    went in, so the write time is used only when it falls on the same day.
    Everything else is stamped at the start of its day, the same rule the
    chemical log uses."""
    stamp = log_timestamps(raw)

    frames = []
    for position, column in CONDUCTIVITY_COLUMNS.items():
        if column not in raw.columns:
            continue
        frames.append(pd.DataFrame({
            "position": position,
            "measured_at": stamp,
            "conductivity": pd.to_numeric(raw[column], errors="coerce"),
            "Name": raw.get("Name"),
        }))
    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["measured_at", "conductivity"])
    return long.sort_values(["position", "measured_at"]).reset_index(drop=True)


# ------------------------------------------------------------ cleaner bath
# qc.vw_cleaner_bath is the cathodic cleaner, vat 17 - the same tank the
# Slotoclean EL DCG goes into. Two pH readings per round, and a Schloetter
# protocol titration that converts to the Slotoclean concentration.
CLEANER_PH_COLUMNS = ["pH 1st", "pH 2nd"]
CLEANER_SCHLOETTER_COLUMNS = ["Schloetter protocol 1", "Schloetter protocol 2"]
# Schloetter protocol titration -> Slotoclean EL DCG concentration. Each
# reading converts on its own and the two calculated values are then
# averaged, which is the order the lab writes it down in.
SLOTOCLEAN_FACTOR = 53.8 / 5


def derive_cleaner_bath(raw):
    """One row per round: measured_at, pH mean, and the raw readings kept.

    Same timestamp rule as the conductivity log - the write time only when it
    lands on the day being reported, the start of that day otherwise. Only 17
    of the 85 rounds record a write time at all."""
    out = pd.DataFrame({
        "measured_at": log_timestamps(raw),
        "Name": raw.get("Name"),
    })
    for column in CLEANER_PH_COLUMNS + CLEANER_SCHLOETTER_COLUMNS:
        if column in raw.columns:
            out[column] = pd.to_numeric(raw[column], errors="coerce")
    # Mean of the two readings where both exist, else whichever one does -
    # the same convention as the plate baths and the pH log
    present = [c for c in CLEANER_PH_COLUMNS if c in out.columns]
    out["pH"] = out[present].mean(axis=1, skipna=True) if present else None
    calculated = []
    for number, column in enumerate(CLEANER_SCHLOETTER_COLUMNS, start=1):
        if column in out.columns:
            name = f"slotoclean_{number}"
            out[name] = (out[column] * SLOTOCLEAN_FACTOR).round(2)
            calculated.append(name)
    if calculated:
        # Mean of the two calculated values, or whichever one exists
        out["slotoclean"] = out[calculated].mean(axis=1,
                                                 skipna=True).round(2)
    out = out.dropna(subset=["measured_at"])
    return out.sort_values("measured_at").reset_index(drop=True)


def _vol(frame, col):
    """Titration volume: coerced (stray text -> blank) and rounded on read."""
    return pd.to_numeric(frame[col], errors="coerce").round(2)


def _pair_mean(a, b):
    """Mean of the two already-rounded calculated values, rounded again."""
    return pd.concat([a, b], axis=1).mean(axis=1).round(2)


def _plate_nickel(bath):
    """Ni2+ g/L per reading, from the 445nm absorbance.

    mg/L = (absorbance - blank) / slope, then to mol/L and back to g/L. Any
    reading the lab did not record stays blank rather than counting as zero.
    """
    missing = [c for c in PLATE_NI_ABSORBANCE if c not in bath.columns]
    if missing:
        print(f"⚠ plate bath has no {missing} — Ni2+ will be blank. "
              f"Check qc.vw_plate_bath_* against PLATE_NI_ABSORBANCE.")
        return [pd.Series(float("nan"), index=bath.index)
                for _ in PLATE_NI_ABSORBANCE]
    out = []
    for column in PLATE_NI_ABSORBANCE:
        mg_per_l = ((pd.to_numeric(bath[column], errors="coerce")
                     - PLATE_NI_BLANK) / PLATE_NI_SLOPE)
        mol_per_l = mg_per_l / (PLATE_NI_MOLAR_MASS * 1000)
        out.append((mol_per_l * PLATE_NI_MOLAR_MASS * 10000).round(2))
    return out


def derive_plate_bath(bath):
    """Add the calculated columns to a plate bath frame (any vat).

    Every titrated section is the same one-step pattern, volume x factor then
    mean — no second-order column like the strike bath's 30 % HCl. Nickel is
    the exception: it is read photometrically and takes three readings.
    """
    bath["pH Mean"] = _pair_mean(_vol(bath, "pH 1st"), _vol(bath, "pH 2nd"))

    readings = _plate_nickel(bath)
    for name, values in zip(("1st", "2nd", "3rd"), readings):
        bath[f"Ni2+ [g/L] {name}"] = values
    bath["Ni2+ [g/L] Mean"] = pd.concat(readings, axis=1).mean(axis=1).round(2)

    for prefix, factor in PLATE_FACTORS:
        v1 = (_vol(bath, f"{prefix} Titration 1st (ml)") * factor).round(2)
        v2 = (_vol(bath, f"{prefix} Titration 2nd (ml)") * factor).round(2)
        bath[f"{prefix} [g/L] 1st"] = v1
        bath[f"{prefix} [g/L] 2nd"] = v2
        bath[f"{prefix} [g/L] Mean"] = _pair_mean(v1, v2)

    bath["Co2+ [g/L] Mean"] = _plate_cobalt(bath)
    return bath


def _plate_cobalt(bath):
    """Cobalt by difference: what the Ni+Co titration counts that the
    photometric nickel does not.

    Both sides have to be in the same units before they can be subtracted,
    so the nickel goes back to mol/L of undiluted bath. Unrounded until the
    end: rounding a difference of two similar numbers at every step is how
    a small concentration turns into noise."""
    missing = [c for c in PLATE_NICO_COLUMNS if c not in bath.columns]
    if missing:
        print(f"⚠ plate bath has no {missing} — Co2+ will be blank.")
        return pd.Series(float("nan"), index=bath.index)

    titrations = [pd.to_numeric(bath[column], errors="coerce")
                  * PLATE_NICO_FACTOR for column in PLATE_NICO_COLUMNS]
    nickel_and_cobalt = pd.concat(titrations, axis=1).mean(axis=1)

    readings = []
    for column in PLATE_NI_ABSORBANCE:
        if column not in bath.columns:
            return pd.Series(float("nan"), index=bath.index)
        mg_per_l = ((pd.to_numeric(bath[column], errors="coerce")
                     - PLATE_NI_BLANK) / PLATE_NI_SLOPE)
        readings.append(mg_per_l / (PLATE_NI_MOLAR_MASS * 1000))
    nickel = pd.concat(readings, axis=1).mean(axis=1) * PLATE_DILUTION

    return ((nickel_and_cobalt - nickel) * PLATE_CO_MOLAR_MASS).round(2)


# ---------------------------------------------------------------- pH control
# qc.vw_ph_control is one row per check, laid out wide: a pH column (sometimes
# two) per vat and, beside it, whatever was dosed in response as free text
# ("2.00 L H2SO4"). Reshaped here to one row per vat and reading, which is
# what an as-of join against an electrode needs.
#
# TIMEZONE: these timestamps are Oslo local, while dbo.MetaData is UTC. Both
# are left as logged; whoever joins them converts. Getting this wrong shifts
# the match by two hours, and readings are only a few hours apart.
PH_VAT_COLUMNS = {
    "28": ["pH Vat. 28 1 meas.", "pH Vat. 28 2 meas."],
    "29": ["pH Vat. 29 1 meas.", "pH Vat. 29 2 meas."],
    "30": ["pH Vat. 30 1 meas.", "pH Vat. 30 2 meas."],
    "31": ["pH Vat. 31 - Strike"],
}
PH_ADDED_COLUMNS = {"28": "Added to vat.28 (L)", "29": "Added to vat.29 (L)",
                    "30": "Added to vat.30 (L)", "31": "Added to vat.31 (kg)"}
# "2.00 L H2SO4", "1,05 L NaOH", "0.5 kg ..." -> amount, chemical
_DOSE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*(?:L|kg)?\s*(.*)",
                   re.IGNORECASE)


def parse_dose(value):
    """Amount and chemical out of a free-text dose, or (None, None)."""
    if not isinstance(value, str) or not value.strip():
        return None, None
    hit = _DOSE.match(value.strip())
    if not hit:
        # Unparseable, but keep the words rather than dropping the event
        return None, value.strip()
    return float(hit.group(1).replace(",", ".")), (hit.group(2).strip()
                                                   or None)


def derive_ph_control(raw):
    """Wide pH log -> long: vat, measured_at, pH, dose, chemical."""
    day = pd.to_datetime(raw["Date"], errors="coerce")
    # When the pH was read, falling back to when the sample was taken and
    # then to when the row was written; 43 old rows carry none of the three
    # and are dropped rather than pinned to midnight.
    clock = pd.Series(pd.NaT, index=raw.index, dtype="timedelta64[ns]")
    for name in ("Time of measurement", "Sample taken time"):
        if name in raw.columns:
            clock = clock.fillna(pd.to_timedelta(raw[name], errors="coerce"))
    stamp = day + clock
    if "Row added" in raw.columns:
        stamp = stamp.fillna(pd.to_datetime(raw["Row added"], errors="coerce"))

    frames = []
    for vat, columns in PH_VAT_COLUMNS.items():
        present = [c for c in columns if c in raw.columns]
        if not present:
            continue
        # Mean of the two readings where the operator took both, otherwise
        # whichever one exists - the same convention as the lab bath columns
        ph = raw[present].mean(axis=1, skipna=True)
        column = PH_ADDED_COLUMNS.get(vat)
        dosed = (raw[column].map(parse_dose) if column in raw.columns
                 else pd.Series([(None, None)] * len(raw), index=raw.index))
        frames.append(pd.DataFrame({
            "vat": vat,
            "measured_at": stamp,
            "pH": ph,
            "dose": [d[0] for d in dosed],
            "chemical": [d[1] for d in dosed],
        }))

    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["measured_at"])
    # A row that says nothing about this vat is not a reading for this vat
    long = long[long["pH"].notna() | long["dose"].notna()]
    return long.sort_values(["vat", "measured_at"]).reset_index(drop=True)


def derive_strike_bath(bath):
    """Add the 11 calculated columns to the strike bath frame.

    Reproduces WeeklyResultSchemeProcessor.process_strike_sheet: volumes are
    coerced (stray text becomes blank, not an error) and rounded to 2 decimals
    on read, each converted value is rounded, and each mean is the mean of the
    already-rounded values — so results match the app display digit for digit.
    """
    def vol(col):
        return pd.to_numeric(bath[col], errors="coerce").round(2)

    def pair_mean(a, b):
        return pd.concat([a, b], axis=1).mean(axis=1).round(2)

    ph1, ph2 = vol("pH 1st"), vol("pH 2nd")
    bath["pH Mean"] = pair_mean(ph1, ph2)

    ni1 = (vol("Ni2+ Titration 1st (ml)") * STRIKE_NI_FACTOR).round(2)
    ni2 = (vol("Ni2+ Titration 2nd (ml)") * STRIKE_NI_FACTOR).round(2)
    bath["Ni2+ [g/L] 1st"] = ni1
    bath["Ni2+ [g/L] 2nd"] = ni2
    bath["Ni2+ [g/L] Mean"] = pair_mean(ni1, ni2)

    cl1 = (vol("Cl- Titration 1st (ml)") * STRIKE_CL_FACTOR).round(2)
    cl2 = (vol("Cl- Titration 2nd (ml)") * STRIKE_CL_FACTOR).round(2)
    bath["Cl- [g/L] 1st"] = cl1
    bath["Cl- [g/L] 2nd"] = cl2
    bath["Cl- [g/L] Mean"] = pair_mean(cl1, cl2)

    hcl1 = (vol("HCl+ Titration 1st (ml)") * STRIKE_HCL_FACTOR).round(2)
    hcl2 = (vol("HCl+ Titration 2nd (ml)") * STRIKE_HCL_FACTOR).round(2)
    bath["HCl [ml/L] 1st"] = hcl1
    bath["HCl [ml/L] 2nd"] = hcl2
    avg36 = pair_mean(hcl1, hcl2)
    bath["avg. ml 36% HCl/L"] = avg36
    # The only two-step calculation: scales the derived 36 % average,
    # not the raw volumes
    bath["avg. ml 30% HCl/L"] = (avg36 * HCL_30_RATIO).round(2)

    return bath

META_COLS = ["generation", "electrode_type", "note", "measured_on"]

# Canonical cracking categories: Fine, Microcracks, Microcracks/small cracks,
# Small cracks, Less cracked, Cracked, Large cracks, Conflicting, Other.
# Maps every observed raw note (after newline normalisation and ' | ' joining
# of multi-position notes) to its category. Unmapped values stay NaN and are
# reported, so new operator phrasings surface instead of vanishing silently.
# Operators write the vat into the note in half a dozen shapes - "Cracked
# Vat 28", "Microcracks vat 30", "vat 29; cracked" - and NOTE_DICT below
# enumerates the ones seen so far by hand. Any new shape silently scored blank
# and the electrode vanished from the charts, so the lookup now falls back to
# a normalised form: vat mentions and punctuation stripped, case folded, the
# common typos repaired. Exact matches are still tried first, so this can only
# rescue notes that scored nothing before - it can never change an existing
# mapping.
_VAT_MENTION = re.compile(r"\b(?:vat|vart|kar)\.?\s*\d+\b", re.IGNORECASE)
_NOTE_TYPOS = {"microcrackes": "microcracks", "microcacks": "microcracks",
               "micro cracks": "microcracks", "craked": "cracked",
               "cacks": "cracks"}


def _normalise_note(note):
    text = str(note).strip().lower()
    text = _VAT_MENTION.sub(" ", text)
    for wrong, right in _NOTE_TYPOS.items():
        text = text.replace(wrong, right)
    return re.sub(r"[\s,;.:|\-]+", " ", text).strip()


def map_note(note):
    """NOTE_DICT category for a note, or None if it says nothing we know."""
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None
    exact = NOTE_DICT.get(str(note).strip())
    if exact is not None:
        return exact
    return _NOTE_LOOKUP.get(_normalise_note(note))


NOTE_DICT = {
    # --- Fine ---
    "Fine": "Fine",
    "Rigid and Fine": "Fine",
    "vat 29; Fine. Notes from production: Burn on the electrode on the "
    "clamp side.": "Fine",
    # --- Microcracks ---
    "Microcracks": "Microcracks",
    "Microcraks": "Microcracks",
    "microcracks": "Microcracks",
    "Microcracked": "Microcracks",
    "microcracked": "Microcracks",
    "Mickrocracked": "Microcracks",
    "Few mickrocrackes": "Microcracks",
    "Few Microcracks, Lots of Ni": "Microcracks",
    "Few microcracks": "Microcracks",
    "few microcracks": "Microcracks",
    "Few Microcracks": "Microcracks",
    "Very few microcracks": "Microcracks",
    "Very few micro cracks": "Microcracks",
    "Few microcacks Vat 28": "Microcracks",
    "Few microcracks Vat 28": "Microcracks",
    "Few microcracks Vat 30": "Microcracks",
    "Microcracks Vat 28": "Microcracks",
    "Microcracks Vat 30": "Microcracks",
    "Microcracked Vat 29": "Microcracks",
    "Microcracked Vat 30": "Microcracks",
    "Microcracked, Vat 28": "Microcracks",
    "Microcracked, Vat 29": "Microcracks",
    "Microcracked, Vat 30": "Microcracks",
    "microcracks 9th electrode - Vat 30": "Microcracks",
    "Microcracks (given as anode but tested as a cathode)": "Microcracks",
    "Microcracks and small clumps of carbon": "Microcracks",
    "Microcracks, some salt deposits present": "Microcracks",
    "Microcracks, few salt deposits present": "Microcracks",
    "Few microcracks, few salt deposits present": "Microcracks",
    "White percipitate on sample, few microcracks": "Microcracks",
    "31/7-2026. 2nd run in new vat 28 Testing dummy electrode plated in 3,5 ph. Few Microcracks": "Microcracks",
    "3rd run in new vat 28 Testing dummy electrode plated in 3,73 ph. Few Microcracks": "Microcracks",
    # --- Microcracks/small cracks ---
    "Microcracks/small cracks": "Microcracks/small cracks",
    "Few small/micro cracks": "Microcracks/small cracks",
    "Few small/micro-cracks": "Microcracks/small cracks",
    "Few small/micro cracks Vat 28": "Microcracks/small cracks",
    "Few small/micro cracks Vat 30": "Microcracks/small cracks",
    "Mostly microcracks/very little cracks": "Microcracks/small cracks",
    "Small cracks/microcracks (kind of on the edge, but a little worse than B14159)": "Microcracks/small cracks",
    "Microcracks/small cracks | Microcracks": "Microcracks/small cracks",
    # --- Small cracks ---
    "Small cracks": "Small cracks",
    "small cracks": "Small cracks",
    "Few small cracks": "Small cracks",
    "Few small cacks": "Small cracks",
    "Few small cracks Vat 30": "Small cracks",
    "Small cracks Vat 30": "Small cracks",
    "Very few small cracks Vat 30": "Small cracks",
    "Very small cracks, Vat 28": "Small cracks",
    "Small cracks, some salt deposits present": "Small cracks",
    "Small cracks visible at higher magnifications": "Small cracks",
    # --- Less cracked ---
    "Less cracked": "Less cracked",
    "Less crack": "Less cracked",
    "Less Cracked": "Less cracked",
    "Less cracking": "Less cracked",
    "Less cracked Vat 28": "Less cracked",
    "Less cracked Vat 29": "Less cracked",
    "Less cracked Vat 30": "Less cracked",
    "less cracked Vat 30": "Less cracked",
    "Less Cracked Vat 28": "Less cracked",
    "Less Cracked Vat 30": "Less cracked",
    "Less cracked, Vat 28": "Less cracked",
    "Less cracked, Vat 29": "Less cracked",
    "Less cracked, Vat 30": "Less cracked",
    "Less Cracked, Vat 29": "Less cracked",
    "Less Cracked, Vat 30": "Less cracked",
    # --- Cracked ---
    "Cracked": "Cracked",
    "Cracks": "Cracked",
    "Cracked Vat 28": "Cracked",
    "Cracked Vat 29": "Cracked",
    "Cracked Vat 30": "Cracked",
    "Cracks, should do post mortem": "Cracked",
    "Cracked | Craked": "Cracked",
    "1st run in new vat.28 First dummy electrode plated in 3,3 pH. Cracked": "Cracked",
    # --- Large cracks ---
    "Large cracks Batch did not receive full plating recipe, hence, it's no surprise that coat lacks adhesion.": "Large cracks",
    # --- Conflicting multi-position notes ---
    "Less Cracked | Cracked | Microcracked | Cracked vat 30": "Conflicting",
    "Microcracked | Less cracked": "Conflicting",
    # --- Other ---
    "Lumps on the surface (espically close to the edges)": "Other",
}

_NOTE_LOOKUP = {}
_NOTE_CLASHES = {}
for _raw, _mapped in NOTE_DICT.items():
    _key = _normalise_note(_raw)
    if _key in _NOTE_LOOKUP and _NOTE_LOOKUP[_key] != _mapped:
        _NOTE_CLASHES[_key] = (_NOTE_LOOKUP[_key], _mapped)
    else:
        _NOTE_LOOKUP.setdefault(_key, _mapped)
if _NOTE_CLASHES:
    print("⚠ notes that normalise together but disagree, exact match still "
          "wins for these:")
    for _key, (_a, _b) in _NOTE_CLASHES.items():
        print(f"    {_key!r}: {_a} vs {_b}")


def unique_joined(series):
    """Collapse repeated metadata to its unique non-null values, ' | '-separated."""
    values = series.dropna().astype(str).str.replace("\n", " ").str.strip().unique()
    return " | ".join(values) if len(values) else None


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {INPUT_FILE.name}")

    gen = df[df["generation"] == GENERATION].copy()
    print(f"{GENERATION}: {len(gen)} rows, {gen['electrode'].nunique()} electrodes")

    # One line per electrode: metadata collapsed, element values spread to
    # <symbol>_p<point> columns (repeated measurements, e.g. multiple
    # positions, are averaged).
    meta = gen.groupby("electrode")[META_COLS].agg(unique_joined)

    values = gen.pivot_table(index="electrode", columns=["symbol", "point"],
                             values="value", aggfunc="mean")
    values.columns = [f"{symbol}_p{point}" for symbol, point in values.columns]

    wide = meta.join(values).reset_index()

    # Normalized cracking category; the raw note column stays untouched.
    wide.insert(wide.columns.get_loc("note") + 1,
                "note_corrected", wide["note"].map(map_note))

    # Official verdict: only open cracks (Cracked, and worse: Large cracks) are
    # bad; all closed-crack categories are good. Conflicting/Other/no note stay
    # empty — no verdict without a clear note.
    SEM_RESULT = {
        "Fine": "good",
        "Microcracks": "good",
        "Microcracks/small cracks": "good",
        "Small cracks": "good",
        "Less cracked": "good",
        "Cracked": "bad",
        "Large cracks": "bad",
    }
    wide.insert(1, "sem_result", wide["note_corrected"].map(SEM_RESULT))

    # Strike vat per electrode from ProcessData (Proces 28/29/30, exported by
    # fetch_sem_data.py). Normally one vat; multiple appear '|'-joined.
    vat_rows = pd.read_csv(VATS_FILE)
    vat_rows["Marking"] = vat_rows["Marking"].astype(str).str.strip()
    vat_map = (vat_rows.groupby("Marking")["Vat"]
               .agg(lambda v: "|".join(str(x) for x in sorted(v.unique()))))
    wide.insert(2, "vats", wide["electrode"].map(vat_map))
    print("\nvats distribution:")
    print(wide["vats"].value_counts(dropna=False).to_string())

    # Production day from dbo.MetaData (exported by fetch_sem_data.py)
    meta_rows = pd.read_csv(METADATA_FILE)
    meta_rows["Marking"] = meta_rows["Marking"].astype(str).str.strip()
    date_map = (meta_rows.dropna(subset=["Production_datetime"])
                .drop_duplicates("Marking")
                .set_index("Marking")["Production_datetime"].str[:10])
    wide.insert(3, "production_date", wide["electrode"].map(date_map))
    # The full timestamp as well, and it is UTC: the pH log is logged in Oslo
    # local time, so anything joining the two has to convert first.
    stamp_map = (meta_rows.dropna(subset=["Production_datetime"])
                 .drop_duplicates("Marking")
                 .set_index("Marking")["Production_datetime"])
    wide.insert(4, "production_datetime_utc",
                wide["electrode"].map(stamp_map))
    missing_date = wide["production_date"].isna()
    if missing_date.any():
        print(f"\n⚠ {int(missing_date.sum())} electrodes without a production date "
              f"in MetaData: {sorted(wide.loc[missing_date, 'electrode'])}")

    # Experimental recipes (e.g. "ANODE 567 FLEM EXP") run their own set
    # times — drip step 16 set to 1s instead of 60s, say — which wrecks any
    # time-against-recipe comparison. Drop them from the whole analysis.
    descriptions = wide["electrode"].map(
        meta_rows.drop_duplicates("Marking")
        .set_index("Marking")["Description"].astype(str)).fillna("")
    experimental = descriptions.str.contains(r"\bEXP", case=False, regex=True)
    if experimental.any():
        dropped = ", ".join(f"{e} ({descriptions[i].strip()})" for i, e
                            in wide.loc[experimental, "electrode"].items())
        print(f"\nDropped {int(experimental.sum())} experimental-recipe "
              f"electrodes: {dropped}")
        wide = wide[~experimental].reset_index(drop=True)
    unmapped = wide["note"].notna() & wide["note_corrected"].isna()
    if unmapped.any():
        print(f"\n⚠ {int(unmapped.sum())} notes not in NOTE_DICT (note_corrected left empty):")
        for value in wide.loc[unmapped, "note"].unique():
            print(f"    {value!r}")
    print("\nnote_corrected distribution:")
    print(wide["note_corrected"].value_counts(dropna=False).to_string())
    print("\nsem_result distribution:")
    print(wide["sem_result"].value_counts(dropna=False).to_string())

    wide.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Filtered data saved to {OUTPUT_FILE.name} "
          f"({len(wide)} rows, {len(wide.columns)} columns)")

    # Strike bath chemistry (qc.vw_strike_bath), clipped to the production
    # date range of the selected electrodes (inclusive)
    first_day = wide["production_date"].min()
    last_day = wide["production_date"].max()
    bath = pd.read_csv(STRIKE_BATH_FILE)
    bath["Date"] = pd.to_datetime(bath["Date"]).dt.strftime("%Y-%m-%d")
    in_range = bath[(bath["Date"] >= first_day) & (bath["Date"] <= last_day)].copy()
    print(f"\nStrike bath: {len(in_range)} of {len(bath)} measurements fall in "
          f"the electrode range {first_day} .. {last_day}")
    in_range = derive_strike_bath(in_range)
    in_range.to_csv(BATH_OUTPUT_FILE, index=False)
    print(f"✓ Strike bath data saved to {BATH_OUTPUT_FILE.name} "
          f"({len(in_range.columns)} columns incl. derived)")

    folder = Path(__file__).resolve().parent

    # Activator bath chemistry, same date clip
    activator = pd.read_csv(folder / "activator_bath.csv")
    activator["Date"] = pd.to_datetime(activator["Date"]).dt.strftime("%Y-%m-%d")
    act_range = activator[(activator["Date"] >= first_day)
                          & (activator["Date"] <= last_day)].copy()
    act_range = derive_activator_bath(act_range)
    act_range.to_csv(folder / "activator_bath_filtered.csv", index=False)
    print(f"✓ Activator bath: {len(act_range)} of {len(activator)} "
          f"measurements in range, saved to activator_bath_filtered.csv")

    # Plate bath chemistry per vat, same date clip and derivation pattern
    for vat in PLATE_VATS:
        plate = pd.read_csv(folder / f"plate_bath_{vat}.csv")
        plate["Date"] = pd.to_datetime(plate["Date"]).dt.strftime("%Y-%m-%d")
        in_r = plate[(plate["Date"] >= first_day)
                     & (plate["Date"] <= last_day)].copy()
        in_r = derive_plate_bath(in_r)
        out = folder / f"plate_bath_{vat}_filtered.csv"
        in_r.to_csv(out, index=False)
        print(f"✓ Plate bath {vat}: {len(in_r)} of {len(plate)} measurements "
              f"in range, saved to {out.name}")

    # pH log, reshaped long and clipped to the electrode window plus a
    # month of run-up so the first electrodes still have a reading behind them
    if PH_CONTROL_FILE.exists():
        ph_long = derive_ph_control(pd.read_csv(PH_CONTROL_FILE))
        # Not clipped to the electrode window the way the baths are: vats 29
        # and 30 go unlogged for a month at a time, so an electrode can need
        # a reading from well before the window to have one at all.
        ph_long.to_csv(PH_OUTPUT_FILE, index=False)
        print(f"\n✓ pH control: {len(ph_long)} readings, saved to "
              f"{PH_OUTPUT_FILE.name}")
        print(ph_long.groupby("vat").agg(
            readings=("pH", "count"), doses=("dose", "count"),
            first=("measured_at", "min"),
            last=("measured_at", "max")).to_string())
    else:
        print(f"\n⚠ {PH_CONTROL_FILE.name} not found - run fetch_sem_data.py "
              f"to pull qc.vw_ph_control; pH columns will be empty")

    # Plating time per electrode (all steps in its vat, cumulative bug undone)
    plating = derive_step_times(vat_rows)
    plating.to_csv(folder / "plating_times.csv", index=False)
    print(f"\n✓ Plating times: {len(plating)} electrodes, "
          f"mean actual {plating['actual_time'].mean():.1f}s vs set "
          f"{plating['set_time'].mean():.1f}s, saved to plating_times.csv")

    # Rinse time per electrode, one row per rinse step. The "all steps"
    # total is summed in the GUI instead of here, because which steps count
    # depends on the checkboxes.
    rinse_rows = pd.read_csv(folder / "rinse_steps.csv")
    rinse = derive_step_times(rinse_rows, per_vat=True)
    rinse.to_csv(folder / "rinse_times.csv", index=False)
    print(f"\n✓ Rinse times: {rinse['electrode'].nunique()} electrodes, "
          f"{len(rinse)} step rows across steps "
          f"{sorted(rinse['vat'].unique(), key=int)}, "
          f"saved to rinse_times.csv")

    # Drip time per electrode, one row per drip step (same shape as rinse)
    drip_rows = pd.read_csv(folder / "drip_steps.csv")
    drip = derive_step_times(drip_rows, per_vat=True)
    drip.to_csv(folder / "drip_times.csv", index=False)
    print(f"✓ Drip times: {drip['electrode'].nunique()} electrodes, "
          f"{len(drip)} step rows across steps "
          f"{sorted(drip['vat'].unique(), key=int)}, saved to drip_times.csv")

    # Strike time per electrode (step 31, both passes, same correction)
    strike_step = pd.read_csv(folder / "strike_step.csv")
    strike_times = derive_step_times(strike_step)
    strike_times.to_csv(folder / "strike_step_times.csv", index=False)
    print(f"✓ Strike times: {len(strike_times)} electrodes, "
          f"mean actual {strike_times['actual_time'].mean():.1f}s vs set "
          f"{strike_times['set_time'].mean():.1f}s, "
          f"saved to strike_step_times.csv")

    # Chemical additions, parsed to long format and clipped to the same range
    chems = pd.read_csv(folder / "adding_chemicals.csv")
    chems["Date"] = pd.to_datetime(chems["Date"]).dt.strftime("%Y-%m-%d")
    chems = chems[(chems["Date"] >= first_day) & (chems["Date"] <= last_day)]
    chem_long = parse_chemicals(chems)
    unparsed = chem_long["amount"].isna().sum() if len(chem_long) else 0
    chem_long.to_csv(folder / "adding_chemicals_filtered.csv", index=False)
    print(f"\n✓ Chemical additions: {len(chem_long)} parsed rows from "
          f"{len(chems)} in-range log entries "
          f"({unparsed} lines unparsed, kept with blank amount), "
          f"saved to adding_chemicals_filtered.csv")

    # The same log again, unclipped and timestamped, for joining to an
    # electrode. The clipped file above feeds a plot whose x-axis is the
    # electrode window anyway; this one has to reach back further, because
    # vats 17, 18 and 31 are dosed months apart and an electrode still needs
    # to know what went in last.
    every = parse_chemicals(pd.read_csv(folder / "adding_chemicals.csv"))
    every.insert(0, "added_at", chemical_timestamps(every))
    every = (every.dropna(subset=["added_at"])
             .sort_values(["vat", "chemical", "added_at"]))
    every.to_csv(folder / CHEM_LONG_FILE, index=False)
    print(f"✓ Chemical additions, full log: {len(every)} rows, "
          f"saved to {CHEM_LONG_FILE}")

    if CONDUCTIVITY_FILE.exists():
        cond = derive_conductivity(pd.read_csv(CONDUCTIVITY_FILE))
        cond.to_csv(folder / CONDUCTIVITY_LONG_FILE, index=False)
        print(f"\n✓ Conductivity: {len(cond)} readings, saved to "
              f"{CONDUCTIVITY_LONG_FILE}")
        print(cond.groupby("position").agg(
            readings=("conductivity", "count"),
            median=("conductivity", "median"),
            first=("measured_at", "min"),
            last=("measured_at", "max")).to_string())
    else:
        print(f"\n⚠ {CONDUCTIVITY_FILE.name} not found - run "
              f"fetch_sem_data.py; conductivity columns will be empty")

    if CLEANER_BATH_FILE.exists():
        cleaner = derive_cleaner_bath(pd.read_csv(CLEANER_BATH_FILE))
        cleaner.to_csv(folder / CLEANER_BATH_LONG_FILE, index=False)
        print(f"\n✓ Cleaner bath: {len(cleaner)} rounds, "
              f"{int(cleaner['pH'].notna().sum())} with a pH, saved to "
              f"{CLEANER_BATH_LONG_FILE}")
        print(f"   pH         median {cleaner['pH'].median():.2f}  "
              f"range {cleaner['pH'].min():.2f}..{cleaner['pH'].max():.2f}  "
              f"dates {cleaner['measured_at'].min().date()}.."
              f"{cleaner['measured_at'].max().date()}")
        if "slotoclean" in cleaner.columns:
            slot = cleaner["slotoclean"]
            print(f"   Slotoclean median {slot.median():.2f}  "
                  f"range {slot.min():.2f}..{slot.max():.2f}  "
                  f"on {int(slot.notna().sum())} rounds")
    else:
        print(f"\n⚠ {CLEANER_BATH_FILE.name} not found - run "
              f"fetch_sem_data.py; the cleaner pH column will be empty")

    # The activator and strike baths again, derived but NOT clipped to the
    # electrode window the way the *_filtered.csv files above are. Both are
    # sampled every week or two, so an electrode regularly needs a reading
    # from before the window to have one at all.
    for source, derive, out_name, what in (
            (folder / "activator_bath.csv", derive_activator_bath,
             ACTIVATOR_LONG_FILE, "Activator bath"),
            (STRIKE_BATH_FILE, derive_strike_bath, STRIKE_LONG_FILE,
             "Strike bath")):
        if not source.exists():
            print(f"\n⚠ {source.name} not found - run fetch_sem_data.py; "
                  f"the {what.lower()} columns will be empty")
            continue
        frame = derive(pd.read_csv(source))
        frame.insert(0, "measured_at", log_timestamps(frame))
        frame = (frame.dropna(subset=["measured_at"])
                 .sort_values("measured_at").reset_index(drop=True))
        frame.to_csv(folder / out_name, index=False)
        print(f"\n✓ {what}: {len(frame)} readings, saved to {out_name}")
        print(f"   {frame['measured_at'].min().date()} .. "
              f"{frame['measured_at'].max().date()}")

    if WATER_FILE.exists():
        water = pd.read_csv(WATER_FILE)
        water.insert(0, "measured_at", log_timestamps(water))
        water = (water.dropna(subset=["measured_at"])
                 .sort_values("measured_at").reset_index(drop=True))
        water.to_csv(folder / WATER_LONG_FILE, index=False)
        conductivity = [c for c in water.columns if "S/cm" in c]
        print(f"\n✓ Water quality: {len(water)} readings, saved to "
              f"{WATER_LONG_FILE}")
        print(f"   {water['measured_at'].min().date()} .. "
              f"{water['measured_at'].max().date()}; "
              f"conductivity on {int(water[conductivity[0]].notna().sum())}, "
              f"pH on {int(water['pH'].notna().sum())}")
    else:
        print(f"\n⚠ {WATER_FILE.name} not found - run fetch_sem_data.py; "
              f"the water quality columns will be empty")

    # All three plate baths in one unclipped, timestamped file, with a vat
    # column so an electrode can be matched to its own.
    plates = []
    for vat in PLATE_VATS:
        source = folder / f"plate_bath_{vat}.csv"
        if not source.exists():
            continue
        frame = derive_plate_bath(pd.read_csv(source))
        frame.insert(0, "vat", str(vat))
        frame.insert(0, "measured_at", log_timestamps(frame))
        plates.append(frame.dropna(subset=["measured_at"]))
    if plates:
        # Not "every": that name holds the chemical log above, which is
        # still printed further down.
        plate_log = (pd.concat(plates, ignore_index=True)
                     .sort_values(["vat", "measured_at"])
                     .reset_index(drop=True))
        plate_log.to_csv(folder / PLATE_LONG_FILE, index=False)
        print(f"\n✓ Plate baths, full log: {len(plate_log)} readings, "
              f"saved to {PLATE_LONG_FILE}")
        print(plate_log.groupby("vat").agg(
            readings=("pH Mean", "size"), ph=("pH Mean", "count"),
            ni=("Ni2+ [g/L] Mean", "count"), co=("Co2+ [g/L] Mean", "count"),
            first=("measured_at", "min"),
            last=("measured_at", "max")).to_string())
    print(every.groupby(["vat", "chemical"]).agg(
        rows=("amount", "size"), kg=("amount", "count"),
        first=("added_at", "min"), last=("added_at", "max")).to_string())


if __name__ == "__main__":
    main()
