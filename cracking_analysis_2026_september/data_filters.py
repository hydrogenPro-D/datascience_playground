"""The filtered view of the all-data table.

Three changes, none of which removes an electrode:

  1  exact duplicate columns are dropped. Three pairs carry the same number
     twice - Decasel is [H+] x 120.06, 30% HCl is 36% x 1.2, and the
     Slotoclean age is the cleaner pH age because both come off the same lab
     round. Their rank correlation is 1.00000, so one of each is pure
     repetition. Dropping them loses nothing and buys power: the
     false-discovery correction is charged per column tested.

  2  a reading older than three days is blanked - but only for the
     measurements that are actually taken that often. Applied to everything
     it would not clean the table, it would delete columns: NiSO4 is DOSED
     about every three weeks, so a nineteen-day-old figure is the dosing
     rhythm rather than a stale reading, and a three-day rule leaves 21 of
     276 electrodes with a value. The slow ones keep their age column, which
     says plainly how old each figure is.

  3  an indicator column is added for every patchy measurement, because
     whether something was measured turns out to predict cracking better
     than some of the measurements: an electrode with no plate-Ni reading
     cracks at 6.5% against 28% for one that has it. That is calendar, not
     chemistry - those measurements started and stopped at particular times -
     and a 0/1 column puts it on the table to be tested rather than leaving
     it to leak in unseen.

Nothing here drops a row. The electrode count is the same before and after,
so a column's n only falls where that column itself is missing or stale.
"""
import pandas as pd

# Rank correlation 1.00000 with the column kept. The kept one is the figure
# people actually quote; the dropped one is a fixed rescale of it.
DUPLICATES = {
    "act_h": "act_decasel",          # [H+] x 120.06 = Decasel g/L
    "stk_hcl30": "stk_hcl36",        # 36% x 1.2 = 30%
    "slotoclean_age": "cleaner_ph_age",   # same lab round, same age
}

MAX_AGE_DAYS = 3
# age column -> the value columns it governs. Only the measurements taken
# often enough for three days to mean "stale" rather than "normal".
AGE_CAPPED = {
    "cond_age": ["cond14", "cond13", "cond12", "cond19", "cond20", "cond25",
                 "cond24", "cond23", "cond22", "cond21"],
    "plate_age": ["plate_ph", "plate_ni", "plate_co", "plate_cl", "plate_tu",
                  "plate_ba"],
    "ph_age": ["ph"],
    "dose_age": ["dose", "chemical"],
    "cleaner_ph_age": ["cleaner_ph"],
    "slotoclean_age": ["slotoclean"],
    "act_age": ["act_h", "act_decasel", "act_ni"],
    "stk_age": ["stk_ph", "stk_ni", "stk_hcl36", "stk_hcl30"],
    "water_age": ["water_cond", "water_ph"],
    "chem_tu_age": ["chem_tu"],
    "chem_ba_age": ["chem_ba"],
    "chem_co_age": ["chem_co"],
}
# Left uncapped on purpose, with a note of how often they are really dosed.
# Their age columns stay, so staleness is still visible per electrode.
AGE_UNCAPPED = {
    "chem_v17_age": "Slotoclean, dosed about every 8 days",
    "chem_v18_age": "Decasel 5, about every 5 days",
    "chem_niso4_age": "NiSO4, about every 19 days",
    "chem_nicl2_age": "NiCl2, about every 19 days",
    "chem_hcl_age": "strike HCl, about every 8 days",
    "chem_s_nicl2_age": "strike NiCl2, about every 15 days",
}
# An age given in hours rather than days
AGE_IN_HOURS = {"ph_age", "dose_age"}

# A column earns an indicator when enough electrodes lack it to compare, but
# not so many that the column is mostly absent.
INDICATOR_MIN_MISSING = 10
INDICATOR_MAX_SHARE = 0.60


def _age_in_days(frame, key):
    age = pd.to_numeric(frame.get(key), errors="coerce")
    return age / 24 if key in AGE_IN_HOURS else age


def apply_filters(frame, columns):
    """(filtered frame, filtered column spec, list of notes)."""
    frame = frame.copy()
    notes = []

    # --- 1. exact duplicates -------------------------------------------
    dropped = [key for key in DUPLICATES if key in frame.columns]
    for key in dropped:
        notes.append(f"dropped {key}: identical to {DUPLICATES[key]}")

    # --- 2. stale readings, per column ----------------------------------
    blanked = 0
    for age_key, governed in AGE_CAPPED.items():
        if age_key not in frame.columns:
            continue
        stale = _age_in_days(frame, age_key) > MAX_AGE_DAYS
        if not stale.any():
            continue
        for value_key in governed:
            if value_key not in frame.columns or value_key in dropped:
                continue
            hit = int((stale & frame[value_key].notna()).sum())
            if hit:
                frame.loc[stale, value_key] = None
                blanked += hit
    if blanked:
        notes.append(f"blanked {blanked} values older than {MAX_AGE_DAYS} "
                     f"days across {len(AGE_CAPPED)} measurements")
    notes.append(f"left {len(AGE_UNCAPPED)} slow measurements uncapped: "
                 + "; ".join(AGE_UNCAPPED.values()))

    # --- 3. indicators, measured after the blanking above ----------------
    kept = [(key, title, kind) for key, title, kind in columns
            if key not in dropped]
    indicators = []
    for key, title, kind in kept:
        if kind not in ("int", "float", "float2") or key == "coating_label":
            continue
        missing = int(frame[key].isna().sum())
        if missing < INDICATOR_MIN_MISSING or \
                missing > INDICATOR_MAX_SHARE * len(frame):
            continue
        flag = f"{key}_measured"
        frame[flag] = frame[key].notna().astype(int)
        indicators.append((flag, f"{title.splitlines()[0]}\\nmeasured 1/0",
                           "int"))
    if indicators:
        notes.append(f"added {len(indicators)} measured/not indicator "
                     f"columns")

    # The indicators sit with the verdict at the end, where they read as
    # bookkeeping about the row rather than as measurements of the coat
    tail = {"coat", "coating_label"}
    body = [c for c in kept if c[0] not in tail]
    verdict = [c for c in kept if c[0] in tail]
    return frame, body + indicators + verdict, notes
