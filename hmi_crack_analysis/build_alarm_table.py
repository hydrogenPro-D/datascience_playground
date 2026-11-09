"""One row per electrode produced while the alarm logs cover, with any
alarm that fired during its own production.

Writes electrode_alarms.csv beside this script.

An electrode's production window is taken as:

    start  the first dbo.ProductionData row for it - the moment a sensor in
           the first tank recorded something for that marking
    end    dbo.MetaData.Date, which is written when the flight bar finishes

An alarm counts for that electrode when it fired at or after the start and
at or before the end.

TIMEZONE, and this is the whole trick: the HMI writes its alarm log in local
wall-clock time, while both database tables are UTC. Checked against the
clock - the newest alarm entry read 15:33 while UTC was 13:53 and local was
15:53 - so the log is local and comparing the two raw would misplace every
alarm by two hours, which is longer than an electrode spends in most tanks.
The alarm times are converted through Europe/Oslo rather than by adding two
hours, so the March and October changeovers stay right.

Only the raises are counted, not the clears. The panel writes a negative
ALARM NR when a condition goes away (53 raised 3713 times, -53 cleared 3706),
so counting both would double everything and imply twice as many events.
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

FOLDER = Path(__file__).resolve().parent
ALARM_FOLDER = FOLDER / "alarmer_files"
OUTPUT = FOLDER / "electrode_alarms.csv"
# The DB helpers and the SEM verdicts live with the cracking analysis
ANALYSIS = FOLDER.parent / "cracking_analysis_2026_september"
SEM_FILE = ANALYSIS / "sem_measurements_filtered.csv"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
from fetch_sem_data import connect_database, load_config   # noqa: E402

PLANT_TZ = ZoneInfo("Europe/Oslo")
LINE_PREFIX = "B"


def read_alarms():
    """Every alarm row from every downloaded log, timestamped in UTC."""
    files = sorted(ALARM_FOLDER.glob("*.CSV"))
    if not files:
        raise SystemExit(f"no alarm logs in {ALARM_FOLDER} - run fetch.py")
    frames = []
    for path in files:
        part = pd.read_csv(path)
        part["source"] = path.name
        frames.append(part)
    alarms = pd.concat(frames, ignore_index=True)
    alarms.columns = [c.strip() for c in alarms.columns]

    local = pd.to_datetime(alarms["Date"].astype(str).str.strip() + " "
                           + alarms["Time"].astype(str).str.strip(),
                           format="%Y/%m/%d %H:%M:%S", errors="coerce")
    unparsed = int(local.isna().sum())
    if unparsed:
        print(f"⚠ {unparsed} alarm rows had an unreadable timestamp")
    alarms["at_local"] = local
    # Local wall clock -> UTC, the clock both database tables use
    alarms["at"] = (local.dt.tz_localize(PLANT_TZ, nonexistent="shift_forward",
                                         ambiguous=True)
                    .dt.tz_convert("UTC").dt.tz_localize(None))
    alarms["number"] = pd.to_numeric(alarms["ALARM NR"], errors="coerce")
    alarms = alarms.dropna(subset=["at", "number"])
    print(f"{len(alarms):,} alarm rows from {len(files)} files, "
          f"{alarms['at_local'].min()} .. {alarms['at_local'].max()} local")

    # A negative number is the same alarm clearing, not a new event
    raises = alarms[alarms["number"] > 0].copy()
    raises["number"] = raises["number"].astype(int)
    print(f"  {len(raises):,} raises, "
          f"{len(alarms) - len(raises):,} clears (ignored), "
          f"{raises['number'].nunique()} distinct alarm numbers")
    return raises.sort_values("at")


def read_electrodes(engine, first, last):
    """Every B-line electrode finished inside the window, with its window."""
    with engine.connect() as conn:
        meta = pd.read_sql(text("""
            SELECT Marking AS electrode, Date AS finished
            FROM dbo.MetaData
            WHERE Date >= :lo AND Date <= :hi
              AND Marking LIKE :prefix
            ORDER BY Date"""), conn,
            params={"lo": first, "hi": last, "prefix": f"{LINE_PREFIX}%"})
        meta["electrode"] = meta["electrode"].astype(str).str.strip()
        meta = meta.drop_duplicates("electrode")
        print(f"{len(meta)} electrodes finished in the window")

        # The first sensor reading is the start of production. Asked for the
        # same markings rather than a date range, so an electrode that
        # started before the window still gets its true start.
        markings = ", ".join(f"'{m}'" for m in meta["electrode"])
        started = pd.read_sql(text(f"""
            SELECT Unik_nummer AS electrode, MIN(Aktuel_tid) AS started
            FROM dbo.ProductionData WITH (INDEX(IX_ProductionData_UnikNummer))
            WHERE Unik_nummer IN ({markings})
            GROUP BY Unik_nummer"""), conn)
    started["electrode"] = started["electrode"].astype(str).str.strip()

    table = meta.merge(started, on="electrode", how="left")
    for column in ("finished", "started"):
        table[column] = pd.to_datetime(table[column], errors="coerce",
                                       utc=True).dt.tz_localize(None)
    missing = int(table["started"].isna().sum())
    if missing:
        print(f"⚠ {missing} electrodes have no ProductionData row, so no "
              f"start time and no alarm window")
    # A start after the finish is a logging inconsistency, not a window
    backwards = table["started"].notna() & (table["started"]
                                            > table["finished"])
    if backwards.any():
        print(f"⚠ {int(backwards.sum())} electrodes start after they finish; "
              f"their windows are left blank")
        table.loc[backwards, "started"] = pd.NaT
    return table


def read_verdicts():
    """The SEM coat verdict per electrode, in three forms."""
    if not SEM_FILE.exists():
        print(f"⚠ {SEM_FILE.name} not found - the coat columns will be empty")
        return pd.DataFrame(columns=["electrode", "coat_integrity",
                                     "coat_cracked", "coating_label"])
    sem = pd.read_csv(SEM_FILE)
    sem["electrode"] = sem["electrode"].astype(str).str.strip()
    verdict = sem["sem_result"].map({"good": 0, "bad": 1})
    out = pd.DataFrame({
        "electrode": sem["electrode"],
        # The operator's categorised wording: Cracked, Microcracks, ...
        "coat_integrity": sem.get("note_corrected"),
        # The same verdict as two categories, and as the 1/0 the analysis uses
        "coat_cracked": verdict.map({1: "cracked", 0: "not cracked"}),
        "coating_label": verdict,
    })
    return out.dropna(subset=["electrode"]).drop_duplicates("electrode")


def alarms_within(table, raises):
    """Alarms that fired inside each electrode's own window.

    Three separate things, because one number cannot say all of it: an
    electrode can see 24 alarm EVENTS made up of only 10 distinct alarm
    NUMBERS, because one alarm often fires repeatedly - 53 alone accounted
    for 8 of those 24. alarm_breakdown spells that out per number."""
    times = raises["at"].to_numpy()
    numbers = raises["number"].to_numpy()
    events, kinds, listed, detail, firsts = [], [], [], [], []
    for started, finished in zip(table["started"], table["finished"]):
        if pd.isna(started) or pd.isna(finished):
            events.append(None)
            kinds.append(None)
            listed.append(None)
            detail.append(None)
            firsts.append(pd.NaT)
            continue
        inside = (times >= started.to_datetime64()) & \
                 (times <= finished.to_datetime64())
        hit = numbers[inside]
        seen = pd.Series(hit).value_counts().sort_index() if len(hit) \
            else pd.Series(dtype=int)
        events.append(int(len(hit)))
        kinds.append(int(len(seen)))
        listed.append("; ".join(str(int(n)) for n in seen.index))
        detail.append("; ".join(f"{int(n)} x{int(c)}"
                                for n, c in seen.items()))
        firsts.append(times[inside][0] if len(hit) else pd.NaT)
    table = table.copy()
    table["alarm_events"] = events        # every raise, repeats included
    table["alarm_types"] = kinds          # how many distinct numbers
    table["alarm_numbers"] = listed       # which numbers, once each
    table["alarm_breakdown"] = detail     # "53 x8; 91 x3; ..."
    table["first_alarm_at"] = firsts
    return table


def main():
    raises = read_alarms()
    # The window the logs actually cover, in UTC, so an electrode is only
    # asked about if there were alarms being recorded at the time
    first, last = raises["at"].min(), raises["at"].max()
    print(f"  window in UTC: {first} .. {last}\n")

    engine = connect_database(load_config())
    table = read_electrodes(engine, first, last)
    table = alarms_within(table, raises)
    table = table.merge(read_verdicts(), on="electrode", how="left")

    table["production_minutes"] = ((table["finished"] - table["started"])
                                   .dt.total_seconds() / 60).round(1)
    table = table[["electrode", "started", "finished", "production_minutes",
                   "coat_integrity", "coat_cracked", "coating_label",
                   "alarm_events", "alarm_types", "alarm_numbers",
                   "alarm_breakdown", "first_alarm_at"]]
    table = table.sort_values("finished").reset_index(drop=True)
    table.to_csv(OUTPUT, index=False)

    print(f"\n✓ {len(table)} electrodes -> {OUTPUT.name}")
    windowed = table["alarm_events"].notna()
    print(f"  {int(windowed.sum())} with a usable window; of those "
          f"{int((table.loc[windowed, 'alarm_events'] > 0).sum())} saw at "
          f"least one alarm")
    graded = table["coating_label"].notna()
    print(f"  {int(graded.sum())} have a SEM coat verdict")
    if int(graded.sum()) < len(table):
        print(f"  ⚠ {len(table) - int(graded.sum())} have none: the SEM data "
              f"ends before the alarm logs begin, so the coat columns are "
              f"blank for almost every row")
    if windowed.any():
        print("\nalarm events per electrode (repeats included):")
        print(table.loc[windowed, "alarm_events"]
              .describe(percentiles=[.5, .9]).round(1).to_string())
        print("\ndistinct alarm numbers per electrode:")
        print(table.loc[windowed, "alarm_types"]
              .describe(percentiles=[.5, .9]).round(1).to_string())
        top = (table.loc[windowed, "alarm_numbers"].str.split("; ")
               .explode().replace("", pd.NA).dropna().value_counts().head(10))
        print("\ncommonest alarm numbers during production:")
        print(top.to_string())


if __name__ == "__main__":
    main()
