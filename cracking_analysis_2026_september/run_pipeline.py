"""Refresh every CSV the apps read, in the order the steps depend on.

    python run_pipeline.py                 # the whole chain
    python run_pipeline.py --list          # what it would run, and why
    python run_pipeline.py --from notes    # resume after a failed step
    python run_pipeline.py steps signals   # just these, in listed order

The database is the only source of truth; the CSVs are snapshots and go
stale silently, which is how a whole batch of electrodes once went missing
from the charts. Run this rather than trusting what is on disk.

Two steps are dictionary checks, not conversions. The SEM verdict is free
text and the operators invent new wordings, so both note dictionaries are
hand-maintained: the scripts cannot learn a new phrase, they can only report
it. Any wording they do not recognise is printed at the end of this run
under NEEDS ATTENTION, and until it is added by hand that electrode carries
a blank verdict through every app downstream.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# The child scripts print check marks and warning signs, and a Windows
# console defaults to cp1252, which cannot encode either. Force UTF-8 on
# both sides rather than losing a warning to a codec error.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
SEM = HERE.parent / "sem_analysis"

# (name, folder, script, what it produces, why it runs where it does)
STEPS = [
    ("fetch", HERE, "fetch_sem_data.py",
     "sem_measurements.csv, plate_bath_28/29/30.csv, strike + activator "
     "baths, chemicals, raw signals",
     "the only step that talks to the database; everything below reads its "
     "output"),
    ("prep", HERE, "data_preprocessing.py",
     "sem_measurements_filtered.csv, plate_bath_*_filtered.csv",
     "filters to the analysed window, converts titration ml and the 445 nm "
     "absorbance to g/L, and maps each SEM note through NOTE_DICT"),
    ("steps", HERE, "export_step_times.py",
     "all_step_times_long.csv, all_step_times_wide.csv",
     "set vs actual time per process step, with the repeated passes "
     "de-accumulated"),
    ("signals", HERE, "build_signal_grid.py",
     "signals_raw_points.parquet, signals_grid.csv",
     "pulls every B-line ProductionData row (~4M) to parquet, then "
     "resamples the change-logged temperature/current/voltage onto a "
     "10 second grid so a median weights real time, not row count. Covers "
     "the cleaner, all three plating vats, the strike and all three dryers. "
     "The slowest step by far; needs the venv, for pyodbc and pyarrow"),
    ("sem", SEM, "fetch_sem.py",
     "sem_raw.csv, plate_bath.csv, electrode_context.csv",
     "the per-point EDS rows, which the filtered file above collapses away"),
    ("notes", SEM, "categorise_notes.py",
     "sem_notes_categorised.csv",
     "scores every SEM note 1-5 through note_dictionary.py and reports any "
     "wording it has never seen"),
    ("models", HERE, "fit_models.py",
     "model_results.json",
     "refits the random forest and the small network on both the full and "
     "the filtered table, with their shuffled-label nulls, and saves the "
     "scores, importances and confusion matrices so the GUI only reads "
     "them. About eight minutes; needs the venv for scikit-learn"),
]

# Lines worth repeating at the end: a new note wording is easy to miss in a
# few hundred lines of distribution tables, and silently costs an electrode
# its verdict.
FLAGS = ("not in NOTE_DICT", "not in the dictionary", "⚠")


def show():
    width = max(len(name) for name, *_rest in STEPS)
    for name, folder, script, makes, why in STEPS:
        where = folder.name
        print(f"  {name:<{width}}  {where}/{script}")
        print(f"  {'':<{width}}  makes : {makes}")
        print(f"  {'':<{width}}  why   : {why}\n")


def run(step):
    name, folder, script, _makes, _why = step
    path = folder / script
    if not path.exists():
        print(f"✗ {name}: {path} not found")
        return None
    print(f"\n{'=' * 72}\n  {name}  ->  {folder.name}/{script}\n{'=' * 72}")
    started = time.time()
    # cwd is the script's own folder: the sem_analysis scripts import
    # note_dictionary as a plain module and will not find it otherwise.
    result = subprocess.run([sys.executable, script], cwd=folder,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            env=CHILD_ENV)
    output = (result.stdout or "") + (result.stderr or "")
    print(output.rstrip())
    print(f"\n[{name}: {'ok' if result.returncode == 0 else 'FAILED'} "
          f"in {time.time() - started:.1f}s]")
    if result.returncode != 0:
        return None
    # A warning is a header line followed by the wordings themselves,
    # indented; repeating the header alone would hide the useful half.
    lines, flagged, collecting = output.splitlines(), [], False
    for line in lines:
        if any(flag in line for flag in FLAGS):
            flagged.append(line)
            collecting = True
        elif collecting and line.startswith("    ") and line.strip():
            flagged.append(line)
        else:
            collecting = False
    return flagged


def main():
    parser = argparse.ArgumentParser(
        description="Refresh the CSVs behind gui_app.py and gui_all_data.py")
    parser.add_argument("only", nargs="*", help="run just these steps")
    parser.add_argument("--list", action="store_true",
                        help="show the steps and what each one makes")
    parser.add_argument("--from", dest="start", metavar="STEP",
                        help="start at this step and run the rest")
    args = parser.parse_args()

    if args.list:
        show()
        return 0

    names = [name for name, *_rest in STEPS]
    chosen = STEPS
    if args.start:
        if args.start not in names:
            raise SystemExit(f"unknown step {args.start!r}; have {names}")
        chosen = STEPS[names.index(args.start):]
    if args.only:
        unknown = [name for name in args.only if name not in names]
        if unknown:
            raise SystemExit(f"unknown step(s) {unknown}; have {names}")
        chosen = [step for step in chosen if step[0] in args.only]

    attention, done = [], []
    for step in chosen:
        flagged = run(step)
        if flagged is None:
            print(f"\n✗ stopped at '{step[0]}'. Fix it, then resume with:\n"
                  f"    python run_pipeline.py --from {step[0]}")
            return 1
        done.append(step[0])
        attention += [(step[0], line) for line in flagged]

    print(f"\n{'=' * 72}\n  {len(done)} steps ok: {', '.join(done)}\n"
          f"{'=' * 72}")
    if attention:
        print("\nNEEDS ATTENTION - the dictionaries do not cover the data:\n")
        for name, line in attention:
            print(f"  [{name}] {line.strip()}")
        print("\n  Add the wordings above by hand, then re-run:\n"
              "    NOTE_DICT in data_preprocessing.py  (the coat verdict)\n"
              "    NOTE_QUALITY in sem_analysis/note_dictionary.py  (1-5)\n"
              "  A wording left out is an electrode with no verdict in every"
              " app.")
    else:
        print("\nEvery SEM note was recognised; no dictionary edits needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
