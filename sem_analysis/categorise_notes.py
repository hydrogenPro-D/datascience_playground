"""Attach a coating quality score to every electrode, from its SEM note.

Reads sem_raw.csv (one row per electrode / position / point / element),
reduces it to one row per electrode, looks the note up in note_dictionary.py
and writes sem_notes_categorised.csv.

Quality runs 1 (cracked) to 5 (fine); higher is better. A note the dictionary
does not recognise, and an electrode with no note at all, both come out blank
rather than guessed. Anything unrecognised is printed at the end so the new
wording can be added to the dictionary rather than silently scoring blank.
"""
from pathlib import Path

import pandas as pd

import note_dictionary as nd

FOLDER = Path(__file__).resolve().parent
SEM_FILE = FOLDER / "sem_raw.csv"
OUTPUT = FOLDER / "sem_notes_categorised.csv"


def main():
    if not SEM_FILE.exists():
        raise SystemExit(f"{SEM_FILE.name} not found - run fetch_sem.py first")
    sem = pd.read_csv(SEM_FILE)
    sem["electrode"] = sem["electrode"].astype(str).str.strip()

    table = sem.groupby("electrode", as_index=False).agg(
        generation=("generation", "first"),
        electrode_type=("electrode_type", "first"),
        measured_on=("measured_on", "first"),
        note=("note", "first"))

    table["quality"] = table["note"].map(nd.quality).astype("Int64")
    table["category"] = table["note"].map(nd.category)
    table = table.sort_values(["generation", "electrode_type", "electrode"])
    table.to_csv(OUTPUT, index=False)

    scored = int(table["quality"].notna().sum())
    print(f"✓ {len(table)} electrodes -> {OUTPUT.name}")
    print(f"  {scored} scored, {len(table) - scored} blank")
    print()
    print("quality distribution (5 = fine, 1 = cracked):")
    counts = table.groupby(["quality", "category"], dropna=False).size()
    print(counts.to_string())
    print()
    print("by generation and type:")
    print(pd.crosstab([table["generation"], table["electrode_type"]],
                      table["quality"], dropna=False).to_string())

    # A note present in the data but missing from the dictionary would score
    # blank forever if it were not surfaced here
    unmapped = table[table["note"].notna() & table["quality"].isna()]
    known = set(nd.NO_VERDICT)
    surprises = sorted(set(unmapped["note"]) - known)
    if surprises:
        print(f"\n⚠ {len(surprises)} note wordings are not in the dictionary "
              f"and scored blank. Add them to note_dictionary.py:")
        for note in surprises:
            print(f"    {note!r}")
    else:
        print("\nEvery note in the data is either scored or listed in "
              "NO_VERDICT; nothing fell through.")


if __name__ == "__main__":
    main()
