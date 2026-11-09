"""Cathode diameters after plating, from "Electrode and sandwich
measurements 0508.xlsx", sheet "Diameter measurements", columns H and I.

The sheet is one electrode per row. Column H is the W Ø1 reading and column I
the C Ø2 reading, both taken after plating; rows where neither was filled in
are electrodes that were measured before plating only, and are left out. Four
rows are marked ANODE in column M - two of them carry readings around 1875,
which are anode diameters sitting in the cathode columns - so those are left
out too. Serials are written both as 14200 and as C14788 in the sheet and are
normalised to the C form."""
import csv
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
BOOK = HERE / "Electrode and sandwich measurements 0508.xlsx"
SHEET = "Diameter measurements"
OUT = HERE / "cathode_diameters_after_plate.csv"
FIRST_ROW = 3           # rows 1-2 are the two header lines
COL_SERIAL, COL_D1, COL_D2, COL_NOTE = 4, 8, 9, 13


def serial(value):
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if text.endswith(".0"):
        text = text[:-2]
    return text if text.upper().startswith("C") else f"C{text}"


def main():
    wb = load_workbook(BOOK, data_only=True)
    ws = wb[SHEET]

    rows, skipped = [], []
    for r in range(FIRST_ROW, ws.max_row + 1):
        d1 = ws.cell(r, COL_D1).value
        d2 = ws.cell(r, COL_D2).value
        if d1 is None and d2 is None:
            continue
        note = " ".join(str(ws.cell(r, COL_NOTE).value or "").split())
        if note.upper() == "ANODE":
            skipped.append((r, serial(ws.cell(r, COL_SERIAL).value), d1, d2))
            continue
        rows.append({
            "serial": serial(ws.cell(r, COL_SERIAL).value),
            "cathode_d1": d1 if isinstance(d1, (int, float)) else "",
            "cathode_d2": d2 if isinstance(d2, (int, float)) else "",
        })
    wb.close()

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["serial", "cathode_d1", "cathode_d2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} electrodes -> {OUT.name}")
    for row in skipped:
        print(f"  skipped row {row[0]} ({row[1]}): marked ANODE, "
              f"readings {row[2]}/{row[3]}")


if __name__ == "__main__":
    main()
