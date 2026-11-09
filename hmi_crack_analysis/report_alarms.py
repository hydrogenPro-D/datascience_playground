"""Do HMI alarms track cracking? Writes alarm_report.md beside this script.

Reads electrode_alarms.csv, so build_alarm_table.py has to have run first.

Two questions, in that order:

  1  do electrodes that cracked see MORE alarms than those that did not
  2  does any PARTICULAR alarm number go with cracking

The second is the interesting one, because the first cannot really work: an
alarm fires during essentially every production, so "had an alarm" has no
contrast to offer. What might have contrast is which alarm.

Each alarm number is a 2x2 table - fired or not, against cracked or not -
tested with Fisher's exact test. Exact rather than chi-square because the
cells are tiny: there are single-figure numbers of cracked electrodes, which
is exactly where the chi-square approximation is least trustworthy. Fisher
needs no approximation at any cell size.

Then Benjamini-Hochberg across the alarm numbers tested, because testing
thirty of them will throw up one or two at p < 0.05 with nothing real behind
it.
"""
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parent
TABLE = FOLDER / "electrode_alarms.csv"
OUTPUT = FOLDER / "alarm_report.md"
# An alarm has to fire for at least this many, and be absent for at least
# this many, or there is no comparison to make
MIN_EACH_SIDE = 3


def fisher_exact(a, b, c, d):
    """Two-sided p for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the same margins
    that is no more likely than the one observed. No approximation, which is
    what makes it usable when a cell holds two electrodes."""
    rows = (a + b, c + d)
    cols = (a + c, b + d)
    total = a + b + c + d
    if total == 0 or min(rows) == 0 or min(cols) == 0:
        return 1.0

    def probability(x):
        return (math.comb(rows[0], x) * math.comb(rows[1], cols[0] - x)
                / math.comb(total, cols[0]))

    observed = probability(a)
    low = max(0, cols[0] - rows[1])
    high = min(rows[0], cols[0])
    # 1e-12 so a table of identical probability is not excluded by rounding
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed + 1e-12))


def benjamini_hochberg(pvalues):
    """Expected false-discovery share at each p, aligned to the input."""
    present = [(i, float(p)) for i, p in enumerate(pvalues)
               if p is not None and not pd.isna(p)]
    out = [None] * len(pvalues)
    if not present:
        return out
    present.sort(key=lambda pair: pair[1])
    total = len(present)
    running = 1.0
    for rank in range(total, 0, -1):
        index, p = present[rank - 1]
        running = min(running, p * total / rank)
        out[index] = running
    return out


def load():
    if not TABLE.exists():
        raise SystemExit(f"{TABLE.name} not found - run build_alarm_table.py")
    table = pd.read_csv(TABLE)
    graded = table[table["coating_label"].notna()
                   & table["alarm_events"].notna()].copy()
    graded["coating_label"] = graded["coating_label"].astype(int)
    return table, graded


def volume_table(graded):
    """Alarm counts split by outcome - question 1."""
    rows = []
    for column, label in (("alarm_events", "alarm events"),
                          ("alarm_types", "distinct alarm numbers")):
        for outcome in ("cracked", "not cracked"):
            part = graded.loc[graded["coat_cracked"] == outcome, column]
            if not len(part):
                continue
            rows.append({"measure": label, "group": outcome,
                         "count": len(part), "mean": round(part.mean(), 1),
                         "median": round(part.median(), 1),
                         "min": int(part.min()), "max": int(part.max())})
    return pd.DataFrame(rows)


def per_alarm(graded):
    """Every alarm number as fired / not fired against cracked / not."""
    seen = {}
    for numbers, cracked in zip(graded["alarm_numbers"].fillna(""),
                                graded["coating_label"]):
        present = {int(n) for n in str(numbers).replace(" ", "").split(";")
                   if n}
        for number in present:
            hit = seen.setdefault(number, {"fired_cracked": 0,
                                           "fired_sound": 0})
            hit["fired_cracked" if cracked else "fired_sound"] += 1

    cracked_total = int(graded["coating_label"].sum())
    sound_total = len(graded) - cracked_total
    rows = []
    for number, hit in seen.items():
        a, b = hit["fired_cracked"], hit["fired_sound"]
        c, d = cracked_total - a, sound_total - b
        fired, quiet = a + b, c + d
        if min(fired, quiet) < MIN_EACH_SIDE:
            note = (f"fired for {fired} of {len(graded)}, "
                    f"no contrast to test")
            p = None
        else:
            note = ""
            p = fisher_exact(a, b, c, d)
        rows.append({
            "alarm": number, "electrodes": fired,
            "cracked": a,
            # Share of ALL cracked electrodes that saw this alarm, not the
            # share of its own electrodes that cracked. Alarm 91 fires for
            # every cracked electrode, so it reads 100% - and 98% for the
            # sound ones too, which is what shows it separates nothing.
            "% of cracked": (f"{100 * a / cracked_total:.0f}%"
                             if cracked_total else ""),
            "not cracked": b,
            "% of not cracked": (f"{100 * b / sound_total:.0f}%"
                                 if sound_total else ""),
            # Kept out of the table but used for the verdict underneath it
            "rate_fired": round(100 * a / fired, 1) if fired else None,
            "rate_quiet": round(100 * c / quiet, 1) if quiet else None,
            "p": p, "note": note})
    out = pd.DataFrame(rows)
    out["q"] = benjamini_hochberg(list(out["p"]))
    # Biggest gap in cracked rate first, untestable ones last
    out["gap"] = (out["rate_fired"].fillna(0)
                  - out["rate_quiet"].fillna(0)).abs()
    out.loc[out["p"].isna(), "gap"] = -1
    # Most cracked electrodes first, which is the column being read
    return out.sort_values(["cracked", "electrodes"], ascending=False)


def as_markdown(frame, columns=None):
    frame = frame if columns is None else frame[columns]
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = []
    for _index, row in frame.iterrows():
        cells = []
        for value in row:
            if value is None or (not isinstance(value, str)
                                 and pd.isna(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}" if 0 < value < 0.001
                             else f"{value:g}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def main():
    table, graded = load()
    cracked = int(graded["coating_label"].sum())
    sound = len(graded) - cracked
    alarms = per_alarm(graded)
    tested = alarms[alarms["p"].notna()]

    lines = [
        "# HMI alarms against coating cracks",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{TABLE.name}`.",
        "",
        f"- **{len(table)}** electrodes in the alarm-log window",
        f"- **{len(graded)}** of them have both a SEM coat verdict and a "
        f"usable production window",
        f"- of those, **{cracked} cracked** and **{sound} not**",
        "",
        "## 1. Does the amount of alarms differ?",
        "",
        as_markdown(volume_table(graded)),
        "",
    ]

    events = graded.groupby("coat_cracked")["alarm_events"].median()
    if len(events) == 2:
        lines += [
            f"Median alarm events: **{events.get('cracked', float('nan')):.0f}"
            f"** when cracked against "
            f"**{events.get('not cracked', float('nan')):.0f}** when not.",
            "",
        ]

    lines += [
        "## 2. Does any particular alarm go with cracking?",
        "",
        as_markdown(alarms, ["alarm", "electrodes", "cracked",
                             "% of cracked", "not cracked",
                             "% of not cracked"]),
        "",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ {OUTPUT.name}: {len(graded)} graded electrodes, "
          f"{cracked} cracked, {len(alarms)} alarm numbers seen, "
          f"{len(tested)} testable")
    if len(tested):
        survivors = tested[tested["q"] <= 0.05]
        # Sorted explicitly: the table itself is ordered by cracked count,
        # so its first row is not the widest gap
        best = tested.sort_values("gap", ascending=False).iloc[0]
        print(f"  widest gap: alarm {int(best['alarm'])} - "
              f"{best['rate_fired']}% cracked when it fired vs "
              f"{best['rate_quiet']}% when it did not, "
              f"p={best['p']:.3f}, q={best['q']:.3f}")
        print(f"  {len(survivors)} of {len(tested)} alarm numbers clear a 5% "
              f"false-discovery rate (Fisher exact, Benjamini-Hochberg)")


if __name__ == "__main__":
    main()
