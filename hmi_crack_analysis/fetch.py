"""Download the HMI alarm logs from the panel at 192.168.1.15.

    python fetch.py                  # the last 30 days
    python fetch.py --days 120       # the last 120 days
    python fetch.py --from 2026-09-01 --to 2026-09-28
    python fetch.py --force          # re-download even what is already here

The panel names one file per day, YYMMDD + "00" + ".CSV", so the file list
can be generated from a date range instead of scraped:

    26092800.CSV  ->  2026-09-28
    26091500.CSV  ->  2026-09-15

That matters, because there is nothing to scrape. The page that lists the
files is drawn by the panel's own javascript, so the HTML the server sends
carries no <a href> rows at all, and every plausible directory URL answers
404 - /ALARMER/, /files/Logs/ALARMER/ and a dozen others were tried. Only
the file URLs themselves respond, and they respond to a plain GET:

    http://192.168.1.15/files/Logs/ALARMER/26092800.CSV?session=null

Generating the names sidesteps the javascript, the session token and the
listing page in one go. A day with no log simply answers 404, which is an
absence rather than a failure.

The panel only keeps about a fortnight, so this folder is the only lasting
copy and nothing here ever deletes: a day the panel has already rolled off
answers 404, which is reported as "no log" and leaves the local file alone.

For the same reason a day already downloaded is still fetched again and the
LARGER copy kept. Skipping on "the file exists" looks cheaper but cannot
repair a file that was saved truncated - and once the panel rolls that day
off, the broken copy is all there is. The files are tens of kilobytes over a
local network, so fetching them again costs nothing worth saving.

Standard library only: neither requests nor beautifulsoup4 is installed on
this machine, and neither is needed.
"""
import argparse
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "http://192.168.1.15/files/Logs/ALARMER/"
# The panel wants the parameter but does not check it
SESSION = "null"
OUTPUT = Path(__file__).resolve().parent / "alarmer_files"
TIMEOUT = 15
# What a real log starts with. Checked before saving, so an error page or a
# login redirect is never written out under a .CSV name.
EXPECTED_HEADER = b"Date,Time"


def log_name(day):
    """The panel's filename for a day: 2026-09-28 -> 26092800.CSV"""
    return f"{day:%y%m%d}00.CSV"


def log_url(day):
    return f"{BASE}{log_name(day)}?session={SESSION}"


def fetch(day):
    """(body, note) for one day. body is None when there is nothing to save."""
    try:
        with urllib.request.urlopen(log_url(day), timeout=TIMEOUT) as answer:
            if answer.status != 200:
                return None, f"HTTP {answer.status}"
            body = answer.read()
    except urllib.error.HTTPError as problem:
        # 404 is the normal answer for a day the line did not run
        return None, ("no log" if problem.code == 404
                      else f"HTTP {problem.code}")
    except Exception as problem:                        # noqa: BLE001
        return None, f"{type(problem).__name__}: {problem}"

    if not body.startswith(EXPECTED_HEADER):
        # The old version of this script would have saved this as a .CSV
        start = body[:40].decode("utf-8", "replace").strip()
        return None, f"not a log, starts {start!r}"
    return body, f"{len(body):,} bytes"


def days_between(first, last):
    while first <= last:
        yield first
        first += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Download the HMI alarm logs by date")
    parser.add_argument("--days", type=int, default=30,
                        help="how many days back from today (default 30)")
    parser.add_argument("--from", dest="start", metavar="YYYY-MM-DD",
                        help="first day, instead of --days")
    parser.add_argument("--to", dest="end", metavar="YYYY-MM-DD",
                        help="last day (default today)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite even when the local copy is bigger")
    args = parser.parse_args()

    last = date.fromisoformat(args.end) if args.end else date.today()
    if args.start:
        first = date.fromisoformat(args.start)
    else:
        first = last - timedelta(days=args.days - 1)
    if first > last:
        raise SystemExit(f"--from {first} is after --to {last}")

    OUTPUT.mkdir(exist_ok=True)
    saved = same = kept = absent = failed = 0

    print(f"{BASE}\n{first} .. {last} -> {OUTPUT}\n")
    for day in days_between(first, last):
        target = OUTPUT / log_name(day)
        body, note = fetch(day)
        if body is None:
            # Nothing is removed here: a day the panel has rolled off simply
            # keeps whatever copy this folder already holds.
            if note == "no log":
                absent += 1
            else:
                failed += 1
                print(f"  {day}  {log_name(day)}  ! {note}")
            continue

        if target.exists():
            held = target.read_bytes()
            if held == body:
                same += 1
                continue
            if len(body) < len(held) and not args.force:
                # The alarm log only ever grows within its day, so a shorter
                # answer means a truncated response or a panel that has
                # started overwriting. Keep what we have.
                kept += 1
                print(f"  {day}  {log_name(day)}  kept local copy: panel "
                      f"offered {len(body):,} < {len(held):,} bytes")
                continue

        # Written whole, so an interrupted run cannot leave a half file
        temporary = target.with_suffix(".part")
        temporary.write_bytes(body)
        temporary.replace(target)
        saved += 1
        print(f"  {day}  {log_name(day)}  {note}")

    print(f"\n{saved} written, {same} unchanged, {kept} local copy kept, "
          f"{absent} with no log, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
