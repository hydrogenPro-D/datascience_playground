"""Every spelling of an operator seen in the chemical log, made one.

The addition log is typed by hand, so the same people arrive in several
spellings: "DC" and "dc", "AMA" and "ama", "AT MA" and "MA AT". Nothing
here expands a code into a name - "MP" stays "MP" - it only makes one
spelling win, so counting or filtering by operator is possible at all.

Two rules, applied when this file was generated and kept as a fallback
for wordings that turn up later:

  * a token that was ever typed in capitals is a CODE, and is upper-cased
    wherever it appears. Read off the data, because length cannot tell
    "Roy" (a name) from "dc" (a code).
  * several people in one cell are split on & / , and "and"/"og", then
    sorted, so the order they were typed in stops mattering.

82 spellings collapse to 69; 13 of those are real merges.

A spelling that is not in INITIALS is put through the same rules rather
than dropped, and canonical() can report it so it gets added here.
"""

# Tokens ever typed in capitals, so treated as initials codes
CODES = frozenset({
    "AB", "AMA", "AT", "BB", "CM", "DC", "JM", "LH",
    "MA", "MP", "RR", "SEK",
})


# Raw value -> the one spelling to use. Grouped: merges first.
INITIALS = {
    # --- several spellings of the same people ---
    'AMA': 'AMA',                                             # 13 log entries
    'ama': 'AMA',                                             # 2 log entries
    'Anders & Andri': 'Anders & Andri',                       # 3 log entries
    'Andri & Anders': 'Anders & Andri',                       # 1 log entry
    'Anders & Mathias': 'Anders & Mathias',                   # 1 log entry
    'Mathias&Anders': 'Anders & Mathias',                     # 1 log entry
    'Andreas & Lasse': 'Andreas & Lasse',                     # 1 log entry
    'Lasse & Andreas': 'Andreas & Lasse',                     # 1 log entry
    'Andri & Lasse': 'Andri & Lasse',                         # 2 log entries
    'Andri&Lasse': 'Andri & Lasse',                           # 2 log entries
    'Roy and Andri': 'Andri & Roy',                           # 1 log entry
    'Roy/Andri': 'Andri & Roy',                               # 1 log entry
    'AT': 'AT',                                               # 3 log entries
    'at': 'AT',                                               # 1 log entry
    'AT MA': 'AT & MA',                                       # 2 log entries
    'MA AT': 'AT & MA',                                       # 1 log entry
    'Daniel & Mads': 'Daniel & Mads',                         # 2 log entries
    'Mads & Daniel': 'Daniel & Mads',                         # 3 log entries
    'Daniel & Søren': 'Daniel & Søren',                       # 1 log entry
    'Søren&Daniel': 'Daniel & Søren',                         # 1 log entry
    'DC': 'DC',                                               # 7 log entries
    'dc': 'DC',                                               # 4 log entries
    'Mads & Roy': 'Mads & Roy',                               # 1 log entry
    'Roy & Mads ': 'Mads & Roy',                              # 2 log entries
    'Richard & Roy': 'Richard & Roy',                         # 1 log entry
    'Roy&Richard': 'Richard & Roy',                           # 1 log entry

    # --- one spelling, relabelled for order or case ---
    'Andri, Andreas, Anders (And^3)': '^3) & Anders ( & Andreas & Andri',
    'AB BB MA': 'AB & BB & MA',
    'Andreas & Anders': 'Anders & Andreas',
    'Andri, Andreas, Mathias & Anders': 'Anders & Andreas & Andri & Mathias',
    'Morten, Mikkel & Anders': 'Anders & Mikkel & Morten',
    'Andri and Andreas': 'Andreas & Andri',
    'Andreas&Julia': 'Andreas & Julia',
    'Mads&Andreas&Julia&Mathias': 'Andreas & Julia & Mads & Mathias',
    'Andreas og Mads': 'Andreas & Mads',
    'Mikkel and Andreas': 'Andreas & Mikkel',
    'Daniel & Andri': 'Andri & Daniel',
    'Julia & Andri': 'Andri & Julia',
    'Mads&Andri': 'Andri & Mads',
    'Mikkel&Andri': 'Andri & Mikkel',
    'at mp': 'AT & MP',
    'Mads & Bardia': 'Bardia & Mads',
    'Roy and Bardia': 'Bardia & Roy',
    'Julia & Carmine': 'Carmine & Julia',
    'Carmine/Mathias': 'Carmine & Mathias',
    'Julia and Daniel': 'Daniel & Julia',
    'Richard & Daniel': 'Daniel & Richard',
    'Julia,Roy, Gustav & Mathias': 'Gustav & Julia & Mathias & Roy',
    'Julia&Lasse': 'Julia & Lasse',
    'Julia&Mads&Mathias': 'Julia & Mads & Mathias',
    'Julia, Roy, Mathias, Thor': 'Julia & Mathias & Roy & Thor',
    'Julia&Søren': 'Julia & Søren',
    'Julia and Tamas': 'Julia & Tamas',
    'Mikkel og Lasse ': 'Lasse & Mikkel',
    'Morten, Lasse, Søren, Mikkel': 'Lasse & Mikkel & Morten & Søren',
    'Roy, Lasse': 'Lasse & Roy',
    'Richard&Mathias': 'Mathias & Richard',
    'Mikkel&Richard': 'Mikkel & Richard',
    'Roy&Mikkel': 'Mikkel & Roy',

    # --- already canonical, listed so nothing looks forgotten ---
    'AB': 'AB',
    'All shifts': 'All shifts',
    'Anders': 'Anders',
    'Anders & Mads': 'Anders & Mads',
    'Anders & Mikkel': 'Anders & Mikkel',
    'Anders & Roy': 'Anders & Roy',
    'Anders B': 'Anders B',
    'Andreas & Mathias': 'Andreas & Mathias',
    'Andreas & Richard': 'Andreas & Richard',
    'Andri': 'Andri',
    'BB': 'BB',
    'CM': 'CM',
    'Daniel': 'Daniel',
    'JM': 'JM',
    'Julia': 'Julia',
    'Julia & Roy': 'Julia & Roy',
    'Lasse & Mathias': 'Lasse & Mathias',
    'LH': 'LH',
    'MA': 'MA',
    'Morten': 'Morten',
    'MP': 'MP',
    'RR': 'RR',
    'SEK': 'SEK',
    'sek': 'SEK',
}

import re

_SPLIT = re.compile(r"\s*(?:&|/|,|\+|\band\b|\bog\b)\s*", re.IGNORECASE)
# Spellings reported by canonical() as not in INITIALS, so a caller can list
# what needs adding instead of each one being silently improvised.
UNKNOWN = set()


def _derive(value):
    """The two rules, for a spelling INITIALS has not seen."""
    parts = []
    for part in (p.strip() for p in _SPLIT.split(value)):
        if not part:
            continue
        pieces = part.split()
        if len(pieces) > 1 and all(p.upper() in CODES for p in pieces):
            parts.extend(pieces)
        else:
            parts.append(part)
    fixed = [p.upper() if p.upper() in CODES else p for p in parts]
    return " & ".join(sorted(fixed, key=str.casefold))


def canonical(value):
    """The one spelling for this operator, or None when there is no value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    known = INITIALS.get(text)
    if known is not None:
        return known
    UNKNOWN.add(text)
    return _derive(text)
