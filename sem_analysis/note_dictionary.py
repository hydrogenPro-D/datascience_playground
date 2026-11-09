"""Every SEM note seen in the data, mapped to a coating quality score.

The operators write the SEM verdict as free text, so the same finding
appears in dozens of spellings: vat numbers appended, typos, asides about
the sample being dirty. This maps each distinct string to one of five
quality levels, HIGHER IS BETTER:

    5  fine
    4  very few microcracks
    3  microcracks
    2  small cracks
    1  cracked

A note that carries no crack verdict at all (a folder name, a stray number,
an invalid measurement) is deliberately absent from the mapping, so it comes
back blank rather than being guessed at.

Lookup is exact on the raw string. A note that is not in here is reported by
categorise_notes.py rather than silently dropped, so new wording gets noticed
and added instead of quietly scoring blank forever.
"""

QUALITY_LEVELS = {
    5: 'fine',
    4: 'very few microcracks',
    3: 'microcracks',
    2: 'small cracks',
    1: 'cracked',
}


# Raw SEM note -> quality level. Grouped by level, commonest first.
NOTE_QUALITY = {
    # --- 5: fine (22 wordings, 177 electrodes) ---
    'Fine': 5,
    'vat 28; fine': 5,
    'vat 29; Fine': 5,
    'vat 30; Fine': 5,
    # The coat is Fine; the burn is a production note about the clamp, not a
    # crack verdict, so it does not change the grade
    'vat 29; Fine. Notes from production: Burn on the electrode on the '
    'clamp side.': 5,
    'Sample fine': 5,
    'No cracks': 5,
    'coating undamaged, some crack-like marks on it': 5,
    'Brown sample, no cracks': 5,
    'Fine Short fibers are present': 5,
    'Fine but not smooth coating': 5,
    'Fine but very high ratio': 5,
    'Fine, First electrode after chemical adjustment, Vat 30': 5,
    'Fine, but high ratio': 5,
    'Fine, close to production start with new FB and new thick coating': 5,
    'Fine, ratio low': 5,
    'No cracks, no dendrites': 5,
    'No cracks, weird dendrites': 5,
    'No visible cracks at 100x': 5,
    'Rigid and Fine': 5,
    'Unsure as it was the label was not clear No Cracks': 5,
    'Vat 28, No Cracks': 5,
    'Vat 29 No Cracks': 5,
    'Vat 30, No Cracks': 5,
    'fine': 5,

    # --- 4: very few microcracks (12 wordings, 54 electrodes) ---
    'Very few microcracks': 4,
    'Very few micro cracks': 4,
    'very few microcracks': 4,
    'VAT 28, very few microcracks on spot 1': 4,
    'Very few small and micro cracks': 4,
    'Very few barely noticeable micro cracks': 4,
    'Very few local small cracks': 4,
    'Very few micro cracks, Vat 28': 4,
    'Very few small cracks Vat 30': 4,
    'Very few small cracks and micro cracks': 4,
    'Very few small cracks, first electrode after adjustment': 4,
    'Very small cracks, Vat 28': 4,

    # --- 3: microcracks (47 wordings, 157 electrodes) ---
    'Few microcracks': 3,
    'Microcracked': 3,
    'microcracked': 3,                   # as 'Microcracked'
    'Mickrocracked': 3,                  # as 'Microcracked'
    'Few mickrocrackes': 3,              # as 'Few microcrackes'
    'Few Microcracks, Lots of Ni': 3,    # as 'Few Microcracks'; the nickel
                                         # remark is an aside, not a verdict
    'Microcracks': 3,
    'Microcraks': 3,                     # as 'Microcracks'
    'microcracks': 3,
    'Few micro cracks': 3,
    'Few Microcracks': 3,
    'Few small/micro cracks': 3,
    'Micro cracks': 3,
    'Microcracking': 3,
    'few microcracks': 3,
    'Microcracked, Vat 29': 3,
    'Microcracked, Vat 30': 3,
    'Mostly microcracks/very little cracks': 3,
    'Sample look unusual, few micro cracks': 3,
    '31/7-2026. 2nd run in new vat 28 Testing dummy electrode plated in 3,5 ph. Few Microcracks': 3,
    '3rd run in new vat 28 Testing dummy electrode plated in 3,73 ph. Few Microcracks': 3,
    'Few microcacks Vat 28': 3,
    'Few microcrackes': 3,
    'Few microcracks Vat 28': 3,
    'Few microcracks Vat 29': 3,
    'Few microcracks Vat 30': 3,
    'Few microcracks, Vat 30': 3,
    'Few microcracks, few salt deposits present': 3,
    'Few microcracks, sample seems a bit dirty': 3,
    'Few small/micro cracks Vat 28': 3,
    'Few small/micro cracks Vat 30': 3,
    'Few small/micro-cracks': 3,
    'Found only one crack on site 3, likely random': 3,
    'Micro cracks, Vat 30': 3,
    'Microcracked Vat 30': 3,
    'Microcracked, Vat 28': 3,
    'Microcracks (given as anode but tested as a cathode)': 3,
    'Microcracks Vat 28': 3,
    'Microcracks Vat 30': 3,
    'Microcracks and small clumps of carbon': 3,
    'Microcracks vat 30': 3,
    'Microcracks, few salt deposits present': 3,
    'Microcracks, some salt deposits present': 3,
    'Micto cracks': 3,
    'Small microcracks': 3,
    'Some local microcracks': 3,
    'VAT 29, few microcracks on spot 2': 3,
    'Very few cracks, appears dirty': 3,
    'White percipitate on sample, few microcracks': 3,
    "extra coupon; it wasn't marked, so sample was cut in a random corner a few microcracks": 3,
    "extra coupon; it wasn't marked, so sample was cut in a random corner few microcracks": 3,
    'microcracks 9th electrode - Vat 30': 3,

    # --- 2: small cracks (37 wordings, 99 electrodes) ---
    'Less cracked': 2,
    'Less crack': 2,                     # as 'Less cracked'
    'Small cracks': 2,
    'Few small cracks': 2,
    'Microcracks/small cracks': 2,
    'Less Cracked': 2,
    'Few cracks': 2,
    'Few small cacks': 2,
    'Slightly cracked': 2,
    'Less cracked Vat 28': 2,
    'Less cracked Vat 30': 2,
    'Microcracks and few small cracks': 2,
    'less cracked': 2,
    'small cracks': 2,
    'various small crack marks': 2,
    'Few small cracks Vat 30': 2,
    'Few small cracks, dirty sample': 2,
    'Less Cracked, Vat 29': 2,
    'Less Cracked, Vat 30': 2,
    "Less cracked (not all over) - did SEM on the rest of coupons and they're microcracked/fine": 2,
    'Less cracked Vat 29': 2,
    'Less cracked, but more cracked than B4148': 2,
    'Less cracked, but more cracked than B4149': 2,
    'Less cracking': 2,
    'Small cracks Vat 30': 2,
    'Small cracks and micro cracks': 2,
    'Small cracks visible at higher magnifications': 2,
    'Small cracks, Vat 29': 2,
    'Small cracks, some salt deposits present': 2,
    'Small cracks/microcracks (kind of on the edge, but a little worse than B14159)': 2,
    'Some small cracks and dirty/dusty sample': 2,
    'Somewhat cracked, Note from bag: ,,Electrode had "tire track" pattern"': 2,
    'less Cracked': 2,
    'less cracked Vat 30': 2,
    'less cracked and with patches': 2,
    'less cracked; Lacking 2 EDS - AZtecOne not working': 2,
    'slightly cracked': 2,
    'small cracks, and microcracking': 2,

    # --- 1: cracked (22 wordings, 113 electrodes) ---
    'Cracked': 1,
    'Cracked Vat 30': 1,
    'Cracked Vat 28': 1,
    'Cracked Vat 29': 1,
    'Cracks': 1,
    "Large cracks Batch did not receive full plating recipe, hence, it's no surprise that coat lacks adhesion.": 1,
    'CRACKED': 1,
    'Some cracks': 1,
    '1st run in new vat.28 First dummy electrode plated in 3,3 pH. Cracked': 1,
    'Cracked vat 28': 1,
    'Cracked vat 29': 1,
    'Cracked, coating peeling off': 1,
    'Cracked, new FB': 1,
    'Cracks, should do post mortem': 1,
    'Craked': 1,
    'Heavy Cracks': 1,
    'Some cracks on spot 3 other microcraks not visible at 100x in the othe spots': 1,
    'Some visible cracks (site 3)': 1,
    'Sopot 1 and 3 show some small cracks. Spot 2 is cracked': 1,
    'White spots on sample, cracks': 1,
    # Written the other way round, vat first, and "vart" is a typo for "vat"
    'vat 29; cracked': 1,
    'vat 30; CRACKED': 1,
    'vart 30; cracked': 1,
}


# Seen in the data but carrying no crack verdict, so intentionally
# unmapped. Listed here so it is clear they were considered, not missed.
NO_VERDICT = [
    'Measured in weight% -unvalid',
    'Lumps on the surface (espically close to the edges)',
    '0.09385018171583047',
    'Extra folder Gen3_cutting_crack_test',
    'RCP80',
    '0.5276924615662871',
    '0.055353410012392204',
]


def quality(note):
    """Quality level for a note, or None when it carries no verdict."""
    if note is None:
        return None
    return NOTE_QUALITY.get(str(note).strip())


def category(note):
    """Human-readable level name, or None."""
    return QUALITY_LEVELS.get(quality(note))
