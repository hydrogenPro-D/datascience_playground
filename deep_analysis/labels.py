"""Turn the operator's free-text SEM note into an ordinal crack severity.

The existing pipeline maps the note to a binary good/bad, which hides what the
notes actually say: on Gen4.1 almost every electrode is described as cracked to
SOME degree, and "good" means "microcracked" rather than "sound". Collapsing a
severity continuum to a threshold throws away most of the signal and puts the
operator's wording of a borderline case straight into the label.

Scale:

    0  sound          "Fine", "No cracks", "No visible cracks at 100x"
    1  very few micro "Very few microcracks"
    2  microcracks    "Microcracks", "Few microcracks", "Very few small cracks"
    3  small cracks   "Small cracks", "Less cracked", "Few cracks"
    4  cracked        "Cracked", "Large cracks", "Heavy cracks"

A base score comes from the coarsest crack word present, then a hedge in the
wording knocks it down. The hedges are deliberately few: "very few" and "only
one" are unambiguous, whereas separating "few microcracks" from "microcracks"
would invent precision the notes do not carry. `audit()` prints every distinct
note beside its score so the mapping can be read and argued with.
"""
import re

import pandas as pd

SEVERITY_NAMES = {0: "sound", 1: "very few micro", 2: "microcracks",
                  3: "small cracks", 4: "cracked"}

# Commentary that says nothing about cracking, stripped before the rules run so
# that "Cracked Vat 30" and "Cracked" score alike.
NOISE = re.compile(
    r"(vat\s*\d+|spot\s*\d+|site\s*\d+|sopot\s*\d+|at\s*100x"
    r"|\d+(st|nd|rd|th)\s+(electrode|run|dummy)"
    r"|sample (seems|looks?|look)[^,.]*|dirty sample|on sample|appears dirty"
    r"|dirty\s*/?\s*dusty sample"
    r"|(few|some) salt deposits present|white percipitate|white spots"
    r"|small clumps of carbon|short fibers are present|weird dendrites"
    r"|no dendrites|with patches|coating peeling off"
    r"|given as anode but tested as a cathode"
    r"|kind of on the edge[^,.]*|a little worse than \w+"
    r"|more cracked than \w+|should do post mortem"
    r"|visible at higher magnifications|likely random"
    r"|first electrode after (chemical )?adjustment"
    r"|close to production start[^,.]*|new fb|new thick coating"
    r"|lacking \d+ eds[^,.]*|aztecone not working"
    r"|note from bag[^,.]*|tire track pattern"
    r"|it wasn.t marked[^,.]*|cut in a random corner"
    r"|but (very high ratio|not smooth coating|high ratio)"
    r"|ratio (low|high))")

# Notes that are not a crack verdict at all
INVALID = re.compile(r"unvalid|invalid|weight%|^rcp\d+$|^[\d.]+$"
                     r"|extra folder|cutting_crack_test|lumps on the surface")

# Known-cause outliers: a batch that never got the full plating recipe is a
# different failure mode and should not train a model about normal production.
KNOWN_CAUSE = re.compile(r"did not receive full plating recipe")


def _normalise(note):
    text = str(note).lower().strip()
    text = text.replace("-", " ").replace("/", " ")
    text = (text.replace("craked", "cracked").replace("cacks", "cracks")
                .replace("micto", "micro").replace("othe ", "other ")
                .replace("microcrackes", "microcracks")
                .replace("microcacks", "microcracks")
                .replace("microcraks", "microcracks")
                .replace("micro cracks", "microcracks")
                .replace("micro crack", "microcrack")
                .replace("microcracking", "microcracks")
                .replace("microcracked", "microcracks"))
    text = NOISE.sub(" ", text)
    return re.sub(r"[\s,.;()\"']+", " ", text).strip()


def severity(note):
    """Ordinal 0-4, or None when the note carries no crack verdict."""
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None
    raw = str(note).lower()
    if INVALID.search(raw.strip()):
        return None
    text = _normalise(note)
    if not text or INVALID.search(text):
        return None

    # An explicit denial wins over any crack word elsewhere in the sentence
    if re.search(r"\bno (visible )?cracks?\b|\bundamaged\b", text):
        return 0

    has_micro = "microcrack" in text
    has_small = "small crack" in text
    bare = re.sub(r"(microcrack\w*|small crack\w*|crack like marks?)", " ", text)
    has_bare = "crack" in bare

    if not (has_micro or has_small or has_bare):
        return 0 if re.search(r"\bfine\b", text) else None

    if has_bare:
        base = 4
    elif has_small:
        base = 3
    else:
        base = 2

    # Hedges, strongest first; a crack that is mentioned never drops below 1
    if re.search(r"\bvery few\b|\bvery little\b|\bonly one\b|\bbarely\b"
                 r"|\bvery small\b", text):
        base -= 2
    elif re.search(r"\bless\b|\bslightly\b|\bsomewhat\b|\bfew\b|\bsome local\b"
                   r"|\bmostly microcrack", text) and base == 4:
        base -= 1
    return max(1, base)


def add_severity(sem):
    """Attach severity and the known-cause flag to a per-electrode frame."""
    out = sem.copy()
    out["severity"] = out["note"].map(severity)
    out["severity_name"] = out["severity"].map(SEVERITY_NAMES)
    out["known_cause"] = out["note"].fillna("").str.lower().str.contains(
        KNOWN_CAUSE)
    return out


def audit(notes):
    """Every distinct note with its score, so the mapping can be eyeballed."""
    rows = [{"note": n, "severity": severity(n),
             "name": SEVERITY_NAMES.get(severity(n), "-- unscored --"),
             "n": c} for n, c in notes.value_counts(dropna=False).items()]
    return pd.DataFrame(rows).sort_values(
        ["severity", "n"], ascending=[True, False], na_position="last")
