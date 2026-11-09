# To discuss next session

Written 2026-09-29 at the end of the day. Four things you asked for, then
the loose ends I am carrying.

## 1. Explain the model findings again, simply

Not the method, the meaning. Plain words, no AUC-speak. Keep to:

- what the models were actually shown, and what they were asked
- what "0.80" and "0.70" mean in terms of an electrode on the line
- why the SEM columns had to be split out, said once and clearly
- what the numbers do **not** let anyone claim

## 2. The four model tabs must not fit on open

They currently fit the first time each tab is shown: about 200 s for a
forest, 25 s for the network. That has to become a load, not a fit.

Plan: a script that runs all four and writes the results to disk
(`model_results.json` or a pickle beside the data), with the tabs reading
that file and showing a date stamp. Add the script as a pipeline step so a
data refresh re-fits. Needs deciding:

- where the file lives and what it holds (scores, null distribution,
  importances, confusion matrices, feature list, fitted-on date)
- what a tab shows when the file is missing or older than the data
- whether re-fitting joins `run_pipeline.py` or stays a separate command

## 3. Confusion matrices

Wanted on each of the four tabs. Points to settle:

- at which threshold - 0.5 is arbitrary when 71 of 277 cracked, so the
  options are 0.5, the rate that matches the real 26%, or a chosen
  cost of a miss versus a false alarm
- out-of-fold predictions, not training ones
- alongside it: how many cracked electrodes the model actually catches,
  which is the number people will ask about

## 4. THE MAIN QUESTION - what causes the cracks

This is the goal; everything above is scaffolding. Where we are:

**Replicated, strong, but not a cause.** Coating Ni:S is the one thing that
keeps appearing - rho +0.419, q = 8e-11, and the forest's top feature by a
wide margin. But it is measured on the same SEM scan that produced the
verdict, so it describes a cracked coating rather than explaining it.

**The process-only forest reaching 0.703 is the real lead.** Something in
the process data does carry signal. Its top features were `chem_v18_age`,
`sig_cleaner_v`, `chem_s_nicl2_age`, `chem_tu`, `plate_ph`.

**What has been ruled out** so far: step timings and overrun (77 features,
day-grouped CV, permutation p = 0.477), drying station and dryer time,
plating vat, HMI alarms.

**What to work out together:**

- are those forest features causes, or all of them proxies for *when*?
  Several are age columns, and a drift in cracking over the autumn would
  light up every one at once
- Ni:S is downstream of the bath - so which bath condition moves Ni:S?
  Earlier work found bath Ni:TU to coating Ni:S inconsistent (vat 28
  rho +0.513, vat 30 rho 0.000), which is worth revisiting now that there
  are 277 electrodes instead of 223
- what would a deliberate trial look like? An observational table of this
  size may simply not be able to settle a cause, and saying so is a result
- is there a physical mechanism anyone on the line already suspects? That
  is worth more than another model

## Loose ends I am carrying

- `'sek'` is not in `initials_dictionary.py`; it is being rule-normalised to
  `SEK`, which is right, but it should be explicit
- `B14450` has no production date in MetaData, so it is dropped every run
- filtering ideas 4 (drop the 3 anodes) and 5 (effective sample size) were
  declined for now. 5 still matters: the activator columns' 277 electrodes
  carry 11 distinct readings, so those p-values are optimistic
- the 28 measured/not indicator columns help as diagnostics but dilute the
  false-discovery correction for every other column - possibly show them
  without testing them
- the alarm work will improve on its own as SEM catches up: the overlap went
  3 -> 56 electrodes with one refresh

## How to run things

The pipeline and anything touching the database or sklearn need the project
venv, not the system Python:

    .venv/Scripts/python.exe run_pipeline.py
