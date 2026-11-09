# dee_analyis — deep analysis of Gen4.1 cathode cracking

Fresh pull from the live database on 2026-09-14, widened to every per-electrode
column the schema holds, then analysed against an ordinal crack-severity label
instead of the old binary one.

## Headline

1. **The binary label was hiding the real result.** `Cracked` → bad and
   `Microcracks`/`Small cracks`/`Less cracked` → good is a threshold drawn on a
   severity continuum. On the ordinal scale, **98.6% of Gen4.1 cathodes show
   cracking of some kind; only 3 of 222 are sound.** Gen3.1 cathodes: 46% sound.
   The question was never "why do some Gen4.1 electrodes crack" — nearly all of
   them do, and what varies is severity.

2. **No process variable predicts severity.** 77 process features (every step's
   time, overrun, amp-hours and temperature, alarm and manual-stop time, flight
   bar, recipe, bath chemistry), day-grouped cross-validation, permutation null:
   **R² = −0.314, p = 0.48.** Indistinguishable from chance. This closes out the
   timing/overrun line of investigation.

3. **Coating composition does, and it replicates.** The S/Ni ratio measured by
   SEM tracks severity in Gen4.1 (rho = −0.344, p = 1.6e−7, n = 222) *and*
   independently in Gen3.1 (rho = −0.212, p = 0.003, n = 190). It is the only
   feature of 85 that survives FDR correction in both populations. Dose-response
   across Gen4.1 quintiles: **49% → 45% → 25% → 14% → 7% fully cracked.**
   Survives partialling out plating charge (rho = −0.305, p = 4e−6).

4. **Gen4.1 plates 4.5× the charge of Gen3.1** (851 vs 189 Ah median), over 3.4×
   the time, ~5 °C hotter, with no overlap between the groups. Thicker
   electrodeposited nickel accumulating more internal stress is the textbook
   cause of this failure mode - but see the caveat below before treating that
   as the explanation.

## What is NOT established

- **Raw element values are NOT comparable across electrodes.** The operator
  picks which elements to quantify and the readings are normalised over that
  set. Every Gen3.1 cathode was quantified with Ni+S only (their two values sum
  to ~43); every Gen4.1 cathode with 5-10 elements (summing to ~100). So the
  raw `comp_Ni` and `comp_S` columns cannot be compared between generations -
  an earlier draft of this file claimed Gen4.1 coatings hold less nickel
  (29.1 vs 34.4) and that was a protocol artefact, now retracted. The S/Ni
  RATIO is immune, because both elements carry the same normalisation: across
  Gen4.1 protocols using 5/6/7/8/9 elements its median moves only 0.271-0.283
  while raw Ni swings 28.8-31.5. Splitting Gen4.1 by protocol, the finding
  holds in each independently - 5-element subset rho = -0.251 (n=124,
  p = 0.005), 7+ element subset rho = -0.512 (n=94, p = 1.3e-7). This is why
  the ratio replicated when raw S and raw Ni did not.
- **Causality.** Composition is measured in the same SEM session that produced
  the crack note, so it is a co-measurement. A cracked coating whose cracks
  expose nickel substrate would show raised Ni and lowered S/Ni for reasons that
  have nothing to do with cause. Against that reading: the spread of Ni across
  measurement points does not rise with severity (rho = +0.046, p = 0.50), which
  is what patchy substrate exposure would produce.
- **The knob.** Bath thiourea titration does not consistently predict coating
  sulfur — positive in Gen4.1 (rho = +0.17), negative in Gen3.1 (rho = −0.17).
  We cannot yet name the setting that moves coating S/Ni.
- **The positive control was uninformative, not negative.** CoSO₄ dose vs
  coating Co came back null, but the dose columns are near-constant (Gen4.1 TU
  dose takes two values, 400 and 405), so there was no variation to detect.
- **S/Ni does not explain the generational jump.** Within each generation higher
  S/Ni means less cracking, but Gen4.1's median S/Ni (0.276) is slightly *higher*
  than Gen3.1's (0.261) while it cracks far more. The between- and within-
  generation directions disagree, so thickness and S/Ni are separate effects.
- Gen3.1 vs Gen4.1 differ in design *and* time period - that comparison is
  confounded by construction.
- **Charge is not thickness, and the thickness story is a hypothesis.** Thickness
  is charge divided by plating area, and there is no electrode area anywhere in
  this schema, so the 4.5x could be partly size rather than coating depth. The
  comparison is also two groups, not 411 electrodes: the independent unit is the
  generation, and everything else changed with it. Inside a generation the charge
  is a fixed recipe (Gen4.1 IQR 851-851 Ah, 5th-95th pct 851-872), so the flat
  within-generation correlation with severity (rho = -0.012, p = 0.86) is
  uninformative rather than contrary - there is no variation to test against.
  Measuring coating thickness on a cross-section, or obtaining the plating area,
  would settle it.

## Files

| file | what |
|---|---|
| `explore_schema.py`, `scout.py`, `scout2.py` | schema reconnaissance; output in `schema.txt` |
| `fetch_deep.py` | wide per-electrode pull (SEM, MetaData, ProcessData) |
| `fetch_baths.py` | bath chemistry, dosing, conductivity, contamination |
| `labels.py` | free-text SEM note → ordinal severity 0–4; `audit()` prints every note beside its score |
| `features.py` | joins everything into `features.csv` (778 electrodes × 270 cols). Note: `Ampere_timer` is a running total across repeated passes, like `Aktuel_tid`, and is de-accumulated here |
| `analyse.py` | the six-part analysis → `analysis_report.txt` |
| `causal_chain.py` | artefact checks and the bath→coating→cracking chain → `causal_report.txt` |
| `final_checks.py` | additive-per-charge, predictive ceiling, `sulfur_finding.png` |

Run with the repo's `.venv` (has sklearn/scipy/pyodbc); set
`PYTHONIOENCODING=utf-8` on Windows or the ✓ characters crash the console.

## Suggested next step

Cut a fresh SEM cross-section on a sample of *stored* electrodes of known S/Ni
and grade them blind, without the operator seeing the composition. That breaks
the co-measurement problem. In parallel, get the plating engineers to confirm
what sets thiourea concentration at the cathode surface — if S/Ni is real and
controllable, it is the first actionable lever this investigation has produced.

`fetch_deep.py`'s ProductionData query (per-vat timestamps, for reconstructing
air time between vats) was killed after 25 minutes — `LTRIM(RTRIM(Unik_nummer))`
in the WHERE clause defeats the index. Drop the wrapper (SQL Server ignores
trailing spaces in `IN`) before re-running it.
