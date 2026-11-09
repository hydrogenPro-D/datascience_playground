# Sampling frequency — the bad-run length approach

The second answer to "is testing every Xth electrode often enough?", the one that
is **not** the confidence-interval / margin-of-error calculation.

Both live in `old_code/Calculate Testing Confidence.ipynb`. The confidence one is
cells 20–23; **this one is cells 27–33**, under the heading
*"BAD TESTED ELECTRODES & CLOSEST GOOD NEIGHBORS ANALYSIS"*.

## The idea in one paragraph

Forget sample size and confidence entirely. Bad electrodes don't appear at
random — they come in **runs**, because whatever goes wrong on the line stays
wrong for a while. So the real question is a comparison of two lengths:
**how long is a typical bad run**, versus **how far apart are your tests**. If
your tests are closer together than the runs are long, at least one test lands
inside every run and you catch it. If a run is shorter than your test interval,
it can start and finish entirely between two tests and you never see it.

## How you did it

**1. Build the electrode timeline.** Every B-electrode in production order,
each marked `Tested` / `Released` / `Scraped`.

**2. Group bad findings into episodes.** From each scrapped electrode, walk
*backwards* until you hit a tested-and-released one, then *forwards* until you
hit another. Everything between is one bad episode. Any scrapped electrodes
found on the way join the same episode, so several bad electrodes with no good
one between them count as one run, not several. That's the
`BadElectrodeSequence` class.

**3. Measure two lengths per episode** — this is the clever part:

- **`tested_bad_length`** — from the first to the last *confirmed* bad
  electrode. What you actually know is bad.
- **`suspected_bad_length`** — from the last good electrode *before* to the
  first good one *after*, minus one. The **uncertainty window**: nothing in
  there was tested, so all of it could be bad.

The second number is the honest one. A single scrapped electrode with good
neighbours 30 apart isn't "1 bad electrode" — it's 1 confirmed bad and 29 you
know nothing about.

**4. Plot both distributions** with a histogram and a fitted normal curve.

**5. Separately, measure the actual test interval** (cell 24) — the gap between
consecutive tested electrodes, as a 30-sample running average, to see the real
sampling rhythm rather than the nominal one.

**6. Compare the two.** That comparison is the answer.

## What came out

| | mean | std dev | min | max |
|---|---|---|---|---|
| Confirmed bad run | **11.4** | 20.3 | 1 | 120 |
| Suspected bad window | **37.2** | 23.5 | 6 | 139 |

Roughly 81 bad episodes across **7 736 electrodes, of which 463 were tested
(5.99 %)** — about one test every 17 electrodes.

Read against each other: a mean uncertainty window of ~37 against a mean test
interval of ~17 means a typical bad run is long enough for a test or two to land
inside it. But the **shortest window is 6** — well under the interval — so the
short episodes are exactly the ones that can pass through untested. The answer
isn't a single yes/no, it's "adequate for typical runs, blind to short ones".

## Where the code and outputs are

| File | What it is |
|---|---|
| `old_code/Calculate Testing Confidence.ipynb`, cells 27–33 | The analysis itself |
| `bad_electrode_analysis.py` | Standalone script, simpler variant — nearest good neighbour before/after each bad electrode, no episode grouping |
| `data/bad_electrode_sequences.csv` | One row per episode, with the neighbours and distances |
| `data/bad_electrode_neighbor_analysis.csv` | Per bad electrode, distance to nearest good before/after |
| `data/electrode_bad_good_neighbors_analysis.png` | The two length distributions |
| `data/bad_sequences_statistics.png` | Min/mean/std/max summary |
| `data/distance_between_tests.png` | Actual sampling interval over time |

## One caveat if you revisit it

A **normal curve is a poor fit** for these lengths. Both are strictly positive
and heavily right-skewed — confirmed run length has mean 11.4 with std dev 20.3,
so the fitted normal puts a lot of its mass below zero, which is impossible. The
histograms are trustworthy; the fitted μ/σ overlay is not. A geometric or
exponential distribution is the natural choice for run lengths, and it would
also give you a direct "probability a run is shorter than N" — which is exactly
the number the sampling-frequency question wants.
