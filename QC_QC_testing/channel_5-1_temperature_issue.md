# Channel 5-1 — faulty T1 temperature reading

**Found:** 2026-07-29, while plotting all QC runs in `analyze.py`.
**Scope:** temperature (`T1`) only. **Voltage on these runs is unaffected and valid.**

## Summary

The `T1` column of the `auxTemp` sheet is unusable for every run on **channel 5-1**.
Readings sit at ~296 °C or ~1420–1433 °C instead of the ~90 °C the rig actually runs at.
No other channel is affected.

The fault follows the **channel**, not the sample: the three bad runs come from three
different studies, three different cathode batches, and three different test campaigns.
That points at the measurement chain on that channel rather than anything about the
cells being tested.

## Evidence

Across all 45 runs, 388 437 T1 samples:

| band | samples | share |
|---|---|---|
| 40–120 °C (plausible) | 358 153 | 92.20 % |
| 120–250 °C | 1 241 | 0.32 % |
| 250–350 °C | 8 709 | 2.24 % |
| > 1000 °C | 20 299 | 5.23 % |

Every sample outside 40–120 °C belongs to channel 5-1.

### The three affected runs

| run | study | median T1 | max T1 | plausible samples | good↔bad transitions |
|---|---|---|---|---|---|
| IC1882_CatB13866-TR | Cathodes_QC_IC_168h | **1424.7 °C** | 1433.4 | **0 of 20 163 (0 %)** | 0 |
| IC1907_CatB13945-TR | Cathodes_QC_IC_48h | 296.2 °C | 1425.2 | 99 of 5 769 (1.7 %) | 130 |
| IC1927_CatB13951-TR | Location study 27-07-2026 | 294.6 °C | 1422.6 | 1 345 of 5 761 (23.3 %) | 1 096 |

For comparison, a healthy channel (6-4, three runs): min 59–78 °C, median ~92 °C, max ~102 °C.

### Reading of the failure

- **IC1882** never once reads plausibly and never transitions — it is pinned at the top
  of the range for all 20 163 samples. That is a rail value, the signature of an open
  (disconnected or broken) thermocouple, not a measurement.
- **IC1907** sits flat at ~296 °C with occasional excursions to ~1420 °C and rare dips to
  a believable 85 °C.
- **IC1927** flips 1 096 times between ~90–120 °C, ~296 °C and ~1420 °C — intermittent contact.

The ~296 °C level is itself suspicious: in an unrelated, healthy file the unused `T2`
column sits at a constant 281.7 °C. So on this rig ~280–300 °C looks like what a
**floating/unconnected input** reads, and ~1420 °C like the **over-range rail**.

Ordered by test date, the three runs go intermittent → mostly bad → fully open, which is
consistent with a connection degrading over time.

### Why this is not a real thermal event

- Voltage on the same three runs is completely normal: 1.63–1.73 V, sitting in the pack
  with every other channel.
- A genuine 1400 °C would have destroyed the cell; the cells finished their runs normally.
- One stray blank T1 cell also exists in IC1907 (coerced to `NaN` on load).

## What the code does about it

`analyze.py` defines:

```python
BAD_T1_CHANNELS = {"5-1"}
```

Channel 5-1 is skipped in `temperature.png` only. It is **still present in `voltage.png`**
and still present in the `runs` dict, so nothing is lost for later aggregation — filter on
`BAD_T1_CHANNELS` wherever T1 is consumed.

Result: temperature plot covers 42 runs across 19 channels; voltage plot covers all 45
runs across 20 channels.

## Recommended follow-up

1. Physically inspect the T1 thermocouple and its wiring/terminal on **channel 5-1**.
2. Treat all historical T1 data from channel 5-1 as void, not just these three runs —
   the earliest run available is already partly bad, so the fault may predate them.
3. Once repaired, re-run `analyze.py` and remove `"5-1"` from `BAD_T1_CHANNELS`.
4. Worth adding a range check (e.g. flag any T1 outside 20–150 °C) to the routine QC so
   this surfaces at acquisition time instead of at analysis time.
