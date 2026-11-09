# HMI alarms against coating cracks

Generated 2026-09-29 10:32 from `electrode_alarms.csv`.

- **551** electrodes in the alarm-log window
- **56** of them have both a SEM coat verdict and a usable production window
- of those, **9 cracked** and **47 not**

## 1. Does the amount of alarms differ?

| measure | group | count | mean | median | min | max |
|---|---|---|---|---|---|---|
| alarm events | cracked | 9 | 56.9 | 49 | 24 | 156 |
| alarm events | not cracked | 47 | 72.9 | 48 | 24 | 315 |
| distinct alarm numbers | cracked | 9 | 12.4 | 12 | 10 | 15 |
| distinct alarm numbers | not cracked | 47 | 12.7 | 12 | 9 | 18 |

Median alarm events: **49** when cracked against **48** when not.

## 2. Does any particular alarm go with cracking?

| alarm | electrodes | cracked | % of cracked | not cracked | % of not cracked |
|---|---|---|---|---|---|
| 91 | 55 | 9 | 100% | 46 | 98% |
| 92 | 55 | 9 | 100% | 46 | 98% |
| 41 | 51 | 9 | 100% | 42 | 89% |
| 93 | 51 | 9 | 100% | 42 | 89% |
| 53 | 43 | 9 | 100% | 34 | 72% |
| 32 | 46 | 8 | 89% | 38 | 81% |
| 58 | 37 | 8 | 89% | 29 | 62% |
| 49 | 33 | 8 | 89% | 25 | 53% |
| 43 | 48 | 7 | 78% | 41 | 87% |
| 33 | 44 | 6 | 67% | 38 | 81% |
| 61 | 20 | 6 | 67% | 14 | 30% |
| 31 | 20 | 6 | 67% | 14 | 30% |
| 47 | 25 | 5 | 56% | 20 | 43% |
| 4 | 26 | 4 | 44% | 22 | 47% |
| 34 | 35 | 3 | 33% | 32 | 68% |
| 52 | 19 | 3 | 33% | 16 | 34% |
| 64 | 24 | 2 | 22% | 22 | 47% |
| 55 | 13 | 1 | 11% | 12 | 26% |
| 42 | 17 | 0 | 0% | 17 | 36% |
| 57 | 7 | 0 | 0% | 7 | 15% |
| 3 | 5 | 0 | 0% | 5 | 11% |
| 51 | 5 | 0 | 0% | 5 | 11% |
| 65 | 4 | 0 | 0% | 4 | 9% |
| 54 | 4 | 0 | 0% | 4 | 9% |
| 66 | 4 | 0 | 0% | 4 | 9% |
| 26 | 3 | 0 | 0% | 3 | 6% |
| 23 | 2 | 0 | 0% | 2 | 4% |
| 100 | 2 | 0 | 0% | 2 | 4% |
| 36 | 2 | 0 | 0% | 2 | 4% |
| 5 | 1 | 0 | 0% | 1 | 2% |
| 6 | 1 | 0 | 0% | 1 | 2% |
| 9 | 1 | 0 | 0% | 1 | 2% |
| 35 | 1 | 0 | 0% | 1 | 2% |
| 1 | 1 | 0 | 0% | 1 | 2% |
| 10 | 1 | 0 | 0% | 1 | 2% |
| 24 | 1 | 0 | 0% | 1 | 2% |
| 59 | 1 | 0 | 0% | 1 | 2% |
| 40 | 1 | 0 | 0% | 1 | 2% |
| 30 | 1 | 0 | 0% | 1 | 2% |
