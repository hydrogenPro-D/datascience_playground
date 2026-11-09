# Plot Design Guide

How the charts in this repo are designed (grouped bar chart in
`cracking_analysis_2026_september/cracking_analysis.py`, lane/timeline plots in
`cracking_analysis_2026_september/gui_app.py`). Follow this exactly when
generating new plots — the rules are deliberate, not taste.

## 1. Color tokens — use these, never matplotlib defaults

Define once at the top of every plotting script:

```python
# Data colors (colorblind-validated pair; extend order: blue, orange, aqua)
COLOR_1  = "#2a78d6"   # blue   — first/positive series (e.g. not cracked)
COLOR_2  = "#eb6834"   # orange — second/negative series (e.g. cracked)
COLOR_3  = "#1baf7a"   # aqua   — third series only

# Chart chrome (light mode)
SURFACE       = "#fcfcfb"  # figure AND axes background — never pure white
INK           = "#0b0b0b"  # titles
INK_SECONDARY = "#52514e"  # axis labels, subtitles, bar labels
INK_MUTED     = "#898781"  # tick labels
GRIDLINE      = "#e1e0d9"  # hairline grid
BASELINE      = "#c3c2b7"  # the one visible axis spine
```

Rules:
- **Never green + red for two categories** — it fails for colorblind readers.
  Blue vs orange is the safe default pair.
- Color follows the *entity*: the same category keeps the same color across
  every chart in the project (here: not-cracked/good = blue, cracked = orange).
- Meaning must never ride on color alone: pair color with position (separate
  lanes/bars), direct labels, or both.
- Text is never colored in a series color — labels/values wear the ink tokens;
  the colored mark next to them carries identity.

## 2. Chrome — the same skeleton on every chart

```python
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# Only the bottom spine survives, in BASELINE; grid is hairline and behind data
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)   # bars: y-grid
ax.tick_params(colors=INK_MUTED, length=0)                     # no tick marks

# Title left-aligned, bold, in INK; context/stats line under it in INK_SECONDARY
ax.set_title("Title", color=INK, fontsize=15, fontweight="bold",
             loc="left", pad=26)
ax.text(0, 1.045, "189 electrodes · 135 good · 54 cracked (29%)",
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=10.5)
```

- One y-axis, always. Never a dual-axis chart.
- Data drawn at `zorder=3`, grid at `zorder=0` — grid never crosses marks.
- Give headroom for labels: `ax.margins(y=0.15)` or explicit `subplots_adjust`.

## 3. Grouped bar charts

- Bar width ~0.32, with a small gap between paired bars (`gap = 0.02`) so
  fills never touch.
- **Every bar gets a direct label** via `ax.bar_label` in INK_SECONDARY:
  count, and the within-group percentage on a second line when shares matter
  (`"34\n(25%)"`). Percentages are computed within each x-group so each group
  sums to 100%.
- Legend only when there are ≥ 2 series: `frameon=False`,
  `labelcolor=INK_SECONDARY` — never a boxed legend.
- Print the underlying counts table to the console too; the chart is not the
  only record.

## 4. Binary outcome over time → lane plot, not a bare 0/1 scatter

A raw scatter of 0/1 values on a big canvas is 90 % dead space. Instead:

- **Tinted lanes**: a translucent band behind each outcome row, in that
  outcome's own color at very low alpha:
  `ax.axhspan(0.72, 1.28, color=COLOR_1, alpha=0.06, zorder=0)`.
- **Stack same-x points vertically** inside the lane (offset ±0.055 per point,
  centered) instead of letting them overprint — density becomes visible.
- Dots: `s≈52`, `alpha=0.9`, with a surface-colored halo so neighbors stay
  separable: `edgecolors=SURFACE, linewidths=1.2`.
- **No legend** — the y-tick labels name each lane directly, with its n:
  `ax.set_yticks([0, 1], ["cracked (0)\nn = 54", "good (1)\nn = 135"])`.
- Time axis: weekly hairline minor grid + stronger month lines, labels like
  "Aug 2026" (`mdates.WeekdayLocator(byweekday=0)` minor,
  `MonthLocator` + `DateFormatter("%b %Y")` major).
- Clamp the y range tight around the lanes (`ax.set_ylim(-0.55, 1.55)`).
- An empty subset shows a centered muted message ("No electrodes for this
  vat"), never a bare axis.

## 5. Anti-patterns — if the output matches one of these, it is wrong

- Matplotlib default color cycle (`C0`, `C1`…) or default white background.
- Green vs red as the only difference between categories.
- Dual y-axes; 3D; pie charts for more than ~3 slices.
- A number printed on every point of a dense series (label selectively or
  aggregate instead).
- Legend box with a frame, or a legend for a single series.
- Bars/points touching with no gap or halo.
- Vast empty space because the y-range is much larger than the data band.
- Tick marks sticking out of the axis (`length=0` always).
- Rainbow or per-value hues for a magnitude — magnitude uses one hue,
  light→dark.

## 6. Output checklist

1. Colors from §1 only, meaning never color-alone.
2. Chrome from §2 (surface, spines, grid, left-aligned bold title + stats line).
3. Direct labels or lane labels present; each group's identity readable
   without the colors.
4. Look at the rendered image before shipping: no label collisions, no dead
   space, no overflow.
