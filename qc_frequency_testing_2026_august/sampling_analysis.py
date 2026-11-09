"""Sampling-frequency analysis: how long are bad sequences vs how often do we test?

Rebuild of the bad-sequence-length analysis described in
SAMPLING_FREQUENCY_RUN_LENGTH_APPROACH.md, as a standalone script that writes a
PDF report.

    python sampling_analysis.py [csv] [-o report.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "electrodes_from_B13862.csv"
DEFAULT_PDF = HERE / "sampling_frequency_report.pdf"

# Status vocabulary in the source CSV.
TESTED_GOOD = "Tested and Released"
TESTED_BAD = "Tested and Scrapped"
NOT_TESTED = "Not Tested"
BAD_STATUSES = {"Scrapped", TESTED_BAD}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


# --------------------------------------------------------------------------- data

def load_electrodes(path):
    """Read the CSV in production order and derive the flags the analysis needs."""
    df = pd.read_csv(path)
    df["number"] = df["marking"].str[1:].astype(int)
    df = df.sort_values("number").reset_index(drop=True)

    df["tested"] = df["status"].isin({TESTED_GOOD, TESTED_BAD})
    df["test_good"] = df["status"] == TESTED_GOOD
    df["test_bad"] = df["status"] == TESTED_BAD
    df["final_bad"] = df["status"].isin(BAD_STATUSES)
    return df


def find_qc_episodes(df):
    """Group QC failures into episodes bounded by the nearest passing tests.

    Mirrors the original: from each failed test, walk back to the previous
    passing test and forward to the next one; every failure in between belongs
    to the same episode.
    """
    test_bad = df["test_bad"].to_numpy()
    test_good = df["test_good"].to_numpy()
    marking = df["marking"].to_numpy()
    number = df["number"].to_numpy()

    episodes = []
    idx, n = 0, len(df)
    while idx < n:
        if not test_bad[idx]:
            idx += 1
            continue

        back = idx - 1
        while back >= 0 and not test_good[back]:
            back -= 1
        good_before = back if back >= 0 else None

        forward = idx + 1
        while forward < n and not test_good[forward]:
            forward += 1
        good_after = forward if forward < n else None

        lo = good_before + 1 if good_before is not None else 0
        hi = good_after - 1 if good_after is not None else n - 1
        bad_idx = [i for i in range(lo, hi + 1) if test_bad[i]]

        # Confirmed: first to last failed test. Suspected: the whole untested
        # window between the two passing tests that bracket them.
        confirmed = bad_idx[-1] - bad_idx[0] + 1
        if good_before is not None and good_after is not None:
            suspected = good_after - good_before - 1
            suspected_numbers = number[good_after] - number[good_before] - 1
        else:
            suspected = confirmed
            suspected_numbers = number[bad_idx[-1]] - number[bad_idx[0]] + 1

        episodes.append({
            "first_bad": marking[bad_idx[0]],
            "last_bad": marking[bad_idx[-1]],
            "n_failed_tests": len(bad_idx),
            "confirmed_length": confirmed,
            "suspected_length": suspected,
            "suspected_length_numbers": suspected_numbers,
            "good_before": marking[good_before] if good_before is not None else None,
            "good_after": marking[good_after] if good_after is not None else None,
            "start_idx": bad_idx[0],
            "end_idx": bad_idx[-1],
        })

        idx = good_after + 1 if good_after is not None else hi + 1

    return pd.DataFrame(episodes)


def find_status_runs(df):
    """Lengths of consecutive runs of finally-bad electrodes, whatever the source."""
    bad = df["final_bad"].to_numpy()
    runs, length = [], 0
    for flag in bad:
        if flag:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return np.array(runs)


# ----------------------------------------------------------------- run-length model

def fit_geometric(lengths):
    """MLE for a geometric run length: p = 1 / mean. Returns (p, mean)."""
    lengths = np.asarray(lengths, dtype=float)
    mean = lengths.mean()
    return (1.0 / mean if mean > 0 else np.nan), mean


def geometric_pmf(p, k):
    return p * (1 - p) ** (np.asarray(k) - 1)


def miss_probability(lengths, interval):
    """P(a bad run is missed entirely) when testing every `interval` electrodes.

    A run of length L is missed only if it fits between two consecutive tests.
    With the run starting at a uniformly random offset, that happens with
    probability max(0, interval - L) / interval.
    """
    lengths = np.asarray(lengths, dtype=float)
    if interval <= 0:
        return 0.0
    return float(np.mean(np.clip(interval - lengths, 0, None) / interval))


def miss_probability_geometric(p, interval, max_k=100000):
    """Same quantity under the fitted geometric model rather than the raw sample."""
    if interval <= 1:
        return 0.0
    k = np.arange(1, int(min(max_k, np.ceil(interval))) + 1)
    return float(np.sum(geometric_pmf(p, k) * (interval - k) / interval))


def test_intervals(df):
    """Gaps between consecutive QC tests, in dataset positions."""
    positions = np.flatnonzero(df["tested"].to_numpy())
    return np.diff(positions), positions


# ------------------------------------------------------------------------- plotting

def style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)


def new_page(title, figsize=(11.7, 8.3)):
    fig = plt.figure(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, fontsize=14, color=INK, x=0.06, ha="left", y=0.965)
    return fig


def page_summary(pdf, df, episodes):
    fig = new_page("Sampling frequency — summary")
    intervals, _ = test_intervals(df)
    n_tested = int(df["tested"].sum())

    # Percentages are over electrodes with a decided status, so they total 100.
    decided = df[df["status"] != NOT_TESTED]
    counts = decided["status"].value_counts()
    n_excluded = len(df) - len(decided)

    lines = [
        f"Number of Gen4.1 electrodes produced: {len(df)} ({df['marking'].iloc[0]} – {df['marking'].iloc[-1]})",
        "",
        f"STATUS BREAKDOWN   ({len(decided)} decided, {n_excluded} still untested)",
    ]
    lines += [f"  {status:<24}{n:>5}   {100 * n / len(decided):>5.1f}%"
              for status, n in counts.items()]
    lines += [
        "",
        "QC TESTING",
        f"  QC tests performed      {n_tested:>5}   {100 * n_tested / len(df):>5.1f}% of electrodes",
        f"    passed                {int(df['test_good'].sum()):>5}",
        f"    failed                {int(df['test_bad'].sum()):>5}",
    ]
    if len(intervals):
        lines += [
            f"  Mean interval           {intervals.mean():>5.1f} electrodes between tests",
            f"  Median interval         {np.median(intervals):>5.1f}",
            f"  Longest gap             {intervals.max():>5.0f}",
        ]

    fig.text(0.06, 0.90, "\n".join(lines), fontsize=10, family="monospace",
             color=INK_2, va="top", linespacing=1.55)

    if len(episodes):
        ax = fig.add_axes((0.14, 0.08, 0.80, 0.30))
        plot_length_summary(ax, episodes)

    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def plot_length_summary(ax, episodes):
    """Mean and ±1σ for each of the two bad-sequence length measures."""
    rows = [
        (1, "Bad sequences\n(upper boundary):", episodes["suspected_length"].to_numpy(), "#d62728"),
        (0, "Bad sequences\n(lower boundary)", episodes["confirmed_length"].to_numpy(), "#1f77b4"),
    ]

    right = 0
    for y, label, values, color in rows:
        mu, sigma = values.mean(), values.std()
        lo, hi = max(0.0, mu - sigma), mu + sigma
        right = max(right, hi, values.max())

        ax.hlines(y, lo, hi, color="#9a9a9a", linewidth=4, zorder=1)
        ax.plot([mu], [y], "o", markersize=13, color=color,
                markeredgecolor="black", markeredgewidth=1.4, zorder=3)

        ax.annotate(f"μ={mu:.2f}", (mu, y), xytext=(0, 26), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="black")
        ax.annotate(f"σ={sigma:.2f}", (hi, y), xytext=(-4, -34), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="black")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=10, fontweight="bold")
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0, np.ceil(right / 10) * 10 + 10)

    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_2)
    for tick in ax.get_yticklabels():
        tick.set_color(INK)
    ax.set_xlabel("Sequence Length", fontsize=11, fontweight="bold", color=INK)
    ax.set_title("Bad Electrode Sequence Lengths", fontsize=12,
                 fontweight="bold", color=INK, pad=10)


def page_timeline(pdf, df):
    """How far apart the QC tests actually are, across the production run."""
    fig = new_page("QC sampling intervals")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.11)

    intervals, positions = test_intervals(df)
    if len(intervals):
        window = 5
        rolling = pd.Series(intervals).rolling(window, min_periods=1).mean()
        ax.plot(positions[1:], rolling, color=BLUE, linewidth=2,
                label=f"moving mean ({window} tests)")
        ax.axhline(intervals.mean(), color=ORANGE, linestyle="--", linewidth=1.5,
                   label=f"overall mean = {intervals.mean():.1f}")
        legend = ax.legend(fontsize=9, loc="upper right", facecolor=SURFACE,
                           edgecolor=GRID, framealpha=0.95)
        for text in legend.get_texts():
            text.set_color(INK_2)
    style(ax, "Position in file", "Electrodes between QC tests")

    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def page_distribution(pdf, lengths, label, color, subtitle):
    """Histogram + fitted geometric for one set of sequence lengths."""
    fig = new_page(f"Sequence-length distribution — {label}")
    fig.text(0.06, 0.915, subtitle, fontsize=9.5, color=INK_2)
    gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.97,
                          top=0.85, bottom=0.12)

    p, mean = fit_geometric(lengths)

    ax = fig.add_subplot(gs[0])
    bins = np.arange(0.5, lengths.max() + 1.5, max(1, lengths.max() // 25))
    ax.hist(lengths, bins=bins, density=True, color=color, alpha=0.65,
            edgecolor="white", linewidth=0.6, label=f"observed (n={len(lengths)})")
    k = np.arange(1, int(lengths.max()) + 1)
    ax.plot(k, geometric_pmf(p, k), color=INK, linewidth=2,
            label=f"geometric fit (p={p:.3f}, mean={mean:.1f})")
    style(ax, "Sequence length (electrodes)", "Probability")
    legend = ax.legend(fontsize=8, loc="upper right", facecolor=SURFACE,
                       edgecolor=GRID, framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(INK_2)

    ax2 = fig.add_subplot(gs[1])
    grid = np.arange(1, int(lengths.max()) + 1)
    empirical = [(lengths <= n).mean() for n in grid]
    ax2.step(grid, empirical, where="post", color=color, linewidth=2,
             label="observed")
    ax2.plot(grid, 1 - (1 - p) ** grid, color=INK, linewidth=1.6,
             linestyle="--", label="geometric fit")
    ax2.set_ylim(0, 1.02)
    style(ax2, "N (electrodes)", "P(sequence length ≤ N)")
    legend = ax2.legend(fontsize=8, loc="lower right", facecolor=SURFACE,
                        edgecolor=GRID, framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(INK_2)

    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def page_miss_curve(pdf, series, df, max_interval=60):
    """The headline: probability a bad sequence is missed, as a function of interval."""
    fig = new_page("How often must we test?")
    fig.text(0.06, 0.915,
             "Probability a bad sequence starts and ends between two consecutive tests, "
             "so nothing catches it.",
             fontsize=9.5, color=INK_2)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12)

    intervals = np.arange(1, max_interval + 1)
    for (label, lengths, color) in series:
        curve = [miss_probability(lengths, x) for x in intervals]
        ax.plot(intervals, curve, color=color, linewidth=2,
                label=f"{label} — observed (n={len(lengths)}, mean {lengths.mean():.1f})")

        # Smooth model estimate: the observed curve is a step function off very
        # few sequences, the fitted geometric interpolates between them.
        p, _ = fit_geometric(lengths)
        model = [miss_probability_geometric(p, x) for x in intervals]
        ax.plot(intervals, model, color=color, linewidth=1.6, linestyle="--",
                alpha=0.85, label=f"{label} — geometric fit (p={p:.3f})")

    actual, _ = test_intervals(df)
    if len(actual):
        current = miss_probability(series[0][1], actual.mean())
        ax.axvline(actual.mean(), color=MUTED, linestyle="--", linewidth=1.5)
        ax.plot([actual.mean()], [current], marker="o", markersize=7,
                color=INK, zorder=5)
        ax.annotate(f"current mean interval {actual.mean():.1f}\n→ {current:.0%} of sequences missed",
                    xy=(actual.mean(), current), xytext=(actual.mean() + 2, current + 0.09),
                    fontsize=9, color=INK, va="bottom")

    for level in (0.05, 0.10):
        ax.axhline(level, color=GRID, linewidth=1)
        ax.annotate(f"{level:.0%}", xy=(max_interval, level), xytext=(-2, 3),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="right")

    ax.set_ylim(0, 1)
    style(ax, "Test every N electrodes", "P(bad sequence missed entirely)")
    legend = ax.legend(fontsize=9, loc="lower right", facecolor=SURFACE,
                       edgecolor=GRID, framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(INK_2)

    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def page_episode_table(pdf, episodes, rows_per_page=26):
    """Every QC failure episode, as a table."""
    if episodes.empty:
        return
    cols = ["first_bad", "last_bad", "n_failed_tests", "confirmed_length",
            "suspected_length", "good_before", "good_after"]
    headers = ["first fail", "last fail", "fails", "confirmed", "suspected",
               "passed before", "passed after"]

    for start in range(0, len(episodes), rows_per_page):
        chunk = episodes.iloc[start:start + rows_per_page]
        fig = new_page(f"QC failure episodes ({start + 1}–{start + len(chunk)} "
                       f"of {len(episodes)})")
        ax = fig.add_subplot(111)
        ax.axis("off")
        table = ax.table(
            cellText=[[("—" if pd.isna(v) else v) for v in row]
                      for row in chunk[cols].to_numpy()],
            colLabels=headers, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_facecolor(SURFACE if r % 2 else "#f4f3ef")
            cell.get_text().set_color(INK if r == 0 else INK_2)
            if r == 0:
                cell.get_text().set_fontweight("bold")
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)


def page_notes(pdf, df, episodes, status_runs, gaps, populations_identical=False):
    fig = new_page("How to read this, and what to watch out for")
    intervals, _ = test_intervals(df)

    notes = [
        ("Method",
         "Bad electrodes arrive in sequences, not at random, so the question is a comparison of two\n"
         "lengths: how long a bad sequence lasts, versus how far apart the QC tests are. A sequence\n"
         "is only missed if it starts and finishes between two consecutive tests."),
        ("Two lengths per episode",
         "Lower boundary = first to last failed test. Upper boundary = the whole window between the\n"
         "passing test before and the passing test after, which is the part nobody actually checked.\n"
         "The upper boundary is the honest number; the lower one understates the sequence."),
        ("Two populations",
         "QC episodes use only genuine QC results ('Tested and …'). Final-status sequences use every\n"
         "electrode ultimately marked Scrapped, including those condemned by inference rather\n"
         "than by test. Check whether they agree before treating them as separate evidence."),
        ("Geometric, not normal",
         "Sequence lengths are strictly positive and right-skewed, so a normal fit would place mass\n"
         "below zero. The geometric distribution is the natural model here and gives\n"
         "P(length ≤ N) directly."),
    ]
    caveats = [
        f"{gaps} marking numbers between {df['marking'].iloc[0]} and {df['marking'].iloc[-1]} are absent from the\n"
        "  file. Lengths are counted in file positions, so if those electrodes exist but were not\n"
        "  exported, every sequence length here is an undercount in real production terms.",
        "The CSV is not sorted by date — marking order and timestamp order disagree. This analysis\n"
        "  uses marking order as the production sequence.",
    ]
    if populations_identical:
        caveats.append(
            "The final-status bad sequences are IDENTICAL to the QC upper boundaries, electrode for\n"
            "  electrode. The scrapped ranges were evidently assigned by the same 'condemn everything\n"
            "  between two passing tests' rule, so they are not independent evidence — they restate\n"
            "  the QC result. Only one population is charted.")
    if len(episodes) < 20:
        caveats.append(
            f"Only {len(episodes)} failure episodes exist in this file. That is far too small a sample to fit a\n"
            "  distribution with confidence — the observed curve is a step function through five points,\n"
            "  and the geometric fit is an extrapolation from a single parameter. Treat every miss\n"
            "  probability here as indicative, not as a measured rate.")
    if len(intervals):
        caveats.append(
            f"Testing is not evenly spaced: gaps range from {intervals.min():.0f} to {intervals.max():.0f} electrodes.\n"
            "  The miss probabilities assume a regular interval, so real-world risk concentrates in\n"
            "  the long gaps.")

    y = 0.885
    for heading, body in notes:
        fig.text(0.06, y, heading, fontsize=10.5, color=INK, fontweight="bold")
        fig.text(0.06, y - 0.028, body, fontsize=9, color=INK_2, va="top", linespacing=1.5)
        y -= 0.028 + 0.038 * (body.count("\n") + 1)

    fig.text(0.06, y - 0.01, "Caveats", fontsize=10.5, color="#d03b3b", fontweight="bold")
    y -= 0.045
    for caveat in caveats:
        fig.text(0.06, y, f"• {caveat}", fontsize=9, color=INK_2, va="top", linespacing=1.5)
        y -= 0.036 * (caveat.count("\n") + 1) + 0.012

    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV), help="electrode CSV")
    ap.add_argument("-o", "--out", default=str(DEFAULT_PDF), help="output PDF")
    ap.add_argument("--max-interval", type=int, default=60,
                    help="widest test interval on the miss-probability curve")
    args = ap.parse_args()

    csv_path, pdf_path = Path(args.csv), Path(args.out)
    df = load_electrodes(csv_path)
    gaps = int(df["number"].iloc[-1] - df["number"].iloc[0] + 1 - len(df))

    episodes = find_qc_episodes(df)
    status_runs = find_status_runs(df)

    # If the scrapped ranges were assigned by the same "everything between two
    # passing tests" rule, the two populations are one and the same — say so
    # rather than drawing the identical curve twice.
    suspected = episodes["suspected_length"].to_numpy() if len(episodes) else np.array([])
    populations_identical = (
        len(suspected) == len(status_runs)
        and len(suspected) > 0
        and np.array_equal(np.sort(suspected), np.sort(status_runs))
    )

    series = []
    if len(suspected):
        series.append(("Bad sequence, upper boundary", suspected.astype(float), ORANGE))
    if len(status_runs) and not populations_identical:
        series.append(("Final-status bad sequence", status_runs.astype(float), BLUE))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        page_summary(pdf, df, episodes)
        page_timeline(pdf, df)
        if series:
            page_miss_curve(pdf, series, df, args.max_interval)
        if len(episodes):
            page_distribution(pdf, episodes["suspected_length"].to_numpy(),
                              "bad sequences, upper boundary", ORANGE,
                              "Window between the passing test before and the passing test "
                              "after each failure episode.")
        if len(status_runs) and not populations_identical:
            page_distribution(pdf, status_runs.astype(float),
                              "final-status bad sequences", BLUE,
                              "Consecutive electrodes ultimately marked Scrapped, by test or "
                              "by inference.")
        page_episode_table(pdf, episodes)
        page_notes(pdf, df, episodes, status_runs, gaps, populations_identical)

        info = pdf.infodict()
        info["Title"] = "QC sampling frequency analysis"
        info["Subject"] = f"Bad-sequence lengths vs test interval — {csv_path.name}"

    episodes.to_csv(pdf_path.with_name("qc_failure_episodes.csv"), index=False)

    print(f"{len(df)} electrodes, {int(df['tested'].sum())} QC tests, "
          f"{len(episodes)} QC episodes, {len(status_runs)} final-status bad sequences")
    intervals, _ = test_intervals(df)
    if len(intervals) and len(series):
        for label, lengths, _ in series:
            print(f"  {label:<28} mean {lengths.mean():5.1f}   "
                  f"P(missed at current interval {intervals.mean():.0f}) = "
                  f"{miss_probability(lengths, intervals.mean()):.1%}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
