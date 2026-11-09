"""QC analysis — terminal app. Generates PNGs into analysis_result/."""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from python_calamine import CalamineWorkbook

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "analysis_result"

# Channel code is the part after the last "_" in the file name, e.g. "2-3".
CHANNEL_RE = re.compile(r"^\d+-\d+$")

# The `record` sheet logs one row every 30 s ("Record settings: 30s/0V/0A").
SAMPLE_INTERVAL_H = 30 / 3600

# Centred rolling mean applied to every plotted line. 0 disables smoothing.
SMOOTH_MINUTES = 10
SMOOTH_WINDOW = round(SMOOTH_MINUTES / (SAMPLE_INTERVAL_H * 60))

# Channels whose T1 probe is faulty — excluded from temperature plots only,
# their voltage is fine. See channel_5-1_temperature_issue.md.
BAD_T1_CHANNELS = {"5-1"}

# Plots are cut to the shortest study length so the 168 h runs line up with the
# 48 h ones. None plots every run in full.
PLOT_MAX_HOURS = 48

# Categorical slots 1-4 in fixed order, each paired with a line style so the
# studies stay separable without relying on hue alone.
STUDY_COLOR = {
    "Cathodes_QC_IC_168h": "#2a78d6",
    "Cathodes_QC_IC_48h": "#eb6834",
    "Location study 27-07-2026 QC": "#1baf7a",
    "Operator study": "#eda100",
}

# Categorical slots in fixed order, for the per-channel plots. Never cycled —
# 8 series is the cap.
CATEGORICAL_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

Cathodes_QC_IC_168h_excel_file = sorted((HERE / "Cathodes_QC_IC_168h").glob("*.xlsx"))
Cathodes_QC_IC_48h_excel_file = sorted((HERE / "Cathodes_QC_IC_48h").glob("*.xlsx"))
Location_study_27_07_2026_QC_excel_file = sorted((HERE / "Location study 27-07-2026 QC").glob("*.xlsx"))
Operator_study_excel_file = sorted((HERE / "Operator study").glob("*.xlsx"))

STUDIES = {
    "Cathodes_QC_IC_168h": Cathodes_QC_IC_168h_excel_file,
    "Cathodes_QC_IC_48h": Cathodes_QC_IC_48h_excel_file,
    "Location study 27-07-2026 QC": Location_study_27_07_2026_QC_excel_file,
    "Operator study": Operator_study_excel_file,
}


def channel_of(path):
    """'..._Fe_2-3.xlsx' -> '2-3'. None when the tail isn't a channel code."""
    tail = path.stem.rsplit("_", 1)[-1]
    return tail if CHANNEL_RE.match(tail) else None


def sheet_df(workbook, name, header_row=0):
    """Sheet -> DataFrame. auxTemp needs header_row=1 (row 0 is a merged banner)."""
    rows = workbook.get_sheet_by_name(name).to_python()
    return pd.DataFrame(rows[header_row + 1:], columns=rows[header_row])


def load_run(path):
    """Voltage(V) from the `record` sheet + T1 from the `auxTemp` sheet."""
    workbook = CalamineWorkbook.from_path(str(path))
    voltage = sheet_df(workbook, "record")["Voltage(V)"]
    t1 = sheet_df(workbook, "auxTemp", header_row=1)["T1"]

    n = min(len(voltage), len(t1))
    if len(voltage) != len(t1):
        print(f"    ! {path.name}: record has {len(voltage)} rows, auxTemp {len(t1)} — trimmed to {n}")

    # Blank cells arrive as '' from calamine, so the columns land as object dtype.
    df = pd.DataFrame({
        "Voltage(V)": pd.to_numeric(voltage.iloc[:n], errors="coerce").to_numpy(),
        "T1": pd.to_numeric(t1.iloc[:n], errors="coerce").to_numpy(),
    })
    df.attrs["file_name"] = path.name
    return df


def load_all():
    """{channel: [DataFrame, ...]} across every study folder."""
    runs = {}
    for study, files in STUDIES.items():
        for path in files:
            channel = channel_of(path)
            if channel is None:
                continue
            df = load_run(path)
            df.attrs["study"] = study
            runs.setdefault(channel, []).append(df)
    return runs


def smooth(series):
    """Centred rolling mean, so plots show trend rather than sample noise."""
    if SMOOTH_WINDOW <= 1:
        return series
    return series.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()


def shade(base_color, i, n):
    """Step a unit's hue light→dark so its channels stay separable."""
    if n == 1:
        return base_color
    rgb = np.array(mcolors.to_rgb(base_color))
    t = -0.32 + 0.64 * i / (n - 1)          # negative = lighter, positive = darker
    mixed = rgb + (1.0 - rgb) * -t if t < 0 else rgb * (1.0 - t)
    return mcolors.to_hex(np.clip(mixed, 0, 1))


def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, fontsize=11, color=INK_PRIMARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def plot_runs(runs, column, ylabel, title, out_file, skip_channels=(),
              max_hours=PLOT_MAX_HOURS):
    """Every run of `column` overlaid on one axes, coloured by study."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    limit = None if max_hours is None else round(max_hours / SAMPLE_INTERVAL_H) + 1
    plotted = channels = 0
    for channel in sorted(runs):
        if channel in skip_channels:
            continue
        channels += 1
        for df in runs[channel]:
            cut = df.iloc[:limit]
            ax.plot(cut.index * SAMPLE_INTERVAL_H, smooth(cut[column]),
                    color=STUDY_COLOR[df.attrs["study"]], linewidth=0.9, alpha=0.8)
            plotted += 1

    style_axes(ax, ylabel)
    ax.set_xlabel("Time (hours)", fontsize=11, color=INK_PRIMARY)
    window = "" if max_hours is None else f", first {max_hours:g} h"
    ax.set_title(f"{title} — {plotted} runs across {channels} channels{window}"
                 f", {SMOOTH_MINUTES}-min rolling mean",
                 fontsize=13, color=INK_PRIMARY, loc="left", pad=12)

    handles = [Line2D([], [], color=color, linewidth=1.8, label=study)
               for study, color in STUDY_COLOR.items()]
    legend = ax.legend(handles=handles, fontsize=9, loc="lower right",
                       facecolor=SURFACE, edgecolor=GRIDLINE, framealpha=0.95)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_file}")


def without_studies(runs, studies):
    """Copy of `runs` with every run from the named studies dropped."""
    kept = {c: [df for df in dfs if df.attrs["study"] not in studies]
            for c, dfs in runs.items()}
    return {c: dfs for c, dfs in kept.items() if dfs}


def channel_mean(dfs, column, limit):
    """Smoothed mean across a channel's runs, aligned on sample index."""
    frames = [df[column].iloc[:limit].reset_index(drop=True) for df in dfs]
    return smooth(pd.concat(frames, axis=1).mean(axis=1))


def plot_channel_means_combined(runs, column, ylabel, title, out_file,
                                skip_channels=(), max_hours=PLOT_MAX_HOURS):
    """Every channel mean on a single axes, coloured by unit."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    limit = None if max_hours is None else round(max_hours / SAMPLE_INTERVAL_H) + 1
    channels = [c for c in sorted(runs) if c not in skip_channels]
    units = sorted({c.split("-")[0] for c in channels})
    if len(units) > len(CATEGORICAL_COLORS):
        raise ValueError(f"{len(units)} units exceeds the {len(CATEGORICAL_COLORS)} colour slots")
    unit_color = dict(zip(units, CATEGORICAL_COLORS))

    # Composite encoding: hue carries the unit, lightness the channel within it,
    # so all 20 channels stay separable without inventing 20 unrelated hues.
    for channel in channels:
        unit = channel.split("-")[0]
        siblings = [c for c in channels if c.split("-")[0] == unit]
        color = shade(unit_color[unit], siblings.index(channel), len(siblings))
        mean = channel_mean(runs[channel], column, limit)
        ax.plot(mean.index.to_numpy() * SAMPLE_INTERVAL_H, mean,
                color=color, linewidth=1.3,
                label=f"{channel}  (n={len(runs[channel])})")

    style_axes(ax, ylabel)
    ax.set_xlabel("Time (hours)", fontsize=11, color=INK_PRIMARY)
    window = "" if max_hours is None else f", first {max_hours:g} h"
    ax.set_title(f"{title} — mean per channel, {len(channels)} channels{window}"
                 f", {SMOOTH_MINUTES}-min rolling mean",
                 fontsize=13, color=INK_PRIMARY, loc="left", pad=12)

    legend = ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       facecolor=SURFACE, edgecolor=GRIDLINE, framealpha=0.95,
                       title="channel")
    legend.get_title().set_color(INK_SECONDARY)
    legend.get_title().set_fontsize(9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_file}")


def plot_channel_means(runs, column, ylabel, title, out_file, skip_channels=(),
                       max_hours=PLOT_MAX_HOURS):
    """One mean line per channel, faceted one panel per unit."""
    limit = None if max_hours is None else round(max_hours / SAMPLE_INTERVAL_H) + 1
    channels = [c for c in sorted(runs) if c not in skip_channels]
    units = sorted({c.split("-")[0] for c in channels})

    ncols = 2
    nrows = -(-len(units) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.1 * nrows), dpi=150,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()

    for ax, unit in zip(axes, units):
        unit_channels = [c for c in channels if c.split("-")[0] == unit]
        if len(unit_channels) > len(CATEGORICAL_COLORS):
            raise ValueError(f"unit {unit} has {len(unit_channels)} channels, "
                             f"more than the {len(CATEGORICAL_COLORS)} colour slots")

        for channel, color in zip(unit_channels, CATEGORICAL_COLORS):
            mean = channel_mean(runs[channel], column, limit)
            hours = mean.index.to_numpy() * SAMPLE_INTERVAL_H
            ax.plot(hours, mean, color=color, linewidth=1.4,
                    label=f"{channel}  (n={len(runs[channel])})")

        style_axes(ax, "")
        ax.set_title(f"unit {unit}", fontsize=10, color=INK_PRIMARY, loc="left", pad=6)
        legend = ax.legend(fontsize=8, loc="lower right", facecolor=SURFACE,
                           edgecolor=GRIDLINE, framealpha=0.95, handlelength=2.4)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    for ax in axes[len(units):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:len(units)]):
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel, fontsize=10, color=INK_PRIMARY)
        if i + ncols >= len(units):  # nothing visible below it — needs its own x axis
            ax.set_xlabel("Time (hours)", fontsize=10, color=INK_PRIMARY)
            ax.tick_params(labelbottom=True)

    window = "" if max_hours is None else f", first {max_hours:g} h"
    fig.suptitle(f"{title} — mean per channel, {len(channels)} channels{window}"
                 f", {SMOOTH_MINUTES}-min rolling mean",
                 fontsize=13, color=INK_PRIMARY, x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_file}")


def main():
    runs = load_all()

    gaps = [(df.attrs["file_name"], df.isna().sum().to_dict())
            for dfs in runs.values() for df in dfs if df.isna().to_numpy().any()]
    for file_name, counts in gaps:
        print(f"! non-numeric cells in {file_name}: {counts}")

    plot_runs(runs, "Voltage(V)", "Voltage (V)", "Voltage",
              OUT_DIR / "voltage.png")
    plot_runs(runs, "T1", "Temperature T1 (°C)", "Temperature",
              OUT_DIR / "temperature.png", skip_channels=BAD_T1_CHANNELS)

    plot_channel_means(runs, "Voltage(V)", "Voltage (V)", "Voltage",
                       OUT_DIR / "voltage_by_channel.png")
    plot_channel_means(runs, "T1", "Temperature T1 (°C)", "Temperature",
                       OUT_DIR / "temperature_by_channel.png",
                       skip_channels=BAD_T1_CHANNELS)

    plot_channel_means_combined(runs, "Voltage(V)", "Voltage (V)", "Voltage",
                                OUT_DIR / "voltage_all_channels.png")
    plot_channel_means_combined(runs, "T1", "Temperature T1 (°C)", "Temperature",
                                OUT_DIR / "temperature_all_channels.png",
                                skip_channels=BAD_T1_CHANNELS)

    short = without_studies(runs, {"Cathodes_QC_IC_168h"})
    plot_channel_means_combined(short, "Voltage(V)", "Voltage (V)",
                                "Voltage, 168 h study excluded",
                                OUT_DIR / "voltage_all_channels_no168h.png")
    plot_channel_means_combined(short, "T1", "Temperature T1 (°C)",
                                "Temperature, 168 h study excluded",
                                OUT_DIR / "temperature_all_channels_no168h.png",
                                skip_channels=BAD_T1_CHANNELS)

    total = sum(len(v) for v in runs.values())
    print(f"\n{total} runs across {len(runs)} channels")
    for channel in sorted(runs):
        studies = ", ".join(df.attrs["study"] for df in runs[channel])
        print(f"  {channel:<5} {len(runs[channel])}x  [{studies}]")


if __name__ == "__main__":
    sys.exit(main())
