"""Unsupervised view of the same per-vat readings.

Clustering is never told which electrodes cracked. It groups electrodes by
how similar their readings are, and the question is then whether those
natural groups happen to line up with cracking — or with something else
entirely, like which vat the electrode was plated in.

Writes clustering.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATASET = HERE / "vat_signals_dataset.csv"
OUTPUT = HERE / "clustering.png"
SEED = 0

COOL = "#0EA5A5"
HOT = "#8B5CF6"
PALETTE = ["#0EA5A5", "#8B5CF6", "#F59E0B", "#6366F1", "#EC4899", "#10B981"]
FIG_BG = "#FAFBFC"
AXES_BG = "#FFFFFF"
GRID = "#E5E7EB"
SPINE = "#D1D5DB"
TEXT = "#111827"
LABEL = "#374151"
SUBTEXT = "#6B7280"


def chrome(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(AXES_BG)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11.5, fontweight="bold",
                 loc="left", pad=10)
    ax.set_xlabel(xlabel, color=LABEL, fontsize=9.5, labelpad=6)
    ax.set_ylabel(ylabel, color=LABEL, fontsize=9.5, labelpad=6)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=4, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def load():
    frame = pd.read_csv(DATASET)
    features = [c for c in frame.columns
                if c.startswith(("strike_", "plating_"))
                and c != "plating_vat"]
    scaled = StandardScaler().fit_transform(frame[features])
    return frame, features, scaled


def choose_k(scaled):
    print("k-means: how many clusters does the data support?")
    print(f"  {'k':>3}{'inertia':>12}{'silhouette':>13}")
    best = None
    for k in range(2, 7):
        model = KMeans(k, n_init=20, random_state=SEED).fit(scaled)
        score = silhouette_score(scaled, model.labels_)
        print(f"  {k:>3}{model.inertia_:>12.1f}{score:>13.3f}")
        if best is None or score > best[1]:
            best = (k, score)
    print(f"  -> best separation at k={best[0]} (silhouette {best[1]:.3f})")
    return best[0]


def describe(frame, labels, name, features):
    frame = frame.assign(cluster=labels)
    table = frame.groupby("cluster").agg(
        electrodes=("cracked", "size"), cracked=("cracked", "sum"))
    table["cracked_rate"] = (table["cracked"] / table["electrodes"])
    vats = pd.crosstab(frame["cluster"], frame["plating_vat"])
    means = frame.groupby("cluster")[features].mean().round(1)

    print(f"\n{name}")
    print(table.to_string(formatters={"cracked_rate": "{:.0%}".format}))
    print("\n  plating vat per cluster:")
    print("  " + vats.to_string().replace("\n", "\n  "))
    print("\n  cluster centres:")
    print("  " + means.to_string().replace("\n", "\n  "))
    print(f"\n  agreement with cracking   (ARI) {adjusted_rand_score(frame['cracked'], labels):+.3f}")
    print(f"  agreement with plating vat (ARI) "
          f"{adjusted_rand_score(frame['plating_vat'], labels):+.3f}")
    return table


def scatter(ax, points, labels, colours, title, legend_labels=None):
    for value in sorted(pd.unique(labels)):
        mask = labels == value
        ax.scatter(points[mask, 0], points[mask, 1], s=30, alpha=0.7,
                   color=colours[int(value) % len(colours)],
                   edgecolors="none", zorder=3,
                   label=legend_labels[int(value)] if legend_labels else
                   f"cluster {value}")
    chrome(ax, title, "principal component 1", "principal component 2")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SUBTEXT, loc="best")


def main():
    frame, features, scaled = load()
    print(f"{len(frame)} electrodes, {int(frame['cracked'].sum())} cracked, "
          f"{len(features)} standardised features\n")

    k = choose_k(scaled)
    kmeans = KMeans(k, n_init=20, random_state=SEED).fit(scaled)
    describe(frame, kmeans.labels_, f"k-means, k={k}", features)

    two = KMeans(2, n_init=20, random_state=SEED).fit(scaled)
    if k != 2:
        describe(frame, two.labels_, "k-means, k=2 (one per verdict?)",
                 features)

    mixture = GaussianMixture(k, random_state=SEED).fit_predict(scaled)
    describe(frame, mixture, f"gaussian mixture, {k} components", features)

    ward = AgglomerativeClustering(k).fit_predict(scaled)
    describe(frame, ward, f"agglomerative (ward), {k} clusters", features)

    dbscan = DBSCAN(eps=1.2, min_samples=5).fit_predict(scaled)
    found = len(set(dbscan)) - (1 if -1 in dbscan else 0)
    print(f"\nDBSCAN: {found} clusters, "
          f"{int((dbscan == -1).sum())} points left as noise")

    points = PCA(2, random_state=SEED).fit_transform(scaled)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    figure.patch.set_facecolor(FIG_BG)
    scatter(axes[0], points, kmeans.labels_, PALETTE,
            f"k-means clusters (k={k})")
    scatter(axes[1], points, frame["cracked"].to_numpy(), [COOL, HOT],
            "the same electrodes, coloured by verdict",
            ["not cracked", "cracked"])
    vat_codes = frame["plating_vat"].astype("category")
    scatter(axes[2], points, vat_codes.cat.codes.to_numpy(), PALETTE,
            "and coloured by plating vat",
            [f"vat {v}" for v in vat_codes.cat.categories])
    figure.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12,
                           wspace=0.25)
    figure.savefig(OUTPUT, facecolor=FIG_BG)
    print(f"\n✓ {OUTPUT.name}")


if __name__ == "__main__":
    main()
