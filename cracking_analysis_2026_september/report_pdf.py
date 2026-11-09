"""A one-page PDF of the cracking findings.

The wording lives in TITLE and SECTIONS below - edit those, not the layout.
A section is a heading and a list of entries; an entry is

    "plain text"          a paragraph
    "- bulleted text"     a bullet
    IMAGES / "name.png"   a Path, drawn at PLOT_WIDTH and its own aspect
    (IMAGES / "x.png", 0.55)   the same, at a width you choose

    python report_pdf.py [output.pdf]
"""
import sys
from pathlib import Path

import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

FOLDER = Path(__file__).resolve().parent
IMAGES = FOLDER / "report_images"
DEFAULT = FOLDER / "coating_integrity_analysis.pdf"

TEXT = "#111827"
BODY = "#374151"

TITLE = "Coating integrity analysis"
SECTIONS = [
    ("Main findings", [
        "Main factors that might contribute towards cracks are:",
        "- Ni:S ratio, especially the nickel content itself",
        IMAGES / "ni_s_ratio.png",
        "- Slight correlation with the cleaner bath voltage",
        IMAGES / "cleaner_voltage.png",
        "- Slight correlation with the Decasel 5 dumps into the "
        "activator bath",
        IMAGES / "decasel_dose_age.png",
    ]),
]

PAGE = (8.27, 11.69)   # A4 portrait
LEFT = 0.10
RIGHT = 0.95
TOP = 0.93
BOTTOM = 0.06
PLOT_WIDTH = 0.76      # of the page width


def _tall(path, wide):
    """How much of the page height the image needs at that width."""
    picture = mpimg.imread(path)
    width, height = PAGE
    return wide * width * (picture.shape[0] / picture.shape[1]) / height


def _image(figure, path, y, wide=PLOT_WIDTH):
    """Draw the image left-aligned under the text and return the new y."""
    tall = _tall(path, wide)
    axes = figure.add_axes((LEFT + 0.02, y - tall, wide, tall))
    axes.imshow(mpimg.imread(path))
    axes.axis("off")
    return y - tall - 0.012


def _page():
    figure = Figure(figsize=PAGE, dpi=200)
    figure.patch.set_facecolor("#FFFFFF")
    return figure


def write(path=DEFAULT):
    with PdfPages(path) as pdf:
        figure = _page()
        y = TOP
        figure.text((LEFT + RIGHT) / 2, y, TITLE, color=TEXT, fontsize=20,
                    fontweight="bold", va="top", ha="center")
        y -= 0.055
        for heading, entries in SECTIONS:
            figure.text(LEFT, y, heading, color=TEXT, fontsize=13,
                        fontweight="bold", va="top")
            y -= 0.032
            for entry in entries:
                picture = (entry if isinstance(entry, Path)
                           else entry[0] if isinstance(entry, tuple) else None)
                wide = entry[1] if isinstance(entry, tuple) else PLOT_WIDTH
                needed = (_tall(picture, wide) + 0.012 if picture
                          else 0.026)
                # a whole plot rather than the top of one
                if y - needed < BOTTOM:
                    pdf.savefig(figure)
                    figure, y = _page(), TOP
                if picture is not None:
                    y = _image(figure, picture, y, wide)
                    continue
                bullet = entry.startswith("- ")
                figure.text(LEFT + (0.02 if bullet else 0), y,
                            f"• {entry[2:]}" if bullet else entry,
                            color=BODY, fontsize=11, va="top")
                y -= 0.026
            y -= 0.020
        pdf.savefig(figure)
    return path


if __name__ == "__main__":
    print(write(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
