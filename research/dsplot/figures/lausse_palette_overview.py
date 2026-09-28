"""lausse_palette_overview - the Lausse Palette in the Optichrome overview layout.

Same figure shape as `optichrome_overview` (whose panel helpers it reuses as-is):
source art up top, then one wide row per spectrum -
[block + continuous strips] · Centroid · Energy Spectrum · Weave.

The palette is measured from three Lausse the Cat covers. Ink, the red ramp
(oxblood -> scarlet, brightness-binned creature reds) and cream / bone (paper
and moon shading) come from the middle CH.1 cover; ochre and the blues from the
outer two. Hand-ordered into a night loop:
ink -> reds -> cream paper -> moonlit blues -> ink.

    PYTHONPATH=research python -m dsplot.figures.lausse_palette_overview
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from PIL import Image

from dsplot import Figure, Heatmap, HeatmapPanel, SuptitlePanel, style

from . import optichrome_overview as opti
from .optichrome_overview import (
    SPECTRUM_ROW_HEIGHT,
    TOTAL_UNITS,
    UNIT_INCHES,
    _draw_section2_chrome,
    _draw_separator,
    _field_panel,
    _img_panel,
    _shrink_strip_pairs,
    _square_cwt,
    _square_field_plots,
    _stacked_strips,
)
from .optichrome_showcase import _build_centroid, _build_weave


VERSION = "v6"
DEFAULT_OUTPUT_DIR = "assets/images/dsp/figures/lausse_palette"
DEFAULT_OUTPUT_FILENAME = f"lausse_palette_{VERSION}.png"
SOURCE_ART = "assets/images/dsp/figures/lausse_palette/source_lausse_covers.webp"

SUPTITLE_TEXT = "Lausse Palette - from Lausse the Cat - DSPlot Showcase"
PAPER_COLOR = "#FCFDD4"
INK_COLOR = "#1A1512"
ART_ROW_HEIGHT = 3.45
REPLICA_BLOCK_COLS = 132

INK = "#0B080F"
OXBLOOD = "#300707"
MAROON = "#5F100E"
CRIMSON = "#861A17"
RED = "#A7221F"
SCARLET = "#CB2B26"
OCHRE = "#C5904C"
BONE = "#CEC5AC"
CREAM = "#FCFDD4"
SKY = "#79ACE8"
CERULEAN = "#30A0DA"
PERIWINKLE = "#4769DF"
COBALT = "#104DB2"
MIDNIGHT = "#123561"
SLATE = "#272C34"

NIGHT_LOOP = [INK, OXBLOOD, MAROON, CRIMSON, RED, SCARLET, OCHRE, BONE, CREAM,
              SKY, CERULEAN, PERIWINKLE, COBALT, MIDNIGHT, SLATE]
CREAM_CENTERED = [INK, MAROON, RED, SCARLET, CREAM, SKY, PERIWINKLE, MIDNIGHT, INK]
EMBER = [INK, OXBLOOD, MAROON, CRIMSON, RED, SCARLET, BONE, CREAM]
MOONLIGHT = [INK, SLATE, MIDNIGHT, COBALT, PERIWINKLE, CERULEAN, SKY, CREAM]
POSTER = [MIDNIGHT, COBALT, PERIWINKLE, SKY, CREAM, SCARLET, RED, CRIMSON, OXBLOOD]
GOUACHE = [INK, SLATE, CRIMSON, OCHRE, BONE, CREAM]


def _spectrum(name: str, colors: list) -> dict:
    return dict(name=name,
                disc=ListedColormap(colors),
                cont=LinearSegmentedColormap.from_list(f"lausse_{name.lower()}", colors, N=256))


def _make_spectra() -> list:
    return [
        _spectrum("Night Loop", NIGHT_LOOP),
        _spectrum("Cream Centered", CREAM_CENTERED),
        _spectrum("Ember", EMBER),
        _spectrum("Moonlight", MOONLIGHT),
        _spectrum("Poster", POSTER),
        _spectrum("Gouache", GOUACHE),
    ]


def _build_replica(source: Image.Image) -> np.ndarray:
    """Mosaic the covers onto a coarse block grid, each block snapped to the
    nearest Night Loop color (same recipe as the Optichrome replica)."""
    w, h = source.size
    cols = REPLICA_BLOCK_COLS
    rows = round(cols * h / w)
    blocks = np.asarray(source.resize((cols, rows), Image.BOX)).astype(float)
    palette = np.array([to_rgb(c) for c in NIGHT_LOOP]) * 255
    nearest = ((blocks.reshape(-1, 3)[:, None] - palette[None]) ** 2).sum(-1).argmin(1)
    mosaic = palette[nearest].reshape(rows, cols, 3).astype(np.uint8)
    return np.asarray(Image.fromarray(mosaic).resize((w, h), Image.NEAREST))


def _paper_separator_panel() -> HeatmapPanel:
    """Invisible spacer row painted in the full paper RGB (the shared helper
    only carries the red channel, which reads as a white bar on cream)."""
    paper = np.array([round(c * 255) for c in to_rgb(PAPER_COLOR)], dtype=np.uint8)
    bar = np.broadcast_to(paper, (1, 2, 3)).copy()
    p = HeatmapPanel(units=(TOTAL_UNITS, 1), title=None, show_border=False,
                     show_xticklabels=False, show_yticklabels=False)
    p.add(Heatmap(bar, extent=(0, 2, 0, 1), origin="upper", vmin=0, vmax=255))
    return p


def _apply_style_knobs() -> None:
    opti._apply_style_knobs()
    opti.TEXT_COLOR = INK_COLOR
    style.BG_COLOR = PAPER_COLOR
    style.SUPTITLE_COLOR = INK_COLOR
    style.TICK_LABEL_COLOR = INK_COLOR


def _build_figure():
    _apply_style_knobs()
    spectra = _make_spectra()

    source = Image.open(SOURCE_ART).convert("RGB")
    art_panels = [
        _img_panel("Original Covers", np.asarray(source), units=(TOTAL_UNITS, 1)),
        _img_panel("Night Loop Replica", _build_replica(source), units=(TOTAL_UNITS, 1)),
    ]

    fields = {
        "centroid": _build_centroid(),
        "cwt": _square_cwt(),
        "weave": _build_weave(base_freq_hz=2.0),
    }

    spectrum_rows, strip_pairs = [], []
    for s in spectra:
        comp, block, cont = _stacked_strips(s)
        strip_pairs.append((block, cont))
        spectrum_rows.append([comp] + [_field_panel(fields[k], s["cont"])
                                       for k in ("centroid", "cwt", "weave")])

    sep_panel = _paper_separator_panel()
    fig = Figure.compose(
        rows=[
            [SuptitlePanel(SUPTITLE_TEXT, units=(TOTAL_UNITS, 1),
                           font_size=opti.SUPTITLE_FONT_PT)],
            [art_panels[0]],
            [art_panels[1]],
            [sep_panel],
            *spectrum_rows,
        ],
        row_heights=([1.8, ART_ROW_HEIGHT, ART_ROW_HEIGHT, 0.30]
                     + [SPECTRUM_ROW_HEIGHT] * len(spectrum_rows)),
        unit_inches=UNIT_INCHES,
        top_reserve_inches=opti.TOP_RESERVE_INCHES,
        show_cell_borders=False,
    )
    fig.render()

    for p in art_panels:
        if p.ax is not None:
            p.ax.set_aspect("equal", anchor="C")

    _shrink_strip_pairs(strip_pairs)
    _square_field_plots(fig, spectrum_rows, strip_pairs)
    _draw_separator(fig, sep_panel)
    _draw_section2_chrome(fig, spectrum_rows, spectra, strip_pairs)
    return fig


def render(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> str:
    fig = _build_figure()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render())
