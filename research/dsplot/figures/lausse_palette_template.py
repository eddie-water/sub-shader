"""lausse_palette_template - the Sample Template Showcase dressed in the Lausse Palette.

Reuses sample_template's panel builders unchanged; only dsplot.style knobs are
retuned (cream paper, ink chrome, scarlet / cobalt / ochre accents, Ember
heatmaps). The stem quartet's magnitude row is rebuilt so it reads through the
Lausse Ember map instead of sample_template's hardcoded inferno.

    PYTHONPATH=research python -m dsplot.figures.lausse_palette_template
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from dsplot import CompositePanel, Figure, Heatmap, HeatmapPanel, SuptitlePanel, style

from . import lausse_palette_overview as lausse
from . import sample_template as template
from .optichrome_overview import _strip_panel


VERSION = "v4"
DEFAULT_OUTPUT_DIR = "assets/images/dsp/figures/lausse_palette"
DEFAULT_OUTPUT_FILENAME = f"lausse_palette_template_{VERSION}.png"

SUPTITLE_TEXT = "dsplot - Sample Template - Lausse Palette"
EMBER_CMAP_NAME = "lausse_ember"

TITLE_FONT_PT = 30
TICK_LABEL_FONT_PT = 24
AXIS_LABEL_FONT_PT = 28
VECTOR_LINEWIDTH = 7.0
PALETTE_ROW_HEIGHT = 0.36


def _register_ember() -> None:
    if EMBER_CMAP_NAME not in mpl.colormaps:
        mpl.colormaps.register(
            LinearSegmentedColormap.from_list(EMBER_CMAP_NAME, lausse.EMBER, N=256),
            name=EMBER_CMAP_NAME,
        )


def _apply_lausse_theme() -> None:
    style.BG_COLOR = lausse.PAPER_COLOR
    style.PRIMARY_COLOR = lausse.SCARLET
    style.SECONDARY_COLOR = lausse.COBALT
    style.TERTIARY_COLOR = lausse.OCHRE
    style.HIGHLIGHT_COLOR = lausse.CERULEAN
    style.NEUTRAL_COLOR = lausse.INK
    style.TICK_LABEL_COLOR = lausse.INK
    style.SUPTITLE_COLOR = lausse.INK
    style.SPINE_COLOR = lausse.BONE
    style.DROPLINE_COLOR = lausse.SLATE
    style.DEFAULT_FRAME_COLOR = lausse.INK
    style.DEFAULT_AXIS_GRID_COLOR = lausse.INK
    style.DEFAULT_SPOTLIGHT_EDGE_COLOR = lausse.INK
    style.DEFAULT_ACCUM_STEM_COLOR = lausse.INK
    style.DEFAULT_ACCUM_READOUT_COLOR = lausse.INK
    style.INST_FREQ_COLOR = lausse.OCHRE
    style.DEFAULT_HEATMAP_CMAP = EMBER_CMAP_NAME
    style.DEFAULT_TITLE_FONT_SIZE = TITLE_FONT_PT
    style.DEFAULT_SUPTITLE_FONT_SIZE = TITLE_FONT_PT
    style.SUPTITLE_FONT_SIZE = TITLE_FONT_PT
    style.DEFAULT_TICK_LABEL_SIZE = TICK_LABEL_FONT_PT
    style.DEFAULT_AXIS_LABEL_SIZE = AXIS_LABEL_FONT_PT
    style.DEFAULT_VECTOR_LINEWIDTH = VECTOR_LINEWIDTH
    style.DEFAULT_VECTOR_BOLD_LINEWIDTH = VECTOR_LINEWIDTH


def _field_heatmap(title: str, data: np.ndarray, colors: list) -> HeatmapPanel:
    panel = HeatmapPanel(
        units=(1, 1),
        title=title,
        x_label="x",
        y_label="y",
        xticks=[-1.0, 0.0, 1.0],
        yticks=[-1.0, 0.0, 1.0],
        show_xticklabels=True,
        show_yticklabels=True,
    )
    cmap = LinearSegmentedColormap.from_list(f"lausse_{title}", colors, N=256)
    panel.add(Heatmap(data, cmap=cmap, extent=(-1.0, 1.0, -1.0, 1.0), aspect="equal"))
    return panel


def _palette_band(width: int) -> CompositePanel:
    """Night Loop as blocks over its continuous blend, full figure width."""
    loop = lausse.NIGHT_LOOP
    return CompositePanel(
        units=(width, 1),
        title="Lausse Palette - Night Loop",
        rows=[
            [_strip_panel(ListedColormap(loop), units=(width, 1))],
            [_strip_panel(LinearSegmentedColormap.from_list("lausse_loop", loop, N=256),
                          units=(width, 1))],
        ],
    )


def _stem_quartet() -> CompositePanel:
    """sample_template's quartet, magnitude row colored through Ember."""
    n = template.STEM_N
    t = np.arange(n, dtype=np.float64)
    square = np.where(t < n // 2, 1.0, -1.0)
    sin_low = np.sin(2.0 * np.pi * template.STEM_SINE_HZ_LOW * t / n)
    sin_high = np.sin(2.0 * np.pi * template.STEM_SINE_HZ_HIGH * t / n)
    triangle = 1.0 - 4.0 * np.abs(t / (n - 1) - 0.5)
    magnitude = np.abs(triangle) / np.abs(triangle).max()
    ember = mpl.colormaps[EMBER_CMAP_NAME]
    triangle_colors = np.array([mpl.colors.to_hex(ember(m)) for m in magnitude])

    stem_row = template._stem_row_panel
    return CompositePanel(
        units=(2, 1),
        title="Stem Quartet",
        rows=[
            [stem_row(t, square, color=style.NEUTRAL_COLOR)],
            [stem_row(t, sin_low, color=style.SECONDARY_COLOR)],
            [stem_row(t, sin_high, color=style.PRIMARY_COLOR)],
            [stem_row(t, triangle, color_per_sample=triangle_colors, show_xticks=True)],
        ],
        share_x=True,
    )


def _build_figure() -> Figure:
    _register_ember()
    _apply_lausse_theme()
    width = template.ROW_WIDTH_UNITS
    return Figure.compose(
        rows=[
            [SuptitlePanel(SUPTITLE_TEXT, units=(width, 1))],
            [_palette_band(width)],
            [template._r1_vector_2d(), template._r1_vector_3d(),
             _field_heatmap("Weave - Poster", template._build_gaf(), lausse.POSTER),
             _field_heatmap("Centroid - Ember", template._build_gaussian(), lausse.EMBER)],
            [template._r2_chirp_with_inst_freq()],
            [_stem_quartet(), template._r3_jargon_panel(), template._r3_vector_projection()],
        ],
        row_heights=[0.25, PALETTE_ROW_HEIGHT, 1.0, 1.0, 1.0],
        show_cell_borders=True,
    )


def render(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> str:
    fig = _build_figure()
    fig.render()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render())
