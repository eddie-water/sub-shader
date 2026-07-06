"""optichrome_overview — the Optichrome palette + DSPlot showcase, one figure.

A landscape, six-column overview built with the dsplot composition API
(`Figure.compose` + a `CompositePanel` per column). Columns, left to right:

    Palette Origin   Original + Replica palette scans (square, undistorted)
    Colormaps        six-strip legend: Block · Block Continuous ·
                     Proportional Block · Proportional Continuous ·
                     Red Continuous · Blue Continuous
    Block Continuous       \
    Proportional Continuous |  one showcase column per CONTINUOUS map, each
    Red Continuous          |  rendering the same Centroid / Energy Spectrum /
    Blue Continuous        /   Weave fields (equal squares) through that cmap.

"Block Continuous" is an equal-spaced smooth interpolation of the 13 block
colors (`optichrome_rwbk_block`), registered inline here — distinct from
`optichrome_rwbk`, whose nodes are placed proportional to canvas area.

The synthetic fields come from `optichrome_showcase` (_build_weave /
_build_centroid / _build_cwt); the CWT is resized to a 96x96 square so its
showcase panel stays square like the others (a wide CWT used to overflow its
title across the narrow columns).

Edit the CONSTANTS block below to retune — version/output name, the three font
sizes, colors, and spacing — then re-run:

    PYTHONPATH=research python -m dsplot.figures.optichrome_overview

A bare run writes the DEFAULT_OUTPUT_FILENAME (a NEW version) so it never
overwrites an existing PNG; pass output_filename=... to render() to override.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

if __package__ in (None, ""):
    _RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _RESEARCH_DIR not in sys.path:
        sys.path.insert(0, _RESEARCH_DIR)
    __package__ = "dsplot.figures"

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from PIL import Image

from dsplot import (
    CompositePanel,
    Figure,
    Heatmap,
    HeatmapPanel,
    SuptitlePanel,
    colormaps,
    style,
)

from .optichrome_showcase import _build_centroid, _build_cwt, _build_weave


# ---------------------------------------------------------------------------
# CONSTANTS — hand-edit these.
# ---------------------------------------------------------------------------
VERSION = "v52"
DEFAULT_OUTPUT_DIR = "assets/images/dsp/figures/optichrome_rwbk"
DEFAULT_OUTPUT_FILENAME = f"overview_{VERSION}.png"

SUPTITLE_TEXT = "Palette - Optichrome by Felipe Pantone - DSPlot Showcase"

# Fonts (points). Suptitle is deliberately the SMALLEST; headers + inner labels
# are larger. White everywhere.
# Sized to read when viewing the whole (very large) figure, not zoomed in.
SUPTITLE_FONT_PT = 76      # the top title (increased a lot)
HEADER_FONT_PT = 48        # the per-column section headers (subtitle, +half)
LABEL_FONT_PT = 48         # every inner panel title (legend / plot / palette)
# White canvas with dark ink so text/lines stay legible on the light bg.
BG_COLOR = "#FFFFFF"       # regular white
TEXT_COLOR = "#1A1A1A"     # near-black ink
SEPARATOR_COLOR = "#1A1A1A"
SPINE_COLOR = "#CCCCCC"    # light-grey spines/borders

# Spacing (inches) — tight landscape layout.
MARGIN_INCHES = 0.5
# Section-2 gutters (row + column) are derived below as ONE THIRD of a spectrum
# row's height — see EVEN_GUTTER_INCHES after UNIT_INCHES / SPECTRUM_ROW_HEIGHT.
INNER_GUTTER_INCHES = 0.22  # gap between the block + continuous strips in a cell
PANEL_TITLE_RESERVE_INCHES = 0.40
UNIT_INCHES = 2.4          # bigger unit -> wider canvas / more horizontal spread
TOP_RESERVE_INCHES = 4.7   # body starts lower -> bigger gap below the suptitle
HEADER_GAP_INCHES = 0.65   # gap from body top up to the (larger) column headers
BODY_ROW_HEIGHT = 6.0      # (legacy) unused by the row-per-spectrum layout

# Row-per-spectrum layout (section 2): each spectrum is one wide row —
# [stacked block+continuous strips] · Centroid · Energy Spectrum · Weave.
FIELD_UNITS = 2            # field cell width (in layout units)
STRIP_UNITS = 2 * FIELD_UNITS  # spectrum spans exactly TWO field columns (clean multiple)
TOTAL_UNITS = STRIP_UNITS + 3 * FIELD_UNITS   # = 10
# Row height is DECOUPLED from FIELD_UNITS so rows stay thin while the canvas
# grows wide — fields read as wide landscape rectangles, not squares.
SPECTRUM_ROW_HEIGHT = float(FIELD_UNITS)        # rows as tall as fields are wide -> SQUARE heatmaps
TOP_ROW_HEIGHT = 3.0       # taller band -> bigger palette-origin paintings (square)

# Section-2 spacing: every gutter (between rows AND between columns) is ONE THIRD
# of a spectrum row's rendered height, so the gap reads identical in both axes.
EVEN_GUTTER_INCHES = (SPECTRUM_ROW_HEIGHT * UNIT_INCHES) / 3.0
GUTTER_INCHES = EVEN_GUTTER_INCHES         # vertical gap between rows
COLUMN_GUTTER_INCHES = EVEN_GUTTER_INCHES  # horizontal gap between columns

# Source scan for the painting replica.
SOURCE_SCAN = "assets/images/claude/diagnostics/optichrome_felipe_pantone.png"

# The new equal-spaced Block Continuous map.
BLOCK_CONTINUOUS_NAME = "optichrome_rwbk_block"

# Native block grid recovered from the source painting.
GRID_COLS, GRID_ROWS = 17, 22


# ---------------------------------------------------------------------------
# Colormap registration
# ---------------------------------------------------------------------------
def _register_block_continuous() -> None:
    """Register `optichrome_rwbk_block`: the 13 block colors spaced EQUALLY
    (idempotent). Distinct from `optichrome_rwbk`'s proportional node spacing.
    """
    if BLOCK_CONTINUOUS_NAME not in mpl.colormaps:
        loop = list(colormaps.RWBK_LOOP_COLORS)
        mpl.colormaps.register(
            cmap=LinearSegmentedColormap.from_list(BLOCK_CONTINUOUS_NAME, loop, N=256),
            name=BLOCK_CONTINUOUS_NAME,
        )


def _apply_style_knobs() -> None:
    """Push the tight-spacing + font/color knobs into dsplot.style (compose
    resolves these lazily at build time)."""
    style.DEFAULT_MARGIN_INCHES = MARGIN_INCHES
    style.DEFAULT_GUTTER_INCHES = GUTTER_INCHES
    style.DEFAULT_COLUMN_GUTTER_INCHES = COLUMN_GUTTER_INCHES
    style.DEFAULT_INNER_GUTTER_INCHES = INNER_GUTTER_INCHES
    style.DEFAULT_PANEL_TITLE_RESERVE_INCHES = PANEL_TITLE_RESERVE_INCHES
    # SuptitlePanel resolves its size from style.SUPTITLE_FONT_SIZE at
    # construction (NOT DEFAULT_SUPTITLE_FONT_SIZE) — set both so the lifted
    # suptitle actually honors SUPTITLE_FONT_PT.
    style.SUPTITLE_FONT_SIZE = SUPTITLE_FONT_PT
    style.DEFAULT_SUPTITLE_FONT_SIZE = SUPTITLE_FONT_PT
    style.DEFAULT_TITLE_FONT_SIZE = LABEL_FONT_PT
    style.TICK_LABEL_COLOR = TEXT_COLOR
    # Bone-white canvas + dark ink (SUPTITLE_COLOR was bound to the old grey at
    # import, so set it explicitly).
    style.BG_COLOR = BG_COLOR
    style.SPINE_COLOR = SPINE_COLOR
    style.SUPTITLE_COLOR = TEXT_COLOR


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def _loop_rgb() -> np.ndarray:
    return np.array([to_rgb(c) for c in colormaps.RWBK_LOOP_COLORS])


def _build_replica(loop_rgb: np.ndarray) -> Image.Image:
    """Snap the source scan to the 13-color palette on its native grid."""
    im = Image.open(SOURCE_SCAN).convert("RGB")
    w, h = im.size
    pal255 = loop_rgb * 255
    blocks = np.asarray(im.resize((GRID_COLS, GRID_ROWS), Image.BOX)).astype(float)
    near = (((blocks.reshape(-1, 3)[:, None] - pal255[None]) ** 2).sum(-1)).argmin(1)
    return Image.fromarray(
        pal255[near].reshape(GRID_ROWS, GRID_COLS, 3).astype(np.uint8)
    ).resize((w, h), Image.NEAREST)


def _pad_square(pil_img: Image.Image) -> np.ndarray:
    """Center the (portrait) image on a square BG-colored canvas so a
    fill-to-cell HeatmapPanel shows it undistorted."""
    arr = np.asarray(pil_img).astype(np.uint8)
    h, w = arr.shape[:2]
    side = max(h, w)
    bg = np.array([round(c * 255) for c in to_rgb(style.BG_COLOR)], dtype=np.uint8)
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = bg
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def _square_cwt() -> np.ndarray:
    """The wide (96x256) CWT resized to a 96x96 square so its showcase panel
    matches Centroid/Weave (avoids cross-column title overflow)."""
    wide = _build_cwt()
    sq = Image.fromarray((wide * 255).astype(np.uint8)).resize((96, 96), Image.BILINEAR)
    return np.asarray(sq).astype(float) / 255.0


# ---------------------------------------------------------------------------
# Derived spectra — the six spectrum columns of the bottom matrix.
#
# Vertical period (symmetric loop): BLACK -> warm -> WHITE*2 -> cool -> BLACK,
# with black at BOTH ends (2 steps) mirroring white's 2 center steps. The four
# derived palettes all spin off the same melt loop; cols 5/6 are its two halves
# split at the center white.
# ---------------------------------------------------------------------------
_C = colormaps
_WARM = [_C.OPTI_MAROON, _C.OPTI_RED, _C.OPTI_ORANGE, _C.OPTI_YELLOW]
_COOL = [_C.OPTI_FROST, _C.OPTI_AQUA, _C.OPTI_CYAN, _C.OPTI_ROYAL,
         _C.OPTI_OCEAN, _C.OPTI_VIOLET, _C.OPTI_NAVY]
_COOL_CONDENSED = [_C.OPTI_FROST, _C.OPTI_CYAN, _C.OPTI_OCEAN, _C.OPTI_NAVY]
_BLK, _WHT = _C.OPTI_BLACK, _C.OPTI_WHITE

# col 3 — raw vertical period (asymmetric: cool keeps all 7).
_PERIOD = [_BLK] + _WARM + [_WHT, _WHT] + _COOL + [_BLK]
# col 4 — white forced dead-center; blues condensed to match the warm count.
_WHITE_CENTERED = [_BLK] + _WARM + [_WHT] + _COOL_CONDENSED + [_BLK]
# col 5 / col 6 — both run BLACK -> WHITE (monotonic magnitude maps). Warm
# climbs black -> warm -> white; cool climbs black -> cool(dark->light) -> white.
_WARM_FROM_WHITE = [_BLK, _C.OPTI_MAROON, _C.OPTI_RED, _C.OPTI_ORANGE,
                    _C.OPTI_YELLOW, _WHT]
_COOL_FROM_WHITE = [_BLK] + _COOL[::-1] + [_WHT]


def _disc(colors) -> ListedColormap:
    """Discrete (hard-edged) palette as a ListedColormap."""
    return ListedColormap(list(colors))


def _cont(colors, name) -> LinearSegmentedColormap:
    """Equal-spaced smooth interpolation of the palette."""
    return LinearSegmentedColormap.from_list(name, list(colors), N=256)


def _make_spectra() -> list:
    """The six spectrum columns: (name, discrete cmap, continuous cmap)."""
    block13 = list(_C.RWBK_LOOP_COLORS)
    return [
        dict(name="Block",
             disc=_disc(block13), cont=colormaps.get(BLOCK_CONTINUOUS_NAME)),
        dict(name="Proportional\nBlock",
             disc=colormaps.get("optichrome_rwbk_steps"),
             cont=colormaps.get("optichrome_rwbk")),
        dict(name="Vertical\nPeriod",
             disc=_disc(_PERIOD), cont=_cont(_PERIOD, "_c_period")),
        dict(name="White\nCentered",
             disc=_disc(_WHITE_CENTERED), cont=_cont(_WHITE_CENTERED, "_c_wcent")),
        dict(name="Heat",
             disc=_disc(_WARM_FROM_WHITE), cont=_cont(_WARM_FROM_WHITE, "_c_warm")),
        dict(name="Frost",
             disc=_disc(_COOL_FROM_WHITE), cont=_cont(_COOL_FROM_WHITE, "_c_cool")),
    ]


MATRIX_ROW_LABELS = ["Block", "Continuous", "Centroid", "Energy\nSpectrum", "Weave"]
MATRIX_HEADER_FONT_PT = 50
ROW_LABEL_FONT_PT = 46
ROW_LABEL_X = 0.018   # figure-fraction x for the rotated row labels


# ---------------------------------------------------------------------------
# Painting builders (top section)
# ---------------------------------------------------------------------------
def _muted_one_period(replica: Image.Image) -> np.ndarray:
    """Replica greyed out except ONE full vertical period of the color cycle.

    A single grid column is let through, but cropped vertically to exactly one
    black-to-black period: from its first black grid row to the next
    non-adjacent black row. Read up the strip it runs
    black -> heat -> white -> frost -> black."""
    arr = np.asarray(replica.convert("RGB")).astype(float)
    h, w, _ = arr.shape
    muted = np.full_like(arr, 255.0)   # white out everything outside the period

    c = 3  # a vertical period from the LEFT-hand side
    fam = _replica_families()[:, c]          # color-loop index per grid row
    black_rows = [r for r, f in enumerate(fam) if f == 0]   # black == loop index 0
    r0 = black_rows[0] if black_rows else 0
    r1 = next((r for r in black_rows if r > r0 + 1), GRID_ROWS - 1)

    col_w, row_h = w / GRID_COLS, h / GRID_ROWS
    x0, x1 = int(round(c * col_w)), int(round((c + 1) * col_w))
    y0, y1 = int(round(r0 * row_h)), int(round((r1 + 1) * row_h))
    out = muted.copy()
    out[y0:y1, x0:x1, :] = arr[y0:y1, x0:x1, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _replica_families() -> np.ndarray:
    """Nearest color-loop family index for every (row, col) block of the
    replica grid — same snap as _build_replica, kept as integer indices."""
    pal255 = _loop_rgb() * 255
    im = Image.open(SOURCE_SCAN).convert("RGB")
    blocks = np.asarray(im.resize((GRID_COLS, GRID_ROWS), Image.BOX)).astype(float)
    near = (((blocks.reshape(-1, 3)[:, None] - pal255[None]) ** 2).sum(-1)).argmin(1)
    return near.reshape(GRID_ROWS, GRID_COLS)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def _img_panel(title: str, rgb_arr: np.ndarray, units=(2, 1)) -> HeatmapPanel:
    s_h, s_w = rgb_arr.shape[:2]
    p = HeatmapPanel(units=units, title=title,
                     show_xticklabels=False, show_yticklabels=False)
    p.add(Heatmap(rgb_arr, extent=(0, s_w, 0, s_h), origin="upper", vmin=0, vmax=255))
    return p


def _strip_panel(cmap, units=(1, 1)) -> HeatmapPanel:
    """A thin full-width gradient strip for one cmap (discrete or continuous)."""
    grad = np.linspace(0.0, 1.0, 512)[None, :]
    p = HeatmapPanel(units=units, title=None,
                     show_xticklabels=False, show_yticklabels=False)
    p.add(Heatmap(grad, cmap=cmap, extent=(0, 512, 0, 1), vmin=0.0, vmax=1.0))
    return p


def _stacked_strips(spec):
    """Block (discrete) over Continuous, stacked, stretched across a wide cell.

    Returns (composite, block_panel, continuous_panel) so the two strips can be
    squeezed into the top / bottom thirds of the row in a post-render pass
    (see _shrink_strip_pairs)."""
    block = _strip_panel(spec["disc"])
    cont = _strip_panel(spec["cont"])
    comp = CompositePanel(rows=[[block], [cont]], units=(STRIP_UNITS, 1))
    return comp, block, cont


def _shrink_strip_pairs(strip_pairs) -> None:
    """Squeeze each block/continuous pair to the top and bottom THIRDS of its
    row, leaving the middle third empty — block hugs the top edge, continuous
    hugs the bottom edge, both at one third of the row's height."""
    for block, cont in strip_pairs:
        if block.ax is None or cont.ax is None:
            continue
        bb_block = block.ax.get_position()
        bb_cont = cont.ax.get_position()
        y_top = bb_block.y1          # outer top edge of the pair
        y_bottom = bb_cont.y0        # outer bottom edge of the pair
        x0, width = bb_block.x0, bb_block.width
        third = (y_top - y_bottom) / 3.0
        block.ax.set_position([x0, y_top - third, width, third])
        cont.ax.set_position([x0, y_bottom, width, third])


def _square_field_plots(fig, spectrum_rows, strip_pairs) -> None:
    """Square the three field plots (Centroid / Energy Spectrum / Weave) in each
    row. Their cells render wider than tall (the width unit stretches to fill the
    figure while the height unit is fixed), so trim each to a centered square of
    its own height — keeps row height + column-header alignment, just drops the
    extra width.

    Centering each square leaves a side pad inside its cell, so the gutter
    between plots grows by twice that pad while the strip (full-width, flush
    right) keeps a tighter gutter to the first plot. Inset every strip's right
    edge by one side-pad so the strip->first-plot gutter matches the rest."""
    mfig = fig._mpl_fig
    w_in, h_in = mfig.get_size_inches()
    side_pad = 0.0
    for row in spectrum_rows:
        for panel in row[1:]:           # row[0] is the full-width strip column
            if panel.ax is None:
                continue
            bb = panel.ax.get_position()
            side = min(bb.width * w_in, bb.height * h_in)
            new_w, new_h = side / w_in, side / h_in
            side_pad = (bb.width - new_w) / 2.0
            new_x0 = bb.x0 + side_pad
            new_y0 = bb.y0 + (bb.height - new_h) / 2.0
            panel.ax.set_position([new_x0, new_y0, new_w, new_h])

    if side_pad:
        for block, cont in strip_pairs:
            for ax in (block.ax, cont.ax):
                if ax is None:
                    continue
                bb = ax.get_position()
                ax.set_position([bb.x0, bb.y0, bb.width - side_pad, bb.height])


def _field_panel(data: np.ndarray, cmap, units=(FIELD_UNITS, 1)) -> HeatmapPanel:
    p = HeatmapPanel(units=units, title=None,
                     show_xticklabels=False, show_yticklabels=False)
    p.add(Heatmap(data, cmap=cmap,
                  extent=(0, data.shape[1], 0, data.shape[0]), vmin=0.0, vmax=1.0))
    return p


def _separator_panel() -> HeatmapPanel:
    """An invisible spacer row; the actual divider is a thin white line drawn
    across its center in post (see _draw_separator)."""
    bg = int(round(to_rgb(style.BG_COLOR)[0] * 255))
    bar = np.full((1, 2, 3), bg, dtype=np.uint8)
    p = HeatmapPanel(units=(TOTAL_UNITS, 1), title=None, show_border=False,
                     show_xticklabels=False, show_yticklabels=False)
    p.add(Heatmap(bar, extent=(0, 2, 0, 1), origin="upper", vmin=0, vmax=255))
    return p


def _draw_separator(fig, sep_panel) -> None:
    """A clean thin white horizontal line across the separator band."""
    from matplotlib.lines import Line2D
    if sep_panel.ax is None:
        return
    bb = sep_panel.ax.get_position()
    y = (bb.y0 + bb.y1) / 2.0
    fig._mpl_fig.add_artist(Line2D(
        [bb.x0, bb.x1], [y, y], color=SEPARATOR_COLOR, linewidth=1.0, alpha=0.9,
        transform=fig._mpl_fig.transFigure, zorder=60, clip_on=False,
    ))


# ---------------------------------------------------------------------------
# Figure assembly — top 3-cell section · separator · 6x5 spectrum matrix.
# ---------------------------------------------------------------------------
def _build_figure():
    _apply_style_knobs()
    _register_block_continuous()

    spectra = _make_spectra()
    loop_rgb = _loop_rgb()
    replica = _build_replica(loop_rgb)
    orig_sq = _pad_square(Image.open(SOURCE_SCAN).convert("RGB"))
    repl_sq = _pad_square(replica)
    muted_sq = _pad_square(Image.fromarray(_muted_one_period(replica)))

    fields = {
        "centroid": _build_centroid(),
        "cwt": _square_cwt(),
        "weave": _build_weave(base_freq_hz=2.0),
    }

    # Top section — three palette-origin cells. TOTAL=10 isn't divisible by 3,
    # so widths are [3,4,3]; the square-padded paintings letterbox to equal
    # size regardless of cell width.
    side_w = TOTAL_UNITS // 3
    mid_w = TOTAL_UNITS - 2 * side_w
    top_panels = [
        _img_panel("Original Palette", orig_sq, units=(side_w, 1)),
        _img_panel("Replica Palette", repl_sq, units=(mid_w, 1)),
        _img_panel("One Vertical Period", muted_sq, units=(side_w, 1)),
    ]

    # Section 2 — one wide ROW per spectrum:
    #   [stacked block+continuous strips] · Centroid · Energy Spectrum · Weave
    spectrum_rows = []
    strip_pairs = []
    for s in spectra:
        comp, block, cont = _stacked_strips(s)
        strip_pairs.append((block, cont))
        spectrum_rows.append([
            comp,
            _field_panel(fields["centroid"], s["cont"]),
            _field_panel(fields["cwt"], s["cont"]),
            _field_panel(fields["weave"], s["cont"]),
        ])

    sep_panel = _separator_panel()
    fig = Figure.compose(
        rows=[
            [SuptitlePanel(SUPTITLE_TEXT, units=(TOTAL_UNITS, 1), font_size=SUPTITLE_FONT_PT)],
            top_panels,
            [sep_panel],
            *spectrum_rows,
        ],
        row_heights=([1.8, TOP_ROW_HEIGHT, 0.30]
                     + [SPECTRUM_ROW_HEIGHT] * len(spectrum_rows)),
        unit_inches=UNIT_INCHES,
        top_reserve_inches=TOP_RESERVE_INCHES,
        show_cell_borders=False,
    )
    fig.render()

    # Square-padded paintings: equal aspect so they stay undistorted.
    for p in top_panels:
        if p.ax is not None:
            p.ax.set_aspect("equal", anchor="C")

    _shrink_strip_pairs(strip_pairs)
    _square_field_plots(fig, spectrum_rows, strip_pairs)
    _draw_separator(fig, sep_panel)
    _draw_section2_chrome(fig, spectrum_rows, spectra, strip_pairs)
    return fig


# Column headers over each spectrum row's four cells (drawn once, up top).
SECTION2_COL_HEADERS = ["Spectrum", "Centroid", "Energy Spectrum", "Weave"]


def _draw_section2_chrome(fig, spectrum_rows, spectra, strip_pairs) -> None:
    """Column headers (Spectrum / Centroid / Energy Spectrum / Weave) above the
    first row, plus each spectrum's name centered horizontally in the empty gap
    between its block (top) and continuous (bottom) strips."""
    mfig = fig._mpl_fig
    _, fig_h_in = mfig.get_size_inches()

    # Column headers — centered above the first spectrum row's four cells.
    first_row = spectrum_rows[0]
    header_y = max(p.ax.get_position().y1 for p in first_row if p.ax is not None)
    header_y += HEADER_GAP_INCHES / fig_h_in
    for panel, label in zip(first_row, SECTION2_COL_HEADERS):
        if panel.ax is None:
            continue
        bb = panel.ax.get_position()
        mfig.text((bb.x0 + bb.x1) / 2.0, header_y, label,
                  color=TEXT_COLOR, fontsize=MATRIX_HEADER_FONT_PT,
                  fontweight="bold", ha="center", va="bottom")

    # Spectrum names — horizontal, centered (both axes) in the empty middle-third
    # gap between each block strip (top) and continuous strip (bottom).
    for (block, cont), spec in zip(strip_pairs, spectra):
        if block.ax is None or cont.ax is None:
            continue
        bb_block = block.ax.get_position()
        bb_cont = cont.ax.get_position()
        xc = (bb_block.x0 + bb_block.x1) / 2.0
        yc = (bb_block.y0 + bb_cont.y1) / 2.0
        name = spec["name"].replace("\n", " ")
        mfig.text(xc, yc, name, color=TEXT_COLOR, fontsize=ROW_LABEL_FONT_PT,
                  fontweight="bold", ha="center", va="center")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> str:
    """Build, render, and save the figure. Returns the absolute output path."""
    fig = _build_figure()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    print(render())
