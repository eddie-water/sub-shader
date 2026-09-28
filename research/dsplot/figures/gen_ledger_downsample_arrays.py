"""Ledger pedagogy - the down-sample change on a toy array (n=16 → 4 bins).

The Fix 3 / overlap discussion in numbers small enough to read: one 16-column
input with two transients - a 7 that happens to sit on a decimation pick, a 9
that doesn't - reduced to 4 display bins by every strategy under discussion:

    before  - index decimation: keep column 0 of each 4-block, drop the rest
    after   - block max-pool: max over each 4-block (Fix 3 as shipped)
    diff    - after − before, bin by bin
    overlap - overlapped max-pool mock: window 8, stride 4

Chrome is the figure-1 stacked template (header band, caption squares, cell
borders, ``_contender_tight_style``) with array diagrams in the plot cells
instead of heatmaps - cells are tinted by value on the heatmap's own
colormap so the arrays read as one-row scalograms. Iteration outputs land in
NEW ``ledger_downsample_arrays_v*.png`` files - never overwrite a PNG.
"""
from __future__ import annotations

import os

from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb

from .. import Figure, BarPanel, Barh, Line, Annotation, SuptitlePanel, style
from .gen_figure_1_stft_vs_cwt import (
    _contender_tight_style, _side_text_panel,
    CONTENDER_CAPTION_FONT_SIZE,
    STACK_PLOT_UNITS, STACK_TEXT_UNITS, STACK_PLOT_ROW_HEIGHT,
    STACK_BAND_HEIGHT, STACK_W_UNIT_INCHES, STACK_UNIT_INCHES,
    STACK_GUTTER_INCHES,
)
from .gen_timing_gantt import _repo_root

# --- the toy signal -----------------------------------------------------------
N, BINS = 16, 4
BLOCK = N // BINS             # 4 columns per display bin (pipeline: 128)
VMAX = 9.0
INPUT = [1, 1, 1, 1, 7, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1]
KEPT = [b * BLOCK for b in range(BINS)]          # decimation's picks: 0,4,8,12

BEFORE = [INPUT[i] for i in KEPT]                          # [1, 7, 1, 1]
AFTER = [max(INPUT[b * BLOCK:(b + 1) * BLOCK]) for b in range(BINS)]  # [1,9,1,1]
DIFF = [a - b for a, b in zip(AFTER, BEFORE)]              # [0, 2, 0, 0]
OVERLAP = [max(AFTER[b:b + 2]) for b in range(BINS)]       # [9, 9, 1, 1]

ROWS = (
    ("before", "index decimation - keep column 0 of each block, drop the "
               "other 3. The 7 sits on a pick and survives by luck; the 9 "
               "falls between picks and vanishes."),
    ("after", "block max-pool - every column lands in exactly one block, the "
              "loudest wins. The 9 cannot be missed. (Fix 3 as shipped.)"),
    ("diff", "after − before, bin by bin - the energy decimation lost. In "
             "the scalogram figure this is the diff row lighting up on "
             "hi-hat columns."),
    ("overlap", "overlapped max-pool - window 8, stride 4: each bin also "
                "sees the next block. The 9 now covers two bins - smoother, "
                "but the attack smears one bin wide."),
)

# --- geometry inside each plot cell (data coordinates) ------------------------
XLIM = (-2.4, 16.4)
YLIM = (0.0, 8.0)
IN_Y, IN_H = 6.9, 1.2          # input line: center / height
OUT_Y, OUT_H = 1.6, 2.2        # output line: center / height
BRACKET_Y = 5.75               # pool brackets sit between the two lines
NUM_SIZE = 44                  # one type size for every number
SYM_SIZE = 60                  # the diff row's − and = glyphs
SEAM = 0.06                    # gap between adjacent cells (data units)
CONNECT_LW = 3.0
BG = style.BG_COLOR


def _val_color(v: float) -> str:
    return to_hex(colormaps[style.DEFAULT_HEATMAP_CMAP](v / VMAX))


def _dim(color: str, keep: float = 0.35) -> str:
    r, g, b = to_rgb(color)
    br, bg_, bb = to_rgb(BG)
    return to_hex((keep * r + (1 - keep) * br, keep * g + (1 - keep) * bg_,
                   keep * b + (1 - keep) * bb))


def _num_color(v: float) -> str:
    # inferno runs dark → bright yellow: dark numbers on the bright cells.
    return "#000000" if v >= 0.6 * VMAX else "#FFFFFF"


def _cells(panel, values, lefts, width, y, h, *, dim_mask=None,
           outline_mask=None) -> None:
    """One line of value-tinted cells with centered numbers."""
    for i, (v, x0) in enumerate(zip(values, lefts)):
        dim = dim_mask[i] if dim_mask is not None else False
        color = _dim(_val_color(v)) if dim else _val_color(v)
        outlined = outline_mask[i] if outline_mask is not None else False
        panel.add(Barh([y], [width - SEAM], lefts=[x0 + SEAM / 2.0],
                       colors=[color], height=h,
                       edgecolor="#FFFFFF" if outlined else None,
                       edgewidth=4.0 if outlined else 0.0))
        panel.add(Annotation(f"{v:g}", (x0 + width / 2.0, y),
                             ha="center", va="center",
                             color="#555555" if dim else _num_color(v),
                             fontsize=NUM_SIZE, fontweight="bold"))


def _bracket(panel, x0, x1, y, *, tick: float = 0.4) -> None:
    for xs, ys in (((x0, x1), (y, y)),
                   ((x0, x0), (y, y + tick)), ((x1, x1), (y, y + tick))):
        panel.add(Line(list(xs), list(ys), color=style.NEUTRAL_COLOR,
                       linewidth=CONNECT_LW))


def _connector(panel, x_top, y_top, x_bot) -> None:
    panel.add(Line([x_top, x_bot], [y_top, OUT_Y + OUT_H / 2.0],
                   color=style.NEUTRAL_COLOR, linewidth=CONNECT_LW))


def _panel() -> BarPanel:
    return BarPanel(units=STACK_PLOT_UNITS, xlim=XLIM, ylim=YLIM,
                    xticks=[], xticklabels=[], show_xticklabels=False,
                    show_border=False)


def _before_panel() -> BarPanel:
    p = _panel()
    kept = [i in KEPT for i in range(N)]
    _cells(p, INPUT, list(range(N)), 1.0, IN_Y, IN_H,
           dim_mask=[not k for k in kept], outline_mask=kept)
    for b, i in enumerate(KEPT):
        _connector(p, i + 0.5, IN_Y - IN_H / 2.0, b * BLOCK + BLOCK / 2.0)
    _cells(p, BEFORE, [b * BLOCK for b in range(BINS)], BLOCK, OUT_Y, OUT_H)
    return p


def _after_panel() -> BarPanel:
    p = _panel()
    _cells(p, INPUT, list(range(N)), 1.0, IN_Y, IN_H)
    for b in range(BINS):
        x0, x1 = b * BLOCK + 0.15, (b + 1) * BLOCK - 0.15
        _bracket(p, x0, x1, BRACKET_Y)
        _connector(p, (x0 + x1) / 2.0, BRACKET_Y, b * BLOCK + BLOCK / 2.0)
    _cells(p, AFTER, [b * BLOCK for b in range(BINS)], BLOCK, OUT_Y, OUT_H)
    return p


def _diff_panel() -> BarPanel:
    p = _panel()
    rows = ((AFTER, 6.9, 1.4, None), (BEFORE, 4.45, 1.4, "−"),
            (DIFF, OUT_Y, OUT_H, "="))
    for values, y, h, sym in rows:
        if sym is not None:
            p.add(Annotation(sym, (-1.2, y), ha="center", va="center",
                             color=style.NEUTRAL_COLOR, fontsize=SYM_SIZE,
                             fontweight="bold"))
        _cells(p, values, [b * BLOCK for b in range(BINS)], BLOCK, y, h)
    return p


def _overlap_panel() -> BarPanel:
    p = _panel()
    _cells(p, INPUT, list(range(N)), 1.0, IN_Y, IN_H)
    for b in range(BINS):
        x0 = b * BLOCK + 0.15
        x1 = min((b + 2) * BLOCK, N) - 0.15        # window = this block + next
        y = BRACKET_Y if b % 2 == 0 else BRACKET_Y - 0.85
        _bracket(p, x0, x1, y)
        _connector(p, (x0 + x1) / 2.0, y, b * BLOCK + BLOCK / 2.0)
    _cells(p, OVERLAP, [b * BLOCK for b in range(BINS)], BLOCK, OUT_Y, OUT_H)
    return p


def build_figure() -> Figure:
    panels = (_before_panel(), _after_panel(), _diff_panel(), _overlap_panel())
    rows = [[panel,
             _side_text_panel(title, caption,
                              font_size=CONTENDER_CAPTION_FONT_SIZE)]
            for (title, caption), panel in zip(ROWS, panels)]

    total_w = STACK_PLOT_UNITS[0] + STACK_TEXT_UNITS[0]
    rows.insert(0, [SuptitlePanel(
        "Down-sample - 16 columns → 4 bins, block = 4",
        units=(total_w, 1), font_size=44)])
    rows.append([SuptitlePanel(
        "pipeline scale: 8,192 columns → 64 bins - block = 128 columns"
        " ≈ 2.9 ms of audio per bin",
        units=(total_w, 1), font_size=34)])
    row_heights = ([STACK_BAND_HEIGHT] + [STACK_PLOT_ROW_HEIGHT] * len(ROWS)
                   + [STACK_BAND_HEIGHT])
    u_w, u_h = STACK_W_UNIT_INCHES, STACK_UNIT_INCHES
    return Figure.compose(
        rows=rows, row_heights=row_heights,
        hspace=STACK_GUTTER_INCHES / (u_h * STACK_PLOT_ROW_HEIGHT),
        wspace=STACK_GUTTER_INCHES / u_w,
        dpi=150, unit_inches=u_w, unit_height_inches=u_h,
        show_cell_borders=True,
    )


def render(output_path: str | None = None) -> str:
    with _contender_tight_style():
        fig = build_figure()
        fig.render()
        for mpl_ax in fig._mpl_fig.axes:
            mpl_ax.tick_params(axis="both", which="both", length=0)
    path = output_path or os.path.join(_repo_root(), "assets", "timing",
                                       "ledger_downsample_arrays_v1.png")
    fig.savefig(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    import sys
    print(render(sys.argv[1] if len(sys.argv) > 1 else None))
