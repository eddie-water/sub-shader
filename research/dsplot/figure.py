"""Figure orchestrator — composes Panels into one matplotlib Figure.

Layout defaults (figsize, hspace, wspace, dpi) resolve LAZILY against
`dsplot.style.DEFAULT_*` at construction time (per D-05) — reassigning a
style constant between `Figure(...)` calls is observable through subsequent
gridspec configuration. The Figure NEVER hardcodes layout numerics.

Per D-04, `add_panel` accepts a `projection` kwarg so 3D cells (Axes3D) can
be composed via the same flow as 2D cells:
    fig = Figure(n_rows=1, n_cols=2)
    fig.add_panel(panel_3d, row=0, col=0, projection="3d")
    fig.add_panel(panel_2d, row=0, col=1)
    fig.render()
    fig.savefig("mixed.png")

A panel that accepts a 3D Axes is expected to do its own 2D-vs-3D branching
(or be a subclass — e.g. a future StaticPanel3D for the 3D foundation figure).
The Figure itself only wires the Axes; it does not adapt panel chrome to
dimensionality.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from . import style
from .panels.base import Panel
from .panels.dynamic_panel import DynamicPanel
from .panels.dynamic_panel_3d import DynamicPanel3D
from .panels.static_panel_3d import StaticPanel3D


# Trailing hold ticks appended to the master clock cycle: the longest panel
# (and therefore every shorter panel that already reached its final frame)
# stays put for this many ticks before the cycle wraps. Gives the eye a
# moment to read the completed state before the next loop starts.
_FIGURE_CLOCK_HOLD_TICKS = 2


_JUPYTER_DARK_CSS_INJECTED = False


def apply_jupyter_dark(bg_color: Optional[str] = None) -> None:
    """Inject one-time CSS into the current Jupyter session that paints the
    cell-output container behind ipympl figures (and other widget outputs)
    dark, matching `style.BG_COLOR`. Safe to call multiple times — the
    injection is idempotent within a session.

    No-op outside Jupyter (silently returns).
    """
    global _JUPYTER_DARK_CSS_INJECTED
    if _JUPYTER_DARK_CSS_INJECTED:
        return
    try:
        from IPython.display import display, HTML
    except ImportError:
        return
    bg = bg_color if bg_color is not None else style.BG_COLOR
    css = (
        "<style>"
        ".cell-output-ipywidget-background,"
        ".jupyter-matplotlib,"
        ".jupyter-widgets.widget-container,"
        ".jp-OutputArea-output {"
        f"  background-color: {bg} !important;"
        "}"
        "</style>"
    )
    display(HTML(css))
    _JUPYTER_DARK_CSS_INJECTED = True


def _composite_dynamic_children(panel) -> List["DynamicPanel"]:
    """DynamicPanel children one level inside a CompositePanel (empty otherwise).

    Lets the figure master clock reach dynamic panels that live inside a
    CompositePanel's sub-grid, so they animate in lockstep with top-level
    dynamic panels instead of each spawning an unsynced per-child timer.
    """
    from .panels.composite_panel import CompositePanel
    if not isinstance(panel, CompositePanel):
        return []
    return [
        child
        for row in panel.rows
        for child in row
        if isinstance(child, DynamicPanel)
    ]


class Figure:
    """Compose Panels into one matplotlib Figure with a gridspec layout."""

    def __init__(
        self,
        *,
        n_rows: int = 1,
        n_cols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        suptitle: Optional[str] = None,
        suptitle_fontsize: Optional[float] = None,
        suptitle_y: Optional[float] = None,
        suptitle_band_inches: Optional[float] = None,
        top_pad: Optional[float] = None,
        bottom_pad: Optional[float] = None,
        figure_number: Optional[str] = None,
        figure_caption: Optional[str] = None,
        figure_number_x: Optional[float] = None,
        footer_text: Optional[str] = None,
        footer_fontsize: Optional[float] = None,
        footer_band_inches: Optional[float] = None,
        bottom_reserve_is_explicit: bool = False,
        subtitle_band_inches: Optional[float] = None,
        subtitle_specs: Optional[List[dict]] = None,
        subtitle_band_center_inches: Optional[float] = None,
        subtitle_band_top_inches: Optional[float] = None,
        subtitle_band_bottom_inches: Optional[float] = None,
        hspace: Optional[float] = None,
        wspace: Optional[float] = None,
        width_ratios: Optional[List[float]] = None,
        height_ratios: Optional[List[float]] = None,
        dpi: Optional[int] = None,
        fill_width: bool = True,
        show_toolbar: bool = False,
        display_width: Optional[str] = None,
        debug_guides: bool = False,
        show_cell_borders: bool = False,
        hold_ticks: Optional[int] = None,
    ) -> None:
        if figsize is None:
            # Width scales by sum(width_ratios) so non-uniform columns produce
            # cells that match the panel's natural aspect ratio. Without this,
            # a column with width_ratio=1 inside a 3-column figsize=(15,10) is
            # 5x5 inches but a column with width_ratio=2 is 10x5 — and square
            # aspect="equal" panels in the wide column leave dead vertical
            # space that pushes panel titles away from their content.
            width_units = sum(width_ratios) if width_ratios else n_cols
            figsize = (
                style.DEFAULT_PANEL_SIZE_INCHES * width_units,
                style.DEFAULT_PANEL_SIZE_INCHES * n_rows,
            )
        if hspace is None:
            hspace = style.DEFAULT_HSPACE
        if wspace is None:
            wspace = style.DEFAULT_WSPACE
        if dpi is None:
            dpi = style.DEFAULT_DPI

        self.n_rows = n_rows
        self.n_cols = n_cols
        self._fill_width = fill_width
        self._show_toolbar = show_toolbar
        self._display_width = display_width
        self._has_suptitle = suptitle is not None
        self._suptitle_band_inches = suptitle_band_inches
        self._top_pad = top_pad
        self._bottom_pad = bottom_pad
        self._figure_number = figure_number
        self._figure_caption = figure_caption
        self._footer_text = footer_text
        self._footer_fontsize = footer_fontsize
        self._footer_band_inches = footer_band_inches
        self._bottom_reserve_is_explicit = bottom_reserve_is_explicit
        self._subtitle_band_inches = subtitle_band_inches
        self._subtitle_specs = subtitle_specs or []
        self._subtitle_band_center_inches = subtitle_band_center_inches
        self._subtitle_band_top_inches = subtitle_band_top_inches
        self._subtitle_band_bottom_inches = subtitle_band_bottom_inches
        self._debug_guides = debug_guides
        self._show_cell_borders = show_cell_borders
        self._hold_ticks = hold_ticks
        # One sans family for all non-numeric text across every figure (the
        # single type-system knob). Set on rcParams so every text artist
        # inherits it; numeric readouts pass family= explicitly and override.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [style.DEFAULT_FONT_FAMILY, "DejaVu Sans"]
        self._mpl_fig = plt.figure(figsize=figsize, dpi=dpi)
        self._mpl_fig.patch.set_facecolor(style.BG_COLOR)
        self._gs = self._mpl_fig.add_gridspec(
            n_rows, n_cols,
            hspace=hspace, wspace=wspace,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        if suptitle is not None:
            # Suptitle lives in a band one `margin` tall at the top of the
            # figure with its text V-centered in that band — equal breathing
            # room above and below the text. Below the band is another full
            # `margin` of gap before the first panel border. So the total
            # top reserve is `2 * margin`.
            _, fig_h = figsize
            sup_fontsize_resolved = (
                suptitle_fontsize if suptitle_fontsize is not None
                else style.DEFAULT_SUPTITLE_FONT_SIZE
            )
            sup_band_center_in = style.DEFAULT_MARGIN_INCHES / 2.0
            self._mpl_fig.suptitle(
                suptitle,
                # Header color is its OWN knob (SUPTITLE_COLOR), independent of
                # tick/body color, so a figure can keep white body text while its
                # header matches the shared gray header tone. Default
                # SUPTITLE_COLOR == TICK_LABEL_COLOR (#888), so unchanged figures
                # are unaffected.
                color=style.SUPTITLE_COLOR,
                fontsize=sup_fontsize_resolved,
                fontweight="bold",
                y=(
                    suptitle_y if suptitle_y is not None
                    else 1.0 - sup_band_center_in / fig_h
                ),
                va="center",
            )

        # Footer band — mirror of the suptitle band, anchored to the figure
        # BOTTOM. Compose lifts the last row's SuptitlePanel here so the
        # footer text sits in a clean band identical in height to the top
        # suptitle band, with no surrounding gridspec chrome.
        if footer_text is not None and footer_band_inches is not None:
            _, fig_h = figsize
            footer_fontsize_resolved = (
                footer_fontsize if footer_fontsize is not None
                else style.DEFAULT_SUPTITLE_FONT_SIZE
            )
            footer_y = (footer_band_inches / 2.0) / fig_h
            self._mpl_fig.text(
                0.5, footer_y, footer_text,
                color=style.SUPTITLE_COLOR,
                fontsize=footer_fontsize_resolved,
                fontweight=style.SUPTITLE_WEIGHT,
                ha="center", va="center",
                transform=self._mpl_fig.transFigure,
            )

        # Figure-level `Figure N` identifier + explanatory caption rendered at
        # the bottom of the figure. Counterpart to the per-panel subtitle +
        # caption — same typographic hierarchy (bold italic identifier,
        # smaller italic-bold caption) but in the figure-bottom reserve band.
        # `figure_number` is centered horizontally and sits in its own padded
        # band; `figure_caption` is LEFT-aligned to the leftmost panel cell's
        # left edge so multi-line captions read like a paragraph caption.
        if figure_number is not None or figure_caption is not None:
            fig_w, fig_h = figsize
            margin = style.DEFAULT_MARGIN_INCHES
            # Identifier sits in the upper portion of the bottom reserve, with
            # its own ~margin of padding above and below for visual breathing
            # room before the caption.
            # figure_number sits well below the row 3 x-axis label (which
            # lands ~DEFAULT_AXIS_LABEL_INSET_INCHES below the bottom plot
            # spine). With breathing room above (x-label gap) and below
            # (caption gap), the identifier reads as its own padded line.
            if figure_number is not None:
                self._mpl_fig.text(
                    figure_number_x if figure_number_x is not None else 0.5,
                    style.DEFAULT_FIGURE_NUMBER_Y_INCHES / fig_h,
                    figure_number,
                    color=style.TICK_LABEL_COLOR,
                    fontsize=style.DEFAULT_FIGURE_NUMBER_FONT_SIZE,
                    fontweight=style.DEFAULT_FIGURE_NUMBER_FONT_WEIGHT,
                    style=style.DEFAULT_FIGURE_NUMBER_FONT_STYLE,
                    ha="center", va="center",
                )
            # Caption: left edge anchors to the leftmost panel cell's left
            # border (figure left margin); top anchored at va="top" so the
            # first line sits at a known y and subsequent lines extend
            # downward toward the figure bottom — predictable layout when
            # the caption is multi-line.
            if figure_caption is not None:
                self._mpl_fig.text(
                    margin / fig_w, style.DEFAULT_FIGURE_CAPTION_Y_INCHES / fig_h,
                    figure_caption,
                    color=style.TICK_LABEL_COLOR,
                    fontsize=style.DEFAULT_FIGURE_CAPTION_FONT_SIZE,
                    fontweight=style.DEFAULT_FIGURE_CAPTION_FONT_WEIGHT,
                    style=style.DEFAULT_FIGURE_CAPTION_FONT_STYLE,
                    ha="left", va="top",
                )

        self.panels: List[Tuple[Panel, int, int, int, int, Optional[str]]] = []
        # Master FuncAnimation when any DynamicPanels are present; load-bearing
        # for matplotlib timer GC the same way DynamicPanel._anim is in the
        # standalone path.
        self._anim = None

    @classmethod
    def compose(
        cls,
        *,
        rows: List[List["Panel"]],
        suptitle: Optional[str] = None,
        suptitle_fontsize: Optional[float] = None,
        unit_inches: Optional[float] = None,
        unit_height_inches: Optional[float] = None,
        total_width_inches: Optional[float] = None,
        total_height_inches: Optional[float] = None,
        header_band_inches: Optional[float] = None,
        row_heights: Optional[List[float]] = None,
        dpi: Optional[int] = None,
        hspace: Optional[float] = None,
        wspace: Optional[float] = None,
        debug_guides: bool = False,
        show_cell_borders: bool = False,
        frame_inset: bool = False,
        top_reserve_inches: Optional[float] = None,
        bottom_reserve_inches: Optional[float] = None,
        figure_number: Optional[str] = None,
        figure_caption: Optional[str] = None,
        hold_ticks: Optional[int] = None,
    ) -> "Figure":
        """Auto-derive figsize, gridspec width-units, and per-panel colspan
        from each panel's `units` (lego-block composition).

        Every row must have the same total width-units; mismatched rows raise
        ValueError. The resulting Figure uses a uniform gridspec sized
        `(n_rows, total_units_per_row)`, with each panel placed via
        `add_panel(..., colspan=panel.units[0])`. StaticPanel3D children get
        `projection="3d"` automatically.
        """
        n_rows = len(rows)
        if n_rows == 0:
            raise ValueError("Figure.compose requires at least one row")

        if row_heights is None:
            row_heights = [1.0] * n_rows
        elif len(row_heights) != n_rows:
            raise ValueError(
                f"Figure.compose row_heights length {len(row_heights)} "
                f"!= n_rows {n_rows}"
            )

        width_units_per_row = [sum(p.units[0] for p in row) for row in rows]
        if any(w != width_units_per_row[0] for w in width_units_per_row):
            raise ValueError(
                f"Figure.compose row widths mismatch: {width_units_per_row}"
            )

        # Common-canvas normalization: back-solve unit_inches so the finished
        # figure is exactly `total_width_inches` wide. Width is
        #   n_cols·unit_inches + (n_cols-1)·col_gutter + 2·margin
        # so unit_inches = (W - 2·margin - (n_cols-1)·col_gutter) / n_cols.
        # unit_height defaults to unit_inches downstream, so the whole figure
        # scales uniformly to the target width — same proportions, one canvas
        # size shared across every figure (→ a shared pt type scale is visually
        # consistent everywhere). Overrides any explicit unit_inches.
        if total_width_inches is not None:
            _ncols_w = width_units_per_row[0]
            if wspace is not None:
                # Explicit wspace: width = unit·(n + (n-1)·wspace) + 2·margin.
                unit_inches = (
                    total_width_inches - 2.0 * style.DEFAULT_MARGIN_INCHES
                ) / (_ncols_w + (_ncols_w - 1) * wspace)
            else:
                unit_inches = style.unit_inches_for_width(
                    _ncols_w, total_width_inches
                )

        # If row 0 is a single SuptitlePanel spanning all cols, lift it out of
        # the gridspec and render it as figure-level chrome in the top_reserve
        # band. Otherwise an explicit row-0 suptitle picks up the full row
        # gutter below its cell (mpl `hspace` × avg_cell, never zero), which
        # visually overwhelms a quarter-height cell with a half-height gap.
        # Putting it in the top_reserve gives a band of exactly
        # `row_heights[0] × unit_height_inches`, no extra gutter.
        from .panels.suptitle_panel import SuptitlePanel
        _suptitle_band_inches: Optional[float] = None
        _footer_band_inches: Optional[float] = None
        _footer_text: Optional[str] = None
        _footer_fontsize: Optional[float] = None

        def _resolved_unit_height() -> float:
            return (
                unit_height_inches if unit_height_inches is not None
                else (unit_inches if unit_inches is not None
                      else style.DEFAULT_PANEL_UNIT_INCHES)
            )

        # Lift row-0 SuptitlePanel into the top band.
        if (
            len(rows[0]) == 1
            and isinstance(rows[0][0], SuptitlePanel)
            and rows[0][0].units[0] == width_units_per_row[0]
        ):
            sup_panel = rows[0][0]
            if suptitle is None:
                suptitle = sup_panel.text
            if suptitle_fontsize is None:
                suptitle_fontsize = sup_panel.font_size
            # Fixed physical band when the caller pins it (header_band_inches),
            # else the legacy relative `row_heights[0] × unit_height`. The fixed
            # path makes every figure's header band identical in inches on the
            # shared canvas regardless of column count (unit_height varies ~3×
            # between a 3-col and 9-col figure, so the relative path renders very
            # different bands). row_heights[0] is sliced off below and unused
            # once the band height is set, so pinning it is complete here.
            _suptitle_band_inches = (
                header_band_inches if header_band_inches is not None
                else row_heights[0] * _resolved_unit_height()
            )
            rows = rows[1:]
            row_heights = row_heights[1:]
            n_rows = len(rows)
            if n_rows == 0:
                raise ValueError(
                    "Figure.compose: row 0 was a SuptitlePanel, but no body "
                    "rows remain after lifting it into the top_reserve band."
                )
            width_units_per_row = [sum(p.units[0] for p in row) for row in rows]

        # Lift row-(-1) SuptitlePanel into the bottom band. Mirrors the top
        # lift: gives the bottom SuptitlePanel a clean band anchored to the
        # figure bottom with no surrounding gridspec chrome, so footer height
        # matches header height exactly.
        if (
            n_rows > 0
            and len(rows[-1]) == 1
            and isinstance(rows[-1][0], SuptitlePanel)
            and rows[-1][0].units[0] == width_units_per_row[-1]
        ):
            footer_panel = rows[-1][0]
            _footer_text = footer_panel.text
            _footer_fontsize = footer_panel.font_size
            _footer_band_inches = row_heights[-1] * _resolved_unit_height()
            rows = rows[:-1]
            row_heights = row_heights[:-1]
            n_rows = len(rows)
            if n_rows == 0:
                raise ValueError(
                    "Figure.compose: last row was a SuptitlePanel, but no "
                    "body rows remain after lifting it into the bottom band."
                )
            width_units_per_row = [sum(p.units[0] for p in row) for row in rows]

        # Lift a MULTI-column SuptitlePanel row (one label per body column) into
        # a thin band directly beneath the body. Mirrors the suptitle/footer
        # band treatment: a row of per-panel subtitles otherwise becomes a real
        # gridspec row whose `show_cell_borders` rectangle balloons to absorb
        # the full inter-row gutter (the label floats far below its panel and
        # reads as a tall empty box). Lifting renders each label as compact
        # figure chrome centered under its column, the same height as the
        # suptitle/footer bands. Detected AFTER the footer lift so a single
        # full-width footer SuptitlePanel is claimed by the footer band first.
        _subtitle_band_inches: Optional[float] = None
        _subtitle_specs: Optional[List[dict]] = None
        if (
            n_rows > 1
            and len(rows[-1]) >= 1
            and all(isinstance(p, SuptitlePanel) for p in rows[-1])
            and sum(p.units[0] for p in rows[-1]) == width_units_per_row[-1]
        ):
            sub_row = rows[-1]
            _subtitle_band_inches = row_heights[-1] * _resolved_unit_height()
            _subtitle_specs = []
            col_cursor = 0
            for p in sub_row:
                _subtitle_specs.append({
                    "text": p.text,
                    "col_start": col_cursor,
                    "colspan": p.units[0],
                    "font_size": p.font_size,
                })
                col_cursor += p.units[0]
            rows = rows[:-1]
            row_heights = row_heights[:-1]
            n_rows = len(rows)
            width_units_per_row = [sum(p.units[0] for p in row) for row in rows]

        n_cols = width_units_per_row[0]
        unit_inches = (
            unit_inches if unit_inches is not None
            else style.DEFAULT_PANEL_UNIT_INCHES
        )
        unit_height_inches_resolved = (
            unit_height_inches if unit_height_inches is not None
            else unit_inches
        )
        # Outer perimeter: `margin` between every panel border and the figure
        # edge (left, right, bottom). Tick + axis labels render WITHIN that
        # margin (eating into it from the spine outward).  Suptitle gets its
        # OWN `margin` of padding above it, plus another `margin` below it
        # before the panel grid starts.  Inter-cell gutters use a separate
        # `gutter` sized to host one set of axis labels + one panel title
        # between adjacent cells.
        margin = style.DEFAULT_MARGIN_INCHES
        # mpl's hspace/wspace are interpreted as fractions of the AVERAGE cell
        # dimension, not as multiples of unit_inches. The naive formula
        # `gutter / unit_inches` only lands the right gap when every cell is
        # uniform — once row_heights has a 0.25-tall suptitle row, mpl's
        # proportional allocation produces both gap and cell sizes that drift
        # from the declared inch values. Closed-form fix (derived from mpl's
        # gridspec law `gap = hspace × avg_cell`, given target gap = g and
        # cells = row_heights[i] × unit_height):
        #     hspace = g × n_rows / sum(cell_heights_inches)
        # Yields cells at exactly `row_heights[i] × unit_height_inches` and
        # gaps at exactly `g` inches.
        col_widths_inches = [unit_inches] * n_cols
        row_heights_inches = [r * unit_height_inches_resolved for r in row_heights]
        col_gutter_in = style.DEFAULT_COLUMN_GUTTER_INCHES
        row_gutter_in = style.DEFAULT_GUTTER_INCHES
        if wspace is not None:
            wspace_resolved = wspace
        elif n_cols > 1:
            wspace_resolved = col_gutter_in * n_cols / sum(col_widths_inches)
        else:
            wspace_resolved = 0.0
        if hspace is not None:
            hspace_resolved = hspace
        elif n_rows > 1:
            hspace_resolved = row_gutter_in * n_rows / sum(row_heights_inches)
        else:
            hspace_resolved = 0.0
        grid_w_inches = (
            sum(col_widths_inches) + (n_cols - 1) * col_gutter_in
            if wspace is None
            else unit_inches * (n_cols + (n_cols - 1) * wspace_resolved)
        )
        grid_h_inches = (
            sum(row_heights_inches) + (n_rows - 1) * row_gutter_in
            if hspace is None
            else unit_height_inches_resolved * (
                sum(row_heights) + (n_rows - 1) * hspace_resolved
            )
        )
        if top_reserve_inches is not None:
            top_reserve = top_reserve_inches
        elif _suptitle_band_inches is not None:
            # Row-0 SuptitlePanel was lifted into top_reserve. Two stacked
            # reservations from figure-top downward:
            #   1) `_suptitle_band_inches` — the band itself, anchored to the
            #      figure top edge with NO leading perimeter margin. Suptitle
            #      text V-centers inside the band.
            #   2) `row_gutter_in / 2` — chrome zone for the first body row's
            #      panel titles (which render half a row-gutter above the
            #      spine via Panel._render_chrome_titles). Skipping it makes
            #      the suptitle text and row-1 panel titles land at the same
            #      y and visually collide.
            top_reserve = _suptitle_band_inches + row_gutter_in / 2.0
        elif suptitle is not None:
            # Default: 1 margin suptitle band (text V-centered) + 1 margin gap
            # below the band before the first panel border.
            top_reserve = 2.0 * margin
        else:
            top_reserve = margin
        # Default bottom reserve = 1.5" when figure_number/caption are set
        # (room for 2 stacked text lines); when a SuptitlePanel was lifted as
        # the footer, the band height owns bottom_reserve PLUS a half-gutter
        # chrome zone ABOVE it (mirrors the top_reserve formula). The chrome
        # zone hosts the last body row's x-axis labels / per-panel subtitle
        # text (rendered below the spine via Panel._render_chrome_*) so they
        # don't pack against the footer text. The cell border bottom touches
        # the footer band top — same as the top side, where the suptitle band
        # bottom touches the row-1 cell border. No visible whitespace between
        # cell border and band. Caller's explicit bottom_reserve_inches still
        # wins.
        # Subtitle-band geometry (inches, measured from the figure bottom up).
        # When a multi-column subtitle row was lifted, the subtitle band sits
        # DIRECTLY ON the footer band (they tile, sharing a cell-border edge —
        # the same way the body row tiles against the suptitle band at the top)
        # so the per-panel subtitles and the figure number read as one tight
        # caption block. Above the band is a half-gutter chrome zone before the
        # body row, mirroring the half-gutter the suptitle band leaves below it.
        # A gap between the subtitle band and the footer would read as a void
        # (both bands centre their text, so the gap compounds with their empty
        # halves) — hence band-on-band with no inter-band gap.
        _subtitle_band_center_in: Optional[float] = None
        _subtitle_band_top_in: Optional[float] = None
        _subtitle_band_bottom_in: Optional[float] = None
        if bottom_reserve_inches is not None:
            bottom_reserve = bottom_reserve_inches
        elif _subtitle_band_inches is not None:
            gap = row_gutter_in / 2.0
            foot = _footer_band_inches if _footer_band_inches is not None else 0.0
            # With a footer band below, the subtitle band tiles on top of it.
            # With nothing below, the subtitle band IS the bottom of the figure,
            # so it sits flush on the perimeter (0.0) instead of floating a
            # margin above it — that float renders as dead space under the
            # footer text.
            _subtitle_band_bottom_in = foot if foot > 0.0 else 0.0
            _subtitle_band_top_in = _subtitle_band_bottom_in + _subtitle_band_inches
            _subtitle_band_center_in = (
                _subtitle_band_bottom_in + _subtitle_band_top_in
            ) / 2.0
            bottom_reserve = _subtitle_band_top_in + gap
        elif _footer_band_inches is not None:
            bottom_reserve = _footer_band_inches + row_gutter_in / 2.0
        elif figure_number is not None or figure_caption is not None:
            bottom_reserve = 1.5
        else:
            bottom_reserve = margin
        # Common-canvas normalization (height): back-solve unit_height so the
        # finished figure is exactly `total_height_inches` tall — the vertical
        # mirror of total_width_inches. figsize[1] = grid_h + bottom_reserve +
        # top_reserve, and the reserves + row gutters are FIXED physical inches
        # (independent of unit_height), so the only height-scaled term is
        # grid_h's row contribution. Solve that for unit_height, then recompute
        # the height-dependent quantities (row heights, hspace gutter fraction,
        # grid_h) before figsize is sized. Lets two structurally-different figures
        # (different row counts) land on ONE shared canvas; trades the square
        # shared-tile invariant for an exact W×H match (opt-in only).
        if total_height_inches is not None:
            avail_h = total_height_inches - top_reserve - bottom_reserve
            if hspace is None:
                avail_h -= (n_rows - 1) * row_gutter_in
                unit_height_inches_resolved = avail_h / sum(row_heights)
                row_heights_inches = [
                    r * unit_height_inches_resolved for r in row_heights
                ]
                if n_rows > 1:
                    hspace_resolved = (
                        row_gutter_in * n_rows / sum(row_heights_inches)
                    )
                grid_h_inches = (
                    sum(row_heights_inches) + (n_rows - 1) * row_gutter_in
                )
            else:
                denom_h = sum(row_heights) + (n_rows - 1) * hspace_resolved
                unit_height_inches_resolved = avail_h / denom_h
                row_heights_inches = [
                    r * unit_height_inches_resolved for r in row_heights
                ]
                grid_h_inches = unit_height_inches_resolved * denom_h
        figsize = (
            grid_w_inches + 2 * margin,
            grid_h_inches + bottom_reserve + top_reserve,
        )

        # Translate the inch-domain top_reserve to the subplots_adjust top
        # fraction so Figure.render uses the same reserve compose just baked
        # into figsize (otherwise render falls back to 2*margin and undoes
        # the override).
        top_pad_resolved = 1.0 - top_reserve / figsize[1]
        # Match: bottom reserve gets baked into figsize above; mirror it on
        # the subplots_adjust bottom so the gridspec doesn't reclaim the
        # space we reserved for the figure_number / figure_caption band.
        bottom_pad_resolved = bottom_reserve / figsize[1]
        # Center the suptitle text vertically in the suptitle BAND when row-0
        # was a SuptitlePanel (band = `_suptitle_band_inches`, chrome zone
        # below it for row-1 panel titles is excluded), otherwise in the full
        # top-reserve. Without this, the suptitle text V-centers in a region
        # that includes the row-1 chrome zone and visually droops onto the
        # body panels.
        if _suptitle_band_inches is not None:
            # Band is anchored to the figure top edge; text V-centers in it.
            suptitle_y_resolved = 1.0 - (
                _suptitle_band_inches / 2.0
            ) / figsize[1]
        else:
            suptitle_y_resolved = 1.0 - (top_reserve / 2.0) / figsize[1]
        # Plot-area center x for figure_number alignment — chrome rows (e.g.
        # TextPanel label columns) are excluded so the figure_number sits
        # under the plot's x-axis label rather than the full figure midline.
        # Falls back to figure center when no TextPanels are present.
        plot_cols = []
        for p_idx in range(len(rows[0])):
            panel = rows[0][p_idx]
            if getattr(panel, "is_text_only", False):
                continue
            col_start = sum(rows[0][i].units[0] for i in range(p_idx))
            col_end = col_start + rows[0][p_idx].units[0]
            plot_cols.append((col_start, col_end))
        gutter_inches = wspace_resolved * unit_inches
        if plot_cols:
            c_lo = min(c for c, _ in plot_cols)
            c_hi = max(c for _, c in plot_cols)
            plot_x_start = margin + c_lo * (unit_inches + gutter_inches)
            plot_x_end = (
                margin + (c_hi - 1) * (unit_inches + gutter_inches) + unit_inches
            )
            figure_number_x_resolved = (
                (plot_x_start + plot_x_end) / 2.0 / figsize[0]
            )
        else:
            figure_number_x_resolved = 0.5

        fig = cls(
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            suptitle=suptitle,
            suptitle_fontsize=suptitle_fontsize,
            suptitle_y=suptitle_y_resolved,
            suptitle_band_inches=_suptitle_band_inches,
            dpi=dpi,
            hspace=hspace_resolved,
            wspace=wspace_resolved,
            height_ratios=row_heights,
            debug_guides=debug_guides,
            show_cell_borders=show_cell_borders,
            top_pad=top_pad_resolved,
            bottom_pad=bottom_pad_resolved,
            figure_number=figure_number,
            figure_caption=figure_caption,
            figure_number_x=figure_number_x_resolved,
            footer_text=_footer_text,
            footer_fontsize=_footer_fontsize,
            footer_band_inches=_footer_band_inches,
            bottom_reserve_is_explicit=bottom_reserve_inches is not None,
            subtitle_band_inches=_subtitle_band_inches,
            subtitle_specs=_subtitle_specs,
            subtitle_band_center_inches=_subtitle_band_center_in,
            subtitle_band_top_inches=_subtitle_band_top_in,
            subtitle_band_bottom_inches=_subtitle_band_bottom_in,
            hold_ticks=hold_ticks,
        )
        fig._frame_inset = frame_inset

        for r in range(n_rows):
            for p in range(len(rows[r])):
                panel = rows[r][p]
                col_start = sum(rows[r][i].units[0] for i in range(p))
                colspan = rows[r][p].units[0]
                projection = (
                    "3d" if isinstance(panel, (StaticPanel3D, DynamicPanel3D))
                    else None
                )
                fig.add_panel(
                    panel,
                    row=r,
                    col=col_start,
                    colspan=colspan,
                    projection=projection,
                )
        return fig

    def add_panel(
        self,
        panel: Panel,
        *,
        row: int = 0,
        col: int = 0,
        rowspan: int = 1,
        colspan: int = 1,
        projection: Optional[str] = None,
    ) -> Panel:
        self.panels.append((panel, row, col, rowspan, colspan, projection))
        return panel

    def render(self) -> None:
        max_bottom_pad = max(
            (panel.requires_bottom_pad for panel, *_ in self.panels),
            default=0.0,
        )
        # Outer perimeter: panel-border-to-figure-edge = `margin` on all four
        # sides. Labels render INSIDE that margin (eating into it). Suptitle
        # gets its own `margin` padding above it plus another `margin` below
        # before the first row's panel border.
        fig_w, fig_h = self._mpl_fig.get_size_inches()
        margin = style.DEFAULT_MARGIN_INCHES
        margin_h_frac = margin / fig_w
        margin_v_frac = margin / fig_h
        if self._has_suptitle:
            top_reserve = 2.0 * margin
        else:
            top_reserve = margin
        # Bottom: use the compose-supplied bottom_pad when present (it already
        # encodes the figure_number/caption reservation). Fall back to the
        # panel-driven max_bottom_pad if compose didn't set anything, then to
        # the standard margin.
        if self._bottom_pad is not None:
            bottom_resolved = self._bottom_pad
        elif max_bottom_pad > 0.0:
            bottom_resolved = max_bottom_pad + style.DEFAULT_BOTTOM_PAD_BUFFER_INCHES
        else:
            bottom_resolved = margin_v_frac

        adjust_kwargs: dict[str, float] = {
            "left":   margin_h_frac,
            "right":  1.0 - margin_h_frac,
            "bottom": bottom_resolved,
            "top": (
                self._top_pad if self._top_pad is not None
                else 1.0 - top_reserve / fig_h
            ),
        }
        if adjust_kwargs:
            self._mpl_fig.subplots_adjust(**adjust_kwargs)

        for panel, row, col, rowspan, colspan, projection in self.panels:
            cell = self._gs[row:row + rowspan, col:col + colspan]
            if projection is None:
                ax = self._mpl_fig.add_subplot(cell)
                ax.set_facecolor(style.BG_COLOR)
                for spine in ax.spines.values():
                    spine.set_edgecolor(style.SPINE_COLOR)
                    spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
            else:
                ax = self._mpl_fig.add_subplot(cell, projection=projection)
                ax.set_facecolor(style.BG_COLOR)
            # Reserve a label strip INSIDE the panel's cell: inset the data
            # axes rightward by `content_left_pad_inches` so the y-axis label +
            # tick numbers render in the reserved strip — inside the panel's
            # cell border rather than in a separate column. Applied BEFORE
            # render() so the axis-label inset math sees the inset spine (its
            # figure-edge clamp keys off the axes' left edge). Opt-in via a
            # panel attribute (default 0.0) so existing figures are untouched.
            # Safe for a col-0 panel: its cell border's left side is hardwired
            # to the figure edge, so insetting the axes doesn't shrink the box.
            left_pad_in = getattr(panel, "content_left_pad_inches", 0.0)
            if left_pad_in:
                pos = ax.get_position()
                dx = left_pad_in / fig_w
                ax.set_position(
                    [pos.x0 + dx, pos.y0, max(pos.width - dx, 0.01), pos.height]
                )
            # Hand control of the animation clock to the figure for any
            # DynamicPanel — the panel itself skips FuncAnimation construction
            # in its render() when this flag is set. CompositePanels are not
            # DynamicPanels themselves, but may hold DynamicPanel children; flag
            # those before composite.render() runs (it calls child.render()), so
            # the master clock — not per-child timers — drives them.
            if isinstance(panel, DynamicPanel):
                panel._managed_externally = True
            else:
                for child in _composite_dynamic_children(panel):
                    child._managed_externally = True
            panel.attach(ax)
            panel.render()
            # A twin axis (created during render via ax.twinx()) does not always
            # inherit the content_left_pad inset, leaving an overlay (e.g. the
            # inst-freq curve) shifted into the label strip. Pin the twin to the
            # main axis' (inset) rect so the overlay aligns with the host data.
            if left_pad_in:
                twin = getattr(panel, "ax_twin", None)
                if twin is not None:
                    twin.set_position(ax.get_position())

        self._install_master_clock()
        self._render_subtitle_band()
        if self._show_cell_borders:
            self._draw_cell_borders()
        if self._debug_guides:
            self._draw_debug_guides()
            self._force_all_tick_labels()
            self._draw_text_bbox_guides()
        # Expand fill_cell panels last — after borders/guides read the original
        # gridspec positions — so the expansion doesn't perturb those reads.
        self._apply_fill_cell()
        # Range-bar y-axes read the FINAL axes positions, so they run after
        # the fill_cell expansion.
        self._apply_range_bar_yaxes()
        self._apply_jupyter_display_styling()

    def _force_all_tick_labels(self) -> None:
        """DEBUG: force every axes' tick labels visible on the side that
        actually hosts them. Used by ``debug_guides=True`` so hidden labels
        surface and the user can spot overlaps with neighbors.
        """
        for ax in self._mpl_fig.axes:
            if ax.xaxis.get_label_position() == "top":
                ax.tick_params(axis="x", labeltop=True, labelbottom=False)
            else:
                ax.tick_params(axis="x", labelbottom=True, labeltop=False)
            if ax.yaxis.get_label_position() == "right":
                ax.tick_params(axis="y", labelright=True, labelleft=False)
            else:
                ax.tick_params(axis="y", labelleft=True, labelright=False)

    def _column_x_extent(
        self, col_start: int, colspan: int
    ) -> tuple[Optional[float], Optional[float]]:
        """Figure-fraction (x0, x1) spanned by the body panels in a column range.

        Read from the RENDERED panel axes so the subtitle band lands on the
        same column geometry as the body grid (survives wspace overrides).
        Returns (None, None) when no body panel occupies the range.
        """
        x0: Optional[float] = None
        x1: Optional[float] = None
        for panel, _row, col, _rspan, cspan, _ in self.panels:
            if panel.ax is None:
                continue
            if col >= col_start and col + cspan <= col_start + colspan:
                bbox = panel.ax.get_position()
                x0 = bbox.x0 if x0 is None else min(x0, bbox.x0)
                x1 = bbox.x1 if x1 is None else max(x1, bbox.x1)
        return x0, x1

    def _render_subtitle_band(self) -> None:
        """Render lifted per-column subtitles as a thin band beneath the body.

        Mirrors the suptitle/footer bands: each label is figure-level chrome
        V-centered in `_subtitle_band_center_inches` and H-centered under its
        column (read from the rendered body panels). No gridspec cell, so no
        ballooning cell border and no full-row gutter above the labels.
        """
        if not self._subtitle_specs or self._subtitle_band_center_inches is None:
            return
        fig = self._mpl_fig
        _, fig_h = fig.get_size_inches()
        y = self._subtitle_band_center_inches / fig_h
        for spec in self._subtitle_specs:
            x0, x1 = self._column_x_extent(spec["col_start"], spec["colspan"])
            if x0 is None:
                continue
            fig.text(
                (x0 + x1) / 2.0, y, spec["text"],
                color=style.SUPTITLE_COLOR,
                fontsize=spec["font_size"],
                fontweight=style.SUPTITLE_WEIGHT,
                ha="center", va="center",
                transform=fig.transFigure,
            )

    def _measure_half_gutter_fracs(self) -> tuple[Optional[float], Optional[float]]:
        """Measure actual half-gutter (inter-cell gap / 2) from adjacent panels.

        Returns (half_row, half_col) in figure-coordinate fractions. Reads
        the ACTUAL spine-to-spine gap between rendered panels instead of
        the style constant — survives `Figure.compose(hspace=...)` overrides
        so cell borders land at the midpoint of the *actual* gutter, not
        the gutter that style.py declares.

        Returns None for a direction if no adjacent panels exist there.
        """
        half_row: Optional[float] = None
        half_col: Optional[float] = None
        for p1, r1, c1, rspan1, cspan1, _ in self.panels:
            if p1.ax is None:
                continue
            for p2, r2, c2, rspan2, cspan2, _ in self.panels:
                if p2.ax is None or p1 is p2:
                    continue
                b1 = p1.ax.get_position()
                b2 = p2.ax.get_position()
                # Vertical neighbor: p2's first row == p1's last row + 1, same col.
                if half_row is None and r2 == r1 + rspan1 and c1 == c2:
                    gap = b1.y0 - b2.y1
                    if gap > 0:
                        half_row = gap / 2.0
                # Horizontal neighbor: p2's first col == p1's last col + 1, same row.
                if half_col is None and c2 == c1 + cspan1 and r1 == r2:
                    gap = b2.x0 - b1.x1
                    if gap > 0:
                        half_col = gap / 2.0
                if half_row is not None and half_col is not None:
                    return half_row, half_col
        return half_row, half_col

    def _cell_rect_fracs(
        self, panel, row, col, rowspan, colspan,
        half_row_g_frac, half_col_g_frac, sup_band_bot_frac, fig_h,
        perim_x_frac=0.0, perim_y_frac=0.0,
    ) -> tuple[float, float, float, float]:
        """Figure-fraction (left, bottom, right, top) of the gridspec CELL that
        encloses one panel — the cell border bounds. Perimeter sides snap to
        the figure edge (or suptitle / footer / subtitle band edge); interior
        sides sit at the half-gutter midline. Shared by the cell-border tiler
        and the fill_cell axes expansion so both agree on cell bounds.

        ``perim_x_frac`` / ``perim_y_frac`` inset the OUTER perimeter sides off
        the figure edge by that fraction (the opt-in ``frame_inset`` path). At
        the default 0.0 perimeter sides snap to the canvas edge as before —
        which renders the line right at the edge where it is effectively clipped
        to invisibility. A non-zero inset (the figure margin) pulls those lines
        inboard so the outer frame and header/footer bands read as full boxes.
        """
        bbox = panel.ax.get_position()
        cell_left = perim_x_frac if col == 0 else bbox.x0 - half_col_g_frac
        cell_right = (
            1.0 - perim_x_frac if col + colspan == self.n_cols
            else bbox.x1 + half_col_g_frac
        )
        if row == 0:
            # With a lifted suptitle band the body's top row touches the band
            # bottom (an interior, already-visible line); otherwise the top row
            # is itself the perimeter and insets by perim_y_frac.
            cell_top = sup_band_bot_frac if self._has_suptitle else 1.0 - perim_y_frac
        else:
            cell_top = bbox.y1 + half_row_g_frac
        # Bottom-most row stops at the footer/subtitle band top when one was
        # lifted into a bottom_reserve band (cell border touches the band edge
        # with no gap); otherwise it stops at the half-gutter midline.
        if row + rowspan == self.n_rows:
            if (
                self._subtitle_band_inches is not None
                and self._subtitle_band_top_inches is not None
            ):
                cell_bot = self._subtitle_band_top_inches / fig_h
            elif self._footer_band_inches is not None:
                cell_bot = self._footer_band_inches / fig_h
            elif self._bottom_reserve_is_explicit and self._bottom_pad is not None:
                # Caller reserved unbordered bottom space via
                # bottom_reserve_inches (already baked into _bottom_pad as a
                # figure fraction) — the bottom row's cell stops at the reserve
                # top so borders and fill_cell don't spill into it.
                cell_bot = self._bottom_pad
            else:
                cell_bot = perim_y_frac
        else:
            cell_bot = bbox.y0 - half_row_g_frac
        return cell_left, cell_bot, cell_right, cell_top

    def _apply_fill_cell(self) -> None:
        """Expand fill_cell panels' axes to fill their gridspec cell.

        Runs AFTER cell borders are drawn (those read the original gridspec
        positions). Cell rects are computed from the un-expanded positions
        first, then applied — so no panel's expansion perturbs another's
        gutter measurement.

        Two opt-in modes:
          - ``fill_cell``: expand to the full cell rect (all four sides).
          - ``fill_cell_vertical``: expand only the vertical extent to the cell
            rect; the horizontal position is preserved (so a panel that already
            inset its left edge for an in-cell label strip via
            ``content_left_pad_inches`` keeps that strip). The twin axis, if
            any, is re-pinned to the host so overlays stay aligned.
        """
        targets = [
            (p, r, c, rs, cs) for p, r, c, rs, cs, _ in self.panels
            if (getattr(p, "fill_cell", False)
                or getattr(p, "fill_cell_vertical", False))
            and p.ax is not None
        ]
        if not targets:
            return

        fig = self._mpl_fig
        fig_w, fig_h = fig.get_size_inches()
        margin = style.DEFAULT_MARGIN_INCHES
        half_row_g_frac, half_col_g_frac = self._measure_half_gutter_fracs()
        if half_row_g_frac is None:
            half_row_g_frac = (style.DEFAULT_GUTTER_INCHES / 2.0) / fig_h
        if half_col_g_frac is None:
            half_col_g_frac = (style.DEFAULT_COLUMN_GUTTER_INCHES / 2.0) / fig_w
        if self._has_suptitle:
            if self._suptitle_band_inches is not None:
                sup_band_bot_frac = 1.0 - self._suptitle_band_inches / fig_h
            else:
                sup_band_bot_frac = 1.0 - margin / fig_h
        else:
            sup_band_bot_frac = 1.0

        rects = [
            (p, self._cell_rect_fracs(
                p, r, c, rs, cs,
                half_row_g_frac, half_col_g_frac, sup_band_bot_frac, fig_h))
            for p, r, c, rs, cs in targets
        ]
        for panel, (left, bot, right, top) in rects:
            cur = panel.ax.get_position()
            # A titled panel keeps its natural chrome zone at the top (the gap
            # between the gridspec axes top and the cell top) so the inline title
            # still has a home above the plot — otherwise fill_cell pushes the
            # axes to the very top of the cell and the title lands in the row
            # above. Bottom + sides still expand to fill.
            if getattr(panel, "title", None):
                top = min(top, cur.y1)
            if (getattr(panel, "fill_cell_vertical", False)
                    and not getattr(panel, "fill_cell", False)):
                # Vertical-only: keep the current horizontal rect (preserves the
                # content_left_pad label strip), grow the y extent to the cell.
                # An optional `fill_cell_pad_inches` insets the filled axes from
                # the cell's top and bottom so the extreme y-tick labels (e.g.
                # 1k at top, 10 at bottom) lift off the cell-border line instead
                # of sitting flush on it — breathing room inside the cell.
                vpad_in = float(getattr(panel, "fill_cell_pad_inches", 0.0) or 0.0)
                vpad_frac = vpad_in / fig_h
                bot_p = bot + vpad_frac
                top_p = top - vpad_frac
                panel.ax.set_position([cur.x0, bot_p, cur.width, top_p - bot_p])
            else:
                panel.ax.set_position([left, bot, right - left, top - bot])
            # Re-pin the twin (e.g. row-1 inst-freq overlay) to the host's new
            # rect so the overlay tracks the expanded plot.
            twin = getattr(panel, "ax_twin", None)
            if twin is not None:
                twin.set_position(panel.ax.get_position())

    def _apply_range_bar_yaxes(self) -> None:
        """Swap opted-in panels' stock y-axes for the range-bar treatment.

        Opt-in via ``panel.range_bar_yaxis = True`` on a panel that reserves an
        in-cell label strip (``content_left_pad_inches``). Post-layout pass —
        runs after ``_apply_fill_cell`` so it reads FINAL axes positions:

          - snaps the y-view to the outermost ticks so the range ENDS at the
            end labels (a heatmap's bin-space view may otherwise run slightly
            past its extreme tick values),
          - hides the stock y-axis (ticks, labels, ylabel),
          - draws a thin vertical bar (``style.RANGE_BAR_LINEWIDTH``) flush
            with the plot's left edge, spanning exactly the plot height,
          - re-draws the tick labels as figure-level text, each label's text
            midpoint horizontally centered in the visible strip between the
            cell border and the bar,
          - tucks the end labels INSIDE the range (top label top-aligned,
            bottom label bottom-aligned, inset by
            ``style.RANGE_BAR_LABEL_END_PAD_INCHES``) so nothing straddles a
            shared cell border or collides with a neighboring row's labels,
          - renders the y unit (the panel's ylabel, e.g. "Hz") horizontal at
            mid-height in the same face as the tick numbers.
        """
        targets = [
            (p, c) for p, _r, c, _rs, _cs, _ in self.panels
            if getattr(p, "range_bar_yaxis", False) and p.ax is not None
        ]
        if not targets:
            return
        import matplotlib.lines as mlines
        fig = self._mpl_fig
        fig.canvas.draw()  # populate tick label text
        fig_w, fig_h = fig.get_size_inches()
        end_pad_v = style.RANGE_BAR_LABEL_END_PAD_INCHES / fig_h
        for panel, col in targets:
            ax = panel.ax
            ticks_and_labels = [
                (t, lab.get_text())
                for t, lab in zip(ax.get_yticks(), ax.get_yticklabels())
            ]
            tick_vals = [t for t, _lab in ticks_and_labels]
            if tick_vals:
                ax.set_ylim(min(tick_vals), max(tick_vals))
            ylim = ax.get_ylim()
            span = ylim[1] - ylim[0]
            if not span:
                continue
            unit = ax.get_ylabel()
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_ylabel("")
            pos = ax.get_position()
            bar_x = pos.x0
            fig.add_artist(mlines.Line2D(
                [bar_x, bar_x], [pos.y0, pos.y1],
                color=style.TICK_LABEL_COLOR,
                linewidth=style.RANGE_BAR_LINEWIDTH,
            ))
            # Labels center between the VISIBLE cell border and the bar. A
            # col-0 panel's border left side is hardwired to the figure edge
            # (see _draw_cell_borders), so its strip starts at the frame edge
            # gap; otherwise the strip starts where the label-strip inset
            # pushed the axes from.
            if col == 0:
                strip_left = style.DEFAULT_FRAME_EDGE_GAP_INCHES / fig_w
            else:
                strip_left = pos.x0 - (
                    getattr(panel, "content_left_pad_inches", 0.0) / fig_w
                )
            label_x = (strip_left + bar_x) / 2.0
            for tick, lab in ticks_and_labels:
                frac = (tick - ylim[0]) / span
                if frac < -0.001 or frac > 1.001:
                    continue
                y = pos.y0 + frac * pos.height
                # End labels tuck INSIDE the range so they don't straddle the
                # cell border or collide with the neighboring row's labels.
                if frac > 0.93:
                    va, y = "top", y - end_pad_v
                elif frac < 0.07:
                    va, y = "bottom", y + end_pad_v
                else:
                    va = "center"
                fig.text(
                    label_x, y, lab, color=style.TICK_LABEL_COLOR,
                    fontsize=style.DEFAULT_TICK_LABEL_SIZE,
                    fontweight=style.DEFAULT_TICK_LABEL_FONT_WEIGHT,
                    ha="center", va=va,
                )
            if unit:
                fig.text(
                    label_x, (pos.y0 + pos.y1) / 2.0, unit,
                    color=style.TICK_LABEL_COLOR,
                    fontsize=style.DEFAULT_TICK_LABEL_SIZE,
                    fontweight=style.DEFAULT_TICK_LABEL_FONT_WEIGHT,
                    ha="center", va="center",
                )

    def _draw_cell_borders(self) -> None:
        """Tile the figure with one bordered Rectangle per gridspec cell.

        Each cell encloses one panel's chrome (title above, x-label below,
        y-labels on left/right) AND the panel's inner spine border — matching
        the "ideal template" reference. Cells tile densely: shared edges
        between adjacent cells appear as a single line.

        Cell boundaries:
          - Non-perimeter sides: midpoint of the inter-cell gutter
          - Perimeter sides: figure edge (or suptitle band bottom for top row)
        """
        from matplotlib.patches import Rectangle

        fig = self._mpl_fig
        fig_w, fig_h = fig.get_size_inches()
        margin = style.DEFAULT_MARGIN_INCHES

        # Outer-perimeter edge gap: every figure's outer frame is inset from the
        # canvas edge by style.DEFAULT_FRAME_EDGE_GAP_INCHES so it reads as a
        # framed figure on a page (and the full stroke renders instead of being
        # half-clipped at fraction 0.0/1.0). A fixed INCH value gives every
        # figure the same visible gap regardless of dpi. One shared default —
        # change the style knob, every figure's edge gap moves together.
        inset_in = style.DEFAULT_FRAME_EDGE_GAP_INCHES
        perim_x_frac = inset_in / fig_w
        perim_y_frac = inset_in / fig_h

        # Use ACTUAL gutter measured from adjacent-panel spine positions so
        # cell borders honor compose(hspace=...) / compose(wspace=...)
        # overrides. Fall back to style constants if no adjacent panel exists
        # in that direction (single-row or single-column figure).
        half_row_g_frac, half_col_g_frac = self._measure_half_gutter_fracs()
        if half_row_g_frac is None:
            half_row_g_frac = (style.DEFAULT_GUTTER_INCHES / 2.0) / fig_h
        if half_col_g_frac is None:
            half_col_g_frac = (style.DEFAULT_COLUMN_GUTTER_INCHES / 2.0) / fig_w

        if self._has_suptitle:
            # The suptitle BAND height differs from `top_reserve`: top_reserve
            # also includes the row-1 chrome zone (= half row gutter) so the
            # first body row's panel titles render between the band bottom and
            # the row-1 spine. The cell border must anchor to the BAND
            # bottom (above the chrome zone & above the panel titles) — not
            # to the gridspec top (= spine top, which puts titles OUTSIDE the
            # border). Preference order:
            #   1. explicit `_suptitle_band_inches` from compose (lifted band)
            #   2. legacy single-margin band when only the constructor path ran
            if self._suptitle_band_inches is not None:
                sup_band_bot_frac = 1.0 - self._suptitle_band_inches / fig_h
            else:
                sup_band_bot_frac = 1.0 - margin / fig_h
        else:
            sup_band_bot_frac = 1.0

        # The cell border IS each panel's single frame (the figure-1 / 2.5
        # border model — spines off, this box is the one visible frame). Styled
        # from the shared DEFAULT_FRAME_* knobs so every figure's frame is one
        # clean visible thin line, tuned in one place (style.py).
        border_kwargs = dict(
            fill=False,
            edgecolor=style.DEFAULT_FRAME_COLOR,
            linewidth=style.DEFAULT_FRAME_LINEWIDTH,
            alpha=style.DEFAULT_FRAME_ALPHA,
            transform=fig.transFigure,
            zorder=50,
            clip_on=False,
        )

        for panel, row, col, rowspan, colspan, _ in self.panels:
            if panel.ax is None:
                continue
            cell_left, cell_bot, cell_right, cell_top = self._cell_rect_fracs(
                panel, row, col, rowspan, colspan,
                half_row_g_frac, half_col_g_frac, sup_band_bot_frac, fig_h,
                perim_x_frac, perim_y_frac,
            )

            rect = Rectangle(
                (cell_left, cell_bot),
                cell_right - cell_left,
                cell_top - cell_bot,
                **border_kwargs,
            )
            fig.add_artist(rect)

        if self._has_suptitle:
            rect = Rectangle(
                (perim_x_frac, sup_band_bot_frac),
                1.0 - 2.0 * perim_x_frac,
                (1.0 - perim_y_frac) - sup_band_bot_frac,
                **border_kwargs,
            )
            fig.add_artist(rect)

        # Footer band cell border — mirror of the suptitle band. Anchored to
        # the figure BOTTOM at y=0, extending up to the band height.
        if self._footer_band_inches is not None:
            footer_band_top_frac = self._footer_band_inches / fig_h
            rect = Rectangle(
                (perim_x_frac, perim_y_frac),
                1.0 - 2.0 * perim_x_frac,
                footer_band_top_frac - perim_y_frac,
                **border_kwargs,
            )
            fig.add_artist(rect)

        # Subtitle band cell borders — one per column, aligned to the body
        # grid columns so each label sits in a thin cell beneath its panel.
        if self._subtitle_band_inches is not None and self._subtitle_specs:
            sub_top = self._subtitle_band_top_inches / fig_h
            sub_bot = self._subtitle_band_bottom_inches / fig_h
            # When the subtitle band is the bottom-most element (no footer band
            # beneath it) its bottom edge IS the figure perimeter, so it honors
            # perim_y_frac exactly like the body cells / header band — otherwise
            # the band floats a margin above the canvas and reads as dead space
            # under the footer text.
            sub_bot_draw = max(sub_bot, perim_y_frac)
            for spec in self._subtitle_specs:
                x0, x1 = self._column_x_extent(spec["col_start"], spec["colspan"])
                if x0 is None:
                    continue
                # Outer columns snap to perim_x_frac (NOT the raw canvas edge) so
                # the band's left/right frame lines up vertically with the header
                # band and the body cells above it.
                cell_left = (
                    perim_x_frac if spec["col_start"] == 0
                    else x0 - half_col_g_frac
                )
                cell_right = (
                    1.0 - perim_x_frac
                    if spec["col_start"] + spec["colspan"] == self.n_cols
                    else x1 + half_col_g_frac
                )
                rect = Rectangle(
                    (cell_left, sub_bot_draw),
                    cell_right - cell_left,
                    sub_top - sub_bot_draw,
                    **border_kwargs,
                )
                fig.add_artist(rect)

    def _draw_text_bbox_guides(self) -> None:
        """Draw horizontal lines above and below every visible Text artist.

        Surfaces where titles, captions, suptitle, footer, and axis labels
        ACTUALLY render — so overflows (text crossing a cell border) and
        misalignments (text not centered in its chrome band) are visible
        at a glance. Each text gets a red bracket: bottom line at the
        glyph baseline area's bottom, top line at the glyph cap height.

        Enabled by ``debug_guides=True``. Runs after _draw_debug_guides so
        the bbox brackets render on top of the layout guides.
        """
        from matplotlib.lines import Line2D

        fig = self._mpl_fig
        # Force a draw so get_window_extent returns final, post-layout
        # bboxes. Justified TextPanel content materializes lazily on first
        # draw, so this must run after at least one canvas pass.
        fig.canvas.draw()
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            return

        # Collect every Text artist on the figure: figure-level (suptitle,
        # figure_number) + per-axes (chrome titles via ax.text, axis labels,
        # tick labels, TextPanel bodies via ax.text).
        texts = list(fig.texts)
        for ax in fig.axes:
            texts.extend(ax.texts)
            for lbl in (ax.xaxis.label, ax.yaxis.label):
                if lbl.get_text():
                    texts.append(lbl)
            texts.extend(ax.get_xticklabels())
            texts.extend(ax.get_yticklabels())

        inv = fig.transFigure.inverted()
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            try:
                bbox = text.get_window_extent(renderer)
            except Exception:
                continue
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            x0, y0 = inv.transform((bbox.x0, bbox.y0))
            x1, y1 = inv.transform((bbox.x1, bbox.y1))
            # Bottom line.
            fig.add_artist(Line2D(
                [x0, x1], [y0, y0],
                color="#ff3030", linewidth=0.5, alpha=0.85,
                transform=fig.transFigure, zorder=210, clip_on=False,
            ))
            # Top line.
            fig.add_artist(Line2D(
                [x0, x1], [y1, y1],
                color="#ff3030", linewidth=0.5, alpha=0.85,
                transform=fig.transFigure, zorder=210, clip_on=False,
            ))

    def _draw_debug_guides(self) -> None:
        """Overlay colored guide lines at every template boundary so layout
        math is visible at a glance.

        Lines drawn (in figure-coord [0, 1] space):
          - magenta: outer perimeter margin (margin inset from all 4 edges)
          - cyan:    suptitle band (top + bottom of the suptitle text reserve)
          - yellow:  per-panel title band (top of band + spine top)
          - orange:  per-panel axes bbox left/right edges (= y-label sit zone)
          - lime:    inter-cell gutter midlines (h + v)
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        fig = self._mpl_fig
        fig_w, fig_h = fig.get_size_inches()
        margin = style.DEFAULT_MARGIN_INCHES
        margin_h_frac = margin / fig_w
        margin_v_frac = margin / fig_h
        title_band_v_frac = style.DEFAULT_PANEL_TITLE_RESERVE_INCHES / fig_h

        def hline(y: float, color: str, label: str | None = None) -> None:
            ln = Line2D(
                [0.0, 1.0], [y, y],
                color=color, linewidth=0.6, alpha=0.7,
                linestyle="--", transform=fig.transFigure,
                zorder=200,
            )
            fig.add_artist(ln)
            if label is not None:
                fig.text(
                    0.002, y, label,
                    color=color, fontsize=7, alpha=0.85,
                    ha="left", va="center",
                    transform=fig.transFigure, zorder=201,
                )

        def vline(x: float, color: str, label: str | None = None) -> None:
            ln = Line2D(
                [x, x], [0.0, 1.0],
                color=color, linewidth=0.6, alpha=0.7,
                linestyle="--", transform=fig.transFigure,
                zorder=200,
            )
            fig.add_artist(ln)
            if label is not None:
                fig.text(
                    x, 0.002, label,
                    color=color, fontsize=7, alpha=0.85,
                    ha="left", va="bottom", rotation=90,
                    transform=fig.transFigure, zorder=201,
                )

        # --- Outer perimeter margin (magenta) ---
        vline(margin_h_frac, "#ff00ff", "L margin")
        vline(1.0 - margin_h_frac, "#ff00ff", "R margin")
        hline(margin_v_frac, "#ff00ff", "B margin")

        # --- Suptitle band (cyan): height = top_reserve band, text V-centered.
        # When compose lifted a row-0 SuptitlePanel into top_reserve, that
        # band height is encoded in `self._top_pad` (= panel_top fraction).
        if self._has_suptitle:
            panel_top_y = (
                self._top_pad if self._top_pad is not None
                else 1.0 - (2.0 * margin) / fig_h
            )
            band_height = 1.0 - panel_top_y
            sup_center_y = 1.0 - band_height / 2.0
            hline(panel_top_y, "#00e5ff", "sup band bot")
            hline(sup_center_y, "#00e5ff", "sup center")
        else:
            hline(1.0 - margin_v_frac, "#ff00ff", "T margin")

        # --- Per-panel: title band + axes bbox sides + axis-label insets ---
        # yellow: title band top + spine top.
        # orange: axes bbox left/right edges (= y-label sit zone reference).
        # red:    axis-label inset markers — where the actual axis label TEXT
        #         centers sit, drawn as a short bracket on each side of the
        #         panel. Derived from style.DEFAULT_{X,Y}_AXIS_LABEL_INSET so
        #         it matches what the panels actually render.
        x_inset_in = style.DEFAULT_X_AXIS_LABEL_INSET_INCHES
        y_inset_in = style.DEFAULT_Y_AXIS_LABEL_INSET_INCHES
        axes_bboxes: list = []
        for panel, *_ in self.panels:
            if panel.ax is None:
                continue
            bbox = panel.ax.get_position()
            axes_bboxes.append(bbox)
            title_band_top = bbox.y1 + title_band_v_frac
            hline(title_band_top, "#ffd400", None)
            hline(bbox.y1, "#ffd400", None)
            # Orange verticals at axes left/right — y-labels live to the left
            # of x0, twin y-labels to the right of x1.
            ln_l = Line2D(
                [bbox.x0, bbox.x0], [bbox.y0, bbox.y1],
                color="#ff8800", linewidth=0.6, alpha=0.55,
                linestyle=":", transform=fig.transFigure, zorder=199,
            )
            ln_r = Line2D(
                [bbox.x1, bbox.x1], [bbox.y0, bbox.y1],
                color="#ff8800", linewidth=0.6, alpha=0.55,
                linestyle=":", transform=fig.transFigure, zorder=199,
            )
            fig.add_artist(ln_l)
            fig.add_artist(ln_r)

            # Red brackets at the axis-label inset positions, with a second
            # marker on the CELL-BORDER side so the user can see whether the
            # axis label sits centered between spine and cell border. Skipped
            # entirely for text-only panels — those have no axis labels, so
            # the brackets would mark nothing and confuse the debug overlay.
            if getattr(panel, "is_text_only", False):
                continue
            cy = (bbox.y0 + bbox.y1) / 2.0
            cx = (bbox.x0 + bbox.x1) / 2.0
            # Use ACTUAL half-gutter so the cell-border marker lands where
            # _draw_cell_borders actually drew the border (post-fix).
            half_row_g_frac, half_col_g_frac = self._measure_half_gutter_fracs()
            if half_row_g_frac is None:
                half_row_g_frac = (style.DEFAULT_GUTTER_INCHES / 2.0) / fig_h
            if half_col_g_frac is None:
                half_col_g_frac = (style.DEFAULT_COLUMN_GUTTER_INCHES / 2.0) / fig_w
            x_lbl_y = bbox.y0 - x_inset_in / fig_h
            x_border_y = bbox.y0 - half_row_g_frac
            y_lbl_x_left = bbox.x0 - y_inset_in / fig_w
            y_border_x_left = bbox.x0 - half_col_g_frac
            y_lbl_x_right = bbox.x1 + y_inset_in / fig_w
            y_border_x_right = bbox.x1 + half_col_g_frac
            # First color = label-center marker (close to spine)
            # Second color = cell-border marker (far from spine) — lighter shade
            # so the user can distinguish which line is which.
            marker_pairs = (
                # x-axis: label marker + outer cell-border marker
                ((cx - 0.02, cx + 0.02), (x_lbl_y, x_lbl_y), "#ff3333"),
                ((cx - 0.02, cx + 0.02), (x_border_y, x_border_y), "#ff99aa"),
                # y-axis LEFT: label marker + outer cell-border marker
                ((y_lbl_x_left, y_lbl_x_left), (cy - 0.02, cy + 0.02), "#ff3333"),
                ((y_border_x_left, y_border_x_left), (cy - 0.02, cy + 0.02), "#ff99aa"),
                # y-axis RIGHT: label marker + outer cell-border marker
                ((y_lbl_x_right, y_lbl_x_right), (cy - 0.02, cy + 0.02), "#ff3333"),
                ((y_border_x_right, y_border_x_right), (cy - 0.02, cy + 0.02), "#ff99aa"),
            )
            for x_pair, y_pair, color in marker_pairs:
                fig.add_artist(Line2D(
                    list(x_pair), list(y_pair),
                    color=color, linewidth=1.2, alpha=0.9,
                    transform=fig.transFigure, zorder=202,
                ))

        # --- Inter-cell gutter midlines (lime) ---
        # Group bboxes by row (y-center) and column (x-center) to find gutter
        # midlines between adjacent cells.
        x_centers = sorted({round((b.x0 + b.x1) / 2.0, 4) for b in axes_bboxes})
        y_centers = sorted({round((b.y0 + b.y1) / 2.0, 4) for b in axes_bboxes})
        # Vertical gutter midlines: between adjacent column centers, find the
        # midpoint of (right edge of left cell, left edge of right cell).
        for i in range(len(x_centers) - 1):
            left = max(
                b.x1 for b in axes_bboxes
                if round((b.x0 + b.x1) / 2.0, 4) == x_centers[i]
            )
            right = min(
                b.x0 for b in axes_bboxes
                if round((b.x0 + b.x1) / 2.0, 4) == x_centers[i + 1]
            )
            mid = (left + right) / 2.0
            vline(mid, "#22ff88", None)
        # Horizontal gutter midlines: between adjacent row centers.
        for j in range(len(y_centers) - 1):
            bot = max(
                b.y1 for b in axes_bboxes
                if round((b.y0 + b.y1) / 2.0, 4) == y_centers[j]
            )
            top = min(
                b.y0 for b in axes_bboxes
                if round((b.y0 + b.y1) / 2.0, 4) == y_centers[j + 1]
            )
            mid = (bot + top) / 2.0
            hline(mid, "#22ff88", None)

    def _install_master_clock(self) -> None:
        """Create a single FuncAnimation that ticks every DynamicPanel in this
        figure from one shared clock.

        Cycle length: ``max(panel.total_frames) + _FIGURE_CLOCK_HOLD_TICKS``.
        Shorter panels naturally hold their final frame (DynamicPanel.tick
        clamps), and every panel sits on its final frame for the trailing
        hold ticks before the cycle wraps.
        """
        dynamic_panels: List[DynamicPanel] = []
        for p, *_ in self.panels:
            if isinstance(p, DynamicPanel):
                dynamic_panels.append(p)
            else:
                dynamic_panels.extend(_composite_dynamic_children(p))
        if not dynamic_panels:
            return

        hold_ticks = (
            self._hold_ticks if self._hold_ticks is not None
            else _FIGURE_CLOCK_HOLD_TICKS
        )
        cycle_length = (
            max(p._total_frames() for p in dynamic_panels)
            + hold_ticks
        )
        # All DynamicPanels share the first panel's interval — they're meant
        # to be in lockstep; mismatched intervals would be a user error.
        interval = dynamic_panels[0].interval_ms

        def _tick(global_idx: int) -> None:
            for panel in dynamic_panels:
                panel.tick(global_idx)

        from matplotlib.animation import FuncAnimation
        self._anim = FuncAnimation(
            self._mpl_fig,
            _tick,
            frames=cycle_length,
            interval=interval,
            repeat=True,
            blit=False,
        )

    def _apply_jupyter_display_styling(self) -> None:
        """Style the ipympl canvas widget and cell-output container — only
        meaningful under `%matplotlib widget` / ipympl. No-op otherwise.

        - Hides the ipympl toolbar/header/footer chrome unless show_toolbar
        - Sets the canvas widget width to fill the cell or to display_width
        - Injects a one-time dark-theme CSS rule for the cell-output container
        """
        canvas = getattr(self._mpl_fig, "canvas", None)
        if canvas is None:
            return
        # ipympl canvases expose a `layout` attribute holding a widget Layout
        # with a writable `width` field. Qt/Agg canvases expose `layout` as a
        # bound method (PyQt's QWidget.layout()) — distinguish by checking
        # for the widget-Layout-specific `width` attribute.
        layout = getattr(canvas, "layout", None)
        if layout is None or not hasattr(layout, "width"):
            return  # not an ipympl Canvas widget — non-interactive backend

        # Disable ipympl's resize-observer BEFORE touching width. With
        # resizable=True the observer reserves the figure's *native* pixel
        # height for the widget box and redraws the figure to fill the cell.
        # When we then constrain only the width, the box keeps the full native
        # height while the canvas scales down → a tall white gap below the
        # figure; in the other direction the box ends up shorter than the
        # canvas and clips the bottom band. Pinning resizable=False renders the
        # canvas once at its exact figsize×dpi aspect and lets CSS scale it.
        if hasattr(canvas, "resizable"):
            canvas.resizable = False

        target_width = None
        if self._fill_width:
            target_width = self._display_width or "100%"
        elif self._display_width is not None:
            target_width = self._display_width
        if target_width is not None:
            canvas.layout.width = target_width
            # height="auto" makes the widget box follow the (aspect-locked)
            # scaled canvas instead of the reserved native height — this is
            # what removes the trailing whitespace and the bottom clipping.
            canvas.layout.height = "auto"

        for attr in ("toolbar_visible", "header_visible", "footer_visible"):
            if hasattr(canvas, attr):
                setattr(canvas, attr, self._show_toolbar)

        apply_jupyter_dark()

    def savefig(self, path: str, **kwargs) -> str:
        """Save the figure at exactly ``figsize × dpi`` pixels.

        Defaults are ``bbox_inches=None, pad_inches=0`` deliberately: the
        figure's pixel dimensions are derived solely from ``figsize`` and
        ``dpi``, never from the bounding boxes of drawn artists. This
        guarantees that:

          * Decorators (labels, callouts, annotations) cannot expand the
            saved canvas. Anything anchored outside [0, 1] figure-coord
            space clips at the figure edge — visibly and predictably.
          * All margins and padding are governed solely by the
            ``subplots_adjust(left=, right=, top=, bottom=)`` knobs the
            Figure already sets (the ``top_pad`` reserved for suptitle,
            mpl defaults for left/right/bottom). Margins do NOT shift
            based on what content was drawn.
          * Two renders of the same Figure produce PNGs of identical
            dimensions, regardless of any decorators added between
            ``render()`` and ``savefig()``.

        Callers can opt back into matplotlib's tight-bbox auto-crop by
        passing ``bbox_inches="tight"`` explicitly.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        bbox_inches = kwargs.pop("bbox_inches", None)
        pad_inches = kwargs.pop("pad_inches", 0)
        self._mpl_fig.savefig(
            path,
            facecolor=self._mpl_fig.get_facecolor(),
            dpi=self._mpl_fig.get_dpi(),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **kwargs,
        )
        return os.path.abspath(path)

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close(self._mpl_fig)
