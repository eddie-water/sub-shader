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
        top_pad: Optional[float] = None,
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
        self._top_pad = top_pad
        self._debug_guides = debug_guides
        self._show_cell_borders = show_cell_borders
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
                color=style.TICK_LABEL_COLOR,
                fontsize=sup_fontsize_resolved,
                fontweight="bold",
                y=(
                    suptitle_y if suptitle_y is not None
                    else 1.0 - sup_band_center_in / fig_h
                ),
                va="center",
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
        dpi: Optional[int] = None,
        hspace: Optional[float] = None,
        wspace: Optional[float] = None,
        debug_guides: bool = False,
        show_cell_borders: bool = False,
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

        width_units_per_row = [sum(p.units[0] for p in row) for row in rows]
        if any(w != width_units_per_row[0] for w in width_units_per_row):
            raise ValueError(
                f"Figure.compose row widths mismatch: {width_units_per_row}"
            )

        n_cols = width_units_per_row[0]
        unit_inches = (
            unit_inches if unit_inches is not None
            else style.DEFAULT_PANEL_UNIT_INCHES
        )
        # Outer perimeter: `margin` between every panel border and the figure
        # edge (left, right, bottom). Tick + axis labels render WITHIN that
        # margin (eating into it from the spine outward).  Suptitle gets its
        # OWN `margin` of padding above it, plus another `margin` below it
        # before the panel grid starts.  Inter-cell gutters use a separate
        # `gutter` sized to host one set of axis labels + one panel title
        # between adjacent cells.
        margin = style.DEFAULT_MARGIN_INCHES
        column_gutter_fraction = style.DEFAULT_COLUMN_GUTTER_INCHES / unit_inches
        row_gutter_fraction = style.DEFAULT_GUTTER_INCHES / unit_inches
        wspace_resolved = wspace if wspace is not None else column_gutter_fraction
        hspace_resolved = hspace if hspace is not None else row_gutter_fraction
        grid_w_inches = unit_inches * (n_cols + (n_cols - 1) * wspace_resolved)
        grid_h_inches = unit_inches * (n_rows + (n_rows - 1) * hspace_resolved)
        if suptitle is not None:
            # Suptitle band (1 margin tall, text V-centered) + 1 margin gap
            # below the band before the first panel border.
            top_reserve = 2.0 * margin
        else:
            top_reserve = margin
        figsize = (
            grid_w_inches + 2 * margin,
            grid_h_inches + margin + top_reserve,
        )

        fig = cls(
            n_rows=n_rows,
            n_cols=n_cols,
            figsize=figsize,
            suptitle=suptitle,
            suptitle_fontsize=suptitle_fontsize,
            dpi=dpi,
            hspace=hspace_resolved,
            wspace=wspace_resolved,
            debug_guides=debug_guides,
            show_cell_borders=show_cell_borders,
        )

        for r in range(n_rows):
            for p in range(len(rows[r])):
                panel = rows[r][p]
                col_start = sum(rows[r][i].units[0] for i in range(p))
                colspan = rows[r][p].units[0]
                projection = "3d" if isinstance(panel, StaticPanel3D) else None
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
        adjust_kwargs: dict[str, float] = {
            "left":   margin_h_frac,
            "right":  1.0 - margin_h_frac,
            "bottom": (
                max_bottom_pad + 0.05 if max_bottom_pad > 0.0
                else margin_v_frac
            ),
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
            # Hand control of the animation clock to the figure for any
            # DynamicPanel — the panel itself skips FuncAnimation construction
            # in its render() when this flag is set.
            if isinstance(panel, DynamicPanel):
                panel._managed_externally = True
            panel.attach(ax)
            panel.render()

        self._install_master_clock()
        if self._show_cell_borders:
            self._draw_cell_borders()
        if self._debug_guides:
            self._draw_debug_guides()
        self._apply_jupyter_display_styling()

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
        col_gutter = style.DEFAULT_COLUMN_GUTTER_INCHES
        row_gutter = style.DEFAULT_GUTTER_INCHES
        margin = style.DEFAULT_MARGIN_INCHES
        half_col_g_frac = (col_gutter / 2.0) / fig_w
        half_row_g_frac = (row_gutter / 2.0) / fig_h

        if self._has_suptitle:
            sup_band_bot_frac = 1.0 - margin / fig_h
        else:
            sup_band_bot_frac = 1.0

        border_kwargs = dict(
            fill=False,
            edgecolor=style.SPINE_COLOR,
            linewidth=style.DEFAULT_SPINE_LINEWIDTH,
            transform=fig.transFigure,
            zorder=50,
            clip_on=False,
        )

        for panel, row, col, rowspan, colspan, _ in self.panels:
            if panel.ax is None:
                continue
            bbox = panel.ax.get_position()

            cell_left = 0.0 if col == 0 else bbox.x0 - half_col_g_frac
            cell_right = (
                1.0 if col + colspan == self.n_cols
                else bbox.x1 + half_col_g_frac
            )
            cell_top = (
                sup_band_bot_frac if row == 0
                else bbox.y1 + half_row_g_frac
            )
            cell_bot = (
                0.0 if row + rowspan == self.n_rows
                else bbox.y0 - half_row_g_frac
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
                (0.0, sup_band_bot_frac),
                1.0,
                margin / fig_h,
                **border_kwargs,
            )
            fig.add_artist(rect)

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

        # --- Suptitle band (cyan): 1 margin tall at top, text V-centered ---
        if self._has_suptitle:
            sup_band_bot_y = 1.0 - margin_v_frac
            sup_center_y = 1.0 - (margin / 2.0) / fig_h
            panel_top_y = 1.0 - (2.0 * margin) / fig_h
            hline(sup_band_bot_y, "#00e5ff", "sup band bot")
            hline(sup_center_y, "#00e5ff", "sup center")
            hline(panel_top_y, "#ff00ff", "T margin")
        else:
            hline(1.0 - margin_v_frac, "#ff00ff", "T margin")

        # --- Per-panel: title band + axes bbox sides (yellow / orange) ---
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
        dynamic_panels = [
            p for p, *_ in self.panels if isinstance(p, DynamicPanel)
        ]
        if not dynamic_panels:
            return

        cycle_length = (
            max(p._total_frames() for p in dynamic_panels)
            + _FIGURE_CLOCK_HOLD_TICKS
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

        if self._fill_width:
            canvas.layout.width = self._display_width or "100%"
        elif self._display_width is not None:
            canvas.layout.width = self._display_width

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
