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
        self._mpl_fig = plt.figure(figsize=figsize, dpi=dpi)
        self._mpl_fig.patch.set_facecolor(style.BG_COLOR)
        self._gs = self._mpl_fig.add_gridspec(
            n_rows, n_cols,
            hspace=hspace, wspace=wspace,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        if suptitle is not None:
            self._mpl_fig.suptitle(
                suptitle,
                color=style.TICK_LABEL_COLOR,
                fontsize=(
                    suptitle_fontsize
                    if suptitle_fontsize is not None
                    else style.DEFAULT_SUPTITLE_FONT_SIZE
                ),
                fontweight="bold",
                y=(suptitle_y if suptitle_y is not None else 0.975),
            )

        self.panels: List[Tuple[Panel, int, int, int, int, Optional[str]]] = []
        # Master FuncAnimation when any DynamicPanels are present; load-bearing
        # for matplotlib timer GC the same way DynamicPanel._anim is in the
        # standalone path.
        self._anim = None

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
        # All four margins resolve from style.DEFAULT_MARGIN_* constants
        # — these are the SOLE source of figure padding because
        # Figure.savefig() saves at exact figsize × dpi (no tight-bbox).
        adjust_kwargs: dict[str, float] = {
            "left":   style.DEFAULT_MARGIN_LEFT,
            "right":  style.DEFAULT_MARGIN_RIGHT,
            "bottom": (
                max_bottom_pad + 0.05 if max_bottom_pad > 0.0
                else style.DEFAULT_MARGIN_BOTTOM
            ),
        }
        if self._has_suptitle:
            adjust_kwargs["top"] = (
                self._top_pad if self._top_pad is not None else 0.84
            )
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
        self._apply_jupyter_display_styling()

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
