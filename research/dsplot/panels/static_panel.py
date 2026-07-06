"""StaticPanel — single-state Panel that renders all Plottables once.

Renders in insertion order; z-ordering between plottables is resolved by each
Plottable's own zorder kwarg. Title/subtitle/lim/axis-style/axis-labels all
flow through `dsplot.axes_setup.setup_vector_axes` so the panel's chrome is
consistent with any other 2D vector panel rendered by the library.

Layout defaults resolve LAZILY against `dsplot.style.DEFAULT_*` at render()
time (per D-05) — reassigning a style constant between construction and
render observes the new value.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from .. import style
from ..axes_setup import setup_vector_axes
from .base import Panel


class StaticPanel(Panel):
    """Single-state panel. `render()` iterates the plottables in insertion order.

    `render()` raises RuntimeError if called before `attach()` — the owning
    Figure handles attach + render in its own render loop.
    """

    def __init__(
        self,
        *,
        units: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        caption: Optional[str] = None,
        lim: Optional[Union[float, Tuple[float, float]]] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        axis_style: str = "line",
        axis_labels: bool = False,
        axis_label_size: Optional[float] = None,
        show_border: bool = True,
        show_ticks: bool = False,
        show_grid: bool = False,
        tick_positions: Optional[Tuple[float, ...]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        show_tick_labels: bool = False,
    ) -> None:
        super().__init__(units=units)
        self.title = title
        self.subtitle = subtitle
        self.caption = caption
        self.lim = lim
        # Explicit per-axis limits override the shared `lim`. An asymmetric pair
        # shifts the origin so a single-quadrant vector fills the cell; keep the
        # two ranges equal to preserve the honest equal-aspect geometry.
        self.xlim = xlim
        self.ylim = ylim
        self.axis_style = axis_style
        self.axis_labels = axis_labels
        self.axis_label_size = axis_label_size
        self.show_border = show_border
        self.show_ticks = show_ticks
        self.show_grid = show_grid
        self.tick_positions = tick_positions
        # x_label / y_label render EXTERNAL axis labels in the gutter via the
        # same path used by HeatmapPanel / TimeSeriesPanel — distinct from
        # axis_labels=True which puts italic "x"/"y" INSIDE the cell near the
        # spine tips. show_tick_labels enables numerical tick labels (only
        # meaningful when show_ticks=True).
        self.x_label = x_label
        self.y_label = y_label
        self.show_tick_labels = show_tick_labels

    def render(self) -> None:
        if self.ax is None:
            raise RuntimeError("StaticPanel.render() called before attach()")

        self.ax.set_facecolor(style.BG_COLOR)
        if self.show_border:
            for spine in self.ax.spines.values():
                spine.set_edgecolor(style.SPINE_COLOR)
                spine.set_linewidth(style.DEFAULT_SPINE_LINEWIDTH)
        else:
            for spine in self.ax.spines.values():
                spine.set_visible(False)

        setup_vector_axes(
            self.ax,
            lim=self.lim,
            xlim=self.xlim,
            ylim=self.ylim,
            panel_title=None,
            show_border=self.show_border,
            axis_style=self.axis_style,
            axis_labels=self.axis_labels,
            axis_label_size=self.axis_label_size,
        )

        if self.show_ticks:
            lim = self._scalar_lim() or style.DEFAULT_VECTOR_LIM
            if self.tick_positions is not None:
                positions = list(self.tick_positions)
            else:
                # Integer ticks across the visible range, skipping 0.
                step = max(1, int(round(lim / 4.0)))
                positions = [
                    float(i) for i in range(-int(lim), int(lim) + 1, step)
                    if i != 0 and abs(i) < lim
                ]
            self.ax.set_xticks(positions)
            self.ax.set_yticks(positions)
            # Numerical tick labels (when show_tick_labels=True) use the same
            # styled size/color as Heatmap/TimeSeries panels so tick chrome is
            # consistent across all panel kinds.
            label_kwargs = dict(
                labelbottom=self.show_tick_labels,
                labelleft=self.show_tick_labels,
            )
            if self.show_tick_labels:
                label_kwargs.update(
                    labelsize=style.DEFAULT_TICK_LABEL_SIZE,
                    labelcolor=style.TICK_LABEL_COLOR,
                )
            self.ax.tick_params(
                axis="both",
                colors=style.SPINE_COLOR,
                length=style.DEFAULT_TICK_LENGTH * style.DEFAULT_INSET_TICK_LENGTH_SCALE,
                width=style.DEFAULT_TICK_WIDTH * style.DEFAULT_INSET_TICK_WIDTH_SCALE,
                direction=style.DEFAULT_TICK_DIRECTION,
                **label_kwargs,
            )

        if self.show_grid:
            # ax.grid() draws lines at TICK positions — but setup_vector_axes
            # cleared ticks to [] (and show_ticks=False panels never set any), so
            # without locators the grid renders nothing. Establish an integer grid
            # of locators (skipping 0 — the axis crosshair owns that line) with the
            # tick MARKS/labels kept invisible, so the light grid shows even on a
            # tickless panel.
            if not self.show_ticks:
                import math
                x_lo, x_hi = self.ax.get_xlim()
                y_lo, y_hi = self.ax.get_ylim()
                xpos = [float(i) for i in range(math.ceil(x_lo), math.floor(x_hi) + 1) if i != 0]
                ypos = [float(i) for i in range(math.ceil(y_lo), math.floor(y_hi) + 1) if i != 0]
                self.ax.set_xticks(xpos)
                self.ax.set_yticks(ypos)
                self.ax.tick_params(axis="both", length=0,
                                    labelbottom=False, labelleft=False)
            self.ax.grid(
                True,
                color=style.DEFAULT_AXIS_GRID_COLOR,
                alpha=style.DEFAULT_AXIS_GRID_ALPHA,
                linewidth=style.DEFAULT_AXIS_GRID_LINEWIDTH,
                zorder=0,
            )

        # External x/y axis labels: route through the shared decoration helper
        # so spacing/font/inset match Heatmap and TimeSeries panels. Passing
        # xticks=None / yticks=None preserves whatever ticks the show_ticks
        # branch above set up — _apply_axis_decoration only restyles ticks
        # when an explicit positions list is passed. Local import to dodge the
        # static_panel ↔ heatmap_panel module cycle.
        if self.x_label is not None or self.y_label is not None:
            from .heatmap_panel import _apply_axis_decoration
            _apply_axis_decoration(
                self.ax,
                x_label=self.x_label,
                y_label=self.y_label,
                xticks=None,
                yticks=None,
                show_xticklabels=self.show_tick_labels,
                show_yticklabels=self.show_tick_labels,
            )
        elif self.show_ticks and self.show_tick_labels:
            # No x/y_label → _apply_axis_decoration is skipped. Pin tick
            # labels directly so the extreme ones don't project past the
            # spine into adjacent cells' chrome zones.
            from .heatmap_panel import _pin_extreme_ticklabels
            _pin_extreme_ticklabels(self.ax)

        self._render_chrome_titles()

        for plottable in self._plottables:
            plottable.draw(self.ax)

    def _scalar_lim(self) -> Optional[float]:
        if self.lim is None:
            return None
        if isinstance(self.lim, tuple):
            lo, hi = self.lim
            return max(abs(lo), abs(hi))
        return float(self.lim)
