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

from typing import Optional, Tuple, Union

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
        lim: Optional[Union[float, Tuple[float, float]]] = None,
        axis_style: str = "line",
        axis_labels: bool = False,
        show_border: bool = True,
    ) -> None:
        super().__init__(units=units)
        self.title = title
        self.subtitle = subtitle
        self.lim = lim
        self.axis_style = axis_style
        self.axis_labels = axis_labels
        self.show_border = show_border

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
            lim=self._scalar_lim(),
            panel_title=None,
            show_border=self.show_border,
            axis_style=self.axis_style,
            axis_labels=self.axis_labels,
        )

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
